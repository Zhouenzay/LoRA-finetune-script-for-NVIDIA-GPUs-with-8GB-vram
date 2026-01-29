import os
import json
import torch
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from datasets import Dataset


# 配置路径
DATA_PATH = r"" # 训练数据（json）的文件路径
OUTPUT_DIR = r"" # 输出目录，用于保存模型和检查点
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" #从hugging face下载的模型名称

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 自动获取本地 HF 模型路径
def get_hf_local_path(repo_id):
    return snapshot_download(repo_id=repo_id, local_files_only=True)

try:
    ACTUAL_MODEL_PATH = get_hf_local_path(MODEL_NAME)
    print(f"📁 使用本地模型: {ACTUAL_MODEL_PATH}")
except Exception as e:
    raise RuntimeError(
        f"未找到本地模型 {MODEL_NAME}。请先运行一次 from_pretrained 下载模型。"
    ) from e


# 加载并预处理数据
def load_and_process_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def format_example(example):
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]
        if input_text:
            prompt = f"指令：{instruction}\n输入：{input_text}\n输出："
        else:
            prompt = f"指令：{instruction}\n输出："
        full_text = prompt + output
        return {"text": full_text}

    formatted_data = [format_example(item) for item in data]
    dataset = Dataset.from_list(formatted_data)
    return dataset


# 加载模型和分词器（4-bit）
def load_model_and_tokenizer():
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4it=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(ACTUAL_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        ACTUAL_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


# 应用 LoRA
def add_lora_to_model(model):
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, lora_config


# 安全合并函数（全程 CPU）
def safe_merge_to_cpu(lora_path, merged_output_dir):
    print("🔄 开始安全合并 LoRA 到 CPU（低内存模式）...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        ACTUAL_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    model = PeftModel.from_pretrained(base_model, lora_path, device_map="cpu")
    merged_model = model.merge_and_unload()
    
    print("  保存为 PyTorch .bin 格式（低内存）...")
    merged_model.save_pretrained(
        merged_output_dir,
        safe_serialization=False,  
        max_shard_size="1GB"       
    )
    
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_output_dir)
    
    del base_model, model, merged_model
    torch.cuda.empty_cache()
    print("✅ 合并完成！")


# 主函数
def main():
    print("Loading dataset...")
    dataset = load_and_process_data(DATA_PATH)

    print("Loading model and tokenizer (4-bit)...")
    model, tokenizer = load_model_and_tokenizer()

    print("Adding LoRA adapters...")
    model, lora_config = add_lora_to_model(model)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding=False,
            add_special_tokens=False,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = os.path.join(OUTPUT_DIR, "lora_checkpoint")

  
    # 自动检测 checkpoint
    checkpoint = None
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            checkpoint = os.path.join(output_dir, latest)
            print(f"✅ Found checkpoint: {checkpoint}")
        else:
            print("ℹ️ No checkpoint found. Starting from scratch.")
    else:
        print("ℹ️ Output directory does not exist. Starting from scratch.")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=1,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("🚀 Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint) 

    print("💾 Saving final LoRA adapter...")
    lora_output_dir = os.path.join(OUTPUT_DIR, "lora_final")
    trainer.model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)

 
    # 安全合并
    print("🔄 开始生成 merged_model...")
    merged_dir = os.path.join(OUTPUT_DIR, "merged_model")
    safe_merge_to_cpu(lora_output_dir, merged_dir)

    # 清理中间文件
    print("🧹 Cleaning up intermediate files...")
    for path in [output_dir, lora_output_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Deleted: {path}")

    print("\n" + "="*60)
    print("合并完成")

if __name__ == "__main__":
    main()
