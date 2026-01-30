# LoRA-finetune-script-for-NVIDIA-GPUs-with-8GB-vram
摘要：此python脚本针对8GB显存的英伟达GPU进行了优化，从而使消费级别显卡用户得以使用本地设备进行LoRA微调  

注意事项：  
  
1.在运行训练脚本前，请先确认您创建了虚拟环境（.venv）且安装了必要的库。运行训练脚本前务必先运行lib_download.py以确保环境完整性。  
2.确保机载内存（RAM）至少为32GB，磁盘上的可用空间至少应为原始模型的三倍。其中，C盘的可用空间应大于原始模型大小。  
3.合并LoRA权重后的模型采用分片存储的策略，在后续操作中请确保选择完整的合并后模型文件夹以确保完整性。  
4.此脚本仅在Windows10/11平台进行验证，其他操作系统对此的支持未知。  
5.为确保稳定性，请使用python3.11解释器。

声明：  

此脚本的开发过程使用了生成式AI。  

项目愿景：  

让个性化AI触手可及，让科技发展的成果惠及广大人民群众

===========================================================================================

Summary: This Python script is optimized for NVIDIA GPUs with 8GB of VRAM, allowing consumer-grade GPU users to fine-tune LoRA locally.

Notes:

1.Before running the training script, ensure you have created a virtual environment (.venv) and installed the necessary libraries. Run lib_download.py before running the training script to ensure environment integrity.

2.Ensure at least 32GB of onboard RAM and at least three times the available disk space of the original model. The C drive should have more available space than the original model size.

3.The model after merging LoRA weights uses a sharded storage strategy. In subsequent operations, ensure you select the complete merged model folder to ensure integrity.

4.This script has only been validated on Windows 10/11 platforms; support for other operating systems is unknown.

5.For stability, please use the Python 3.11 interpreter.

Declaration:

This script was developed with generative AI.  
  
Project Vision:

Personalized AI models will no longer be out of reach, and the fruits of technological advancement will benefit all people.
