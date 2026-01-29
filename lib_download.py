import sys
import subprocess

packages = ["torch", "transformers", "datasets", "huggingface-hub", "peft", "llama-cpp-python"]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
