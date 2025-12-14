import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
# 尝试显式导入 OPT 模型
try:
    from transformers import OPTForCausalLM
    print("Successfully imported OPTForCausalLM")
except ImportError as e:
    print(f"Warning: Could not import OPTForCausalLM: {e}")
    # 可以继续使用 Auto 类
import time
import os

# ... 保持其他函数不变 ...

class TransformersBlend:
    """
    使用 Hugging Face Transformers 实现 KV 缓存管理和生成。
    """
    def __init__(self, model_path, device="cuda"):
        print(f"Initializing model from: {model_path}")
        
        # 首先尝试加载 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                print("Setting pad_token = eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            raise

        # 尝试多种方式加载模型
        try:
            # 方法1: 使用 AutoModelForCausalLM，修复参数名
            print("Attempting to load with AutoModelForCausalLM...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,  # 保持原有参数名，某些版本仍支持
                device_map="auto",
                trust_remote_code=True  # 添加这个参数
            )
        except Exception as e1:
            print(f"AutoModel failed: {e1}")
            try:
                # 方法2: 使用新的参数名
                print("Trying with 'dtype' parameter...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            except Exception as e2:
                print(f"New parameter format failed: {e2}")
                try:
                    # 方法3: 显式使用 OPTForCausalLM
                    print("Trying with explicit OPTForCausalLM...")
                    from transformers import OPTForCausalLM
                    self.model = OPTForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                except Exception as e3:
                    print(f"Explicit OPT model failed: {e3}")
                    raise Exception(f"All model loading attempts failed. Last error: {e3}")

        self.model.eval()
        # 获取模型实际所在的设备，以确保张量在正确的设备间移动
        self.device = next(self.model.parameters()).device
        print(f"Model loaded successfully on device: {self.device}")

    # ... 保持其他方法不变 ...
