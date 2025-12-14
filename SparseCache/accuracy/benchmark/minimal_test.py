#!/usr/bin/env python3
"""最小化测试：直接生成文本"""
import os
import sys

# 强制使用我们修复的代码
os.system("rm -f /root/sparse-load/playground/libs/transformers/src/transformers/models/llama/modeling_llama.py")
os.system("ln -s /root/sparse-load/SparseCache/accuracy/benchmark/source/modeling_llama_ours.py /root/sparse-load/playground/libs/transformers/src/transformers/models/llama/modeling_llama.py")

sys.path.insert(0, '/root/sparse-load/playground/libs/transformers/src')

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

print("=" * 60)
print("最小化测试")
print("=" * 60)

model_path = "/root/data_djl/model/Llama-2-7B-32K-Instruct"

print(f"\n1. 加载tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(model_path)

print(f"\n2. 加载模型...")
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print(f"   模型类: {type(model).__module__}.{type(model).__name__}")
print(f"   Attention类: {type(model.model.layers[0].self_attn).__module__}.{type(model.model.layers[0].self_attn).__name__}")

# 简单测试
prompt = "What is the capital of France?"
print(f"\n3. 测试生成...")
print(f"   提示: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n4. 结果:")
print(f"   {result}")

# 检查乱码
has_gibberish = any(x in result for x in ['<<<', '111', '666', '```', '\n\n\n  <'])
print(f"\n5. 判断:")
if has_gibberish:
    print("   ❌ 检测到乱码特征")
else:
    print("   ✅ 输出正常")

print("\n" + "=" * 60)
