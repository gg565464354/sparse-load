#!/usr/bin/env python3
"""检查模型实际使用的类"""
import os
import sys
sys.path.insert(0, '/root/sparse-load/playground/libs/transformers/src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("检查模型使用的类")
print("=" * 60)

model_path = "/root/data_djl/model/Llama-2-7B-32K-Instruct"

print(f"\n1. 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

print(f"\n2. 检查模型类:")
print(f"   Model class: {type(model).__name__}")
print(f"   Model module: {type(model).__module__}")

print(f"\n3. 检查第一层attention:")
attn = model.model.layers[0].self_attn
print(f"   Attention class: {type(attn).__name__}")
print(f"   Attention module: {type(attn).__module__}")

print(f"\n4. 检查rotary_emb:")
rotary = attn.rotary_emb if hasattr(attn, 'rotary_emb') else model.model.rotary_emb if hasattr(model.model, 'rotary_emb') else None
if rotary:
    print(f"   RotaryEmbedding class: {type(rotary).__name__}")
    print(f"   RotaryEmbedding module: {type(rotary).__module__}")
    print(f"   Has inv_freq: {hasattr(rotary, 'inv_freq')}")
    print(f"   Has cos_cached: {hasattr(rotary, 'cos_cached')}")
    print(f"   Has sin_cached: {hasattr(rotary, 'sin_cached')}")
else:
    print(f"   ❌ 未找到rotary_emb")

print(f"\n5. 检查模型文件路径:")
import inspect
model_file = inspect.getfile(type(model))
print(f"   Model file: {model_file}")

attn_file = inspect.getfile(type(attn))
print(f"   Attention file: {attn_file}")

if rotary:
    rotary_file = inspect.getfile(type(rotary))
    print(f"   RotaryEmbedding file: {rotary_file}")

print(f"\n6. 检查apply_rotary_pos_emb函数:")
if hasattr(attn, 'forward'):
    import inspect
    source = inspect.getsource(type(attn).forward)
    if 'apply_rotary_pos_emb' in source:
        # 查找apply_rotary_pos_emb函数
        module = sys.modules[type(attn).__module__]
        if hasattr(module, 'apply_rotary_pos_emb'):
            func = module.apply_rotary_pos_emb
            func_source = inspect.getsource(func)
            print(f"   apply_rotary_pos_emb函数长度: {len(func_source)} 字符")
            if 'clamp' in func_source:
                print(f"   ❌ 函数中仍包含clamp!")
            else:
                print(f"   ✅ 函数中没有clamp")
            if 'Check if position_ids are within bounds' in func_source:
                print(f"   ✅ 函数中有边界检查")
            else:
                print(f"   ❌ 函数中没有边界检查")

print("\n" + "=" * 60)
