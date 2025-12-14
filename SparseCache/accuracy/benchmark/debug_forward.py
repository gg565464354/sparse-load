#!/usr/bin/env python3
"""调试forward过程，找出乱码的原因"""
import os
import sys
sys.path.insert(0, '/root/sparse-load/playground/libs/transformers/src')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 60)
print("调试forward过程")
print("=" * 60)

model_path = "/root/data_djl/model/Llama-2-7B-32K-Instruct"
print(f"\n加载模型: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 添加hook来监控第一个attention层
first_attn = model.model.layers[0].self_attn

original_forward = first_attn.forward

def debug_forward(self, *args, **kwargs):
    print(f"\n=== Layer 0 Attention Forward ===")
    hidden_states = args[0] if len(args) > 0 else kwargs.get('hidden_states')
    attention_mask = args[1] if len(args) > 1 else kwargs.get('attention_mask')
    position_ids = args[2] if len(args) > 2 else kwargs.get('position_ids')
    
    print(f"  hidden_states: {hidden_states.shape if hidden_states is not None else None}")
    print(f"  attention_mask: {attention_mask.shape if attention_mask is not None else None}")
    print(f"  position_ids: {position_ids.shape if position_ids is not None else None}")
    if position_ids is not None:
        print(f"  position_ids values: {position_ids[0, :10] if position_ids.shape[1] > 10 else position_ids[0]}")
        print(f"  position_ids min/max: {position_ids.min().item()}/{position_ids.max().item()}")
    
    # 调用原始forward
    result = original_forward(*args, **kwargs)
    
    # 检查输出
    attn_output = result[0]
    print(f"  attn_output: {attn_output.shape}")
    print(f"  attn_output有NaN: {torch.isnan(attn_output).any().item()}")
    print(f"  attn_output有Inf: {torch.isinf(attn_output).any().item()}")
    print(f"  attn_output均值/std: {attn_output.mean().item():.4f}/{attn_output.std().item():.4f}")
    
    return result

first_attn.forward = lambda *args, **kwargs: debug_forward(first_attn, *args, **kwargs)

# 测试
prompt = "Hello, my name is"
print(f"\n测试提示: '{prompt}'")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"Input IDs: {inputs.input_ids}")
print(f"Input shape: {inputs.input_ids.shape}")

try:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n生成结果:\n{result}")
    
    # 检查是否是乱码
    if any(char in result for char in ['<<<', '111', '666', '```', '\\n\\n\\n']):
        print("\n❌ 检测到乱码特征！")
    else:
        print("\n✅ 输出看起来正常")
        
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
