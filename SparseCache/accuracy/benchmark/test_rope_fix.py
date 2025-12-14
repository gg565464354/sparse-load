#!/usr/bin/env python3
"""快速测试RoPE修复是否生效"""
import torch
import sys
sys.path.insert(0, '/root/sparse-load/playground/libs/transformers/src')

from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("测试RoPE修复")
print("=" * 60)

# 加载模型
model_path = "/root/data_djl/model/Llama-2-7B-32K-Instruct"
print(f"\n加载模型: {model_path}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("✅ 模型加载成功")
    
    # 简单测试
    prompt = "Hello, my name is"
    print(f"\n测试提示: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Input shape: {inputs.input_ids.shape}")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n生成结果:\n{result}")
    
    # 检查是否是乱码
    if any(char in result for char in ['\\n\\n\\n', '<<<', '111', '666', '```']):
        print("\n❌ 检测到乱码特征！")
    else:
        print("\n✅ 输出看起来正常")
        
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
