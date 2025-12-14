import sys
import os
import torch
import time
import numpy as np

# 添加 libs 路径
local_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
sys.path.insert(0, local_lib_path)

from transformers import AutoTokenizer, AutoModelForCausalLM

def test_with_disabled_masking(model_path, test_text, num_runs=3):
    """测试完全禁用masking的性能"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型并手动禁用masking
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager",
        padding_strategy="least_important"
    )
    
    # 手动设置所有层的masker参数为0，相当于禁用masking
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'heavy_hitter_masker'):
            layer.self_attn.heavy_hitter_masker.heavy_budget_ratio = 0.0
            layer.self_attn.heavy_hitter_masker.recent_budget_ratio = 0.0
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 准备输入
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    # 预热
    print("预热中...")
    with torch.no_grad():
        for _ in range(2):
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0
            )
    
    # 性能测试
    print(f"开始性能测试，运行 {num_runs} 次...")
    times = []
    
    for i in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0
            )
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
        print(f"运行 {i+1}: {times[-1]:.4f}s")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n禁用masking性能结果:")
    print(f"平均时间: {mean_time:.4f}s")
    print(f"标准差: {std_time:.4f}s")
    
    return mean_time, std_time

def test_with_enabled_masking(model_path, test_text, num_runs=3):
    """测试启用masking的性能"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型，使用默认的masking设置
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager",
        padding_strategy="least_important"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 准备输入
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    # 预热
    print("预热中...")
    with torch.no_grad():
        for _ in range(2):
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0
            )
    
    # 性能测试
    print(f"开始性能测试，运行 {num_runs} 次...")
    times = []
    
    for i in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0
            )
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
        print(f"运行 {i+1}: {times[-1]:.4f}s")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n启用masking性能结果:")
    print(f"平均时间: {mean_time:.4f}s")
    print(f"标准差: {std_time:.4f}s")
    
    return mean_time, std_time

def main():
    model_path = "/root/model/Qwen2-1.5B-Instruct"
    test_text = "请解释一下人工智能的基本概念和应用领域。"
    
    print("=" * 80)
    print("Masking开关测试：禁用 vs 启用")
    print("=" * 80)
    
    print("\n1. 测试完全禁用masking的性能:")
    print("-" * 40)
    disabled_time, disabled_std = test_with_disabled_masking(model_path, test_text)
    
    print("\n2. 测试启用masking的性能:")
    print("-" * 40)
    enabled_time, enabled_std = test_with_enabled_masking(model_path, test_text)
    
    print("\n" + "=" * 80)
    print("性能对比结果:")
    print("=" * 80)
    print(f"禁用masking平均时间: {disabled_time:.4f}s ± {disabled_std:.4f}s")
    print(f"启用masking平均时间: {enabled_time:.4f}s ± {enabled_std:.4f}s")
    
    if enabled_time < disabled_time:
        speedup = disabled_time / enabled_time
        improvement = ((disabled_time - enabled_time) / disabled_time) * 100
        print(f"\n✅ 启用masking更快！")
        print(f"   加速比: {speedup:.2f}x")
        print(f"   性能提升: {improvement:.1f}%")
    else:
        slowdown = enabled_time / disabled_time
        overhead = ((enabled_time - disabled_time) / disabled_time) * 100
        print(f"\n❌ 启用masking更慢！")
        print(f"   减速比: {slowdown:.2f}x")
        print(f"   性能开销: {overhead:.1f}%")
    
    print("\n建议:")
    if overhead > 10:
        print("- masking开销过大，建议进一步优化或在特定场景下禁用")
    elif overhead < 5:
        print("- masking开销可接受，可以继续使用")
    else:
        print("- masking开销适中，可根据具体需求选择是否启用")

if __name__ == "__main__":
    main() 