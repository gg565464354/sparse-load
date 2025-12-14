import sys
import os
import torch
import time
import numpy as np

# 添加 libs 路径
local_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
sys.path.insert(0, local_lib_path)

from transformers import AutoTokenizer, AutoModelForCausalLM

def test_without_cache(model_path, test_text, num_runs=3):
    """测试没有缓存的原版性能"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载原版模型（不使用padding_strategy参数）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager"
        # 不传入padding_strategy参数，使用原版attention
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
    
    print(f"\n原版性能结果:")
    print(f"平均时间: {mean_time:.4f}s")
    print(f"标准差: {std_time:.4f}s")
    
    return mean_time, std_time

def test_with_cache(model_path, test_text, num_runs=3):
    """测试有缓存的优化版性能"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载优化版模型
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
    
    print(f"\n优化版性能结果:")
    print(f"平均时间: {mean_time:.4f}s")
    print(f"标准差: {std_time:.4f}s")
    
    return mean_time, std_time

def main():
    model_path = "/root/model/Qwen2-1.5B-Instruct"
    test_text = "请解释一下人工智能的基本概念和应用领域。"
    
    print("=" * 80)
    print("性能对比测试：原版 vs 优化版 CachedHeavyRecentAttentionMasker")
    print("=" * 80)
    
    print("\n1. 测试原版（无缓存）性能:")
    print("-" * 40)
    original_time, original_std = test_without_cache(model_path, test_text)
    
    print("\n2. 测试优化版（有缓存）性能:")
    print("-" * 40)
    optimized_time, optimized_std = test_with_cache(model_path, test_text)
    
    print("\n" + "=" * 80)
    print("性能对比结果:")
    print("=" * 80)
    print(f"原版平均时间:     {original_time:.4f}s ± {original_std:.4f}s")
    print(f"优化版平均时间:   {optimized_time:.4f}s ± {optimized_std:.4f}s")
    
    if optimized_time < original_time:
        speedup = original_time / optimized_time
        improvement = ((original_time - optimized_time) / original_time) * 100
        print(f"\n✅ 优化版更快！")
        print(f"   加速比: {speedup:.2f}x")
        print(f"   性能提升: {improvement:.1f}%")
    else:
        slowdown = optimized_time / original_time
        degradation = ((optimized_time - original_time) / original_time) * 100
        print(f"\n❌ 优化版更慢！")
        print(f"   减速比: {slowdown:.2f}x")
        print(f"   性能下降: {degradation:.1f}%")
    
    print("\n注意：性能差异可能受到以下因素影响：")
    print("- GPU内存状态")
    print("- 缓存命中率")
    print("- 输入序列长度")
    print("- 模型参数设置")

if __name__ == "__main__":
    main() 