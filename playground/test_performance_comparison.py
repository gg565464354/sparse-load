import sys
import os
import torch
import time
import numpy as np

# 添加 libs 路径
local_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
sys.path.insert(0, local_lib_path)

from transformers import AutoTokenizer, AutoModelForCausalLM

def benchmark_model(model_path, test_text, use_cache=True, num_runs=5):
    """基准测试模型性能"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"测试配置: use_heavy_hitter_cache={use_cache}")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager",
        use_heavy_hitter_cache=use_cache
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
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    for i in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.3f}s")
    
    # 统计结果
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n{'='*50}")
    print(f"use_heavy_hitter_cache={use_cache} 结果:")
    print(f"平均时间: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"最快时间: {min(times):.3f}s")
    print(f"最慢时间: {max(times):.3f}s")
    print(f"{'='*50}\n")
    
    return avg_time, std_time

def main():
    # 测试配置
    model_path = "facebook/opt-125m"  # 使用小模型进行快速测试
    test_text = "The quick brown fox jumps over the lazy dog. This is a test sentence for performance benchmarking."
    
    print("开始性能对比测试...")
    print(f"模型: {model_path}")
    print(f"测试文本: {test_text}")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # 测试vanilla attention (禁用缓存)
    vanilla_time, vanilla_std = benchmark_model(model_path, test_text, use_cache=False)
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 测试heavy-hitter缓存
    cache_time, cache_std = benchmark_model(model_path, test_text, use_cache=True)
    
    # 对比结果
    print("="*60)
    print("性能对比结果:")
    print("="*60)
    print(f"Vanilla Attention:      {vanilla_time:.3f}s ± {vanilla_std:.3f}s")
    print(f"Heavy-Hitter Cache:     {cache_time:.3f}s ± {cache_std:.3f}s")
    
    if vanilla_time > 0:
        speedup = vanilla_time / cache_time
        if speedup > 1:
            print(f"加速比: {speedup:.2f}x (缓存更快)")
        else:
            print(f"减速比: {1/speedup:.2f}x (缓存更慢)")
    
    print("="*60)

if __name__ == "__main__":
    main() 