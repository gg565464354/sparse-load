import sys
import os
import torch
import time
import numpy as np

# 添加 libs 路径
local_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
sys.path.insert(0, local_lib_path)

from transformers import AutoTokenizer, AutoModelForCausalLM

def test_performance(model_path, test_text, num_runs=5):
    """测试模型性能"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型和分词器
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
    
    # 统计结果
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n性能测试结果:")
    print(f"平均时间: {mean_time:.4f}s")
    print(f"标准差: {std_time:.4f}s")
    print(f"最快时间: {min(times):.4f}s")
    print(f"最慢时间: {max(times):.4f}s")
    
    # 解码输出
    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(f"\n生成的文本: {response[:100]}...")
    
    return mean_time, std_time

def main():
    model_path = "/root/model/Qwen2-1.5B-Instruct"
    test_text = "请解释一下人工智能的基本概念和应用领域。"
    
    print("=" * 60)
    print("优化版本 CachedHeavyRecentAttentionMasker 性能测试")
    print("=" * 60)
    
    mean_time, std_time = test_performance(model_path, test_text)
    
    print(f"\n总结:")
    print(f"优化版本平均生成时间: {mean_time:.4f}s ± {std_time:.4f}s")

if __name__ == "__main__":
    main() 