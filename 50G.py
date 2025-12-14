import sys
import os
import torch
import time
import pickle  # 用于加载 partial_weight
local_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
sys.path.insert(0, local_lib_path)
# 不再加载 transformers，避免加载模型
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import transformers
# POOL_PATH = "qwen2-1.5b-low-attention-pool.json" 

def allocate_gpu_memory(target_gb=50):
    """预分配GPU显存到指定大小（默认50GB）"""
    if not torch.cuda.is_available():
        print("未检测到CUDA，跳过显存分配。")
        return []
    
    print(f"开始预分配 {target_gb}GB 显存...")
    memory_tensors = []
    
    try:
        # 每次分配5GB
        chunk_gb = 5
        elements_per_chunk = int((chunk_gb * 1024**3) / 2)  # float16 = 2 bytes
        
        while True:
            current_memory = torch.cuda.memory_allocated() / (1024**3)
            if current_memory >= target_gb * 0.95:  # 达到目标的95%
                break
                
            tensor = torch.randn(elements_per_chunk, dtype=torch.float16, device='cuda')
            memory_tensors.append(tensor)
            
            new_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"分配了 {chunk_gb}GB，总显存: {new_memory:.2f}GB")
            
    except RuntimeError as e:
        print(f"显存分配达到上限或出错: {e}")
    
    final_memory = torch.cuda.memory_allocated() / (1024**3)
    print(f"最终显存占用: {final_memory:.2f}GB")
    return memory_tensors

def parse_target_gb(default_gb=50):
    # 优先命令行参数，其次环境变量 TARGET_GPU_GB，最后默认值
    # 用法: python run_chat_qwen2_50G.py 60
    if len(sys.argv) >= 2:
        try:
            return float(sys.argv[1])
        except Exception:
            pass
    env_val = os.environ.get("TARGET_GPU_GB")
    if env_val:
        try:
            return float(env_val)
        except Exception:
            pass
    return float(default_gb)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_gb = parse_target_gb(90)

    # 预分配显存（默认50GB）
    memory_tensors = allocate_gpu_memory(target_gb)

    # 保持显存占用与进程存活（不加载任何模型）
    if device == "cuda":
        print(f"当前显存占用: {torch.cuda.memory_allocated() / (1024**3):.2f}GB")
    else:
        print("当前为CPU环境，没有显存分配。")

    print("进入常驻循环，按 Ctrl+C 退出。")
    try:
        while True:
            if device == "cuda":
                used = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                print(f"[心跳] 已占用: {used:.2f}GB, 已保留: {reserved:.2f}GB")
            time.sleep(10)
    except KeyboardInterrupt:
        print("接收到退出信号，准备结束。")
        # 释放显存（如果希望退出时立即释放），一般不需要手动清理
        # memory_tensors.clear()
        # torch.cuda.empty_cache()

if __name__ == "__main__":
    main()