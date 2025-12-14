import torch
import torch.nn.functional as F
import my_cache_load._C as _C
from concurrent.futures import ThreadPoolExecutor

import sys

import time

def select_kv(prefetch_idx, k_cache, v_cache):
    """Selects and aggregates critical KV caches using speculated indices"""
    prefetch_idx = prefetch_idx.squeeze().to(k_cache.device)
    ind = prefetch_idx * k_cache.shape[1] + torch.arange(k_cache.shape[1], device=k_cache.device)[None, :]
    selected_k = F.embedding(ind, k_cache.reshape(-1, k_cache.shape[2]))
    selected_v = F.embedding(ind, v_cache.reshape(-1, v_cache.shape[2]))
    return selected_k, selected_v

# ======================
# 配置参数
# ======================
num_tokens_list = [1024, 2048, 4096]   # 不同长度的 cache
batch_size = 4
num_heads = 8
head_dim = 64

device = 'cpu'

# ======================
# 测试循环
# ======================
for n_prefetch in [64, 128, 256]:  # prefetch token 数量
    for n in num_tokens_list:
        print(f"\nTesting with k_cache/v_cache size: ({n}, {num_heads}, {head_dim}), "
              f"prefetch tokens: {n_prefetch}")
        
        # 构建 k_cache 和 v_cache（可以不同长度）
        k_cache = torch.randn(n, num_heads, head_dim, device=device)
        v_cache = torch.randn(n, num_heads, head_dim, device=device)

        # 构建 prefetch_idx (n', 1, bh) -> 每个 batch-head 对应一个 token index
        prefetch_idx = torch.randint(0, n, (n_prefetch, 1, num_heads), device=device)

        # 同步 CUDA 时间
        if device == 'cuda':
            torch.cuda.synchronize()

        # 计时开始
        start_time = time.time()

        # 执行函数多次以获得稳定结果
        for _ in range(100):
            selected_k, selected_v = select_kv(prefetch_idx, k_cache, v_cache)
        
        if device == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        print(f"Average execution time: {avg_time * 1000:.4f} ms")


