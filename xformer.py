import torch
import torch.nn.functional as F
import time
import json
from flash_attn.flash_attn_interface import flash_attn_func

# **🚀 1. 标准 SDPA（带 Mask）**
def sdpa_with_mask(Q, K_padded, V_padded, attn_mask):
    return F.scaled_dot_product_attention(Q, K_padded, V_padded, attn_mask=attn_mask)

# **🚀 2. 直接去掉 Mask 的 SDPA**
def sdpa_no_mask(Q, K_padded, V_padded):
    return F.scaled_dot_product_attention(Q, K_padded, V_padded)

# **🚀 3. 使用 Flash Attention 处理变长 KV Attention**
def flash_attention(Q, K_padded, V_padded, key_lengths):
    B, H, QL, D = Q.shape
    max_seq_len = K_padded.shape[2]

    # 调整 Q 形状为 Flash Attention 需要的 (B * QL, H, D)
    Q = Q.permute(0, 2, 1, 3).reshape(B * QL, H, D)
    
    # 将 K, V 调整为 Flash Attention 需要的形状 (B * H, L, D)
    K_list, V_list = [], []
    for i in range(B * H):
        length = key_lengths[i]
        K_list.append(K_padded[:, i % H, :length, :])  # 取有效的 K
        V_list.append(V_padded[:, i % H, :length, :])  # 取有效的 V
    
    K_flat = torch.cat(K_list, dim=1)  # (B * H, L, D)
    V_flat = torch.cat(V_list, dim=1)  # (B * H, L, D)
    
    # 计算 cu_seqlens_k
    cu_seqlens_k = torch.cat([torch.tensor([0], device=Q.device), torch.cumsum(torch.tensor(key_lengths, device=Q.device), dim=0)])

    return flash_attn_func(Q, K_flat, V_flat, dropout_p=0.0, softmax_scale=None, causal=False)

# **🚀 测试性能**
def benchmark(func, *args):
    func(*args)  # 预热
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):  # 运行 10 次取平均值
        func(*args)
    torch.cuda.synchronize()
    end_time = time.time()
    cost = (end_time - start_time) * 1000  # 转换为毫秒
    print(f"{func.__name__} 运行时间: {cost:.2f} ms")
    return cost

# **🚀 配置参数**
BATCH_SIZE = 1
NUM_HEADS = 64
QUERY_LEN = 1
DIM = 128

for i in range(1, 11):
    max_key_len = int(1024 * i)
    min_key_len = int(512 * i)
    
    # **生成变长 KV 长度**
    key_lengths = torch.randint(min_key_len, max_key_len, (BATCH_SIZE * NUM_HEADS,)).tolist()
    
    # **生成 Q, K, V**
    Q = torch.randn(BATCH_SIZE, NUM_HEADS, QUERY_LEN, DIM, device="cuda", dtype=torch.float16)
    K_padded = torch.zeros(BATCH_SIZE, NUM_HEADS, max_key_len, DIM, device="cuda", dtype=torch.float16)
    V_padded = torch.zeros(BATCH_SIZE, NUM_HEADS, max_key_len, DIM, device="cuda", dtype=torch.float16)
    attn_mask = torch.ones(BATCH_SIZE, NUM_HEADS, QUERY_LEN, max_key_len, device="cuda", dtype=torch.float16) * float("-inf")
    
    # **填充不同 head 的 KV 并生成 mask**
    for i, length in enumerate(key_lengths):
        K_padded[:, i // NUM_HEADS, :length, :] = torch.randn(BATCH_SIZE, 1, length, DIM, device="cuda", dtype=torch.float16)
        V_padded[:, i // NUM_HEADS, :length, :] = torch.randn(BATCH_SIZE, 1, length, DIM, device="cuda", dtype=torch.float16)
        attn_mask[:, i // NUM_HEADS, :, :length] = 0  # 允许访问的部分设为 0
    
    # **🚀 运行所有测试**
    print("\n=== SDPA vs Flash Attention 性能测试 ===")
    hete_attn = benchmark(sdpa_with_mask, Q, K_padded, V_padded, attn_mask)
    common_attn = benchmark(sdpa_no_mask, Q, K_padded, V_padded)
    flash_attn = benchmark(flash_attention, Q, K_padded, V_padded, key_lengths)
    
    # # **保存结果**
    # result = {"hete_attn": hete_attn, "common_attn": common_attn, "flash_attn": flash_attn}
    # with open("../cache_test/result/cache_attn_cost.jsonl", "a", encoding="utf-8") as file:
    #     file.write(json.dumps(result, ensure_ascii=False) + "\n")
