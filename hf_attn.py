import torch
import torch.nn.functional as F
import time
from collections import defaultdict
import json
from transformers.modeling_flash_attention_utils import _flash_attention_forward


# **🚀 1. 标准 SDPA（带 Mask）**
def sdpa_with_mask(Q, K_padded, V_padded, attn_mask):
    # return F.scaled_dot_product_attention(Q, K_padded, V_padded, attn_mask=attn_mask)

    _flash_attention_forward(Q, K_padded, V_padded, attention_mask=attn_mask, query_length=1, is_causal=True)

# **🚀 2. 直接去掉 Mask 的 SDPA**
def sdpa_no_mask(Q, K_padded, V_padded):
    return F.scaled_dot_product_attention(Q, K_padded, V_padded)  # 🚀 最高效（如果 mask 仅用于 padding）

# **🚀 3. 仅裁剪 K/V，而不使用 Mask**
def sdpa_trimmed_kv(Q, K_padded, V_padded):
    outputs = []
    for head_idx, key_len in enumerate(key_lengths):
        Q_head = Q[:, head_idx:head_idx+1, :, :]  # 取该 head 的 Q
        K_head = K_padded[:, head_idx:head_idx+1, :key_len, :]  # 只取有效的 K
        V_head = V_padded[:, head_idx:head_idx+1, :key_len, :]  # 只取有效的 V

        output = F.scaled_dot_product_attention(Q_head, K_head, V_head)  # 计算 SDPA
        outputs.append(output)

    return torch.cat(outputs, dim=1)  # 重新拼接所有 head 计算结果


# **🚀 4. Grouped SDPA（针对不同 Key 长度分组计算）**
def sdpa_grouped(Q, K_padded, V_padded):
    grouped_heads = defaultdict(list)
    for head_idx, length in enumerate(key_lengths):
        grouped_heads[length].append(head_idx)

    outputs = []
    for length, head_list in grouped_heads.items():
        Q_group = Q[:, head_list, :, :]
        K_group = K_padded[:, head_list, :length, :]
        V_group = V_padded[:, head_list, :length, :]

        output = F.scaled_dot_product_attention(Q_group, K_group, V_group)
        outputs.append(output)

    return torch.cat(outputs, dim=1)

# **🚀 测试性能**
def benchmark(func, *args):
    # 预热
    func(*args)
    torch.cuda.synchronize()

    # 正式测试
    start_time = time.time()
    for _ in range(10):  # 运行 10 次取平均值
        func(*args)
    torch.cuda.synchronize()
    end_time = time.time()

    cost = end_time - start_time

    print(f"{func.__name__} 运行时间: {(cost) * 1000:.2f} ms")
    return cost


# 配置参数
BATCH_SIZE = 1
NUM_HEADS = 16
QUERY_LEN = 1
DIM = 128

for i in range(1, 11):
    max_key_len = int(1024*i)
    min_key_len = int(512*i)

    # 模拟不同 head 具有不同 key-value 长度的情况
    key_lengths = torch.randint(min_key_len, max_key_len, (NUM_HEADS,)).tolist()  # 每个 head 的 key 长度


    # 生成 Query, Key, Value
    Q = torch.randn(BATCH_SIZE, NUM_HEADS, QUERY_LEN, DIM, device="cuda", dtype=torch.float16)
    K_padded = torch.zeros(BATCH_SIZE, NUM_HEADS, max_key_len, DIM, device="cuda", dtype=torch.float16)
    V_padded = torch.zeros(BATCH_SIZE, NUM_HEADS, max_key_len, DIM, device="cuda", dtype=torch.float16)

    # 生成 Mask
    attn_mask = torch.zeros(BATCH_SIZE, NUM_HEADS, QUERY_LEN, max_key_len, device="cuda", dtype=torch.float16)

    # 逐个填充不同 head 的 key-value，并生成 mask
    for i, length in enumerate(key_lengths):
        K_padded[:, i, :length, :] = torch.randn(BATCH_SIZE, 1, length, DIM, device="cuda", dtype=torch.float16)
        V_padded[:, i, :length, :] = torch.randn(BATCH_SIZE, 1, length, DIM, device="cuda", dtype=torch.float16)
        attn_mask[:, i, :, length:] = float("-inf")  # Mask padding 部分


    # **🚀 运行所有测试**
    print("\n=== SDPA 性能测试 ===")
    hete_attn = benchmark(sdpa_with_mask, Q, K_padded, V_padded, attn_mask)
    common_attn = benchmark(sdpa_no_mask, Q, K_padded, V_padded)
    # benchmark(sdpa_trimmed_kv, Q, K_padded, V_padded)
    # benchmark(sdpa_grouped, Q, K_padded, V_padded)

    result = {"hete_attn": hete_attn, "common_attn":common_attn}

    with open("../cache_test/result/cache_attn_cost.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")
