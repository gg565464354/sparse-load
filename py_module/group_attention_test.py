import torch
import time
from typing import List, Tuple

NUM_RUNS = 10

class GroupInfo:
    def __init__(self, head_indices: List[int], max_kv_len: int):
        self.head_indices = head_indices
        self.max_kv_len = max_kv_len

def standard_attention(query, key, value, scale_factor, mask=None):
    attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scale_factor
    if mask is not None and mask.numel() > 0:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_weights, value)

def group_attention_core(group_q, group_k, group_v, group_heads, output):
    start = 0
    for q, k, v, heads in zip(group_q, group_k, group_v, group_heads):
        # print("k shape = ", k.shape)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        # print(f"output shape = {output.shape}, attn_out shape = {attn_out.shape}")
        # output.index_copy_(1, heads, attn_out)
        # output[:, heads, :, :] = attn_out
        output[:, start:start+len(heads), :, :] = attn_out
        start += len(heads)
    return output

def groupwise_flash_attention(query, key, value, groups: List[GroupInfo]) -> Tuple[torch.Tensor, int]:
    output = torch.zeros_like(query, device=query.device)
    group_k, group_v, group_q, group_heads = [], [], [], []

    for group in groups:
        heads = torch.tensor(group.head_indices, dtype=torch.long, device=query.device)
        q = query.index_select(1, heads)
        k = key.index_select(1, heads).narrow(2, 0, group.max_kv_len)
        v = value.index_select(1, heads).narrow(2, 0, group.max_kv_len)
        group_q.append(q)
        group_k.append(k)
        group_v.append(v)
        group_heads.append(heads)

    torch.cuda.synchronize()
    start = time.time()

    group_attention_core(group_q, group_k, group_v, group_heads, output)

    torch.cuda.synchronize()
    end = time.time()
    communication_time_us = int((end - start) * 1e6)
    return output, communication_time_us

def grouped_attention(query, key, value, groups: List[GroupInfo], scale_factor: float, mask=None):
    output = torch.zeros_like(query)
    for group in groups:
        heads = torch.tensor(group.head_indices, dtype=torch.long, device=query.device)
        q = query.index_select(1, heads)
        k = key.index_select(1, heads).narrow(2, 0, group.max_kv_len)
        v = value.index_select(1, heads).narrow(2, 0, group.max_kv_len)
        m = mask.index_select(1, heads).narrow(3, 0, group.max_kv_len) if mask is not None else None
        attn = standard_attention(q, k, v, scale_factor, m)
        output.index_copy_(1, heads, attn)
    return output

def generate_test_data(batch=2, num_heads=6, seq_len=8, dim=64, device='cuda'):
    groups = [
        GroupInfo([], seq_len),
        GroupInfo([], int(3 * seq_len / 4)),
        # GroupInfo([], int(1 * seq_len / 4))
    ]
    group_num = 2
    half_num_head = num_heads // group_num
    for i in range(num_heads):
        gid = i % group_num
        groups[gid].head_indices.append(i)

    query = torch.randn(batch, num_heads, 1, dim, device=device)
    key = torch.randn(batch, num_heads, seq_len, dim, device=device)
    value = torch.randn(batch, num_heads, seq_len, dim, device=device)

    for group in groups:
        if group.max_kv_len < seq_len:
            for head in group.head_indices:
                key[:, head, group.max_kv_len:] = 0
                value[:, head, group.max_kv_len:] = 0
    return query, key, value, groups

def build_mask(batch, heads, q_len, kv_len, groups: List[GroupInfo], device='cuda'):
    mask = torch.ones(batch, heads, q_len, kv_len, device=device)
    for group in groups:
        for head in group.head_indices:
            mask[:, head, :, group.max_kv_len:] = 0
    return mask

def benchmark(func, warmup=2, runs=NUM_RUNS):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        func()
    torch.cuda.synchronize()
    end = time.time()
    return int((end - start) * 1e6 / runs)

def test_attention_equivalence():
    torch.manual_seed(42)
    device = 'cuda'
    query, key, value, groups = generate_test_data(16, 32, 1000, 128, device)
    scale = 1.0 / (query.size(-1) ** 0.5)
    mask = build_mask(query.size(0), query.size(1), query.size(2), key.size(2), groups, device)
    bool_mask = mask.to(torch.bool)

    query_fp16 = query.to(torch.half)
    key_fp16 = key.to(torch.half)
    value_fp16 = value.to(torch.half)

    def flash_func():
        return torch.nn.functional.scaled_dot_product_attention(query_fp16, key_fp16, value_fp16, attn_mask=bool_mask, is_causal=False)

    def flash_nomask_func():
        return torch.nn.functional.scaled_dot_product_attention(query_fp16, key_fp16, value_fp16, attn_mask=None, is_causal=False)


    flash_time = benchmark(flash_func)
    
    flash_nomask_time = benchmark(flash_nomask_func)
    print("standard success")

    print("groups indices = ", [g.head_indices for g in groups])
    print("groups max len = ", [g.max_kv_len for g in groups])

    grouped_time_total = 0
    for _ in range(NUM_RUNS):
        out_grouped, grouped_time = groupwise_flash_attention(query_fp16, key_fp16, value_fp16, groups)
        grouped_time_total += grouped_time
    grouped_time_avg = grouped_time_total // NUM_RUNS

    out_flash = flash_func()
    diff = (out_flash - out_grouped).abs().max().item()

    print(f"最大差异: {diff}")
    print(f"dtype: {out_flash.dtype}")
    print("=== 多轮平均性能对比 ===")
    print(f"Flash/SDPA Attention 平均耗时: {flash_time} us")
    print(f"Flash/SDPA Attention no mask 平均耗时: {flash_nomask_time} us")
    print(f"Grouped Attention 平均耗时:    {grouped_time_avg} us")
    print(f"加速比 (Grouped vs Flash):    {flash_time / grouped_time_avg:.2f}x")

if __name__ == "__main__":
    test_attention_equivalence()
