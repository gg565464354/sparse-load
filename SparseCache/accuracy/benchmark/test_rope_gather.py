#!/usr/bin/env python3
"""测试RoPE gather逻辑是否正确"""
import torch

print("=" * 60)
print("测试RoPE gather逻辑")
print("=" * 60)

# 模拟参数
bs = 1
seq_len = 5
max_seq_len = 10
head_dim = 128
num_heads = 32

# 创建测试数据
position_ids = torch.tensor([[0, 1, 2, 3, 4]])  # [1, 5]
cos_cached = torch.arange(max_seq_len * head_dim).reshape(1, 1, max_seq_len, head_dim).float()

print(f"\n输入形状:")
print(f"  position_ids: {position_ids.shape} = {list(position_ids.shape)}")
print(f"  cos_cached: {cos_cached.shape} = {list(cos_cached.shape)}")
print(f"  position_ids值: {position_ids}")

# 当前的gather逻辑
print(f"\n当前gather逻辑:")
gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
print(f"  Step 1 - gather_indices: {gather_indices.shape}")

gather_indices = gather_indices.repeat(1, cos_cached.shape[1], 1, cos_cached.shape[3])
print(f"  Step 2 - gather_indices after repeat: {gather_indices.shape}")
print(f"  (repeated with cos.shape[1]={cos_cached.shape[1]}, cos.shape[3]={cos_cached.shape[3]})")

cos_result = torch.gather(cos_cached.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
print(f"  Step 3 - cos_result: {cos_result.shape}")

# 验证结果
print(f"\n验证gather结果:")
print(f"  cos_cached[0, 0, 0, :5] (position 0的前5个值): {cos_cached[0, 0, 0, :5]}")
print(f"  cos_result[0, 0, 0, :5] (gather后position 0的前5个值): {cos_result[0, 0, 0, :5]}")
print(f"  cos_cached[0, 0, 2, :5] (position 2的前5个值): {cos_cached[0, 0, 2, :5]}")
print(f"  cos_result[0, 0, 2, :5] (gather后position 2的前5个值): {cos_result[0, 0, 2, :5]}")

# 测试增量生成场景
print(f"\n" + "=" * 60)
print("测试增量生成场景（KV cache）")
print("=" * 60)

position_ids_gen = torch.tensor([[5]])  # 生成第6个token，position_id=5
print(f"  position_ids: {position_ids_gen} (shape: {position_ids_gen.shape})")

gather_indices_gen = position_ids_gen[:, None, :, None]
gather_indices_gen = gather_indices_gen.repeat(1, 1, 1, head_dim)
print(f"  gather_indices: {gather_indices_gen.shape}")

cos_result_gen = torch.gather(cos_cached, 2, gather_indices_gen)
print(f"  cos_result: {cos_result_gen.shape}")
print(f"  cos_cached[0, 0, 5, :5]: {cos_cached[0, 0, 5, :5]}")
print(f"  cos_result[0, 0, 0, :5]: {cos_result_gen[0, 0, 0, :5]}")

if torch.allclose(cos_cached[0, 0, 5, :], cos_result_gen[0, 0, 0, :]):
    print("  ✅ gather正确提取了position 5的值")
else:
    print("  ❌ gather结果不正确!")

print("\n" + "=" * 60)
