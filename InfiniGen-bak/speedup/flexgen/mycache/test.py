import torch

# 假设尺寸 b=4, n_head=8, head_dim=64, k=3
b, _, n_head, head_dim = 4, 1, 8, 64
k = 3

# 创建一个随机的 query tensor 作为例子
query_tensor = torch.randn(b, 1, n_head, head_dim)

# 给定 id_list，长度为 k，例如我们想要索引第 1, 3, 和第 6 个 head（注意这里是从0开始计数）
id_list = torch.tensor([0, 2, 5], dtype=torch.long)  # 确保类型为 long

# 使用 index_select 方法来选择指定的 heads
selected_heads = query_tensor.index_select(dim=2, index=id_list)

# 检查是否连续
print("Is the output tensor contiguous after index_select? ", selected_heads.is_contiguous())

# 确保输出张量是连续的
output_tensor = selected_heads.contiguous()

# 再次检查是否连续
print("Is the output tensor contiguous after calling contiguous()? ", output_tensor.is_contiguous())

print(output_tensor.shape)  # 应该输出 torch.Size([4, 1, 3, 64])