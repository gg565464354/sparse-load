import torch
import time

# 配置
N, D = 8000, 128   # 原始数据尺寸 (N, D)
M = 2000           # 每次随机取 M 行
BUFFER_SIZE = M    # Pinned Memory 缓冲区大小，等于每次传输数据量
TEST_NUM = 100

# 在 CPU 上创建 (8000, 128) 的随机 Tensor
cpu_tensor = torch.randn(N, D)
indices0 = torch.randint(0, N, (M,))

# warm up
selected_data = cpu_tensor[indices0]
gpu_tensor = selected_data.to("cuda")  # 直接传输
torch.cuda.synchronize()


# 方式 1：直接传输
cost1 = []
for i in range(TEST_NUM):
    indices1 = torch.randint(0, N, (M,))
    
    start_time = time.time()
    continue_data = cpu_tensor[indices1].contiguous()
    gpu_tensor = continue_data.to("cuda")  # 直接传输
    torch.cuda.synchronize()
    end_time = time.time()
    
    cost1.append(end_time - start_time)
avg_cost1 = sum(cost1)/len(cost1)

print(f"直接传输时间: {(avg_cost1) * 1000:.2f} ms")


# 方式 2：使用 Pinned Memory 缓存区
pinned_buffer = torch.empty(BUFFER_SIZE, D, pin_memory=True)  # 预分配 Pinned Memory
cost2 = []
for i in range(TEST_NUM):
    indices2 = torch.randint(0, N, (M,))
    start_time = time.time()
    pinned_buffer.copy_(cpu_tensor[indices2])  # 先拷贝到 Pinned Memory
    gpu_tensor = pinned_buffer.to("cuda", non_blocking=True)  # 再传输到 GPU
    torch.cuda.synchronize()
    end_time = time.time()
    cost2.append(end_time - start_time)
avg_cost2 = sum(cost2)/len(cost2)
print(f"Pinned Memory 缓存传输时间: {(avg_cost2) * 1000:.2f} ms")


# 方式 3：使用 Pinned Memory 缓存区 + 稀疏加载
# cache_size = BUFFER_SIZE * 
hit_rate = 0.6
pinned_buffer = torch.empty(BUFFER_SIZE, D, pin_memory=True)  # 预分配 Pinned Memory
cost2 = []
for i in range(TEST_NUM):
    indices2 = torch.randint(0, N, (M,))
    
    un_hit_num = int(M*(1-hit_rate))
    un_hit_ids = torch.randint(0, N, (un_hit_num,))

    start_time = time.time()
    # hot
    # pinned_buffer.copy_(cpu_tensor[indices2])  # 先拷贝到 Pinned Memory
    gpu_tensor = pinned_buffer.to("cuda", non_blocking=True)  # 再传输到 GPU

    # cold
    continue_un_hit = cpu_tensor[un_hit_ids].contiguous()
    gpu_tensor = continue_un_hit.to("cuda")
    
    torch.cuda.synchronize()
    end_time = time.time()
    cost2.append(end_time - start_time)
avg_cost2 = sum(cost2)/len(cost2)
print(f"Pinned Memory 缓存传输时间: {(avg_cost2) * 1000:.2f} ms")


# 更新pinned memory需要的时间
