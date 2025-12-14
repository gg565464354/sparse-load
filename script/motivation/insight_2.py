import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

def select_kv(prefetch_idx, k_cache, v_cache):
    prefetch_idx = prefetch_idx.squeeze().to(k_cache.device)
    ind = prefetch_idx * k_cache.shape[1] + torch.arange(k_cache.shape[1])[None, :]
    selected_k = F.embedding(ind, k_cache.reshape(-1, k_cache.shape[2]))
    selected_v = F.embedding(ind, v_cache.reshape(-1, v_cache.shape[2]))
    return selected_k, selected_v

# 参数设置
seq_len_list = [512, 1024, 2048, 4096, 8192]
batch_size = 4
head_num = 32
n_head = head_num * batch_size
dim = 128

select_times = []
transfer_times = []
total_times = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Warmup（使用中等长度）
print("Running warmup...")
with torch.no_grad():
    warm_seq_len = 1024
    n_prime = warm_seq_len // 4
    warm_prefetch_idx = torch.randint(0, warm_seq_len, (n_prime, 1, n_head), dtype=torch.long)
    warm_k_cache = torch.randn(warm_seq_len, n_head, dim)
    warm_v_cache = torch.randn(warm_seq_len, n_head, dim)
    selected_k, selected_v = select_kv(warm_prefetch_idx, warm_k_cache, warm_v_cache)
    selected_k = selected_k.to(device, non_blocking=True)
    selected_v = selected_v.to(device, non_blocking=True)
torch.cuda.synchronize()
print("Warmup done.\n")

# 正式测试
for seq_len in seq_len_list:
    with torch.no_grad():
        n_prime = seq_len // 4
        prefetch_idx = torch.randint(0, seq_len, (n_prime, 1, n_head), dtype=torch.long)
        k_cache = torch.randn(seq_len, n_head, dim)
        v_cache = torch.randn(seq_len, n_head, dim)

        # 选择 KV 时间测量
        torch.cuda.synchronize()
        t0 = time.time()
        selected_k, selected_v = select_kv(prefetch_idx, k_cache, v_cache)
        torch.cuda.synchronize()
        t1 = time.time()
        select_time = t1 - t0

        # 传输时间测量
        torch.cuda.synchronize()
        t2 = time.time()
        selected_k_gpu = selected_k.to(device, non_blocking=True)
        selected_v_gpu = selected_v.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t3 = time.time()
        transfer_time = t3 - t2

        # 总耗时
        total_time = select_time + transfer_time

        # 记录
        select_times.append(select_time)
        transfer_times.append(transfer_time)
        total_times.append(total_time)

        select_rate = 100 * select_time/total_time
        transfer_rate = 100 * transfer_time/total_time

        print(f"[Seq_len={seq_len}] Select Time: {select_time:.6f}s ({select_rate:.2f}%), "
              f"Transfer Time: {transfer_time:.6f}s ({transfer_rate:.2f}%), Total: {total_time:.6f}s")



# 绘图
plt.figure(figsize=(10, 6))
plt.plot(seq_len_list, transfer_times, label='Continue Transfer', marker='s')
plt.plot(seq_len_list, total_times, label='Discrete Transfer', marker='^')
plt.plot(seq_len_list, select_times, label='Data Select', marker='o')
plt.xlabel('KV Cache Length (seq_len)')
plt.ylabel('Time (s)')
plt.title('Discrete vs. Continue Transfer Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("insight_2.png")
