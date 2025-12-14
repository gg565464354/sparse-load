import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------- 模拟 OPT 6.7B 的 Decoder Layer ---------
class OPTDecoderLayer(nn.Module):
    def __init__(self, hidden_size=4096, num_heads=32, ffn_dim=16384):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, hidden_size)
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, k, v):
        # print("shape k = ", k.shape)

        attn_output, _ = self.self_attn(x, k, v)
        x = x + attn_output
        x = self.ln1(x)

        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.ln2(x)
        return x

# --------- 新的稀疏 KV 选择函数（基于 F.embedding）---------
def select_kv(prefetch_idx, k_cache, v_cache):
    prefetch_idx = prefetch_idx.squeeze().to(k_cache.device)
    ind = prefetch_idx * k_cache.shape[1] + torch.arange(k_cache.shape[1])[None, :]
    selected_k = F.embedding(ind, k_cache.reshape(-1, k_cache.shape[2]))
    selected_v = F.embedding(ind, v_cache.reshape(-1, v_cache.shape[2]))
    return selected_k, selected_v


# --------- 测试设置 ---------
hidden_size = 4096
num_heads = 32
ffn_dim = 16384
batch_size = 1
head_dim = hidden_size // num_heads

test_num = 10
seq_lens = [1024 * i for i in range(1, 1 + test_num)]
n_warmup = 10
n_repeat = 10

sparse_rate = 0.2

decoder_layer = OPTDecoderLayer(hidden_size, num_heads, ffn_dim).half().to(device)
timing_decode = []
timing_transfer = []
timing_sparse_transfer = []
timing_sparse_decode = []

# 预热
print("Running warmup...")
for _ in range(n_warmup):
    test_seq_len = 512
    dummy_x = torch.randn(batch_size, 1, hidden_size, dtype=torch.float16, device=device)
    dummy_k = torch.randn(test_seq_len, batch_size * num_heads, head_dim, dtype=torch.float16, device=device)
    dummy_v = torch.randn(test_seq_len, batch_size * num_heads, head_dim, dtype=torch.float16, device=device)
    dummy_k = dummy_k.permute(1, 0, 2).reshape(batch_size, test_seq_len, -1)
    dummy_v = dummy_v.permute(1, 0, 2).reshape(batch_size, test_seq_len, -1)
    _ = decoder_layer(dummy_x, dummy_k, dummy_v)
torch.cuda.synchronize()

print("Start benchmark...")
for seq_len in seq_lens:
    total_decode_time = 0.0
    total_transfer_time = 0.0
    total_sparse_time = 0.0
    total_sparse_decode_time = 0.0

    for i in range(n_repeat+1):
        q = torch.randn(batch_size, 1, hidden_size, dtype=torch.float16, device=device)

        # CPU上的k和v: (seq_len, batch_size*num_heads, head_dim)
        k_cpu = torch.randn(seq_len, batch_size * num_heads, head_dim, dtype=torch.float16, device="cpu")
        v_cpu = torch.randn(seq_len, batch_size * num_heads, head_dim, dtype=torch.float16, device="cpu")
        
        k_cpu_2 = k_cpu.permute(1, 0, 2).reshape(batch_size, seq_len, -1)
        v_cpu_2 = v_cpu.permute(1, 0, 2).reshape(batch_size, seq_len, -1)

        # # --------- 1. 全量 KV Cache Transfer 时间 + Decoder 时间 ---------
        # torch.cuda.synchronize()
        # start = time.time()
        k_gpu = k_cpu_2.to(device)
        v_gpu = v_cpu_2.to(device)
        # torch.cuda.synchronize()
        # transfer_time = time.time()

        _ = decoder_layer(q, k_gpu, v_gpu)
        # torch.cuda.synchronize()
        # finish_time = time.time()
        # total_transfer_time += (transfer_time - start) * 1000
        # total_decode_time += (finish_time - transfer_time) * 1000

        # --------- 2. 稀疏 KV Cache 选取 + 传输时间 + Decoder 时间 ---------
        sparse_num = max(int(seq_len * sparse_rate), 1)
        prefetch_idx = torch.randint(0, seq_len, (sparse_num, 1, batch_size * num_heads), device="cpu")
        sparse_k_gpu = k_gpu[:, :sparse_num, :].contiguous()
        sparse_v_gpu = v_gpu[:, :sparse_num, :].contiguous()

        torch.cuda.synchronize()
        start = time.time()

        # 使用新的 select_kv 实现
        selected_k, selected_v = select_kv(prefetch_idx, k_cpu, v_cpu)
        selected_k = selected_k.to(device)
        selected_v = selected_v.to(device)

        torch.cuda.synchronize()
        transfer_time = time.time()

        _ = decoder_layer(q, sparse_k_gpu, sparse_v_gpu)
        torch.cuda.synchronize()
        finish_time = time.time()

        total_sparse_time += (transfer_time - start) * 1000
        total_sparse_decode_time += (finish_time - transfer_time) * 1000

        # print(f"total_sparse_decode_time = {total_sparse_decode_time}, finish_time={finish_time}, transfer_time={transfer_time}")

        if i == 0:
            total_sparse_decode_time = 0
            total_sparse_time = 0

        # 可选：测试稀疏 KV 的计算时间
        # _ = decoder_layer(q, selected_k, selected_v)

    # avg_decode = total_decode_time / n_repeat
    # avg_transfer = total_transfer_time / n_repeat
    avg_sparse = total_sparse_time / n_repeat
    avg_sparse_decode = total_sparse_decode_time / n_repeat
    
    # timing_decode.append(avg_decode)
    # timing_transfer.append(avg_transfer)
    timing_sparse_transfer.append(avg_sparse)
    timing_sparse_decode.append(avg_sparse_decode)

    # print(f"SeqLen={seq_len:>4} | Decode: {avg_decode:.3f} ms | Transfer: {avg_transfer:.3f} ms | Sparse Transfer: {avg_sparse:.3f} ms | Sparse Decode: {avg_sparse_decode:.3f} ms")

    print(f"Sparse Transfer: {avg_sparse:.3f} ms | Sparse Decode: {avg_sparse_decode:.3f} ms")

# 画图
plt.figure(figsize=(10, 6))
# plt.plot(seq_lens, timing_decode, label="Decoder Compute (ms)", marker='o')
# plt.plot(seq_lens, timing_transfer, label="Full KV Transfer (ms)", marker='s')
plt.plot(seq_lens, timing_sparse_transfer, label="Sparse Load", marker='^')
plt.plot(seq_lens, timing_sparse_decode, label="Sparse Decode", marker='^')
plt.xlabel("KV Cache Length")
plt.ylabel("Time (ms)")
plt.title("OPT 6.7B: Decoder vs KV Transfer vs Sparse Transfer")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("insight_1.png")
print("Saved plot to insight_1.png")