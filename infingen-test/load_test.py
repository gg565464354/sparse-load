import torch
import torch.nn.functional as F
import time

DEVICE = 'cpu'

def speculate_attention(hidden, p_w_q, p_k_c, n_head, alpha, max_num_kv):
    """Speculates the indices of the critical KV caches of next attention layer.

    On the decoding stage, by using the hidden states (layer i), partial query
    weight (layer i+1), and partial key cache (layer i+1), speculates the
    attention score of the next layer. After that, counts the number of
    critical tokens and gets the indcies of the top-k KV cache tokens with high
    attention scores.

    Args:
        hidden: Hidden states of layer i (b, 1, D)
        p_w_q: Partial query weight (D', D)
        p_k_c: Partial key cache (n, bh, d')

        Note that bh * d' == D'

    Returns:
        prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
    """
    b = hidden.shape[0]
    p_q = F.linear(hidden, p_w_q, bias=None)
    p_q = p_q.view(b, 1, n_head, -1)
    p_q = p_q.permute(0, 2, 1, 3).reshape(b * n_head, 1, -1)

    print(f"p_q shape = {p_q.shape}")
    print(f"p_k_c shape = {p_k_c.permute(1, 2, 0).shape}")


    p_attn = torch.bmm(p_q, p_k_c.permute(1, 2, 0))
    max_ = torch.max(p_attn, dim=-1)[0]
    thr_ = (max_ - alpha).unsqueeze(-1).repeat(1, 1, p_attn.shape[-1])
    count = torch.where(
        p_attn > thr_, torch.ones_like(p_attn), torch.zeros_like(p_attn)
    )
    mean = torch.mean(torch.sum(count, dim=-1)).item()
    prefetch_idx = torch.topk(
        p_attn.permute(2, 1, 0), min(int(mean), max_num_kv), dim=0
    )[1]

    return prefetch_idx

# 运行 select_kv
def select_kv(prefetch_idx, k_cache, v_cache):
    """Selects and aggregates critical KV caches using speculated indices

    On the decoding stage, aggregates the critical KV caches corresponding to
    the speculated prefetch index using embedding function.

    Args:
        prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
        k_cache: Key cache (n, bh, d)
        v_cache: Value cache (n, bh, d)

    Returns:
        selected_k: selected key cache (n', bh, d)
        selected_v: selected value cache (n', bh, d)
    """
    # print("prefetch_idx shape = ", prefetch_idx.shape)

    # start = time.time()
    prefetch_idx = prefetch_idx.squeeze().to(k_cache.device)
    ind = prefetch_idx * k_cache.shape[1] + torch.arange(k_cache.shape[1])[None, :]

    selected_k = F.embedding(ind, k_cache.reshape(-1, k_cache.shape[2]))
    selected_v = F.embedding(ind, v_cache.reshape(-1, v_cache.shape[2]))
    # end = time.time()

    return selected_k, selected_v, 0
    # return selected_k, selected_v, (end-start)

def rand_tensor(shape, a, b):
    # 生成服从标准正态分布的张量
    tensor = torch.randn(shape, device=DEVICE)

    # 将张量缩放到 [a, b] 范围并取整
    scaled_tensor = (b - a) * (tensor - tensor.min()) / (tensor.max() - tensor.min()) + a
    integer_tensor = torch.round(scaled_tensor).int()  # 四舍五入取整

    # print("rand_tensor.shape = ", integer_tensor.shape)

    return integer_tensor


# 假设的 key 和 value 缓存
N = 4096
N = 8192
# N = 10240
N_prime = int(N*0.5)

hit_rate = 0.8
N_cached = int(N_prime*hit_rate)
N_un_cached = int(N_prime*(1-hit_rate))
bh = 64
d = 128

#
TEST_CASE = 10
full_cost = []
cache_cost = []
full_cmp_cost = []
cache_cmp_cost = []

for i in range(TEST_CASE):
    prefetch_idx = rand_tensor((N_prime, 1, bh), 0, N_prime-1)  # (n=5, bh=2, d=3)
    un_cached_idx = rand_tensor((N_un_cached, 1, bh), 0, N_prime-1)
    cached_k = torch.randn((N_cached, bh, d), device=DEVICE)
    cached_v = cached_k + 0.1

    k_cache = torch.randn((N_prime, bh, d), device=DEVICE)
    v_cache = k_cache + 0.1  # 让 v_cache 和 k_cache 不同


    torch.cuda.synchronize()
    start1 = time.time()
    selected_k, selected_v, tcost = select_kv(prefetch_idx, k_cache, v_cache)
    gpu_tensor_k = selected_k.cuda()
    gpu_tensor_v = selected_v.cuda()
    torch.cuda.synchronize()
    end1 = time.time()
    full_cost.append(end1 - start1)
    full_cmp_cost.append(tcost)
    

    torch.cuda.synchronize()
    start2 = time.time()
    un_cached_k, un_cached_v, tcost = select_kv(un_cached_idx, k_cache, v_cache)
    gpu_uncached_k = un_cached_k.cuda()
    gpu_uncached_v = un_cached_v.cuda()

    gpu_cached_k = cached_k.cuda()
    gpu_cached_v = cached_v.cuda()

    final_k = torch.concat((gpu_cached_k, gpu_uncached_k), dim=0)
    final_v = torch.concat((gpu_cached_v, gpu_uncached_v), dim=0)

    torch.cuda.synchronize()
    end2 = time.time()
    cache_cost.append(end2 - start2)
    cache_cmp_cost.append(tcost)

avg_full_cost = sum(full_cost)/TEST_CASE
avg_cache_cost = sum(cache_cost)/TEST_CASE
print("avg_full_cost = ", avg_full_cost)
print("avg_cache_cost = ", avg_cache_cost)


# avg_full_cmp_cost = sum(full_cmp_cost)/TEST_CASE
# avg_cache_cmp_cost = sum(cache_cmp_cost)/TEST_CASE
# print("avg_full_cmp_cost = ", avg_full_cmp_cost)
# print("avg_cache_cmp_cost = ", avg_cache_cmp_cost)

# print("Selected K:")
# print(selected_k)
# print("Selected V:")
# print(selected_v)

################################### 


# # 测试代码
# b, D, D_prime, n, bh = 1, 4, 4, 5, 2  # batch_size, hidden_dim, proj_dim, num_tokens, num_heads
# alpha = 0.5
# max_num_kv = 5

# # 随机初始化输入
# torch.manual_seed(42)
# hidden = torch.randn(b, 1, D)
# p_w_q = torch.randn(D_prime*bh, D)
# p_k_c = torch.randn(n, bh, D_prime)

# # 运行函数
# prefetch_idx = speculate_attention(hidden, p_w_q, p_k_c, n_head=bh, alpha=alpha, max_num_kv=max_num_kv)


# # 输出结果
# print("Prefetch Indices Shape:", prefetch_idx.shape)
# print("Prefetch Indices:", prefetch_idx)