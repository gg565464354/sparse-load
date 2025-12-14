import torch
import torch.nn.functional as F
import sys

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

    prefetch_idx = prefetch_idx.squeeze().to(k_cache.device)
    ind = prefetch_idx * k_cache.shape[1] + torch.arange(k_cache.shape[1])[None, :]
    selected_k = F.embedding(ind, k_cache.reshape(-1, k_cache.shape[2]))
    selected_v = F.embedding(ind, v_cache.reshape(-1, v_cache.shape[2]))
    return selected_k, selected_v


# def speculate_attention(hidden, p_w_q, p_k_c, n_head, alpha, max_num_kv):
#     """Speculates the indices of the critical KV caches of next attention layer.

#     On the decoding stage, by using the hidden states (layer i), partial query
#     weight (layer i+1), and partial key cache (layer i+1), speculates the
#     attention score of the next layer. After that, counts the number of
#     critical tokens and gets the indcies of the top-k KV cache tokens with high
#     attention scores.

#     Args:
#         hidden: Hidden states of layer i (b, 1, D)
#         p_w_q: Partial query weight (D', D)
#         p_k_c: Partial key cache (n, bh, d')

#         Note that bh * d' == D'

#     Returns:
#         prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
#     """
#     b = hidden.shape[0]
#     p_q = F.linear(hidden, p_w_q, bias=None)
#     p_q = p_q.view(b, 1, n_head, -1)
#     p_q = p_q.permute(0, 2, 1, 3).reshape(b * n_head, 1, -1)

#     p_attn = torch.bmm(p_q, p_k_c.permute(1, 2, 0))
#     max_ = torch.max(p_attn, dim=-1)[0]
#     thr_ = (max_ - alpha).unsqueeze(-1).repeat(1, 1, p_attn.shape[-1])
#     count = torch.where(
#         p_attn > thr_, torch.ones_like(p_attn), torch.zeros_like(p_attn)
#     )
#     mean = torch.mean(torch.sum(count, dim=-1)).item()
#     prefetch_idx = torch.topk(
#         p_attn.permute(2, 1, 0), min(int(mean), max_num_kv), dim=0
#     )[1]

#     return prefetch_idx

cnt = 0

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
    
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    b = hidden.shape[0]
    p_q = F.linear(hidden, p_w_q, bias=None)
    p_q = p_q.view(b, 1, n_head, -1)
    p_q = p_q.permute(0, 2, 1, 3).reshape(b * n_head, 1, -1)

    p_attn = torch.bmm(p_q, p_k_c.permute(1, 2, 0))

    # max_ = torch.max(p_attn, dim=-1)[0]
    # thr_ = (max_ - alpha).unsqueeze(-1).repeat(1, 1, p_attn.shape[-1])
    # count = torch.where(
    #     p_attn > thr_, torch.ones_like(p_attn), torch.zeros_like(p_attn)
    # )
    # mean = torch.mean(torch.sum(count, dim=-1)).item()

    tmp_p_attn = p_attn.permute(2, 1, 0)
    # prefetch_idx = torch.topk(
    #     tmp_p_attn, min(int(mean), max_num_kv), dim=0
    # )[1]
    
    prefetch_idx = torch.topk(
        tmp_p_attn, max_num_kv, dim=0
    )[1]

    
    # global cnt
    # # path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_inf_l{cnt}.pt"
    # # path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_b16_l{cnt}.pt"
    # path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_inf_b16_l{cnt}.pt"
    # cnt += 1
    # torch.save(prefetch_idx, path)

    return prefetch_idx



def new_speculate_attention(hidden, p_w_q, p_k_c, n_head, alpha, max_num_kv):
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
    
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # print(hidden.dtype, p_w_q.dtype, p_k_c.dtype)  # 如果是 torch.float16 或 bfloat16 就可能出现误差
    # print("kv_select hidden close", torch.allclose(hidden[0], hidden[1], atol=1e-6))
    
    b = hidden.shape[0]
    
    # if b > 1:
    #     torch.allclose(hidden[0], hidden[1], atol=1e-6)

    # hidden_fp32 = hidden.float()

    # print("kv_select hidden close", torch.allclose(hidden[0], hidden[1], atol=1e-6))
    if b > 1:
        ready = torch.allclose(hidden[0], hidden[1], atol=1e-6)
        # ready_cnt = 0
        while ready == False:
            ready = torch.allclose(hidden[0], hidden[1], atol=1e-6)
            # ready_cnt += 1
            print("ready")
            # if ready_cnt > 100:
            #     sys.exit(0)

    # print("kv_select 1 hidden close", )
    # print("kv_select 2 hidden close", torch.allclose(hidden[0], hidden[1], atol=1e-6))

    p_q = F.linear(hidden, p_w_q, bias=None)
    # if b > 1:
    #     while not torch.allclose(hidden[0], hidden[1], atol=1e-6):
    #         p_q = F.linear(hidden, p_w_q, bias=None)
    #         print("here?")

    # print("kv_select pre p_q close = ", torch.allclose(p_q[0], p_q[1], atol=1e-6))

    p_q = p_q.view(b, 1, n_head, -1)
    p_q = p_q.permute(0, 2, 1, 3).reshape(b * n_head, 1, -1)

    p_attn = torch.bmm(p_q, p_k_c.permute(1, 2, 0)) # (bh, 1, d) * (bh, d, n)
    # max_ = torch.max(p_attn, dim=-1)[0]
    # thr_ = (max_ - alpha).unsqueeze(-1).repeat(1, 1, p_attn.shape[-1])
    # count = torch.where(
    #     p_attn > thr_, torch.ones_like(p_attn), torch.zeros_like(p_attn)
    # )
    # mean = torch.mean(torch.sum(count, dim=-1)).item()

    tmp_p_attn = p_attn.permute(2, 1, 0)
    
    prefetch_idx = torch.topk(
        tmp_p_attn, max_num_kv, dim=0
    )[1]
    
    pad_idx = torch.topk(
        -tmp_p_attn, 1, dim=0
    )[1]

    # global cnt
    # # path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_inf_l{cnt}.pt"
    # path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_new_b16_l{cnt}.pt"
    # # path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_my_unhit_l{cnt}.pt"
    # # path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_my_unhit_l{cnt}.pt"
    # cnt += 1
    # torch.save(prefetch_idx, path)

    # if b > 1:
    #     print("kv_select hidden close", torch.allclose(hidden[0], hidden[1], atol=1e-6))
    #     print("kv_select p_q close", torch.allclose(p_q[0], p_q[n_head], atol=1e-6))
    #     print("kv_select p_attn close", torch.allclose(p_attn[0], p_attn[n_head], atol=1e-6))
    

    # hidden_cmp = [torch.equal(hidden[i], hidden[i+1]) for i in range(b-1)]
    # print("kv_select: hidden_cmp = ", hidden_cmp)

    # p_k_c_cmp = []
    # p_k_c_T = p_k_c.permute(1, 0, 2) # bh, n, d
    # for i in range(32):
    #     pidx_h1 = p_k_c_T[i]
    #     pidx_h2 = p_k_c_T[i+32]
    #     p_k_c_cmp.append(torch.equal(pidx_h1, pidx_h2))
    # print("kv_select: p_k_c_cmp = ", p_k_c_cmp)

    
    # tmp_p_attn_cmp = []
    # tmp_p_attn_T = tmp_p_attn.permute(2, 1, 0) # bh, n, d
    # for i in range(32):
    #     pidx_h1 = tmp_p_attn_T[i]
    #     pidx_h2 = tmp_p_attn_T[i+32]
    #     tmp_p_attn_cmp.append(torch.equal(pidx_h1, pidx_h2))
    # print("kv_select: tmp_p_attn_cmp = ", tmp_p_attn_cmp)

    
    # prefetch_idx_cmp = []
    # prefetch_idx_T = prefetch_idx.squeeze(1).T
    # for i in range(32):
    #     pidx_h1 = prefetch_idx_T[i]
    #     pidx_h2 = prefetch_idx_T[i+32]
    #     prefetch_idx_cmp.append(torch.equal(pidx_h1, pidx_h2))
    
    # print("kv_select: prefetch_idx_cmp = ", prefetch_idx_cmp)
    
    return prefetch_idx, pad_idx



# def new_speculate_attention(hidden, p_w_q, p_k_c, n_head, alpha, max_num_kv):
#     """
#     Speculates the indices of the critical KV caches of next attention layer.

#     Args:
#         hidden: Hidden states of layer i (b, 1, D)
#         p_w_q: Partial query weight (D', D)
#         p_k_c: Partial key cache (n, bh, d')

#     Returns:
#         prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
#     """
#     torch.use_deterministic_algorithms(True)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     print("[dtype check]", hidden.dtype, p_w_q.dtype, p_k_c.dtype)

#     b = hidden.shape[0]

#     # === Step 1: float32 精度计算 F.linear ===
#     print("kv_select hidden close", torch.allclose(hidden[0], hidden[1], atol=1e-6))

#     hidden_fp32 = hidden.float()
#     p_w_q_fp32 = p_w_q.float()

#     print("kv_select hidden close", torch.allclose(hidden[0], hidden[1], atol=1e-6))
#     print("kv_select hidden_fp32 close", torch.allclose(hidden_fp32[0], hidden_fp32[1], atol=1e-6))

#     # p_q = F.linear(hidden_fp32, p_w_q_fp32, bias=None)  # (b, 1, D')
#     p_q = torch.matmul(hidden_fp32, p_w_q_fp32.T)  # (b, 1, D')

#     print("hidden mean:", hidden_fp32.mean().item())
#     print("hidden std:", hidden_fp32.std().item())
#     print("hidden abs max:", hidden_fp32.abs().max().item())

#     print("kv_select p_q close", torch.allclose(p_q[0], p_q[1], atol=1e-6))

#     p_q = p_q.view(b, 1, n_head, -1)                    # (b, 1, nh, d')
#     p_q = p_q.permute(0, 2, 1, 3).reshape(b * n_head, 1, -1)  # (bh, 1, d')

#     # === Step 2: 注意力计算也转成 float32 后 half 回来 ===
#     p_k_c_fp32 = p_k_c.float()                          # (n, bh, d')
#     p_attn = torch.bmm(p_q, p_k_c_fp32.permute(1, 2, 0))  # (bh, 1, n)
#     p_attn = p_attn.half()  # 保持输出类型一致

#     # === Step 3: Permute 以便 topk ===
#     tmp_p_attn = p_attn.permute(2, 1, 0)  # shape: (n, 1, bh)

#     # === Step 4: topk 选出预取索引 ===
#     prefetch_idx = torch.topk(
#         tmp_p_attn, max_num_kv, dim=0
#     )[1]  # shape: (max_num_kv, 1, bh)

#     pad_idx = torch.topk(
#         -tmp_p_attn, 1, dim=0
#     )[1]  # shape: (1, 1, bh)

    
#     # print("kv_select p_q close", torch.allclose(p_q[0], p_q[n_head], atol=1e-6))
#     print("kv_select p_attn close", torch.allclose(p_attn[0], p_attn[n_head], atol=1e-6))
    

#     hidden_cmp = [torch.equal(hidden[i], hidden[i+1]) for i in range(b-1)]
#     print("kv_select: hidden_cmp = ", hidden_cmp)

#     p_k_c_cmp = []
#     p_k_c_T = p_k_c.permute(1, 0, 2) # bh, n, d
#     for i in range(32):
#         pidx_h1 = p_k_c_T[i]
#         pidx_h2 = p_k_c_T[i+32]
#         p_k_c_cmp.append(torch.equal(pidx_h1, pidx_h2))
#     print("kv_select: p_k_c_cmp = ", p_k_c_cmp)

    
#     tmp_p_attn_cmp = []
#     tmp_p_attn_T = tmp_p_attn.permute(2, 1, 0) # bh, n, d
#     for i in range(32):
#         pidx_h1 = tmp_p_attn_T[i]
#         pidx_h2 = tmp_p_attn_T[i+32]
#         tmp_p_attn_cmp.append(torch.equal(pidx_h1, pidx_h2))
#     print("kv_select: tmp_p_attn_cmp = ", tmp_p_attn_cmp)

    
#     prefetch_idx_cmp = []
#     prefetch_idx_T = prefetch_idx.squeeze(1).T
#     for i in range(32):
#         pidx_h1 = prefetch_idx_T[i]
#         pidx_h2 = prefetch_idx_T[i+32]
#         prefetch_idx_cmp.append(torch.equal(pidx_h1, pidx_h2))
    
#     print("kv_select: prefetch_idx_cmp = ", prefetch_idx_cmp)

#     return prefetch_idx, pad_idx