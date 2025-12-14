import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_fwd(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B, H, N_CTX, D_HEAD,
    stride_qb, stride_qh, stride_ql,
    stride_kb, stride_kh, stride_kl,
    stride_vb, stride_vh, stride_vl,
    stride_ob, stride_oh, stride_ol,
    is_decode,  # 0 or 1
    current_len,  # length to attend to in decode mode
    BLOCK_M: tl.constexpr,  # block size along M dimension (queries)
    BLOCK_N: tl.constexpr,  # block size along N dimension (keys)
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    Compute out = softmax(Q*K^T) * V blockwise.
    Q: [B, H, N_CTX, D_HEAD]
    K: [B, H, N_CTX, D_HEAD]
    V: [B, H, N_CTX, D_HEAD]
    is_decode: 1 => decode mode, 0 => full attention (prefill)
    current_len: in decode mode, how many tokens are valid so far?
                 in prefill mode, typically = N_CTX
    BLOCK_M, BLOCK_N: block sizes
    """
    # ------------------------------------------------------------
    # Program ID and offsets
    # ------------------------------------------------------------
    pid_m = tl.program_id(0)
    pid_hb = tl.program_id(1)

    # Extract batch and head index from pid_hb
    # e.g. if total B*H blocks are used
    # "hb" = (batch_index * H) + head_index
    # THOUGHTS: We'll decode batch/h from pid_hb
    b_idx = pid_hb // H
    h_idx = pid_hb % H

    # The row offset for Q in the M dimension
    # M dimension => sequence length dimension for Q
    m_start = pid_m * BLOCK_M

    # If decode=1, we only have one block of queries (the "new" token),
    # so we can limit ourselves to the final row in Q.
    # THOUGHTS: For a single token decode, m_start=0 is often fixed,
    # but let's keep generalization for partial block decode.
    if is_decode == 1:
        # In decode mode, the "effective" M dimension is just 1 or some small region
        # We clamp m_start to the last chunk of the current sequence
        m_start = current_len - BLOCK_M
        m_start = tl.maximum(m_start, 0)

    # ------------------------------------------------------------
    # Pointers for Q this block
    # ------------------------------------------------------------
    # THOUGHTS: We get the pointer to Q for batch b_idx, head h_idx, row offset m_start
    Q_block_ptr = Q_ptr + b_idx * stride_qb + h_idx * stride_qh + m_start * stride_ql

    # We also track the dimension offsets
    # We'll need a loop over the K dimension in blocks
    # Partial: we do an iteration over the block of K dimension
    # We'll produce partial sums that get reduced

    # We'll store partial accumulations for the final V-product
    # We'll store partial max and sum (for the softmax)

    # Each block of Q is [BLOCK_M, D_HEAD]. We'll load it into registers.
    # For K, we do it block by block in the N dimension.

    # We'll define a loop over n_block in [0, N_CTX, BLOCK_N] if is_decode=0
    # or in [0, current_len, BLOCK_N] if is_decode=1
    # We'll accumulate partial results in shared or registers. For simplicity, we show a direct approach.

    # Create accumulators for the partial softmax
    # We'll do a row for each of the M rows
    # "m" dimension is the row dimension, up to BLOCK_M
    # We'll have partial sum and max.
    scores_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    scores_sum = tl.zeros((BLOCK_M,), tl.float32)

    # We'll create a partial output accumulator for the result of multiplying by V
    # shape: [BLOCK_M, D_HEAD]
    out_accum = tl.zeros((BLOCK_M, D_HEAD), tl.float32)

    # Actually load Q block from memory into registers
    # shape = [BLOCK_M, D_HEAD]
    # THOUGHTS: We might do a 2D load, but in Triton we typically do vector loads for contiguous memory
    # We'll do a for loop for the dimension. For brevity, let's do a naive approach:
    q_block = tl.zeros((BLOCK_M, D_HEAD), dtype=tl.float32)
    for d in range(D_HEAD):
        q_block[:, d] = tl.load(Q_block_ptr + d, mask=(m_start + tl.arange(0, BLOCK_M) < N_CTX))

    # ------------------------------------------------------------
    # Now iterate over the blocks of K dimension
    # ------------------------------------------------------------
    n_range = current_len if is_decode == 1 else N_CTX
    # We'll iterate in steps of BLOCK_N
    # For each n_block, we load K and V, compute partial attention scores, update softmax stats, ...
    # Then after we have stable partial softmax stats, we do a second pass to compute final results.
    # For simplicity, let's do a single-pass approach with a running max-sum for stability.
    
    # For demonstration: do a simple approach with global max for the row so far
    for n_start in range(0, n_range, BLOCK_N):
        # 1) Load K-block
        K_block_ptr = K_ptr + b_idx * stride_kb + h_idx * stride_kh + n_start * stride_kl
        # shape = [BLOCK_N, D_HEAD]
        k_block = tl.zeros((BLOCK_N, D_HEAD), dtype=tl.float32)
        for d in range(D_HEAD):
            k_block[:, d] = tl.load(K_block_ptr + d, mask=(n_start + tl.arange(0, BLOCK_N) < n_range))

        # 2) Compute Scores = Q_block * K_block^T
        # We'll do: [BLOCK_M, D_HEAD] * [D_HEAD, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        # Then we do the softmax in the row dimension. For memory reasons, we do partial row-level transformations
        # Let's do naive matmul in Pythonic pseudo-code inside Triton:
        # shape of scores = [BLOCK_M, BLOCK_N]
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in range(BLOCK_M):
            for j in range(BLOCK_N):
                acc_ij = tl.float32(0.)
                for d in range(D_HEAD):
                    acc_ij += q_block[i, d] * k_block[j, d]
                # We typically scale by 1/sqrt(D_HEAD)
                # (FlashAttention does scale early with Q, but for illustration):
                scores[i, j] = acc_ij * (1.0 / tl.sqrt(tl.float32(D_HEAD)))

        # 3) Update row-wise max for stability
        # For each i in [0, BLOCK_M], find the max over j in [0, BLOCK_N], update scores_max
        for i in range(BLOCK_M):
            row_max = tl.maximum(scores_max[i], tl.max(scores[i, :]))
            # shift the scores by (row_max - old_max), then exponentiate
            old_max = scores_max[i]
            diff = row_max - old_max
            # update partial sums
            # old sum must be scaled by exp(-diff)
            old_sum = scores_sum[i] * tl.exp(-diff)
            # new sum = old_sum + sum of e^(scores[i, j] - row_max)
            # let's do that in two steps
            # step1: shift and exponentiate
            for j in range(BLOCK_N):
                scores[i, j] = tl.exp(scores[i, j] - row_max)
            # step2: add to old_sum
            row_sum = old_sum + tl.sum(scores[i, :])
            # update scores_max, scores_sum
            scores_max[i] = row_max
            scores_sum[i] = row_sum

        # 4) For the partial V accumulation, we do out_accum[i, :] += sum( scores[i, j] * V_block[j, : ] ) 
        #    We'll load V block for the same range [n_start : n_start+BLOCK_N] and accumulate
        V_block_ptr = V_ptr + b_idx * stride_vb + h_idx * stride_vh + n_start * stride_vl
        v_block = tl.zeros((BLOCK_N, D_HEAD), dtype=tl.float32)
        for d in range(D_HEAD):
            v_block[:, d] = tl.load(V_block_ptr + d, mask=(n_start + tl.arange(0, BLOCK_N) < n_range))

        for i in range(BLOCK_M):
            # We'll accumulate in registers: out_accum[i, :] 
            acc_i = out_accum[i, :]
            for j in range(BLOCK_N):
                score_ij = scores[i, j]
                for d in range(D_HEAD):
                    acc_i[d] += score_ij * v_block[j, d]
            out_accum[i, :] = acc_i

    # ------------------------------------------------------------
    # Now we finalize the output 
    # out[i, :] = out_accum[i, :] / scores_sum[i]
    # We'll store the result in Out_ptr
    # ------------------------------------------------------------
    Out_block_ptr = Out_ptr + b_idx * stride_ob + h_idx * stride_oh + m_start * stride_ol

    for i in range(BLOCK_M):
        denom = scores_sum[i]
        # Avoid dividing by zero
        denom = tl.where(denom == 0.0, 1e-10, denom)
        for d in range(D_HEAD):
            val = out_accum[i, d] / denom
            tl.store(
                Out_block_ptr + i * stride_ol + d,
                val,
                mask=(m_start + i < N_CTX)
            )


@torch.no_grad()
def flash_attention_v2(Q, K, V, is_decode=False, current_len=None, block_size=128):
    """
    Wrapper to launch the above kernel in Triton.
    Q, K, V: [B, H, N_CTX, D_HEAD], typically float16 or bfloat16
    is_decode: bool, True => decode mode
    current_len: if decode=True, must be provided; else ignored
    block_size: block dimension for M, N
    """
    B, H, N_CTX, D_HEAD = Q.shape
    if is_decode and current_len is None:
        raise ValueError("In decode mode, must specify current_len")

    # We'll create an output tensor
    Out = torch.empty_like(Q)

    # Convert Q,K,V to float16 or BF16 if not already
    # (In practice you'd do more thorough checks here)
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # We can let Triton handle the grid
    # We'll have grid0 = number of blocks along M dimension
    # If decode => effectively 1 or so. We’ll keep it general though:
    grid0 = ( (N_CTX + block_size - 1) // block_size )
    grid1 = B * H
    # We pass in stride info to the kernel
    # Q.stride() => (H*N_CTX*D_HEAD, N_CTX*D_HEAD, D_HEAD, 1) typically 
    stride_qb, stride_qh, stride_ql, stride_qd = Q.stride()
    stride_kb, stride_kh, stride_kl, stride_kd = K.stride()
    stride_vb, stride_vh, stride_vl, stride_vd = V.stride()
    stride_ob, stride_oh, stride_ol, stride_od = Out.stride()

    # We'll define kernel launcher
    flash_attention_fwd[grid0, grid1](
        Q, K, V, Out,
        B, H, N_CTX, D_HEAD,
        stride_qb, stride_qh, stride_ql,
        stride_kb, stride_kh, stride_kl,
        stride_vb, stride_vh, stride_vl,
        stride_ob, stride_oh, stride_ol,
        int(is_decode),
        current_len if current_len else N_CTX,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        num_warps=4,
        num_stages=2
    )

    return Out

def example_run():
    import time

    B = 4
    H = 8
    N_CTX = 1024
    D_HEAD = 64

    # Synthetic data
    Q = torch.randn((B, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda")
    K = torch.randn((B, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda")
    V = torch.randn((B, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda")

    # 1. Prefill (full attention) run
    start_t = time.time()
    out_prefill = flash_attention_v2(Q, K, V, is_decode=False, block_size=128)
    torch.cuda.synchronize()
    end_t = time.time()
    print(f"[Prefill] Triton FlashAttn v2 time: {end_t - start_t:.4f}s")

    # 2. Decode run (imagine we have 1 new token, current_len=512)
    # In real decode scenarios, you might have cached K, V from prior calls.
    # For demonstration, we just reuse K, V as is but only attend up to current_len=512
    current_len = 512
    start_t = time.time()
    out_decode = flash_attention_v2(Q, K, V, is_decode=True, current_len=current_len, block_size=128)
    torch.cuda.synchronize()
    end_t = time.time()
    print(f"[Decode] Triton FlashAttn v2 time: {end_t - start_t:.4f}s")

    # Compare to PyTorch scaled_dot_product_attention
    # We'll cast to float32 as PyTorch 2.0's scaled_dot_product_attention is flexible with dtype
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    # 1. Prefill with SDPA
    start_t = time.time()
    Qf = Q.float()
    Kf = K.float()
    Vf = V.float()
    out_sdpa = sdpa(Qf.flatten(0,1), Kf.flatten(0,1), Vf.flatten(0,1)).view(B, H, N_CTX, D_HEAD)
    torch.cuda.synchronize()
    end_t = time.time()
    print(f"[Prefill] SDPA time: {end_t - start_t:.4f}s")

    # Quick check: difference in output
    diff = (out_prefill.float() - out_sdpa).abs().max()
    print(f"Max difference between Triton FlashAttn v2 and SDPA: {diff.item()}")

example_run()