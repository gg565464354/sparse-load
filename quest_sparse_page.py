# quest_sparse_page_gqa_robust.py
import torch
import time
from typing import List, Tuple

class QuestSparseKVManagerPageAware:
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int = 16,
        dtype=torch.bfloat16,
        device: str = 'cuda',
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = dtype
        self.gpu_device = device
        self.cpu_device = 'cpu'

        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        ratio = num_heads // num_kv_heads
        self.q_head_to_kv_head = torch.arange(num_kv_heads).repeat_interleave(ratio).to(self.cpu_device)

        # Page setup
        self.num_pages = (max_seq_len + page_size - 1) // page_size
        self.page_start_end = [
            (p * page_size, min((p + 1) * page_size, max_seq_len))
            for p in range(self.num_pages)
        ]

        # === KV Cache: [B, H_kv, P, page_size, D] ===
        self.kv_cache_k = torch.zeros(
            batch_size, num_kv_heads, self.num_pages, page_size, head_dim,
            dtype=dtype, device=self.cpu_device, pin_memory=True
        )
        self.kv_cache_v = torch.zeros_like(self.kv_cache_k)

        # === Flat view: [Total_P, page_size, D] ===
        self.total_page_count = batch_size * num_kv_heads * self.num_pages
        self.flat_page_k = self.kv_cache_k.view(self.total_page_count, page_size, head_dim)
        self.flat_page_v = self.kv_cache_v.view(self.total_page_count, page_size, head_dim)

        # === 全局 page ID 映射函数 ===
        def global_pid(bid: int, hid: int, pid: int) -> int:
            return bid * num_kv_heads * self.num_pages + hid * self.num_pages + pid

        self._global_pid = global_pid

        # === 预分配 pinned buffer（动态扩展）===
        self.BH_kv = batch_size * num_kv_heads
        self._pinned_k_out = None
        self._pinned_v_out = None

        # Min/Max: [P, B, H_kv, D]
        self.page_key_min = torch.full(
            (self.num_pages, batch_size, num_kv_heads, head_dim),
            float('inf'), dtype=dtype, device=self.cpu_device
        )
        self.page_key_max = torch.full(
            (self.num_pages, batch_size, num_kv_heads, head_dim),
            float('-inf'), dtype=dtype, device=self.cpu_device
        )
        self.page_valid = torch.zeros(self.num_pages, dtype=torch.bool, device=self.cpu_device)

        # === 预计算缓存：映射到 query head 视图 [P, B, H_q, D] ===
        self.H_q = num_heads
        self.cached_page_key_min = torch.empty(
            self.num_pages, batch_size, self.H_q, head_dim,
            dtype=dtype, device=self.cpu_device
        )
        self.cached_page_key_max = torch.empty_like(self.cached_page_key_min)

        self.seq_length = 0
        self.last_page_id = -1
        self.last_page_valid_len = 0

    def _update_last_page_info(self):
        if self.seq_length == 0:
            self.last_page_id = -1
            return
        self.last_page_id = (self.seq_length - 1) // self.page_size
        start, _ = self.page_start_end[self.last_page_id]
        self.last_page_valid_len = self.seq_length - start

    def _update_page_stats(self, start: int, end: int):
        start_page = start // self.page_size
        end_page = (end - 1) // self.page_size + 1
        for p in range(start_page, end_page):
            if p >= self.num_pages:
                continue
            s, e = self.page_start_end[p]
            valid_e = min(e, self.seq_length)
            if s >= valid_e:
                continue
            k_page = self.kv_cache_k[:, :, p, :valid_e - s]  # [B, H_kv, L_p, D]
            k_reshaped = k_page.permute(0, 1, 3, 2)  # [B, H_kv, D, L_p]
            self.page_key_min[p] = torch.amin(k_reshaped, dim=-1)
            self.page_key_max[p] = torch.amax(k_reshaped, dim=-1)
            self.page_valid[p] = True

    def _update_cached_min_max(self):
        mapped_min = self.page_key_min.index_select(2, self.q_head_to_kv_head)  # [P, B, H_q, D]
        mapped_max = self.page_key_max.index_select(2, self.q_head_to_kv_head)
        self.cached_page_key_min.copy_(mapped_min)
        self.cached_page_key_max.copy_(mapped_max)

    def append_prefill_kv(self, k: torch.Tensor, v: torch.Tensor):
        k = k.to(self.cpu_device)
        v = v.to(self.cpu_device)
        S, BH, D = k.shape
        assert BH == self.batch_size * self.num_kv_heads

        start = self.seq_length
        end = start + S
        if end > self.max_seq_len:
            raise ValueError("Prefill exceeds max_seq_len")

        k_reshaped = k.transpose(0, 1).reshape(self.batch_size, self.num_kv_heads, S, D)
        v_reshaped = v.transpose(0, 1).reshape(self.batch_size, self.num_kv_heads, S, D)

        for pos in range(S):
            abs_pos = start + pos
            page_id = abs_pos // self.page_size
            page_offset = abs_pos % self.page_size
            for b in range(self.batch_size):
                for h in range(self.num_kv_heads):
                    self.kv_cache_k[b, h, page_id, page_offset] = k_reshaped[b, h, pos]
                    self.kv_cache_v[b, h, page_id, page_offset] = v_reshaped[b, h, pos]

        self.seq_length = end
        self._update_last_page_info()
        self._update_page_stats(start, end)
        self._update_cached_min_max()

    def append_decode_kv(self, k: torch.Tensor, v: torch.Tensor):
        k = k.to(self.cpu_device).reshape(1, -1, self.head_dim)
        v = v.to(self.cpu_device).reshape(1, -1, self.head_dim)
        _, BH, D = k.shape
        assert BH == self.batch_size * self.num_kv_heads

        pos = self.seq_length
        if pos >= self.max_seq_len:
            raise RuntimeError("Sequence length exceeds max_seq_len")

        page_id = pos // self.page_size
        page_offset = pos % self.page_size

        k_unflat = k[0].reshape(self.batch_size, self.num_kv_heads, D)
        v_unflat = v[0].reshape(self.batch_size, self.num_kv_heads, D)

        self.kv_cache_k[:, :, page_id, page_offset] = k_unflat
        self.kv_cache_v[:, :, page_id, page_offset] = v_unflat

        self.seq_length += 1
        self._update_last_page_info()
        self._update_page_stats(pos, pos + 1)
        self._update_cached_min_max()

    def quest_select_for_batch(
        self,
        query: torch.Tensor,
        top_k: int
    ) -> List[List[List[int]]]:
        query = query.to(self.cpu_device)
        B, H_q, _, D = query.shape

        # 延迟扩展 min/max
        kv_min = self.page_key_min.index_select(2, self.q_head_to_kv_head)  # [P, B, H_q, D]
        kv_max = self.page_key_max.index_select(2, self.q_head_to_kv_head)

        q_flat = query.squeeze(2)  # [B, H_q, D]
        sign_q = q_flat.unsqueeze(0)  # [1, B, H_q, D]

        # 更快的 bound: 选择最可能激活的 kv
        k_cand = torch.where(sign_q >= 0, kv_max, kv_min)  # [P, B, H_q, D]
        U = (sign_q * k_cand).sum(dim=-1)  # [P, B, H_q]

        valid_mask = self.page_valid.view(-1, 1, 1)
        U = torch.where(valid_mask, U, torch.tensor(-1e9, dtype=U.dtype, device=U.device))

        if self.last_page_id != -1:
            U[self.last_page_id] = -1e9  # 排除 last page

        scores = U.permute(2, 1, 0)  # [H_q, B, P]
        _, topk_page_ids = torch.topk(scores, k=top_k, dim=-1)  # [H_q, B, K]

        # 向量化构造 selected_pages
        kv_head_ids = self.q_head_to_kv_head.unsqueeze(1).unsqueeze(2).expand(H_q, B, top_k)
        b_expand = torch.arange(B).view(1, -1, 1).expand(H_q, -1, top_k)
        hkv_expand = kv_head_ids

        flat_topk = topk_page_ids.reshape(-1)
        flat_b = b_expand.reshape(-1)
        flat_hkv = hkv_expand.reshape(-1)

        # 使用字典聚合
        from collections import defaultdict
        page_dict = defaultdict(set)
        for i in range(flat_topk.shape[0]):
            b_idx = flat_b[i].item()
            hkv_idx = flat_hkv[i].item()
            p_idx = flat_topk[i].item()
            page_dict[(b_idx, hkv_idx)].add(p_idx)

        # 构造输出
        selected_pages = [[[] for _ in range(self.num_kv_heads)] for _ in range(B)]
        for (b, h), pids in page_dict.items():
            selected_pages[b][h].extend(pids)

        # 强制加入 last page
        if self.last_page_id != -1:
            for b in range(B):
                for h in range(self.num_kv_heads):
                    selected_pages[b][h].append(self.last_page_id)

        return selected_pages
    

    def gather_sparse_kv_to_gpu(
        self,
        selected_pages: List[List[List[int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = self.batch_size
        H_kv = self.num_kv_heads
        BH_kv = self.BH_kv
        D = self.head_dim
        Ps = self.page_size
        P = self.num_pages
        device = self.gpu_device

        if self.seq_length == 0 or BH_kv == 0:
            empty = torch.zeros(0, BH_kv, D, device=device, dtype=self.dtype)
            return empty, empty

        last_page_id = self.last_page_id
        last_page_valid_len = self.last_page_valid_len if last_page_id != -1 else 0

        # === Step 1: 收集每个 head 的 page IDs，并 padding 到相同长度 ===
        page_ids_per_head = []
        max_pages = 0

        for b in range(B):
            for h_kv in range(H_kv):
                ids = [p_id for p_id in selected_pages[b][h_kv] if p_id < P]
                page_ids_per_head.append(ids)
                max_pages = max(max_pages, len(ids))

        if max_pages == 0:
            empty = torch.zeros(0, BH_kv, D, device=device, dtype=self.dtype)
            return empty, empty

        # === Step 2: 构造 padded page ID 矩阵 [BH_kv, max_pages] ===
        padded_page_ids = torch.full((BH_kv, max_pages), fill_value=0, dtype=torch.long, device=self.cpu_device)
        page_mask = torch.zeros(BH_kv, max_pages, dtype=torch.bool, device=self.cpu_device)

        for idx, ids in enumerate(page_ids_per_head):
            L = len(ids)
            if L > 0:
                padded_page_ids[idx, :L] = torch.tensor(ids, dtype=torch.long)
                page_mask[idx, :L] = True

        # === Step 3: 向量化计算 global page ID: global_pid = b*H_kv*P + h*P + p_id ===
        # 展开为 [B, H_kv, max_pages]
        page_ids_3d = padded_page_ids.view(B, H_kv, max_pages)
        batch_ids = torch.arange(B, device=self.cpu_device).view(-1, 1, 1)
        head_ids = torch.arange(H_kv, device=self.cpu_device).view(1, -1, 1)

        global_ids_3d = (batch_ids * H_kv * P +
                         head_ids * P +
                         page_ids_3d)  # [B, H_kv, max_pages]
        global_ids_flat = global_ids_3d.reshape(-1, max_pages)  # [BH_kv, max_pages]

        # === Step 4: 向量化 gather 所有 pages ===
        flat_indices = global_ids_flat.flatten()  # [BH_kv * max_pages]
        gathered_k = self.flat_page_k.index_select(0, flat_indices)  # [N, Ps, D]
        gathered_v = self.flat_page_v.index_select(0, flat_indices)
        gathered_k = gathered_k.view(BH_kv, max_pages, Ps, D)  # [BH_kv, K, Ps, D]
        gathered_v = gathered_v.view(BH_kv, max_pages, Ps, D)

        # === Step 5: 构造 token_mask: 只有 last_page_id 需要特殊处理 ===
        # 初始化为 full mask: [BH_kv, max_pages, Ps]
        token_mask = page_mask.unsqueeze(-1).expand(-1, -1, Ps).clone()  # [BH_kv, K, Ps]

        if last_page_id != -1:
            # 找到哪些位置是 last_page_id
            is_last_page = (padded_page_ids == last_page_id) & page_mask  # [BH_kv, K], bool
            # 对这些位置，只保留前 last_page_valid_len 个 token
            valid_range = torch.arange(Ps, device=self.cpu_device) < last_page_valid_len  # [Ps]
            # 扩展并应用
            last_page_mask = is_last_page.unsqueeze(-1) & valid_range.unsqueeze(0).unsqueeze(0)  # [BH_kv, K, Ps]
            # 非 last_page 的位置保持 full mask
            # last_page 的位置用 valid_range 替代
            token_mask = torch.where(is_last_page.unsqueeze(-1), last_page_mask, token_mask)

        # === Step 6: 应用 mask，无效 token 置 0 ===
        gathered_k = gathered_k.masked_fill(~token_mask.unsqueeze(-1), 0)
        gathered_v = gathered_v.masked_fill(~token_mask.unsqueeze(-1), 0)

        # === Step 7: 合并 page 维度 → [BH_kv, L_max, D] ===
        max_tokens = max_pages * Ps
        flat_k = gathered_k.reshape(BH_kv, max_tokens, D)
        flat_v = gathered_v.reshape(BH_kv, max_tokens, D)

        # === Step 8: 使用 pinned memory 传输 ===
        if (self._pinned_k_out is None or 
            self._pinned_k_out.size(0) < max_tokens or 
            self._pinned_k_out.size(1) < BH_kv):
            self._pinned_k_out = torch.zeros(
                max_tokens, BH_kv, D, dtype=self.dtype, device='cpu', pin_memory=True
            )
            self._pinned_v_out = torch.zeros_like(self._pinned_k_out)

        # 复制并转置
        self._pinned_k_out[:max_tokens].copy_(flat_k.permute(1, 0, 2))
        self._pinned_v_out[:max_tokens].copy_(flat_v.permute(1, 0, 2))

        sparse_k = self._pinned_k_out[:max_tokens].to(device, non_blocking=True)
        sparse_v = self._pinned_v_out[:max_tokens].to(device, non_blocking=True)

        return sparse_k, sparse_v


# ========================================
# 测试函数
# ========================================

def benchmark_quest_performance(
    batch_size: int = 4,
    max_seq_len: int = 2048,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    dtype=torch.bfloat16,
    device: str = 'cuda',
    top_k_pages: int = 5,
    target_token_count: int = 2000,
):
    print(f"🚀 Starting GQA-Robust Page-Aware Benchmark")
    print(f"  Batch size: {batch_size}")
    print(f"  Target length: {target_token_count}")
    print(f"  Top-{top_k_pages} + last page (forced)")
    print("-" * 60)

    manager = QuestSparseKVManagerPageAware(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        dtype=dtype,
        device=device
    )

    BH_kv = batch_size * num_kv_heads
    query = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype)

    # Prefill
    print("🧪 Prefilling KV cache (on CPU)...")
    k_prefill = torch.randn(target_token_count, BH_kv, head_dim, device='cpu', dtype=dtype)
    v_prefill = torch.randn_like(k_prefill)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    manager.append_prefill_kv(k_prefill, v_prefill)
    end_event.record()
    torch.cuda.synchronize()
    prefill_time = start_event.elapsed_time(end_event) / 1000
    print(f"    Prefill done in {prefill_time:.4f}s, seq_length = {manager.seq_length}")

    # Selection
    print("🔍 Running CPU selection (last page forced)...")
    torch.cuda.synchronize()
    start_time = time.time()
    selected_pages = manager.quest_select_for_batch(query, top_k=top_k_pages)
    torch.cuda.synchronize()
    select_time = time.time() - start_time

    total_tokens = 0
    for b in range(batch_size):
        for h_kv in range(num_kv_heads):
            for p_id in selected_pages[b][h_kv]:
                if p_id < len(manager.page_start_end):
                    start, end = manager.page_start_end[p_id]
                    valid_end = min(end, manager.seq_length)
                    total_tokens += max(0, valid_end - start)

    print(f"    Page selection took {select_time * 1000:.2f} ms")
    print(f"    Estimated sparse tokens: {total_tokens}")

    # Gather
    print("🚚 Running sparse KV gather (GQA-robust page-aware)...")
    torch.cuda.synchronize()
    start_event.record()
    sparse_k, sparse_v = manager.gather_sparse_kv_to_gpu(selected_pages)
    end_event.record()
    torch.cuda.synchronize()
    gather_time = start_event.elapsed_time(end_event) / 1000

    print(f"    Sparse gather took {gather_time * 1000:.2f} ms")
    print(f"    sparse_k.shape = {sparse_k.shape}")

    # Full copy baseline
    print("🚛 Running full KV copy (baseline)...")
    full_k_cpu = torch.cat([
        manager.kv_cache_k[b, h, :manager.seq_length // manager.page_size + 1, :, :]
        for b in range(batch_size) for h in range(num_kv_heads)
    ], dim=0)
    full_k_cpu = full_k_cpu.reshape(BH_kv, -1, head_dim)[:, :manager.seq_length]
    full_k_cpu = full_k_cpu.transpose(0, 1).contiguous()

    torch.cuda.synchronize()
    start_event.record()
    full_k = full_k_cpu.to(device)
    full_v = torch.zeros_like(full_k_cpu).to(device)
    end_event.record()
    torch.cuda.synchronize()
    full_time = start_event.elapsed_time(end_event) / 1000

    print(f"    Full copy took {full_time * 1000:.2f} ms")
    print(f"    full_k.shape = {full_k.shape}")

    # Summary
    total_sparse = select_time + gather_time
    speedup = full_time / (total_sparse + 1e-8)
    print("📊 SUMMARY")
    print(f"  Total sparse path: {total_sparse * 1000:.2f} ms")
    print(f"  Full copy:         {full_time * 1000:.2f} ms")
    print(f"  Speedup:           {speedup:.2f}x")
    print(f"  {'✅ Sparse faster' if total_sparse < full_time else '❌ Full faster'}")

    return {
        "select_time_ms": select_time * 1000,
        "gather_time_ms": gather_time * 1000,
        "full_copy_ms": full_time * 1000,
        "speedup": speedup,
        "sparse_tokens": total_tokens,
    }


if __name__ == "__main__":
    page_size = 16
    token_len = 2000
    topk_pages = int((token_len/page_size)*0.1)
    
    result = benchmark_quest_performance(
        batch_size=4,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        top_k_pages=topk_pages,
        target_token_count=token_len,
        page_size=page_size,
        device='cuda'
    )
