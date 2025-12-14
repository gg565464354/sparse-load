# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union

import torch
from torch import nn
import time
from typing import Callable, Optional, Union, List, Tuple
from collections import defaultdict # Quest Manager 依赖
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import check_model_inputs
from .configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)


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


        # NEW: 计算覆盖 "最近 recent_tokens 个 token" 所需的页列表（包含 last_page）
        def recent_pages_to_cover(self, recent_tokens: int) -> list[int]:
            if self.seq_length <= 0 or recent_tokens <= 0:
                return []
            start_pos = max(0, self.seq_length - recent_tokens)
            start_page = start_pos // self.page_size
            # 覆盖从 start_page 到 last_page 的所有页
            end_page = self.last_page_id if self.last_page_id != -1 else (self.seq_length - 1) // self.page_size
            return list(range(start_page, end_page + 1))

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

    def quest_select_with_kv_query(
        self,
        query_kv: torch.Tensor,   # [B, H_kv or 1, 1, D]
        top_k: int
    ) -> List[List[List[int]]]:
        # 设备与形状处理
        query_kv = query_kv.to(self.cpu_device)
        B, H_in, _, D = query_kv.shape
        assert D == self.head_dim, "head_dim mismatch"

        # 若是单一合并，广播到所有 H_kv
        if H_in == 1:
            query_kv = query_kv.expand(B, self.num_kv_heads, 1, D)  # [B, H_kv, 1, D]

        # 直接用 page_key_min/max（已经是 [P, B, H_kv, D]）
        kv_min = self.page_key_min    # [P, B, H_kv, D]
        kv_max = self.page_key_max

        q_flat = query_kv.squeeze(2)          # [B, H_kv, D]
        sign_q = q_flat.unsqueeze(0)          # [1, B, H_kv, D]
        k_cand = torch.where(sign_q >= 0, kv_max, kv_min)  # [P, B, H_kv, D]
        U = (sign_q * k_cand).sum(dim=-1)     # [P, B, H_kv]

        valid_mask = self.page_valid.view(-1, 1, 1)
        U = torch.where(valid_mask, U, torch.tensor(-1e9, dtype=U.dtype, device=U.device))

        if self.last_page_id != -1:
            U[self.last_page_id] = -1e9

        # [H_kv, B, P]
        scores = U.permute(2, 1, 0)
        _, topk_page_ids = torch.topk(scores, k=top_k, dim=-1)  # [H_kv, B, K]

        # 组装成 List[List[List[int]]]: [B][H_kv][K]
        selected_pages = [[[] for _ in range(self.num_kv_heads)] for _ in range(B)]
        H_kv = self.num_kv_heads
        for h in range(H_kv):
            for b in range(B):
                pids = topk_page_ids[h, b].tolist()
                if self.last_page_id != -1 and self.last_page_id not in pids:
                    pids.append(self.last_page_id)
                selected_pages[b][h].extend(pids)

        return selected_pages


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
# === Step 1: 收集每个 head 的 page IDs，并 padding 到相同长度 ===
        page_ids_per_head = []
        max_pages = 0
        for b in range(B):
            for h_kv in range(H_kv):
                # 原：ids = [p_id for p_id in selected_pages[b][h_kv] if p_id < P]
                ids = [p_id for p_id in selected_pages[b][h_kv] if 0 <= p_id < P]
                # NEW: 去重 + 排序（按时间从早到晚）
                ids = sorted(set(ids))
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
    def reset(self):
        """Resets the state of the cache manager for a new sequence."""
        print("Resetting Quest KV Manager State...")
        self.seq_length = 0
        self.last_page_id = -1
        self.last_page_valid_len = 0

        # Reset page statistics
        self.page_key_min.fill_(float('inf'))
        self.page_key_max.fill_(float('-inf'))
        self.page_valid.fill_(False)

        # Re-update cached min/max based on reset values (clears them)
        self._update_cached_min_max()

@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
    def _merge_queries_to_kv(self, q: torch.Tensor, reduce: str = "mean", single: bool = False) -> torch.Tensor:
        """
        q: [B, H_q, S(=1), D]
        return:
        if single=False: [B, H_kv, 1, D]
        if single=True:  [B, 1,    1, D]
        """
        B, Hq, S, D = q.shape
        g = self.num_heads // self.num_key_value_heads  # group_size
        # [B, H_kv, g, S, D]
        qg = q.view(B, self.num_key_value_heads, g, S, D)

        if reduce == "mean":
            merged = qg.mean(dim=2)                      # [B, H_kv, S, D]
        elif reduce == "sum":
            merged = qg.sum(dim=2)
        elif reduce == "maxabs":
            # 选绝对值最大的 head 作为代表（稳定的稀疏检索基准）
            idx = qg.abs().amax(dim=-1, keepdim=True).argmax(dim=2, keepdim=True)  # [B, H_kv, 1, S, 1]
            merged = qg.gather(2, idx.expand(-1, -1, 1, S, D)).squeeze(2)          # [B, H_kv, S, D]
        elif reduce == "l2":
            merged = torch.nn.functional.normalize(qg, dim=-1).sum(dim=2)
        else:
            merged = qg.mean(dim=2)

        if single:
            merged = merged.mean(dim=1, keepdim=True)    # [B, 1, S, D]

        return merged

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        quest_manager: Optional["QuestSparseKVManagerPageAware"] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        # +++ QUEST START +++
        # bsz, q_len, _ = hidden_states.shape # 获取 B 和 S(q_len)
        # GQA/MHA: q_proj shape [B, S, H_q * D] -> view -> [B, S, H_q, D] -> transpose -> [B, H_q, S, D]
        # GQA: k_proj/v_proj shape [B, S, H_kv * D] -> view -> [B, S, H_kv, D] -> transpose -> [B, H_kv, S, D]
        # The head dimension calculation for k/v needs num_key_value_heads, not num_heads (self.num_heads)
        # Let's verify the original implementation's view logic.
        # hidden_shape = (*input_shape, -1, self.head_dim) will be (B, S, num_heads, head_dim) for Q
        # and (B, S, num_key_value_heads, head_dim) for K/V, if k_proj/v_proj output dim matches.

        # Calculate K/V view shape specifically for GQA
        bsz, q_len, _ = hidden_states.shape
        query_shape = (bsz, q_len, self.num_heads, self.head_dim)
        key_value_shape = (bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(query_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(key_value_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(key_value_shape).transpose(1, 2)
        # +++ QUEST END +++
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        self.rope_query = query_states
        self.rope_key = key_states 
        attn_mask_to_use = attention_mask

        # --- Quest Sparse Attention Logic ---
        is_quest_active = quest_manager is not None and not self.training
        # is_decode = q_len == 1
        is_decode = True

        if is_quest_active:
            # [B, H_kv, S, D]  ->  [S, B*H_kv, D]
            current_B, H_kv, current_S, D_head = key_states.shape
            num_kv_heads_times_batch = current_B * H_kv
            k_for_manager = key_states.permute(2, 0, 1, 3).reshape(current_S, num_kv_heads_times_batch, D_head)
            v_for_manager = value_states.permute(2, 0, 1, 3).reshape(current_S, num_kv_heads_times_batch, D_head)

            if is_decode:
                # 1) 追加当前步 KV
                quest_manager.append_decode_kv(k_for_manager, v_for_manager)

                # 2) 代表性 Query 选页
                quest_top_k = getattr(self.config, "quest_top_k", 5)
                merge_mode  = getattr(self.config, "quest_merge_mode", "kv")      # "kv" | "single" | "none"
                reduce_mode = getattr(self.config, "quest_merge_reduce", "mean")  # "mean" | "sum" | "maxabs" | "l2"
                quest_top_k = max(1, min(quest_top_k, getattr(quest_manager, "num_pages", quest_top_k)))

                if merge_mode in ("kv", "single"):
                    q_for_select = self._merge_queries_to_kv(
                        query_states, reduce=reduce_mode, single=(merge_mode == "single")
                    )  # [B,H_kv,1,D] 或 [B,1,1,D]
                    selected_pages = quest_manager.quest_select_with_kv_query(q_for_select, top_k=quest_top_k)
                else:
                    selected_pages = quest_manager.quest_select_for_batch(query_states, top_k=quest_top_k)
                # === NEW: 并入最近窗口 ===
                recent_ratio = float(getattr(self.config, "quest_recent_ratio", 0.0))
                if recent_ratio > 0.0 and quest_manager.seq_length > 0:
                    # 按比例换算 token 预算（向上取整，至少 1）
                    recent_tokens = max(1, int(round(recent_ratio * quest_manager.seq_length)))
                    force_pages = quest_manager.recent_pages_to_cover(recent_tokens)
                    if force_pages:
                        # 每个 (b, h_kv) 并集去重（可选：排序保证时间顺序）
                        force_pages_set = set(force_pages)
                        for b in range(current_B):
                            for h in range(H_kv):
                                pooled = set(selected_pages[b][h])
                                # 并入最近页
                                pooled |= force_pages_set
                                # 可选：排序以保证页面时间顺序（建议）
                                selected_pages[b][h] = sorted(pooled)
                # 3) 聚合稀疏 KV 到 GPU，覆盖本步 K/V
                sparse_key_history, sparse_value_history = quest_manager.gather_sparse_kv_to_gpu(selected_pages)
                if sparse_key_history.shape[0] > 0:
                    Ls = sparse_key_history.shape[0]
                    key_states = sparse_key_history.view(Ls, current_B, H_kv, D_head).permute(1, 2, 0, 3)   # [B,H_kv,Ls,D]
                    value_states = sparse_value_history.view(Ls, current_B, H_kv, D_head).permute(1, 2, 0, 3)
                else:
                    key_states = torch.empty(current_B, H_kv, 0, D_head, dtype=query_states.dtype, device=query_states.device)
                    value_states = torch.empty_like(key_states)

                # 4) 掩码失配：选页后 KV 顺序已改变，禁用原 causal mask
                attn_mask_to_use = None

            else:
                # PREFILL：整段写入页式缓存（不做稀疏选页）
                if quest_manager.seq_length == 0:
                    quest_manager.append_prefill_kv(k_for_manager, v_for_manager)
        # 如需增量 prefill，可在此续写
            # else (prefill phase): We use the key_states and value_states calculated from the current input chunk directly.
            # The standard causal mask will handle self-attention within the prefill chunk.

        elif past_key_value is not None:
            # --- Standard Cache Logic (if Quest is disabled) ---
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # +++ QUEST END +++
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attn_mask_to_use,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        quest_manager: Optional["QuestSparseKVManagerPageAware"] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            quest_manager=quest_manager, # 向下传递
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # +++ QUEST START +++
        # S-2: 初始化 Quest Manager
        self.use_quest_sparse = getattr(config, "use_quest_sparse", True)
        self.quest_manager = None
        if self.use_quest_sparse:
            # 从 config 中获取 Quest 特定参数 (假设已添加到 LlamaConfig 中)
            # 注意：Quest manager 的 batch size 是固定的。这在实际使用中需要小心处理，
            # 确保推理时的 batch size 与初始化时一致。
            # 这里我们假设 config 中有一个 'inference_batch_size' 字段用于初始化。
            inference_batch_size = getattr(config, "inference_batch_size", 1) # 示例默认值
            page_size = getattr(config, "quest_page_size", 16) # 示例默认值

            logger.info("Initializing QuestSparseKVManagerPageAware...")
            self.quest_manager = QuestSparseKVManagerPageAware(
                batch_size=inference_batch_size, # 关键：需要匹配实际推理的batch size
                max_seq_len=config.max_position_embeddings,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.hidden_size // config.num_attention_heads,
                page_size=page_size,
                dtype=torch.get_default_dtype(), # or config.torch_dtype
                device='cuda' if torch.cuda.is_available() else 'cpu' # 管理器本身在CPU上运行，但需要知道GPU device
            )
        # +++ QUEST END +++
        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.use_quest_sparse:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            if self.use_quest_sparse and self.quest_manager is not None:
                past_seen_tokens = self.quest_manager.seq_length            
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                quest_manager=self.quest_manager,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
