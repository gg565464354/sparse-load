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
import sys
import os
quest_path = '/workspace/quest'
if quest_path not in sys.path:
    sys.path.append(quest_path)
import quest.utils as quest_utils
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

class CachedHeavyRecentAttentionMasker(nn.Module):
    def __init__(self, heavy_budget_ratio=0.1, recent_budget_ratio=0.1, layer_idx: int = None, rebuild_cap_ratio: float = 0.75):
        super().__init__()
        self.heavy_budget_ratio = heavy_budget_ratio
        self.recent_budget_ratio = recent_budget_ratio
        self.layer_idx = layer_idx
        self.rebuild_cap_ratio = rebuild_cap_ratio  # 新增：阈值上限比例（默认 0.75）

        self.cached_indices: Optional[torch.Tensor] = None
        self.step_count = 0

        self.head_hit_tokens = {}
        self.head_candidate_tokens = {}
        self.forward_count = 0

    def forward(self, attn_weights: torch.Tensor, group_size: int = 7):
        """
        修改版forward方法，返回与原kv_cache_mask兼容的格式
        """
        bs, head, query_len, key_len = attn_weights.shape
        
        # 确保所有计算使用相同的dtype和device
        dtype = attn_weights.dtype
        device = attn_weights.device
        min_value = torch.finfo(dtype).min
        if key_len == 0:
            min_value = torch.finfo(dtype).min
            return torch.full_like(attn_weights, min_value), 0.0
        assert head % group_size == 0, "head must be divisible by group_size"
        num_groups = head // group_size

        # 软化 -> 按 query 求和，拿到 token 重要度
        # 形状: [B, G, group, L, S]
        attn_g = attn_weights.view(bs, num_groups, group_size, query_len, key_len)
        tmp_attn = nn.functional.softmax(attn_g, dim=-1, dtype=torch.float32).to(dtype)
        importance = tmp_attn.sum(dim=-2)  # [B, G, group, S]，沿 query 维求和

        # 预算
        heavy_budget = max(1, min(int(self.heavy_budget_ratio * key_len), key_len))

        # 每个 head 先各取 top-k
        _, topk_idx_head = importance.topk(k=heavy_budget, dim=-1)  # [B, G, group, k]

        # 先做 head 内的 one-hot，再在 group_size 上求并集
        heavy_mask_head = torch.zeros(bs, num_groups, group_size, key_len, dtype=torch.bool, device=device)
        heavy_mask_head.scatter_(-1, topk_idx_head, True)  # [B, G, group, S]
        heavy_mask_group = heavy_mask_head.any(dim=2)      # [B, G, S] —— 并集完成

        # ===== 跨 step 的缓存（按 group 存）=====
        if self.cached_indices is None:
            # 首次：直接按当前 heavy 结果建立缓存
            max_keep = heavy_mask_group.sum(dim=-1).amax().item()
            if max_keep == 0:
                cached_idx = torch.zeros(bs, num_groups, 1, dtype=torch.long, device=device)
            else:
                cached_idx = torch.full((bs, num_groups, max_keep), -1, dtype=torch.long, device=device)
                for b in range(bs):
                    for g in range(num_groups):
                        ids = torch.nonzero(heavy_mask_group[b, g], as_tuple=False).flatten()
                        if ids.numel():
                            cached_idx[b, g, :ids.numel()] = ids
            self.cached_indices = cached_idx.detach()
            final_mask_group = heavy_mask_group
        else:
            # 先把旧缓存还原成 bool mask
            prev_idx = self.cached_indices  # [B, G, K’]
            prev_mask_group = torch.zeros(bs, num_groups, key_len, dtype=torch.bool, device=device)
            for b in range(bs):
                for g in range(num_groups):
                    ids = prev_idx[b, g]
                    ids = ids[(ids >= 0) & (ids < key_len)]
                    if ids.numel():
                        prev_mask_group[b, g].scatter_(0, ids, True)

            # 合并旧缓存与本步 heavy 选择
            merged = prev_mask_group | heavy_mask_group  # [B, G, S]

            # 统计命中（可选）
            inter = (prev_mask_group & heavy_mask_group).sum().item()
            cand = heavy_mask_group.sum().item()
            self.head_hit_tokens[0] = self.head_hit_tokens.get(0, 0) + inter
            self.head_candidate_tokens[0] = self.head_candidate_tokens.get(0, 0) + cand

            # ========= 基于阈值的重建判定 =========
            # 当前步关键 KV 数量（每组）
            k_cur = heavy_mask_group.sum(dim=-1)                        # [B, G]
            # 两倍关键数
            thr_2k = k_cur * 2                                          # [B, G]
            # 0.75 * 总 token 上限
            thr_cap_val = int(self.rebuild_cap_ratio * key_len)         # 标量
            threshold = torch.minimum(thr_2k, torch.full_like(thr_2k, thr_cap_val))  # [B, G]

            # 如果合并后的缓存规模超过阈值，则对该组重建（用当前步的 heavy）
            merged_count = merged.sum(dim=-1)                            # [B, G]
            rebuild_flags = merged_count > threshold                     # [B, G]

            # 超阈值的 group 用 heavy_mask_group，其余 group 用 merged
            final_mask_group = torch.where(rebuild_flags[..., None], heavy_mask_group, merged)  # [B, G, S]
            # =====================================

            # 维护新的缓存 indices（按最终 mask）
            max_keep = final_mask_group.sum(dim=-1).amax().item()
            if max_keep == 0:
                new_idx = torch.zeros(bs, num_groups, 1, dtype=torch.long, device=device)
            else:
                new_idx = torch.full((bs, num_groups, max_keep), -1, dtype=torch.long, device=device)
                for b in range(bs):
                    for g in range(num_groups):
                        ids = torch.nonzero(final_mask_group[b, g], as_tuple=False).flatten()
                        if ids.numel():
                            new_idx[b, g, :ids.numel()] = ids
            self.cached_indices = new_idx.detach()

        self.step_count += 1
        self.forward_count += 1

        # ===== 严格因果（j ≤ i）=====
        causal_full = torch.tril(
            torch.ones((bs, 1, query_len, key_len), dtype=torch.bool, device=device), diagonal=0
        )  # [B,1,L,S]广播到 head 维

        # recent 带（可选）
        if int(self.recent_budget_ratio * key_len) > 0:
            recent_budget = int(self.recent_budget_ratio * key_len)
            recent_band = torch.triu(causal_full, diagonal=-recent_budget)  # [B,1,L,S]
        else:
            recent_band = torch.eye(query_len, key_len, dtype=torch.bool, device=device)[None, None].expand(bs, 1, -1, -1)

        # ===== 把 group 级 mask 展开回 head 级（同组共享）=====
        # final_mask_group: [B, G, S] -> [B, G, 1, L, S] -> tile 到 group_size
        final_mask_group_4d = final_mask_group[:, :, None, None, :].expand(-1, -1, group_size, query_len, -1)
        final_keep_mask = final_mask_group_4d.reshape(bs, head, query_len, key_len)

        # 强制因果
        final_keep_mask = final_keep_mask & causal_full.expand(-1, head, -1, -1)

        # 合并 recent
        final_keep_mask = final_keep_mask | recent_band.expand(-1, head, -1, -1)

        # 转加性 mask
        fetch_mask = torch.where(
            final_keep_mask,
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(min_value, dtype=dtype, device=device),
        )

        # 密度（相对完整下三角）
        heads = bs * head
        density = final_keep_mask.float().sum().item() / heads / (query_len * (query_len + 1) / 2)

        return fetch_mask, density

    def get_hit_rate(self, head_idx: int = None):
        """获取指定头或所有头的命中率"""
        if head_idx is not None:
            if head_idx in self.head_candidate_tokens and self.head_candidate_tokens[head_idx] > 0:
                return self.head_hit_tokens[head_idx] / self.head_candidate_tokens[head_idx]
            return 0.0
        else:
            total_hits = sum(self.head_hit_tokens.values())
            total_candidates = sum(self.head_candidate_tokens.values())
            if total_candidates > 0:
                return total_hits / total_candidates
            return 0.0

    def get_hit_stats(self):
        """获取当前层所有头的详细统计信息"""
        head_stats = {}
        for head_idx in sorted(self.head_hit_tokens.keys()):
            head_stats[head_idx] = {
                'hit_tokens': self.head_hit_tokens[head_idx],
                'candidate_tokens': self.head_candidate_tokens[head_idx],
                'hit_rate': self.get_hit_rate(head_idx)
            }
        
        return {
            'layer_idx': self.layer_idx,
            'forward_count': self.forward_count,
            'step_count': self.step_count,
            'current_cache_size': self.cached_indices.shape[-1] if self.cached_indices is not None else 0,
            'head_stats': head_stats,
            'average_hit_rate': self.get_hit_rate()
        }


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
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
        self.heavy_hitter_masker = CachedHeavyRecentAttentionMasker(
            heavy_budget_ratio=0.10,   # 可调
            recent_budget_ratio=0.00,  # 可调
            layer_idx=layer_idx,
        )
        self.density = None  # 观测最终保留密度
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        iController=None,   # <=== NEW: 从上层传进来的 Quest 控制器
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # =============== PATH A: Quest 稀疏（KV 由 Controller 接管） ===============
        if iController is not None:
            # 限制：Quest 目前仅支持 batch_size == 1（与你的 QuestAttention 一致）
            bsz, q_len, _ = hidden_states.shape
            assert bsz == 1, "QuestAttention only supports batch_size=1."

            # 1) 线性投影（保持与你的 QuestAttention 完全一致的布局 NHD）
            query_states = self.q_proj(hidden_states)  # [1, L, H*D]
            key_states   = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(q_len, self.config.num_attention_heads, self.head_dim)
            key_states   = key_states.view(q_len, self.config.num_key_value_heads, self.head_dim)
            value_states = value_states.view(q_len, self.config.num_key_value_heads, self.head_dim)

            # 2) RoPE（用你的 in-place 版本，offset=已缓存长度）
            #    注意：Quest 自己算 offset，不使用上游 cos/sin
            rope_offset = iController.kv_cache.seqlen - q_len
            quest_utils.apply_rope_in_place(
                query_states, key_states, rope_offset, rope_scale=1.0  # 若有 scale，可从 config 读
            )

            # 3) 追加 KV 到 Quest 的缓存（内部是分页 NHD 格式）
            quest_utils.append_kv(key_states, value_states, iController, self.layer_idx)

            # 4) prefill / decode 分支
            if q_len > 1:
                # prefill：密集计算（但由 Quest 的 prefill kernel 完成）
                attn_output = quest_utils.prefill_forward(query_states, iController, self.layer_idx)
            else:
                # decode：估计 -> topk -> 稀疏解码（或直连 full）
                if not iController.need_estimate():
                    attn_output = quest_utils.decode_sparse_attn(
                        query_states, iController, self.layer_idx, iController.kv_indices_without_last
                    )
                else:
                    est = quest_utils.decode_estimate(query_states, iController, self.layer_idx)
                    quest_utils.decode_topk(est, iController)
                    attn_output = quest_utils.decode_sparse_attn(
                        query_states, iController, self.layer_idx, iController.topk_dindices_buffer
                    )

            # 5) 还原 [B, L, H*D]，再过 o_proj
            attn_output = attn_output.unsqueeze(0)  # [1, L, H, D]
            exp_shape = (1, q_len, self.config.num_attention_heads, self.head_dim)
            if attn_output.size() != exp_shape:
                raise ValueError(f"`attn_output` should be {exp_shape}, got {attn_output.size()}")
            attn_output = attn_output.reshape(1, q_len, self.config.num_attention_heads * self.head_dim)
            attn_output = self.o_proj(attn_output)
            return attn_output, None  # Quest 路径不返回权重

        # =============== PATH B: 保留原有（heavy+recent 稀疏 或 默认） ===============
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        bs, H, Lq, D = query_states.shape
        kvH = key_states.shape[1]
        group_size = self.num_key_value_groups
        Lk = key_states.shape[-2]
        dtype = query_states.dtype
        device = query_states.device
        minv = torch.finfo(dtype).min

        # 因果 bias（仅用于 heavy 选取阶段，避免未来泄漏）
        causal_keep = torch.tril(torch.ones((1, 1, Lq, Lk), dtype=torch.bool, device=device), diagonal=0)
        causal_bias = torch.where(causal_keep, torch.tensor(0.0, dtype=dtype, device=device),
                                             torch.tensor(minv, dtype=dtype, device=device))

        base_mask = attention_mask[:, :, :, :Lk] if attention_mask is not None else causal_bias
        use_sparse = (self.layer_idx >= 2)

        if use_sparse:
            qg  = query_states.view(bs, kvH, group_size, Lq, D)
            kgT = key_states.transpose(-2, -1)[:, :, None, :, :]
            attn_g = torch.matmul(qg, kgT) * self.scaling
            attn_for_topk = attn_g.reshape(bs, H, Lq, Lk)

            attn_mask_sparse, density = self.heavy_hitter_masker(attn_for_topk + causal_bias, group_size=group_size)
            self.density = density
            combined_mask = base_mask + attn_mask_sparse
        else:
            combined_mask = base_mask

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            combined_mask,
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
            iController=kwargs.pop("iController", None),
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

        # Initialize weights and apply final processing
        self.post_init()
        # === NEW: Quest hooks (默认关闭) ===
        self.iController = None
        self._quest_page_size = None
        self._quest_page_budget = None
        self._quest_max_page_limit = None
        self._quest_skip_layer = 0  # 与你的 Quest 版本对齐：>=2 层后启用稀疏可在 Attention 内部判断

    def quest_init(
        self,
        page_size: int,
        max_seq_len: int,
        token_budget: int = 512,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda:0"),
        skip_until_layer: int = 2,           # 与你的 QuestAttention 逻辑保持一致
        max_page_limit: int = 1024 * 1024,   # 任意大的上限
    ):
        assert self.iController is None, "Quest Controller already initialized."
        self._quest_page_size = page_size
        self._quest_page_budget = max(1, token_budget // page_size)
        self._quest_max_page_limit = max_page_limit
        self._quest_skip_layer = int(skip_until_layer)

        cfg = self.config
        self.iController = quest_utils.InferenceController(
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
            page_size=page_size,
            page_budget=self._quest_page_budget,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )

    def quest_clear(self):
        assert self.iController is not None, "Quest Controller is not initialized."
        self.iController.clean_states()

    def _quest_prepare_seq(self, q_len: int):
        """在每个 forward 的开头调用，准备本 step 的元数据/页预算"""
        assert self.iController is not None
        # 为新追加的 tokens 准备 metadata（仿照你的 llama.py）
        self.iController.prepare_metadata(q_len)
 
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

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
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
        if self.iController is not None:
            # inputs_embeds.shape[1] 即本 step 的 q_len
            self._quest_prepare_seq(inputs_embeds.shape[1])

            # “跳层”策略与 token 预算：参考你的 Quest 代码
            # 这里用简化版：在第 skip_until_layer 之前不做近似（让 Controller 进入估计态）
            # 在进入第 skip_until_layer 层时，切到“常规预算”
            self.iController.set_page_budget(self._quest_max_page_limit)  # 预过程给大预算
            self.iController.begin_forward(inputs_embeds.shape[1])
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if self.iController is not None and layer_idx == self._quest_skip_layer:
                self.iController.end_forward()
                self.iController.set_page_budget(self._quest_page_budget)
                self.iController.begin_forward(inputs_embeds.shape[1], updateTensor=True)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                iController=self.iController,   # <=== NEW
                **kwargs,
            )
        if self.iController is not None:
            self.iController.end_forward()
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
    def quest_init(self, *args, **kwargs):
        return self.model.quest_init(*args, **kwargs)

    def quest_clear(self):
        return self.model.quest_clear()
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
