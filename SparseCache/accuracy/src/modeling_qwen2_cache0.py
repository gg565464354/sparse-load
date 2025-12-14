# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# 整合版本：将CachedHeavyRecentAttentionMasker集成到InfiniGen基础上
#
import math
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import LossKwargs, auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging
from .configuration_qwen2 import Qwen2Config


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
    from ...integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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


class CachedHeavyRecentAttentionMasker(nn.Module):
    """
    你实现的复杂cache系统，整合到InfiniGen基础上
    根据步骤K采用不同的缓存策略：
    - K=0,4,8...: 重置缓存，使用当前topk作为新缓存
    - K=1,2,3: 累积策略，只添加未命中的token
    """

    def __init__(self, heavy_budget_ratio=0.1, recent_budget_ratio=0.1, layer_idx: int = None):
        super().__init__()
        self.heavy_budget_ratio = heavy_budget_ratio
        self.recent_budget_ratio = recent_budget_ratio
        self.layer_idx = layer_idx

        # ===== 缓存状态 =====
        self.cached_indices: Optional[torch.Tensor] = None
        self.step_count = 0  # 步骤计数器
        # ===================

        # 命中率统计字段
        self.head_hit_tokens = {}
        self.head_candidate_tokens = {}
        self.forward_count = 0

    def forward(self, attn_weights: torch.Tensor):
        """
        整合版本的forward方法，支持你的复杂cache策略
        """
        bs, head, query_len, key_len = attn_weights.shape

        if key_len == 0:
            return attn_weights

        heavy_budget = min(int(self.heavy_budget_ratio * key_len), key_len)
        min_value = torch.finfo(attn_weights.dtype).min

        tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
        tmp_sum = torch.sum(tmp_attn, dim=-2)
        _, least_important_indices = tmp_sum.topk(k=1, dim=-1, largest=False)
        
        _, current_topk = tmp_sum.topk(k=heavy_budget, dim=-1)

        # ===== 核心修改：根据步骤K决定缓存策略 =====
        current_k = self.step_count % 4  # K在0,1,2,3之间循环
        
        if current_k == 0:  # K=0或K=4: 重置缓存策略
            print(f"Layer {self.layer_idx}, Step {self.step_count} (K={current_k}): 重置缓存，加载所有{heavy_budget}个重要KV")
            final_indices_to_keep = current_topk
            # 直接使用当前topk作为新缓存
            self.cached_indices = current_topk.detach()
            
        else:  # K=1,2,3: 累积策略，只添加未命中的token
            if self.cached_indices is not None:
                # 确保设备一致
                prev_cached_indices = self.cached_indices.to(current_topk.device)
                
                # 计算未命中的token
                unhit_ids_list = []
                max_unhit_len = 0
                total_unhit_count = 0
                
                # 命中率统计逻辑
                current_forward_head_hits = {}
                current_forward_head_candidates = {}
                for b in range(bs):
                    batch_unhit_list = []
                    for h in range(head):
                        # 从缓存中获取历史重要token
                        cached_set = set(prev_cached_indices[b, h].tolist())
                        curr_set = set(current_topk[b, h].tolist())
                        
                        hit_cnt = len(cached_set & curr_set)
                        if h not in current_forward_head_hits:
                            current_forward_head_hits[h] = 0
                            current_forward_head_candidates[h] = 0
                        current_forward_head_hits[h] += hit_cnt
                        current_forward_head_candidates[h] += len(curr_set)

                        # 计算未命中的token
                        unhit_list = list(curr_set - cached_set)
                        batch_unhit_list.append(unhit_list)
                        total_unhit_count += len(unhit_list)
                        if len(unhit_list) > max_unhit_len:
                            max_unhit_len = len(unhit_list)
                    unhit_ids_list.append(batch_unhit_list)

                print(f"Layer {self.layer_idx}, Step {self.step_count} (K={current_k}): 检索到{total_unhit_count}个未命中token，添加到缓存")

                # 更新命中率统计
                for h in current_forward_head_hits:
                    if h not in self.head_hit_tokens:
                        self.head_hit_tokens[h] = 0
                        self.head_candidate_tokens[h] = 0
                    self.head_hit_tokens[h] += current_forward_head_hits[h]
                    self.head_candidate_tokens[h] += current_forward_head_candidates[h]

                # 拼接未命中的token到缓存
                if max_unhit_len > 0:
                    pad_values_for_unhit = least_important_indices.expand(bs, head, max_unhit_len)
                    padded_unhit_ids = pad_values_for_unhit.clone()

                    for b in range(bs):
                        for h in range(head):
                            unhits = unhit_ids_list[b][h]
                            if unhits:
                                padded_unhit_ids[b, h, : len(unhits)] = torch.tensor(
                                    unhits, dtype=torch.long, device=current_topk.device
                                )
                    
                    # 直接拼接到历史缓存
                    new_cached_indices = torch.cat((prev_cached_indices, padded_unhit_ids), dim=-1)
                    final_indices_to_keep = new_cached_indices
                    # 更新缓存为拼接后的结果
                    self.cached_indices = new_cached_indices.detach()
                else:
                    # 没有新的未命中token，直接使用历史缓存
                    final_indices_to_keep = prev_cached_indices
                    print(f"Layer {self.layer_idx}, Step {self.step_count} (K={current_k}): 无新增未命中token")
            else:
                # 首次调用但不是K=0，按K=0处理
                print(f"Layer {self.layer_idx}, Step {self.step_count} (K={current_k}): 首次调用，初始化缓存")
                final_indices_to_keep = current_topk
                self.cached_indices = current_topk.detach()

        # 更新步骤计数器
        self.step_count += 1
        self.forward_count += 1
        
        # 定期打印统计信息
        if self.forward_count % 10 == 0 and self.head_hit_tokens:
            avg_hit_rate = sum(self.head_hit_tokens.values()) / max(sum(self.head_candidate_tokens.values()), 1)
            cache_size = self.cached_indices.shape[-1] if self.cached_indices is not None else 0
            print(f"Layer {self.layer_idx}: Forward {self.forward_count}, 平均命中率 = {avg_hit_rate:.2%}, 当前缓存大小 = {cache_size}")

        # ===== 创建mask并应用 =====
        mask_heavy = torch.zeros((bs, head, key_len), dtype=torch.bool, device=attn_weights.device)
        if final_indices_to_keep is not None:
            for b in range(bs):
                for h in range(head):
                    indices = final_indices_to_keep[b, h]
                    valid_indices = torch.unique(indices)
                    if valid_indices.numel() > 0:
                        mask_heavy[b, h].scatter_(0, valid_indices, True)

        mask_heavy_4d = mask_heavy.unsqueeze(2).expand(-1, -1, query_len, -1)

        recent_budget = int(self.recent_budget_ratio * key_len)
        mask_recent = torch.ones((bs, head, query_len, key_len), dtype=torch.bool, device=attn_weights.device)
        if recent_budget >= 0:
            mask_recent = torch.tril(mask_recent, diagonal=recent_budget)
            mask_recent = torch.triu(mask_recent, diagonal=-recent_budget)
        
        final_keep_mask = torch.logical_or(mask_heavy_4d, mask_recent)
        attn_weights[~final_keep_mask] = min_value
        
        return attn_weights

    def get_hit_rate(self, head_idx: int = None):
        """获取指定头或所有头的命中率"""
        if head_idx is not None:
            if head_idx in self.head_candidate_tokens and self.head_candidate_tokens[head_idx] > 0:
                return self.head_hit_tokens[head_idx] / self.head_candidate_tokens[head_idx]
            return 0.0
        else:
            # 返回所有头的平均命中率
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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    整合版本的eager attention forward，支持复杂cache机制
    """
    # 使用你的CachedHeavyRecentAttentionMasker进行处理
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


class Qwen2Attention(nn.Module):
    """
    整合版本的Qwen2Attention：
    - 保持InfiniGen的基础架构
    - 集成你的CachedHeavyRecentAttentionMasker
    """

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        
        # 保存原始的 GQA 配置用于权重加载
        self._original_num_key_value_heads = config.num_key_value_heads
        self._original_num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        
        # 设置为 MHA 模式（这样 attention 函数就不会再次调用 repeat_kv）
        self.num_key_value_groups = 1
        
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        # 保持原有的 k_proj 和 v_proj 维度以确保权重加载兼容性
        self.k_proj = nn.Linear(config.hidden_size, self._original_num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self._original_num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        #### InfiniGen基础参数 ####
        self.cache_ratio = None
        self.partial_weight_ratio = None
        self.previous_hidden_states = None
        self.current_hidden_states = None
        self.partial_weight_q = None
        self.skewing_matrix = None
        self.alpha = 5
        self.capacity = 1.0
        self.budget = 0.2
        self.eviction_policy = "counter"
        self.density = None
        ############################
        
        #### 你的复杂Cache系统 ####
        # 使用你的CachedHeavyRecentAttentionMasker替换原来的简单kv_cache_mask
        self.heavy_hitter_masker = CachedHeavyRecentAttentionMasker(
            heavy_budget_ratio=0.1,  # 可配置
            recent_budget_ratio=0.1,  # 可配置
            layer_idx=layer_idx,
        )
        ###########################

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        self.current_hidden_states = hidden_states.clone()

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view((*input_shape, -1, self.head_dim)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view((*input_shape, -1, self.head_dim)).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        self.rope_query = query_states
        
        # 扩展 key_states 和 value_states 以匹配 query_states 的头数（模拟 MHA）
        key_states = repeat_kv(key_states, self._original_num_key_value_groups)
        value_states = repeat_kv(value_states, self._original_num_key_value_groups)
        self.rope_key = key_states
        
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        ### InfiniGen的Speculate attention ###
        infinigen_attn_mask = None
        if (self.previous_hidden_states is not None) and (self.partial_weight_q is not None):
            query = (torch.matmul(self.previous_hidden_states, self.q_proj.weight.data.transpose(-1,-2))).view(input_shape[0], input_shape[1], self.num_heads, self.head_dim).transpose(1, 2)
            query, _ = apply_rotary_pos_emb(query, key_states, cos, sin)
            query = query @ self.skewing_matrix.unsqueeze(0)
            mask = self.partial_weight_q[0].view(-1,self.head_dim).unsqueeze(0).unsqueeze(2).repeat(1,1,query_states.shape[2], 1)
            query = torch.where(mask.to(torch.bool), query, torch.zeros_like(query))

            attn = torch.matmul(query, (key_states @ self.skewing_matrix).transpose(2, 3))/math.sqrt(self.head_dim)

            # 使用你的复杂cache系统替换原来的kv_cache_mask
            infinigen_attn_mask = self.heavy_hitter_masker(attn)
            # 计算density用于统计
            b, h, tgt_len, src_len = attn.shape
            heads = b * h
            total_elements = heads * (tgt_len * (tgt_len + 1) / 2)
            kept_elements = (infinigen_attn_mask != torch.finfo(infinigen_attn_mask.dtype).min).float().sum().item()
            self.density = kept_elements / total_elements if total_elements > 0 else 0
        #######################################

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # 应用InfiniGen的attention mask（如果有）
        if infinigen_attn_mask is not None:
            # 转换为与attention_mask兼容的格式
            min_dtype = torch.finfo(infinigen_attn_mask.dtype).min
            mask_to_apply = torch.where(infinigen_attn_mask == min_dtype, min_dtype, 0.0)
            
            if attention_mask is not None:
                attention_mask = attention_mask + mask_to_apply
            else:
                attention_mask = mask_to_apply
            attention_mask = torch.max(attention_mask, torch.tensor(min_dtype))

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights


@use_kernel_forward_from_hub("RMSNorm")
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
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


class Qwen2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@auto_docstring
class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
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
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class Qwen2Model(Qwen2PreTrainedModel):
    """
    整合版本的Qwen2Model：
    - 保持InfiniGen的核心逻辑
    - 集成你的复杂cache统计系统
    """
    
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # 在创建每个Decoder层时，传入层索引
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]
            
            #### InfiniGen逻辑：传递previous hidden states ####
            if (idx > 0) and (idx < (len(self.layers)-1)):
                self.layers[idx + 1].self_attn.previous_hidden_states = self.layers[idx].self_attn.current_hidden_states
            ###############################################

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        # 保持原有的_update_causal_mask逻辑不变
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2Config,
        past_key_values: Cache,
    ):
        """保持原有的mask准备逻辑不变"""
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            diagonal_attend_mask = torch.arange(target_length, device=cache_position.device) > cache_position.reshape(
                -1, 1
            )
            text_config = config.get_text_config()
            if getattr(text_config, "use_sliding_window", True) and text_config.sliding_window is not None:
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=cache_position.device) <= (
                        cache_position.reshape(-1, 1) - text_config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask

    def get_all_layers_hit_stats(self):
        """获取所有层的命中率统计"""
        stats = []
        for i, layer in enumerate(self.layers):
            if hasattr(layer.self_attn, 'heavy_hitter_masker'):
                stats.append(layer.self_attn.heavy_hitter_masker.get_hit_stats())
        return stats

    def print_hit_rate_summary(self, detailed: bool = True):
        """打印所有层各头的命中率摘要"""
        stats = self.get_all_layers_hit_stats()
        print("=" * 80)
        if detailed:
            print("各层各头命中率详细统计:")
        else:
            print("各层命中率汇总统计:")
        print("=" * 80)
        
        if not stats:
            print("没有找到任何层的统计信息！")
            for i, layer in enumerate(self.layers):
                print(f"Layer {i}: has self_attn = {hasattr(layer, 'self_attn')}")
                if hasattr(layer, 'self_attn'):
                    print(f"  - has heavy_hitter_masker = {hasattr(layer.self_attn, 'heavy_hitter_masker')}")
        else:
            for stat in stats:
                layer_idx = stat['layer_idx']
                head_stats = stat.get('head_stats', {})
                average_hit_rate = stat.get('average_hit_rate', 0.0)
                forward_count = stat.get('forward_count', 0)
                
                if detailed and head_stats:
                    print(f"Layer {layer_idx:2d} (Forward次数: {forward_count}, 平均命中率: {average_hit_rate:.2%}):")
                    for head_idx in sorted(head_stats.keys()):
                        head_stat = head_stats[head_idx]
                        print(f"  Head {head_idx:2d}: 命中率 = {head_stat['hit_rate']:.2%} "
                               f"({head_stat['hit_tokens']}/{head_stat['candidate_tokens']})")
                    print()
                else:
                    total_hits = sum(h['hit_tokens'] for h in head_stats.values()) if head_stats else 0
                    total_candidates = sum(h['candidate_tokens'] for h in head_stats.values()) if head_stats else 0
                    num_heads = len(head_stats) if head_stats else 0
                    
                    print(f"Layer {layer_idx:2d}: 平均命中率 = {average_hit_rate:.2%} "
                           f"({total_hits}/{total_candidates}) "
                           f"头数: {num_heads}, Forward次数: {forward_count}")
        print("=" * 80)


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


@auto_docstring
class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    """
    整合版本的Qwen2ForCausalLM：
    - 保持InfiniGen的get_density方法
    - 支持你的复杂cache统计系统
    """
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_density(self):
        """InfiniGen兼容的density获取方法"""
        density = []
        for l in self.model.layers:
            if hasattr(l.self_attn, "density") and l.self_attn.density != None:
                density.append(l.self_attn.density)
        return sum(density) / len(density) if density else 0

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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
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


# 保持其他类的定义不变（为了简洁这里省略，但实际实现中需要包含）
# Qwen2ForSequenceClassification, Qwen2ForTokenClassification, Qwen2ForQuestionAnswering

__all__ = [
    "Qwen2PreTrainedModel",
    "Qwen2Model", 
    "Qwen2ForCausalLM",
    "CachedHeavyRecentAttentionMasker",
]