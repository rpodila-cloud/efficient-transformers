"""GLM DSA (DeepSeek sparse attention) indexer + sparse-MLA forward.

Math is a line-for-line port of ``glm_dsa/glm_mla_dsa_microbench.py``: the
autograd-Function custom ops become ``cache.gather_rows`` / ``cache.paged_scatter``
(``custom_ops.call``), and only the paths the benchmark uses are kept
(decode Top-K, parallel prefill Top-K, sparse prefill/decode attention).
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .cache import gather_rows, paged_scatter

MIN_MASKED_ATTENTION_VALUE = float("-inf")
TS16_NUM_DEVICES = 16
AI100_CORES_PER_DEVICE = 16
DSA_INDEXER_CP = TS16_NUM_DEVICES
DSA_ATTENTION_DP = TS16_NUM_DEVICES
@dataclass
class DeepseekV3Config:
    attention_dropout: float
    hidden_size: int
    num_attention_heads: int
    max_position_embeddings: int
    rope_theta: float
    q_lora_rank: Optional[int]
    qk_rope_head_dim: int
    kv_lora_rank: int
    v_head_dim: int
    qk_nope_head_dim: int
    attention_bias: bool = False
    rope_scaling: Optional[dict[str, Any]] = None
    model_type: str = "kimi_k25"
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 64
class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Spell rsqrt as Reciprocal(Sqrt) so ONNX does not lower it to a
        # generic Div node.  The reciprocal is scalar per row, so this keeps
        # the same RMSNorm computation without an element-div kernel.
        hidden_states = hidden_states * torch.reciprocal(
            torch.sqrt(variance + self.variance_epsilon)
        )
        return self.weight * hidden_states.to(input_dtype)
class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype),
            self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype),
        )
class DeepseekV3LinearScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        device: Optional[torch.device] = None,
        scaling_factor: float = 1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
class DeepseekV3DynamicNTKScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        device: Optional[torch.device] = None,
        scaling_factor: float = 1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
def yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))
def yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> tuple[int, int]:
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)
def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
def yarn_linear_ramp_mask(min_value: float, max_value: float, dim: int) -> torch.Tensor:
    if min_value == max_value:
        max_value += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_value) / (max_value - min_value)
    return torch.clamp(linear_func, 0, 1)
class DeepseekV3YarnRotaryEmbedding(DeepseekV3RotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim
        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False)
class DeepseekV3Attention(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            warnings.warn(
                "Instantiating DeepseekV3Attention without `layer_idx` may break caching-dependent forward paths.",
                stacklevel=2,
            )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.is_causal = True
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()
        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
    def _init_rope(self) -> None:
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV3RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
            return
        scaling_type = self.config.rope_scaling["type"]
        scaling_factor = self.config.rope_scaling["factor"]
        if scaling_type == "linear":
            self.rotary_emb = DeepseekV3LinearScalingRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        elif scaling_type == "dynamic":
            self.rotary_emb = DeepseekV3DynamicNTKScalingRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        elif scaling_type == "yarn":
            kwargs = {
                key: self.config.rope_scaling[key]
                for key in [
                    "original_max_position_embeddings",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                ]
                if key in self.config.rope_scaling
            }
            self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
@dataclass
class MLAConfig:
    absorption: bool = True
    online: bool = False
    cache_compressed: bool = True
@dataclass
class DSAConfig:
    enabled: bool = False
    indexer_only: bool = False
    attention_only: bool = False
    qkv_only: bool = False
    topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 64
    indexer_dp: int = 1
    indexer_cp: int = DSA_INDEXER_CP
    indexer_kvp: int = 1
    attention_dp: int = DSA_ATTENTION_DP
    attention_cp: int = 1
    attention_kvp: int = 1
    indexer_num_blocks: int = 16
    index_block_size: int = 128
    indexer_ql_chunk: int = 128
    indexer_q_block_size: int = 32
    indexer_topk_blocking: int = 16
    indexer_prefill_parallel: bool = False
    sparse_q_block_size: int = 128
    sparse_kv_num_blocks: int = 1
    num_cores: int = AI100_CORES_PER_DEVICE
    num_devices: int = 1
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
def apply_single_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)




def scatter_dsa_sharded_cache(
    cache: torch.Tensor,
    global_positions: torch.Tensor,
    updates: torch.Tensor,
    dp: int,
    cp: int,
    layout_name: str,
) -> torch.Tensor:
    """Write attention-cache updates in the legacy token-interleaved layout.

    Physical retained-state shape is
    [B/DP, DP*CP, T/CP, D].  The DSA indexer does not use this mapping; it uses
    :func:`scatter_dsa_indexer_cache` with MiniMax's block-striped layout.
    """
    batch, seq_len, dim = updates.shape
    if batch % dp:
        raise ValueError(f"DSA batch size must be divisible by {layout_name}_dp.")
    batch_local = batch // dp

    expected_prefix = (batch_local, dp * cp)
    if cache.shape[:2] != expected_prefix:
        raise ValueError(
            f"DSA {layout_name} cache must start with [B/DP,DP*CP]; "
            f"expected {expected_prefix}, got {tuple(cache.shape[:2])}."
        )

    # Runtime batch order is DP-major: [dp0 local batch, dp1 local batch, ...].
    # Convert once to [B_local, DP, ...], exactly as MiniMax _select_blocks does.
    updates_dp = updates.view(dp, batch_local, seq_len, dim).permute(1, 0, 2, 3)
    positions_dp = global_positions.view(dp, batch_local, seq_len).permute(1, 0, 2)

    rows = dp * cp
    updates_rows = updates_dp.unsqueeze(2).expand(
        batch_local, dp, cp, seq_len, dim
    ).reshape(batch_local, rows, seq_len, dim)

    if cp == 1:
        # With one CP lane, every row is live and logical position equals the
        # physical address. Keep this path entirely free of Div/Mod nodes.
        block_id = torch.arange(batch_local, device=updates.device).view(
            batch_local, 1, 1
        ).expand(batch_local, rows, seq_len).to(torch.int32)
        addr = positions_dp.unsqueeze(2).reshape(
            batch_local, rows, seq_len
        ).to(torch.int32)
    else:
        # MiniMax-style row ownership: calculate the live CP way from the
        # DP-local logical position, then compare it with the physical row
        # way. Do not first replicate positions to every row and derive
        # ownership from that expanded tensor.
        live_way = (positions_dp % cp).unsqueeze(2).expand(
            batch_local, dp, cp, seq_len
        ).reshape(batch_local, rows, seq_len)
        row_idx = torch.arange(rows, device=updates.device).view(1, rows, 1)
        row_way = row_idx % cp
        row_live = row_way == live_way
        block_id = torch.arange(batch_local, device=updates.device).view(
            batch_local, 1, 1
        ).expand(batch_local, rows, seq_len)
        block_id = torch.where(
            row_live,
            block_id,
            torch.full_like(block_id, torch.iinfo(torch.int32).max),
        ).to(torch.int32)
        addr = positions_dp.unsqueeze(2).expand(
            batch_local, dp, cp, seq_len
        ).reshape(batch_local, rows, seq_len)
        addr = _trunc_div_pow2_nonnegative(addr, cp).to(torch.int32)

    # One row-parallel scatter. Invalid row lanes carry the sentinel block ID
    # and are ignored by V4CtxPagedScatterFunc; no row-by-row Python loop and
    # no zero update is needed.
    return paged_scatter(cache, block_id, addr, updates_rows)


def scatter_dsa_indexer_cache(
    cache: torch.Tensor,
    global_positions: torch.Tensor,
    updates: torch.Tensor,
    dp: int,
    cp: int,
    index_block_size: int,
) -> torch.Tensor:
    """Write indexer keys using MiniMax's block-striped GP layout.

    A logical position ``p`` is stored in CP way ``(p // block_size) % cp``
    at local address ``(p // (block_size * cp)) * block_size + p % block_size``.
    The cache rows are folded as [B/DP, DP*CP, T/CP, D].
    """
    batch, seq_len, dim = updates.shape
    if batch % dp:
        raise ValueError("DSA indexer batch size must be divisible by indexer_dp.")
    if cp < 1 or index_block_size < 1:
        raise ValueError("DSA indexer cp and index_block_size must be positive.")
    if index_block_size % cp:
        raise ValueError(
            f"index_block_size ({index_block_size}) must be divisible by indexer_cp ({cp})."
        )
    batch_local = batch // dp
    rows = dp * cp
    if cache.shape[:2] != (batch_local, rows):
        raise ValueError(
            "DSA indexer cache must start with [B/indexer_dp,indexer_dp*indexer_cp]."
        )

    updates_dp = updates.view(dp, batch_local, seq_len, dim).permute(1, 0, 2, 3)
    positions_dp = global_positions.view(dp, batch_local, seq_len).permute(1, 0, 2)
    updates_rows = updates_dp.unsqueeze(2).expand(
        batch_local, dp, cp, seq_len, dim
    ).reshape(batch_local, rows, seq_len, dim)

    if cp == 1:
        # For CP=1 the MiniMax block-striped mapping is the identity:
        # logical position p -> row 0, local address p.
        block_id = torch.arange(batch_local, device=updates.device).view(
            batch_local, 1, 1
        ).expand(batch_local, rows, seq_len).to(torch.int32)
        addr = positions_dp.unsqueeze(2).reshape(
            batch_local, rows, seq_len
        ).to(torch.int32)
    else:
        logical_index_block = _trunc_div_pow2_nonnegative(
            positions_dp,
            index_block_size,
        )
        live_way = (logical_index_block % cp).unsqueeze(2).expand(
            batch_local, dp, cp, seq_len
        ).reshape(batch_local, rows, seq_len)
        row_way = (
            torch.arange(rows, device=updates.device).view(1, rows, 1) % cp
        )
        row_live = row_way == live_way
        block_id = torch.arange(batch_local, device=updates.device).view(
            batch_local, 1, 1
        ).expand(batch_local, rows, seq_len)
        block_id = torch.where(
            row_live,
            block_id,
            torch.full_like(block_id, torch.iinfo(torch.int32).max),
        ).to(torch.int32)
        addr = (
            _trunc_div_pow2_nonnegative(logical_index_block, cp)
            * index_block_size
            + positions_dp % index_block_size
        )
        addr = addr.unsqueeze(2).expand(
            batch_local, dp, cp, seq_len
        ).reshape(batch_local, rows, seq_len).to(torch.int32)
    return paged_scatter(cache, block_id, addr, updates_rows)


def _read_dsa_indexer_block(
    cache: torch.Tensor,
    start: int,
    block_size: int,
    ctx_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather one indexer block with the lightweight custom op.

    Shapes:
      cache:       [batch_local, rows, cache_slots, index_head_dim]
      ctx_indices: [batch_local, rows, block_len] (int32)
      result:      [batch_local, rows, block_len, index_head_dim]

    ``ctx_indices`` may contain ``INT32_MAX`` sentinel entries.  The custom
    gather maps those entries to row zero; callers must mask those positions
    after the gather.  This matches the MLA gather contract while allowing
    the caller to retain KV-block reuse.
    """
    if cache.ndim != 4:
        raise ValueError(
            "DSA indexer cache must have shape "
            "[batch_local, rows, cache_slots, index_head_dim]."
        )
    end = start + block_size
    if not (0 <= start < end <= cache.shape[2]):
        raise ValueError(
            f"Invalid DSA cache block [{start}:{end}] for "
            f"cache_slots={cache.shape[2]}."
        )

    batch_local, rows, _, _ = cache.shape
    if ctx_indices is None:
        ctx_indices = torch.arange(
            start,
            end,
            device=cache.device,
            dtype=torch.int32,
        ).view(1, 1, block_size).expand(batch_local, rows, block_size)
    return gather_rows(cache, ctx_indices)


def prepare_dsa_indexer_state(
    module: nn.Module,
    hidden_states: torch.Tensor,
    q_a_proj_out: torch.Tensor,
    position_ids: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the exact DSA indexer query, key, and head-weight tensors."""
    batch, q_len, _ = hidden_states.shape
    index_n_heads = module.dsa_index_n_heads
    index_head_dim = module.dsa_index_head_dim
    rope_dim = module.qk_rope_head_dim
    q_idx = module.dsa_wq_b(q_a_proj_out).view(
        batch, q_len, index_n_heads, index_head_dim
    )
    q_pe, q_nope = torch.split(
        q_idx, [rope_dim, index_head_dim - rope_dim], dim=-1
    )
    k_idx = module.dsa_k_norm(module.dsa_wk(hidden_states))
    k_pe, k_nope = torch.split(
        k_idx, [rope_dim, index_head_dim - rope_dim], dim=-1
    )
    q_pe = apply_single_rotary_pos_emb(
        q_pe, cos, sin, position_ids, unsqueeze_dim=2
    )
    k_pe = apply_single_rotary_pos_emb(
        k_pe.unsqueeze(2), cos, sin, position_ids, unsqueeze_dim=2
    ).squeeze(2)
    q_idx = torch.cat((q_pe, q_nope), dim=-1)
    k_idx = torch.cat((k_pe, k_nope), dim=-1)
    weights = module.dsa_weights_proj(hidden_states).float() * (
        index_n_heads**-0.5
    )
    return q_idx, k_idx, weights


_FP16_MAX_VALUE = 65504.0


def _trunc_div_pow2_nonnegative(
    value: torch.Tensor,
    divisor: int,
) -> torch.Tensor:
    """Divide a non-negative integer tensor by a power of two without Div.

    QAIC lowers integer ``Div`` to an expensive element-div kernel even when
    the divisor is a compile-time power of two.  All uses in the radix path
    have non-negative operands, so converting to FP32, multiplying by the
    compile-time reciprocal, and casting back is exactly truncating division.
    The cast back to an integer is truncation toward zero.
    """
    if divisor <= 0 or divisor & (divisor - 1):
        raise ValueError(
            "_trunc_div_pow2_nonnegative requires a power-of-two divisor."
        )
    reciprocal = 2.0 ** (-(divisor.bit_length() - 1))
    return (value.to(torch.float32) * reciprocal).to(value.dtype)


def _block_index_from_boundaries_nonnegative(
    value: torch.Tensor,
    block_span: int,
    num_blocks: int,
) -> torch.Tensor:
    """Return ``floor(value / block_span)`` using boundary comparisons.

    ``block_span`` is not necessarily a power of two.  Using integer Div for
    this value creates an expensive element-div kernel in QAIC, while using a
    floating reciprocal can be off by one at an exact block boundary.  The
    number of KV blocks is static and small (normally four), so count the
    crossed boundaries instead:

      floor(pos / span) == sum(pos >= boundary * span,
                                boundary=1 .. num_blocks-1)

    Input shape is preserved.  The DSA caller passes ``[local, rows, 1]`` and
    then reshapes the result to ``[local, rows, 1, 1]`` before expanding over
    cores.
    """
    if block_span <= 0:
        raise ValueError("block_span must be positive.")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive.")
    if num_blocks == 1:
        return torch.zeros_like(value, dtype=torch.int32)

    block_index = torch.zeros_like(value, dtype=torch.int32)
    for boundary in range(1, num_blocks):
        block_index = block_index + (
            value >= boundary * block_span
        ).to(torch.int32)
    return block_index




def _select_device_topk_fp16(
    replicated_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    topk: int,
    cores: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select Top-K with the native ``torch.topk`` operator.

    Input shapes:
      replicated_scores: [1, rows, C, T]
      candidate_indices: [1, rows, T]

    Output shapes:
      device_scores:  [1, rows, min(topk, T)]
      device_indices: [1, rows, min(topk, T)]

    Stage 1 uses T=CL/CP.  The CP-replicated stage uses T=CP*device_topk.
    The rows axis is never reduced or exchanged inside this function.

    The C dimension is an explicit replication dimension.  All C slices are
    identical before this function is called, so only the first slice is
    consumed for TopK; this avoids an unnecessary C-wise reduction before the
    native TopK operator.
    """
    batch_step, rows, score_cores, device_width = replicated_scores.shape
    if batch_step != 1:
        raise ValueError("Device Top-K selection expects one local batch.")
    if score_cores != cores:
        raise ValueError("Device Top-K score core axis must equal num_cores.")
    if candidate_indices.shape != (
        batch_step,
        rows,
        device_width,
    ):
        raise ValueError(
            "Device Top-K candidate indices must be [1,rows,T]."
        )

    replicated_scores = replicated_scores.to(torch.float16)
    scores = replicated_scores[:, :, 0, :]
    # scores: [1, rows, T]
    device_width = scores.shape[-1]
    device_topk = min(topk, device_width)
    if device_topk < 1:
        raise ValueError("Device Top-K requires T > 0.")

    # TopK positions: [1, rows, K], int64.
    device_scores, topk_positions = torch.topk(
        scores,
        k=device_topk,
        dim=-1,
        largest=True,
        sorted=False,
    )
    # Gather the original candidate ids: [1, rows, K], int32.
    device_indices = torch.gather(
        candidate_indices.to(torch.int32),
        dim=-1,
        index=topk_positions,
    )

    # If a row contains fewer than K valid scores, TopK returns masked
    # positions as well.  Keep the output width fixed and mark those slots
    # with the same sentinels used by the rest of the DSA path.
    output_valid = device_scores > MIN_MASKED_ATTENTION_VALUE
    device_scores = torch.where(
        output_valid,
        device_scores,
        torch.full_like(device_scores, MIN_MASKED_ATTENTION_VALUE),
    )
    device_indices = torch.where(
        output_valid,
        device_indices,
        torch.full_like(device_indices, torch.iinfo(torch.int32).max),
    )
    # device_scores/device_indices: [1, rows, min(topk, T)]
    return device_scores, device_indices


def build_dsa_parallel_topk_indices(
    module: nn.Module,
    hidden_states: torch.Tensor,
    q_a_proj_out: torch.Tensor,
    position_ids: torch.Tensor,
    indexer_key_cache: torch.Tensor,
    dsa_config: DSAConfig,
    indexer_state: Optional[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
    ql_chunk: int = 128,
    q_block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DSA Top-K with the physical KV/core/device hierarchy.

    The indexer cache is consumed as compact KV blocks.  For each local batch,
    all core-local KV-block scores are joined on the token dimension.  Native
    Top-K selects each device's candidates, then a second Top-K merges the
    candidates exchanged across CP devices.
    """
    batch, q_len, _ = hidden_states.shape
    if q_len != 1:
        raise ValueError("DSA Top-K selection is decode-only and requires Q=1.")
    dp = dsa_config.indexer_dp
    cp = dsa_config.indexer_cp
    cores = dsa_config.num_cores
    num_blocks = dsa_config.indexer_num_blocks
    if batch % dp:
        raise ValueError("DSA batch size must be divisible by indexer_dp.")
    batch_local = batch // dp
    folded = (
        indexer_key_cache.ndim == 4
        and indexer_key_cache.shape[0] == batch_local
        and indexer_key_cache.shape[1] == dp * cp
    )
    if not folded:
        raise ValueError(
            "DSA indexer cache must have shape "
            "[B/indexer_dp,indexer_dp*indexer_cp,T/indexer_cp,D]."
        )
    local_slots = int(indexer_key_cache.shape[2])
    index_n_heads = module.dsa_index_n_heads
    index_head_dim = module.dsa_index_head_dim
    if num_blocks < 1 or cores < 1:
        raise ValueError("DSA indexer_num_blocks and num_cores must be positive.")
    if local_slots % num_blocks:
        raise ValueError("CP-local context must divide indexer_num_blocks.")
    kv_block = num_blocks
    cache_block_size = local_slots // kv_block
    if cache_block_size % cores:
        raise ValueError(
            "Each DSA KV block must divide evenly across num_cores."
        )
    tokens_per_block_per_core = cache_block_size // cores
    logical_block_span = cache_block_size * cp
    # A core owns its portion of the complete local attention space, not just
    # its portion of one KV block.  Keep this expression explicit because it
    # is a static scheduling quantity used by the compiler mapping.
    tokens_per_core = cache_block_size * kv_block // cores

    if indexer_state is None:
        cos, sin = module.rotary_emb(
            hidden_states, seq_len=module.max_position_embeddings
        )
        q_idx, k_idx, weights = prepare_dsa_indexer_state(
            module, hidden_states, q_a_proj_out, position_ids, cos, sin
        )
    else:
        q_idx, k_idx, weights = indexer_state
    indexer_key_cache = scatter_dsa_indexer_cache(
        indexer_key_cache,
        position_ids,
        k_idx,
        dp=dsa_config.indexer_dp,
        cp=dsa_config.indexer_cp,
        index_block_size=dsa_config.index_block_size,
    )
    # Runtime batch order is DP-major.  Keep DP explicit while the weighted
    # query is built, then replicate it over the CP rows of each DP lane.
    q_idx_dp = q_idx.view(
        dp, batch_local, q_len, index_n_heads, index_head_dim
    ).permute(1, 0, 2, 3, 4)
    weights_dp = weights.view(
        dp, batch_local, q_len, index_n_heads
    ).permute(1, 0, 2, 3)
    weighted_q = (q_idx_dp.float() * weights_dp.unsqueeze(-1)).sum(dim=3)
    rows = dp * cp
    query_rows = weighted_q.unsqueeze(2).expand(
        batch_local, dp, cp, q_len, index_head_dim
    ).reshape(batch_local, rows, q_len, index_head_dim)
    position_ids_dp = position_ids.view(
        dp, batch_local, q_len
    ).permute(1, 0, 2)
    position_rows = position_ids_dp.unsqueeze(2).expand(
        batch_local, dp, cp, q_len
    ).reshape(batch_local, rows, q_len)

    row_way = torch.arange(rows, device=hidden_states.device) % cp
    block_core_offsets = (
        torch.arange(cores, device=hidden_states.device).view(cores, 1)
        * tokens_per_block_per_core
        + torch.arange(
            tokens_per_block_per_core, device=hidden_states.device
        ).view(1, -1)
    )
    # Logical positions inside one CP-wide KV block.  This is independent of
    # block_idx and is reused to build one current-block causal mask per local
    # batch, instead of rebuilding a full [rows, C, T] token mask for every
    # KV block.
    base_cache_addr = block_core_offsets.view(
        1, 1, cores, tokens_per_block_per_core
    )
    local_block_positions = (
        (base_cache_addr // dsa_config.index_block_size)
        * (dsa_config.index_block_size * cp)
        + row_way.view(1, rows, 1, 1) * dsa_config.index_block_size
        + base_cache_addr % dsa_config.index_block_size
    )
    # local_block_positions: [1, rows, C, tokens_per_block_per_core]

    # Compute each KV-block score first.  Merge the score tensors only after
    # the KV-block loop has completed for this local batch.
    topk_index_groups: list[torch.Tensor] = []
    local_batch_size = 1
    for batch_start in range(0, batch_local, local_batch_size):
        batch_end = min(batch_start + local_batch_size, batch_local)
        local = batch_end - batch_start
        q_local = query_rows[batch_start:batch_end]
        key_local = indexer_key_cache[batch_start:batch_end]
        pos_rows_local = position_rows[batch_start:batch_end]
        # Keep the core axis explicit, matching the score layout.  This is
        # the same row/core organization used by the paged selector:
        # row_position_for_mask: [local, rows, C, 1].
        row_position_for_mask = pos_rows_local.reshape(
            local, rows, 1, 1
        ).expand(local, rows, cores, 1)
        # The current block and its offset are computed once per row/core.
        # pos_rows_local: [local, rows, 1].
        # query_block before reshape: [local, rows, 1].
        # logical_block_span can be non-power-of-two (for example,
        # ctx_len=10240, kv_block=4, cp=1 gives 2560).  Count crossed block
        # boundaries instead of emitting integer Div.
        query_block = _block_index_from_boundaries_nonnegative(
            pos_rows_local.to(torch.int32),
            logical_block_span,
            kv_block,
        ).reshape(local, rows, 1, 1).expand(local, rows, cores, 1)
        query_offset = row_position_for_mask.to(torch.int32) - (
            query_block * logical_block_span
        )
        # current_causal_mask: [local, rows, C, tokens_per_block_per_core]
        current_causal_mask = local_block_positions > query_offset
        block_score_groups: list[torch.Tensor] = []
        block_index_groups: list[torch.Tensor] = []

        for block_idx in range(kv_block):
            start = block_idx * cache_block_size
            end = start + cache_block_size
            key_block = _read_dsa_indexer_block(
                key_local,
                start,
                cache_block_size,
            )
            key_core = key_block.view(
                local,
                rows,
                cores,
                tokens_per_block_per_core,
                index_head_dim,
            ).float()
            # key_core:
            # [1, rows, C, cache_block_size/C, D]
            score_block = torch.matmul(
                q_local.float().unsqueeze(2),
                key_core.transpose(-1, -2),
            ).reshape(
                local, rows, cores, tokens_per_block_per_core
            )
            # score_block:
            # [1, rows, C, cache_block_size/C]
            block_indices = (
                local_block_positions + block_idx * logical_block_span
            ).expand(
                local, rows, cores, tokens_per_block_per_core
            )
            score_block = score_block * module.dsa_softmax_scale
            # score_block = torch.where(
            #     score_block > _FP16_MAX_VALUE,
            #     torch.full_like(score_block, _FP16_MAX_VALUE),
            #     score_block,
            # )
            # score_block = torch.where(
            #     score_block < -_FP16_MAX_VALUE,
            #     torch.full_like(score_block, -_FP16_MAX_VALUE),
            #     score_block,
            # ).to(torch.float16)
            # Match _select_blocks_paged: past blocks are untouched, future
            # blocks are fully masked, and only the current block uses the
            # precomputed per-token causal mask.
            block_is_current = query_block == block_idx
            block_is_future = query_block < block_idx
            # block_is_current/block_is_future: [local, rows, C, 1]
            # current_causal_mask/causal_mask:
            # [local, rows, C, tokens_per_block_per_core]
            causal_mask = block_is_future | (
                block_is_current & current_causal_mask
            )
            score_block = torch.where(
                causal_mask,
                torch.full_like(score_block, MIN_MASKED_ATTENTION_VALUE),
                score_block,
            )
            block_score_groups.append(score_block)
            block_index_groups.append(
                block_indices
            )

        if not block_score_groups:
            raise ValueError("DSA indexer selection produced no cache blocks.")
        core_scores = torch.cat(block_score_groups, dim=3)
        core_indices = torch.cat(block_index_groups, dim=3)
        # core_scores:
        # [1, rows, C, CL/(C*CP)]
        # core_indices:
        # [1, rows, C, CL/(C*CP)]

        # Merge the C core-owned score T ranges into the complete device-local
        # score T, then replicate that merged score range over C cores.
        device_width = cores * tokens_per_core
        merged_scores = core_scores.reshape(
            local, rows, 1, device_width
        )
        # merged_scores:
        # [1, rows, 1, CL/CP]
        # Concatenate explicit C copies instead of Expand/Tile.  The C-sharded
        # scatter split can then propagate through independent Concat inputs;
        # Expand/Tile would incorrectly try to slice the singleton C source.
        replicated_scores = torch.cat(
            [merged_scores] * cores,
            dim=2,
        )
        # replicated_scores:
        # [1, rows, C, CL/CP]
        candidate_indices = core_indices.reshape(local, rows, device_width)
        # candidate_indices: [1, rows, CL/CP]
        # replicated_scores: [1, rows, C, CL/CP]
        # Stage 1: select the device-local Top-K from [1, rows, CL/CP].
        device_scores, device_indices = _select_device_topk_fp16(
            replicated_scores,
            candidate_indices,
            dsa_config.topk,
            cores,
        )
        # device_scores/device_indices:
        # [1, rows, min(topk, CL/CP)]

        device_topk = device_scores.shape[-1]
        if device_topk != dsa_config.topk:
            raise ValueError(
                "CP-replicated Top-K requires topk <= CL/CP."
            )
        if cp == 1:
            # With one CP device, Stage 1 is already the final device Top-K.
            # Do not concatenate/repeat the candidates or run a second Top-K.
            # device_indices: [1, DP, topk]
            topk_indices_local = device_indices.view(
                local, dp, device_topk
            )
        else:
            # Multiple CP devices require one final merge Top-K over their
            # local candidates.
            scores_by_cp = device_scores.view(
                local, dp, cp, device_topk
            )
            indices_by_cp = device_indices.view(
                local, dp, cp, device_topk
            )
            # scores_by_cp/indices_by_cp: [1, DP, CP, topk]
            merged_cp_scores = scores_by_cp.reshape(
                local, dp, cp * device_topk
            )
            merged_cp_indices = indices_by_cp.reshape(
                local, dp, cp * device_topk
            )
            # merged_cp_scores/indices: [1, DP, CP*topk]

            repeated_cp_scores = merged_cp_scores.unsqueeze(2).repeat(
                1, 1, cp, 1
            ).reshape(
                local, rows, cp * device_topk
            )
            repeated_cp_indices = merged_cp_indices.unsqueeze(2).repeat(
                1, 1, cp, 1
            ).reshape(
                local, rows, cp * device_topk
            )
            # repeated_cp_scores/indices: [1, rows, CP*topk]

            repeated_cp_core_scores = torch.cat(
                [repeated_cp_scores.unsqueeze(2)] * cores,
                dim=2,
            )
            # repeated_cp_core_scores: [1, rows, C, CP*topk]
            _, repeated_topk_indices = _select_device_topk_fp16(
                repeated_cp_core_scores,
                repeated_cp_indices,
                dsa_config.topk,
                cores,
            )
            # repeated_topk_indices: [1, rows, topk]

            # CP rows contain identical results because Stage 2 is duplicated.
            repeated_topk_indices = repeated_topk_indices.view(
                local, dp, cp, repeated_topk_indices.shape[-1]
            )
            repeated_topk_sum = repeated_topk_indices.sum(
                dim=2, dtype=torch.int32
            )
            topk_indices_local = _trunc_div_pow2_nonnegative(
                repeated_topk_sum,
                cp,
            )
        # topk_indices_local:
        # [1, DP, topk]
        topk_index_groups.append(topk_indices_local)

    # Restore the external DP-major [B, K] contract used by attention gather.
    topk_indices_dp = torch.cat(topk_index_groups, dim=0)
    final_topk = topk_indices_dp.shape[-1]
    topk_indices = topk_indices_dp.permute(1, 0, 2).reshape(
        batch, final_topk
    )
    valid_topk = topk_indices.to(position_ids.dtype) <= position_ids[:, -1:]
    return topk_indices, valid_topk, indexer_key_cache




def build_dsa_prefill_topk_indices_par(
    module: nn.Module,
    hidden_states: torch.Tensor,
    q_a_proj_out: torch.Tensor,
    position_ids: torch.Tensor,
    indexer_key_cache: torch.Tensor,
    dsa_config: DSAConfig,
    indexer_state: Optional[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
    indexer_ql_chunk: int = 128,
    indexer_q_block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build prefill Top-K with K split across cores and Q replicated.

    The original prefill selector keeps Q split and replicates each KV block
    over cores.  This variant uses the opposite dataflow:

      q_core:   [B, C, Q_BLOCK, D]       # Q replicated over C
      key_core: [B, C, KV_BLOCK/C, D]    # K split over C
      scores:   [B, C, Q_BLOCK, KV_BLOCK/C]

    Scores are reordered to [B, Q_BLOCK, KV_BLOCK] before they are saved, so
    the Top-K result and causal semantics are identical to the original
    selector.  The existing selector remains the default path.
    """
    batch, q_len, _ = hidden_states.shape
    dp = dsa_config.indexer_dp
    cp = dsa_config.indexer_cp

    if indexer_state is None:
        cos, sin = module.rotary_emb(
            hidden_states, seq_len=module.max_position_embeddings
        )
        q_idx, k_idx, weights = prepare_dsa_indexer_state(
            module, hidden_states, q_a_proj_out, position_ids, cos, sin
        )
    else:
        q_idx, k_idx, weights = indexer_state

    weighted_q = (
        q_idx.float() * weights.float().unsqueeze(-1)
    ).sum(dim=2)

    indexer_key_cache = scatter_dsa_indexer_cache(
        indexer_key_cache,
        position_ids,
        k_idx,
        dp=dp,
        cp=cp,
        index_block_size=dsa_config.index_block_size,
    )

    ctx_len = int(indexer_key_cache.shape[2])
    topk = min(int(dsa_config.topk), ctx_len)
    num_blocks = int(dsa_config.indexer_num_blocks)
    num_cores = int(dsa_config.num_cores)
    assert ctx_len % num_blocks == 0, (
        "DSA parallel prefill context length must divide evenly into KV blocks: "
        f"ctx_len={ctx_len}, num_blocks={num_blocks}"
    )
    kv_block_size = ctx_len // num_blocks
    assert kv_block_size % num_cores == 0, (
        "DSA parallel prefill KV block must divide evenly across cores: "
        f"kv_block_size={kv_block_size}, num_cores={num_cores}"
    )
    tokens_per_core = kv_block_size // num_cores
    core_token_offsets = (
        torch.arange(num_cores, device=hidden_states.device).view(
            1, 1, num_cores, 1
        )
        * tokens_per_core
        + torch.arange(
            tokens_per_core,
            device=hidden_states.device,
        ).view(1, 1, 1, tokens_per_core)
    ).to(torch.int32)
    mask_value = float(MIN_MASKED_ATTENTION_VALUE)
    topk_index_chunks: list[torch.Tensor] = []
    valid_chunks: list[torch.Tensor] = []

    for q_chunk_start in range(0, q_len, indexer_ql_chunk):
        q_chunk_end = min(q_chunk_start + indexer_ql_chunk, q_len)
        query_chunk = q_chunk_end - q_chunk_start
        assert query_chunk % indexer_q_block_size == 0, (
            "DSA parallel prefill indexer_q_block_size must divide evenly "
            "into Q chunk: "
            f"query_chunk={query_chunk}, "
            f"indexer_q_block_size={indexer_q_block_size}"
        )
        topk_blocking = dsa_config.indexer_topk_blocking
        assert indexer_q_block_size % topk_blocking == 0, (
            "DSA parallel prefill Q block must divide evenly by the "
            "indexer Top-K blocking factor: "
            f"indexer_q_block_size={indexer_q_block_size}, "
            f"indexer_topk_blocking={topk_blocking}"
        )
        q_block_size = indexer_q_block_size
        weighted_q_chunk = weighted_q[:, q_chunk_start:q_chunk_end]
        query_positions = position_ids[:, q_chunk_start:q_chunk_end]
        q_block_ranges = list(
            range(q_chunk_start, q_chunk_end, q_block_size)
        )
        score_blocks_by_q: list[list[torch.Tensor]] = [
            [] for _ in q_block_ranges
        ]
        index_blocks_by_q: list[list[torch.Tensor]] = [
            [] for _ in q_block_ranges
        ]

        q_block_metadata: list[
            tuple[torch.Tensor, torch.Tensor]
        ] = []
        for q_block_start in q_block_ranges:
            q_local_start = q_block_start - q_chunk_start
            q_local_end = q_local_start + q_block_size
            weighted_q_block = weighted_q_chunk[
                :, q_local_start:q_local_end
            ]
            query_positions_block = query_positions[
                :, q_local_start:q_local_end
            ].to(torch.int32)
            q_block_metadata.append(
                (weighted_q_block, query_positions_block)
            )

        for block_idx in range(num_blocks):
            start = block_idx * kv_block_size
            end = start + kv_block_size
            key_positions = torch.arange(
                start,
                end,
                device=hidden_states.device,
                dtype=torch.int32,
            ).view(1, 1, kv_block_size)
            gather_limit = (
                query_positions.to(torch.int32)
                .max(dim=1, keepdim=True)
                .values
                .unsqueeze(1)
            )
            gather_indices = torch.where(
                key_positions > gather_limit,
                torch.full_like(
                    key_positions,
                    torch.iinfo(torch.int32).max,
                ),
                key_positions,
            ).expand(
                batch,
                indexer_key_cache.shape[1],
                kv_block_size,
            )
            key_block = _read_dsa_indexer_block(
                indexer_key_cache,
                start,
                kv_block_size,
                ctx_indices=gather_indices,
            ).squeeze(1).float()
            key_core = key_block.reshape(
                batch,
                num_cores,
                tokens_per_core,
                key_block.shape[-1],
            )

            for q_block_idx, q_block_start in enumerate(q_block_ranges):
                q_block, query_positions_block = (
                    q_block_metadata[q_block_idx]
                )
                # Keep Q as a singleton C-axis broadcast for MatMul; do not
                # materialize [B, C, Q_BLOCK, D].
                q_core = q_block.unsqueeze(1)
                block_scores_core = torch.matmul(
                    q_core,
                    key_core.transpose(-1, -2),
                ) * module.dsa_softmax_scale
                # Keep the core/token hierarchy until the Q-block merge:
                #   [B, C, Q_BLOCK, KV_BLOCK/C]
                #   -> [B, Q_BLOCK, C, KV_BLOCK/C]
                # The reshape to [B, Q_BLOCK, CTX] is intentionally performed
                # after all KV blocks are concatenated below.
                block_scores = block_scores_core.permute(
                    0, 2, 1, 3
                )
                block_key_positions = start + core_token_offsets
                block_indices = block_key_positions.expand(
                    batch,
                    q_block_size,
                    num_cores,
                    tokens_per_core,
                )
                # Match MLA: compare global KV positions with query
                # positions directly.  This single predicate covers past,
                # current, and future positions without separate
                # block-current/block-future masks.
                causal_mask = block_key_positions > (
                    query_positions_block.unsqueeze(-1).unsqueeze(-1)
                )
                block_scores = block_scores.masked_fill(
                    causal_mask,
                    mask_value,
                )
                score_blocks_by_q[q_block_idx].append(block_scores)
                index_blocks_by_q[q_block_idx].append(block_indices)

        best_scores_blocks: list[torch.Tensor] = []
        best_indices_blocks: list[torch.Tensor] = []
        # Match build_dsa_prefill_topk_indices(): concatenate all KV blocks
        # for one Q block.  Only reshape and Top-K are split into the
        # configured number of smaller Q sub-blocks.
        for q_block_idx in range(len(q_block_ranges)):
            candidate_scores_core = torch.cat(
                score_blocks_by_q[q_block_idx], dim=-1
            )
            candidate_indices_core = torch.cat(
                index_blocks_by_q[q_block_idx], dim=-1
            )
            q_sub_block_size = q_block_size // topk_blocking
            for q_sub_block_idx in range(topk_blocking):
                q_start = q_sub_block_idx * q_sub_block_size
                q_end = q_start + q_sub_block_size
                q_scores_core = candidate_scores_core[:, q_start:q_end]
                q_indices_core = candidate_indices_core[:, q_start:q_end]
                # Only this Q sub-block is reshaped before its Top-K:
                #   [B, Q_SUB_BLOCK, C, CTX/C]
                #   -> [B, Q_SUB_BLOCK, CTX]
                candidate_scores = q_scores_core.reshape(
                    batch,
                    q_sub_block_size,
                    ctx_len,
                )
                candidate_indices = q_indices_core.reshape(
                    batch,
                    q_sub_block_size,
                    ctx_len,
                )
                _, candidate_order = torch.topk(
                    candidate_scores,
                    k=topk,
                    dim=-1,
                )
                best_scores_blocks.append(
                    torch.gather(candidate_scores, -1, candidate_order)
                )
                best_indices_blocks.append(
                    torch.gather(candidate_indices, -1, candidate_order)
                )

        best_scores = torch.cat(best_scores_blocks, dim=1)
        best_indices = torch.cat(best_indices_blocks, dim=1)
        topk_index_chunks.append(best_indices)
        valid_chunks.append(best_scores > mask_value)

    topk_indices = torch.cat(topk_index_chunks, dim=1)
    valid_topk = torch.cat(valid_chunks, dim=1)
    return topk_indices, valid_topk, indexer_key_cache


def gather_dsa_attention_cache_local(
    cache: torch.Tensor,
    topk_indices: torch.Tensor,
    valid_topk: torch.Tensor,
    dsa_config: DSAConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather exact Top-K rows from the folded attention cache layout."""
    batch, gather_len = topk_indices.shape
    dp = dsa_config.attention_dp
    cp = dsa_config.attention_cp
    if batch % dp:
        raise ValueError("DSA batch size must be divisible by attention_dp.")
    batch_local = batch // dp
    expected_prefix = (batch_local, dp * cp)
    if cache.ndim != 4 or cache.shape[:2] != expected_prefix:
        raise ValueError(
            "DSA attention cache must be "
            "[B/attention_dp,attention_dp*attention_cp,T/attention_cp,D]."
        )

    topk_dp = topk_indices.view(dp, batch_local, gather_len).permute(1, 0, 2)
    valid_dp = valid_topk.view(dp, batch_local, gather_len).permute(1, 0, 2)

    # The public top-k contract is a global logical token position.  Convert
    # it once through a static logical->physical address table instead of
    # calculating `position % cp` for every [batch, CP, topk] element.
    #
    # Attention cache layout:
    #   cache: [batch_local, DP*CP, T/CP, D]
    #   flat_cache: [batch_local, DP, CP*(T/CP), D]
    # For logical position p, the absolute address in one DP lane is:
    #   (p % CP) * (T/CP) + (p // CP)
    # The table stores that address as a constant, so runtime uses Gather.
    local_slots = int(cache.shape[2])
    global_context = local_slots * cp
    physical_address_table = torch.tensor(
        [
            (position % cp) * local_slots + (position // cp)
            for position in range(global_context)
        ],
        device=cache.device,
        dtype=torch.int32,
    ).view(1, 1, global_context)
    safe_topk_dp = torch.where(
        valid_dp,
        topk_dp,
        torch.zeros_like(topk_dp),
    )
    physical_address = torch.gather(
        physical_address_table,
        -1,
        safe_topk_dp.to(torch.int64).reshape(1, 1, -1),
    ).reshape(batch_local, dp, gather_len)
    # physical_address: [batch_local, DP, topk]
    if cp == 1:
        # One-device/one-CP layout has no owner calculation to perform.
        owner_cp = torch.zeros_like(physical_address, dtype=torch.int32)
    else:
        owner_cp = _trunc_div_pow2_nonnegative(
            physical_address,
            local_slots,
        ).to(torch.int32)
    # owner_cp: [batch_local, DP, topk]
    flat_cache = cache.reshape(
        batch_local,
        dp,
        cp * local_slots,
        cache.shape[-1],
    )
    cp_rows = torch.arange(cp, device=cache.device, dtype=torch.int32).view(
        1, 1, cp, 1
    )
    row_valid = valid_dp.unsqueeze(2) & (owner_cp.unsqueeze(2) == cp_rows)
    physical_indices = physical_address.unsqueeze(2).expand(
        batch_local, dp, cp, gather_len
    )
    invalid_index = torch.full_like(
        physical_indices,
        torch.iinfo(torch.int32).max,
    )
    gather_indices = torch.where(
        row_valid,
        physical_indices,
        invalid_index,
    )

    gathered = gather_rows(flat_cache, gather_indices)
    gathered = torch.where(
        row_valid.unsqueeze(-1),
        gathered,
        torch.zeros_like(gathered),
    )
    return (
        gathered.view(batch_local, dp, cp, gather_len, -1),
        row_valid,
    )




def scatter_dsa_attention_cache(
    cache: torch.Tensor,
    global_positions: torch.Tensor,
    updates: torch.Tensor,
    dsa_config: DSAConfig,
) -> torch.Tensor:
    """Scatter attention updates into the folded DP/CP cache rows."""
    return scatter_dsa_sharded_cache(
        cache,
        global_positions,
        updates,
        dp=dsa_config.attention_dp,
        cp=dsa_config.attention_cp,
        layout_name="attention",
    )


def update_running_softmax(
    current_max: torch.Tensor,
    attn_weights_block: torch.Tensor,
    current_denominator: torch.Tensor,
    output: torch.Tensor,
    v_block: torch.Tensor,
    skip_kv: bool = False,
    skip_future: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the same online-softmax update pattern as the MLA prefill path."""
    prev_max = current_max
    current_max_updated = torch.max(
        prev_max,
        attn_weights_block.max(dim=-1).values,
    )
    delta_max = prev_max - current_max_updated
    current_exp = torch.exp(
        attn_weights_block - current_max_updated.unsqueeze(-1)
    )
    curr_exp_sum = current_exp.sum(dim=-1)
    current_denominator_updated = (
        current_denominator * torch.exp(delta_max) + curr_exp_sum
    )
    if v_block.ndim == current_exp.ndim + 1:
        if v_block.shape[2] != current_exp.shape[2]:
            # Sparse prefill keeps cKV shared by all query heads:
            #   current_exp: [B, C, Q/C*H, KV_BLOCK]
            #   v_block:     [B, C, Q/C,   KV_BLOCK, R]
            # Keep the H axis on the left operand and let MatMul consume the
            # matching [B, C, Q/C] batch axes.  Do not expand cKV over H:
            # QAIC materializes an explicit Expand and spills the replicated
            # value tile to DDR.
            queries_per_core = v_block.shape[2]
            heads_per_query = current_exp.shape[2] // queries_per_core
            current_exp_grouped = current_exp.reshape(
                current_exp.shape[0],
                current_exp.shape[1],
                queries_per_core,
                heads_per_query,
                current_exp.shape[-1],
            )
            value_update = torch.matmul(
                current_exp_grouped,
                v_block,
            ).reshape(
                current_exp.shape[0],
                current_exp.shape[1],
                current_exp.shape[2],
                v_block.shape[-1],
            )
        else:
            # Query-head-specific V:
            #   current_exp: [B, C, QH, KV_BLOCK]
            #   v_block:     [B, C, QH, KV_BLOCK, R]
            value_update = torch.matmul(
                current_exp.unsqueeze(-2),
                v_block,
            ).squeeze(-2)
    else:
        # Dense MLA path: V is shared by the folded query/head axis.
        value_update = torch.matmul(current_exp, v_block)
    output_updated = (
        output * torch.exp(delta_max.unsqueeze(-1))
        + value_update
    )
    if skip_kv:
        if skip_future is None:
            raise ValueError("skip_future is required when skip_kv is enabled.")
        current_max = torch.where(
            skip_future,
            prev_max,
            current_max_updated,
        )
        current_denominator = torch.where(
            skip_future,
            current_denominator,
            current_denominator_updated,
        )
        output = torch.where(
            skip_future.unsqueeze(-1),
            output,
            output_updated,
        )
    else:
        current_max = current_max_updated
        current_denominator = current_denominator_updated
        output = output_updated
    return current_max, current_denominator, output


def dsa_parallel_sparse_mla_attention_prefill(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    per_head_v_up: torch.Tensor,
    scaling: float,
    ckv_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    valid_topk: torch.Tensor,
    position_ids: torch.Tensor,
    num_cores: int,
    ql_chunk: int,
    kv_num_blocks: int,
) -> torch.Tensor:
    """Single-device sparse prefill with Q-only core partitioning.

    Shapes:
      query_latent: [B, H, QL, R]
      query_rope:   [B, H, QL, PE]
      topk_indices: [B, QL, K]
      cKV cache:    [B, 1, T, R]
      k_PE cache:   [B, 1, T, PE]

    Each core owns a Q sequence slice and gathers that slice's query-specific
    K entries inside the selected-KV block loop.  The running softmax state
    follows the MLA prefill ``update_running_softmax`` layout; there is no
    K/core split, KV multicast, or head blocking.
    """
    batch, num_query_heads, q_len, latent_dim = query_latent.shape
    selected_per_query = int(topk_indices.shape[-1])
    mask_value = float(MIN_MASKED_ATTENTION_VALUE)

    if q_len % num_cores:
        raise ValueError(
            "Sparse prefill Q length must be divisible by num_cores."
        )

    output_chunks: list[torch.Tensor] = []
    for q_chunk_start in range(0, q_len, ql_chunk):
        q_chunk_end = min(q_chunk_start + ql_chunk, q_len)
        query_chunk = q_chunk_end - q_chunk_start
        if query_chunk % num_cores:
            raise ValueError(
                "Every sparse prefill Q chunk must be divisible by num_cores."
            )
        q_per_core = query_chunk // num_cores
        # Keep the core dimension first in the attention work tensor.  The Q
        # sequence, not K, is split:
        #   q_latent_core: [B, C, Q/C, H, R]
        #   q_rope_core:   [B, C, Q/C, H, PE]
        #   topk_core:     [B, C, Q/C, K]
        # With Q/C=1, the per-core work is [B, C, H*Q/C, D] == [B, C, H, D],
        # so all heads for the core-owned query are computed in parallel.
        q_latent_core = query_latent[:, :, q_chunk_start:q_chunk_end].reshape(
            batch, num_query_heads, num_cores, q_per_core, latent_dim
        ).permute(0, 2, 3, 1, 4).float()
        q_rope_core = query_rope[:, :, q_chunk_start:q_chunk_end].reshape(
            batch, num_query_heads, num_cores, q_per_core, query_rope.shape[-1]
        ).permute(0, 2, 3, 1, 4).float()
        topk_core = topk_indices[:, q_chunk_start:q_chunk_end].reshape(
            batch, num_cores, q_per_core, selected_per_query
        )
        valid_core = valid_topk[:, q_chunk_start:q_chunk_end].reshape(
            batch, num_cores, q_per_core, selected_per_query
        )
        # Prefill Top-K indices are always ordinary in-range context
        # positions, including slots whose validity is false because they
        # came from a future position.  The validity mask controls whether a
        # slot contributes; it is not needed to make the gather index safe.
        # Keeping the original indices avoids another broadcast Where before
        # each cache gather.
        gather_topk_core = topk_core

        # All query heads are evaluated in one tensor.  Fold the query/head
        # axes exactly like MLA's [B, split, rc*tc, D] online-softmax path:
        #   q_abs:  [B, C, Q/C, H, R+PE]
        #   score work: [B, C, (Q/C)*H, R+PE]
        q_abs = torch.cat((q_latent_core, q_rope_core), dim=-1)
        q_head_width = q_per_core * num_query_heads

        # Keep the same accumulator layout as MLA's update_running_softmax:
        #   m_acc: [B, C, (Q/C)*H]
        #   s_acc: [B, C, (Q/C)*H]
        #   o_acc: [B, C, (Q/C)*H, R]
        # Use a finite lower bound instead of -inf.  This makes an entirely
        # invalid selected-KV block contribute exp(-inf)=0 without needing
        # three state-restoring Where nodes in update_running_softmax().
        # Valid scores are finite, so this is equivalent to the usual
        # -inf-initialized running maximum for the attention result.
        work_dtype = q_abs.dtype
        m_acc = torch.full(
            (batch, num_cores, q_head_width),
            -_FP16_MAX_VALUE,
            dtype=work_dtype,
            device=q_abs.device,
        )
        s_acc = torch.zeros_like(m_acc)
        o_acc = torch.zeros(
            batch,
            num_cores,
            q_head_width,
            latent_dim,
            dtype=work_dtype,
            device=q_abs.device,
        )
        for kv_block_idx in range(kv_num_blocks):
            # Split the selected Top-K axis into exactly ``kv_num_blocks``
            # contiguous pieces without padding.  If K is not divisible by
            # the block count, neighboring blocks differ by at most one.
            kv_block_start = (
                selected_per_query * kv_block_idx
            ) // kv_num_blocks
            kv_block_end = (
                selected_per_query * (kv_block_idx + 1)
            ) // kv_num_blocks
            topk_block = gather_topk_core[
                ..., kv_block_start:kv_block_end
            ]
            valid_block = valid_core[
                ..., kv_block_start:kv_block_end
            ]

            # Gather only the current sparse-KV block.  The gather sees:
            #   cache:      [B, 1, T, D]
            #   indices:    [B, 1, C, Q/C, KV_BLOCK]
            #   gathered:   [B, C, Q/C, KV_BLOCK, D]
            # This keeps the cache read inside the KV-block loop and avoids
            # materializing the complete [B, C, Q/C, K, D] gathered cache.
            ckv_block = gather_rows(
                ckv_cache,
                topk_block.unsqueeze(1),
            ).squeeze(1).float()
            k_pe_block = gather_rows(
                k_pe_cache,
                topk_block.unsqueeze(1),
            ).squeeze(1).float()
            key_block = torch.cat((ckv_block, k_pe_block), dim=-1)
            # Keep the previous Q-chunk -> KV-block dataflow, but flatten the
            # matching [B, C, Q/C] batch axes before the score MatMul.  This
            # follows the 0907 dataflow: Q and K are never expanded over H.
            #   q_work:   [B*C*(Q/C), H, R+PE]
            #   k_work:   [B*C*(Q/C), R+PE, KV_BLOCK]
            #   scores:   [B*C*(Q/C), H, KV_BLOCK]
            q_score_work = q_abs.reshape(
                batch * num_cores * q_per_core,
                num_query_heads,
                q_abs.shape[-1],
            )
            k_score_work = key_block.reshape(
                batch * num_cores * q_per_core,
                kv_block_end - kv_block_start,
                key_block.shape[-1],
            ).transpose(1, 2)
            scores_block = torch.bmm(
                q_score_work,
                k_score_work,
            ).reshape(
                batch,
                num_cores,
                q_per_core,
                num_query_heads,
                kv_block_end - kv_block_start,
            ) * scaling
            # Match MLA's mask layout: keep the validity predicate at the
            # query level and broadcast only over the shared query-head axis.
            #   scores_block: [B, C, Q/C, H, KV_BLOCK]
            #   valid_block:  [B, C, Q/C, KV_BLOCK]
            #   mask:         [B, C, Q/C, 1, KV_BLOCK]
            attn_block = scores_block.masked_fill(
                ~valid_block.unsqueeze(3),
                mask_value,
            ).reshape(
                batch,
                num_cores,
                q_head_width,
                kv_block_end - kv_block_start,
            )

            # cKV is shared by all query heads.  Keep it as
            # [B, C, Q/C, KV_BLOCK, R] and let update_running_softmax()
            # perform the weighted V MatMul with the explicit head axis.
            # Do not materialize [B, C, Q/C, H, KV_BLOCK, R].
            m_acc, s_acc, o_acc = update_running_softmax(
                m_acc,
                attn_block,
                s_acc,
                o_acc,
                ckv_block,
                skip_kv=False,
            )

        latent_output = o_acc / s_acc.clamp_min(1.0).unsqueeze(-1)
        latent_output = latent_output.reshape(
            batch,
            num_cores,
            q_per_core,
            num_query_heads,
            latent_dim,
        )
        # latent_output: [B, C, Q/C, H, R].  Restore H before flattening the
        # disjoint core/Q ranges for the per-head V-up projection:
        #   latent_output_flat: [B, H, Q, R]
        latent_output_flat = latent_output.permute(0, 3, 1, 2, 4).reshape(
            batch,
            num_query_heads,
            query_chunk,
            latent_dim,
        )
        output_flat = torch.matmul(
            latent_output_flat.to(per_head_v_up.dtype),
            per_head_v_up,
        )
        output_core = output_flat.reshape(
            batch,
            num_query_heads,
            num_cores,
            q_per_core,
            output_flat.shape[-1],
        )
        # [B, H, C, Q/C, V] -> [B, H, Q, V].  The Q ranges are disjoint, so
        # reshape is a concatenation, not a reduction or synchronization.
        output_chunks.append(
            output_core.reshape(
                batch,
                num_query_heads,
                query_chunk,
                output_core.shape[-1],
            )
        )

    return torch.cat(output_chunks, dim=2).transpose(1, 2).contiguous()


def dsa_parallel_sparse_mla_attention_decode(
    module: nn.Module,
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    per_head_v_up: torch.Tensor,
    absorption: bool,
    scaling: float,
    ckv_cache: Optional[torch.Tensor],
    k_pe_cache: Optional[torch.Tensor],
    topk_indices: torch.Tensor,
    valid_topk: torch.Tensor,
    dsa_config: DSAConfig,
    current_ckv: Optional[torch.Tensor] = None,
    current_k_pe: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    batch, num_query_heads, q_len, query_latent_dim = query_latent.shape
    kv_lora_rank = module.config.kv_lora_rank
    if query_latent_dim != kv_lora_rank:
        raise ValueError("DSA latent query width must equal kv_lora_rank.")
    if not absorption:
        raise ValueError("The DSA benchmark requires MLA absorption mode.")
    cores = dsa_config.num_cores
    dp = dsa_config.attention_dp
    cp = dsa_config.attention_cp
    if dp < 1 or batch % dp:
        raise ValueError("DSA attention batch must be divisible by attention_dp.")
    batch_local = batch // dp
    if ckv_cache is None or k_pe_cache is None:
        raise ValueError("DSA attention requires cKV and k_PE caches.")

    selected_per_row = topk_indices.shape[-1]
    if selected_per_row < 1:
        raise ValueError("DSA attention requires at least one Top-K slot.")
    if tuple(valid_topk.shape) != tuple(topk_indices.shape):
        raise ValueError("DSA Top-K indices and validity shapes must match.")

    # External inputs are DP-major: [DP, batch_local, K].  Convert once to
    # the local-batch view used by the loop below:
    #   topk_dp_all:   [batch_local, DP, K]
    #   valid_dp_all:  [batch_local, DP, K]
    topk_dp_all = topk_indices.view(dp, batch_local, selected_per_row).permute(
        1, 0, 2
    )
    valid_dp_all = valid_topk.view(
        dp, batch_local, selected_per_row
    ).permute(1, 0, 2)

    if current_ckv is not None and current_k_pe is not None and position_ids is not None:
        # Keep current-token tensors in the same local-batch/DP order:
        #   current_ckv_dp:   [batch_local, DP, 1, kv_lora_rank]
        #   current_k_pe_dp: [batch_local, DP, 1, rope_dim]
        #   position_dp:     [batch_local, DP, 1]
        current_ckv_dp = current_ckv.view(
            dp, batch_local, 1, kv_lora_rank
        ).permute(1, 0, 2, 3)
        current_k_pe_dp = current_k_pe.view(
            dp, batch_local, 1, query_rope.shape[-1]
        ).permute(1, 0, 2, 3)
        position_dp = position_ids.view(dp, batch_local, -1).permute(1, 0, 2)

    if selected_per_row % cores:
        raise ValueError(
            "DSA Top-K slots must divide num_cores per device."
        )
    tokens_per_core = selected_per_row // cores
    rows = dp * cp
    query_width = num_query_heads * q_len
    q_latent_rows = query_latent.view(
        dp, batch_local, num_query_heads, q_len, kv_lora_rank
    ).permute(1, 0, 2, 3, 4).float()
    q_latent_rows = q_latent_rows.unsqueeze(2).expand(
        batch_local,
        dp,
        cp,
        num_query_heads,
        q_len,
        kv_lora_rank,
    ).reshape(batch_local, rows, query_width, kv_lora_rank)
    rope_dim = query_rope.shape[-1]
    q_rope_rows = query_rope.view(
        dp, batch_local, num_query_heads, q_len, rope_dim
    ).permute(1, 0, 2, 3, 4).float()
    q_rope_rows = q_rope_rows.unsqueeze(2).expand(
        batch_local,
        dp,
        cp,
        num_query_heads,
        q_len,
        rope_dim,
    ).reshape(batch_local, rows, query_width, rope_dim)
    # Keep one local batch entry per attention graph.  This preserves the
    # same general batch scheduling used by the MiniMax GP implementation:
    # every loop iteration is independent in the batch axis.
    local_batch_size = 1
    output_groups: list[torch.Tensor] = []
    for batch_start in range(0, batch_local, local_batch_size):
        batch_end = min(batch_start + local_batch_size, batch_local)
        local = batch_end - batch_start

        # Gather only this local batch.  The helper expects external DP-major
        # input [local*DP, K] and returns folded rows:
        #   ckv_sparse_local: [local, DP, CP, K, kv_lora_rank]
        #   k_pe_sparse_local: [local, DP, CP, K, rope_dim]
        topk_local_dp = topk_dp_all[batch_start:batch_end]
        valid_local_dp = valid_dp_all[batch_start:batch_end]
        topk_local = topk_local_dp.permute(1, 0, 2).reshape(
            local * dp, selected_per_row
        )
        valid_local = valid_local_dp.permute(1, 0, 2).reshape(
            local * dp, selected_per_row
        )
        ckv_sparse_local, sparse_valid_local = gather_dsa_attention_cache_local(
            ckv_cache[batch_start:batch_end],
            topk_local,
            valid_local,
            dsa_config,
        )
        k_pe_sparse_local, _ = gather_dsa_attention_cache_local(
            k_pe_cache[batch_start:batch_end],
            topk_local,
            valid_local,
            dsa_config,
        )

        if current_ckv is not None and current_k_pe is not None and position_ids is not None:
            # The current token is not in the retained cache yet.  Overlay it
            # only in this local batch's selected CP rows:
            #   current_mask: [local, DP, 1, K, 1]
            current_mask = (
                topk_local_dp == position_dp[batch_start:batch_end]
            ).unsqueeze(2).unsqueeze(-1)
            current_ckv_local = current_ckv_dp[batch_start:batch_end].unsqueeze(2)
            current_k_pe_local = current_k_pe_dp[batch_start:batch_end].unsqueeze(2)
            ckv_sparse_local = torch.where(
                current_mask, current_ckv_local, ckv_sparse_local
            )
            k_pe_sparse_local = torch.where(
                current_mask, current_k_pe_local, k_pe_sparse_local
            )

        # Per-batch shapes before the core split:
        #   q_latent_local: [local, rows=DP*CP, query_width, kv_lora_rank]
        #   q_rope_local:   [local, rows=DP*CP, query_width, rope_dim]
        #   ckv_local:      [local, rows=DP*CP, topk, kv_lora_rank]
        #   k_pe_local:     [local, rows=DP*CP, topk, rope_dim]
        q_latent_local = q_latent_rows[batch_start:batch_end]
        q_rope_local = q_rope_rows[batch_start:batch_end]
        ckv_local = ckv_sparse_local
        k_pe_local = k_pe_sparse_local
        valid_local = sparse_valid_local

        ckv_core = ckv_local.reshape(
            local, rows, cores, tokens_per_core, kv_lora_rank
        ).float()
        k_pe_core = k_pe_local.reshape(
            local, rows, cores, tokens_per_core, rope_dim
        ).float()
        valid_core = valid_local.reshape(
            local, rows, cores, tokens_per_core
        )
        # Explicit core replication of the query:
        #   q_core:      [local, DP*CP, C, query_width, kv_lora_rank]
        #   q_rope_core: [local, DP*CP, C, query_width, rope_dim]
        q_core = q_latent_local.unsqueeze(2).expand(
            local, rows, cores, query_width, kv_lora_rank
        )
        q_rope_core = q_rope_local.unsqueeze(2).expand(
            local, rows, cores, query_width, rope_dim
        )

        # Absorption score:
        #   attn: [local, DP*CP, C, query_width, tokens_per_core]
        attn = torch.matmul(
            torch.cat((q_core, q_rope_core), dim=-1),
            torch.cat((ckv_core, k_pe_core), dim=-1).transpose(-1, -2),
        )
        attn = attn * scaling
        valid_mask = valid_core.unsqueeze(3)
        attn = attn.masked_fill(~valid_mask, -3.0e4)
        local_max = attn.max(dim=-1).values
        exp_attn = torch.exp(attn - local_max.unsqueeze(-1))
        exp_attn = torch.where(valid_mask, exp_attn, torch.zeros_like(exp_attn))
        local_sum = exp_attn.sum(dim=-1)
        local_out = torch.matmul(exp_attn, ckv_core)

        # Restore the physical hierarchy explicitly:
        #   local_max: [local, DP, CP, C, query_width]
        #   local_sum: [local, DP, CP, C, query_width]
        #   local_out: [local, DP, CP, C, query_width, kv_lora_rank]
        local_max = local_max.reshape(local, dp, cp, cores, query_width)
        local_sum = local_sum.reshape(local, dp, cp, cores, query_width)
        local_out = local_out.reshape(
            local, dp, cp, cores, query_width, kv_lora_rank
        )

        # Stage 1: reduce the physical cores inside each CP lane.
        #   core_max: [local, DP, CP, query_width]
        #   core_sum: [local, DP, CP, query_width]
        #   core_out: [local, DP, CP, query_width, kv_lora_rank]
        core_max = local_max.max(dim=3).values
        core_weights = torch.exp(local_max - core_max.unsqueeze(3))
        core_sum = (core_weights * local_sum).sum(dim=3)
        core_out = (core_weights.unsqueeze(-1) * local_out).sum(dim=3)

        if cp == 1:
            # No CP exchange/reduction graph when CP is one.  Remove the
            # singleton CP axis directly:
            #   [local, DP, 1, ...] -> [local, DP, ...]
            denominator = core_sum[:, :, 0]
            numerator = core_out[:, :, 0]
        else:
            # Stage 2: reduce the CP lanes after the core reduction.
            #   global_max: [local, DP, query_width]
            #   denominator: [local, DP, query_width]
            #   numerator: [local, DP, query_width, kv_lora_rank]
            global_max = core_max.max(dim=2).values
            cp_weights = torch.exp(core_max - global_max.unsqueeze(2))
            denominator = (cp_weights * core_sum).sum(dim=2)
            numerator = (cp_weights.unsqueeze(-1) * core_out).sum(dim=2)

        output_groups.append(numerator / denominator.unsqueeze(-1))

    # output_groups entries: [1, DP, query_width, kv_lora_rank]
    output = torch.cat(output_groups, dim=0)
    output = output.permute(1, 0, 2, 3).reshape(
        batch, num_query_heads, q_len, kv_lora_rank
    )
    output = output.to(query_latent.dtype)
    attn_output = torch.matmul(output, per_head_v_up)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


class QEffDeepseekV3Attention(nn.Module):
    """QEff-style GLM DSA decode implementation for the attention module."""
    def __qeff_init__(
        self,
    ):
        q_up, q_rope = self.q_b_proj.weight.T.view(
            -1, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim
        ).split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_up = q_up.reshape(-1, self.num_heads * self.qk_nope_head_dim).unsqueeze(0)
        self.q_up = torch.nn.Parameter(q_up.detach().clone())
        q_rope = q_rope.reshape(-1, self.num_heads * self.qk_rope_head_dim).unsqueeze(0)
        self.q_rope = torch.nn.Parameter(q_rope.detach().clone())
        k_up, v_up = self.kv_b_proj.weight.T.view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_up = k_up.reshape(-1, self.num_heads * self.qk_nope_head_dim).unsqueeze(0)
        v_up = v_up.reshape(-1, self.num_heads * self.v_head_dim).unsqueeze(0)
        self.k_up = torch.nn.Parameter(k_up.detach())
        self.v_up = torch.nn.Parameter(v_up.detach())
        per_head_q_up = self.q_up.squeeze(0).view(-1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1)
        per_head_k_up = (
            self.k_up.squeeze(0).view(-1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1).transpose(1, 2)
        )
        per_head_v_up = self.v_up.squeeze(0).view(-1, self.num_heads, self.v_head_dim).transpose(0, 1)
        self.per_head_v_up = torch.nn.Parameter(per_head_v_up.unsqueeze(0).detach().clone())
        self.per_head_q_up = torch.nn.Parameter(per_head_q_up.unsqueeze(0).detach().clone())
        self.per_head_k_up = torch.nn.Parameter(per_head_k_up.unsqueeze(0).detach().clone())
        # Build one ordinary Linear projection with all heads concatenated in
        # the output-channel dimension.  For H heads, R=q_lora_rank, and
        # K=kv_lora_rank:
        #   temporary per-head product: [H, R, K]
        #   fused linear weight:         [H*K, R]
        #   Linear input:                [B, Q, R]
        #   Linear output:               [B, Q, H*K]
        # The temporary per-head product exists only during initialization; it
        # is not retained as an ONNX initializer.
        fusedqk_per_head = torch.bmm(
            per_head_q_up, per_head_k_up
        ).reshape(
            self.num_heads,
            self.q_lora_rank,
            self.kv_lora_rank,
        )
        fusedqk_linear_weight = fusedqk_per_head.permute(0, 2, 1).reshape(
            self.num_heads * self.kv_lora_rank,
            self.q_lora_rank,
        )
        num_devices = int(
            getattr(getattr(self, "dsa_config", None), "num_devices", 1)
        )
        # When the device world is large, duplicate the complete output
        # channel set so the compiler has multiple independent channel groups
        # to partition.  The compiler handles up to eight devices per group,
        # so the normal TS16 setup uses ceil(16/8)=2 replicas.
        self.fusedqk_output_replicas = max(1, math.ceil(num_devices / 8))
        replicated_linear_weight = fusedqk_linear_weight.repeat(
            self.fusedqk_output_replicas,
            1,
        )
        self.fusedqk_linear = nn.Linear(
            self.q_lora_rank,
            self.fusedqk_output_replicas
            * self.num_heads
            * self.kv_lora_rank,
            bias=False,
        )
        self.fusedqk_linear.weight = nn.Parameter(
            replicated_linear_weight.detach().clone()
        )
        self.dsa_index_head_dim = self.config.index_head_dim
        self.dsa_index_n_heads = self.config.index_n_heads
        self.dsa_softmax_scale = self.dsa_index_head_dim ** (-0.5)
        self.dsa_wq_b = nn.Linear(self.q_lora_rank, self.dsa_index_n_heads * self.dsa_index_head_dim, bias=False)
        self.dsa_wk = nn.Linear(self.hidden_size, self.dsa_index_head_dim, bias=False)
        self.dsa_k_norm = nn.LayerNorm(self.dsa_index_head_dim, eps=1e-6)
        self.dsa_weights_proj = nn.Linear(self.hidden_size, self.dsa_index_n_heads, bias=False)
    def fused_forward_dsa_parallel(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        indexer_key_cache: Optional[torch.Tensor] = None,
        compressed_kvs: Optional[List[List[torch.Tensor]]] = None,
        mla_absorption: Optional[Dict[str, bool]] = None,
        topk_indices: Optional[torch.Tensor] = None,
        valid_topk: Optional[torch.Tensor] = None,
        query_latent: Optional[torch.Tensor] = None,
        query_rope: Optional[torch.Tensor] = None,
    ) -> tuple[Any, ...]:
        dsa_config = getattr(self, "dsa_config", DSAConfig(enabled=True))
        active_modes = sum(
            [
                dsa_config.indexer_only,
                dsa_config.attention_only,
                dsa_config.qkv_only,
            ]
        )
        if active_modes > 1:
            raise ValueError(
                "DSA indexer_only, attention_only, and qkv_only are mutually exclusive."
            )

        # Attention-only is a pure sparse-attention graph.  Q/K/V tensors,
        # retained caches, and Top-K indices are graph inputs; no QKV
        # projection, indexer, cache update, or final output projection runs.
        if dsa_config.attention_only:
            if any(
                value is None
                for value in (
                    topk_indices,
                    valid_topk,
                    compressed_kvs,
                    query_latent,
                    query_rope,
                )
            ):
                raise ValueError(
                    "Attention-only DSA requires QKV, cache, and Top-K inputs."
                )
            if len(compressed_kvs[0]) != 2:
                raise ValueError("DSA attention cache entry must contain cKV and k_PE.")
            ckv_cache, k_pe_cache = compressed_kvs[0]
            absorption = bool(
                mla_absorption and mla_absorption.get("absorption", False)
            )
            if not absorption:
                raise ValueError("DSA supports only MLA absorption mode.")
            is_prefill = query_latent.shape[2] > 1
            if is_prefill:
                # Prefill and decode have separate attention functions.  The
                # sequence length selects the dataflow exactly once here.
                attn_output = dsa_parallel_sparse_mla_attention_prefill(
                    query_latent=query_latent,
                    query_rope=query_rope,
                    per_head_v_up=self.per_head_v_up,
                    scaling=self.softmax_scale,
                    ckv_cache=ckv_cache,
                    k_pe_cache=k_pe_cache,
                    topk_indices=topk_indices,
                    valid_topk=valid_topk,
                    position_ids=position_ids,
                    num_cores=dsa_config.num_cores,
                    ql_chunk=dsa_config.sparse_q_block_size,
                    kv_num_blocks=dsa_config.sparse_kv_num_blocks,
                )
            else:
                attn_output, _ = dsa_parallel_sparse_mla_attention_decode(
                    module=self,
                    query_latent=query_latent,
                    query_rope=query_rope,
                    per_head_v_up=self.per_head_v_up,
                    absorption=absorption,
                    scaling=self.softmax_scale,
                    ckv_cache=ckv_cache,
                    k_pe_cache=k_pe_cache,
                    topk_indices=topk_indices,
                    valid_topk=valid_topk,
                    dsa_config=dsa_config,
                    position_ids=position_ids,
                )
            return attn_output, None, None

        if hidden_states is None or position_ids is None:
            raise ValueError("QKV/indexer DSA modes require hidden_states and position_ids.")
        bsz, q_len, _ = hidden_states.size()
        is_prefill = q_len > 1
        if not dsa_config.qkv_only and indexer_key_cache is None:
            raise ValueError("DSA requires indexer_key_cache input.")

        # Keep the Q projection working set bounded for prefill.  The large
        # [B, QL, hidden] input otherwise makes the compiler materialize a
        # large q_a_proj/dequant intermediate in DDR and can prevent the
        # projection from being distributed across cores.  Split only along
        # the sequence dimension; the hidden dimension and projection weights
        # are unchanged.
        q_a_proj_seq_chunk = 256
        if q_len > q_a_proj_seq_chunk:
            q_a_proj_chunks = []
            # Use the minimum number of chunks needed to keep every chunk at
            # most 256 tokens, then distribute QL evenly across those chunks.
            # For example, QL=376 becomes [188, 188], not [256, 120].
            num_q_chunks = math.ceil(q_len / q_a_proj_seq_chunk)
            base_q_chunk = q_len // num_q_chunks
            extra_q_tokens = q_len % num_q_chunks
            q_start = 0
            for q_chunk_idx in range(num_q_chunks):
                q_chunk_len = base_q_chunk + (q_chunk_idx < extra_q_tokens)
                q_end = q_start + q_chunk_len
                q_a_proj_chunks.append(
                    self.q_a_layernorm(
                        self.q_a_proj(hidden_states[:, q_start:q_end, :])
                    )
                )
                q_start = q_end
            q_a_proj_out = torch.cat(q_a_proj_chunks, dim=1)
        else:
            q_a_proj_out = self.q_a_layernorm(self.q_a_proj(hidden_states))
        cos, sin = self.rotary_emb(
            hidden_states, seq_len=self.max_position_embeddings
        )
        if not dsa_config.qkv_only:
            indexer_state = prepare_dsa_indexer_state(
                self, hidden_states, q_a_proj_out, position_ids, cos, sin
            )
            if not is_prefill:
                topk_indices, valid_topk, indexer_key_cache = build_dsa_parallel_topk_indices(
                    module=self,
                    hidden_states=hidden_states,
                    q_a_proj_out=q_a_proj_out,
                    position_ids=position_ids,
                    indexer_key_cache=indexer_key_cache,
                    dsa_config=dsa_config,
                    indexer_state=indexer_state,
                )
            else:
                if not dsa_config.indexer_prefill_parallel:
                    raise ValueError("Only the parallel prefill indexer is ported (--indexer-prefill-parallel).")
                prefill_indexer_builder = build_dsa_prefill_topk_indices_par
                topk_indices, valid_topk, indexer_key_cache = (
                    prefill_indexer_builder(
                        module=self,
                        hidden_states=hidden_states,
                        q_a_proj_out=q_a_proj_out,
                        position_ids=position_ids,
                        indexer_key_cache=indexer_key_cache,
                        dsa_config=dsa_config,
                        indexer_state=indexer_state,
                        indexer_ql_chunk=dsa_config.indexer_ql_chunk,
                        indexer_q_block_size=dsa_config.indexer_q_block_size,
                    )
                )
            if dsa_config.indexer_only:
                return hidden_states, indexer_key_cache, topk_indices, valid_topk

        ckv_cache = None
        k_pe_cache = None
        if not dsa_config.qkv_only:
            if compressed_kvs is None:
                raise ValueError("DSA requires compressed MLA retained state.")
            cache_entry = compressed_kvs[0]
            if len(cache_entry) != 2:
                raise ValueError("DSA attention cache entry must contain cKV and k_PE.")
            ckv_cache, k_pe_cache = cache_entry
            expected_batch_local = bsz // dsa_config.attention_dp
            expected_rows = dsa_config.attention_dp * dsa_config.attention_cp
            expected_cache_shape = (expected_batch_local, expected_rows)
            if ckv_cache.ndim != 4 or ckv_cache.shape[:2] != expected_cache_shape:
                raise ValueError(
                    "DSA cKV cache must use folded [B/DP,DP*CP,T/CP,D] layout."
                )
            if k_pe_cache.ndim != 4 or k_pe_cache.shape[:3] != ckv_cache.shape[:3]:
                raise ValueError("DSA cKV and k_PE caches must use the same layout.")

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = compressed_kv.view(
            bsz, q_len, 1, self.kv_lora_rank + self.qk_rope_head_dim
        ).transpose(1, 2)
        kva = compressed_kv[:, :, :, : self.kv_lora_rank]
        k_pe = compressed_kv[:, :, :, self.kv_lora_rank :]
        q_pe = torch.matmul(q_a_proj_out, self.q_rope)
        q_pe = q_pe.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)
        kva = self.kv_a_layernorm(kva)
        # GLM-MoE-DSA uses the split-half HF RoPE convention for its DSA path.
        q_pe = apply_single_rotary_pos_emb(q_pe, cos, sin, position_ids)
        k_pe = apply_single_rotary_pos_emb(k_pe, cos, sin, position_ids)

        absorption = bool(mla_absorption and mla_absorption.get("absorption", False))
        online = bool(mla_absorption and mla_absorption.get("online", False))
        if not absorption:
            raise ValueError("DSA supports only MLA absorption mode.")
        if online:
            qup_kupT = torch.matmul(self.per_head_q_up, self.per_head_k_up)
            dq_qup_kupT = torch.matmul(q_a_proj_out, qup_kupT)
        else:
            # Use one dense Linear projection with no explicit head/device
            # axis in the graph:
            #   q_a_proj_out: [B, Q, R]
            #   Linear output: [B, Q, replicas*H*K]
            #   replica view:  [B, Q, replicas, H, K]
            #   reduced Q:     [B, H, Q, K]
            #   query_latent: [B, H, Q, K]
            fusedqk_output = self.fusedqk_linear(q_a_proj_out)
            fusedqk_output = fusedqk_output.reshape(
                bsz,
                q_len,
                self.fusedqk_output_replicas,
                self.num_heads,
                self.kv_lora_rank,
            )
            # Keep every replica live in the graph.  Since all replicas use
            # identical weights, their mean is numerically equal to one copy.
            fusedqk_output = fusedqk_output.mean(dim=2)
            dq_qup_kupT = fusedqk_output.permute(0, 2, 1, 3)

        if dsa_config.qkv_only:
            # Return exactly the tensors consumed by the pure sparse-attention
            # graph: latent Q, rotary Q, current compressed KV, and rotary K.
            return (
                dq_qup_kupT,
                ((q_pe, kva.squeeze(1), k_pe),),
                None,
            )

        # The sequence length is the only sparse-attention route selector:
        #   QL == 1 -> decode (overlay current KV, attend, then commit it)
        #   QL >  1 -> prefill (commit the whole Q sequence, then attend)
        if is_prefill:
            ckv_cache = scatter_dsa_attention_cache(
                ckv_cache,
                position_ids,
                kva.squeeze(1),
                dsa_config,
            )
            k_pe_cache = scatter_dsa_attention_cache(
                k_pe_cache,
                position_ids,
                k_pe.squeeze(1),
                dsa_config,
            )
            attention_current_ckv = None
            attention_current_k_pe = None
            attention_position_ids = position_ids
        else:
            attention_current_ckv = kva.squeeze(1)
            attention_current_k_pe = k_pe.squeeze(1)
            attention_position_ids = position_ids

        if is_prefill:
            attn_output = dsa_parallel_sparse_mla_attention_prefill(
                query_latent=dq_qup_kupT,
                query_rope=q_pe,
                per_head_v_up=self.per_head_v_up,
                scaling=self.softmax_scale,
                ckv_cache=ckv_cache,
                k_pe_cache=k_pe_cache,
                topk_indices=topk_indices,
                valid_topk=valid_topk,
                position_ids=position_ids,
                num_cores=dsa_config.num_cores,
                ql_chunk=dsa_config.sparse_q_block_size,
                kv_num_blocks=dsa_config.sparse_kv_num_blocks,
            )
        else:
            attn_output, _ = dsa_parallel_sparse_mla_attention_decode(
                module=self,
                query_latent=dq_qup_kupT,
                query_rope=q_pe,
                per_head_v_up=self.per_head_v_up,
                absorption=absorption,
                scaling=self.softmax_scale,
                ckv_cache=ckv_cache,
                k_pe_cache=k_pe_cache,
                topk_indices=topk_indices,
                valid_topk=valid_topk,
                dsa_config=dsa_config,
                current_ckv=attention_current_ckv,
                current_k_pe=attention_current_k_pe,
                position_ids=attention_position_ids,
            )
        if not is_prefill:
            ckv_cache = scatter_dsa_attention_cache(
                ckv_cache, position_ids, kva.squeeze(1), dsa_config
            )
            k_pe_cache = scatter_dsa_attention_cache(
                k_pe_cache, position_ids, k_pe.squeeze(1), dsa_config
            )
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        present = ((ckv_cache, k_pe_cache),)
        return (
            attn_output,
            present,
            None if dsa_config.attention_only else indexer_key_cache,
        )
    def fused_forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        indexer_key_cache: Optional[torch.Tensor] = None,
        compressed_kvs: Optional[List[List[torch.Tensor]]] = None,
        topk_indices: Optional[torch.Tensor] = None,
        valid_topk: Optional[torch.Tensor] = None,
        query_latent: Optional[torch.Tensor] = None,
        query_rope: Optional[torch.Tensor] = None,
    ) -> tuple[Any, ...]:
        mla_absorption = getattr(self, "mla_absorption", None)
        return self.fused_forward_dsa_parallel(
            hidden_states=hidden_states,
            position_ids=position_ids,
            indexer_key_cache=indexer_key_cache,
            compressed_kvs=compressed_kvs,
            mla_absorption=mla_absorption,
            topk_indices=topk_indices,
            valid_topk=valid_topk,
            query_latent=query_latent,
            query_rope=query_rope,
        )


def get_attention_config(profile: str) -> DeepseekV3Config:
    if profile == "glm5":
        return DeepseekV3Config(
            attention_dropout=0.0,
            hidden_size=6144,
            num_attention_heads=64,
            max_position_embeddings=202752,
            rope_theta=1000000.0,
            q_lora_rank=2048,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=256,
            qk_nope_head_dim=192,
            attention_bias=False,
            rope_scaling=None,
            model_type="glm5",
            index_topk=2048,
            index_head_dim=128,
            index_n_heads=32,
        )
    if profile == "tiny":
        return DeepseekV3Config(
            attention_dropout=0.0,
            hidden_size=32,
            num_attention_heads=4,
            max_position_embeddings=32,
            rope_theta=10000.0,
            q_lora_rank=16,
            qk_rope_head_dim=4,
            kv_lora_rank=8,
            v_head_dim=8,
            qk_nope_head_dim=8,
            index_topk=4,
            index_head_dim=8,
            index_n_heads=2,
            model_type="glm5_parity",
        )
    raise ValueError(f"Unknown model profile: {profile}")


def build_module(config: DeepseekV3Config, dsa_config: DSAConfig, dtype=torch.float32) -> DeepseekV3Attention:
    """DeepseekV3Attention with the QEff DSA methods and absorbed-MLA weights installed."""
    module = DeepseekV3Attention(config=config, layer_idx=0)
    module.dsa_config = dsa_config
    for name in ("fused_forward", "fused_forward_dsa_parallel", "__qeff_init__"):
        setattr(module, name, getattr(QEffDeepseekV3Attention, name).__get__(module, module.__class__))
    module.forward = module.fused_forward
    module.__qeff_init__()
    module = module.to(dtype=dtype).eval()
    module.mla_absorption = {"absorption": True, "online": False, "cache_compressed": True}
    return module
