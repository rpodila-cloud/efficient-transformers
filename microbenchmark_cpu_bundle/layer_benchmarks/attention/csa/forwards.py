"""DeepSeek-V4 compressed sparse attention forward implementation.

CSA uses a sliding window, overlapping (2*ratio) compressor, and lightning
indexer top-k over a CP-tiled indexer cache.

The math is the ``DeepseekV4FoldedCSAParallelDecodeAttention`` path of
``deepseek_v4_csa_hca_decode_microbench.py`` verbatim (only the branches that benchmark uses), so
the exported graph and its QAIC perf are unchanged. Parameter names are kept
so the original module's state dict loads directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .cache import DeepseekV4FoldedAttentionCache

MASKED_ATTENTION_LOGIT = -3.0e4


@dataclass
class Config:
    profile: str
    attn_mode: str
    hidden_size: int
    num_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    window_size: int
    compress_ratio: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    ctx_len: int
    norm_eps: float = 1.0e-6
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    yarn_factor: float = 16.0
    yarn_original_max_position_embeddings: int = 65536
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0

    @property
    def comp_slots(self) -> int:
        return max(1, (self.ctx_len + self.compress_ratio - 1) // self.compress_ratio)

    @property
    def overlap(self) -> bool:
        return self.attn_mode == "csa"

    @property
    def compressor_width(self) -> int:
        return self.head_dim * (2 if self.overlap else 1)

    @property
    def compressor_state_len(self) -> int:
        return self.compress_ratio * (2 if self.overlap else 1)

    @property
    def indexer_state_len(self) -> int:
        return self.compress_ratio * 2


PROFILES = {
    "smoke": dict(hidden_size=256, num_heads=4, head_dim=64, rope_head_dim=16, q_lora_rank=64, o_groups=2,
                  o_lora_rank=32, window_size=8, index_n_heads=4, index_head_dim=32, index_topk=4),
    "tiny": dict(hidden_size=256, num_heads=4, head_dim=512, rope_head_dim=64, q_lora_rank=128, o_groups=2,
                 o_lora_rank=128, window_size=128, index_n_heads=32, index_head_dim=128, index_topk=1024),
    "v4_flash": dict(hidden_size=4096, num_heads=64, head_dim=512, rope_head_dim=64, q_lora_rank=1024, o_groups=8,
                     o_lora_rank=1024, window_size=128, index_n_heads=64, index_head_dim=128, index_topk=512),
    "v4_pro": dict(hidden_size=7168, num_heads=128, head_dim=512, rope_head_dim=64, q_lora_rank=1536, o_groups=16,
                   o_lora_rank=1024, window_size=128, index_n_heads=64, index_head_dim=128, index_topk=512),
}
DEFAULT_RATIO = {"csa": 4, "hca": 128}


def yarn_inv_freq(c: Config) -> torch.Tensor:
    dim, base = c.rope_head_dim, c.compress_rope_theta

    def correction_dim(rotations: float) -> float:
        return dim * math.log(c.yarn_original_max_position_embeddings / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction_dim(c.yarn_beta_fast)), 0)
    high = min(math.ceil(correction_dim(c.yarn_beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    extrapolation = 1 - ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    base_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    return base_inv_freq / c.yarn_factor * (1 - extrapolation) + base_inv_freq * extrapolation


def _div(position: torch.Tensor, ratio: int) -> torch.Tensor:
    """position // ratio without an integer Div: dynamo lowers int floor/trunc division
    with a Sign correction that AIC rejects for int32. Exact for power-of-two ratios."""
    if ratio & (ratio - 1):
        raise ValueError(f"compress_ratio must be a power of two, got {ratio}")
    return torch.floor(position.float() * (1.0 / ratio)).to(position.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype, x = x.dtype, x.float()
        return (self.weight * x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)).to(dtype)


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)).to(x.dtype)


class CompressedAttention(DeepseekV4FoldedAttentionCache, nn.Module):
    """forward(hidden [B,1,H], position_ids [B,1], local, compressed, comp_kv, comp_score[, idx_kv, idx_kv_state, idx_score_state])."""

    def __init__(self, c: Config, *, attn_dp: int, indexer_cp: int, num_kv_blocks: int = 2, num_attention_cores: int = 16,
                 hca_compressed_kv_cp: int = 1, hca_attn_blocks: int | None = None):
        super().__init__()
        if c.attn_mode not in ("csa", "hca"):
            raise ValueError("attn_mode must be csa or hca")
        self.config = c
        self.attn_dp, self.indexer_cp = attn_dp, indexer_cp
        self.num_kv_blocks, self.num_attention_cores = num_kv_blocks, num_attention_cores
        # HCA only: CP ways folded into the compressed-KV slot axis, and core tiles per way.
        self.hca_compressed_kv_cp = hca_compressed_kv_cp
        self.hca_attn_blocks = hca_attn_blocks if hca_attn_blocks is not None else num_attention_cores
        self.q_a_proj = nn.Linear(c.hidden_size, c.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(c.q_lora_rank, c.norm_eps)
        self.q_b_proj = nn.Linear(c.q_lora_rank, c.num_heads * c.head_dim, bias=False)
        self.q_b_norm = UnweightedRMSNorm(c.norm_eps)
        self.kv_proj = nn.Linear(c.hidden_size, c.head_dim, bias=False)
        self.kv_norm = RMSNorm(c.head_dim, c.norm_eps)
        self.sinks = nn.Parameter(torch.zeros(c.num_heads))
        self.o_a_weight = nn.Parameter(torch.empty(c.o_groups, c.o_lora_rank, c.num_heads * c.head_dim // c.o_groups))
        self.o_b_proj = nn.Linear(c.o_groups * c.o_lora_rank, c.hidden_size, bias=False)
        nn.init.normal_(self.o_a_weight, std=0.02)
        self.comp_kv_proj = nn.Linear(c.hidden_size, c.compressor_width, bias=False)
        self.comp_gate_proj = nn.Linear(c.hidden_size, c.compressor_width, bias=False)
        self.comp_norm = RMSNorm(c.head_dim, c.norm_eps)
        self.comp_position_bias = nn.Parameter(torch.zeros(c.compress_ratio, c.compressor_width))
        if c.attn_mode == "csa":
            width = 2 * c.index_head_dim
            self.index_kv_proj = nn.Linear(c.hidden_size, width, bias=False)
            self.index_gate_proj = nn.Linear(c.hidden_size, width, bias=False)
            self.index_norm = RMSNorm(c.index_head_dim, c.norm_eps)
            self.index_position_bias = nn.Parameter(torch.zeros(c.compress_ratio, width))
            self.index_q_b_proj = nn.Linear(c.q_lora_rank, c.index_n_heads * c.index_head_dim, bias=False)
            self.index_weights_proj = nn.Linear(c.hidden_size, c.index_n_heads, bias=False)
        main = 1.0 / (c.rope_theta ** (torch.arange(0, c.rope_head_dim, 2).float() / c.rope_head_dim))
        self.register_buffer("main_inv_freq", main, persistent=False)
        self.register_buffer("compress_inv_freq", yarn_inv_freq(c), persistent=False)

    # ── RoPE ─────────────────────────────────────────────────────────────────
    def _rope(self, x: torch.Tensor, pos_dp: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        """Interleaved RoPE on the trailing rope_head_dim channels; x is [l, p, (h,) s, d], pos_dp [l, p, s].

        Phases stay fp32: fp16 position*freq overflows at long context.
        """
        rope_dim = self.config.rope_head_dim
        freqs = pos_dp.float().unsqueeze(-1) * self.compress_inv_freq.float().view(1, 1, -1)
        cos, sin = freqs.cos(), freqs.sin()
        if inverse:
            sin = -sin
        if x.ndim == 5:
            cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        lead = int(x.shape[-1]) - rope_dim
        nope, rope = x[..., :lead], x[..., lead:]
        even, odd = rope[..., 0::2].float(), rope[..., 1::2].float()
        rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2).to(x.dtype)
        return rotated if lead == 0 else torch.cat((nope, rotated), dim=-1)

    # ── compressors ──────────────────────────────────────────────────────────
    def _pool(self, kv_dp, score_dp, width, norm, pos_dp, dtype):
        """Softmax-pool a [l, p, 2r, 2w] overlap state (or [l, p, r, w] when width is None)."""
        r = self.config.compress_ratio
        if width is not None:
            kv_dp = torch.cat([kv_dp[:, :, :r, :width], kv_dp[:, :, r:, width:]], dim=2)
            score_dp = torch.cat([score_dp[:, :, :r, :width], score_dp[:, :, r:, width:]], dim=2)
        pooled = (kv_dp * score_dp.softmax(dim=2).to(kv_dp.dtype)).sum(dim=2, keepdim=True)
        return self._rope(norm(pooled.to(dtype)), pos_dp + 1 - r)

    def _rotate(self, kv_dp, score_dp, should_compress):
        """At a window boundary shift the current half into the previous half."""
        r = self.config.compress_ratio
        shifted_kv = torch.cat([kv_dp[:, :, r:].clone(), kv_dp[:, :, r:].clone()], dim=2)
        shifted_score = torch.cat([score_dp[:, :, r:].clone(), score_dp[:, :, r:].clone()], dim=2)
        return torch.where(should_compress, shifted_kv, kv_dp), torch.where(should_compress, shifted_score, score_dp)

    def _compress(self, hidden, position_ids, pos_dp, compressed_cache, kv_state, score_state, kv, score):
        """Write this token into the compressor state, pool, and write the provisional compressed row.

        The row at pos // ratio is rewritten every step; validity masks keep it
        out of attention until the boundary, and the (CSA) state rotation is
        boundary-gated, so one graph serves every position.
        """
        c = self.config
        batch, dp = hidden.shape[0], self.attn_dp
        slot = torch.remainder(pos_dp, c.compress_ratio).to(torch.long).reshape(batch, -1)
        write = c.compress_ratio + slot if c.overlap else slot
        kv_state = self.scatter_folded(kv_state, write, kv.to(kv_state.dtype))
        score_state = self.scatter_folded(score_state, write, score.to(score_state.dtype))
        kv_dp = kv_state.view(batch // dp, dp, kv_state.shape[2], kv_state.shape[3])
        score_dp = score_state.view(batch // dp, dp, score_state.shape[2], score_state.shape[3])
        pooled = self._pool(kv_dp, score_dp, c.head_dim if c.overlap else None, self.comp_norm, pos_dp, hidden.dtype)
        slot = _div(position_ids, c.compress_ratio).to(torch.int32)
        if not c.overlap and self.hca_compressed_kv_cp > 1:
            # HCA CP: global slot g lives at (g % cp) * slots_per_cp + g // cp.
            cp = self.hca_compressed_kv_cp
            slot = torch.remainder(slot, cp) * (compressed_cache.shape[2] // cp) + _div(slot, cp)
        compressed_cache = self.scatter_dp(compressed_cache, slot, pooled.reshape(batch, 1, -1), dp)
        if not c.overlap:
            return compressed_cache, kv_state, score_state
        should_compress = (torch.remainder(position_ids + 1, c.compress_ratio) == 0).view(batch, 1, 1)
        kv_dp, score_dp = self._rotate(kv_dp, score_dp, should_compress)
        return compressed_cache, kv_dp.reshape_as(kv_state), score_dp.reshape_as(score_state)

    # ── lightning indexer (CSA) ──────────────────────────────────────────────
    def _topk_blockwise(self, q, weights, score_cache, pos_dp):
        """Per-(CP way, KV block) scores + top-k, merged per way, then globally. Returns global slot ids."""
        c = self.config
        batch_local, dp, cp, local_slots, _ = score_cache.shape
        if local_slots % self.num_kv_blocks:
            raise ValueError("CP-local indexer slots must divide num_kv_blocks.")
        block_width = local_slots // self.num_kv_blocks
        q_sp = q.permute(0, 1, 3, 2, 4).float()
        weights = weights.float()
        cp_ids = torch.arange(cp, dtype=torch.int64).view(1, 1, cp, 1)
        threshold = _div(pos_dp[:, :, -1] + 1, c.compress_ratio).view(batch_local, dp, 1, 1)
        block_scores, block_ids = [], []
        for block in range(self.num_kv_blocks):
            start = block * block_width
            keys = score_cache[:, :, :, start : start + block_width, :].float()
            head_scores = torch.matmul(q_sp.unsqueeze(2), keys.unsqueeze(3).transpose(-1, -2))
            scores = (torch.relu(head_scores) * weights.unsqueeze(2).unsqueeze(-1)).sum(dim=-2)[:, :, :, 0, :]
            local_ids = torch.arange(start, start + block_width, dtype=torch.int64).view(1, 1, 1, block_width)
            scores = scores.masked_fill(local_ids * cp + cp_ids >= threshold, MASKED_ATTENTION_LOGIT)
            selected = torch.topk(scores, k=min(c.index_topk, block_width), dim=-1)
            block_scores.append(selected.values)
            block_ids.append(torch.gather(local_ids.expand(batch_local, dp, cp, block_width), -1, selected.indices))
        candidate_scores = torch.cat(block_scores, dim=-1)
        local_k = min(c.index_topk, self.num_kv_blocks * block_scores[0].shape[-1])
        local_scores, local_idx = torch.topk(candidate_scores, k=local_k, dim=-1)
        local_ids = torch.gather(torch.cat(block_ids, dim=-1), -1, local_idx)
        candidate_global = (local_ids * cp + cp_ids.expand(batch_local, dp, cp, local_k)).reshape(batch_local, dp, -1)
        _, selected = torch.topk(local_scores.reshape(batch_local, dp, -1), k=min(c.index_topk, cp * local_k), dim=-1)
        return torch.gather(candidate_global, 2, selected).to(torch.int32)

    def _indexer(self, hidden, position_ids, pos_dp, index_cache, kv_state, score_state, kv, score, q, weights):
        c = self.config
        batch, dp, cp, r = hidden.shape[0], self.attn_dp, self.indexer_cp, c.compress_ratio
        batch_local = batch // dp
        write = r + torch.remainder(pos_dp, r).to(torch.long).reshape(batch, -1)
        kv_state = self.scatter_1d(kv_state, write, kv.to(kv_state.dtype))
        score_state = self.scatter_1d(score_state, write, score.to(score_state.dtype))
        kv_dp = kv_state.view(batch_local, dp, kv_state.shape[1], kv_state.shape[2])
        score_dp = score_state.view(batch_local, dp, score_state.shape[1], score_state.shape[2])
        pooled = self._pool(kv_dp, score_dp, c.index_head_dim, self.index_norm, pos_dp, hidden.dtype)
        index_cache = self.scatter_indexer(
            index_cache, _div(position_ids, r).to(torch.int32), pooled.reshape(batch, 1, -1), dp, cp
        )
        should_compress = (torch.remainder(position_ids + 1, r) == 0).view(batch_local, dp, 1, 1)
        kv_dp, score_dp = self._rotate(kv_dp, score_dp, should_compress)

        score_cache = index_cache.view(batch_local, dp, cp, index_cache.shape[2] // cp, index_cache.shape[3])
        if c.index_topk >= c.comp_slots:
            topk = torch.arange(c.comp_slots).view(1, 1, -1).expand(batch_local, dp, c.comp_slots)
        else:
            topk = self._topk_blockwise(q, weights, score_cache, pos_dp)
        valid = topk < _div(pos_dp[:, :, -1] + 1, r).view(batch_local, dp, 1)
        return (
            topk.reshape(batch, -1).to(torch.int32),
            valid.reshape(batch, -1),
            index_cache,
            kv_dp.reshape_as(kv_state),
            score_dp.reshape_as(score_state),
        )

    # ── attention ────────────────────────────────────────────────────────────
    def _attend_blocked(self, q_dp, kv, valid, parts: int):
        """Unnormalised (max, sum, out) online-softmax partials over ``parts`` tiles; heads fold into matmul rows."""
        batch_local, dp, heads, seq_len, head_dim = q_dp.shape
        if kv.shape[2] % parts:
            raise ValueError("Attention length must divide the tile count.")
        tile = kv.shape[2] // parts
        kv_tiled = kv.view(batch_local, dp, parts, tile, head_dim)
        valid_tiled = valid.view(batch_local, dp, parts, tile)
        q_folded = q_dp.reshape(batch_local, dp, 1, heads * seq_len, head_dim)
        scores = torch.matmul(q_folded, kv_tiled.float().transpose(-1, -2)) * (head_dim**-0.5)
        scores = scores.masked_fill(~valid_tiled[:, :, :, None, :], MASKED_ATTENTION_LOGIT)
        max_scores = scores.max(dim=-1).values
        # Float reduce: dynamo lowers bool any() to an int32 ReduceMax that costs ~8% of HCA on AIC.
        all_masked = (valid_tiled.to(scores.dtype).amax(dim=-1) == 0)[:, :, :, None]
        max_scores = torch.where(all_masked, torch.full_like(max_scores, MASKED_ATTENTION_LOGIT), max_scores)
        exp_scores = torch.exp(scores - max_scores.unsqueeze(-1))
        exp_scores = torch.where(valid_tiled[:, :, :, None, :], exp_scores, torch.zeros_like(exp_scores))
        sums = exp_scores.sum(dim=-1)
        out = torch.matmul(exp_scores.to(kv_tiled.dtype), kv_tiled)
        sums = torch.where(all_masked, torch.zeros_like(sums), sums)
        out = torch.where(all_masked.unsqueeze(-1), torch.zeros_like(out), out).float()
        max_scores = max_scores.view(batch_local, dp, parts, heads, seq_len).permute(0, 1, 3, 2, 4)
        sums = sums.view(batch_local, dp, parts, heads, seq_len).permute(0, 1, 3, 2, 4)
        out = out.view(batch_local, dp, parts, heads, seq_len, head_dim).permute(0, 1, 3, 2, 4, 5)
        return max_scores, sums, out

    def _attend(self, q_dp, local_kv, local_valid, comp_kv, comp_valid, pos_dp):
        """Local + compressed partials merged with the per-head sink; HCA's compressed read is core-tiled."""
        c = self.config
        batch_local, dp, _, seq_len, _ = q_dp.shape
        q = q_dp.float()
        sources = [
            self._attend_blocked(q, local_kv, local_valid, 1),
            self._attend_blocked(
                q, comp_kv, comp_valid, self.hca_compressed_kv_cp * self.hca_attn_blocks if c.attn_mode == "hca" else 1
            ),
        ]
        part_max = torch.cat([m for m, _, _ in sources], dim=3)
        part_sum = torch.cat([s for _, s, _ in sources], dim=3)
        part_out = torch.cat([o for _, _, o in sources], dim=3)
        sinks = self.sinks.view(1, 1, c.num_heads, 1).expand(batch_local, dp, -1, seq_len).float()
        global_max = torch.maximum(part_max.max(dim=3).values, sinks)
        part_weight = torch.exp(part_max - global_max.unsqueeze(3))
        denom = (part_weight * part_sum).sum(dim=3) + torch.exp(sinks - global_max)
        merged = (part_weight.unsqueeze(-1) * part_out).sum(dim=3) / denom.unsqueeze(-1)
        return self._rope(merged.to(q_dp.dtype), pos_dp, inverse=True)

    def _project_output(self, attn: torch.Tensor) -> torch.Tensor:
        c = self.config
        batch_local, dp, _, seq_len, _ = attn.shape
        out = attn.transpose(2, 3).reshape(batch_local, dp, seq_len, c.o_groups, -1)
        out = torch.einsum("lpsgd,grd->lpsgr", out, self.o_a_weight.to(out.dtype)).flatten(3)
        return self.o_b_proj(out).reshape(batch_local * dp, seq_len, c.hidden_size)

    def forward(self, hidden_states, position_ids, local_kv_cache, compressed_kv_cache, compressor_kv_state,
                compressor_score_state, *indexer_state):
        c = self.config
        batch, seq_len, _ = hidden_states.shape
        if seq_len != 1:
            raise ValueError("decode-only: seq_len must be 1")
        dp = self.attn_dp
        batch_local = batch // dp
        pos_dp = position_ids.view(batch_local, dp, seq_len)
        hidden_dp = hidden_states.view(batch_local, dp, seq_len, c.hidden_size)

        q_residual_dp = self.q_a_norm(self.q_a_proj(hidden_dp))
        q_dp = self.q_b_proj(q_residual_dp).view(batch_local, dp, seq_len, c.num_heads, c.head_dim).permute(0, 1, 3, 2, 4)
        q_dp = self._rope(self.q_b_norm(q_dp), pos_dp)

        kv = self._rope(self.kv_norm(self.kv_proj(hidden_dp)), pos_dp).reshape(batch, seq_len, c.head_dim)
        local_kv_cache = self.scatter_folded(local_kv_cache, (position_ids % c.window_size).to(torch.int32), kv)
        local_kv, local_valid = self.gather_folded(local_kv_cache, *self._sliding_indices(position_ids), dp)

        slot_dp = torch.remainder(pos_dp, c.compress_ratio).to(torch.long)
        comp_kv = self.comp_kv_proj(hidden_dp).float().reshape(batch, seq_len, -1)
        comp_score = self.comp_gate_proj(hidden_dp).float() + nn.functional.embedding(slot_dp, self.comp_position_bias)
        compressed_kv_cache, compressor_kv_state, compressor_score_state = self._compress(
            hidden_states, position_ids, pos_dp, compressed_kv_cache, compressor_kv_state, compressor_score_state,
            comp_kv, comp_score.reshape(batch, seq_len, -1),
        )

        if c.attn_mode == "csa":
            index_kv = self.index_kv_proj(hidden_dp).float().reshape(batch, seq_len, -1)
            index_score = self.index_gate_proj(hidden_dp).float() + nn.functional.embedding(slot_dp, self.index_position_bias)
            index_q = self.index_q_b_proj(q_residual_dp).view(batch_local, dp, seq_len, c.index_n_heads, c.index_head_dim)
            index_q = self._rope(index_q.permute(0, 1, 3, 2, 4), pos_dp)
            index_weights = self.index_weights_proj(hidden_dp).float() * (c.index_head_dim**-0.5) * (c.index_n_heads**-0.5)
            topk, topk_valid, *indexer_state = self._indexer(
                hidden_states, position_ids, pos_dp, *indexer_state,
                index_kv, index_score.reshape(batch, seq_len, -1), index_q, index_weights,
            )
            comp_kv_rows, comp_valid = self.gather_dp(compressed_kv_cache, topk, topk_valid, dp)
        else:
            # No indexer: attend every ready compressed row in place.
            ready = _div(position_ids[:, -1] + 1, c.compress_ratio).view(batch_local, dp, 1)
            comp_kv_rows = compressed_kv_cache
            slot_ids = torch.arange(compressed_kv_cache.shape[2])
            cp = self.hca_compressed_kv_cp
            if cp > 1:  # storage address -> global slot id (inverse of the CP scatter)
                per_cp = compressed_kv_cache.shape[2] // cp
                slot_ids = torch.remainder(slot_ids, per_cp) * cp + _div(slot_ids, per_cp)
            comp_valid = slot_ids.view(1, 1, -1) < ready

        attn = self._attend(q_dp, local_kv, local_valid, comp_kv_rows, comp_valid, pos_dp)
        dtype = hidden_states.dtype
        return (
            self._project_output(attn),
            local_kv_cache,
            compressed_kv_cache,
            compressor_kv_state.to(dtype),
            compressor_score_state.to(dtype),
            *([indexer_state[0], indexer_state[1].to(dtype), indexer_state[2].to(dtype)] if indexer_state else []),
        )

    def _sliding_indices(self, position_ids: torch.Tensor):
        """Ring-buffer order (oldest first) once the window is full; slot order before."""
        window = self.config.window_size
        pos = position_ids[:, -1:].to(torch.long)
        offsets = torch.arange(window).view(1, -1)
        ring = (pos % window + 1 + offsets) % window
        indices = torch.where(pos >= window - 1, ring, offsets.expand(pos.shape[0], -1))
        return indices.to(torch.int32), offsets < (pos + 1)


class CSA(CompressedAttention):
    """CSA model with the lightning indexer and top-k compressed attention."""

    def __init__(self, config: Config, **kwargs):
        if config.attn_mode != "csa":
            raise ValueError(f"CSA requires config.attn_mode='csa', got {config.attn_mode!r}")
        super().__init__(config, **kwargs)


__all__ = ["CSA", "CompressedAttention", "Config", "DEFAULT_RATIO", "PROFILES", "yarn_inv_freq"]
