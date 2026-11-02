"""Independent dense eager reference for MLA parity.

The reference materializes the updated latent KV cache and performs ordinary
causal attention. It deliberately does not import the production MLA forward,
cache, or custom-op modules.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


MIN_MASKED = -3.0e4


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * y.to(dtype)


def _correction_dim(rotations: float, dim: int, base: float, max_pos: int) -> float:
    return dim * math.log(max_pos / (rotations * 2 * math.pi)) / (2 * math.log(base))


def _correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_pos: int) -> tuple[int, int]:
    low = math.floor(_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def _mscale(scale: float, value: float) -> float:
    return 1.0 if scale <= 1 else 0.1 * value * math.log(scale) + 1.0


class YarnRotary(nn.Module):
    def __init__(self, dim: int, cfg):
        super().__init__()
        base, factor = cfg.rope_theta, cfg.rope_factor
        extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        inter = 1.0 / (factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        low, high = _correction_range(cfg.rope_beta_fast, cfg.rope_beta_slow, dim, base,
                                      cfg.rope_original_max_position_embeddings)
        if low == high:
            high += 0.001
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
        inv = inter * ramp + extra * (1.0 - ramp)
        positions = torch.arange(cfg.ctx_len, dtype=torch.float32)
        emb = torch.outer(positions, inv)
        scale = _mscale(factor, cfg.rope_mscale) / _mscale(factor, cfg.rope_mscale_all_dim)
        emb = torch.cat((emb, emb), dim=-1)
        self.register_buffer("cos", emb.cos() * scale, persistent=False)
        self.register_buffer("sin", emb.sin() * scale, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[positions.long()], self.sin[positions.long()]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-b, a), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    d = cos.shape[-1]
    rotated, tail = x[..., :d], x[..., d:]
    rotated = rotated.view(*rotated.shape[:-1], d // 2, 2).transpose(-1, -2).reshape_as(rotated)
    rotated = rotated * cos + _rotate_half(rotated) * sin
    return torch.cat((rotated, tail), dim=-1)


def _scatter(cache: torch.Tensor, positions: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    out = cache.clone()
    batch = torch.arange(cache.shape[0], device=cache.device)
    pos = positions.long()
    for t in range(updates.shape[2]):
        out[batch, :, pos[:, t], :] = updates[:, :, t, :]
    return out


class MLAReference(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(cfg.q_lora_rank)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * (cfg.kv_lora_rank + cfg.qk_rope_head_dim), bias=False)
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank)
        self.kv_b_proj = nn.Linear(cfg.kv_lora_rank,
                                   cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.v_head_dim, cfg.hidden_size, bias=False)
        self.rotary_emb = YarnRotary(cfg.qk_rope_head_dim, cfg)

    def _up_matrices(self):
        c = self.cfg
        q_up, q_rope = self.q_b_proj.weight.T.view(-1, c.num_attention_heads,
                                                    c.qk_nope_head_dim + c.qk_rope_head_dim).split(
                                                        [c.qk_nope_head_dim, c.qk_rope_head_dim], dim=-1)
        k_up, v_up = self.kv_b_proj.weight.T.view(-1, c.num_attention_heads,
                                                   c.qk_nope_head_dim + c.v_head_dim).split(
                                                       [c.qk_nope_head_dim, c.v_head_dim], dim=-1)
        q_up = q_up.permute(1, 0, 2)
        q_rope = q_rope.permute(1, 0, 2)
        k_up = k_up.permute(1, 2, 0)
        v_up = v_up.permute(1, 0, 2)
        return q_up, q_rope, k_up, v_up

    def forward(self, hidden_states, position_ids, compressed_kv, k_pe):
        c = self.cfg
        b, qlen, _ = hidden_states.shape
        packed = self.kv_a_proj_with_mqa(hidden_states).view(
            b, qlen, c.num_kv_heads, c.kv_lora_rank + c.qk_rope_head_dim).transpose(1, 2)
        ckv_new = self.kv_a_layernorm(packed[..., :c.kv_lora_rank])
        kpe_new = packed[..., c.kv_lora_rank:]
        qlatent = self.q_a_layernorm(self.q_a_proj(hidden_states))
        _, q_rope, k_up, v_up = self._up_matrices()
        qpe = torch.matmul(qlatent.unsqueeze(1), q_rope)
        qpe = qpe.view(b, c.num_attention_heads, qlen, c.qk_rope_head_dim)
        cos, sin = self.rotary_emb(position_ids)
        qpe = _apply_rope(qpe, cos, sin)
        kpe_new = _apply_rope(kpe_new, cos, sin)
        compressed_kv = _scatter(compressed_kv, position_ids, ckv_new)
        k_pe = _scatter(k_pe, position_ids, kpe_new)
        repeats = c.num_attention_heads // c.num_kv_heads
        if c.mla_absorption:
            q_nope = torch.matmul(qlatent.unsqueeze(1), torch.matmul(self._up_matrices()[0], k_up))
            keys = torch.cat((compressed_kv, k_pe), dim=-1)
            values = compressed_kv
        else:
            q_nope = torch.matmul(qlatent.unsqueeze(1), self._up_matrices()[0])
            ckv_heads = compressed_kv.repeat_interleave(repeats, dim=1)
            keys = torch.einsum("bhtk,hdk->bhtd", ckv_heads, k_up)
            keys = torch.cat((keys, k_pe.repeat_interleave(repeats, dim=1)), dim=-1)
            values = ckv_heads
        if c.mla_absorption:
            keys = keys.repeat_interleave(repeats, dim=1)
            values = values.repeat_interleave(repeats, dim=1)
        query = torch.cat((q_nope, qpe), dim=-1)
        scores = torch.matmul(query.float(), keys.float().transpose(-1, -2)) * (query.shape[-1] ** -0.5)
        key_pos = torch.arange(c.ctx_len, device=hidden_states.device).view(1, 1, 1, -1)
        scores = scores.masked_fill(key_pos > position_ids[:, None, :, None], MIN_MASKED)
        probs = torch.softmax(scores, dim=-1).to(values.dtype)
        latent = torch.matmul(probs, values)
        out = torch.matmul(latent, v_up.unsqueeze(0).repeat(b, 1, 1, 1))
        out = out.transpose(1, 2).reshape(b, qlen, c.num_attention_heads * c.v_head_dim)
        return self.o_proj(out), compressed_kv, k_pe
