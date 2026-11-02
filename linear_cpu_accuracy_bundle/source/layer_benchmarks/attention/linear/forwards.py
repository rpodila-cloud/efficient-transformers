"""Qwen3.5 / Qwen3-Next linear attention (GatedDeltaNet), one layer.

Two paths, one module:

* ``chunk``     – prefill: split q/k/v depthwise convs + chunked delta-rule scan
  (row-by-row forward substitution); returns the final recurrent state.
* ``recurrent`` – decode: conv-state update + one delta-rule step; returns the
  new conv and recurrent state.

The decode path includes matrix-state head blocking and a combined K/Q
projection through the recurrent state. Parameter names and creation order are
kept, so the same seed gives the original module's weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from . import ops
from .cache import Qwen35LinearAttentionCache


@dataclass
class Config:
    path: str = "chunk"  # "chunk" | "recurrent"
    hidden_size: int = 2048
    num_k_heads: int = 16
    num_v_heads: int = 32
    head_k_dim: int = 128
    head_v_dim: int = 128
    conv_kernel: int = 4
    rms_norm_eps: float = 1e-6
    chunk_size: int = 64
    conv_state_layout: str = "channel_first"
    head_block_size: int = 0

    @property
    def key_dim(self) -> int:
        return self.head_k_dim * self.num_k_heads

    @property
    def value_dim(self) -> int:
        return self.head_v_dim * self.num_v_heads

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim

    @property
    def num_head_blocks(self) -> int:
        return self.num_v_heads // self.head_block_size if self.head_block_size else 1


class GatedRMSNorm(nn.Module):
    """CustomRMSNorm(x) * silu(gate); only the norm is the QAIC custom op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return ops.rms_norm(hidden_states, self.weight, self.variance_epsilon) * F.silu(gate.to(torch.float32))


def _depthwise(weight: torch.Tensor, bias: torch.Tensor, kernel: int) -> nn.Conv1d:
    dim = weight.shape[0]
    conv = nn.Conv1d(dim, dim, kernel, groups=dim, padding=kernel - 1, bias=True)
    conv.weight.data.copy_(weight)
    conv.bias.data.copy_(bias)
    return conv


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


class LinearAttention(Qwen35LinearAttentionCache, nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        kd, vd, K, C = c.key_dim, c.value_dim, c.conv_kernel, c.chunk_size
        self.in_proj_qkv = nn.Linear(c.hidden_size, c.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(c.hidden_size, vd, bias=False)
        self.in_proj_b = nn.Linear(c.hidden_size, c.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(c.hidden_size, c.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(c.conv_dim, c.conv_dim, K, groups=c.conv_dim, padding=K - 1, bias=True)
        # Per-group copies of conv1d for the prefill path (head-parallel on AIC).
        w, b = self.conv1d.weight, self.conv1d.bias
        self.q_conv1d = _depthwise(w[:kd], b[:kd], K)
        self.k_conv1d = _depthwise(w[kd : 2 * kd], b[kd : 2 * kd], K)
        self.v_conv1d = _depthwise(w[2 * kd :], b[2 * kd :], K)
        self.dt_bias = nn.Parameter(torch.ones(c.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(c.num_v_heads).uniform_(0, 16)))
        self.norm = GatedRMSNorm(c.head_v_dim, eps=c.rms_norm_eps)
        self.out_proj = nn.Linear(vd, c.hidden_size, bias=False)
        self.register_buffer("_mask_strict", torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1), persistent=False)
        self.register_buffer("_mask_upper", torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0), persistent=False)
        self.register_buffer("_eye", torch.eye(C).view(1, 1, 1, C, C), persistent=True)

    # ── shared ──
    def project(self, hidden_states: torch.Tensor):
        """-> mixed_qkv [B, conv_dim, T], z [B, T, Hv, dv], beta [B, T, Hv], g [B, T, Hv]."""
        batch, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch, seq_len, -1, self.c.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
        return mixed_qkv, z, beta, g

    def split_qkv(self, mixed_qkv: torch.Tensor):
        """[B, T, conv_dim] -> q, k [B, T, Hv, dk], v [B, T, Hv, dv] (k heads repeated)."""
        c = self.c
        batch, seq_len, _ = mixed_qkv.shape
        query = mixed_qkv[..., : c.key_dim].reshape(batch, seq_len, -1, c.head_k_dim)
        key = mixed_qkv[..., c.key_dim : 2 * c.key_dim].reshape(batch, seq_len, -1, c.head_k_dim)
        value = mixed_qkv[..., 2 * c.key_dim :].reshape(batch, seq_len, -1, c.head_v_dim)
        rep = c.num_v_heads // c.num_k_heads
        if rep > 1:
            query, key = query.repeat_interleave(rep, dim=2), key.repeat_interleave(rep, dim=2)
        return query, key, value

    def gated_norm_out(self, core: torch.Tensor, z: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        dv = self.c.head_v_dim
        return self.out_proj(self.norm(core.reshape(-1, dv), z.reshape(-1, dv)).reshape(batch, seq_len, -1))

    # ── chunk (prefill) ──
    def chunk_delta_rule(self, query, key, value, g, beta, position_ids):
        """q/k/v head-first [B, H, T, D]; g/beta [B, T, H] -> (core [B, T, H, dv], S [B, H, dk, dv])."""
        C = self.c.chunk_size
        mask_strict = self._mask_strict
        initial_dtype = query.dtype
        query, key = l2norm(query), l2norm(key)
        query, key, value = (x.contiguous().to(torch.float32) for x in (query, key, value))
        beta = beta.transpose(1, 2).contiguous().to(torch.float32)
        g = g.transpose(1, 2).contiguous().to(torch.float32)

        # position_ids == -1 marks padding tokens.
        mask = (position_ids != -1).unsqueeze(1)
        g = torch.where(mask, g, torch.zeros(g.shape, dtype=g.dtype))
        qkv_zeros = torch.zeros(key.shape, dtype=key.dtype)
        key = torch.where(mask.unsqueeze(-1), key, qkv_zeros)
        query = torch.where(mask.unsqueeze(-1), query, qkv_zeros)
        value = torch.where(mask.unsqueeze(-1), value, qkv_zeros)

        batch, heads, seq_len, dk = key.shape
        dv = value.shape[-1]
        pad = (C - seq_len % C) % C
        query, key, value = (F.pad(x, (0, 0, 0, pad), mode="constant", value=0.0) for x in (query, key, value))
        beta, g = F.pad(beta, (0, pad), mode="constant", value=0.0), F.pad(g, (0, pad), mode="constant", value=0.0)
        total = seq_len + pad
        query = query * (1 / (query.shape[-1] ** 0.5))

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, value, k_beta, v_beta = (
            x.reshape(x.shape[0], x.shape[1], -1, C, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
        )
        g = g.reshape(g.shape[0], g.shape[1], -1, C)

        # Cumulative decay as a lower-triangular matmul (not cumsum).
        idx = torch.arange(C)
        g = g @ (idx.unsqueeze(1) >= idx.unsqueeze(0)).to(g.dtype).T
        diff = (g.unsqueeze(-1) - g.unsqueeze(-2)) * (~mask_strict).float()
        decay_mask = diff.exp().float() * (~mask_strict).float()

        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(self._mask_upper, 0)
        # (I - A)^-1 by row-by-row forward substitution.  Build each solved
        # row functionally: Dynamo lowers slice assignment to ScatterElements,
        # which QAIC compiles very poorly for the 64-row chunk.  The original
        # benchmark's functional stack form lowers to the efficient scatter
        # pattern instead.
        solved_rows = [attn[..., 0, :]]
        for i in range(1, C):
            row = attn[..., i, :i]
            sub = torch.stack(solved_rows, dim=-2)[..., :i]
            solved = row + (row.unsqueeze(-1) * sub).sum(-2)
            solved_rows.append(torch.cat((solved, attn[..., i, i:]), dim=-1))
        attn = torch.stack(solved_rows, dim=-2)
        attn = attn + self._eye.to(dtype=attn.dtype)

        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        state = torch.zeros(batch, heads, dk, dv).to(value)
        outputs = []
        for i in range(total // C):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_strict, 0)
            v_new = v_i - k_cumdecay[:, :, i] @ state
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
            outputs.append(attn_inter + attn @ v_new)
            state = (
                state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )
        core = torch.stack(outputs, dim=2)
        core = core.reshape(core.shape[0], core.shape[1], -1, core.shape[-1])[:, :, :seq_len]
        return core.transpose(1, 2).contiguous().to(initial_dtype), state

    # ── recurrent (decode) ──
    def recurrent_head_block(self, query, key, value, g, beta, recurrent_state):
        """Evaluate one contiguous value-head block for a decode token."""
        dtype = query.dtype
        q, k, v = query.float(), key.float(), value.float()
        q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
        q = q * (1.0 / (q.shape[-1] ** 0.5))
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        b = beta.transpose(1, 2).float().unsqueeze(-1)
        decay = g.transpose(1, 2).float().exp().unsqueeze(-1).unsqueeze(-1)

        state = recurrent_state.float() * decay[:, :, 0]
        k0, q0 = k[:, :, 0], q[:, :, 0]
        # One state matmul supplies both k@S and q@S.
        kq = torch.cat((k0.unsqueeze(-2), q0.unsqueeze(-2)), dim=-2)
        state_proj = torch.matmul(kq, state)
        kv_mem, q_mem = state_proj[:, :, 0], state_proj[:, :, 1]
        delta = (v[:, :, 0] - kv_mem) * b[:, :, 0]
        out = q_mem + (q0 * k0).sum(dim=-1, keepdim=True) * delta
        state = state + k0.unsqueeze(-1) * delta.unsqueeze(-2)
        return out.unsqueeze(2).transpose(1, 2).to(dtype), state.to(recurrent_state.dtype)

    def recurrent_step(self, query, key, value, g, beta, recurrent_state):
        """One-token recurrence, optionally split into contiguous head blocks."""
        if recurrent_state.ndim != 4:
            raise ValueError("recurrent_state must have shape [B,H,DK,DV]")
        block = self.c.head_block_size
        if block == 0:
            return self.recurrent_head_block(query, key, value, g, beta, recurrent_state)

        blocked_state = recurrent_state.reshape(
            recurrent_state.shape[0],
            self.c.num_head_blocks,
            block,
            self.c.head_k_dim,
            self.c.head_v_dim,
        )
        outputs, states = [], []
        for block_idx in range(self.c.num_head_blocks):
            start, end = block_idx * block, (block_idx + 1) * block
            block_out, block_state = self.recurrent_head_block(
                query[:, :, start:end],
                key[:, :, start:end],
                value[:, :, start:end],
                g[:, :, start:end],
                beta[:, :, start:end],
                blocked_state[:, block_idx],
            )
            outputs.append(block_out)
            states.append(block_state)
        return torch.cat(outputs, dim=2), torch.cat(states, dim=1)

    def forward(self, hidden_states, position_ids, conv_state=None, recurrent_state=None):
        c = self.c
        batch, seq_len, _ = hidden_states.shape
        mixed_qkv, z, beta, g = self.project(hidden_states)

        if c.path == "recurrent":
            mixed_qkv, new_conv_state = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                position_ids,
                self.conv1d.bias,
                c.conv_state_layout,
            )
            query, key, value = self.split_qkv(mixed_qkv.transpose(1, 2))
            core, new_state = self.recurrent_step(query, key, value, g, beta, recurrent_state)
            return self.gated_norm_out(core, z, batch, seq_len), new_conv_state, new_state

        kd = c.key_dim
        q = F.silu(self.q_conv1d(mixed_qkv[:, :kd])[:, :, :seq_len])
        k = F.silu(self.k_conv1d(mixed_qkv[:, kd : 2 * kd])[:, :, :seq_len])
        v = F.silu(self.v_conv1d(mixed_qkv[:, 2 * kd :])[:, :, :seq_len])
        # [B, dim, T] -> head-first [B, H, T, D].
        query = q.reshape(batch, c.num_k_heads, c.head_k_dim, seq_len).permute(0, 1, 3, 2)
        key = k.reshape(batch, c.num_k_heads, c.head_k_dim, seq_len).permute(0, 1, 3, 2)
        value = v.reshape(batch, c.num_v_heads, c.head_v_dim, seq_len).permute(0, 1, 3, 2)
        rep = c.num_v_heads // c.num_k_heads
        if rep > 1:
            query, key = query.repeat_interleave(rep, dim=1), key.repeat_interleave(rep, dim=1)
        core, final_state = self.chunk_delta_rule(query, key, value, g, beta, position_ids)
        return self.gated_norm_out(core, z, batch, seq_len), final_state
