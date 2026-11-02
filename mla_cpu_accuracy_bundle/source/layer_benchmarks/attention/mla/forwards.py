"""Kimi-K2.5/DeepSeek-style MLA attention benchmark variants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import MLACompressedKVRopeCache

MIN_MASKED = -3.0e4


@dataclass
class Config:
    hidden_size: int = 7168
    num_attention_heads: int = 64
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    kv_lora_rank: int = 512
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    ctx_len: int = 32768
    rope_theta: float = 50000.0
    max_position_embeddings: int = 262144
    rope_factor: float = 64.0
    rope_original_max_position_embeddings: int = 4096
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    num_kv_heads: int = 4
    num_kv_blocks: int = 4
    par_num_split: int = 16
    mode: str = "par"
    head_block_size: int = 1
    mla_absorption: bool = True
    mla_online: bool = False
    skip_kv: bool = True


def _correction_dim(rotations: float, dim: int, base: float, max_pos: int) -> float:
    return (dim * math.log(max_pos / (rotations * 2 * math.pi))) / (2 * math.log(base))


def _correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_pos: int) -> tuple[int, int]:
    low = math.floor(_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def _mscale(scale: float, value: float) -> float:
    return 1.0 if scale <= 1 else 0.1 * value * math.log(scale) + 1.0


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


class YarnRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, cfg: Config):
        super().__init__()
        self.dim = dim
        base = cfg.rope_theta
        factor = cfg.rope_factor
        freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freq_inter = 1.0 / (factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        low, high = _correction_range(cfg.rope_beta_fast, cfg.rope_beta_slow, dim, base,
                                      cfg.rope_original_max_position_embeddings)
        if low == high:
            high += 0.001
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
        mask = 1.0 - ramp
        inv_freq = freq_inter * (1 - mask) + freq_extra * mask
        t = torch.arange(cfg.ctx_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        scale = _mscale(factor, cfg.rope_mscale) / _mscale(factor, cfg.rope_mscale_all_dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos() * scale, persistent=False)
        self.register_buffer("sin_cached", emb.sin() * scale, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[position_ids.long()], self.sin_cached[position_ids.long()]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def _expand_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    if x.shape[1] == heads:
        return x
    repeats = (heads + x.shape[1] - 1) // x.shape[1]
    return x.unsqueeze(2).expand(-1, -1, repeats, -1, -1).reshape(
        x.shape[0], repeats * x.shape[1], x.shape[2], x.shape[3]
    )[:, :heads]


def _project_block(ckv: torch.Tensor, k_pe: torch.Tensor, query_heads: int,
                   absorption: bool, model: "MLA") -> tuple[torch.Tensor, torch.Tensor]:
    """Return K and V in the head layout consumed by an attention variant."""
    if absorption:
        return torch.cat((ckv, k_pe), dim=-1), ckv
    ckv = _expand_heads(ckv, query_heads)
    k_pe = _expand_heads(k_pe, query_heads)
    k_nope = torch.matmul(ckv, model.per_head_k_up_normal)
    return torch.cat((k_nope, k_pe), dim=-1), ckv


def _causal_mask(position_ids: torch.Tensor, start: int, length: int) -> torch.Tensor:
    indices = torch.arange(start, start + length, device=position_ids.device)
    return indices.view(1, 1, 1, -1) > position_ids[:, None, :, None]


def _full_attention(query: torch.Tensor, ckv: torch.Tensor, k_pe: torch.Tensor,
                    position_ids: torch.Tensor, cfg: Config, scaling: float,
                    model: "MLA") -> torch.Tensor:
    keys, values = _project_block(ckv, k_pe, query.shape[1], cfg.mla_absorption, model)
    keys = _expand_heads(keys, query.shape[1])
    values = _expand_heads(values, query.shape[1])
    scores = torch.matmul(query, keys.transpose(2, 3)) * scaling
    scores = scores.masked_fill(_causal_mask(position_ids, 0, cfg.ctx_len), MIN_MASKED)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(probs, values)
    return torch.matmul(output, model.per_head_v_up).transpose(1, 2).contiguous()


def _parallel_attention(query: torch.Tensor, ckv: torch.Tensor, k_pe: torch.Tensor,
                        position_ids: torch.Tensor, cfg: Config, scaling: float,
                        model: "MLA", online: bool = False) -> torch.Tensor:
    bsz, nqh, qlen, d = query.shape
    hkv = cfg.num_kv_heads if cfg.mla_absorption else nqh
    nrep = nqh // hkv
    split = cfg.par_num_split
    q_fold = query.reshape(bsz, hkv, qlen * nrep, d)
    q5 = q_fold.unsqueeze(2).expand(bsz, hkv, split, qlen * nrep, d)
    block = (cfg.ctx_len + cfg.num_kv_blocks - 1) // cfg.num_kv_blocks
    nominal = (block + split - 1) // split
    offsets = (torch.arange(split, device=query.device)[:, None] * nominal
               + torch.arange(nominal, device=query.device)[None, :]).view(1, 1, split, 1, nominal)
    max_buf, sum_buf, out_buf = [], [], []
    for j in range(cfg.num_kv_blocks):
        start = j * block
        length = cfg.ctx_len - start if j == cfg.num_kv_blocks - 1 else block
        end = start + length
        cb = MLACompressedKVRopeCache.read_block(ckv, position_ids, start, end)
        pb = MLACompressedKVRopeCache.read_block(k_pe, position_ids, start, end)
        kb, vb = _project_block(cb, pb, nqh, cfg.mla_absorption, model)
        pad = (-length) % split
        if pad:
            kb = F.pad(kb, (0, 0, 0, pad))
            vb = F.pad(vb, (0, 0, 0, pad))
        th = (length + pad) // split
        k5 = kb.view(bsz, hkv, split, th, d)
        v5 = vb.view(bsz, hkv, split, th, cfg.kv_lora_rank)
        attn = torch.matmul(q5, k5.transpose(-1, -2)) * scaling
        if pad:
            starts = torch.arange(split, device=query.device) * th
            valid = length - starts
            attn = attn.masked_fill(torch.arange(th, device=query.device)[None, :] >= valid[:, None]
                                    .view(1, 1, split, 1, th), MIN_MASKED)
        off = offsets[..., :th]
        causal = off > (position_ids - start)[:, None, None, :, None]
        attn = attn.masked_fill(causal, MIN_MASKED)
        m = attn.max(dim=-1).values
        e = torch.exp(attn - m.unsqueeze(-1))
        if cfg.skip_kv:
            skip_future = (torch.tensor(start, device=query.device) > position_ids.max(dim=-1).values).all()
            m = torch.where(skip_future, torch.full_like(m, MIN_MASKED), m)
            e = torch.where(skip_future, torch.zeros_like(e), e)
        s = e.sum(dim=-1)
        o = torch.matmul(e, v5)
        max_buf.append(m)
        sum_buf.append(s)
        out_buf.append(o)
    ms = torch.stack(max_buf)
    ss = torch.stack(sum_buf)
    os = torch.stack(out_buf)
    m1 = ms.max(dim=0).values
    w1 = torch.exp(ms - m1.unsqueeze(0))
    s1 = (w1 * ss).sum(dim=0)
    o1 = (w1.unsqueeze(-1) * os).sum(dim=0)
    m2 = m1.max(dim=2).values
    w2 = torch.exp(m1 - m2.unsqueeze(2))
    s2 = (w2 * s1).sum(dim=2)
    o2 = (w2.unsqueeze(-1) * o1).sum(dim=2) / s2.unsqueeze(-1)
    latent = o2.view(bsz, hkv, nrep, qlen, cfg.kv_lora_rank).reshape(bsz, nqh, qlen, cfg.kv_lora_rank)
    return torch.matmul(latent, model.per_head_v_up).transpose(1, 2).contiguous()


def _parallel_prefill(query: torch.Tensor, ckv: torch.Tensor, k_pe: torch.Tensor,
                      position_ids: torch.Tensor, cfg: Config, scaling: float,
                      model: "MLA", online: bool) -> torch.Tensor:
    """Prefill-parallel implementation; online controls the block merge order."""
    bsz, nqh, qlen, d = query.shape
    hkv = cfg.num_kv_heads if cfg.mla_absorption else nqh
    nrep = nqh // hkv
    split = cfg.par_num_split
    q6 = query.reshape(bsz, hkv, nrep, qlen, d).unsqueeze(2).expand(
        bsz, hkv, split, nrep, qlen, d
    )
    block = (cfg.ctx_len + cfg.num_kv_blocks - 1) // cfg.num_kv_blocks
    nominal = (block + split - 1) // split
    offsets = (torch.arange(split, device=query.device)[:, None] * nominal
               + torch.arange(nominal, device=query.device)[None, :]).view(1, 1, split, 1, nominal)

    m_acc = s_acc = o_acc = None
    max_buf, sum_buf, out_buf = [], [], []
    for j in range(cfg.num_kv_blocks):
        start = j * block
        length = cfg.ctx_len - start if j == cfg.num_kv_blocks - 1 else block
        cb = MLACompressedKVRopeCache.read_block(ckv, position_ids, start, start + length)
        pb = MLACompressedKVRopeCache.read_block(k_pe, position_ids, start, start + length)
        kb, vb = _project_block(cb, pb, nqh, cfg.mla_absorption, model)
        pad = (-length) % split
        if pad:
            kb = F.pad(kb, (0, 0, 0, pad))
            vb = F.pad(vb, (0, 0, 0, pad))
        th = (length + pad) // split
        k5 = kb.view(bsz, hkv, split, th, d)
        v5 = vb.view(bsz, hkv, split, th, cfg.kv_lora_rank)
        attn = torch.matmul(q6, k5.unsqueeze(3).transpose(-1, -2)) * scaling
        if pad:
            starts = torch.arange(split, device=query.device) * th
            valid = length - starts
            bad = torch.arange(th, device=query.device)[None, :] >= valid[:, None]
            attn = attn.masked_fill(bad.view(1, 1, split, 1, 1, th), MIN_MASKED)
        off = offsets[..., :th]
        bad = off.unsqueeze(3) > (position_ids - start)[:, None, None, None, :, None]
        attn = attn.masked_fill(bad, MIN_MASKED)
        m = attn.max(dim=-1).values
        e = torch.exp(attn - m.unsqueeze(-1))
        if cfg.skip_kv:
            skip_future = (torch.tensor(start, device=query.device) > position_ids.max(dim=-1).values).all()
            m = torch.where(skip_future, torch.full_like(m, MIN_MASKED), m)
            e = torch.where(skip_future, torch.zeros_like(e), e)
        s = e.sum(dim=-1)
        o = torch.matmul(e, v5.unsqueeze(3))
        if online:
            if m_acc is None:
                m_acc, s_acc, o_acc = m, s, o
            else:
                new_m = torch.maximum(m_acc, m)
                old_w = torch.exp(m_acc - new_m)
                new_w = torch.exp(m - new_m)
                m_acc = new_m
                s_acc = old_w * s_acc + new_w * s
                o_acc = old_w.unsqueeze(-1) * o_acc + new_w.unsqueeze(-1) * o
        else:
            max_buf.append(m)
            sum_buf.append(s)
            out_buf.append(o)

    if not online:
        ms, ss, os = torch.stack(max_buf), torch.stack(sum_buf), torch.stack(out_buf)
        m_acc = ms.max(dim=0).values
        weights = torch.exp(ms - m_acc.unsqueeze(0))
        s_acc = (weights * ss).sum(dim=0)
        o_acc = (weights.unsqueeze(-1) * os).sum(dim=0)
    m2 = m_acc.max(dim=2).values
    weights = torch.exp(m_acc - m2.unsqueeze(2))
    denom = (weights * s_acc).sum(dim=2)
    latent = (weights.unsqueeze(-1) * o_acc).sum(dim=2) / denom.unsqueeze(-1)
    latent = latent.reshape(bsz, nqh, qlen, cfg.kv_lora_rank)
    return torch.matmul(latent, model.per_head_v_up).transpose(1, 2).contiguous()


def _kv_attention(query: torch.Tensor, ckv: torch.Tensor, k_pe: torch.Tensor,
                  position_ids: torch.Tensor, cfg: Config, scaling: float,
                  model: "MLA") -> torch.Tensor:
    bsz, nqh, qlen, _ = query.shape
    max_acc = sum_acc = out_acc = None
    block = (cfg.ctx_len + cfg.num_kv_blocks - 1) // cfg.num_kv_blocks
    for j in range(cfg.num_kv_blocks):
        start = j * block
        length = cfg.ctx_len - start if j == cfg.num_kv_blocks - 1 else block
        cb = MLACompressedKVRopeCache.read_block(ckv, position_ids, start, start + length)
        pb = MLACompressedKVRopeCache.read_block(k_pe, position_ids, start, start + length)
        keys, values = _project_block(cb, pb, nqh, cfg.mla_absorption, model)
        keys = _expand_heads(keys, nqh)
        values = _expand_heads(values, nqh)
        scores = torch.matmul(query, keys.transpose(2, 3)) * scaling
        scores = scores.masked_fill(_causal_mask(position_ids, start, length), MIN_MASKED)
        m = scores.max(dim=-1).values
        e = torch.exp(scores - m.unsqueeze(-1))
        if cfg.skip_kv:
            skip_future = (torch.tensor(start, device=query.device) > position_ids.max(dim=-1).values).all()
            m = torch.where(skip_future, torch.full_like(m, MIN_MASKED), m)
            e = torch.where(skip_future, torch.zeros_like(e), e)
        s = e.sum(dim=-1)
        o = torch.matmul(e, values)
        if max_acc is None:
            max_acc, sum_acc, out_acc = m, s, o
        else:
            new_m = torch.maximum(max_acc, m)
            old_w, new_w = torch.exp(max_acc - new_m), torch.exp(m - new_m)
            max_acc = new_m
            sum_acc = old_w * sum_acc + new_w * s
            out_acc = old_w.unsqueeze(-1) * out_acc + new_w.unsqueeze(-1) * o
    latent = out_acc / sum_acc.unsqueeze(-1)
    return torch.matmul(latent, model.per_head_v_up).transpose(1, 2).contiguous()


def _head_attention(query: torch.Tensor, ckv: torch.Tensor, k_pe: torch.Tensor,
                    position_ids: torch.Tensor, cfg: Config, scaling: float,
                    model: "MLA") -> torch.Tensor:
    keys, values = _project_block(ckv, k_pe, query.shape[1], cfg.mla_absorption, model)
    keys = _expand_heads(keys, query.shape[1])
    values = _expand_heads(values, query.shape[1])
    size = cfg.head_block_size if cfg.head_block_size > 0 else query.shape[1]
    outputs = []
    mask = _causal_mask(position_ids, 0, cfg.ctx_len)
    for start in range(0, query.shape[1], size):
        end = min(start + size, query.shape[1])
        scores = torch.matmul(query[:, start:end], keys[:, start:end].transpose(2, 3)) * scaling
        scores = scores.masked_fill(mask, MIN_MASKED)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        outputs.append(torch.matmul(torch.matmul(probs, values[:, start:end]),
                                    model.per_head_v_up[:, start:end]))
    return torch.cat(outputs, dim=1).transpose(1, 2).contiguous()


class MLA(MLACompressedKVRopeCache, nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.config = cfg
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_key_value_heads = 1
        self.q_lora_rank = cfg.q_lora_rank
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.kv_lora_rank = cfg.kv_lora_rank
        self.v_head_dim = cfg.v_head_dim
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(cfg.q_lora_rank)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, cfg.num_attention_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(cfg.hidden_size, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank)
        self.kv_b_proj = nn.Linear(cfg.kv_lora_rank,
                                   cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.v_head_dim, cfg.hidden_size, bias=False)
        if cfg.num_kv_heads != 1:
            old = self.kv_a_proj_with_mqa
            new = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * (cfg.kv_lora_rank + cfg.qk_rope_head_dim), bias=False)
            new.weight.data.copy_(old.weight.data.repeat(cfg.num_kv_heads, 1))
            self.kv_a_proj_with_mqa = new
            self.num_key_value_heads = cfg.num_kv_heads
        self.rotary_emb = YarnRotaryEmbedding(cfg.qk_rope_head_dim, cfg)
        self.softmax_scale = self.q_head_dim ** -0.5
        self._qeff_init()

    def _qeff_init(self) -> None:
        q_up, q_rope = self.q_b_proj.weight.T.view(-1, self.num_heads, self.q_head_dim).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        self.q_up = nn.Parameter(q_up.reshape(-1, self.num_heads * self.qk_nope_head_dim).unsqueeze(0).detach().clone())
        self.q_rope = nn.Parameter(q_rope.reshape(-1, self.num_heads * self.qk_rope_head_dim).unsqueeze(0).detach().clone())
        k_up, v_up = self.kv_b_proj.weight.T.view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        self.k_up = nn.Parameter(k_up.reshape(-1, self.num_heads * self.qk_nope_head_dim).detach().clone().unsqueeze(0))
        self.v_up = nn.Parameter(v_up.reshape(-1, self.num_heads * self.v_head_dim).detach().clone().unsqueeze(0))
        qh = self.q_up.squeeze(0).view(-1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1)
        kh = self.k_up.squeeze(0).view(-1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1).transpose(1, 2)
        vh = self.v_up.squeeze(0).view(-1, self.num_heads, self.v_head_dim).transpose(0, 1)
        self.per_head_q_up = nn.Parameter(qh.unsqueeze(0).detach().clone())
        self.per_head_k_up = nn.Parameter(kh.unsqueeze(0).detach().clone())
        self.per_head_v_up = nn.Parameter(vh.unsqueeze(0).detach().clone())
        self.per_head_k_up_normal = nn.Parameter(self.per_head_k_up.transpose(2, 3).detach().clone())
        self.fusedqk = nn.Parameter(torch.bmm(qh, kh).reshape(-1, self.num_heads, self.q_lora_rank, self.kv_lora_rank).detach().clone())

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor,
                compressed_kv: torch.Tensor, k_pe: torch.Tensor):
        c = self.cfg
        bsz, qlen, _ = hidden_states.shape
        packed = self.kv_a_proj_with_mqa(hidden_states).view(
            bsz, qlen, c.num_kv_heads, c.kv_lora_rank + c.qk_rope_head_dim).transpose(1, 2)
        kva = self.kv_a_layernorm(packed[..., :c.kv_lora_rank])
        kpe = packed[..., c.kv_lora_rank:]
        qlatent = self.q_a_layernorm(self.q_a_proj(hidden_states))
        qpe = torch.matmul(qlatent, self.q_rope).view(bsz, qlen, self.num_heads, c.qk_rope_head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(position_ids)
        qpe, kpe = apply_rope(qpe, kpe, cos, sin, position_ids)
        compressed_kv = self.scatter(compressed_kv, position_ids, kva)
        k_pe = self.scatter(k_pe, position_ids, kpe)
        if c.mla_absorption:
            if c.mla_online:
                q_nope = torch.matmul(
                    qlatent.unsqueeze(1), torch.matmul(self.per_head_q_up, self.per_head_k_up)
                )
            else:
                q_nope = torch.matmul(qlatent.unsqueeze(1), self.fusedqk)
        else:
            q_nope = torch.bmm(qlatent, self.q_up)
            q_nope = q_nope.view(bsz, qlen, self.num_heads, c.qk_nope_head_dim).transpose(1, 2)
        query = torch.cat((q_nope, qpe), dim=-1)
        if c.mode == "none":
            attn = _full_attention(query, compressed_kv, k_pe, position_ids, c, self.softmax_scale, self)
        elif c.mode == "kv":
            attn = _kv_attention(query, compressed_kv, k_pe, position_ids, c, self.softmax_scale, self)
        elif c.mode == "h":
            attn = _head_attention(query, compressed_kv, k_pe, position_ids, c, self.softmax_scale, self)
        elif c.mode == "par":
            # The standalone ``par`` kernel is a decode layout.  For a caller
            # that supplies multiple query tokens, use the equivalent 6-D
            # prefill layout instead of flattening Q and replicated heads.
            if qlen == 1:
                attn = _parallel_attention(query, compressed_kv, k_pe, position_ids,
                                           c, self.softmax_scale, self)
            else:
                attn = _parallel_prefill(query, compressed_kv, k_pe, position_ids,
                                         c, self.softmax_scale, self, online=False)
        elif c.mode == "prefill_par":
            attn = _parallel_prefill(query, compressed_kv, k_pe, position_ids, c,
                                     self.softmax_scale, self, online=False)
        elif c.mode == "prefill_par_online":
            attn = _parallel_prefill(query, compressed_kv, k_pe, position_ids, c,
                                     self.softmax_scale, self, online=True)
        else:
            raise ValueError(f"unknown MLA attention mode: {c.mode}")
        output = self.o_proj(attn.reshape(bsz, qlen, self.num_heads * c.v_head_dim))
        return output, compressed_kv, k_pe


def build(cfg: Config, dtype: torch.dtype = torch.float32) -> MLA:
    return MLA(cfg).to(dtype).eval()
