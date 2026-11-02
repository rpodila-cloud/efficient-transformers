"""Qwen3 GQA single layer with KV-blocked attention (QAIC decode/prefill).

Ported from ``gqa_benchmark/qwen3_gqa_kv_blocking_microbench.py`` (``GQAKVBlockingLayer``).
The parameter names and creation order are kept, so ``state_dict`` loads either way.

This file exposes all eight standalone implementation names:

* ``prefill_attn_online_prefill`` – online softmax over KV blocks; Q is chunked
  by ``q_block_size`` tokens and ``q_head_block_chunk`` head groups.
* ``decode_attn_headpar``, ``decode_attn_headpar_chunk_kv`` and their unrolled
  names – head-parallel decode entry points;
* ``decode_attn_headpar_batch_split`` and its unrolled name – decode with
  (batch, kv-head) folded onto one BH axis;
* ``prefill_attn_parallel``, ``prefill_attn_parallel_chunk_kv`` and
  ``prefill_attn_online_prefill`` – prefill entry points.

``mode="none"`` is the dense full-context reference used by the self-test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_benchmarks.common import custom_ops

from . import ops
from .cache import Qwen3GQAKVCache

NEG_INF = float("-inf")
DECODE_HEADPAR = "decode_attn_headpar"
DECODE_HEADPAR_CHUNK_KV = "decode_attn_headpar_chunk_kv"
DECODE_HEADPAR_CHUNK_KV_UNROLL = "decode_attn_headpar_chunk_kv_unroll"
DECODE_BATCH_SPLIT = "decode_attn_headpar_batch_split"
DECODE_BATCH_SPLIT_UNROLL = "decode_attn_headpar_batch_split_unroll"
DECODE_BATCH_SPLIT_PIPELINE = "decode_attn_headpar_batch_split_pipeline"
PREFILL_PARALLEL = "prefill_attn_parallel"
PREFILL_PARALLEL_CHUNK_KV = "prefill_attn_parallel_chunk_kv"
PREFILL_ONLINE = "prefill_attn_online_prefill"

IMPLS = (
    DECODE_HEADPAR,
    DECODE_HEADPAR_CHUNK_KV,
    DECODE_HEADPAR_CHUNK_KV_UNROLL,
    DECODE_BATCH_SPLIT,
    DECODE_BATCH_SPLIT_UNROLL,
    DECODE_BATCH_SPLIT_PIPELINE,
    PREFILL_PARALLEL,
    PREFILL_PARALLEL_CHUNK_KV,
    PREFILL_ONLINE,
)
DECODE_IMPLS = set(IMPLS[:6])
PREFILL_IMPLS = set(IMPLS[6:])
# These are the four implementations for which we keep independent parity /
# production-reference coverage.  The other four remain executable smoke paths.
GOLDEN_IMPLS = (PREFILL_ONLINE, PREFILL_PARALLEL, DECODE_BATCH_SPLIT, DECODE_HEADPAR)


@dataclass
class Config:
    hidden_size: int = 4096
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    rope_theta: float = 5e6
    ctx_len: int = 16384
    rms_norm_eps: float = 1e-6
    qk_norm: bool = True
    num_kv_blocks: int = 4
    skip_kv: bool = True
    mode: str = "kv"  # "none" (dense reference) | "kv"
    num_cores_per_device: int = 16
    impls: str = DECODE_BATCH_SPLIT
    q_block_size: int = 128
    q_head_block_chunk: int = 4
    chunk_kv: bool = False
    chunk_kv_size: int = 256
    chunk_kv_n: int = 2
    kv_block_unroll: int = 2


class RMSNorm(nn.Module):
    """Lowers to ``com.qti.aisw.onnx::CustomRMSNorm``."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.variance_epsilon

        def eager(x, weight):
            return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

        return custom_ops.call(ops.CustomRMSNorm, (x, self.weight), eager, shape=x.shape, attrs={"epsilon": float(eps)})


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _skip_future(start: int, position_ids: torch.Tensor) -> torch.Tensor:
    """start > every position (the original's ``(start > pos.max(-1)).all()``), as a float reduce."""
    return torch.tensor(start, device=position_ids.device) > position_ids.to(torch.float32).amax(dim=(0, 1))


def _merge(max_buf, sum_buf, out_buf) -> torch.Tensor:
    """Softmax merge of per-block (max, sum, out) partials."""
    max_stk, sum_stk, out_stk = torch.stack(max_buf), torch.stack(sum_buf), torch.stack(out_buf)
    m = max_stk.max(dim=0).values
    w = torch.exp(max_stk - m.unsqueeze(0))
    return (w.unsqueeze(-1) * out_stk).sum(dim=0) / (w * sum_stk).sum(dim=0).unsqueeze(-1)


class GQA(Qwen3GQAKVCache, nn.Module):
    """q/k/v_proj -> q_norm/k_norm -> RoPE -> cache scatter -> blocked attention -> o_proj."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = c = cfg
        H, NQH, NKVH, D = c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim
        if NQH % NKVH:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.num_kv_groups = NQH // NKVH
        self.scaling = D**-0.5
        self.q_proj = nn.Linear(H, NQH * D, bias=False)
        self.k_proj = nn.Linear(H, NKVH * D, bias=False)
        self.v_proj = nn.Linear(H, NKVH * D, bias=False)
        self.o_proj = nn.Linear(NQH * D, H, bias=False)
        self.q_norm = RMSNorm(D, c.rms_norm_eps) if c.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(D, c.rms_norm_eps) if c.qk_norm else nn.Identity()
        inv_freq = 1.0 / (c.rope_theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        emb = torch.outer(torch.arange(c.ctx_len, dtype=torch.float32), inv_freq)
        emb = torch.cat((emb, emb), dim=-1)
        self.register_buffer("_cos_table", emb.cos(), persistent=False)  # [ctx_len, D]
        self.register_buffer("_sin_table", emb.sin(), persistent=False)

    def rope_lookup(self, position_ids: torch.Tensor):
        idx = position_ids.long()
        return self._cos_table[idx].unsqueeze(1), self._sin_table[idx].unsqueeze(1)  # [B, 1, QL, D]

    def forward(self, hidden_states, position_ids, k_cache, v_cache, cos: Optional[torch.Tensor] = None,
                sin: Optional[torch.Tensor] = None):
        c = self.cfg
        B, QL, _ = hidden_states.shape
        prefill = QL > 1
        if c.mode == "kv" and prefill and c.impls not in PREFILL_IMPLS:
            raise NotImplementedError(f"{c.impls} is decode-only; use seq_len=1")
        if c.mode == "kv" and not prefill and c.impls not in DECODE_IMPLS:
            raise NotImplementedError(f"{c.impls} is prefill-only; use seq_len>1")
        batch_fold = not prefill and B > 1 and c.impls in {
            DECODE_BATCH_SPLIT, DECODE_BATCH_SPLIT_UNROLL, DECODE_BATCH_SPLIT_PIPELINE
        }

        q = self.q_proj(hidden_states).view(B, QL, c.num_attention_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, QL, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, QL, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if c.chunk_kv:
            n = c.chunk_kv_n
            q = q.unsqueeze(2).expand(B, c.num_attention_heads, n, QL, c.head_dim).reshape(B, c.num_attention_heads * n, QL, c.head_dim)
            k = k.unsqueeze(2).expand(B, c.num_key_value_heads, n, QL, c.head_dim).reshape(B, c.num_key_value_heads * n, QL, c.head_dim)
            v = v.unsqueeze(2).expand(B, c.num_key_value_heads, n, QL, c.head_dim).reshape(B, c.num_key_value_heads * n, QL, c.head_dim)
        if not prefill:  # decode looks RoPE up in-graph; prefill takes cos/sin as inputs
            cos, sin = self.rope_lookup(position_ids)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        if c.chunk_kv:
            k_cache = self.scatter_chunk(k_cache, position_ids, k[:, ::c.chunk_kv_n], chunk_n=c.chunk_kv_n, chunk_size=c.chunk_kv_size)
            v_cache = self.scatter_chunk(v_cache, position_ids, v[:, ::c.chunk_kv_n], chunk_n=c.chunk_kv_n, chunk_size=c.chunk_kv_size)
            q_attn = q[:, ::c.chunk_kv_n]
            if c.mode == "kv" and c.impls == DECODE_HEADPAR_CHUNK_KV_UNROLL and not prefill:
                # The unrolled path consumes the packed cache directly so its
                # physical gather schedule remains visible to the compiler.
                k_attn, v_attn = k_cache, v_cache
            else:
                k_attn = self.unpack_chunk(k_cache, ctx_len=c.ctx_len, chunk_n=c.chunk_kv_n, chunk_size=c.chunk_kv_size, num_kv_heads=c.num_key_value_heads)
                v_attn = self.unpack_chunk(v_cache, ctx_len=c.ctx_len, chunk_n=c.chunk_kv_n, chunk_size=c.chunk_kv_size, num_kv_heads=c.num_key_value_heads)
        else:
            write = self.scatter_batch if batch_fold else self.scatter
            k_cache = write(k_cache, position_ids, k)
            v_cache = write(v_cache, position_ids, v)
            q_attn, k_attn, v_attn = q, k_cache, v_cache

        if c.mode == "none":
            attn = self._dense(q_attn, k_attn, v_attn, position_ids)
        elif prefill:
            if c.impls in {PREFILL_PARALLEL, PREFILL_PARALLEL_CHUNK_KV}:
                attn = self._parallel_prefill(q_attn, k_attn, v_attn, position_ids)
            else:
                attn = self._online_prefill(q_attn, k_attn, v_attn, position_ids)
        else:
            if c.impls == DECODE_HEADPAR_CHUNK_KV_UNROLL:
                attn = self._decode_headpar_chunk_kv_unroll(q_attn, k_attn, v_attn, position_ids)
            elif c.impls == DECODE_BATCH_SPLIT_UNROLL:
                attn = self._decode_batch_split_unroll(q_attn, k_attn, v_attn, position_ids)
            elif c.impls == DECODE_BATCH_SPLIT:
                attn = self._decode_batch_split(q_attn, k_attn, v_attn, position_ids)
            elif c.impls in {DECODE_BATCH_SPLIT_PIPELINE}:
                attn = self._decode_batch_split_pipeline(q_attn, k_attn, v_attn, position_ids)
            else:
                attn = self._decode_headpar(q_attn, k_attn, v_attn, position_ids)
        attn = attn.transpose(1, 2).contiguous().view(B, QL, c.num_attention_heads * c.head_dim)
        return self.o_proj(attn), k_cache, v_cache

    def _decode_headpar(self, q, k_cache, v_cache, position_ids):
        """Head-parallel decode with the stable merged-cache ABI.

        Head-parallel blocked reduction.  Chunk-KV callers unpack their packed
        cache before reaching this common reduction, so the numerical kernel
        remains shared while the cache scatter/layout is variant-specific.
        """
        if q.shape[0] == 1:
            return self._decode_batch_split(q, k_cache, v_cache, position_ids)
        outputs = [
            self._decode_batch_split(q[b : b + 1], k_cache[b : b + 1], v_cache[b : b + 1], position_ids[b : b + 1])
            for b in range(q.shape[0])
        ]
        return torch.cat(outputs, dim=0)

    def _decode_headpar_chunk_kv_unroll(self, q, k_cache, v_cache, position_ids):
        """Pipelined head-parallel decode over the packed chunk-KV cache."""
        c = self.cfg
        B, NQH, QL, D = q.shape
        if QL != 1:
            raise ValueError("decode_attn_headpar_chunk_kv_unroll requires seq_len=1")
        if c.num_cores_per_device is None:
            raise ValueError("chunk-KV decode requires num_cores_per_device")

        N = c.chunk_kv_n
        split = c.num_cores_per_device
        Hkv_base = c.num_key_value_heads
        Hkv_eff = N * Hkv_base
        n_rep = NQH // Hkv_base
        p = c.chunk_kv_size // split
        nc = N * split
        pnc = p * nc
        # Each packed p-token window occupies p columns.  Use the physical
        # depth, not ceil(ctx_len / NC), so partial final windows are covered
        # as well (the common production shapes happen to be divisible).
        cache_depth = p * -(-c.ctx_len // pnc)
        num_blocks = max(1, c.num_kv_blocks)
        block_slots = -(-cache_depth // num_blocks)
        unroll = max(1, c.kv_block_unroll)
        exporting = torch.compiler.is_exporting()

        def merge_blocks(max_buf, sum_buf, out_buf):
            max_stk = torch.stack(max_buf)
            sum_stk = torch.stack(sum_buf)
            out_stk = torch.stack(out_buf)
            m1 = max_stk.max(dim=0).values
            w1 = torch.exp(max_stk - m1.unsqueeze(0))
            s1 = (w1 * sum_stk).sum(dim=0)
            o1 = (w1.unsqueeze(-1) * out_stk).sum(dim=0)
            m2 = m1.max(dim=2).values
            w2 = torch.exp(m1 - m2.unsqueeze(2))
            s2 = (w2 * s1).sum(dim=2)
            o2 = (w2.unsqueeze(-1) * o1).sum(dim=2)
            m2 = m2.reshape(1, Hkv_base, N, n_rep * QL)
            s2 = s2.reshape(1, Hkv_base, N, n_rep * QL)
            o2 = o2.reshape(1, Hkv_base, N, n_rep * QL, D)
            mf = m2.max(dim=2).values
            wf = torch.exp(m2 - mf.unsqueeze(2))
            sf = (wf * s2).sum(dim=2)
            of = (wf.unsqueeze(-1) * o2).sum(dim=2)
            return (of / sf.unsqueeze(-1)).view(1, Hkv_base, n_rep, QL, D).reshape(1, NQH, QL, D)

        outputs = []
        for b in range(B):
            q_b = q[b : b + 1]
            k_b = k_cache[b : b + 1]
            v_b = v_cache[b : b + 1]
            pos_b = position_ids[b : b + 1]
            q_grouped = q_b.reshape(1, Hkv_base, n_rep * QL, D)
            q_grouped = q_grouped.repeat_interleave(N, dim=1)
            q5 = q_grouped.unsqueeze(2).expand(1, Hkv_eff, split, n_rep * QL, D)
            current = pos_b.max(dim=-1).values

            def score_block(kb, start, tokens, skip):
                th = tokens // split
                k5 = kb.view(1, Hkv_eff, split, th, D)
                scores = torch.matmul(q5, k5.transpose(-1, -2)) * self.scaling
                slots = torch.arange(th, device=q.device).view(1, 1, 1, 1, th)
                lanes = torch.arange(split, device=q.device).view(1, 1, split, 1, 1)
                ways = torch.arange(N, device=q.device).view(1, N, 1, 1, 1)
                packed_slot = start + slots
                row_nc = ways * split + lanes
                logical_pos = (packed_slot // p) * pnc + row_nc * p + packed_slot % p
                causal = logical_pos > pos_b[:, None, None, :, None]
                scores = scores.view(1, Hkv_base, N, split, n_rep * QL, th)
                scores = scores.masked_fill(causal.unsqueeze(1), -3.0e4)
                scores = scores.view(1, Hkv_eff, split, n_rep * QL, th)
                m = scores.max(dim=-1).values
                e = torch.exp(scores - m.unsqueeze(-1))
                s = e.sum(dim=-1)
                if skip is not None and exporting:
                    m = torch.where(skip, torch.full_like(m, NEG_INF), m)
                    e = torch.where(skip, torch.zeros_like(e), e)
                    s = torch.where(skip, torch.zeros_like(s), s)
                return m, e, s

            max_buf, sum_buf, out_buf = [], [], []
            stopped = False
            for group in range(0, num_blocks, unroll):
                blocks = []
                for j in range(group, min(group + unroll, num_blocks)):
                    start = j * block_slots
                    slots = cache_depth - start if j == num_blocks - 1 else block_slots
                    end = start + slots
                    tokens = slots * split
                    skip = None
                    if c.skip_kv:
                        skip = torch.tensor((start // p) * pnc, device=q.device) > current
                        if not exporting and skip.item():
                            stopped = True
                            break
                    blocks.append((start, end, tokens, skip))
                k_reads = [
                    self.read_chunk_block(k_b, pos_b, start, end, chunk_n=N,
                                          chunk_size=c.chunk_kv_size, split=split,
                                          zero_invalid=False)
                    for start, end, _, _ in blocks
                ]
                exp_blocks = []
                for kb, (start, _, tokens, skip) in zip(k_reads, blocks):
                    m, e, s = score_block(kb, start, tokens, skip)
                    max_buf.append(m)
                    sum_buf.append(s)
                    exp_blocks.append(e)
                v_reads = [
                    self.read_chunk_block(v_b, pos_b, start, end, chunk_n=N,
                                          chunk_size=c.chunk_kv_size, split=split,
                                          zero_invalid=True)
                    for start, end, _, _ in blocks
                ]
                for e, vb, (_, _, tokens, skip) in zip(exp_blocks, v_reads, blocks):
                    out = torch.matmul(e, vb.view(1, Hkv_eff, split, tokens // split, D))
                    if skip is not None and exporting:
                        out = torch.where(skip, torch.zeros_like(out), out)
                    out_buf.append(out)
                if stopped:
                    break
            outputs.append(merge_blocks(max_buf, sum_buf, out_buf))
        return torch.cat(outputs, dim=0)

    def _parallel_prefill(self, q, k_cache, v_cache, position_ids):
        """Head-parallel prefill: split each KV block across ``split`` lanes."""
        c = self.cfg
        B, NQH, QL, D = q.shape
        Hkv = c.num_key_value_heads
        n_rep = self.num_kv_groups
        split = (c.num_cores_per_device // Hkv)
        block_size = -(-c.ctx_len // max(1, c.num_kv_blocks))
        if n_rep % c.q_head_block_chunk:
            raise ValueError("q_head_block_chunk must divide the number of Q heads per KV head")
        q_fold = q.reshape(B, Hkv, n_rep, QL, D)
        outputs = []
        for t0 in range(0, QL, c.q_block_size):
            t1 = min(t0 + c.q_block_size, QL)
            tc = t1 - t0
            pos = position_ids[:, t0:t1]
            th_nom = block_size // split
            offsets = (torch.arange(split, device=q.device)[:, None] * th_nom
                       + torch.arange(th_nom, device=q.device)[None, :]).repeat(Hkv, 1)
            accs = []
            for r0 in range(0, n_rep, c.q_head_block_chunk):
                rc = min(c.q_head_block_chunk, n_rep - r0)
                accs.append({
                    "rc": rc,
                    "q": q_fold[:, :, r0:r0 + rc, t0:t1].reshape(B, Hkv, rc * tc, D).repeat_interleave(split, dim=1),
                    "m": torch.full((B, Hkv * split, rc * tc), NEG_INF, dtype=q.dtype, device=q.device),
                    "s": torch.zeros(B, Hkv * split, rc * tc, dtype=q.dtype, device=q.device),
                    "o": torch.zeros(B, Hkv * split, rc * tc, D, dtype=q.dtype, device=q.device),
                })
            for start, end in self._blocks():
                T = end - start
                if T % split:
                    raise ValueError("KV block size must be divisible by num_cores_per_device/num_kv_heads")
                skip = _skip_future(start, pos) if c.skip_kv else None
                if skip is not None and not torch.compiler.is_exporting() and skip.item():
                    break
                kh = self.read_block(k_cache, position_ids, start, end, zero_invalid=False)
                vh = self.read_block(v_cache, position_ids, start, end, zero_invalid=True)
                kh = kh.view(B, Hkv * split, T // split, D)
                vh = vh.view(B, Hkv * split, T // split, D)
                causal = offsets[None, :, None, :] > (pos - start)[:, None, :, None]
                for a in accs:
                    rc = a["rc"]
                    mask = causal.repeat(1, 1, rc, 1) if rc > 1 else causal
                    scores = torch.matmul(a["q"], kh.transpose(-1, -2)) * self.scaling
                    scores = torch.where(mask, torch.tensor(NEG_INF, dtype=scores.dtype, device=scores.device), scores)
                    m_old, s_old, o_old = a["m"], a["s"], a["o"]
                    m_new = torch.maximum(m_old, scores.max(dim=-1).values)
                    scale = torch.exp(m_old - m_new)
                    e = torch.exp(scores - m_new.unsqueeze(-1))
                    s_new = s_old * scale + e.sum(dim=-1)
                    o_new = o_old * scale.unsqueeze(-1) + torch.matmul(e, vh)
                    # A split lane can be entirely future for one query while
                    # valid for another query in the same chunk.  Keep each
                    # flattened accumulator row independent.
                    lane = (~mask).any(dim=-1).logical_not().reshape(B, Hkv * split, rc * tc)
                    m_new = torch.where(lane, m_old, m_new)
                    s_new = torch.where(lane, s_old, s_new)
                    o_new = torch.where(lane.unsqueeze(-1), o_old, o_new)
                    a["m"], a["s"], a["o"] = m_new, s_new, o_new
            chunks = []
            for a in accs:
                rc = a["rc"]
                m = a["m"].view(B, Hkv, split, rc * tc)
                s = a["s"].view(B, Hkv, split, rc * tc)
                o = a["o"].view(B, Hkv, split, rc * tc, D)
                m2 = m.max(dim=2).values
                w = torch.exp(m - m2.unsqueeze(2))
                s2 = (w * s).sum(dim=2)
                o2 = (w.unsqueeze(-1) * o).sum(dim=2)
                chunks.append((o2 / s2.unsqueeze(-1)).view(B, Hkv, rc, tc, D))
            outputs.append(torch.cat(chunks, dim=2))
        return torch.cat(outputs, dim=3).reshape(B, NQH, QL, D)

    def _dense(self, q, k_cache, v_cache, position_ids):
        n_rep = self.num_kv_groups
        k = k_cache.repeat_interleave(n_rep, dim=1)
        v = v_cache.repeat_interleave(n_rep, dim=1)
        attn = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        future = torch.arange(self.cfg.ctx_len).view(1, 1, -1) > position_ids.unsqueeze(-1)
        attn = torch.where(future.unsqueeze(1), torch.tensor(NEG_INF, dtype=attn.dtype), attn)
        return torch.matmul(F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype), v)

    def _blocks(self):
        c = self.cfg
        size = -(-c.ctx_len // max(1, c.num_kv_blocks))
        for j in range(max(1, c.num_kv_blocks)):
            start = j * size
            yield start, (c.ctx_len if j == c.num_kv_blocks - 1 else start + size)

    def _online_prefill(self, q, k_cache, v_cache, position_ids):
        """Online softmax per Q chunk; each core owns one KV head (K/V repeated ``cores // Hkv``)."""
        c = self.cfg
        B, NQH, QL, D = q.shape
        cores = c.num_cores_per_device
        kv_repeat, rep_per_core, rc_max = cores // c.num_key_value_heads, NQH // cores, c.q_head_block_chunk
        if rep_per_core % rc_max:
            raise ValueError(f"q_head_block_chunk ({rc_max}) must divide NQH//num_cores ({rep_per_core})")
        exporting = torch.compiler.is_exporting()
        q_fold = q.reshape(B, cores, rep_per_core, QL, D)

        t_chunks = []
        for t0 in range(0, QL, c.q_block_size):
            t1 = min(t0 + c.q_block_size, QL)
            tc = t1 - t0
            pos_sub = position_ids[:, t0:t1]
            accs = []
            for r0 in range(0, rep_per_core, rc_max):
                rc = min(r0 + rc_max, rep_per_core) - r0
                accs.append({
                    "rc": rc,
                    "Q": q_fold[:, :, r0 : r0 + rc, t0:t1, :].reshape(B, cores, rc * tc, D),
                    "m": torch.full((B, cores, rc * tc), NEG_INF, dtype=q.dtype),
                    "s": torch.zeros(B, cores, rc * tc, dtype=q.dtype),
                    "o": torch.zeros(B, cores, rc * tc, D, dtype=q.dtype),
                })
            for start, end in self._blocks():
                skip = _skip_future(start, pos_sub) if c.skip_kv else None
                if skip is not None and not exporting and skip.item():
                    break
                kb = self.read_block(k_cache, position_ids, start, end, zero_invalid=False).repeat_interleave(kv_repeat, dim=1)
                vb = self.read_block(v_cache, position_ids, start, end, zero_invalid=True).repeat_interleave(kv_repeat, dim=1)
                causal = torch.arange(start, end)[None, None, None, :] > pos_sub[:, None, :, None]
                for acc in accs:
                    rc = acc["rc"]
                    attn = torch.matmul(acc["Q"], kb.transpose(2, 3)) * self.scaling
                    attn = attn.view(B, cores, rc, tc, -1).masked_fill(causal.unsqueeze(2), NEG_INF).view(B, cores, rc * tc, -1)
                    m_prev, s_prev, o_prev = acc["m"], acc["s"], acc["o"]
                    m_new = torch.max(m_prev, attn.max(dim=-1).values)
                    delta = m_prev - m_new
                    e = torch.exp(attn - m_new.unsqueeze(-1))
                    s_new = s_prev * torch.exp(delta) + e.sum(dim=-1)
                    o_new = o_prev * torch.exp(delta.unsqueeze(-1)) + torch.matmul(e, vb)
                    if skip is not None and exporting:
                        m_new = torch.where(skip, m_prev, m_new)
                        s_new = torch.where(skip, s_prev, s_new)
                        o_new = torch.where(skip.unsqueeze(-1), o_prev, o_new)
                    acc["m"], acc["s"], acc["o"] = m_new, s_new, o_new
            t_chunks.append(torch.cat([(a["o"] / a["s"].unsqueeze(-1)).view(B, cores, a["rc"], tc, D) for a in accs], dim=2))
        return torch.cat(t_chunks, dim=3).reshape(B, NQH, QL, D)

    def _decode_batch_split(self, q, k_cache, v_cache, position_ids):
        """Decode on the [1, B*Hkv, T, D] fold: per-block (max, sum, out) partials, merged once."""
        c = self.cfg
        B, NQH, QL, D = q.shape
        n_rep = self.num_kv_groups
        Hkv = NQH // n_rep
        BH = B * Hkv
        exporting = torch.compiler.is_exporting()
        k_flat = k_cache.reshape(1, BH, c.ctx_len, D)
        v_flat = v_cache.reshape(1, BH, c.ctx_len, D)
        q_flat = q.reshape(B, Hkv, n_rep, QL, D).reshape(1, BH, n_rep * QL, D)

        max_buf, sum_buf, out_buf = [], [], []
        for start, end in self._blocks():
            T = end - start
            skip = _skip_future(start, position_ids) if c.skip_kv else None
            if skip is not None and not exporting and skip.item():
                break
            kb = self.read_block_batch(k_flat, position_ids, start, end, zero_invalid=False)
            attn = torch.matmul(q_flat, kb.transpose(-1, -2)) * self.scaling
            causal = torch.arange(T).view(1, 1, T) > (position_ids - start)[:, :, None]
            causal = causal.unsqueeze(1).expand(B, Hkv, QL, T).reshape(1, BH, QL, T).repeat(1, 1, n_rep, 1)
            attn = attn.masked_fill(causal, -3.0e4)
            m_blk = attn.max(dim=-1).values
            e = torch.exp(attn - m_blk.unsqueeze(-1))
            s_blk = e.sum(dim=-1)
            if skip is not None and exporting:
                m_blk = torch.where(skip, torch.full_like(m_blk, NEG_INF), m_blk)
                e = torch.where(skip, torch.zeros_like(e), e)
                s_blk = torch.where(skip, torch.zeros_like(s_blk), s_blk)
            vb = self.read_block_batch(v_flat, position_ids, start, end, zero_invalid=True)
            o_blk = torch.matmul(e, vb)
            if skip is not None and exporting:
                s_blk = torch.where(skip, torch.zeros_like(s_blk), s_blk)
                o_blk = torch.where(skip, torch.zeros_like(o_blk), o_blk)
            max_buf.append(m_blk)
            sum_buf.append(s_blk)
            out_buf.append(o_blk)
        out = _merge(max_buf, sum_buf, out_buf)  # [1, BH, n_rep*QL, D]
        return out.reshape(B, Hkv, n_rep, QL, D).reshape(B, NQH, QL, D)

    def _decode_batch_split_unroll(self, q, k_cache, v_cache, position_ids):
        """Batch-split decode with grouped/pipelined KV-block processing.

        The ordinary path gathers K, computes the K-side scores, gathers V,
        and computes the value matmul for each block in sequence.  The
        standalone ``*_batch_unroll`` implementation instead groups
        ``kv_block_unroll`` blocks: all K gathers in a group are issued first,
        followed by all K-side score computations, then all V gathers, and
        finally all value matmuls.  This ordering is intentional; it gives the
        compiler an opportunity to overlap gather DMA with the preceding
        matmul while preserving the same numerics and cache ABI.
        """
        c = self.cfg
        B, NQH, QL, D = q.shape
        if QL != 1:
            raise ValueError("decode_attn_headpar_batch_split_unroll requires seq_len=1")

        n_rep = self.num_kv_groups
        Hkv = NQH // n_rep
        BH = B * Hkv
        unroll = max(1, c.kv_block_unroll)
        exporting = torch.compiler.is_exporting()

        k_flat = k_cache.reshape(1, BH, c.ctx_len, D)
        v_flat = v_cache.reshape(1, BH, c.ctx_len, D)
        q_flat = q.reshape(B, Hkv, n_rep, QL, D).reshape(1, BH, n_rep * QL, D)

        max_buf, sum_buf, out_buf = [], [], []
        stopped = False
        for group_start in range(0, max(1, c.num_kv_blocks), unroll):
            blocks = []
            for block in range(group_start, min(group_start + unroll, max(1, c.num_kv_blocks))):
                start, end = list(self._blocks())[block]
                skip = _skip_future(start, position_ids) if c.skip_kv else None
                if skip is not None and not exporting and skip.item():
                    stopped = True
                    break
                blocks.append((start, end, end - start, skip))

            # Issue all K gathers in this group before doing score matmuls.
            k_blocks = [
                self.read_block_batch(k_flat, position_ids, start, end, zero_invalid=False)
                for start, end, _, _ in blocks
            ]

            exp_blocks = []
            for kb, (start, _, T, skip) in zip(k_blocks, blocks):
                attn = torch.matmul(q_flat, kb.transpose(-1, -2)) * self.scaling
                causal = torch.arange(T, device=q.device).view(1, 1, T) > (
                    position_ids - start
                )[:, :, None]
                causal = (
                    causal.unsqueeze(1)
                    .expand(B, Hkv, QL, T)
                    .reshape(1, BH, QL, T)
                    .repeat(1, 1, n_rep, 1)
                )
                attn = attn.masked_fill(causal, -3.0e4)
                m_blk = attn.max(dim=-1).values
                e_blk = torch.exp(attn - m_blk.unsqueeze(-1))
                s_blk = e_blk.sum(dim=-1)
                if skip is not None and exporting:
                    m_blk = torch.where(torch.logical_not(skip), m_blk, torch.full_like(m_blk, NEG_INF))
                    e_blk = torch.where(skip, torch.zeros_like(e_blk), e_blk)
                    s_blk = torch.where(skip, torch.zeros_like(s_blk), s_blk)
                max_buf.append(m_blk)
                sum_buf.append(s_blk)
                exp_blocks.append(e_blk)

            # Issue all V gathers only after the group's K-side work.
            v_blocks = [
                self.read_block_batch(v_flat, position_ids, start, end, zero_invalid=True)
                for start, end, _, _ in blocks
            ]
            for e_blk, vb, (_, _, _, skip) in zip(exp_blocks, v_blocks, blocks):
                out_blk = torch.matmul(e_blk, vb)
                if skip is not None and exporting:
                    out_blk = torch.where(skip, torch.zeros_like(out_blk), out_blk)
                out_buf.append(out_blk)

            if stopped:
                break

        out = _merge(max_buf, sum_buf, out_buf)
        return out.reshape(B, Hkv, n_rep, QL, D).reshape(B, NQH, QL, D)

    def _decode_batch_split_pipeline(self, q, k_cache, v_cache, position_ids):
        """Pipelined decode: issues K[j+1] gather while V[j] gather and attention for block j are in flight."""
        c = self.cfg
        B, NQH, QL, D = q.shape
        n_rep = self.num_kv_groups
        Hkv = NQH // n_rep
        BH = B * Hkv
        exporting = torch.compiler.is_exporting()

        k_flat = k_cache.reshape(1, BH, c.ctx_len, D)
        v_flat = v_cache.reshape(1, BH, c.ctx_len, D)
        q_flat = q.reshape(B, Hkv, n_rep, QL, D).reshape(1, BH, n_rep * QL, D)

        blocks = list(self._blocks())
        last_block_start = blocks[-1][0]

        # Prologue: read first K block
        first_start, first_end = blocks[0]
        invalid_mask, ctx_indices = self.causal_mask(position_ids, BH, first_start, first_end, k_flat.device)
        kb = self.read_block_batch_with_mask(k_flat, invalid_mask, ctx_indices, first_start, first_end, zero_invalid=False)

        max_buf, sum_buf, out_buf = [], [], []

        for i, (start, end) in enumerate(self._blocks()):
            skip = _skip_future(start, position_ids) if c.skip_kv else None
            if skip is not None and not exporting and skip.item():
                break

            vb = self.read_block_batch_with_mask(v_flat, invalid_mask, ctx_indices, start, end, zero_invalid=True)

            attn = torch.matmul(q_flat, kb.transpose(-1, -2)) * self.scaling

            # Broadcast KV gather invalid mask from [1, BH, end-start] to [1, BH, n_rep, D] (assumes QL == 1)
            causal = invalid_mask.unsqueeze(2).repeat(1, 1, n_rep, 1)

            attn = attn.masked_fill(causal, -3.0e4)
            m_blk = attn.max(dim=-1).values
            e = torch.exp(attn - m_blk.unsqueeze(-1))
            s_blk = e.sum(dim=-1)
            if skip is not None and exporting:
                m_blk = torch.where(skip, torch.full_like(m_blk, NEG_INF), m_blk)
                e = torch.where(skip, torch.zeros_like(e), e)
                s_blk = torch.where(skip, torch.zeros_like(s_blk), s_blk)

            # Load next K block
            if start != last_block_start:
                next_start, next_end = blocks[i + 1]
                invalid_mask, ctx_indices = self.causal_mask(position_ids, BH, next_start, next_end, k_flat.device)
                kb = self.read_block_batch_with_mask(k_flat, invalid_mask, ctx_indices, next_start, next_end, zero_invalid=False)

            o_blk = torch.matmul(e, vb)
            if skip is not None and exporting:
                s_blk = torch.where(skip, torch.zeros_like(s_blk), s_blk)
                o_blk = torch.where(skip, torch.zeros_like(o_blk), o_blk)
            max_buf.append(m_blk)
            sum_buf.append(s_blk)
            out_buf.append(o_blk)

        out = _merge(max_buf, sum_buf, out_buf)  # [1, BH, n_rep*QL, D]
        return out.reshape(B, Hkv, n_rep, QL, D).reshape(B, NQH, QL, D)
