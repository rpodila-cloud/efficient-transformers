"""MiniMax-M3 GQA/MSA parity loops.

* ``self_test`` – tiny shapes for every ported path, each checked against:
  - an independent dense reference (below). It keeps logical ``[B, H, ctx, D]``
    caches and selects blocks as ``{current} ∪ top-(k-1) past blocks by
    max_t(q·k_index)``. It runs multi-step from position 0, with the retained
    state fed back, across index-block and page boundaries;
  - ORT on the export with custom ops decomposed.

  Every QAIC export is also checked for Sign/NonZero.
* ``hf_parity`` – GQA + MSA decode vs HF ``MiniMaxM3VLAttention`` (``transformers.models.minimax_m3_vl``)
  over the full sequence. It needs 1 KV head and 1 index head. HF picks one block set per query,
  from the max over index heads; the port, like the standalone, lets index head h pick the blocks
  for KV head h. The two agree only when there is one head of each.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from layer_benchmarks.common import custom_ops, parity
from layer_benchmarks.common.export import op_histogram

_OVERRIDES = ("hidden_size", "num_heads", "num_kv_heads", "head_dim", "rotary_dim", "rope_theta", "index_n_heads",
              "indexer_n_head", "index_head_dim", "index_block_size", "index_topk_blocks", "index_local_blocks")
_DEFAULTS = dict(model_profile="tiny", impl="msa_attention_GP", num_kv_blocks=1, msa_num_kv_blocks=None,
                 indexer_num_blocks=None, q_block_size=64, q_block_chunk=None, msa_prefill_q_chunk=None,
                 indexer_ql_chunk=None, indexer_q_block_size=None, q_head_block_chunk=1, attn_dp=1, attn_cp=1,
                 paged_kv=False, random_slot_ids=False, page_block_size=8, kv_pool_gb=2.0, msa_attn_dp=1,
                 msa_attn_cp=1, msa_indexer_dp=1, msa_indexer_cp=1, msa_indexer_kv_pool_gb=None,
                 indexer_prefill_parallel=False, compile_num_cores=2, skip_kv=True)
CASES = (
    dict(name="gqa_decode", g=1, m=0, batch=4, seq=1, steps=12, ctx_len=128, num_kv_blocks=2, attn_dp=4),
    dict(name="gqa_msa_decode", g=1, m=1, batch=4, seq=1, steps=44, ctx_len=128, num_kv_blocks=2, attn_dp=4,
         msa_attn_dp=4, msa_indexer_dp=4, msa_indexer_cp=2),
    dict(name="msa_paged_decode", g=0, m=1, batch=4, seq=1, steps=44, ctx_len=256, num_kv_blocks=2, paged_kv=True,
         msa_attn_cp=2, msa_indexer_cp=2, random_slot_ids=True, skip_kv=False, pool=128),
    dict(name="gqa_prefill", g=1, m=0, batch=1, seq=32, steps=2, ctx_len=128, num_kv_blocks=4, q_block_size=16,
         q_block_chunk=32),
    dict(name="msa_paged_prefill", g=0, m=1, batch=1, seq=32, steps=2, ctx_len=128, num_kv_blocks=2,
         msa_num_kv_blocks=2, indexer_num_blocks=2, q_block_size=32, q_block_chunk=32, msa_prefill_q_chunk=8,
         indexer_ql_chunk=32, indexer_q_block_size=32, paged_kv=True, indexer_prefill_parallel=True, pool=64),
    dict(name="gqa_msa_paged_decode", g=1, m=1, batch=4, seq=1, steps=44, ctx_len=256, num_kv_blocks=2,
         attn_cp=2, paged_kv=True, msa_attn_cp=2, msa_indexer_cp=2, random_slot_ids=True, skip_kv=False, pool=128),
    dict(name="gqa_msa_paged_prefill", g=1, m=1, batch=1, seq=32, steps=2, ctx_len=128, num_kv_blocks=4,
         msa_num_kv_blocks=2, indexer_num_blocks=2, q_block_size=16, q_block_chunk=32, msa_prefill_q_chunk=8,
         indexer_ql_chunk=32, indexer_q_block_size=32, paged_kv=True, indexer_prefill_parallel=True, pool=64),
)
ORT_CASES = ("gqa_msa_decode", "msa_paged_decode", "gqa_prefill", "msa_paged_prefill", "gqa_msa_paged_decode",
             "gqa_msa_paged_prefill")


def _config(case):
    from .bench import make_config

    kw = {k: v for k, v in case.items() if k not in ("name", "g", "m", "batch", "seq", "steps", "pool")}
    a = argparse.Namespace(**{**_DEFAULTS, **dict.fromkeys(_OVERRIDES), **kw, "num_gqa_layers": case["g"]})
    c = make_config(a)
    if c.paged_kv:
        c.num_physical_blocks = c.num_indexer_physical_blocks = c.num_msa_attn_physical_blocks = case["pool"]
    return c


# ---------------------------------------------------------------- dense reference (independent of forwards/cache)


def _rms(x, w, eps):
    return w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rope(x, pos, rotary_dim, theta):
    """x [B, H, S, D]; rotates the first min(rotary_dim, D) dims."""
    inv = 1.0 / theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    f = pos.float()[:, None, :, None] * inv
    emb = torch.cat((f, f), -1)[..., : x.shape[-1]]
    r, rest = x[..., : emb.shape[-1]], x[..., emb.shape[-1]:]
    a, b = r.chunk(2, -1)
    return torch.cat((r * emb.cos() + torch.cat((-b, a), -1) * emb.sin(), rest), -1)


def _heads(lin, x, n, d):
    return lin(x)[..., : n * d].view(*x.shape[:2], n, d).transpose(1, 2)


def _write(cache, pos, upd):
    for b in range(cache.shape[0]):
        cache[b, :, pos[b].long()] = upd[b]


def _attend(c, q, k_cache, v_cache, pos, allowed):
    """q [B, H, S, D]; allowed(b, h_kv, s) -> bool [ctx] token mask."""
    out = torch.zeros_like(q)
    rep = c.num_heads // c.num_kv_heads
    for b in range(q.shape[0]):
        for h in range(c.num_heads):
            for s in range(q.shape[2]):
                mask = allowed(b, h // rep, s)
                k, v = k_cache[b, h // rep, mask], v_cache[b, h // rep, mask]
                p = torch.softmax(q[b, h, s] @ k.T * c.head_dim**-0.5, -1)
                out[b, h, s] = p @ v
    return out


class Reference:
    """Dense logical-cache GQA + MSA stack, weights read from the port module."""

    def __init__(self, c, model, batch):
        self.c, self.model = c, model
        ctx = torch.arange(c.ctx_len)
        self.ctx = ctx
        kv = lambda d, h: torch.zeros(batch, h, c.ctx_len, d)  # noqa: E731
        self.gqa = [(kv(c.head_dim, c.num_kv_heads), kv(c.head_dim, c.num_kv_heads)) for _ in model.gqa_layers]
        self.msa = [(kv(c.head_dim, c.num_kv_heads), kv(c.head_dim, c.num_kv_heads),
                     kv(c.index_head_dim, c.indexer_n_head)) for _ in model.msa_layers]

    def _qkv(self, layer, x, pos):
        c = self.c
        q = _rms(_heads(layer.q_proj, x, c.num_heads, c.head_dim), layer.q_norm.weight, c.norm_eps)
        k = _rms(_heads(layer.k_proj, x, c.num_kv_heads, c.head_dim), layer.k_norm.weight, c.norm_eps)
        v = _heads(layer.v_proj, x, c.num_kv_heads, c.head_dim)
        return _rope(q, pos, c.rotary_dim, c.rope_theta), _rope(k, pos, c.rotary_dim, c.rope_theta), v

    def _out(self, layer, o):
        return layer.o_proj(o.transpose(1, 2).reshape(o.shape[0], o.shape[2], -1))

    def _selected(self, layer, ik_cache, x, pos):
        """[B, H_idx, S, ctx] token mask: {current block} ∪ top-(k-1) past blocks by block-max index score."""
        c, bs = self.c, self.c.index_block_size
        iq = _rms(_heads(layer.index_q_proj, x, c.index_n_heads, c.index_head_dim), layer.index_q_norm.weight,
                  c.norm_eps)
        ik = _rms(_heads(layer.index_k_proj, x, c.indexer_n_head, c.index_head_dim), layer.index_k_norm.weight,
                  c.norm_eps)
        iq, ik = _rope(iq, pos, c.rotary_dim, c.rope_theta), _rope(ik, pos, c.rotary_dim, c.rope_theta)
        _write(ik_cache, pos, ik)
        rep = c.index_n_heads // c.indexer_n_head
        scores = torch.einsum("bhsd,bhtd->bhst", iq, ik_cache.repeat_interleave(rep, 1))
        block = scores.view(*scores.shape[:3], c.num_blocks, bs).amax(-1)
        cur = (pos // bs).long()[:, None, :, None]
        ids = torch.arange(c.num_blocks)
        block = block.masked_fill(ids >= cur, float("-inf"))
        top = block.topk(c.selected_blocks - 1, -1)
        chosen = torch.zeros_like(block, dtype=torch.bool).scatter(-1, top.indices, top.values.isfinite())
        chosen |= ids == cur
        tokens = chosen.repeat_interleave(bs, -1)
        return tokens & (self.ctx <= pos[:, None, :, None])

    @torch.no_grad()
    def __call__(self, x, pos):
        c = self.c
        for layer, (kc, vc) in zip(self.model.gqa_layers, self.gqa):
            q, k, v = self._qkv(layer, x, pos)
            _write(kc, pos, k)
            _write(vc, pos, v)
            x = self._out(layer, _attend(c, q, kc, vc, pos, lambda b, h, s: self.ctx <= pos[b, s]))
        for layer, (kc, vc, ikc) in zip(self.model.msa_layers, self.msa):
            assert c.index_n_heads == c.num_kv_heads, "index head h selects blocks for kv head h"
            sel = self._selected(layer, ikc, x, pos)
            q, k, v = self._qkv(layer, x, pos)
            _write(kc, pos, k)
            _write(vc, pos, v)
            x = self._out(layer, _attend(c, q, kc, vc, pos, lambda b, h, s: sel[b, h, s]))
        return x


# ---------------------------------------------------------------- port loop


def _feeds(case, c, dtype=torch.float32):
    """Per-step port inputs from position 0; caches are replaced by the retained state as it runs."""
    from .bench import _block_table, _gqa_prefill_block_table, _prefill_rope, inputs

    g, m, batch, seq = case["g"], case["m"], case["batch"], case["seq"]
    x = inputs(c, batch, seq, 0, dtype, g, m)
    if c.paged_kv and m:
        # Allocate every page up front so later steps can write. Per-batch random_slot_ids draws can share
        # physical pages across batches (fine for one timed step, wrong over many), so use the shuffled
        # non-overlapping table here.
        full = torch.full((batch, 1), c.ctx_len - 1, dtype=torch.int32)
        tc = replace(c, random_slot_ids=False)
        x["msa_indexer_block_table"] = _block_table(tc, full, c.msa_indexer_dp, c.msa_indexer_cp,
                                                    c.num_indexer_physical_blocks, 305)
        x["msa_attn_block_table"] = _block_table(tc, full, c.msa_attn_dp, c.msa_attn_cp,
                                                 c.num_msa_attn_physical_blocks, 307)
    if c.paged_kv and g:
        full = torch.full((batch, 1), c.ctx_len - 1, dtype=torch.int32)
        tc = replace(c, random_slot_ids=False)
        x["gqa_block_table"] = (_gqa_prefill_block_table(tc, full, 303) if seq > 1 else
                                _block_table(tc, full, c.attn_dp, c.attn_cp, c.num_physical_blocks, 303))
    gen = torch.Generator().manual_seed(29)
    for step in range(case["steps"]):
        x["hidden_states"] = torch.randn(batch, seq, c.hidden_size, generator=gen, dtype=dtype)
        x["position_ids"] = (torch.arange(seq, dtype=torch.int32) + step * seq).expand(batch, -1).clone()
        if g and seq > 1:
            x["gqa_cos"], x["gqa_sin"] = _prefill_rope(c, x["position_ids"], dtype)
        yield x


@torch.no_grad()
def _case(case, out_dir: Path) -> dict:
    from .bench import build, cache_names, export_onnx, io_names

    c = _config(case)
    g, m = case["g"], case["m"]
    model = build(c, g, m, torch.float32)
    ref = Reference(c, model, case["batch"])
    names, out_names = io_names(c, g, m, case["seq"])
    states = cache_names(g, m)
    checks, last = [], None
    for step, x in enumerate(_feeds(case, c)):
        last = {n: t.clone() for n, t in x.items()}
        outs = model(*(x[n] for n in names))
        expected = ref(x["hidden_states"], x["position_ids"])
        checks.append(parity.compare(f"step{step}.attn_output vs dense", outs[0], expected, atol=1e-4, rtol=1e-3))
        for n, t in zip(states, outs[1:]):
            x[n] = t
    result = {"checks": checks}
    if case["name"] in ORT_CASES:
        with custom_ops.enabled(False):
            onnx_path = export_onnx(model, last, (names, out_names), out_dir / case["name"] / "ref.onnx",
                                    custom_ops=False)
        want = model(*(last[n] for n in names))
        got = parity.ort_run(onnx_path, {n: last[n].numpy() for n in names}, out_names)
        for n, t in zip(out_names, want):
            checks.append(parity.compare(f"ort.{n}", got[n], t, atol=2e-5, rtol=1e-4))
        ops = op_histogram(export_onnx(model, last, (names, out_names), out_dir / case["name"] / "qaic.onnx"))
        result["banned_ops"] = {k: v for k, v in ops.items() if k in ("Sign", "NonZero")}
        result["qaic_ops"] = ops
    result["passed"] = all(x["passed"] for x in checks) and not result.get("banned_ops")
    parity.require(result, f"msa {case['name']}")
    worst = max(checks, key=lambda x: x["max_abs"] if x["name"].endswith("dense") else -1)
    return {"passed": True, "steps": case["steps"], "max_abs_dense": worst["max_abs"],
            "max_abs_ort": max((x["max_abs"] for x in checks if x["name"].startswith("ort.")), default=None),
            "banned_ops": result.get("banned_ops")}


def self_test(out_dir: Path) -> dict:
    torch.set_num_threads(8)
    return {case["name"]: _case(case, out_dir) for case in CASES}


@torch.no_grad()
def hf_parity(tokens: int) -> dict:
    """Decode ``tokens`` steps through 1 GQA + 1 MSA layer and compare with HF full-sequence attention."""
    from transformers.models.minimax_m3_vl import configuration_minimax_m3_vl as hf_cfg
    from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as hf_mod

    from .bench import build, cache_names, io_names

    torch.set_num_threads(8)
    case = dict(name="hf", g=1, m=1, batch=1, seq=1, steps=tokens, ctx_len=128, num_kv_blocks=2, num_kv_heads=1,
                index_n_heads=1)
    c = _config(case)
    model = build(c, 1, 1, torch.float32)
    hc = hf_cfg.MiniMaxM3VLTextConfig(
        hidden_size=c.hidden_size, num_hidden_layers=2, num_attention_heads=c.num_heads,
        num_key_value_heads=c.num_kv_heads, head_dim=c.head_dim, rotary_dim=c.rotary_dim, rms_norm_eps=c.norm_eps,
        index_n_heads=c.index_n_heads, index_head_dim=c.index_head_dim, index_block_size=c.index_block_size,
        index_topk_blocks=c.index_topk_blocks, index_local_blocks=c.index_local_blocks,
        layer_types=["full_attention", "minimax_m3_sparse"],
        rope_parameters={"rope_type": "default", "rope_theta": c.rope_theta,
                         "partial_rotary_factor": c.rotary_dim / c.head_dim},
    )
    hc._attn_implementation = "eager"
    ref = [hf_mod.MiniMaxM3VLAttention(hc, i).eval() for i in range(2)]
    rope = hf_mod.MiniMaxM3VLRotaryEmbedding(hc)
    # HF norms scale by (1 + weight); the port stores the full scale.
    for r, layer in zip(ref, (model.gqa_layers[0], model.msa_layers[0])):
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(r, n).weight.copy_(getattr(layer, n).weight)
        for n in ("q_norm", "k_norm"):
            getattr(r, n).weight.copy_(getattr(layer, n).weight - 1)
        if r.indexer is not None:
            r.indexer.q_proj.weight.copy_(layer.index_q_proj.weight)
            r.indexer.k_proj.weight.copy_(layer.index_k_proj.weight)
            r.indexer.q_norm.weight.copy_(layer.index_q_norm.weight - 1)
            r.indexer.k_norm.weight.copy_(layer.index_k_norm.weight - 1)

    names, _ = io_names(c, 1, 1, 1)
    states = cache_names(1, 1)
    xs, outs = [], []
    for x in _feeds(case, c):
        xs.append(x["hidden_states"].clone())
        o = model(*(x[n] for n in names))
        outs.append(o[0])
        for n, t in zip(states, o[1:]):
            x[n] = t
    hidden = torch.cat(xs, 1)
    pos = torch.arange(tokens)[None]
    pe = rope(hidden, pos)
    causal = torch.ones(tokens, tokens, dtype=torch.bool).tril()[None, None]
    y = ref[0](hidden, pe, torch.zeros(causal.shape).masked_fill(~causal, torch.finfo(torch.float32).min),
               position_ids=pos)[0]
    y = ref[1](y, pe, causal, position_ids=pos)[0]
    check = parity.compare(f"{tokens} decode steps vs HF MiniMaxM3VLAttention", torch.cat(outs, 1), y,
                           atol=1e-4, rtol=1e-3)
    result = {"passed": check["passed"], "checks": [check]}
    parity.require(result, "msa hf_parity")
    return result
