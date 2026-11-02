"""Qwen3 GQA parity loops.

* ``self_test`` – small shapes for both impls. It checks eager against:
  - the dense ``mode="none"`` reference;
  - ORT on the export with custom ops decomposed.

  Decode runs several steps with the retained caches fed back and staggered
  per-batch positions.
* ``hf_parity`` – dense reference vs HF ``Qwen3Attention`` (transformers 5.5.4 in the env)
  over a full causal prefill with weights copied.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from layer_benchmarks.common import parity

from .forwards import (
    DECODE_BATCH_SPLIT,
    DECODE_BATCH_SPLIT_UNROLL,
    DECODE_HEADPAR,
    DECODE_HEADPAR_CHUNK_KV,
    DECODE_HEADPAR_CHUNK_KV_UNROLL,
    GQA,
    PREFILL_ONLINE,
    PREFILL_PARALLEL,
    PREFILL_PARALLEL_CHUNK_KV,
    Config,
)

SMALL = dict(hidden_size=256, num_attention_heads=32, num_key_value_heads=4, head_dim=32, ctx_len=256, num_kv_blocks=4)
CASES = (
    # 8 cores: kv_repeat 2, 4 Q heads per core in chunks of 2; start 100 skips the last block.
    # All eight standalone implementation names are smoke-tested here.  The
    # four names in GOLDEN_IMPLS are the production/reference set; the other
    # four are intentionally not promoted to golden performance baselines.
    dict(name="prefill_online", impls=PREFILL_ONLINE, cores=8, q_block_size=16, q_head_block_chunk=2, batch=1,
         seq_len=40, starts=(100, 0, 216)),
    dict(name="prefill_parallel", impls=PREFILL_PARALLEL, cores=8, q_block_size=16, q_head_block_chunk=2, batch=1,
         seq_len=40, starts=(100, 0, 216)),
    dict(name="prefill_parallel_chunk_kv", impls=PREFILL_PARALLEL_CHUNK_KV, cores=8, q_block_size=16, q_head_block_chunk=2, batch=1,
         seq_len=40, starts=(100, 0, 216)),
    dict(name="decode_headpar", impls=DECODE_HEADPAR, cores=16, batch=1, seq_len=1,
         starts=(60, 61, 62, 130, 190)),
    dict(name="decode_headpar_chunk_kv", impls=DECODE_HEADPAR_CHUNK_KV, cores=16, batch=1, seq_len=1,
         starts=(60, 61, 62, 130, 190)),
    dict(name="decode_headpar_chunk_kv_unroll", impls=DECODE_HEADPAR_CHUNK_KV_UNROLL, cores=16, batch=1, seq_len=1,
         starts=(60, 61, 62, 130, 190)),
    dict(name="decode_batch_split", impls=DECODE_BATCH_SPLIT, cores=16, batch=4, seq_len=1, starts=(60, 61, 62, 130, 190)),
    dict(name="decode_batch_split_unroll", impls=DECODE_BATCH_SPLIT_UNROLL, cores=16, batch=4, seq_len=1,
         starts=(60, 61, 62, 130, 190)),
)


def _config(case) -> Config:
    return Config(**SMALL, impls=case["impls"], num_cores_per_device=case["cores"],
                  q_block_size=case.get("q_block_size", 128), q_head_block_chunk=case.get("q_head_block_chunk", 1),
                  chunk_kv=case["impls"] in {DECODE_HEADPAR_CHUNK_KV, DECODE_HEADPAR_CHUNK_KV_UNROLL, PREFILL_PARALLEL_CHUNK_KV})


@torch.no_grad()
def _steps(model, first, case):
    """Run each start position, feeding the retained caches back; returns per-step outputs."""
    from .bench import inputs

    hidden, pos, k, v, *rope = first
    outs = []
    for i, start in enumerate(case["starts"]):
        if i:
            torch.manual_seed(start)
            nxt = inputs(model, case["batch"], case["seq_len"], start)
            hidden, pos, rope = nxt[0], nxt[1], list(nxt[4:])
        if case["seq_len"] == 1:  # stagger decode positions per batch row
            pos = pos + torch.arange(case["batch"], dtype=torch.int32).view(-1, 1) * 3
        out, k, v = model(hidden, pos, k, v, *rope)
        outs.append({"attn_out": out, "k_cache": k, "v_cache": v, "_feed": (hidden, pos, *rope)})
    return outs


@torch.no_grad()
def _case(case, out_dir: Path) -> dict:
    from .bench import build, export_onnx, inputs

    c = _config(case)
    ours = build(c)
    dense = GQA(replace(c, mode="none")).eval()
    dense.load_state_dict(ours.state_dict())

    torch.manual_seed(7)
    first = inputs(ours, case["batch"], case["seq_len"], case["starts"][0])
    got = _steps(ours, first, case)
    ref_dense = _steps(dense, first, case)
    checks = []
    for i, (g, d) in enumerate(zip(got, ref_dense)):
        checks.append(parity.compare(f"step{i}.attn_out vs dense", g["attn_out"], d["attn_out"], atol=1e-5, rtol=1e-4))

    # ORT on the decomposed export, at the last step's inputs and the caches from the step before.
    hidden, pos, *rope = got[-1]["_feed"]
    k, v = (got[-2]["k_cache"], got[-2]["v_cache"]) if len(got) > 1 else first[2:4]
    args = (hidden, pos, k, v, *rope)
    onnx_path = export_onnx(ours, args, out_dir / case["name"] / "ref.onnx", custom_ops=False)
    from .bench import io_names

    in_names, out_names = io_names(case["seq_len"] > 1)
    ort = parity.ort_run(onnx_path, {n: t.numpy() for n, t in zip(in_names, args)}, out_names)
    for name, ref_t in zip(out_names, (got[-1]["attn_out"], got[-1]["k_cache"], got[-1]["v_cache"])):
        checks.append(parity.compare(f"ort.{name}", ort[name], ref_t, atol=2e-5, rtol=1e-4))
    from layer_benchmarks.common.export import op_histogram

    ops = op_histogram(export_onnx(ours, args, out_dir / case["name"] / "qaic.onnx"))
    banned = {k: v for k, v in ops.items() if k in ("Sign", "NonZero")}
    result = {"passed": all(x["passed"] for x in checks) and not banned, "checks": checks, "banned_ops": banned}
    parity.require(result, f"gqa {case['name']}")
    return {"passed": True, "max_abs": {x["name"]: x["max_abs"] for x in checks}, "qaic_ops": ops}


def self_test(out_dir: Path) -> dict:
    torch.set_num_threads(8)
    return {case["name"]: _case(case, out_dir) for case in CASES}


@torch.no_grad()
def hf_parity(tokens: int = 24) -> dict:
    """Dense prefill of ``tokens`` tokens from an empty cache vs HF Qwen3Attention."""
    from transformers import Qwen3Config
    from transformers.models.qwen3 import modeling_qwen3 as hf_qwen3

    from .bench import build

    c = Config(**{**SMALL, "ctx_len": tokens}, impls=PREFILL_ONLINE, num_cores_per_device=8, q_block_size=8,
               q_head_block_chunk=2)
    ours = build(c)
    ours.q_norm.weight.uniform_(0.5, 1.5)  # non-trivial norm weights
    ours.k_norm.weight.uniform_(0.5, 1.5)
    dense = GQA(replace(c, mode="none")).eval()
    dense.load_state_dict(ours.state_dict())
    hf_cfg = Qwen3Config(hidden_size=c.hidden_size, num_attention_heads=c.num_attention_heads,
                         num_key_value_heads=c.num_key_value_heads, head_dim=c.head_dim, rms_norm_eps=c.rms_norm_eps,
                         max_position_embeddings=tokens, rope_parameters={"rope_type": "default", "rope_theta": c.rope_theta},
                         attention_bias=False, attn_implementation="eager")
    ref = hf_qwen3.Qwen3Attention(hf_cfg, layer_idx=0).eval()
    ref.load_state_dict(ours.state_dict())
    rope = hf_qwen3.Qwen3RotaryEmbedding(hf_cfg)

    x = torch.randn(1, tokens, c.hidden_size)
    pos = torch.arange(tokens).view(1, -1)
    causal = torch.ones(tokens, tokens, dtype=torch.bool).tril()
    mask = torch.zeros(1, 1, tokens, tokens).masked_fill(~causal, torch.finfo(torch.float32).min)
    expected, _ = ref(x, rope(x, pos), mask)
    cache = torch.zeros(1, c.num_key_value_heads, tokens, c.head_dim)
    cos, sin = ours.rope_lookup(pos.to(torch.int32))
    checks = [
        parity.compare("dense vs hf", dense(x, pos.to(torch.int32), cache, cache, cos, sin)[0], expected, atol=2e-5, rtol=1e-4),
        parity.compare("online_prefill vs hf", ours(x, pos.to(torch.int32), cache, cache, cos, sin)[0], expected,
                       atol=2e-5, rtol=1e-4),
        parity.compare("rope cos vs hf", cos[:, 0], rope(x, pos)[0], atol=1e-6, rtol=0),
    ]
    result = {"passed": all(x["passed"] for x in checks), "checks": checks}
    parity.require(result, "gqa hf")
    return {x["name"]: x["max_abs"] for x in checks}
