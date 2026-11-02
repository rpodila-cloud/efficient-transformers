#!/usr/bin/env python3
"""Qwen3 GQA KV-blocking single-layer benchmark (Dynamo → QAIC).

    python -m layer_benchmarks.attention.gqa.bench --self-test --artifact-dir /tmp/gqa
    python -m layer_benchmarks.attention.gqa.bench --impls decode_attn_headpar_batch_split --batch-size 64 \\
        --ctx-len 16384 --start-pos-id 12287 --num-devices 4 --device-group 30..33 \\
        --artifact-dir out/gqa --dump-io --run-compile --run-perf

See README.md for the cache ABI and verified numbers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from layer_benchmarks.common import cli, export, qaic

from .forwards import (DECODE_HEADPAR_CHUNK_KV, DECODE_HEADPAR_CHUNK_KV_UNROLL,
                       GQA, IMPLS, PREFILL_IMPLS, PREFILL_PARALLEL_CHUNK_KV,
                       PREFILL_ONLINE, Config)

STATES = ["k_cache", "v_cache"]


def io_names(prefill: bool):
    return ["hidden_states", "position_ids", *STATES] + (["cos", "sin"] if prefill else []), [
        "attn_out", *[n + qaic.RETAINED_SUFFIX for n in STATES]]


def build(c: Config, seed: int = 1234) -> GQA:
    torch.manual_seed(seed)
    return GQA(c).eval()


def inputs(model: GQA, batch: int, seq_len: int, start: int):
    """The original ``build_inputs`` order (global RNG after the module is built)."""
    c = model.cfg
    hidden = torch.randn(batch, seq_len, c.hidden_size)
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(batch, -1).clone() + start
    if c.chunk_kv:
        split = c.num_cores_per_device
        rows = c.num_key_value_heads * c.chunk_kv_n * split
        nc = c.chunk_kv_n * split
        p = c.chunk_kv_size // split
        # Keep enough columns for the final partial p-wide window when a
        # small smoke-test context is shorter than one packed window.
        depth = max((c.ctx_len + nc - 1) // nc, p)
        k = torch.randn(batch, rows, depth, c.head_dim)
        v = torch.randn(batch, rows, depth, c.head_dim)
    else:
        k = torch.randn(batch, c.num_key_value_heads, c.ctx_len, c.head_dim)
        v = torch.randn(batch, c.num_key_value_heads, c.ctx_len, c.head_dim)
    args = [hidden, position_ids, k, v]
    if seq_len > 1:
        args += list(model.rope_lookup(position_ids))
    return tuple(args)


def export_onnx(model: GQA, args, path, *, custom_ops: bool = True):
    input_names, output_names = io_names(args[0].shape[1] > 1)
    return export.export(model, args, path, input_names, output_names, use_custom_ops=custom_ops)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impls", choices=IMPLS, default="decode_attn_headpar_batch_split")
    p.add_argument("--attn-blocking-mode", choices=["none", "kv"], default="kv")
    p.add_argument("--attn-num-kv-blocks", type=int, default=4)
    p.add_argument("--skip-kv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--num-attention-heads", type=int, default=64)
    p.add_argument("--num-key-value-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--rope-theta", type=float, default=5e6)
    p.add_argument("--rms-norm-eps", type=float, default=1e-6)
    p.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--ctx-len", type=int, default=16384)
    p.add_argument("--start-pos-id", type=int, default=0)
    p.add_argument("--q-block-size", type=int, default=128)
    p.add_argument("--q-head-block-chunk", type=int, default=4)
    p.add_argument("--kv-block-unroll", type=int, default=2,
                   help="Number of KV blocks grouped by the unrolled implementation point.")
    p.add_argument("--chunk-kv-size", type=int, default=256,
                   help="Tokens per packed chunk for chunk-KV implementations.")
    p.add_argument("--chunk-kv-n", type=int, default=2,
                   help="Number of packed chunk ways for chunk-KV implementations.")
    p.add_argument("--num-cores-per-device", type=int, help="Default: --compile-num-cores.")
    p.add_argument("--kv-cache-dtype", choices=["float16", "mxint8"], default="mxint8")
    p.add_argument("--mxfp6-matmul", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--self-test", action="store_true", help="Parity vs the dense reference + ORT on small shapes, then exit.")
    p.add_argument("--hf-parity", action="store_true", help="Parity vs HF Qwen3Attention, then exit.")
    cli.add_qaic_args(p)
    a = p.parse_args()
    if a.artifact_dir is None and (a.self_test or a.hf_parity):
        p.error("--artifact-dir is required")
    if a.self_test or a.hf_parity:
        a.artifact_dir = a.artifact_dir.resolve()
        a.artifact_dir.mkdir(parents=True, exist_ok=True)
        return a
    if (a.impls in PREFILL_IMPLS) != (a.seq_len > 1):
        p.error(f"{a.impls} needs {'--seq-len > 1' if a.impls in PREFILL_IMPLS else '--seq-len 1'}")
    if a.start_pos_id + a.seq_len > a.ctx_len:
        p.error("start_pos_id + seq_len must not exceed ctx_len")
    cli.finalize_qaic_args(a)
    return a


def main() -> None:
    a = parse_args()
    if a.self_test or a.hf_parity:
        from .parity import hf_parity, self_test

        print(json.dumps(self_test(a.artifact_dir) if a.self_test else hf_parity(), indent=2))
        return

    c = Config(
        hidden_size=a.hidden_size, num_attention_heads=a.num_attention_heads, num_key_value_heads=a.num_key_value_heads,
        head_dim=a.head_dim, rope_theta=a.rope_theta, ctx_len=a.ctx_len, rms_norm_eps=a.rms_norm_eps, qk_norm=a.qk_norm,
        num_kv_blocks=a.attn_num_kv_blocks, skip_kv=a.skip_kv, mode=a.attn_blocking_mode,
        num_cores_per_device=a.num_cores_per_device or a.compile_num_cores, impls=a.impls,
        q_block_size=a.q_block_size, q_head_block_chunk=a.q_head_block_chunk,
        kv_block_unroll=a.kv_block_unroll,
        chunk_kv=a.impls in {DECODE_HEADPAR_CHUNK_KV, DECODE_HEADPAR_CHUNK_KV_UNROLL, PREFILL_PARALLEL_CHUNK_KV},
        chunk_kv_size=a.chunk_kv_size, chunk_kv_n=a.chunk_kv_n,
    )
    model = build(c, a.seed)
    args = inputs(model, a.batch_size, a.seq_len, a.start_pos_id)
    with torch.no_grad():
        attn_out = model(*args)[0]
    onnx_path = export_onnx(model, args, a.artifact_dir / f"qwen3_kv_blocking_{c.mode}_dynamic.onnx")

    report = {"config": asdict(c), "batch_size": a.batch_size, "seq_len": a.seq_len, "start_pos_id": a.start_pos_id,
              "kv_cache_dtype": a.kv_cache_dtype, "onnx": str(onnx_path), "onnx_ops": export.op_histogram(onnx_path)}
    input_names, _ = io_names(a.seq_len > 1)
    # Host IO stays fp32 / int32 as in the original; the compiler converts to fp16.
    visible = {n: t.numpy() for n, t in zip(input_names, args) if n not in STATES}
    cli.run_qaic(
        a,
        report,
        onnx_path=onnx_path,
        specialization={"batch_size": a.batch_size, "seq_len": a.seq_len, "kv_slots": a.ctx_len},
        visible_inputs=visible,
        visible_outputs={"attn_out": attn_out.numpy()},
        custom_io=qaic.retained_custom_io(STATES, a.kv_cache_dtype),
        compile_kwargs={"mxfp6_matmul": a.mxfp6_matmul, "allow_mxint8_mdp_io": a.kv_cache_dtype == "mxint8"},
    )
    cli.write_report(a, report)


if __name__ == "__main__":
    main()
