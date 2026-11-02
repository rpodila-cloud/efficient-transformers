#!/usr/bin/env python3
"""Kimi-K2.5 MLA attention benchmark variants."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

from layer_benchmarks.common import cli, export, qaic

from .forwards import Config, MLA, build

STATES = ["compressed_kv.0", "k_pe.0"]


def io_names():
    return ["hidden_states", "position_ids", *STATES], ["attn_output", *[s + qaic.RETAINED_SUFFIX for s in STATES]]


def make_config(a: argparse.Namespace) -> Config:
    num_kv_blocks = a.attn_num_kv_blocks
    if num_kv_blocks is None:
        num_kv_blocks = 1
    par_num_split = a.par_num_split
    if par_num_split is None:
        par_num_split = a.repeat_kv_heads
    head_block_size = a.attn_head_block_size
    if head_block_size is None:
        head_block_size = a.num_attention_heads
    return Config(
        hidden_size=a.hidden_size, num_attention_heads=a.num_attention_heads, q_lora_rank=a.q_lora_rank,
        qk_rope_head_dim=a.qk_rope_head_dim, kv_lora_rank=a.kv_lora_rank, v_head_dim=a.v_head_dim,
        qk_nope_head_dim=a.qk_nope_head_dim, ctx_len=a.ctx_len, num_kv_heads=a.repeat_kv_heads,
        num_kv_blocks=num_kv_blocks, par_num_split=par_num_split, mode=a.attn_blocking_mode,
        head_block_size=head_block_size, mla_absorption=a.mla_absorption, mla_online=a.mla_online,
    )


def inputs(c: Config, batch: int, seq_len: int, start: int, dtype: torch.dtype = torch.float32):
    hidden = torch.randn(batch, seq_len, c.hidden_size, dtype=dtype)
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(batch, -1).clone() + start
    ckv = torch.zeros(batch, c.num_kv_heads, c.ctx_len, c.kv_lora_rank, dtype=dtype)
    kpe = torch.zeros(batch, c.num_kv_heads, c.ctx_len, c.qk_rope_head_dim, dtype=dtype)
    return hidden, position_ids, ckv, kpe


def export_onnx(model: MLA, args, path, *, custom_ops: bool = True):
    return export.export(model, args, path, *io_names(), use_custom_ops=custom_ops)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden-size", type=int, default=7168)
    p.add_argument("--num-attention-heads", type=int, default=64)
    p.add_argument("--q-lora-rank", type=int, default=1536)
    p.add_argument("--qk-rope-head-dim", type=int, default=64)
    p.add_argument("--kv-lora-rank", type=int, default=512)
    p.add_argument("--v-head-dim", type=int, default=128)
    p.add_argument("--qk-nope-head-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--ctx-len", type=int, default=32768)
    p.add_argument("--start-pos-id", type=int, default=0)
    p.add_argument("--mla-absorption", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mla-online", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--attn-blocking-mode",
                   choices=["none", "kv", "h", "par", "prefill_par", "prefill_par_online"],
                   default="none")
    p.add_argument("--attn-num-kv-blocks", type=int, default=None)
    p.add_argument("--attn-head-block-size", type=int, default=None)
    p.add_argument("--repeat-kv-heads", type=int, default=1)
    p.add_argument("--par-num-split", type=int, default=None)
    p.add_argument("--kv-cache-dtype", choices=["float16", "mxint8"], default="float16")
    p.add_argument("--self-test", action="store_true")
    cli.add_qaic_args(p, num_devices=16, num_cores=16)
    a = p.parse_args()
    if a.self_test:
        if a.artifact_dir is None:
            p.error("--artifact-dir is required with --self-test")
        a.artifact_dir = a.artifact_dir.resolve()
        a.artifact_dir.mkdir(parents=True, exist_ok=True)
        return a
    if a.start_pos_id + a.seq_len > a.ctx_len:
        p.error("start-pos-id + seq-len must not exceed ctx-len")
    if a.num_attention_heads % a.repeat_kv_heads:
        p.error("num-attention-heads must be divisible by repeat-kv-heads")
    if a.attn_blocking_mode in ("kv", "par", "prefill_par", "prefill_par_online"):
        if a.attn_num_kv_blocks is None or a.attn_num_kv_blocks <= 0:
            p.error("this blocking mode requires --attn-num-kv-blocks > 0")
    if a.attn_blocking_mode in ("par", "prefill_par", "prefill_par_online"):
        if a.par_num_split is None:
            a.par_num_split = a.repeat_kv_heads
        if a.par_num_split <= 0:
            p.error("--par-num-split must be positive")
    if a.attn_blocking_mode == "h" and (a.attn_head_block_size is None or a.attn_head_block_size <= 0):
        p.error("head blocking requires --attn-head-block-size > 0")
    cli.finalize_qaic_args(a)
    return a


def main() -> None:
    a = parse_args()
    if a.self_test:
        from .parity import self_test

        print(json.dumps(self_test(a.artifact_dir), indent=2))
        return
    c = make_config(a)
    torch.manual_seed(a.seed)
    model = build(c).eval()
    args = inputs(c, a.batch_size, a.seq_len, a.start_pos_id)
    with torch.no_grad():
        attn_output, ckv_out, kpe_out = model(*args)
    onnx_path = export_onnx(model, args, a.artifact_dir / "mla_par_ctx_dynamic.onnx")
    report = {
        "config": asdict(c), "batch_size": a.batch_size, "seq_len": a.seq_len,
        "start_pos_id": a.start_pos_id, "kv_cache_dtype": a.kv_cache_dtype,
        "onnx": str(onnx_path), "onnx_ops": export.op_histogram(onnx_path),
        "attn_output_mean": float(attn_output.float().mean()),
    }
    input_names, _ = io_names()
    visible_inputs = {n: t.detach().numpy() for n, t in zip(input_names, args) if n not in STATES}
    visible_outputs = {"attn_output": attn_output.detach().numpy()}
    cli.run_qaic(
        a, report, onnx_path=onnx_path,
        specialization={"batch_size": a.batch_size, "seq_len": a.seq_len, "ctx_len": a.ctx_len},
        visible_inputs=visible_inputs, visible_outputs=visible_outputs,
        custom_io=qaic.retained_custom_io(STATES, a.kv_cache_dtype),
        compile_kwargs={"mxfp6_matmul": True, "allow_mxint8_mdp_io": a.kv_cache_dtype == "mxint8"},
    )
    cli.write_report(a, report)


if __name__ == "__main__":
    main()
