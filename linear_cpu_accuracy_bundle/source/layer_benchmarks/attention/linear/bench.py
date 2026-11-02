#!/usr/bin/env python3
"""Qwen3.5 linear attention (GatedDeltaNet), one layer: chunk prefill or recurrent decode (Dynamo → QAIC).

    python -m layer_benchmarks.attention.linear.bench --self-test --artifact-dir /tmp/la
    python -m layer_benchmarks.attention.linear.bench --hf-parity --artifact-dir /tmp/la
    python -m layer_benchmarks.attention.linear.bench --path chunk --seq-len 2048 --device-group 30 \\
        --artifact-dir out/la_chunk --dump-io --run-compile --run-hw --run-perf

See README.md for the state ABI and verified numbers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from layer_benchmarks.common import cli, export, qaic

from .forwards import Config, LinearAttention

STATES = ("conv_state", "recurrent_state")


# The shared standard environment disables VTCM load hoisting. On this
# retained-state graph that makes QAIC replicate the weight set on every
# core, multiplying DDR traffic by roughly 6x. Keep the other standard
# settings, but leave hoisting enabled for linear attention.
LINEAR_COMPILER_ENV = {
    "QAIC_COMPILER_OPTS_UNSUPPORTED": "-aic-op-stats-verbosity 2 -aic-userdma-async=0 -aic-hmx-async=0",
}


def io_names(path: str):
    if path == "recurrent":
        return (["hidden_states", "position_ids", "conv_state", "recurrent_state"],
                ["attn_out", "conv_state_RetainedState", "recurrent_state_RetainedState"])
    return ["hidden_states", "position_ids"], ["attn_out", "recurrent_state_RetainedState"]


def build(c: Config, seed: int = 1234) -> LinearAttention:
    """Same seed order as the original bench: weights first, then inputs."""
    torch.manual_seed(seed)
    return LinearAttention(c).eval()


def inputs(c: Config, batch: int, seq_len: int, pos: int):
    args = [torch.randn(batch, seq_len, c.hidden_size),
            torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(batch, -1).clone() + pos]
    if c.path == "recurrent":
        conv_shape = ((batch, c.conv_kernel, c.conv_dim)
                      if c.conv_state_layout == "channel_last"
                      else (batch, c.conv_dim, c.conv_kernel))
        args += [torch.randn(*conv_shape),
                 torch.randn(batch, c.num_v_heads, c.head_k_dim, c.head_v_dim)]
    return tuple(args)


def export_onnx(model: LinearAttention, args, path, *, custom_ops: bool = True):
    input_names, output_names = io_names(model.c.path)
    # The recurrent graph is a one-token decode kernel, but keeping dynamic
    # batch/sequence boundary dimensions lets QAIC specialize the actual
    # shape instead of materializing the full static activation plan.  This is
    # especially important for retained-state graphs: the static Dynamo
    # export compiled correctly but inflated DDR traffic by ~6x.
    dynamic_axes = None
    if model.c.path == "recurrent":
        dynamic_axes = {
            "hidden_states": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "conv_state": {0: "batch_size"},
            "recurrent_state": {0: "batch_size"},
            "attn_out": {0: "batch_size", 1: "seq_len"},
            "conv_state_RetainedState": {0: "batch_size"},
            "recurrent_state_RetainedState": {0: "batch_size"},
        }
    return export.export(model, args, path, input_names, output_names,
                         dynamic_axes=dynamic_axes, use_custom_ops=custom_ops)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", choices=("chunk", "recurrent"), default="chunk")
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--num-key-heads", type=int, default=16)
    p.add_argument("--num-value-heads", type=int, default=32)
    p.add_argument("--key-head-dim", type=int, default=128)
    p.add_argument("--value-head-dim", type=int, default=128)
    p.add_argument("--conv-kernel", type=int, default=4)
    p.add_argument("--conv-state-layout", choices=("channel_first", "channel_last"), default="channel_first")
    p.add_argument("--head-block-size", type=int, default=0,
                   help="Decode recurrent-state head block. Zero disables blocking.")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--rms-norm-eps", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1, help="1 = decode (--path recurrent); >1 = prefill.")
    p.add_argument("--start-pos-id", type=int, default=0)
    p.add_argument("--state-dtype", choices=("float16", "mxfp6", "mxint8"), default="float16")
    p.add_argument("--conv-state-dtype", choices=("float16", "mxfp6", "mxint8"), default=None)
    p.add_argument("--recurrent-state-dtype", choices=("float16", "mxfp6", "mxint8"), default=None)
    p.add_argument("--mxfp6-matmul", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--self-test", action="store_true", help="Eager vs ORT on small shapes, then exit.")
    p.add_argument("--hf-parity", action="store_true", help="Eager vs HF Qwen3_5MoeGatedDeltaNet, then exit.")
    cli.add_qaic_args(p, num_devices=1, num_cores=8)
    a = p.parse_args()
    if a.self_test or a.hf_parity:
        if a.artifact_dir is None:
            p.error("--artifact-dir is required")
        a.artifact_dir = a.artifact_dir.resolve()
        a.artifact_dir.mkdir(parents=True, exist_ok=True)
        return a
    if a.num_value_heads % a.num_key_heads:
        p.error("--num-value-heads must be divisible by --num-key-heads")
    if a.path == "recurrent" and a.seq_len != 1:
        p.error("--path recurrent requires --seq-len 1")
    if a.conv_state_layout == "channel_last" and a.path != "recurrent":
        p.error("--conv-state-layout channel_last is decode-only")
    if a.head_block_size < 0:
        p.error("--head-block-size must be zero or positive")
    if a.head_block_size and a.path != "recurrent":
        p.error("--head-block-size is decode-only")
    if a.head_block_size and a.num_value_heads % a.head_block_size:
        p.error("--head-block-size must divide --num-value-heads")
    cli.finalize_qaic_args(a)
    return a


def main() -> None:
    a = parse_args()
    if a.self_test or a.hf_parity:
        from . import parity

        result = parity.self_test(a.artifact_dir) if a.self_test else parity.hf_parity()
        print(json.dumps(result, indent=2))
        return

    c = Config(path=a.path, hidden_size=a.hidden_size, num_k_heads=a.num_key_heads, num_v_heads=a.num_value_heads,
               head_k_dim=a.key_head_dim, head_v_dim=a.value_head_dim, conv_kernel=a.conv_kernel,
               rms_norm_eps=a.rms_norm_eps, chunk_size=a.chunk_size,
               conv_state_layout=a.conv_state_layout, head_block_size=a.head_block_size)
    model = build(c, a.seed)
    args = inputs(c, a.batch_size, a.seq_len, a.start_pos_id)
    with torch.no_grad():
        attn_out = model(*args)[0]
    onnx_path = export_onnx(model, args, a.artifact_dir / f"qwen3_linear_attn_{c.path}.onnx")

    conv_state_dtype = a.conv_state_dtype or a.state_dtype
    recurrent_state_dtype = a.recurrent_state_dtype or a.state_dtype
    report = {"config": asdict(c), "batch_size": a.batch_size, "seq_len": a.seq_len,
              "start_pos_id": a.start_pos_id, "conv_state_dtype": conv_state_dtype,
              "recurrent_state_dtype": recurrent_state_dtype, "mxfp6_matmul": a.mxfp6_matmul,
              "onnx": str(onnx_path), "onnx_ops": export.op_histogram(onnx_path)}
    visible_inputs = {"hidden_states": args[0].numpy(), "position_ids": args[1].numpy()}
    custom_io = None
    compile_kwargs = {"mxfp6_matmul": a.mxfp6_matmul}
    if c.path == "recurrent":
        custom_io = qaic.retained_custom_io(
            STATES,
            a.state_dtype,
            {"conv_state": conv_state_dtype, "recurrent_state": recurrent_state_dtype},
        )
        compile_kwargs["allow_mxint8_mdp_io"] = "mxint8" in (conv_state_dtype, recurrent_state_dtype)
    cli.run_qaic(
        a,
        report,
        onnx_path=onnx_path,
        specialization={"batch_size": a.batch_size, "seq_len": a.seq_len},
        visible_inputs=visible_inputs,
        visible_outputs={"attn_out": attn_out.numpy()},
        compiler_env=LINEAR_COMPILER_ENV,
        perf_compile_env=LINEAR_COMPILER_ENV,
        custom_io=custom_io,
        compile_kwargs=compile_kwargs,
    )
    cli.write_report(a, report)


if __name__ == "__main__":
    main()
