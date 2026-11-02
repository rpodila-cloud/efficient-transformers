#!/usr/bin/env python3
"""Expert-blocked NSP MoE FFN benchmark (Dynamo → QAIC).

    python -m layer_benchmarks.moe.bench --self-test --artifact-dir /tmp/moe
    python -m layer_benchmarks.moe.bench --impls decode_gather_bmm --batch-size 16 --T 1 --H 4096 --I 2048 \\
        --num-nsp 256 --num-experts-per-tok 6 --num-devices 16 --device-group 30..45 \\
        --artifact-dir out/moe --dump-io --run-compile --run-perf

See README.md for the impls, IO and verified numbers.
"""

from __future__ import annotations

import argparse

import numpy as np
import onnx
import torch

from layer_benchmarks.common import cli, export

from .forwards import (
    CumsumScatterGatherUpdate,
    CumsumScatterGatherUpdateWithRouter,
    GatherBmmDecode,
    GatherBmmLoopDecode,
    Reference,
    RoutedCumsumScatterGatherUpdate,
    build_router_inputs,
)

DECODE = "decode_gather_bmm"
DECODE_LOOP = "decode_gather_bmm_loop"
ROUTED = "decode_qwen3vl_cumsum_scatter_gather_update_with_router"
PREFILL = "cumsum_scatter_gather_update"
PREFILL_ROUTED = "cumsum_scatter_gather_update_with_router"
REFERENCE = "reference"
DECODE_IMPLS = (DECODE, DECODE_LOOP)
PREFILL_IMPLS = (REFERENCE, PREFILL, PREFILL_ROUTED)
IMPLS = PREFILL_IMPLS + (ROUTED,) + DECODE_IMPLS
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def build_local_masks(num_nsp, local_experts, seq_len, valid_rows):
    """Random [N, L, T] routing mask, ``valid_rows`` tokens per expert (original ``pattern=random``)."""
    masks = torch.zeros(num_nsp * local_experts, seq_len, dtype=torch.bool)
    for e in range(num_nsp * local_experts):
        masks[e, torch.randperm(seq_len)[:valid_rows]] = True
    return masks.view(local_experts, num_nsp, seq_len).transpose(0, 1).contiguous()


def build(a):
    """(module, example inputs, input names, dim_params, specialization) with the original's RNG order."""
    torch.manual_seed(a.seed)
    dtype = DTYPES[a.dtype]
    E = a.num_nsp * a.local_experts
    tb = a.ffn_token_block_size if getattr(a, "ffn_blocking_mode", "default") == "token" else None
    tokens = a.batch_size * a.T
    x = torch.randn(a.batch_size, a.T, a.H, dtype=dtype) if a.impls in (ROUTED, PREFILL_ROUTED) else \
        torch.randn(tokens if a.impls in DECODE_IMPLS else a.T, a.H, dtype=dtype)
    local_T2E = build_local_masks(a.num_nsp, a.local_experts, a.T, max(1, round(a.T * a.num_experts_per_tok / E)))
    if a.impls in DECODE_IMPLS:
        ri, rw = build_router_inputs(tokens, a.num_experts_per_tok, E, torch.float32)
        cls = GatherBmmDecode if a.impls == DECODE else GatherBmmLoopDecode
        m = cls(a.num_nsp, a.local_experts, a.H, a.I, torch.float32, a.num_experts_per_tok, a.shared_experts,
                token_block=tb)
        dims = {n: {0: "seq_len"} for n in ("x", "router_indices", "router_weights", "O")}
        return m, {"x": x.float(), "router_indices": ri, "router_weights": rw}, dims, {"seq_len": tokens}
    if a.impls == ROUTED:
        eps = a.experts_per_soc or (a.num_nsp // a.num_devices if a.num_devices > 1 else None)
        m = RoutedCumsumScatterGatherUpdate(a.num_nsp, a.local_experts, a.H, a.I, dtype, a.packed_chunk_size,
                                            a.num_experts_per_tok, eps, tb)
        return m, {"x": x}, {"x": {0: "batch_size", 1: "seq_len"}}, {"seq_len": a.T, "batch_size": a.batch_size}
    if a.impls == PREFILL_ROUTED:
        eps = a.experts_per_soc or (a.num_nsp // a.num_devices if a.num_devices > 1 else None)
        m = CumsumScatterGatherUpdateWithRouter(
            a.num_nsp, a.local_experts, a.H, a.I, dtype, a.packed_chunk_size,
            a.num_experts_per_tok, eps, tb,
        )
        return m, {"x": x}, {"x": {0: "batch_size", 1: "seq_len"}}, {"seq_len": a.T, "batch_size": a.batch_size}
    if a.impls == REFERENCE:
        m = Reference(a.num_nsp, a.local_experts, a.H, a.I, dtype)
        dims = {"x": {0: "seq_len"}, "local_T2E": {2: "seq_len"}, "O": {0: "seq_len"}}
        return m, {"x": x, "local_T2E": local_T2E}, dims, {"seq_len": a.T}
    m = CumsumScatterGatherUpdate(a.num_nsp, a.local_experts, a.H, a.I, dtype, a.packed_chunk_size, tb)
    dims = {"x": {0: "seq_len"}, "local_T2E": {2: "seq_len"}, "O": {0: "seq_len"}}
    return m, {"x": x, "local_T2E": local_T2E}, dims, {"seq_len": a.T}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impls", choices=IMPLS, default=PREFILL)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--T", type=int, default=512, help="Per-item seq_len; tokens = batch_size * T.")
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--I", type=int, default=128)
    p.add_argument("--num-nsp", type=int, default=16)
    p.add_argument("--local-experts", type=int, default=None, help="Default 1 for the routed impl, else 2.")
    p.add_argument("--num-experts-per-tok", type=int, default=8)
    p.add_argument("--shared-experts", type=int, default=0, help="Dense shared experts (decode_gather_bmm).")
    p.add_argument("--packed-chunk-size", type=int, default=256)
    p.add_argument("--experts-per-soc", type=int, default=None, help="Routed: default num_nsp // num_devices.")
    p.add_argument("--ffn-blocking-mode", choices=("default", "token"), default="default",
                   help="token: SwiGLU in --ffn-token-block-size row blocks (packed impls; no-op for decode, as in the original).")
    p.add_argument("--ffn-token-block-size", type=int, default=None)
    p.add_argument("--dtype", choices=tuple(DTYPES), default="float16")
    p.add_argument("--self-test", action="store_true", help="Eager/reference vs ORT (CPU, small shapes).")
    p.add_argument("--hf-parity", action="store_true", help="Eager vs HF GptOss / Qwen3-VL-MoE experts.")
    cli.add_qaic_args(p)
    a = p.parse_args()
    if a.local_experts is None:
        a.local_experts = 1 if a.impls == ROUTED else 2
    cli.finalize_qaic_args(a)
    return a


def main():
    a = parse_args()
    if a.self_test or a.hf_parity:
        from . import parity

        if a.self_test:
            parity.self_test(a.artifact_dir)
        if a.hf_parity:
            parity.hf_parity()
        return
    module, feed, dims, spec = build(a)
    onnx_path = export.export(module, tuple(feed.values()), a.artifact_dir / f"moe_{a.impls}.onnx", list(feed), ["O"],
                              dim_params=dims, optimize=False)
    hist = export.op_histogram(onnx_path)
    print(f"ONNX: {onnx_path}\n  ops: {hist}")
    with torch.no_grad():
        out = module(*feed.values())
    # The exported O can be fp32 even when the model runs fp16; size the runner buffer from the ONNX.
    o_type = onnx.load(str(onnx_path), load_external_data=False).graph.output[0].type.tensor_type.elem_type
    report = {"config": {k: v for k, v in vars(a).items() if k != "device_ids"}, "onnx_ops": hist}
    cli.run_qaic(
        a, report, onnx_path=onnx_path, specialization=spec,
        visible_inputs={k: v.numpy() for k, v in feed.items()},
        visible_outputs={"O": np.zeros(out.shape, dtype=onnx.helper.tensor_dtype_to_np_dtype(o_type))},
        compile_kwargs={"retained_state": False, "user_tiled": a.impls not in DECODE_IMPLS},
        perf_kwargs={"perf_metrics": False},  # the original compile_perf has no -aic-perf-metrics
    )
    cli.write_report(a, report)


if __name__ == "__main__":
    main()
