#!/usr/bin/env python3
"""DeepSeek-V4 CSA/HCA single-layer decode benchmark, folded DP/CP layout (Dynamo → QAIC).

    python -m layer_benchmarks.attention.csa.bench --self-test --artifact-dir /tmp/csa
    python -m layer_benchmarks.attention.hca.bench --self-test --artifact-dir /tmp/hca
    python -m layer_benchmarks.attention.csa.bench --start-pos-id 262143 --device-group 30..45 \\
        --artifact-dir out/csa --dump-io --run-compile --run-hw --run-perf

See README.md for the cache ABI, parallelism rules and verified numbers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

from layer_benchmarks.common import cli, export, qaic

from .forwards import HCA
from .forwards import DEFAULT_RATIO, CompressedAttention, PROFILES, HCA, Config

MODES = {"hca_decode": "hca"}
OVERRIDES = {
    "hidden_size": int, "num_heads": int, "head_dim": int, "rope_head_dim": int, "q_lora_rank": int, "o_groups": int,
    "o_lora_rank": int, "window_size": int, "rope_theta": float, "compress_rope_theta": float, "yarn_factor": float,
    "yarn_original_max_position_embeddings": int, "yarn_beta_fast": float, "yarn_beta_slow": float,
    "index_n_heads": int, "index_head_dim": int, "index_topk": int,
}


def state_names(mode: str):
    names = ["local_kv_cache", "compressed_kv_cache", "compressor_kv_state", "compressor_score_state"]
    return names + (["indexer_kv_cache", "indexer_kv_state", "indexer_score_state"] if mode == "csa" else [])


def io_names(mode: str):
    states = state_names(mode)
    return ["hidden_states", "position_ids", *states], ["attn_output", *[n + qaic.RETAINED_SUFFIX for n in states]]


def make_config(profile: str, mode: str, ctx_len: int, compress_ratio: int | None = None, **overrides) -> Config:
    values = dict(PROFILES[profile])
    values.update({k: v for k, v in overrides.items() if v is not None})
    ratio = compress_ratio if compress_ratio is not None else DEFAULT_RATIO[mode]
    return Config(profile=profile, attn_mode=mode, compress_ratio=ratio, ctx_len=ctx_len, **values)


def build(c: Config, dtype: torch.dtype, *, attn_dp: int, indexer_cp: int, num_kv_blocks: int, cores: int,
          hca_compressed_kv_cp: int = 1, hca_attn_blocks: int | None = None) -> CompressedAttention:
    torch.manual_seed(123)
    model_cls = HCA
    model = model_cls(c, attn_dp=attn_dp, indexer_cp=indexer_cp, num_kv_blocks=num_kv_blocks,
                      num_attention_cores=cores, hca_compressed_kv_cp=hca_compressed_kv_cp,
                      hca_attn_blocks=hca_attn_blocks)
    return model.to(dtype).eval()


def inputs(c: Config, batch: int, pos: int, dtype: torch.dtype, *, attn_dp: int, indexer_cp: int):
    """(hidden, position_ids, *states) with empty caches; compressor scores start at -1e4 (≈ -inf)."""
    g = torch.Generator().manual_seed(17)
    batch_local = batch // attn_dp
    comp_state = (1, batch, c.compressor_state_len, c.compressor_width)
    args = [
        torch.randn(batch, 1, c.hidden_size, generator=g, dtype=dtype),
        torch.full((batch, 1), pos, dtype=torch.int64),
        torch.zeros(1, batch, c.window_size, c.head_dim, dtype=dtype),
        torch.zeros(batch_local, attn_dp, c.comp_slots, c.head_dim, dtype=dtype),
        torch.zeros(comp_state, dtype=dtype),
        torch.full(comp_state, -1.0e4, dtype=dtype),
    ]
    if c.attn_mode == "csa":
        index_state = (batch, c.indexer_state_len, 2 * c.index_head_dim)
        slots = c.comp_slots // indexer_cp
        args += [
            torch.zeros(batch_local, attn_dp, indexer_cp * slots, c.index_head_dim, dtype=dtype),
            torch.zeros(index_state, dtype=dtype),
            torch.full(index_state, -1.0e4, dtype=dtype),
        ]
    return tuple(args)


def export_onnx(model: CompressedAttention, args, path, *, custom_ops: bool = True):
    input_names, output_names = io_names(model.config.attn_mode)
    return export.export(model, args, path, input_names, output_names, use_custom_ops=custom_ops)


def parse_args(mode_override: str | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=sorted(MODES), default=mode_override or "hca_decode")
    p.add_argument("--model-profile", choices=sorted(PROFILES), default="v4_flash")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ctx-len", type=int, default=262144)
    p.add_argument("--start-pos-id", type=int, help="Default: ctx_len - 1.")
    p.add_argument("--compress-ratio", type=int, help="Default: 4 (csa) / 128 (hca).")
    for name, kind in OVERRIDES.items():
        p.add_argument("--" + name.replace("_", "-"), type=kind)
    p.add_argument("--attn-dp", type=int, default=16)
    p.add_argument("--indexer-cp", type=int, default=16)
    p.add_argument("--num-kv-blocks", type=int, default=2)
    p.add_argument("--hca-compressed-kv-cp", type=int, default=1, help="HCA: CP ways folded into the compressed slot axis.")
    p.add_argument("--hca-attn-blocks", type=int, help="HCA: core tiles per CP way (default: core count).")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    p.add_argument("--kv-cache-dtype", choices=["float16"], default="float16")
    p.add_argument("--self-test", action="store_true", help="Smoke cases: eager vs ORT/reference, then exit.")
    p.add_argument("--hf-parity", action="store_true", help="fp32 decode loop vs HF DeepseekV4Attention, then exit.")
    cli.add_qaic_args(p, num_devices=16)
    a = p.parse_args()
    if mode_override is not None and a.mode != mode_override:
        p.error(f"this entry point only supports --mode {mode_override}")
    if a.self_test or a.hf_parity:
        if a.artifact_dir is None:
            p.error("--artifact-dir is required")
        a.artifact_dir = a.artifact_dir.resolve()
        a.artifact_dir.mkdir(parents=True, exist_ok=True)
        return a
    a.start_pos_id = a.ctx_len - 1 if a.start_pos_id is None else a.start_pos_id
    if not 0 <= a.start_pos_id < a.ctx_len:
        p.error("--start-pos-id must be < --ctx-len")
    if a.batch_size % a.attn_dp:
        p.error("--batch-size must be divisible by --attn-dp")
    if a.mode == "csa_decode" and (a.hca_compressed_kv_cp != 1 or a.hca_attn_blocks is not None):
        p.error("--hca-compressed-kv-cp / --hca-attn-blocks only apply to --mode hca_decode")
    cli.finalize_qaic_args(a)
    return a


def main(mode_override: str | None = None) -> None:
    a = parse_args(mode_override)
    if a.self_test or a.hf_parity:
        from . import parity

        active_mode = MODES[a.mode] if mode_override is not None else None
        result = parity.self_test(a.artifact_dir, mode=active_mode) if a.self_test else parity.hf_parity(mode=active_mode)
        print(json.dumps(result, indent=2))
        return

    mode = MODES[a.mode]
    c = make_config(a.model_profile, mode, a.ctx_len, a.compress_ratio, **{k: getattr(a, k) for k in OVERRIDES})
    cores = qaic.HW_CORES_PER_DEVICE[a.hw_version]
    if mode == "csa" and c.comp_slots % a.indexer_cp:
        raise SystemExit("compressed slot count must be divisible by --indexer-cp")
    hca_blocks = a.hca_attn_blocks or cores
    if mode == "hca" and (c.comp_slots % a.hca_compressed_kv_cp or (c.comp_slots // a.hca_compressed_kv_cp) % hca_blocks):
        raise SystemExit(f"HCA comp_slots={c.comp_slots} must split into {a.hca_compressed_kv_cp} CP ways x {hca_blocks} blocks")
    dtype = getattr(torch, a.dtype)
    model = build(c, dtype, attn_dp=a.attn_dp, indexer_cp=a.indexer_cp, num_kv_blocks=a.num_kv_blocks, cores=cores,
                  hca_compressed_kv_cp=a.hca_compressed_kv_cp, hca_attn_blocks=a.hca_attn_blocks)
    args = inputs(c, a.batch_size, a.start_pos_id, dtype, attn_dp=a.attn_dp, indexer_cp=a.indexer_cp)
    with torch.no_grad():
        attn_output = model(*args)[0]
    onnx_path = export_onnx(model, args, a.artifact_dir / f"deepseek_v4_{c.profile}_{mode}_parallel_decode.onnx")

    report = {
        "config": asdict(c),
        "parallel": {"attn_dp": a.attn_dp, "indexer_cp": a.indexer_cp, "num_kv_blocks": a.num_kv_blocks,
                     "num_attention_cores": cores, "hca_compressed_kv_cp": a.hca_compressed_kv_cp,
                     "hca_attn_blocks": hca_blocks},
        "dtype": a.dtype,
        "start_pos_id": a.start_pos_id,
        "cache_bytes": sum(t.numel() * t.element_size() for t in args[2:]),
        "attn_output_mean": float(attn_output.float().mean()),
        "onnx": str(onnx_path),
    }
    np_dtype = {"float16": np.float16, "float32": np.float32}.get(a.dtype, np.float16)
    cli.run_qaic(
        a,
        report,
        onnx_path=onnx_path,
        specialization={"batch_size": a.batch_size, "seq_len": 1, "ctx_len": a.ctx_len},
        visible_inputs={"hidden_states": args[0].float().numpy().astype(np_dtype), "position_ids": args[1].numpy()},
        visible_outputs={"attn_output": attn_output.float().numpy().astype(np_dtype)},
        custom_io=qaic.retained_custom_io(state_names(mode), a.kv_cache_dtype),
        compile_kwargs={"mxfp6_matmul": False},
    )
    cli.write_report(a, report)


if __name__ == "__main__":
    main("hca_decode")
