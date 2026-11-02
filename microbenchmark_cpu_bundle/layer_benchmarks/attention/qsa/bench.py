#!/usr/bin/env python3
"""Qwen3.8-Flash-Next QSA single-layer decode/prefill benchmark (Dynamo → QAIC).

    python -m layer_benchmarks.attention.qsa.bench --profile tiny --ctx-len 64 --self-test --artifact-dir /tmp/qsa
    python -m layer_benchmarks.attention.qsa.bench --hf-parity --hf-parity-tokens 12
    python -m layer_benchmarks.attention.qsa.bench --start-pos-id 262143 --device-group 30..45 \\
        --artifact-dir out/qsa --dump-io --run-compile --run-hw --run-perf

See README.md for the cache ABI, parallelism rules and verified numbers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

from layer_benchmarks.common import cli, export, qaic

from .forwards import QSA, TINY, Config
from .ops import TRANSLATIONS


def state_names(c: Config):
    if c.split_cache_cp:
        return [f"key_cache_{i}" for i in range(c.attn_cp)] + [f"value_cache_{i}" for i in range(c.attn_cp)] + [
            "index_cache",
            "index_partial",
        ]
    return ["key_cache", "value_cache", "index_cache", "index_partial"]


def io_names(c: Config):
    states = state_names(c)
    inputs = ["hidden_states", "position_ids", *states] + (["block_table"] if c.paged_kv else [])
    return inputs, ["attention_output", *[n + qaic.RETAINED_SUFFIX for n in states]]


def paged_block_table(c: Config) -> torch.Tensor:
    """Deterministic [attn_dp, B_local, grouped_pages] table."""
    if c.context % (c.ratio * c.attn_cp):
        raise ValueError("paged QSA context must divide ratio*attn_cp")
    batch_local = c.batch // c.attn_dp
    grouped_pages = c.context // (c.ratio * c.attn_cp)
    ids = torch.arange(grouped_pages, dtype=torch.int32).view(1, 1, -1)
    ids = ids + torch.arange(batch_local, dtype=torch.int32).view(1, -1, 1) * grouped_pages
    return ids.expand(c.attn_dp, -1, -1).contiguous()


def kv_shape(c: Config):
    if c.paged_kv:
        # Physical four-token pages shared across DP lanes; one row per [DP, CP, KV head].
        blocks = (c.batch // c.attn_dp) * (c.context // (c.ratio * c.attn_cp))
        return (blocks, c.attn_dp * c.attn_cp * c.kv_heads, c.ratio, c.dim)
    if c.split_cache_cp:
        return (c.batch // c.attn_dp, c.attn_dp * c.kv_heads, c.context // c.attn_cp, c.dim)
    if c.cache_layout == "batch_major":
        return (c.batch, c.attn_cp * c.kv_heads, c.context // c.attn_cp, c.dim)
    return (c.batch // c.attn_dp, c.attn_dp * c.attn_cp * c.kv_heads, c.context // c.attn_cp, c.dim)


def inputs(c: Config, dtype: torch.dtype, pos: int = 0, empty: bool = False):
    alloc = torch.empty if empty else torch.zeros
    if c.seq_len == 1:
        position_ids = torch.full((c.batch,), pos, dtype=torch.int64)
    else:
        position_ids = torch.arange(c.seq_len, dtype=torch.int64)[None] + torch.full(
            (c.batch, 1), pos, dtype=torch.int64
        )
    args = [torch.randn(c.batch, c.seq_len, c.hidden, dtype=dtype), position_ids]
    n_kv = 2 * c.attn_cp if c.split_cache_cp else 2
    args += [alloc(*kv_shape(c), dtype=dtype) for _ in range(n_kv)]
    index_shape = (c.batch // c.indexer_dp, c.indexer_dp * c.indexer_cp, c.context // (c.ratio * c.indexer_cp), c.index_dim)
    args += [alloc(*index_shape, dtype=dtype), torch.zeros(c.batch, 1, c.index_dim, dtype=torch.float32)]
    if c.paged_kv:
        args.append(paged_block_table(c))
    return tuple(args)


def build(c: Config, dtype: torch.dtype, seed: int = 17) -> QSA:
    torch.manual_seed(seed)
    model = QSA(c).eval().to(dtype)
    # RoPE frequencies stay fp32 even for fp16 compute.
    model.inv_freq = 1 / (c.theta ** (torch.arange(0, c.rotary_dim, 2).float() / c.rotary_dim))
    return model


def export_onnx(model: QSA, args, path, *, custom_ops: bool = True):
    input_names, output_names = io_names(model.c)
    return export.export(
        model, args, path, input_names, output_names, custom_translation_table=TRANSLATIONS, use_custom_ops=custom_ops
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", choices=["qwen38", "tiny"], default="qwen38")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=1, help="query length; >1 selects prefill")
    p.add_argument("--ctx-len", type=int, default=262144)
    p.add_argument("--start-pos-id", type=int, help="Default: ctx_len - 1 (boundary).")
    p.add_argument("--indexer-dp", type=int, default=16)
    p.add_argument("--indexer-cp", type=int, default=1)
    p.add_argument("--attn-dp", type=int, default=16)
    p.add_argument("--attn-cp", type=int, default=1)
    p.add_argument("--num-cores-per-device", type=int)
    p.add_argument("--indexer-num-blocks", type=int, default=4)
    p.add_argument("--indexer-ql-chunk", type=int, default=512)
    p.add_argument("--indexer-q-block-size", type=int, default=512)
    p.add_argument("--indexer-topk-blocking", type=int, default=4)
    p.add_argument("--sparse-q-block-size", type=int, default=16)
    p.add_argument("--sparse-kv-num-blocks", type=int, default=8)
    p.add_argument("--skip-kv", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--paged-kv", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--cache-layout", choices=["dp_rows", "batch_major"], default="dp_rows")
    p.add_argument("--split-cache-cp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--legacy-attn-cp-gather",
        action="store_true",
        help="use the superseded cross-CP gather instead of owner-local attention",
    )
    p.add_argument("--dtype", choices=["float32", "float16"], help="Default: tiny=float32, qwen38=float16.")
    p.add_argument("--kv-cache-dtype", choices=["float16", "mxint8"], default="float16")
    p.add_argument("--mxfp6-matmul", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--self-test", action="store_true", help="Tiny recurrent ORT + dense-reference parity.")
    p.add_argument("--hf-parity", action="store_true", help="Parity vs HF Qwen4ExpTextAttention, then exit.")
    p.add_argument("--hf-parity-tokens", type=int, default=12)
    cli.add_qaic_args(p, num_devices=16)
    a = p.parse_args()
    a.dtype = a.dtype or ("float32" if a.profile == "tiny" else "float16")
    if a.num_cores_per_device is None:
        a.num_cores_per_device = a.compile_num_cores
    if a.hf_parity:
        return a
    if a.batch_size % a.indexer_dp or a.batch_size % a.attn_dp or a.ctx_len % (4 * a.indexer_cp) or a.ctx_len % a.attn_cp:
        p.error("DP must divide batch; context must divide 4*indexer_cp and attn_cp")
    if a.indexer_dp * a.indexer_cp != a.num_devices or a.attn_dp * a.attn_cp != a.num_devices:
        p.error("requires indexer_dp*indexer_cp == attn_dp*attn_cp == num_devices")
    if a.self_test and (a.profile != "tiny" or a.ctx_len > 256 or a.dtype != "float32"):
        p.error("--self-test requires --profile tiny, --ctx-len <= 256 and float32")
    if a.split_cache_cp and (a.attn_cp < 2 or a.paged_kv):
        p.error("--split-cache-cp requires non-paged attention CP >= 2")
    a.start_pos_id = a.ctx_len - a.seq_len if a.start_pos_id is None else a.start_pos_id
    if not 0 <= a.start_pos_id or a.start_pos_id + a.seq_len > a.ctx_len:
        p.error("start-pos-id + seq-len must be within the cache")
    if a.seq_len > 1:
        pooled_context = a.ctx_len // 4
        if a.paged_kv or a.split_cache_cp or a.cache_layout != "dp_rows":
            p.error("QSA prefill supports only non-paged dp_rows cache")
        if a.indexer_cp != 1 or a.attn_cp != 1:
            p.error("QSA prefill currently requires indexer-cp=1 and attn-cp=1")
        if a.start_pos_id % 4 or a.seq_len % 4:
            p.error("QSA prefill requires start-pos-id and seq-len divisible by 4")
        if pooled_context % a.indexer_num_blocks:
            p.error("context/ratio must be divisible by indexer-num-blocks")
        if a.indexer_q_block_size % a.num_cores_per_device:
            p.error("indexer-q-block-size must be divisible by num-cores-per-device")
        kv_heads = 2 if a.profile == "qwen38" else TINY["kv_heads"]
        if a.num_cores_per_device % kv_heads:
            p.error("num-cores-per-device must be divisible by kv-heads")
        if (
            a.indexer_ql_chunk % a.indexer_q_block_size
            or a.seq_len % a.indexer_q_block_size
            or a.indexer_q_block_size % a.indexer_topk_blocking
        ):
            p.error("invalid indexer prefill blocking")
        attention_q_divisor = a.num_cores_per_device // kv_heads
        if (
            a.seq_len % a.sparse_q_block_size
            or a.sparse_q_block_size % attention_q_divisor
        ):
            p.error("invalid sparse attention Q/core blocking")
        budget = 2048 if a.profile == "qwen38" else TINY["budget"]
        if a.sparse_kv_num_blocks > budget // 4 + 1:
            p.error("sparse-kv-num-blocks exceeds selected KV width")
    cli.finalize_qaic_args(a)
    return a


def main() -> None:
    a = parse_args()
    torch.set_num_threads(1)
    if a.hf_parity:
        from .parity import hf_parity

        print(json.dumps(hf_parity(a.hf_parity_tokens, 17, a.profile), indent=2))
        return

    c = Config(
        batch=a.batch_size,
        seq_len=a.seq_len,
        context=a.ctx_len,
        indexer_dp=a.indexer_dp,
        indexer_cp=a.indexer_cp,
        attn_dp=a.attn_dp,
        attn_cp=a.attn_cp,
        num_devices=a.num_devices,
        num_cores=a.num_cores_per_device,
        indexer_num_blocks=a.indexer_num_blocks,
        indexer_ql_chunk=a.indexer_ql_chunk,
        indexer_q_block_size=a.indexer_q_block_size,
        indexer_topk_blocking=a.indexer_topk_blocking,
        sparse_q_block_size=a.sparse_q_block_size,
        sparse_kv_num_blocks=a.sparse_kv_num_blocks,
        skip_kv=a.skip_kv,
        paged_kv=a.paged_kv,
        cache_layout=a.cache_layout,
        split_cache_cp=a.split_cache_cp,
        **(TINY if a.profile == "tiny" else {}),
    )
    dtype = getattr(torch, a.dtype)
    model = build(c, dtype)
    model.use_owner_local_attention_cp = not a.legacy_attn_cp_gather
    # Hardware graphs specialise the pooling residue of start_pos_id.
    specialized = c.seq_len == 1 and not a.self_test and (a.run_compile or a.run_hw)
    if specialized:
        model.position_mode = (a.start_pos_id + 1) % c.ratio == 0
    args = inputs(c, dtype, a.start_pos_id, empty=True)
    graph_kind = "decode" if c.seq_len == 1 else "prefill"
    onnx_path = export_onnx(
        model,
        args,
        a.artifact_dir / f"qwen38_qsa_{graph_kind}.onnx",
        custom_ops=not a.self_test,
    )

    report = {
        "config": asdict(c),
        "dtype": a.dtype,
        "kv_cache_dtype": a.kv_cache_dtype,
        "mxfp6_matmul": a.mxfp6_matmul,
        "position_mode": {True: "boundary", False: "nonboundary", None: "dynamic"}[model.position_mode],
        "owner_local_attention_cp": model.use_owner_local_attention_cp,
        "graph_kind": graph_kind,
        "cache_bytes": sum(t.numel() * t.element_size() for t in args[2:]),
        "onnx": str(onnx_path),
    }
    if a.self_test:
        from .parity import prefill_self_test, self_test

        report["verification"] = (
            self_test(model, onnx_path, inputs)
            if c.seq_len == 1
            else prefill_self_test(model, onnx_path, inputs, a.start_pos_id)
        )
        print(json.dumps(report["verification"], indent=2))

    input_names, _ = io_names(c)
    precision = a.kv_cache_dtype if a.kv_cache_dtype == "mxint8" else ("float16" if a.dtype == "float16" else "float")
    custom_io = qaic.retained_custom_io(
        state_names(c),
        precision,
        {"index_partial": "float"},
    )
    np_dtype = np.float16 if a.dtype == "float16" else np.float32
    visible = {"hidden_states": args[0].numpy().astype(np_dtype, copy=False), "position_ids": args[1].numpy()}
    if c.paged_kv:
        visible["block_table"] = args[-1].numpy()
    cli.run_qaic(
        a,
        report,
        onnx_path=onnx_path,
        specialization={"batch_size": a.batch_size, "seq_len": c.seq_len, "ctx_len": a.ctx_len},
        visible_inputs=visible,
        visible_outputs={
            "attention_output": np.zeros(
                (a.batch_size, c.seq_len, c.hidden), dtype=np_dtype
            )
        },
        custom_io=custom_io,
        compile_kwargs={
            "mxfp6_matmul": a.mxfp6_matmul,
            "allow_mxint8_mdp_io": a.kv_cache_dtype == "mxint8",
            # Retained caches are explicit model IO; keep the split for MDP placement.
            "extra_flags": ["-split-model-io"],
        },
    )
    if "hardware" in report:
        report["hardware"]["attention_rows_per_second"] = (
            c.batch * c.seq_len * 1000 / report["hardware"]["mean_ms"]
        )
    cli.write_report(a, report)


if __name__ == "__main__":
    main()
