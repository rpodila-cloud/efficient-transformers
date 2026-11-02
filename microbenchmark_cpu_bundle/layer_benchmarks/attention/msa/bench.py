#!/usr/bin/env python3
"""MiniMax-M3 stacked GQA + MSA (sparse block) attention benchmark, decode and prefill."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch
from torch.export import Dim

from layer_benchmarks.common import cli, export, qaic

from .forwards import (
    PROFILES,
    Config,
    MiniMaxM3StackedDecodeModel,
)
from .ops import TRANSLATIONS

INT32_MAX = torch.iinfo(torch.int32).max


def _layer_cache_names(i: int, include_index: bool = False) -> list[str]:
    names = [f"key_cache_{i}", f"value_cache_{i}"]
    return names + [f"index_key_cache_{i}"] if include_index else names


def cache_names(num_gqa: int, num_msa: int) -> list[str]:
    names = [n for i in range(num_gqa) for n in _layer_cache_names(i)]
    return names + [n for j in range(num_msa) for n in _layer_cache_names(num_gqa + j, True)]


def io_names(c: Config, num_gqa: int, num_msa: int, seq_len: int):
    states = cache_names(num_gqa, num_msa)
    names = ["hidden_states", "position_ids", *states]
    if c.paged_kv and num_gqa:
        names.append("gqa_block_table")
    if c.paged_kv and num_msa:
        names.append("msa_indexer_block_table")
        if not c.indexer_only:
            names.append("msa_attn_block_table")
    if num_gqa and seq_len > 1:
        names += ["gqa_cos", "gqa_sin"]
    if num_msa:
        names += ["msa_cos", "msa_sin"]
    return names, ["attn_output", *[s + qaic.RETAINED_SUFFIX for s in states]]


def _gqa_cache_shape(c: Config, batch: int, seq_len: int):
    if c.paged_kv and seq_len > 1:
        return (c.num_physical_blocks, c.num_kv_heads, c.page_block_size, c.head_dim)
    if c.paged_kv:
        return (c.num_physical_blocks, c.attn_dp * c.attn_cp * c.num_kv_heads, c.page_block_size, c.head_dim)
    if seq_len == 1:
        return (batch // c.attn_dp, c.attn_dp * c.attn_cp * c.num_kv_heads, c.ctx_len // c.attn_cp, c.head_dim)
    return (batch, c.num_kv_heads, c.ctx_len, c.head_dim)


def _msa_cache_shape(c: Config, batch: int):
    rows = c.msa_attn_dp * c.msa_attn_cp * c.num_kv_heads
    if c.paged_kv:
        return (c.num_msa_attn_physical_blocks, rows, c.page_block_size, c.head_dim)
    return (batch // c.msa_attn_dp, rows, c.ctx_len // c.msa_attn_cp, c.head_dim)


def _index_cache_shape(c: Config, batch: int):
    rows = c.msa_indexer_dp * c.msa_indexer_cp * c.indexer_n_head
    if c.paged_kv:
        return (c.num_indexer_physical_blocks, rows, c.page_block_size, c.index_head_dim)
    return (batch // c.msa_indexer_dp, rows, c.ctx_len // c.msa_indexer_cp, c.index_head_dim)


def _block_table(c: Config, position_ids, dp: int, cp: int, num_physical_blocks: int, seed: int):
    """[dp, batch_local, page_groups] physical page ids, INT32_MAX for unused groups."""
    groups = (c.ctx_len // c.page_block_size + cp - 1) // cp
    batch_local = position_ids.shape[0] // dp
    table = torch.full((dp, batch_local, groups), INT32_MAX, dtype=torch.int32)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # The page mapping is shared.  [DP, B_local, page_groups] is only the
    # runtime table shape; it must not allocate a new physical-page mapping
    # for every batch or DP row.  CP lanes use the same page-group ID and
    # select their lane through logical_page % cp in the attention kernel.
    pages = torch.randperm(num_physical_blocks, generator=gen)[:groups].to(torch.int32)
    for d in range(dp):
        for b in range(batch_local):
            used = (int(position_ids[d * batch_local + b].max()) // c.page_block_size + 1 + cp - 1) // cp
            table[d, b, :used] = pages[:used]
    return table


def _gqa_prefill_block_table(c: Config, position_ids, seed: int):
    """[B, pages] physical page ids for paged GQA prefill, INT32_MAX for unused pages."""
    batch = position_ids.shape[0]
    table = torch.full((batch, c.ctx_len // c.page_block_size), INT32_MAX, dtype=torch.int32)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm, cursor = (None, 0) if c.random_slot_ids else (torch.randperm(c.num_physical_blocks, generator=gen), 0)
    for b in range(batch):
        used = int(position_ids[b].max()) // c.page_block_size + 1
        if c.random_slot_ids:
            table[b, :used] = torch.randperm(c.num_physical_blocks, generator=gen)[:used].to(torch.int32)
        else:
            table[b, :used] = perm[cursor:cursor + used].to(torch.int32)
            cursor += used
    return table


def _prefill_rope(c: Config, position_ids, dtype):
    inv_freq = 1.0 / c.rope_theta ** (
        torch.arange(0, c.rotary_dim, 2, dtype=torch.float32) / c.rotary_dim
    )
    freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype).unsqueeze(1), emb.sin().to(dtype).unsqueeze(1)


def inputs(c: Config, batch: int, seq_len: int, start: int, dtype: torch.dtype, num_gqa: int, num_msa: int):
    gen = torch.Generator(device="cpu").manual_seed(17)
    x = {
        "hidden_states": torch.randn(batch, seq_len, c.hidden_size, generator=gen, dtype=dtype),
        "position_ids": torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(batch, -1).clone() + start,
    }
    for i in range(num_gqa):
        for n in _layer_cache_names(i):
            x[n] = torch.zeros(_gqa_cache_shape(c, batch, seq_len), dtype=dtype)
    for j in range(num_msa):
        k, v, idx = _layer_cache_names(num_gqa + j, True)
        x[k] = torch.zeros(_msa_cache_shape(c, batch), dtype=dtype)
        x[v] = torch.zeros(_msa_cache_shape(c, batch), dtype=dtype)
        x[idx] = torch.zeros(_index_cache_shape(c, batch), dtype=dtype)
    pos = x["position_ids"]
    if c.paged_kv and num_gqa:
        x["gqa_block_table"] = (_gqa_prefill_block_table(c, pos, 303) if seq_len > 1 else
                                _block_table(c, pos, c.attn_dp, c.attn_cp, c.num_physical_blocks, 303))
    if c.paged_kv and num_msa:
        x["msa_indexer_block_table"] = _block_table(
            c, pos, c.msa_indexer_dp, c.msa_indexer_cp, c.num_indexer_physical_blocks, 305)
        if not c.indexer_only:
            x["msa_attn_block_table"] = _block_table(
                c, pos, c.msa_attn_dp, c.msa_attn_cp, c.num_msa_attn_physical_blocks, 307)
    if num_msa:
        x["msa_cos"], x["msa_sin"] = _prefill_rope(c, pos, dtype)
    if num_gqa and seq_len > 1:
        x["gqa_cos"], x["gqa_sin"] = _prefill_rope(c, pos, dtype)
    return x


def make_config(a: argparse.Namespace) -> Config:
    values = dict(PROFILES[a.model_profile])
    for key in ("hidden_size", "num_heads", "num_kv_heads", "head_dim", "rotary_dim", "rope_theta", "index_n_heads",
                "indexer_n_head", "index_head_dim", "index_block_size", "index_topk_blocks", "index_local_blocks"):
        if getattr(a, key) is not None:
            values[key] = getattr(a, key)
    c = Config(
        **values, impl=a.impl, ctx_len=a.ctx_len, num_kv_blocks=a.num_kv_blocks, msa_num_kv_blocks=a.msa_num_kv_blocks,
        indexer_num_blocks=a.indexer_num_blocks, q_block_size=a.q_block_size, q_block_chunk=a.q_block_chunk,
        msa_prefill_q_chunk=a.msa_prefill_q_chunk, indexer_ql_chunk=a.indexer_ql_chunk,
        indexer_q_block_size=a.indexer_q_block_size, q_head_block_chunk=a.q_head_block_chunk, attn_dp=a.attn_dp,
        attn_cp=a.attn_cp, paged_kv=a.paged_kv, random_slot_ids=a.random_slot_ids, page_block_size=a.page_block_size,
        kv_pool_gb=a.kv_pool_gb, msa_attn_dp=a.msa_attn_dp, msa_attn_cp=a.msa_attn_cp,
        msa_indexer_dp=a.msa_indexer_dp,
        msa_indexer_cp=a.msa_indexer_cp, msa_indexer_kv_pool_gb=a.msa_indexer_kv_pool_gb,
        indexer_prefill_parallel=a.indexer_prefill_parallel, num_cores_per_device=a.compile_num_cores,
        skip_kv=a.skip_kv,
    )
    return c.with_pools(a.num_gqa_layers)


def build(
    c: Config,
    num_gqa: int,
    num_msa: int,
    dtype: torch.dtype,
) -> MiniMaxM3StackedDecodeModel:
    torch.manual_seed(123)
    return MiniMaxM3StackedDecodeModel(c, num_gqa, num_msa).to(dtype=dtype).eval()


def onnx_name(c: Config, num_gqa: int, num_msa: int) -> str:
    impl = f"{c.impl}_paged" if c.paged_kv else c.impl
    return f"minimax_m3_{c.profile}_{impl}_{num_gqa}gqa_{num_msa}msa_decode.onnx"


def _page_decode_dynamic_shapes(model, x, input_names):
    """Return torch.export constraints for paged decode only.

    The model has a ``*caches`` forward argument, so the dynamic-shape
    structure must be ``(hidden, position, (cache_0, cache_1, ...))`` rather
    than a flat dict.  Using ``dynamic_shapes`` directly is required by the
    torch 2.13 exporter; passing the equivalent ONNX ``dynamic_axes`` makes
    its compatibility converter fail for this varargs signature.
    """
    if not (model.config.paged_kv and x["hidden_states"].shape[1] == 1):
        return None

    config = model.config
    max_dims = {
        "gqa_physical_pages": max(1, int(config.num_physical_blocks)),
        "attn_physical_pages": max(1, int(config.num_msa_attn_physical_blocks)),
        "indexer_physical_pages": max(1, int(config.num_indexer_physical_blocks)),
        "gqa_page_groups": max(1, int(x["gqa_block_table"].shape[2]))
        if "gqa_block_table" in x
        else 1,
        "indexer_page_groups": max(1, int(x["msa_indexer_block_table"].shape[2]))
        if "msa_indexer_block_table" in x
        else 1,
        "attn_page_groups": max(1, int(x["msa_attn_block_table"].shape[2]))
        if "msa_attn_block_table" in x
        else 1,
    }
    dim_registry = {
        name: Dim(name, min=1, max=max_value) for name, max_value in max_dims.items() if max_value > 1
    }

    def shape_spec(name: str) -> dict[int, Dim]:
        axes = {
            "gqa_block_table": {2: "gqa_page_groups"},
            "msa_indexer_block_table": {2: "indexer_page_groups"},
            "msa_attn_block_table": {2: "attn_page_groups"},
        }.get(name)
        if axes is None:
            axes = {}
            if name.startswith(("key_cache_", "value_cache_")):
                layer_idx = int(name.rsplit("_", 1)[1])
                physical_axis = (
                    "gqa_physical_pages" if layer_idx < model.num_gqa_layers else "attn_physical_pages"
                )
                axes = {0: physical_axis}
            elif name.startswith("index_key_cache_"):
                axes = {0: "indexer_physical_pages"}
        return {axis: dim_registry[dim_name] for axis, dim_name in axes.items() if dim_name in dim_registry}

    # ``input_names`` is ordered as forward(hidden_states, position_ids, *caches).
    specs = [shape_spec(name) for name in input_names]
    return (specs[0], specs[1], tuple(specs[2:]))


def _page_decode_page_dim_params(model, x, input_names, output_names):
    """Mark cache page-size axes symbolically after torch.export.

    The attention implementation uses the configured page size in Python
    reshapes, so asking torch.export to make that axis symbolic would collapse
    it back to the configured value. ``dim_params`` changes only the ONNX
    boundary metadata and keeps the page-size symbol visible to QAIC.
    """
    if not (model.config.paged_kv and x["hidden_states"].shape[1] == 1):
        return None

    params = {}
    cache_names = [
        name
        for name in (*input_names, *output_names)
        if name.startswith(("key_cache_", "value_cache_", "index_key_cache_"))
    ]
    for name in cache_names:
        params[name] = {2: "page_size"}
    return params or None


def export_onnx(model, x, names, path, *, custom_ops: bool = True, optimize: bool = True):
    in_names, out_names = names
    dynamic_shapes = _page_decode_dynamic_shapes(model, x, in_names)
    dim_params = _page_decode_page_dim_params(model, x, in_names, out_names)

    return export.export(model, [x[n] for n in in_names], path, in_names, out_names,
                         custom_translation_table=TRANSLATIONS, dynamic_shapes=dynamic_shapes,
                         dim_params=dim_params,
                         use_custom_ops=custom_ops, optimize=optimize)


def _page_decode_symbol_flags(model, x) -> list[str]:
    """Bind exported page symbols to this run's compile specialization."""
    if not (model.config.paged_kv and x["hidden_states"].shape[1] == 1):
        return []

    values: dict[str, int] = {}

    def add(name: str, tensor_name: str, axis: int) -> None:
        if tensor_name in x:
            values[name] = int(x[tensor_name].shape[axis])

    if model.num_gqa_layers:
        add("gqa_physical_pages", "key_cache_0", 0)
        add("page_size", "key_cache_0", 2)
        add("gqa_page_groups", "gqa_block_table", 2)
    if model.num_msa_layers:
        msa_layer = model.num_gqa_layers
        add("attn_physical_pages", f"key_cache_{msa_layer}", 0)
        add("indexer_physical_pages", f"index_key_cache_{msa_layer}", 0)
        add("page_size", f"key_cache_{msa_layer}", 2)
        add("indexer_page_groups", "msa_indexer_block_table", 2)
        add("attn_page_groups", "msa_attn_block_table", 2)

    return [f"-onnx-define-symbol={name},{value}" for name, value in values.items()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-profile", choices=tuple(PROFILES), default="m3")
    p.add_argument(
        "--impl", choices=["msa_attention_GP", "msa_attention_GP_batched", "msa_indexer_attention_GP_unified"],
        default="msa_attention_GP",
    )
    for name in ("hidden-size", "num-heads", "num-kv-heads", "head-dim", "rotary-dim", "index-n-heads",
                 "indexer-n-head", "index-head-dim", "index-block-size", "index-topk-blocks", "index-local-blocks"):
        p.add_argument(f"--{name}", type=int, default=None)
    p.add_argument("--rope-theta", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--ctx-len", type=int, default=4096)
    p.add_argument("--start-pos-id", type=int, default=0)
    p.add_argument("--num-gqa-layers", type=int, default=0)
    p.add_argument("--num-msa-layers", type=int, default=1)
    p.add_argument("--num-kv-blocks", type=int, default=1)
    p.add_argument("--msa-num-kv-blocks", type=int, default=None)
    p.add_argument("--indexer-num-blocks", type=int, default=None)
    p.add_argument("--q-block-size", type=int, default=64)
    p.add_argument("--q-block-chunk", type=int, default=None)
    p.add_argument("--msa-prefill-q-chunk", type=int, default=None)
    p.add_argument("--indexer-ql-chunk", type=int, default=None)
    p.add_argument("--indexer-q-block-size", type=int, default=None)
    p.add_argument("--q-head-block-chunk", type=int, default=1)
    p.add_argument("--attn-dp", type=int, default=1)
    p.add_argument("--attn-cp", type=int, default=1)
    p.add_argument("--msa-attn-dp", type=int, default=1)
    p.add_argument("--msa-attn-cp", type=int, default=1)
    p.add_argument("--msa-indexer-dp", type=int, default=1)
    p.add_argument("--msa-indexer-cp", type=int, default=1)
    p.add_argument("--paged-kv", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--random-slot-ids", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--page-block-size", type=int, default=128)
    p.add_argument("--kv-pool-gb", type=float, default=2.0)
    p.add_argument("--msa-indexer-kv-pool-gb", type=float, default=None)
    p.add_argument("--indexer-prefill-parallel", action=argparse.BooleanOptionalAction, default=True,
                   help="Parallel indexer block selection for MSA prefill (the only ported selector); decode ignores it.")
    p.add_argument("--skip-kv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    p.add_argument("--kv-cache-dtype", choices=["float16", "mxint8", "bfloat16"], default="float16")
    p.add_argument("--onnx-optimize", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable PyTorch ONNX-export optimization (default: enabled); does not control QAIC optimization.")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--hf-parity", action="store_true", help="Parity vs HF MiniMaxM3VLAttention, then exit.")
    p.add_argument("--hf-parity-tokens", type=int, default=64)
    cli.add_qaic_args(p, num_devices=1)
    a = p.parse_args()
    if a.hf_parity:
        return a
    if a.self_test:
        if a.artifact_dir is None:
            p.error("--artifact-dir is required with --self-test")
        a.artifact_dir = a.artifact_dir.resolve()
        a.artifact_dir.mkdir(parents=True, exist_ok=True)
        return a
    if a.kv_cache_dtype == "bfloat16" and a.dtype != "bfloat16":
        p.error("--kv-cache-dtype bfloat16 requires --dtype bfloat16 to preserve native BF16 cache IO")
    if a.start_pos_id + a.seq_len > a.ctx_len:
        p.error("start-pos-id + seq-len must not exceed ctx-len")
    if a.impl == "msa_indexer_attention_GP_unified" and a.num_msa_layers:
        if a.seq_len != 1 or a.paged_kv or a.msa_attn_cp <= 1 or a.msa_indexer_cp <= 1:
            p.error("unified GP requires non-paged decode with msa-attn-cp > 1 and msa-indexer-cp > 1")
    if a.seq_len > 1 and a.num_msa_layers and not a.indexer_prefill_parallel:
        p.error("MSA prefill is ported only for --indexer-prefill-parallel")
    cli.finalize_qaic_args(a)
    return a


def _numpy_io(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().contiguous()
    if tensor.dtype == torch.bfloat16:
        # NumPy has no native BF16 dtype; preserve the raw 16-bit representation.
        tensor = tensor.view(torch.uint16)
    return tensor.numpy()


def main() -> None:
    a = parse_args()
    if a.hf_parity:
        from .parity import hf_parity

        print(json.dumps(hf_parity(a.hf_parity_tokens), indent=2))
        return
    if a.self_test:
        from .parity import self_test

        print(json.dumps(self_test(a.artifact_dir), indent=2))
        return
    dtype = getattr(torch, a.dtype)
    c = make_config(a)
    g, m = a.num_gqa_layers, a.num_msa_layers
    model = build(c, g, m, dtype)
    x = inputs(c, a.batch_size, a.seq_len, a.start_pos_id, dtype, g, m)
    names = io_names(c, g, m, a.seq_len)
    with torch.no_grad():
        outputs = model(*(x[n] for n in names[0]))
    onnx_path = export_onnx(model, x, names, a.artifact_dir / onnx_name(c, g, m), optimize=a.onnx_optimize)
    report = {
        "config": asdict(c), "num_gqa_layers": g, "num_msa_layers": m, "batch_size": a.batch_size,
        "seq_len": a.seq_len, "start_pos_id": a.start_pos_id, "dtype": a.dtype,
        "kv_cache_dtype": a.kv_cache_dtype, "onnx_optimize": a.onnx_optimize,
        "inputs": {n: list(t.shape) for n, t in x.items()},
        "onnx": str(onnx_path), "onnx_ops": export.op_histogram(onnx_path),
        "attn_output_mean": float(outputs[0].float().mean()),
    }
    states = cache_names(g, m)
    visible_inputs = {n: _numpy_io(x[n]) for n in names[0] if n not in states}
    cli.run_qaic(
        a, report, onnx_path=onnx_path,
        specialization={"batch_size": a.batch_size, "seq_len": a.seq_len, "ctx_len": a.ctx_len},
        visible_inputs=visible_inputs, visible_outputs={"attn_output": _numpy_io(outputs[0])},
        # BF16 cache IO retains its ONNX type; the compiler rejects BF16 custom-IO overrides.
        custom_io=qaic.retained_custom_io(states, a.kv_cache_dtype) if a.kv_cache_dtype != "bfloat16" else None,
        compile_kwargs={
            "convert_to_fp16": a.dtype != "bfloat16",
            "mxfp6_matmul": False,
            "allow_mxint8_mdp_io": a.kv_cache_dtype == "mxint8",
            # Keep retained KV-cache inputs/outputs in every MDP partition's
            # binding set. Without this, qaic-qpc validate may show only the
            # visible attention inputs and omit the cache IO entirely.
            "extra_flags": [
                "-split-model-io",
                *_page_decode_symbol_flags(model, x),
            ],
            # QAIC's user-tiled scheduler currently asserts on this graph
            # when page-count/page-size symbols are present. Static prefill
            # graphs keep the normal user-tiled path.
            "user_tiled": not (c.paged_kv and a.seq_len == 1),
        },
    )
    cli.write_report(a, report)


if __name__ == "__main__":
    main()
