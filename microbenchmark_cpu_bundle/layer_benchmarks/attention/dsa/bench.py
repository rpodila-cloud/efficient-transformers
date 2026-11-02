#!/usr/bin/env python3
"""GLM-5 DSA (indexer Top-K + sparse absorbed MLA) attention benchmark, decode and prefill."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch
from torch import nn

from layer_benchmarks.common import cli, export, qaic

from .forwards import DSAConfig, build_module, get_attention_config
from .ops import TRANSLATIONS

INDEXER_CACHE = "indexer_key_cache"
ATTN_CACHES = ["compressed_kv.0", "k_pe.0"]


class DsaExportWrapper(nn.Module):
    """Flat-tensor boundary: (hidden, pos, [ckv, k_pe,] indexer) -> flat outputs."""

    def __init__(self, wrapped: nn.Module, indexer_only: bool):
        super().__init__()
        self.wrapped = wrapped
        self.indexer_only = indexer_only

    def forward(self, *flat):
        if self.indexer_only:
            hidden_states, position_ids, indexer_key_cache = flat
            return self.wrapped(hidden_states=hidden_states, position_ids=position_ids,
                                indexer_key_cache=indexer_key_cache)
        hidden_states, position_ids, ckv, k_pe, indexer_key_cache = flat
        out, present, indexer = self.wrapped(hidden_states=hidden_states, position_ids=position_ids,
                                             compressed_kvs=[[ckv, k_pe]], indexer_key_cache=indexer_key_cache)
        return (out, *present[0], indexer)


def cache_names(indexer_only: bool) -> list[str]:
    return [INDEXER_CACHE] if indexer_only else [*ATTN_CACHES, INDEXER_CACHE]


def io_names(indexer_only: bool):
    states = cache_names(indexer_only)
    ins = ["hidden_states", "position_ids", *states]
    if indexer_only:
        return ins, ["attn_output", INDEXER_CACHE + qaic.RETAINED_SUFFIX, "topk_indices", "valid_topk"]
    return ins, ["attn_output", *[s + qaic.RETAINED_SUFFIX for s in states]]


def inputs(config, d: DSAConfig, batch: int, seq_len: int, ctx_len: int, start: int, dtype=torch.float32):
    gen = torch.Generator(device="cpu").manual_seed(17)
    x = {
        "hidden_states": torch.randn(batch, seq_len, config.hidden_size, generator=gen, dtype=dtype),
        "position_ids": torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch, -1) + start,
    }
    if not d.indexer_only:
        rows, local = d.attention_dp * d.attention_cp, ctx_len // d.attention_cp
        x["compressed_kv.0"] = torch.zeros(batch // d.attention_dp, rows, local, config.kv_lora_rank, dtype=dtype)
        x["k_pe.0"] = torch.zeros(batch // d.attention_dp, rows, local, config.qk_rope_head_dim, dtype=dtype)
    x[INDEXER_CACHE] = torch.zeros(batch // d.indexer_dp, d.indexer_dp * d.indexer_cp, ctx_len // d.indexer_cp,
                                   config.index_head_dim, dtype=dtype)
    return x


def make_dsa_config(a: argparse.Namespace, config) -> DSAConfig:
    return DSAConfig(
        enabled=True, indexer_only=a.indexer_only, topk=config.index_topk, index_head_dim=config.index_head_dim,
        index_n_heads=config.index_n_heads, indexer_dp=a.indexer_dp, indexer_cp=a.indexer_cp,
        indexer_kvp=a.indexer_kvp, attention_dp=a.attn_dp, attention_cp=a.attn_cp, attention_kvp=a.attn_kvp,
        indexer_num_blocks=a.indexer_num_blocks, index_block_size=a.index_block_size,
        indexer_ql_chunk=a.indexer_ql_chunk, indexer_q_block_size=a.indexer_q_block_size,
        indexer_topk_blocking=a.indexer_topk_blocking, indexer_prefill_parallel=a.indexer_prefill_parallel,
        sparse_q_block_size=a.sparse_q_block_size, sparse_kv_num_blocks=a.sparse_kv_num_blocks,
        num_cores=a.num_cores_per_device, num_devices=a.num_devices,
    )


def onnx_name(profile: str, d: DSAConfig) -> str:
    suffix = "_indexer_only" if d.indexer_only else ""
    return (f"dsa_{profile}_idx_dp{d.indexer_dp}_cp{d.indexer_cp}_attn_dp{d.attention_dp}_cp{d.attention_cp}_"
            f"topk{d.topk}{suffix}.onnx")


def export_onnx(wrapper, x, names, path, *, custom_ops: bool = True):
    in_names, out_names = names
    return export.export(wrapper, [x[n] for n in in_names], path, in_names, out_names,
                         custom_translation_table=TRANSLATIONS, use_custom_ops=custom_ops)


def _validate(p: argparse.ArgumentParser, a: argparse.Namespace, config) -> None:
    if a.start_pos_id + a.seq_len > a.ctx_len or a.ctx_len <= a.seq_len:
        p.error("need seq-len < ctx-len and start-pos-id + seq-len <= ctx-len")
    if a.indexer_kvp != 1 or a.attn_kvp != 1:
        p.error("MLA retains one compressed stream: --indexer-kvp and --attn-kvp must be 1")
    if a.indexer_dp * a.indexer_cp != a.num_devices:
        p.error("indexer DP*CP must equal --num-devices")
    if not a.indexer_only and a.attn_dp * a.attn_cp != a.num_devices:
        p.error("attention DP*CP must equal --num-devices")
    if a.batch_size % a.indexer_dp or (not a.indexer_only and a.batch_size % a.attn_dp):
        p.error("--batch-size must be divisible by the indexer/attention DP")
    if a.index_block_size % a.indexer_cp:
        p.error("--index-block-size must be divisible by --indexer-cp")
    if (a.ctx_len // a.indexer_cp) % (a.indexer_num_blocks * a.num_cores_per_device):
        p.error("ctx_len/indexer_cp must be divisible by indexer_num_blocks*num_cores_per_device")
    if config.index_topk > a.ctx_len:
        p.error("--dsa-topk must be <= ctx_len")
    if not a.indexer_only and config.index_topk % a.num_cores_per_device:
        p.error("--dsa-topk must be divisible by --num-cores-per-device")
    if a.seq_len > 1 and not a.indexer_prefill_parallel:
        p.error("DSA prefill is ported only for --indexer-prefill-parallel")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-profile", choices=["glm5", "tiny"], default="glm5")
    p.add_argument("--dsa-topk", type=int, default=None)
    p.add_argument("--indexer-only", action="store_true", help="Indexer Top-K only (no sparse attention).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--ctx-len", type=int, default=4096)
    p.add_argument("--start-pos-id", type=int, default=None, help="Default ctx_len - seq_len.")
    p.add_argument("--max-position-embeddings", type=int, default=None)
    for name, default in (("indexer-dp", 1), ("indexer-cp", 16), ("indexer-kvp", 1), ("attn-dp", 16),
                          ("attn-cp", 1), ("attn-kvp", 1), ("indexer-num-blocks", 16), ("index-block-size", 128),
                          ("indexer-ql-chunk", 128), ("indexer-q-block-size", 32), ("indexer-topk-blocking", 16),
                          ("sparse-q-block-size", 128), ("sparse-kv-num-blocks", 1)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--indexer-prefill-parallel", action=argparse.BooleanOptionalAction, default=True,
                   help="Parallel prefill indexer (the only ported prefill selector); decode ignores it.")
    p.add_argument("--num-cores-per-device", type=int, default=None, help="Default --compile-num-cores.")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                   help="Target precision; the graph is exported in FP32 and -convert-to-fp16 applies it.")
    p.add_argument("--kv-cache-dtype", choices=["float16", "mxint8"], default="float16")
    p.add_argument("--mxfp6-matmul", "--enable-mxfp6", dest="mxfp6_matmul",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--self-test", action="store_true")
    cli.add_qaic_args(p, num_devices=16)
    a = p.parse_args()
    if a.self_test:
        if a.artifact_dir is None:
            p.error("--artifact-dir is required with --self-test")
        a.artifact_dir = a.artifact_dir.resolve()
        a.artifact_dir.mkdir(parents=True, exist_ok=True)
        return a
    if a.num_cores_per_device is None:
        a.num_cores_per_device = a.compile_num_cores
    if a.start_pos_id is None:
        a.start_pos_id = a.ctx_len - a.seq_len
    config = get_attention_config(a.model_profile)
    if a.dsa_topk is not None:
        config.index_topk = a.dsa_topk
    _validate(p, a, config)
    cli.finalize_qaic_args(a)
    a.config = config
    return a


def main() -> None:
    a = parse_args()
    if a.self_test:
        from .parity import self_test

        print(json.dumps(self_test(a.artifact_dir), indent=2))
        return
    config = a.config
    if a.max_position_embeddings is not None:
        config.max_position_embeddings = a.max_position_embeddings
    config.max_position_embeddings = max(config.max_position_embeddings, a.ctx_len)
    d = make_dsa_config(a, config)
    torch.manual_seed(123)
    module = build_module(config, d)
    wrapper = DsaExportWrapper(module, d.indexer_only).eval()
    x = inputs(config, d, a.batch_size, a.seq_len, a.ctx_len, a.start_pos_id)
    names = io_names(d.indexer_only)
    with torch.no_grad():
        outputs = dict(zip(names[1], wrapper(*(x[n] for n in names[0]))))
    onnx_path = export_onnx(wrapper, x, names, a.artifact_dir / onnx_name(a.model_profile, d))
    states = cache_names(d.indexer_only)
    report = {
        "config": asdict(config), "dsa_config": asdict(d), "batch_size": a.batch_size, "seq_len": a.seq_len,
        "ctx_len": a.ctx_len, "start_pos_id": a.start_pos_id, "kv_cache_dtype": a.kv_cache_dtype,
        "inputs": {n: list(t.shape) for n, t in x.items()},
        "onnx": str(onnx_path), "onnx_ops": export.op_histogram(onnx_path),
        "attn_output_mean": float(outputs["attn_output"].float().mean()),
    }
    visible_out = ["attn_output", "topk_indices", "valid_topk"] if d.indexer_only else ["attn_output"]
    cli.run_qaic(
        a, report, onnx_path=onnx_path,
        specialization={"batch_size": a.batch_size, "seq_len": a.seq_len, "ctx_len": a.ctx_len},
        visible_inputs={n: x[n].numpy() for n in names[0] if n not in states},
        visible_outputs={n: outputs[n].numpy() for n in visible_out},
        custom_io=qaic.retained_custom_io(states, a.kv_cache_dtype),
        compile_kwargs={"mxfp6_matmul": a.mxfp6_matmul, "allow_mxint8_mdp_io": False},
    )
    cli.write_report(a, report)


if __name__ == "__main__":
    main()
