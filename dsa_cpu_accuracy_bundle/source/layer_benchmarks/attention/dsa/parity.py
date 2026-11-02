"""GLM DSA parity loops.

``self_test`` checks every ported path on tiny shapes:

* ``decode_cp2`` is ``run_dsa_parity`` from the standalone: indexer CP2 and attention CP2 decode, compared with
  the dense HF ``GlmMoeDsaIndexer``/``GlmMoeDsaAttention`` equations. It checks the Top-K set, the output, and
  all three retained caches.
* ``decode_dp2`` is the same check at DP2/CP1, the layout the decode benchmark uses.
* ``prefill`` checks the parallel prefill indexer and the sparse prefill attention against a dense causal
  per-query Top-K plus the same HF attention equations.
* ORT runs on the export with the custom ops decomposed. The QAIC export is also checked for Sign/NonZero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from layer_benchmarks.common import custom_ops, parity
from layer_benchmarks.common.export import op_histogram

from .forwards import (
    MIN_MASKED_ATTENTION_VALUE,
    DSAConfig,
    apply_single_rotary_pos_emb,
    build_dsa_parallel_topk_indices,
    build_module,
    get_attention_config,
)

TOL = 5.0e-5


def _pack_dense_dsa_cache(
    dense: torch.Tensor,
    dp: int,
    cp: int,
    index_block_size: Optional[int] = None,
) -> torch.Tensor:
    """Pack [B,T,D] into a folded attention or indexer cache layout."""
    batch, ctx_len, dim = dense.shape
    if batch % dp or ctx_len % cp:
        raise ValueError("Parity cache dimensions must divide DP and CP.")
    if index_block_size is not None:
        if index_block_size <= 0 or index_block_size % cp:
            raise ValueError(
                "Indexer block size must be positive and divisible by CP."
            )
        if ctx_len % (index_block_size * cp):
            raise ValueError(
                "Indexer parity context must be divisible by block_size * CP."
            )
    dense_dp = dense.view(dp, batch // dp, ctx_len, dim).permute(1, 0, 2, 3)
    rows = dp * cp
    row_idx = torch.arange(rows, device=dense.device)
    row_dp = row_idx // cp
    row_way = row_idx % cp
    local_addr = torch.arange(ctx_len // cp, device=dense.device)
    if index_block_size is None:
        logical_pos = local_addr.view(1, -1) * cp + row_way.view(-1, 1)
    else:
        logical_pos = (
            (local_addr.view(1, -1) // index_block_size)
            * (index_block_size * cp)
            + row_way.view(-1, 1) * index_block_size
            + local_addr.view(1, -1) % index_block_size
        )
    return dense_dp[:, row_dp[:, None], logical_pos, :]


def _unpack_dense_dsa_cache(
    cache: torch.Tensor,
    batch: int,
    ctx_len: int,
    dp: int,
    cp: int,
    index_block_size: Optional[int] = None,
) -> torch.Tensor:
    """Unpack a folded attention or indexer cache to logical [B,T,D]."""
    batch_local, rows, _, dim = cache.shape
    if (batch_local, rows) != (batch // dp, dp * cp):
        raise ValueError("Unexpected parity cache layout.")
    if index_block_size is not None:
        if index_block_size <= 0 or index_block_size % cp:
            raise ValueError(
                "Indexer block size must be positive and divisible by CP."
            )
        if ctx_len % (index_block_size * cp):
            raise ValueError(
                "Indexer parity context must be divisible by block_size * CP."
            )
    dense = torch.zeros(
        batch_local, dp, ctx_len, dim, dtype=cache.dtype, device=cache.device
    )
    row_idx = torch.arange(rows, device=cache.device)
    row_dp = row_idx // cp
    row_way = row_idx % cp
    local_addr = torch.arange(ctx_len // cp, device=cache.device)
    if index_block_size is None:
        logical_pos = local_addr.view(1, -1) * cp + row_way.view(-1, 1)
    else:
        logical_pos = (
            (local_addr.view(1, -1) // index_block_size)
            * (index_block_size * cp)
            + row_way.view(-1, 1) * index_block_size
            + local_addr.view(1, -1) % index_block_size
        )
    dense[:, row_dp[:, None], logical_pos, :] = cache
    return dense.permute(1, 0, 2, 3).reshape(batch, ctx_len, dim)


def _reference_dsa_indexer_topk(
    module: nn.Module,
    hidden_states: torch.Tensor,
    q_a_proj_out: torch.Tensor,
    position_ids: torch.Tensor,
    dense_key_cache: torch.Tensor,
) -> torch.Tensor:
    """Dense HF-style DSA indexer reference, without any external import."""
    batch, q_len, _ = hidden_states.shape
    q = module.dsa_wq_b(q_a_proj_out).view(
        batch, q_len, module.dsa_index_n_heads, module.dsa_index_head_dim
    )
    q_pe, q_nope = torch.split(
        q, [module.qk_rope_head_dim, module.dsa_index_head_dim - module.qk_rope_head_dim], dim=-1
    )
    k = module.dsa_k_norm(module.dsa_wk(hidden_states))
    k_pe, k_nope = torch.split(
        k, [module.qk_rope_head_dim, module.dsa_index_head_dim - module.qk_rope_head_dim], dim=-1
    )
    cos, sin = module.rotary_emb(hidden_states, seq_len=module.max_position_embeddings)
    q_pe = apply_single_rotary_pos_emb(q_pe, cos, sin, position_ids, unsqueeze_dim=2)
    k_pe = apply_single_rotary_pos_emb(
        k_pe.unsqueeze(2), cos, sin, position_ids, unsqueeze_dim=2
    ).squeeze(2)
    current_key = torch.cat((k_pe, k_nope), dim=-1)
    key_cache = dense_key_cache.clone()
    key_cache[torch.arange(batch), position_ids[:, 0]] = current_key[:, 0]
    q = torch.cat((q_pe, q_nope), dim=-1)
    weights = module.dsa_weights_proj(hidden_states).float() * (
        module.dsa_index_n_heads**-0.5
    )
    scores = torch.einsum("bshd,btd->bsht", q.float(), key_cache.float())
    scores = scores * module.dsa_index_head_dim**-0.5
    scores = (scores * weights.unsqueeze(-1)).sum(dim=2)
    valid = torch.arange(key_cache.shape[1], device=hidden_states.device).view(1, 1, -1)
    scores = scores.masked_fill(valid > position_ids.unsqueeze(-1), MIN_MASKED_ATTENTION_VALUE)
    return torch.topk(scores, k=module.dsa_config.topk, dim=-1).indices


def _reference_dsa_attention_output(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    dense_ckv_cache: torch.Tensor,
    dense_k_pe_cache: torch.Tensor,
    topk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense sparse-MLA reference matching the HF attention equations."""
    batch, q_len, _ = hidden_states.shape
    q_a_proj_out = module.q_a_layernorm(module.q_a_proj(hidden_states))
    q = module.q_b_proj(q_a_proj_out).view(
        batch, q_len, module.num_heads, module.q_head_dim
    ).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1
    )
    compressed_kv = module.kv_a_proj_with_mqa(hidden_states)
    current_ckv, current_k_pe = torch.split(
        compressed_kv, [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1
    )
    current_ckv = module.kv_a_layernorm(current_ckv)
    cos, sin = module.rotary_emb(hidden_states, seq_len=module.max_position_embeddings)
    q_pe = apply_single_rotary_pos_emb(q_pe, cos, sin, position_ids)
    current_k_pe = apply_single_rotary_pos_emb(
        current_k_pe.unsqueeze(1), cos, sin, position_ids
    ).squeeze(1)
    ckv = dense_ckv_cache.clone()
    k_pe = dense_k_pe_cache.clone()
    ckv[torch.arange(batch), position_ids[:, 0]] = current_ckv[:, 0]
    k_pe[torch.arange(batch), position_ids[:, 0]] = current_k_pe[:, 0]
    kv_expanded = module.kv_b_proj(ckv).view(
        batch, ckv.shape[1], module.num_heads,
        module.qk_nope_head_dim + module.v_head_dim,
    )
    k_nope, value_states = torch.split(
        kv_expanded, [module.qk_nope_head_dim, module.v_head_dim], dim=-1
    )
    key_states = torch.cat(
        (k_nope.transpose(1, 2), k_pe.unsqueeze(1).expand(-1, module.num_heads, -1, -1)),
        dim=-1,
    )
    selected = topk_indices[:, 0]
    selected_keys = key_states.gather(
        2, selected.unsqueeze(1).unsqueeze(-1).expand(
            -1, module.num_heads, -1, key_states.shape[-1]
        )
    )
    selected_values = value_states.transpose(1, 2).gather(
        2, selected.unsqueeze(1).unsqueeze(-1).expand(
            -1, module.num_heads, -1, module.v_head_dim
        )
    )
    scores = torch.matmul(torch.cat((q_nope, q_pe), dim=-1), selected_keys.transpose(-1, -2))
    scores = scores * module.softmax_scale
    probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
    output = torch.matmul(probs, selected_values).transpose(1, 2).reshape(batch, q_len, -1)
    return module.o_proj(output), current_ckv, current_k_pe






def _current_indexer_key(module, hidden_states, position_ids):
    key = module.dsa_k_norm(module.dsa_wk(hidden_states))
    key_pe, key_nope = torch.split(
        key, [module.qk_rope_head_dim, module.dsa_index_head_dim - module.qk_rope_head_dim], dim=-1)
    cos, sin = module.rotary_emb(hidden_states, seq_len=module.max_position_embeddings)
    key_pe = apply_single_rotary_pos_emb(key_pe.unsqueeze(2), cos, sin, position_ids, unsqueeze_dim=2).squeeze(2)
    return torch.cat((key_pe, key_nope), dim=-1)


def _tiny(dsa: DSAConfig):
    torch.manual_seed(17)
    config = get_attention_config("tiny")
    return config, build_module(config, dsa)


def _record(checks, name, actual, expected, exact=False):
    checks.append(parity.compare(name, actual, expected, atol=0.0 if exact else TOL, rtol=0.0))


@torch.no_grad()
def _decode_case(name: str, dp: int, cp: int, out_dir: Path) -> dict:
    """Standalone ``run_dsa_parity`` at the given indexer/attention DP and CP."""
    from .bench import DsaExportWrapper, io_names

    batch, ctx_len, topk = 2, 16, 4
    dsa = DSAConfig(enabled=True, topk=topk, index_head_dim=8, index_n_heads=2, indexer_dp=dp, indexer_cp=cp,
                    attention_dp=dp, attention_cp=cp, indexer_num_blocks=2, index_block_size=4, num_cores=2,
                    num_devices=1)
    config, module = _tiny(dsa)
    hidden_states = torch.randn(batch, 1, config.hidden_size)
    position_ids = torch.tensor([[ctx_len - 1], [ctx_len - 3]], dtype=torch.long)
    dense_ckv = torch.randn(batch, ctx_len, config.kv_lora_rank)
    dense_k_pe = torch.randn(batch, ctx_len, config.qk_rope_head_dim)
    dense_indexer = torch.randn(batch, ctx_len, config.index_head_dim)
    ckv_cache = _pack_dense_dsa_cache(dense_ckv, dp, cp)
    k_pe_cache = _pack_dense_dsa_cache(dense_k_pe, dp, cp)
    indexer_cache = _pack_dense_dsa_cache(dense_indexer, dp, cp, dsa.index_block_size)

    q_a_proj_out = module.q_a_layernorm(module.q_a_proj(hidden_states))
    reference_topk = _reference_dsa_indexer_topk(module, hidden_states, q_a_proj_out, position_ids, dense_indexer)
    actual_topk, _, _ = build_dsa_parallel_topk_indices(
        module, hidden_states, q_a_proj_out, position_ids, indexer_cache.clone(), dsa)
    actual_output, actual_present, actual_indexer = module.fused_forward_dsa_parallel(
        hidden_states, position_ids=position_ids, compressed_kvs=[[ckv_cache.clone(), k_pe_cache.clone()]],
        indexer_key_cache=indexer_cache.clone(), mla_absorption={"absorption": True, "online": False})
    reference_output, current_ckv, current_k_pe = _reference_dsa_attention_output(
        module, hidden_states, position_ids, dense_ckv, dense_k_pe, reference_topk)

    rows = torch.arange(batch)
    expected_ckv, expected_k_pe, expected_indexer = dense_ckv.clone(), dense_k_pe.clone(), dense_indexer.clone()
    expected_ckv[rows, position_ids[:, 0]] = current_ckv[:, 0]
    expected_k_pe[rows, position_ids[:, 0]] = current_k_pe[:, 0]
    expected_indexer[rows, position_ids[:, 0]] = _current_indexer_key(module, hidden_states, position_ids)[:, 0]
    actual_ckv, actual_k_pe = actual_present[0]
    checks = []
    _record(checks, "topk_set", torch.sort(actual_topk, dim=-1).values,
            torch.sort(reference_topk[:, 0], dim=-1).values, exact=True)
    _record(checks, "attn_output", actual_output, reference_output)
    _record(checks, "ckv_cache", _unpack_dense_dsa_cache(actual_ckv, batch, ctx_len, dp, cp), expected_ckv)
    _record(checks, "k_pe_cache", _unpack_dense_dsa_cache(actual_k_pe, batch, ctx_len, dp, cp), expected_k_pe)
    _record(checks, "indexer_cache", _unpack_dense_dsa_cache(
        actual_indexer, batch, ctx_len, dp, cp, dsa.index_block_size), expected_indexer)

    wrapper = DsaExportWrapper(module, False).eval()
    feeds = {"hidden_states": hidden_states, "position_ids": position_ids, "compressed_kv.0": ckv_cache,
             "k_pe.0": k_pe_cache, "indexer_key_cache": indexer_cache}
    return _finish(name, checks, wrapper, feeds, io_names(False), out_dir)


def _reference_prefill(module, hidden_states, position_ids, dense_ckv, dense_k_pe, dense_indexer, topk):
    """Dense causal reference: per-query indexer Top-K, then HF sparse attention over the selected positions."""
    batch, q_len, _ = hidden_states.shape
    rows = torch.arange(batch).view(-1, 1)
    q_a_proj_out = module.q_a_layernorm(module.q_a_proj(hidden_states))
    q = module.dsa_wq_b(q_a_proj_out).view(batch, q_len, module.dsa_index_n_heads, module.dsa_index_head_dim)
    q_pe, q_nope = torch.split(
        q, [module.qk_rope_head_dim, module.dsa_index_head_dim - module.qk_rope_head_dim], dim=-1)
    cos, sin = module.rotary_emb(hidden_states, seq_len=module.max_position_embeddings)
    q_idx = torch.cat((apply_single_rotary_pos_emb(q_pe, cos, sin, position_ids, unsqueeze_dim=2), q_nope), dim=-1)
    key_cache = dense_indexer.clone()
    key_cache[rows, position_ids] = _current_indexer_key(module, hidden_states, position_ids)
    weights = module.dsa_weights_proj(hidden_states).float() * module.dsa_index_n_heads**-0.5
    scores = torch.einsum("bshd,btd->bsht", q_idx, key_cache) * module.dsa_index_head_dim**-0.5
    scores = (scores * weights.unsqueeze(-1)).sum(dim=2)
    causal = torch.arange(key_cache.shape[1]).view(1, 1, -1) > position_ids.unsqueeze(-1)
    scores = scores.masked_fill(causal, MIN_MASKED_ATTENTION_VALUE)
    top = torch.topk(scores, k=topk, dim=-1)
    valid = top.values > MIN_MASKED_ATTENTION_VALUE

    q = module.q_b_proj(q_a_proj_out).view(batch, q_len, module.num_heads, module.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1)
    q_pe = apply_single_rotary_pos_emb(q_pe, cos, sin, position_ids)
    current_ckv, current_k_pe = torch.split(
        module.kv_a_proj_with_mqa(hidden_states), [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1)
    current_ckv = module.kv_a_layernorm(current_ckv)
    current_k_pe = apply_single_rotary_pos_emb(current_k_pe.unsqueeze(1), cos, sin, position_ids).squeeze(1)
    ckv, k_pe = dense_ckv.clone(), dense_k_pe.clone()
    ckv[rows, position_ids], k_pe[rows, position_ids] = current_ckv, current_k_pe
    kv = module.kv_b_proj(ckv).view(batch, ckv.shape[1], module.num_heads, module.qk_nope_head_dim + module.v_head_dim)
    k_nope, values = torch.split(kv, [module.qk_nope_head_dim, module.v_head_dim], dim=-1)
    keys = torch.cat((k_nope.transpose(1, 2), k_pe.unsqueeze(1).expand(-1, module.num_heads, -1, -1)), dim=-1)
    scores = torch.matmul(torch.cat((q_nope, q_pe), dim=-1), keys.transpose(-1, -2)) * module.softmax_scale
    allowed = torch.zeros(batch, q_len, ckv.shape[1], dtype=torch.bool)
    allowed.scatter_(-1, top.indices, valid)
    scores = scores.masked_fill(~allowed.unsqueeze(1), MIN_MASKED_ATTENTION_VALUE)
    output = torch.matmul(torch.softmax(scores, dim=-1), values.transpose(1, 2))
    output = module.o_proj(output.transpose(1, 2).reshape(batch, q_len, -1))
    return output, top.indices, valid, ckv, k_pe, key_cache


@torch.no_grad()
def _prefill_case(out_dir: Path) -> dict:
    from .bench import DsaExportWrapper, io_names

    batch, seq, ctx_len, start, topk = 1, 16, 32, 8, 4
    dsa = DSAConfig(enabled=True, topk=topk, index_head_dim=8, index_n_heads=2, indexer_dp=1, indexer_cp=1,
                    attention_dp=1, attention_cp=1, indexer_num_blocks=2, index_block_size=4, indexer_ql_chunk=16,
                    indexer_q_block_size=8, indexer_topk_blocking=2, indexer_prefill_parallel=True,
                    sparse_q_block_size=4, sparse_kv_num_blocks=2, num_cores=2, num_devices=1)
    config, module = _tiny(dsa)
    hidden_states = torch.randn(batch, seq, config.hidden_size)
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0) + start
    dense_ckv = torch.randn(batch, ctx_len, config.kv_lora_rank)
    dense_k_pe = torch.randn(batch, ctx_len, config.qk_rope_head_dim)
    dense_indexer = torch.randn(batch, ctx_len, config.index_head_dim)
    ckv_cache, k_pe_cache = _pack_dense_dsa_cache(dense_ckv, 1, 1), _pack_dense_dsa_cache(dense_k_pe, 1, 1)
    indexer_cache = _pack_dense_dsa_cache(dense_indexer, 1, 1, dsa.index_block_size)

    wrapper = DsaExportWrapper(module, False).eval()
    feeds = {"hidden_states": hidden_states, "position_ids": position_ids, "compressed_kv.0": ckv_cache,
             "k_pe.0": k_pe_cache, "indexer_key_cache": indexer_cache}
    names = io_names(False)
    out, ckv, k_pe, indexer = wrapper(*(feeds[n].clone() for n in names[0]))
    ref_out, ref_topk, ref_valid, ref_ckv, ref_k_pe, ref_indexer = _reference_prefill(
        module, hidden_states, position_ids, dense_ckv, dense_k_pe, dense_indexer, topk)

    module.dsa_config.indexer_only = True
    _, _, topk_indices, valid_topk = DsaExportWrapper(module, True)(
        hidden_states, position_ids, indexer_cache.clone())
    module.dsa_config.indexer_only = False
    sentinel = torch.iinfo(torch.int32).max
    checks = []
    _record(checks, "valid_count", valid_topk.sum(-1), ref_valid.sum(-1), exact=True)
    _record(checks, "topk_set", torch.sort(torch.where(valid_topk, topk_indices, sentinel).long(), dim=-1).values,
            torch.sort(torch.where(ref_valid, ref_topk, sentinel), dim=-1).values, exact=True)
    _record(checks, "attn_output", out, ref_out)
    _record(checks, "ckv_cache", _unpack_dense_dsa_cache(ckv, batch, ctx_len, 1, 1), ref_ckv)
    _record(checks, "k_pe_cache", _unpack_dense_dsa_cache(k_pe, batch, ctx_len, 1, 1), ref_k_pe)
    _record(checks, "indexer_cache", _unpack_dense_dsa_cache(indexer, batch, ctx_len, 1, 1, 4), ref_indexer)
    return _finish("prefill", checks, wrapper, feeds, names, out_dir)


def _finish(name, checks, wrapper, feeds, names, out_dir: Path) -> dict:
    from .bench import export_onnx

    in_names, out_names = names
    with custom_ops.enabled(False):
        ref_path = export_onnx(wrapper, feeds, names, out_dir / name / "ref.onnx", custom_ops=False)
    want = wrapper(*(feeds[n].clone() for n in in_names))
    got = parity.ort_run(ref_path, {n: feeds[n].numpy() for n in in_names}, out_names)
    for n, t in zip(out_names, want):
        checks.append(parity.compare(f"ort.{n}", got[n], t, atol=2e-5, rtol=1e-4))
    ops = op_histogram(export_onnx(wrapper, feeds, names, out_dir / name / "qaic.onnx"))
    result = {"checks": checks, "banned_ops": {k: v for k, v in ops.items() if k in ("Sign", "NonZero")},
              "qaic_ops": {k: v for k, v in ops.items() if "::" in k}}
    result["passed"] = all(c["passed"] for c in checks) and not result["banned_ops"]
    parity.require(result, f"dsa {name}")
    return {"passed": True, "max_abs": {c["name"]: c["max_abs"] for c in checks}, "qaic_ops": result["qaic_ops"]}


def self_test(out_dir: Path) -> dict:
    torch.set_num_threads(8)
    return {
        "decode_cp2": _decode_case("decode_cp2", 1, 2, out_dir),
        "decode_dp2": _decode_case("decode_dp2", 2, 1, out_dir),
        "prefill": _prefill_case(out_dir),
    }
