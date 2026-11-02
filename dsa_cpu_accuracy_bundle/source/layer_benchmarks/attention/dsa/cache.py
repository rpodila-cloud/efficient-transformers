"""GLM DSA retained-cache reads and writes.

Cache ABI (see README):

  indexer_key_cache  [B/idx_dp, idx_dp*idx_cp, ctx/idx_cp, index_head_dim]   block-striped over CP
  compressed_kv.0    [B/attn_dp, attn_dp*attn_cp, ctx/attn_cp, kv_lora_rank]  token-interleaved over CP
  k_pe.0             [B/attn_dp, attn_dp*attn_cp, ctx/attn_cp, qk_rope_head_dim]

Each function runs eager PyTorch, or emits the QAIC custom op while exporting.
"""

from __future__ import annotations

import torch

from layer_benchmarks.common import custom_ops

from . import ops

INVALID = torch.iinfo(torch.int32).max


def gather_rows(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """data [B, R, T, D] at ctx_indices [B, R, *S] -> [B, R, *S, D]; INT32_MAX reads row 0."""

    def eager(data, ctx_indices):
        if ctx_indices.ndim < 3:
            raise ValueError("ctx_indices must have shape [B, rows, ...].")
        b = torch.arange(data.shape[0]).view(data.shape[0], *(1,) * (ctx_indices.ndim - 1))
        r = torch.arange(data.shape[1]).view(1, data.shape[1], *(1,) * (ctx_indices.ndim - 2))
        idx = torch.where(ctx_indices == INVALID, 0, ctx_indices)
        return data[b, r, idx.long()]

    shape = (data.shape[0], data.shape[1], *ctx_indices.shape[2:], data.shape[3])
    return custom_ops.call(ops.V4CtxGatherRows, (data, ctx_indices), eager, shape=shape)


def paged_scatter(data, block_id, addr, updates) -> torch.Tensor:
    """data [P, R, T, D][block_id, r, addr] = updates [B, R, S, D]; INT32_MAX block ids are dropped."""

    def eager(data, block_id, addr, updates):
        out = data.clone()
        rows = torch.arange(updates.shape[1]).view(1, -1, 1).expand_as(block_id)
        valid = block_id != INVALID
        out[block_id[valid].long(), rows[valid], addr[valid].long()] = updates[valid]
        return out

    return custom_ops.call(ops.V4CtxPagedScatter, (data, block_id, addr, updates), eager, shape=data.shape)
