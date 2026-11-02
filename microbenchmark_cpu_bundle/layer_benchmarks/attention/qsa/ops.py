"""QSA QAIC custom ops (ONNX function bodies) and Dynamo translations.

Only ops the QSA graph actually emits live here. Domains, op types and
signatures match the original qwen38 benchmark so QAIC lowering is unchanged.
"""

from __future__ import annotations

import onnxscript
import torch
from onnxscript import opset18 as op

from layer_benchmarks.common.custom_ops import qaic_function


@qaic_function()
def CtxPagedScatter(
    data: onnxscript.FLOAT,
    block_id: onnxscript.INT32,
    addr: onnxscript.INT32,
    updates: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    """data[block_id, row, addr] = updates; INT32_MAX block ids are no-ops on QAIC."""
    update_shape = op.Shape(updates)
    batch_size = op.Gather(update_shape, [0])
    num_rows = op.Gather(update_shape, [1])
    seq_len = op.Gather(update_shape, [2])
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    exp_shape = op.Concat(batch_size, num_rows, seq_len, one, axis=0)
    row_idx = op.Expand(op.Unsqueeze(op.Range(zero, num_rows, one), [0, 2, 3]), exp_shape)
    indices = op.Concat(
        op.Unsqueeze(op.Cast(block_id, to=onnxscript.INT64.dtype), [-1]),
        op.Cast(row_idx, to=onnxscript.INT64.dtype),
        op.Unsqueeze(op.Cast(addr, to=onnxscript.INT64.dtype), [-1]),
        axis=3,
    )
    return op.ScatterND(data, indices, updates)


@qaic_function()
def CtxGatherBlockRangeKVDP(data: onnxscript.FLOAT, block_ids: onnxscript.INT32) -> onnxscript.FLOAT:
    """[B, rows, blocks, 4, D] x [B, rows, K] -> [B, rows, K, 4, D] without mapping the full cache."""
    return op.GatherND(data, op.Unsqueeze(block_ids, [-1]), batch_dims=2)


@qaic_function()
def V4CtxGatherRows(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    """Gather context entries independently from each physical cache row."""
    invalid_sentinel = op.CastLike(op.Constant(value_int=2147483647), ctx_indices)
    invalid = op.Equal(ctx_indices, invalid_sentinel)
    safe_indices = op.Where(invalid, op.Mul(ctx_indices, 0), ctx_indices)
    return op.GatherND(
        data,
        op.Unsqueeze(op.Cast(safe_indices, to=onnxscript.INT64.dtype), [-1]),
        batch_dims=2,
    )


# torch.library ops lowered through the Dynamo custom translation table.
@torch.library.custom_op("qsa_bench::block_gather", mutates_args=())
def block_gather(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather complete contiguous blocks from [A, G, N, R, D]."""
    a = torch.arange(data.shape[0], device=data.device)[:, None, None]
    g = torch.arange(data.shape[1], device=data.device)[None, :, None]
    return data[a, g, indices[..., 0]].contiguous()


@block_gather.register_fake
def _(data, indices):
    return data.new_empty((*indices.shape[:-1], *data.shape[3:]))


@torch.library.custom_op("qsa_bench::topk", mutates_args=())
def stable_topk(data: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Match ONNX TopK's lower-index tie behaviour in eager PyTorch."""
    ids = torch.argsort(data, dim=-1, descending=True, stable=True)[..., :k]
    return data.gather(-1, ids), ids.contiguous()


@stable_topk.register_fake
def _(data, k):
    shape = (*data.shape[:-1], k)
    return data.new_empty(shape), torch.empty(shape, dtype=torch.int64, device=data.device)


def _onnx_block_gather(data, indices):
    return op.GatherND(data, indices, batch_dims=2)


def _onnx_topk(data, k: int):
    return op.TopK(data, op.Constant(value_ints=[k]), axis=-1, largest=1, sorted=1)


TRANSLATIONS = {
    torch.ops.qsa_bench.block_gather.default: _onnx_block_gather,
    torch.ops.qsa_bench.topk.default: _onnx_topk,
}
