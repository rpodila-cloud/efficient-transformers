"""DeepSeek-V4 CSA/HCA QAIC custom ops (ONNX function bodies).

Only the six ops the folded DP/CP decode graph emits. Domains, op types and
bodies match the original DeepSeek-V4 bench so QAIC
lowering is unchanged.
"""

from __future__ import annotations

import onnxscript
from onnxscript import opset18 as op

from layer_benchmarks.common.custom_ops import CLOUD_DOMAIN, qaic_function


@qaic_function(CLOUD_DOMAIN)
def V4CtxScatter1D(data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    """data [B, T, D][b, position_ids[b, s]] = updates [B, S, D]."""
    batch_size = op.Gather(op.Shape(data), [0])
    seq_len = op.Gather(op.Shape(position_ids), [1])
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    exp_shape = op.Concat(batch_size, seq_len, one, axis=0)
    batch_idx = op.Expand(op.Unsqueeze(op.Range(zero, batch_size, one), [1, 2]), exp_shape)
    ctx_idx = op.Cast(op.Expand(op.Unsqueeze(position_ids, [2]), exp_shape), to=onnxscript.INT64.dtype)
    return op.ScatterND(data, op.Concat(batch_idx, ctx_idx, axis=2), updates)


@qaic_function()
def FoldedV4CtxFoldedScatter(
    data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    """Scatter updates [B, 1, S, D] into data [1, B, T, D], one static row per batch item."""
    update_shape = op.Shape(updates)
    batch_size = op.Gather(update_shape, [0])
    seq_len = op.Gather(update_shape, [2])
    head_dim = op.Gather(update_shape, [3])
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    updates_folded = op.Reshape(updates, op.Concat(one, batch_size, seq_len, head_dim, axis=0))
    row = op.Expand(
        op.Unsqueeze(op.Range(zero, batch_size, one), [0, 2, 3]),
        op.Concat(one, batch_size, seq_len, one, axis=0),
    )
    pos = op.Unsqueeze(op.Cast(position_ids, to=7), [0, 3])
    indices = op.Concat(op.Mul(row, zero), row, pos, axis=3)
    return op.ScatterND(data, indices, updates_folded)


@qaic_function()
def FoldedV4CtxGatherFoldedRows(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    """data [1, B, T, D], ctx_indices [B, L] -> [1, B, L, D]."""
    indices = op.Unsqueeze(op.Unsqueeze(ctx_indices, [0]), [-1])
    return op.GatherND(data, indices, batch_dims=2)


@qaic_function()
def V4CtxPagedScatter(
    data: onnxscript.FLOAT, block_id: onnxscript.INT32, addr: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    """Masked data[block_id, row, addr] = updates; INT32_MAX block ids keep the current row."""
    update_shape = op.Shape(updates)
    batch_size = op.Gather(update_shape, [0])
    num_rows = op.Gather(update_shape, [1])
    seq_len = op.Gather(update_shape, [2])
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    zero_scalar = op.Cast(op.Constant(value_int=0), to=onnxscript.INT32.dtype)
    invalid_index = op.Cast(op.Constant(value_int=2147483647), to=onnxscript.INT32.dtype)
    exp_shape = op.Concat(batch_size, num_rows, seq_len, one, axis=0)
    row_idx = op.Expand(op.Unsqueeze(op.Range(zero, num_rows, one), [0, 2, 3]), exp_shape)
    valid = op.Not(op.Equal(block_id, invalid_index))
    safe_block = op.Where(valid, block_id, zero_scalar)
    safe_addr = op.Where(valid, addr, zero_scalar)
    indices = op.Concat(
        op.Unsqueeze(op.Cast(safe_block, to=7), [-1]),
        op.Cast(row_idx, to=7),
        op.Unsqueeze(op.Cast(safe_addr, to=7), [-1]),
        axis=3,
    )
    current = op.GatherND(data, indices)
    return op.ScatterND(data, indices, op.Where(op.Unsqueeze(valid, [3]), updates, current))


@qaic_function()
def FoldedV4CtxPagedScatter(
    data: onnxscript.FLOAT, block_id: onnxscript.INT32, addr: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    """Unmasked data[block_id, row, addr] = updates (QAIC drops INT32_MAX block ids)."""
    update_shape = op.Shape(updates)
    batch_size = op.Gather(update_shape, [0])
    num_rows = op.Gather(update_shape, [1])
    seq_len = op.Gather(update_shape, [2])
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    exp_shape = op.Concat(batch_size, num_rows, seq_len, one, axis=0)
    row_idx = op.Expand(op.Unsqueeze(op.Range(zero, num_rows, one), [0, 2, 3]), exp_shape)
    indices = op.Concat(
        op.Unsqueeze(op.Cast(block_id, to=7), [-1]),
        op.Cast(row_idx, to=7),
        op.Unsqueeze(op.Cast(addr, to=7), [-1]),
        axis=3,
    )
    return op.ScatterND(data, indices, updates)


@qaic_function()
def V4CtxGatherRows(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    """data [B, R, T, D], ctx_indices [B, R, L] -> [B, R, L, D]."""
    return op.GatherND(data, op.Unsqueeze(ctx_indices, [-1]), batch_dims=2)
