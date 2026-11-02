"""MiniMax-M3 GQA/MSA QAIC custom ops (ONNX function bodies) and Dynamo translations.

Only ops the stacked GQA/MSA decode and prefill graphs emit. Domains, op
types and bodies match ``msa/minimax_m3_msa_decode_microbench.py`` so QAIC
lowering is unchanged.
"""

from __future__ import annotations

import onnxscript
import torch
from onnxscript import opset18 as op

from layer_benchmarks.common import custom_ops
from layer_benchmarks.common.custom_ops import CLOUD_DOMAIN, qaic_function


@qaic_function(CLOUD_DOMAIN)
def M3CtxScatter(data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    """data [B, H, T, D][b, h, position_ids[b, s]] = updates [B, H, S, D]."""
    batch_size = op.Gather(op.Shape(data), [0])
    num_heads = op.Gather(op.Shape(data), [1])
    seq_len = op.Gather(op.Shape(position_ids), [1])
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    exp_shape = op.Concat(batch_size, num_heads, seq_len, one, axis=0)
    batch_idx = op.Expand(op.Unsqueeze(op.Range(zero, batch_size, one), [1, 2, 3]), exp_shape)
    head_idx = op.Expand(op.Unsqueeze(op.Range(zero, num_heads, one), [0, 2, 3]), exp_shape)
    ctx_idx = op.Expand(op.Unsqueeze(op.Cast(position_ids, to=onnxscript.INT64.dtype), [1, 3]), exp_shape)
    indices = op.Concat(batch_idx, head_idx, ctx_idx, axis=3)
    return op.ScatterND(data, indices, updates)


@qaic_function()
def CustomRMSNorm(hidden_states: onnxscript.FLOAT, weight: onnxscript.FLOAT, epsilon: float) -> onnxscript.FLOAT:
    weight = op.Cast(weight, to=onnxscript.FLOAT.dtype)
    variance = op.ReduceMean(op.Pow(hidden_states, 2), axes=[-1], keepdims=1)
    epsilon = op.Expand(epsilon, op.Shape(variance))
    hidden_states = hidden_states * op.Reciprocal(op.Sqrt(variance + epsilon))
    return weight * hidden_states


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    def eager(hidden_states, weight):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return weight * hidden_states * torch.rsqrt(variance + eps)

    return custom_ops.call(CustomRMSNorm, (hidden_states, weight), eager, shape=hidden_states.shape,
                           attrs={"epsilon": float(eps)})


@qaic_function()
def CtxPagedScatter(
    data: onnxscript.FLOAT, block_id: onnxscript.INT32, addr: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    """data [P, R, T, D][block_id[b, r, s], r, addr[b, r, s]] = updates [B, R, S, D]."""
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
def CtxPagedScatterPage(data: onnxscript.FLOAT, block_id: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    """data [P, H, page, D][block_id[p, h], h] = updates [pages, H, page, D]."""
    update_shape = op.Shape(updates)
    num_pages = op.Gather(update_shape, [0])
    num_heads = op.Gather(update_shape, [1])
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    head_idx = op.Expand(op.Unsqueeze(op.Range(zero, num_heads, one), [0]), op.Concat(num_pages, num_heads, axis=0))
    indices = op.Concat(
        op.Unsqueeze(op.Cast(block_id, to=7), [-1]),
        op.Unsqueeze(op.Cast(head_idx, to=7), [-1]),
        axis=2,
    )
    return op.ScatterND(data, indices, updates)


@qaic_function()
def CtxGatherBlockedKVDP(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    """data [B, R, T, D], ctx_indices [B, R, L] -> [B, R, L, D]."""
    return op.GatherND(data, op.Unsqueeze(ctx_indices, [-1]), batch_dims=2)


@qaic_function()
def CtxGatherBlockedKV(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    """data [B, Hkv, T, D], ctx_indices [B, Hkv, L] -> [B, Hkv, L, D]."""
    return op.GatherND(data, op.Unsqueeze(ctx_indices, [-1]), batch_dims=2)


@qaic_function()
def CtxGatherBlockRangeKVDP(data: onnxscript.FLOAT, block_ids: onnxscript.INT32) -> onnxscript.FLOAT:
    """data [B, R, blocks, block, D], block_ids [B, R, K] -> [B, R, K, block, D]."""
    return op.GatherND(data, op.Unsqueeze(block_ids, [-1]), batch_dims=2)


@qaic_function()
def CtxGatherPagedKVDP(data: onnxscript.FLOAT, block_ids: onnxscript.INT32) -> onnxscript.FLOAT:
    """data [P, R, page, D], block_ids [pages, R] -> [1, R, pages*page, D]."""
    data_shape = op.Shape(data)
    ids_shape = op.Shape(block_ids)
    num_rows = op.Gather(data_shape, [1])
    page_size = op.Gather(data_shape, [2])
    head_dim = op.Gather(data_shape, [3])
    num_pages = op.Gather(ids_shape, [0])
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    row_idx = op.Expand(op.Unsqueeze(op.Range(zero, num_rows, one), [0]), op.Concat(num_pages, num_rows, axis=0))
    indices = op.Concat(
        op.Unsqueeze(op.Cast(block_ids, to=7), [-1]),
        op.Unsqueeze(op.Cast(row_idx, to=7), [-1]),
        axis=2,
    )
    # Make GatherND emit [row, page, token, D] directly.  The transpose is
    # only over the small index tensor; avoid transposing the gathered KV
    # payload [page, row, token, D].
    indices = op.Transpose(indices, perm=[1, 0, 2])
    pages = op.GatherND(data, indices, batch_dims=0)
    return op.Reshape(pages, op.Concat(one, num_rows, op.Mul(num_pages, page_size), head_dim, axis=0))


@qaic_function()
def CtxGatherPagedKVHeads(data: onnxscript.FLOAT, block_ids: onnxscript.INT32) -> onnxscript.FLOAT:
    """data [P, H, page, D], block_ids [pages, H] -> [1, H, pages*page, D]."""
    ids_shape = op.Shape(block_ids)
    num_pages = op.Gather(ids_shape, [0])
    num_heads = op.Gather(ids_shape, [1])
    data_shape = op.Shape(data)
    page_size = op.Gather(data_shape, [2])
    head_dim = op.Gather(data_shape, [3])
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    head_idx = op.Expand(op.Unsqueeze(op.Range(zero, num_heads, one), [0]), ids_shape)
    indices = op.Concat(
        op.Unsqueeze(op.Cast(block_ids, to=7), [-1]),
        op.Unsqueeze(op.Cast(head_idx, to=7), [-1]),
        axis=2,
    )
    # GatherND now emits [head, page, token, D] directly.  Only the small
    # index tensor is transposed; the gathered KV payload is not.
    indices = op.Transpose(indices, perm=[1, 0, 2])
    pages = op.GatherND(data, indices, batch_dims=0)
    return op.Reshape(pages, op.Concat(one, num_heads, op.Mul(num_pages, page_size), head_dim, axis=0))


# Dynamo lowers integer ``//`` to Div + Sign/floor fix-ups. Every divide here is
# on non-negative ints, so a plain ONNX Div (truncating) is exact and matches
# the TorchScript graph the golden was compiled from.
def _onnx_floordiv(a, b):
    return op.Div(a, op.CastLike(b, a))


def _onnx_div_mode(a, b, rounding_mode: str = None):
    return op.Div(a, op.CastLike(b, a))


TRANSLATIONS = {
    torch.ops.aten.floor_divide.default: _onnx_floordiv,
    torch.ops.aten.div.Tensor_mode: _onnx_div_mode,
}
