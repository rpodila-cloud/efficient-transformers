"""QAIC custom cache functions used by the MLA retained-state ABI."""

from __future__ import annotations

import onnxscript
from onnxscript import opset18 as op

from layer_benchmarks.common.custom_ops import CLOUD_DOMAIN, qaic_function


@qaic_function(CLOUD_DOMAIN)
def CtxScatter(data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    batch = op.Gather(op.Shape(data), [0])
    heads = op.Gather(op.Shape(data), [1])
    seq = op.Gather(op.Shape(position_ids), [1])
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    shape = op.Concat(batch, heads, seq, one, axis=0)
    b = op.Expand(op.Unsqueeze(op.Range(zero, batch, one), [1, 2, 3]), shape)
    h = op.Expand(op.Unsqueeze(op.Range(zero, heads, one), [0, 2, 3]), shape)
    p = op.Expand(op.Unsqueeze(op.Cast(position_ids, to=onnxscript.INT64.dtype), [1, 3]), shape)
    indices = op.Concat(b, h, p, axis=3)
    return op.ScatterND(data, indices, updates)


@qaic_function(CLOUD_DOMAIN)
def CtxGatherBlockedKV(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    indices = op.Cast(op.Unsqueeze(ctx_indices, [-1]), to=onnxscript.INT64.dtype)
    return op.GatherND(data, indices, batch_dims=2)
