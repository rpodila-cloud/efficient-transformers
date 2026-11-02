"""MoE QAIC custom ops: 3-D context gather / scatter along the token axis.

Names, domain and bodies match the original bench's opset-18 functions
(``com.qualcomm.cloud::CtxGather3D_v18`` …) so QAIC lowering is unchanged.
INT32_MAX indices mark padding rows; on QAIC they are no-ops.
"""

from __future__ import annotations

import onnxscript
from onnxscript import opset18 as op

from layer_benchmarks.common.custom_ops import CLOUD_DOMAIN, qaic_function


@qaic_function(CLOUD_DOMAIN)
def CtxGather3D_v18(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    """out[b, s] = data[b, ctx_indices[b, s]]"""
    ctx_indices = op.Cast(ctx_indices, to=7)
    ctx_indices = op.Unsqueeze(ctx_indices, [-1])
    return op.GatherND(data, ctx_indices, batch_dims=1)


@qaic_function(CLOUD_DOMAIN)
def CtxScatter3D_v18(data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    """data[b, position_ids[b, s]] = updates[b, s]"""
    batch_size = op.Reshape(op.Gather(op.Shape(data), [0]), op.Constant(value_ints=[]))
    seq_len = op.Reshape(op.Gather(op.Shape(position_ids), [1]), op.Constant(value_ints=[]))
    zero = op.Reshape(op.Cast(op.Constant(value_ints=[0]), to=7), op.Constant(value_ints=[]))
    one_scalar = op.Reshape(op.Cast(op.Constant(value_ints=[1]), to=7), op.Constant(value_ints=[]))
    one_vec = op.Cast(op.Constant(value_ints=[1]), to=7)
    exp_shape = op.Concat(op.Unsqueeze(batch_size, [0]), op.Unsqueeze(seq_len, [0]), one_vec, axis=0)
    batch_idx = op.Expand(op.Unsqueeze(op.Range(zero, batch_size, one_scalar), [1, 2]), exp_shape)
    ctx_idx = op.Expand(op.Unsqueeze(op.Cast(position_ids, to=7), [2]), exp_shape)
    indices = op.Concat(batch_idx, ctx_idx, axis=2)
    return op.ScatterND(data, indices, updates)


@qaic_function(CLOUD_DOMAIN)
def CtxScatter3DInt_v18(data: onnxscript.INT32, position_ids: onnxscript.INT32, updates: onnxscript.INT32) -> onnxscript.INT32:
    """Integer CtxScatter3D_v18 (builds the packed row index)."""
    batch_size = op.Reshape(op.Gather(op.Shape(data), [0]), op.Constant(value_ints=[]))
    seq_len = op.Reshape(op.Gather(op.Shape(position_ids), [1]), op.Constant(value_ints=[]))
    zero = op.Reshape(op.Cast(op.Constant(value_ints=[0]), to=7), op.Constant(value_ints=[]))
    one_scalar = op.Reshape(op.Cast(op.Constant(value_ints=[1]), to=7), op.Constant(value_ints=[]))
    one_vec = op.Cast(op.Constant(value_ints=[1]), to=7)
    exp_shape = op.Concat(op.Unsqueeze(batch_size, [0]), op.Unsqueeze(seq_len, [0]), one_vec, axis=0)
    batch_idx = op.Expand(op.Unsqueeze(op.Range(zero, batch_size, one_scalar), [1, 2]), exp_shape)
    ctx_idx = op.Expand(op.Unsqueeze(op.Cast(position_ids, to=7), [2]), exp_shape)
    indices = op.Concat(batch_idx, ctx_idx, axis=2)
    return op.ScatterND(data, indices, updates)
