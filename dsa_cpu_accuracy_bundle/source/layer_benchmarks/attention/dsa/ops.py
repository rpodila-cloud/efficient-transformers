"""GLM DSA QAIC custom ops (ONNX function bodies) and Dynamo translations.

Only the ops the DSA decode/prefill graphs emit. Domains, op types and bodies
match ``glm_dsa/glm_mla_dsa_microbench.py`` so QAIC lowering is unchanged.
"""

from __future__ import annotations

import onnxscript
import torch
from onnxscript import opset18 as op

from layer_benchmarks.common.custom_ops import QAIC_DOMAIN, qaic_function


@qaic_function(QAIC_DOMAIN)
def V4CtxGatherRows(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    """data [B, R, T, D], ctx_indices [B, R, *S] -> [B, R, *S, D]."""
    return op.GatherND(data, op.Unsqueeze(ctx_indices, [-1]), batch_dims=2)


@qaic_function(QAIC_DOMAIN)
def V4CtxPagedScatter(
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
