"""Qwen3 GQA KV-blocking QAIC custom ops (ONNX function bodies).

The ops that the two ported paths emit: online prefill and batch-split decode.
Domains, op types and bodies match ``gqa_benchmark/qwen3_gqa_kv_blocking_microbench.py``,
so QAIC lowering is unchanged.
"""

from __future__ import annotations

import onnxscript
from onnxscript import opset18 as op

from layer_benchmarks.common.custom_ops import qaic_function


@qaic_function()
def CtxChunkScatter(data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    """Interleaved chunk-KV scatter: row=h*NC+pos%NC, col=pos//NC."""
    batch_size = op.Gather(op.Shape(data), [0])
    num_heads = op.Gather(op.Shape(updates), [1])
    seq_len = op.Gather(op.Shape(position_ids), [1])
    total_rows = op.Gather(op.Shape(data), [1])
    nc = op.Div(total_rows, num_heads)
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    shape = op.Concat(batch_size, num_heads, seq_len, one, axis=0)
    b = op.Expand(op.Unsqueeze(op.Range(zero, batch_size, one), [1, 2, 3]), shape)
    h = op.Expand(op.Unsqueeze(op.Range(zero, num_heads, one), [0, 2, 3]), shape)
    pos = op.Cast(position_ids, to=7)
    group = op.Expand(op.Unsqueeze(op.Mod(pos, nc), [1, 3]), shape)
    col = op.Expand(op.Unsqueeze(op.Div(pos, nc), [1, 3]), shape)
    row = op.Cast(h, to=7) * nc + group
    return op.ScatterND(data, op.Concat(b, row, col, axis=3), updates)


@qaic_function()
def CtxChunkScatterPacked(data: onnxscript.FLOAT, position_ids: onnxscript.INT32,
                          updates: onnxscript.FLOAT, p: int) -> onnxscript.FLOAT:
    """Packed chunk-KV scatter for p consecutive tokens per lane."""
    batch_size = op.Gather(op.Shape(data), [0])
    num_heads = op.Gather(op.Shape(updates), [1])
    seq_len = op.Gather(op.Shape(position_ids), [1])
    total_rows = op.Gather(op.Shape(data), [1])
    nc = op.Div(total_rows, num_heads)
    ps = op.Constant(value_int=p)
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    shape = op.Concat(batch_size, num_heads, seq_len, one, axis=0)
    b = op.Expand(op.Unsqueeze(op.Range(zero, batch_size, one), [1, 2, 3]), shape)
    h = op.Expand(op.Unsqueeze(op.Range(zero, num_heads, one), [0, 2, 3]), shape)
    pos = op.Cast(position_ids, to=7)
    group = op.Expand(op.Unsqueeze(op.Mod(op.Div(pos, ps), nc), [1, 3]), shape)
    col = op.Expand(op.Unsqueeze(op.Add(op.Mul(op.Div(pos, op.Mul(nc, ps)), ps), op.Mod(pos, ps)), [1, 3]), shape)
    row = op.Add(op.Mul(op.Cast(h, to=7), op.Cast(nc, to=7)), op.Cast(group, to=7))
    return op.ScatterND(data, op.Concat(b, row, col, axis=3), updates)


@qaic_function()
def CtxGatherBlockedKV(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    """data [B, H, T, D], ctx_indices [B, H, Tb] -> [B, H, Tb, D]."""
    ctx_indices = op.Unsqueeze(ctx_indices, [-1])
    return op.GatherND(data, ctx_indices, batch_dims=2)


@qaic_function()
def CtxChunkScatterBatch(
    data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    """data [1, B*H, T, D][0, b*H + h, position_ids[b, s]] = updates [B, H, S, D].

    Updates and indices are folded onto the BH axis, so the compiler can split
    that axis across devices.
    """
    batch_size = op.Gather(op.Shape(updates), [0])
    num_heads = op.Gather(op.Shape(updates), [1])
    seq_len = op.Gather(op.Shape(updates), [2])
    head_dim = op.Gather(op.Shape(updates), [3])
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    bh = op.Mul(batch_size, num_heads)
    updates_folded = op.Reshape(updates, op.Concat(one, bh, seq_len, head_dim, axis=0))
    head_flat = op.Expand(op.Unsqueeze(op.Range(zero, bh, one), [0, 2, 3]), op.Concat(one, bh, seq_len, one, axis=0))
    pos_i64 = op.Cast(position_ids, to=7)
    pos_tiled = op.Expand(op.Unsqueeze(pos_i64, [1]), op.Concat(batch_size, num_heads, seq_len, axis=0))
    pos_exp = op.Reshape(pos_tiled, op.Concat(one, bh, seq_len, one, axis=0))
    indices = op.Concat(op.Mul(head_flat, zero), head_flat, pos_exp, axis=3)
    return op.ScatterND(data, indices, updates_folded)


@qaic_function()
def CtxGatherBlockedKVBatch(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    """data [1, BH, T, D], ctx_indices [1, BH, Tb] -> [1, BH, Tb, D]."""
    ctx_indices = op.Unsqueeze(ctx_indices, [-1])
    return op.GatherND(data, ctx_indices, batch_dims=2)


@qaic_function()
def CustomRMSNorm(hidden_states: onnxscript.FLOAT, weight: onnxscript.FLOAT, epsilon: float) -> onnxscript.FLOAT:
    weight = op.Cast(weight, to=1)
    variance = op.ReduceMean(op.Pow(hidden_states, 2), axes=[-1], keepdims=1)
    epsilon = op.Expand(epsilon, op.Shape(variance))
    hidden_states = hidden_states * op.Reciprocal(op.Sqrt(variance + epsilon))
    return weight * hidden_states
