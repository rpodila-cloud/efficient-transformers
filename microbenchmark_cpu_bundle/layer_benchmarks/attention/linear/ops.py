"""Qwen3.5 linear-attention QAIC custom op: the norm half of the gated RMSNorm.

Domain, op type and body match ``la_benchmark/qwen3_5_linear_attn_microbench.py``
(``com.qti.aisw.onnx::CustomRMSNorm``) so QAIC lowering is unchanged.
"""

from __future__ import annotations

import onnxscript
import torch
from onnxscript import opset18 as op

from layer_benchmarks.common import custom_ops
from layer_benchmarks.common.custom_ops import qaic_function


@qaic_function()
def CustomRMSNorm(hidden_states: onnxscript.FLOAT, weight: onnxscript.FLOAT, epsilon: float) -> onnxscript.FLOAT:
    weight = op.Cast(weight, to=1)
    variance = op.ReduceMean(op.Pow(hidden_states, 2), axes=[-1], keepdims=1)
    epsilon = op.Expand(epsilon, op.Shape(variance))
    hidden_states = hidden_states * op.Reciprocal(op.Sqrt(variance + epsilon))
    return weight * hidden_states


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    def eager(hidden_states, weight):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return weight * (hidden_states * torch.rsqrt(variance + eps))

    return custom_ops.call(CustomRMSNorm, (hidden_states, weight), eager, shape=hidden_states.shape,
                           attrs={"epsilon": float(eps)})
