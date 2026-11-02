"""One pattern for every QAIC custom op in every benchmark.

A custom op is an ``@qaic_function`` onnxscript body (the ONNX function that
ships inside the model) plus an eager PyTorch implementation. ``call`` runs
the eager code normally and emits a ``torch.onnx.ops.symbolic`` node while
Dynamo is exporting; ``common.export`` then attaches the matching function
protos. This mirrors ``QEfficient/customop`` so ops port one-to-one.

Setting ``enabled(False)`` exports the eager decomposition instead, which is
what the ORT self-tests use.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Dict, Sequence, Tuple

import onnxscript
import torch

QAIC_DOMAIN = "com.qti.aisw.onnx"
CLOUD_DOMAIN = "com.qualcomm.cloud"

FUNCTIONS: Dict[Tuple[str, str], onnxscript.OnnxFunction] = {}
_ENABLED = True


def qaic_function(domain: str = QAIC_DOMAIN):
    """Compile an onnxscript function in ``domain`` and register it for export."""

    def wrap(fn: Callable) -> onnxscript.OnnxFunction:
        compiled = onnxscript.script(onnxscript.values.Opset(domain, 1))(fn)
        FUNCTIONS[(domain, compiled.name)] = compiled
        return compiled

    return wrap


@contextlib.contextmanager
def enabled(flag: bool):
    global _ENABLED
    prev, _ENABLED = _ENABLED, flag
    try:
        yield
    finally:
        _ENABLED = prev


def is_enabled() -> bool:
    return _ENABLED


def call(
    fn: onnxscript.OnnxFunction,
    inputs: Sequence[torch.Tensor],
    eager: Callable[..., torch.Tensor],
    *,
    shape: Sequence[int],
    dtype: torch.dtype | None = None,
    attrs: Dict[str, object] | None = None,
) -> torch.Tensor:
    """Eager ``eager(*inputs)``; under export, a ``domain::fn`` node of ``shape``."""
    if _ENABLED and torch.compiler.is_exporting():
        return torch.onnx.ops.symbolic(
            f"{fn.opset.domain}::{fn.name}",
            tuple(inputs),
            attrs or {},
            dtype=dtype if dtype is not None else inputs[0].dtype,
            shape=tuple(shape),
            version=1,
        )
    return eager(*inputs)


def call_multi(
    fn: onnxscript.OnnxFunction,
    inputs: Sequence[torch.Tensor],
    eager: Callable[..., Tuple[torch.Tensor, ...]],
    *,
    shapes: Sequence[Sequence[int]],
    dtypes: Sequence[torch.dtype],
    attrs: Dict[str, object] | None = None,
) -> Tuple[torch.Tensor, ...]:
    if _ENABLED and torch.compiler.is_exporting():
        return tuple(
            torch.onnx.ops.symbolic_multi_out(
                f"{fn.opset.domain}::{fn.name}",
                tuple(inputs),
                attrs or {},
                dtypes=list(dtypes),
                shapes=[tuple(s) for s in shapes],
                version=1,
            )
        )
    return eager(*inputs)
