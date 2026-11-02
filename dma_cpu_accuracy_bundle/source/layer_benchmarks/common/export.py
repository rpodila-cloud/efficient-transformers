"""Dynamo ONNX export shared by all benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import onnx
import torch

from . import custom_ops

OPSET = 18


def export(
    model: torch.nn.Module,
    args: Sequence[Any],
    onnx_path: Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    *,
    custom_translation_table: Optional[Dict[Any, Any]] = None,
    dim_params: Optional[Dict[str, Dict[int, str]]] = None,
    dynamic_shapes: Any = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    use_custom_ops: bool = True,
    optimize: bool = True,
    check: bool = True,
) -> Path:
    """Export ``model(*args)`` with ``torch.onnx.export(dynamo=True)``.

    * Weights go to ``<name>.onnx.data`` (torch's external-data layout).
    * QAIC custom ops emitted via ``custom_ops.call`` get their function protos
      attached afterwards (only the ones actually used by the graph).
    * ``dim_params`` renames boundary dims, e.g. ``{"k_cache": {0: "batch_size"}}``.
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad(), custom_ops.enabled(use_custom_ops):
        program = torch.onnx.export(
            model,
            tuple(args),
            dynamo=True,
            opset_version=OPSET,
            optimize=optimize,
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_shapes=dynamic_shapes,
            dynamic_axes=dynamic_axes,
            custom_translation_table=custom_translation_table or {},
        )
    # input_names are applied after tracing, so an internal value can share a
    # name with an input (e.g. an aten.cos node vs a "cos" input). Keep SSA.
    taken = {v.name for v in program.model.graph.inputs}
    for node in program.model.graph:
        for value in node.outputs:
            if value.name in taken and value not in program.model.graph.outputs:
                value.name = f"{value.name}_{node.name}"
    program.save(str(onnx_path), external_data=True)
    if check:
        onnx.checker.check_model(str(onnx_path))
    # Post-edit the graph only; the weights file stays as torch wrote it.
    # (Re-externalizing with onnx>=1.18 would also move Constants inside
    # function bodies to disk, which ORT/QAIC cannot read.)
    proto = onnx.load(str(onnx_path), load_external_data=False)
    used = {(n.domain, n.op_type) for n in _all_nodes(proto)}
    have = {(f.domain, f.name) for f in proto.functions}
    changed = False
    for key, fn in custom_ops.FUNCTIONS.items():
        if key in used and key not in have:
            proto.functions.append(fn.to_function_proto())
            _ensure_opset(proto, key[0])
            changed = True
    for value in list(proto.graph.input) + list(proto.graph.output):
        for axis, name in (dim_params or {}).get(value.name, {}).items():
            dim = value.type.tensor_type.shape.dim[axis]
            dim.ClearField("dim_value")
            dim.dim_param = name
            changed = True
    if changed:
        onnx.save(proto, str(onnx_path))
    return onnx_path


def _all_nodes(proto: onnx.ModelProto) -> Iterable[onnx.NodeProto]:
    yield from proto.graph.node
    for fn in proto.functions:
        yield from fn.node


def _ensure_opset(proto: onnx.ModelProto, domain: str) -> None:
    if not any(o.domain == domain for o in proto.opset_import):
        proto.opset_import.append(onnx.helper.make_opsetid(domain, 1))


def op_histogram(onnx_path: Path) -> Dict[str, int]:
    proto = onnx.load(str(onnx_path), load_external_data=False)
    hist: Dict[str, int] = {}
    for n in proto.graph.node:
        key = f"{n.domain}::{n.op_type}" if n.domain else n.op_type
        hist[key] = hist.get(key, 0) + 1
    return hist
