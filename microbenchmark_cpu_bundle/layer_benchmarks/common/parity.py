"""Numerical parity helpers: eager vs reference, eager vs ORT."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch


def _np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def compare(name: str, actual: Any, expected: Any, *, atol: float, rtol: float) -> Dict[str, Any]:
    a, e = _np(actual), _np(expected)
    if a.shape != e.shape:
        return {"name": name, "passed": False, "error": f"shape {a.shape} != {e.shape}"}
    diff = np.abs(a - e)
    finite = bool(np.isfinite(a).all() and np.isfinite(e).all())
    return {
        "name": name,
        "passed": finite and bool(np.allclose(a, e, atol=atol, rtol=rtol)),
        "max_abs": float(diff.max(initial=0.0)),
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "atol": atol,
        "rtol": rtol,
    }


def compare_all(actual: Mapping[str, Any], expected: Mapping[str, Any], *, atol: float, rtol: float) -> Dict[str, Any]:
    checks = [compare(k, actual[k], expected[k], atol=atol, rtol=rtol) for k in expected]
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def report(title: str, result: Dict[str, Any]) -> None:
    status = "PASSED" if result["passed"] else "FAILED"
    print(f"Parity [{title}]: {status}")
    for c in result.get("checks", [result]):
        if "error" in c:
            print(f"  {c['name']:<32} {c['error']}")
        else:
            mean_abs = c.get("mean_abs")
            suffix = f" mean_abs={mean_abs:.3e}" if mean_abs is not None else ""
            print(f"  {c['name']:<32} max_abs={c.get('max_abs', float('nan')):.3e}{suffix}")


def ort_run(onnx_path: Path, feed: Mapping[str, np.ndarray], output_names: Sequence[str] | None = None) -> Dict[str, np.ndarray]:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    outs = sess.run(list(output_names) if output_names else None, {n: feed[n] for n in names})
    return dict(zip(output_names or [o.name for o in sess.get_outputs()], outs))


def require(result: Dict[str, Any], title: str) -> None:
    report(title, result)
    if not result["passed"]:
        raise SystemExit(f"parity failed: {title}")
