"""Runtime dependency checks for the layer benchmark package."""

from __future__ import annotations

import importlib.metadata as metadata
import re
import sys
from pathlib import Path


MIN_PYTHON = (3, 12)
MIN_TORCH = (2, 13, 0)
REQUIRED_DISTRIBUTIONS = (
    "torch",
    "torchvision",
    "onnx",
    "onnxscript",
    "onnxruntime",
    "numpy",
    "transformers",
)


def _numeric_version(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    if not match:
        return ()
    parts = tuple(int(part) for part in match.group(1).split("."))
    return parts + (0,) * max(0, len(MIN_TORCH) - len(parts))


def requirements_path() -> Path:
    return Path(__file__).with_name("requirements.txt")


def check_runtime() -> list[str]:
    errors: list[str] = []
    if sys.version_info[:2] < MIN_PYTHON:
        errors.append(
            f"Python 3.12+ is required (found {sys.version_info.major}.{sys.version.minor})."
        )
    installed: dict[str, str] = {}
    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            installed[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            errors.append(f"Missing package: {distribution}.")
    torch_version = installed.get("torch")
    if torch_version and _numeric_version(torch_version) < MIN_TORCH:
        errors.append(f"torch>=2.13.0 is required (found {torch_version}).")
    return errors


def require_runtime() -> None:
    errors = check_runtime()
    if not errors:
        return
    install = f"{sys.executable} -m pip install -r {requirements_path()}"
    message = [
        "layer_benchmarks cannot start because its Python environment is not ready.",
        "Use Python 3.12+ and install the benchmark dependencies with:",
        f"  {install}",
        "",
        "Dependency check:",
        *(f"  - {error}" for error in errors),
    ]
    raise SystemExit("\n".join(message))
