#!/usr/bin/env python3
"""DMA microbench: stream ``num_xfers`` fp16 constants of ``xfer_size`` bytes per core through Adds.

    python -m layer_benchmarks.dma.bench --self-test --artifact-dir /tmp/dma
    python -m layer_benchmarks.dma.bench --device-group 30 --artifact-dir out/dma --dump-io --run-compile --run-perf

See README.md.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from layer_benchmarks.common import cli, export, parity, qaic

INPUT_NAMES, OUTPUT_NAMES = ["in"], ["out"]


@dataclass
class Config:
    xfer_size: int
    num_xfers: int
    num_cores: int

    def xfer_shape(self) -> int:
        return int(self.xfer_size / 2 * self.num_cores)  # fp16 elements across all cores


class DMA(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.cfg = c
        # Plain tensors (not buffers), drawn in the original order, so they export as initializers.
        self.constants = [torch.randn(c.xfer_shape(), dtype=torch.float16) for _ in range(c.num_xfers)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c in self.constants:
            x = x + c
        return x


def build(c: Config, seed: int):
    """(module, input) with the original's RNG order: constants first, then the input."""
    torch.manual_seed(seed)
    model = DMA(c).eval()
    return model, torch.randn(c.xfer_shape(), dtype=torch.float16)


def export_onnx(model: DMA, x: torch.Tensor, path: Path) -> Path:
    return export.export(model, (x,), path, INPUT_NAMES, OUTPUT_NAMES)


def self_test(out_dir: Path) -> dict:
    """Eager vs ORT on a small and the default config."""
    checks = []
    for xfer_size, num_xfers, cores in ((4096, 3, 4), (128 * 1024, 10, 16)):
        c = Config(xfer_size, num_xfers, cores)
        model, x = build(c, 1234)
        with torch.no_grad():
            want = model(x)
            from .ref import DMAReference
            reference = DMAReference(model.constants)
            parity.require(parity.compare("eager vs reference", want, reference(x), atol=0.0, rtol=0.0),
                          f"dma reference x{xfer_size}")
        path = export_onnx(model, x, out_dir / f"dma_x{xfer_size}_n{num_xfers}_c{cores}.onnx")
        got = parity.ort_run(path, {"in": x.numpy()})["out"]
        # ORT's CPU fp16 Add is not correctly rounded (eager is); allow ~1 fp16 ulp.
        checks.append(parity.compare(f"x{xfer_size}_n{num_xfers}_c{cores}_vs_ort", torch.from_numpy(got), want,
                                     atol=1e-2, rtol=2e-3))
        hist = export.op_histogram(path)
        checks.append({"name": f"x{xfer_size}_n{num_xfers}_c{cores}_ops", "max_abs": 0.0, "mean_abs": 0.0,
                       "ops": hist, "passed": hist == {"Add": num_xfers}})
    result = {"passed": all(r["passed"] for r in checks), "checks": checks}
    parity.require(result, "dma self-test")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xfer-size", type=int, default=128 * 1024, help="Bytes per core per transfer.")
    p.add_argument("--num-xfers", type=int, default=10)
    p.add_argument("--num-cores", type=int, help="Alias of --compile-num-cores (original flag).")
    p.add_argument("--json", action="store_true", help="Print the report JSON.")
    p.add_argument("--self-test", action="store_true", help="Eager vs ORT on small configs, then exit.")
    cli.add_qaic_args(p, num_devices=1)
    a = p.parse_args()
    if a.num_cores is not None:
        a.compile_num_cores = a.num_cores
    if a.self_test:
        if a.artifact_dir is None:
            p.error("--artifact-dir is required")
        a.artifact_dir = a.artifact_dir.resolve()
        a.artifact_dir.mkdir(parents=True, exist_ok=True)
        return a
    cli.finalize_qaic_args(a)
    return a


def main() -> None:
    a = parse_args()
    if a.self_test:
        print(json.dumps(self_test(a.artifact_dir), indent=2))
        return
    c = Config(a.xfer_size, a.num_xfers, a.compile_num_cores)
    model, x = build(c, a.seed)
    with torch.no_grad():
        out = model(x)
    onnx_path = export_onnx(model, x, a.artifact_dir / "dma_bench.onnx")
    report = {"config": asdict(c), "bytes_per_xfer": c.xfer_shape() * 2, "onnx": str(onnx_path),
              "ops": export.op_histogram(onnx_path)}
    cli.run_qaic(
        a,
        report,
        onnx_path=onnx_path,
        specialization={"batch_size": 1},  # the graph is static; qaic-compile still wants one specialization
        visible_inputs={"in": x.numpy()},
        visible_outputs={"out": out.numpy()},
    )
    cli.write_report(a, report)
    if a.json:
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
