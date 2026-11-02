"""CLI flags and the export → compile → run → perf flow every bench shares."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from . import qaic


def add_qaic_args(p: argparse.ArgumentParser, *, num_devices: int = 1, num_cores: Optional[int] = None) -> None:
    g = p.add_argument_group("qaic")
    g.add_argument("--num-devices", type=int, default=num_devices)
    g.add_argument("--device-group", default=None, help="Physical QIDs: 30,31 or [30,31] or 30..45.")
    g.add_argument("--hw-version", choices=tuple(qaic.HW_CORES_PER_DEVICE), default="ai100")
    g.add_argument("--compile-num-cores", type=int, default=num_cores, help="Default: 16 on ai100, 4 on ai200.")
    g.add_argument("--extra-compiler-args", default="")
    g.add_argument("--compiler-sdk", type=int, default=1, help="0 = use the custom compiler-glow build.")
    g.add_argument("--artifact-dir", type=Path, default=None)
    g.add_argument("--dump-io", action="store_true", help="Write io/ + perf_dump/*.sh.")
    g.add_argument("--run-compile", action="store_true")
    g.add_argument("--run-hw", action="store_true", help="Time session.run on device.")
    g.add_argument("--run-perf", action="store_true", help="compile_perf + run_perf + opstats (implies --dump-io).")
    g.add_argument("--defer-hw", action="store_true",
                   help="Build only (export, compile, perf compile); leave --run-hw/--run-perf device steps in "
                        "hw_plan.json for run_deferred_hw (used by verify to overlap builds with device runs).")
    g.add_argument("--hw-warmup", type=int, default=5)
    g.add_argument("--hw-iters", type=int, default=50)
    g.add_argument("--perf-num-iters", type=int, default=30)
    g.add_argument("--perf-profile-start-iter", type=int, default=20)
    g.add_argument("--perf-num-samples", type=int, default=2)
    g.add_argument("--perf-stats-level", type=int, default=70)
    g.add_argument("--seed", type=int, default=1234)


def finalize_qaic_args(a: argparse.Namespace) -> None:
    if a.compile_num_cores is None:
        a.compile_num_cores = qaic.HW_CORES_PER_DEVICE[a.hw_version]
    if a.run_perf:
        a.dump_io = True
    a.device_ids = qaic.resolve_device_ids(a.num_devices, a.device_group)
    if a.artifact_dir is None:
        raise SystemExit("--artifact-dir is required")
    a.artifact_dir = a.artifact_dir.resolve()
    a.artifact_dir.mkdir(parents=True, exist_ok=True)


def run_qaic(
    a: argparse.Namespace,
    report: Dict[str, Any],
    *,
    onnx_path: Path,
    specialization: Dict[str, Any],
    visible_inputs: Dict[str, np.ndarray],
    visible_outputs: Dict[str, np.ndarray],
    custom_io: Optional[Dict[str, str]] = None,
    compile_kwargs: Optional[Dict[str, Any]] = None,
    compiler_env: Optional[Dict[str, str]] = None,
    perf_compile_env: Optional[Dict[str, str]] = None,
    perf_kwargs: Optional[Dict[str, Any]] = None,
    hw: Optional[Callable[[Path], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Emit compile assets, then honour --dump-io/--run-compile/--run-hw/--run-perf.

    ``visible_*`` are the runtime-visible buffers (retained state excluded).
    ``hw(qpc_dir)`` overrides the default timing (feed dumped IO, skip the rest).
    """
    out = a.artifact_dir
    assets = qaic.emit_assets(
        out,
        onnx_path=onnx_path,
        specialization=specialization,
        num_devices=len(a.device_ids),
        num_cores=a.compile_num_cores,
        custom_io=custom_io,
        compiler_sdk=a.compiler_sdk,
        compiler_env=compiler_env,
        hw_version=a.hw_version,
        extra_args=a.extra_compiler_args,
        **(compile_kwargs or {}),
    )
    report["assets"] = {k: v for k, v in assets.items() if k != "compile_command_list"}
    report["device_ids"] = a.device_ids
    qpc = Path(assets["qpc_dir"])
    io_json = qaic.dump_qaic_io(visible_inputs, visible_outputs, out / "io")
    report["io_json"] = str(io_json)
    if a.dump_io:
        perf_dir = qaic.write_perf_scripts(
            out,
            assets["compile_command_list"],
            io_json,
            device_ids=a.device_ids,
            num_iters=a.perf_num_iters,
            profile_start_iter=a.perf_profile_start_iter,
            num_samples=a.perf_num_samples,
            stats_level=a.perf_stats_level,
            compiler_sdk=a.compiler_sdk,
            compile_env=perf_compile_env,
            **(perf_kwargs or {}),
        )
        report["perf_dir"] = str(perf_dir)
    if a.run_compile or (a.run_hw and not (qpc / "programqpc.bin").is_file()):
        print(f"Compiling: {assets['compile_script']}", flush=True)
        qaic.run_script(assets["compile_script"], out / "compile.log")
    if a.defer_hw:
        if hw:
            raise SystemExit("--defer-hw does not support a custom hw()")
        if a.run_perf:
            qaic.run_script(Path(report["perf_dir"]) / "compile_perf.sh", Path(report["perf_dir"]) / "compile_perf.log")
        plan = {"qpc": str(qpc), "io_json": str(io_json), "device_ids": a.device_ids, "run_hw": a.run_hw,
                "warmup": a.hw_warmup, "iters": a.hw_iters, "perf_dir": report.get("perf_dir") if a.run_perf else None}
        (out / "hw_plan.json").write_text(json.dumps(plan, indent=2))
        return report
    if a.run_hw:
        report["hardware"] = hw(qpc) if hw else qaic.time_from_io(qpc, io_json, a.device_ids, a.hw_warmup, a.hw_iters)
        print(f"HW latency: mean={report['hardware']['mean_ms']:.4f} ms median={report['hardware']['median_ms']:.4f} ms")
    if a.run_perf:
        report["perf"] = qaic.run_perf(Path(report["perf_dir"]))
        print(f"Perf: max_core_pcycles={report['perf'].get('max_core_pcycles')} ddr_mb={report['perf'].get('total_ddr_mb')}")
    return report


def run_deferred_hw(artifact_dir: Path) -> None:
    """Run the device steps a ``--defer-hw`` build left in ``hw_plan.json`` and add them to report.json."""
    artifact_dir = Path(artifact_dir)
    plan = json.loads((artifact_dir / "hw_plan.json").read_text())
    report_path = artifact_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    if plan["run_hw"]:
        report["hardware"] = h = qaic.time_from_io(Path(plan["qpc"]), Path(plan["io_json"]), plan["device_ids"],
                                                   plan["warmup"], plan["iters"])
        print(f"HW latency: mean={h['mean_ms']:.4f} ms median={h['median_ms']:.4f} ms")
    if plan["perf_dir"]:
        perf_dir = Path(plan["perf_dir"])
        for name in ("run_perf.sh", "decode_perf.sh"):
            print(f"Running {perf_dir / name}", flush=True)
            qaic.run_script(perf_dir / name, perf_dir / (name[:-3] + ".log"))
        report["perf"] = p = qaic.parse_opstats(perf_dir / "opstats")
        print(f"Perf: max_core_pcycles={p.get('max_core_pcycles')} ddr_mb={p.get('total_ddr_mb')}")
    report_path.write_text(json.dumps(report, indent=2, default=str))


def write_report(a: argparse.Namespace, report: Dict[str, Any]) -> Path:
    path = a.artifact_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Report: {path}")
    return path
