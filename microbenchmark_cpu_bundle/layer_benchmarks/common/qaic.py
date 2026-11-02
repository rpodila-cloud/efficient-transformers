"""QAIC compile/runtime harness shared by every layer benchmark.

Everything here is model-agnostic: device parsing, compile assets, IO dumps
for qaic-runner, perf scripts, the qaicrt session, and timing/perf parsing.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

COMPILER = "/opt/qti-aic/exec/qaic-compile"
# Every compile (baseline and port, normal and perf) runs with these, so numbers stay comparable.
COMPILER_OPTS_UNSUPPORTED = "-aic-hoist-vtcm-loads=false -aic-op-stats-verbosity 2 -aic-userdma-async=0 -aic-hmx-async=0 -aic-sync-ts-starts"
RUNNER = "/opt/qti-aic/exec/qaic-runner"
OPSTATS = "/opt/qti-aic/exec/qaic-opstats"
HW_CORES_PER_DEVICE = {"ai100": 16, "ai200": 4}
# --compiler-sdk 0 points qaic-compile at this custom compiler build.
COMPILER_SDK_LIB_DIR = (
    "/local/mnt/p_drive/qranium/users/chullee/compiler-glow/build/"
    "compiler-qualcomm-release-assert/lib/QAicCompilerManagers"
)
RETAINED_SUFFIX = "_RetainedState"

try:
    import qaicrt
except ImportError:
    sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
    try:
        import qaicrt
    except ImportError:
        qaicrt = None
try:
    import QAicApi_pb2 as aicapi
except ImportError:
    sys.path.append("/opt/qti-aic/dev/python")
    try:
        import QAicApi_pb2 as aicapi
    except ImportError:
        aicapi = None


# ─────────────────────────────────────────────────────────────────────────────
# Runtime session.
# ─────────────────────────────────────────────────────────────────────────────
class QAICInferenceSession:
    def __init__(self, qpc_path: Path | str, device_ids: Optional[List[int]] = None):
        if qaicrt is None or aicapi is None:
            raise ImportError("qaicrt and/or QAicApi_pb2 are unavailable.")
        self.np_dtype = {
            aicapi.FLOAT_TYPE: np.dtype(np.float32),
            aicapi.FLOAT_16_TYPE: np.dtype(np.float16),
            aicapi.INT8_Q_TYPE: np.dtype(np.int8),
            aicapi.UINT8_Q_TYPE: np.dtype(np.uint8),
            aicapi.INT16_Q_TYPE: np.dtype(np.int16),
            aicapi.INT32_Q_TYPE: np.dtype(np.int32),
            aicapi.INT32_I_TYPE: np.dtype(np.int32),
            aicapi.INT64_I_TYPE: np.dtype(np.int64),
            aicapi.INT8_TYPE: np.dtype(np.int8),
        }
        if hasattr(aicapi, "BFLOAT16_TYPE"):
            # BF16 runtime buffers use uint16 storage without a numeric conversion.
            self.np_dtype[aicapi.BFLOAT16_TYPE] = np.dtype(np.uint16)
        if device_ids:
            self.context = qaicrt.Context(qaicrt.QIDList(device_ids))
            self.queue = qaicrt.Queue(self.context, device_ids[0])
        else:
            self.context = qaicrt.Context()
            self.queue = qaicrt.Queue(self.context, 0)
        qpc = qaicrt.Qpc(str(qpc_path))
        status, iodesc_data = qpc.getIoDescriptor()
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to getIoDescriptor")
        iodesc = aicapi.IoDesc()
        iodesc.ParseFromString(bytes(iodesc_data))
        self.bindings = iodesc.selected_set.bindings
        self.binding_index_map = {b.name: b.index for b in self.bindings}
        props = qaicrt.QAicProgramProperties()
        if hasattr(props, "SubmitRetryTimeoutMs"):
            props.SubmitRetryTimeoutMs = 60_000
        if hasattr(props, "dataPathTimeoutMs"):
            props.dataPathTimeoutMs = 60_000
        if device_ids and len(device_ids) > 1:
            props.devMapping = ":".join(map(str, device_ids))
        self.program = qaicrt.Program(self.context, None, qpc, props)
        if self.program.load() != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to load program")
        self.program.activate()
        self.execObj = qaicrt.ExecObj(self.context, self.program)
        self.qbuffers = [qaicrt.QBuffer(bytes(b.size)) for b in self.bindings]
        self.buf_dims = qaicrt.BufferDimensionsVecRef(
            [(self.np_dtype[b.type].itemsize, list(b.dims)) for b in self.bindings]
        )
        self._skipped_buffers = set()

    @property
    def input_names(self) -> List[str]:
        return [b.name for b in self.bindings if b.dir == aicapi.BUFFER_IO_TYPE_INPUT]

    @property
    def output_names(self) -> List[str]:
        return [b.name for b in self.bindings if b.dir == aicapi.BUFFER_IO_TYPE_OUTPUT]

    def binding_dtype(self, name: str) -> np.dtype:
        return self.np_dtype[self.bindings[self.binding_index_map[name]].type]

    def set_buffers(self, buffers: Dict[str, np.ndarray]) -> None:
        for name, buf in buffers.items():
            if name not in self.binding_index_map:
                continue
            idx = self.binding_index_map[name]
            self.qbuffers[idx] = qaicrt.QBuffer(buf.tobytes())
            self.buf_dims[idx] = (buf.itemsize, buf.shape if buf.shape else (1,))

    def skip_buffers(self, names: Iterable[str]) -> None:
        # QBuffer objects are allocated at their descriptor sizes in __init__.
        # Leaving omitted bindings untouched supplies zero-filled retained
        # state and, importantly, preserves the required dimensions.  Replacing
        # them with zero-byte buffers makes setData reject retained-state
        # outputs on graphs whose state shape is non-empty.  The runtime may
        # return no payload for omitted retained-state outputs, so omit them
        # from result extraction too.
        self._skipped_buffers.update(name for name in names if name in self.binding_index_map)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.set_buffers(inputs)
        if self.execObj.setData(self.qbuffers, self.buf_dims) != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to setData")
        if self.queue.enqueue(self.execObj) != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to enqueue")
        if self.execObj.waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            raise ValueError("Failed to run")
        status, out_buffers = self.execObj.getData()
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to getData")
        outputs = {}
        for name in self.output_names:
            if name in self._skipped_buffers:
                continue
            idx = self.binding_index_map[name]
            outputs[name] = np.frombuffer(
                bytes(out_buffers[idx]), self.np_dtype[self.bindings[idx].type]
            ).reshape(self.buf_dims[idx][1])
        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Devices.
# ─────────────────────────────────────────────────────────────────────────────
def parse_device_group(device_group: str | None) -> List[int]:
    """Parse ``[30,31]``, ``30,31`` or ``30..45``."""
    text = (device_group or "").strip().strip("[]").strip()
    ids: List[int] = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        if ".." in part:
            lo, hi = part.split("..")
            ids += list(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def resolve_device_ids(num_devices: int, device_group: str | None) -> List[int]:
    """--device-group wins when given; otherwise expand 0..num_devices-1."""
    parsed = parse_device_group(device_group)
    if parsed:
        if num_devices not in (1, len(parsed)):
            raise ValueError(
                f"--num-devices ({num_devices}) disagrees with --device-group ({device_group})."
            )
        return parsed
    return list(range(num_devices)) if num_devices > 1 else [0]


def runner_device_arg(device_ids: List[int]) -> List[str]:
    if not device_ids:
        return []
    if len(device_ids) == 1:
        return ["-d", str(device_ids[0])]
    if device_ids == list(range(device_ids[0], device_ids[-1] + 1)):
        return ["-D", f"{device_ids[0]}..{device_ids[-1]}"]
    return ["-D", ",".join(map(str, device_ids))]


# ─────────────────────────────────────────────────────────────────────────────
# Compile assets.
# ─────────────────────────────────────────────────────────────────────────────
def write_specializations(path: Path, **kwargs: Any) -> Path:
    path.write_text(json.dumps({"specializations": [{k: str(v) for k, v in kwargs.items()}]}, indent=2))
    return path


def write_mdp_ts(path: Path, num_devices: int, num_cores: int) -> Optional[Path]:
    """MDP partition over logical devices 0..N-1 (physical QIDs map at runtime)."""
    if num_devices <= 1:
        return None
    devices = [{"deviceId": i, "numCores": num_cores} for i in range(num_devices)]
    payload = {
        "connections": [{"devices": list(range(num_devices)), "type": "p2p"}],
        "partitions": [{"name": "Partition0", "devices": devices}],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_custom_io(path: Path, mapping: Dict[str, str]) -> Path:
    path.write_text("\n".join(f" - IOName: {n}\n   Precision: {p}\n" for n, p in mapping.items()) + "\n")
    return path


def retained_custom_io(names: Iterable[str], precision: str, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """custom_io entries for a retained input and its ``_RetainedState`` output."""
    overrides = overrides or {}
    mapping: Dict[str, str] = {}
    for name in names:
        p = overrides.get(name, precision)
        mapping[name] = p
        mapping[name + RETAINED_SUFFIX] = p
    return mapping


def compile_command(
    *,
    onnx_path: Path,
    qpc_dir: Path,
    specialization_json: Path,
    num_cores: int,
    hw_version: str = "ai100",
    custom_io_yaml: Optional[Path] = None,
    mdp_ts_json: Optional[Path] = None,
    retained_state: bool = True,
    convert_to_fp16: bool = True,
    mxfp6_matmul: bool = True,
    user_tiled: bool = True,
    allow_mxint8_mdp_io: bool = False,
    extra_flags: Iterable[str] = (),
    extra_args: str = "",
    compiler: str = COMPILER,
) -> List[str]:
    cmd = [compiler, "-aic-hw", f"-aic-hw-version={hw_version}", f"-m={onnx_path}"]
    cmd += ["-retained-state"] if retained_state else []
    cmd += ["-convert-to-fp16"] if convert_to_fp16 else []
    cmd += ["-mxfp6-matmul"] if mxfp6_matmul else []
    cmd += [
        f"-aic-num-cores={num_cores}",
        f"-network-specialization-config={specialization_json}",
        f"-aic-binary-dir={qpc_dir}",
    ]
    cmd += ["-user-tiled"] if user_tiled else []
    cmd += [f"-mdp-load-partition-config={mdp_ts_json}"] if mdp_ts_json else []
    cmd += [f"-custom-IO-list-file={custom_io_yaml}"] if custom_io_yaml else []
    cmd += ["-allow-mxint8-mdp-io"] if allow_mxint8_mdp_io else []
    cmd += list(extra_flags)
    cmd += shlex.split(extra_args)
    return cmd


def env_lines(compiler_sdk: int = 1, env: Optional[Dict[str, str]] = None) -> str:
    lines = ""
    if compiler_sdk == 0:
        lines += f"export AIC_COMPILER_LIB_DIR={COMPILER_SDK_LIB_DIR}\n"
        lines += f"export QAIC_COMPILER_LIB={COMPILER_SDK_LIB_DIR}/libQAicCompiler.so\n"
    # An explicit QAIC_COMPILER_OPTS_UNSUPPORTED (e.g. a verify case's golden opts) wins over the default.
    opts = os.environ.get("QAIC_COMPILER_OPTS_UNSUPPORTED", COMPILER_OPTS_UNSUPPORTED)
    env = {"QAIC_COMPILER_OPTS_UNSUPPORTED": opts, **(env or {})}
    for key, value in env.items():
        lines += f"export {key}={shlex.quote(value)}\n"
    return lines


def write_script(path: Path, body: str, env: str = "") -> Path:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + env + body + "\n")
    os.chmod(path, 0o755)
    return path


def emit_assets(
    artifact_dir: Path,
    *,
    onnx_path: Path,
    specialization: Dict[str, Any],
    num_devices: int,
    num_cores: int,
    custom_io: Optional[Dict[str, str]] = None,
    compiler_sdk: int = 1,
    compiler_env: Optional[Dict[str, str]] = None,
    **compile_kwargs: Any,
) -> Dict[str, Any]:
    """Write specializations/mdp/custom_io + compile.sh and return the command."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    spec = write_specializations(artifact_dir / "specializations.json", **specialization)
    mdp = write_mdp_ts(artifact_dir / "mdp_ts_config.json", num_devices, num_cores)
    cio = write_custom_io(artifact_dir / "custom_io.yaml", custom_io) if custom_io else None
    qpc_dir = artifact_dir / "qpc"
    cmd = compile_command(
        onnx_path=onnx_path,
        qpc_dir=qpc_dir,
        specialization_json=spec,
        num_cores=num_cores,
        custom_io_yaml=cio,
        mdp_ts_json=mdp,
        **compile_kwargs,
    )
    script = write_script(
        artifact_dir / "compile.sh",
        f"rm -rf {qpc_dir}\n" + shlex.join(cmd),
        env_lines(compiler_sdk, compiler_env),
    )
    return {
        "onnx_path": str(onnx_path),
        "specializations_json": str(spec),
        "mdp_ts_json": str(mdp) if mdp else None,
        "custom_io_yaml": str(cio) if cio else None,
        "compile_script": str(script),
        "compile_command": shlex.join(cmd),
        "compile_command_list": cmd,
        "qpc_dir": str(qpc_dir),
    }


# ─────────────────────────────────────────────────────────────────────────────
# IO dump for qaic-runner --aic-batch-json-input.
# ─────────────────────────────────────────────────────────────────────────────
def dump_qaic_io(inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray], io_dir: Path) -> Path:
    """Write runtime-visible buffers as .raw + aic_batch_io.json.

    Callers pass only the visible IO; retained state stays device-resident.
    """
    data_dir = io_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for direction, arrays in (("in", inputs), ("out", outputs)):
        for name, arr in arrays.items():
            arr = np.ascontiguousarray(arr)
            fname = name.replace(".", "_") + ".raw"
            arr.tofile(data_dir / fname)
            entries.append(
                {"path": f"data/{fname}", "io-direction": direction, "elem-size": arr.itemsize, "map-to": name, "dims": list(arr.shape)}
            )
    path = io_dir / "aic_batch_io.json"
    path.write_text(json.dumps({"IO-files": [entries]}, indent=2))
    return path


def load_qaic_io(io_json: Path) -> Dict[str, Dict[str, Any]]:
    """Inverse of dump_qaic_io: {name: {direction, path, dims, elem_size}}."""
    entries = json.loads(Path(io_json).read_text())["IO-files"][0]
    root = Path(io_json).parent
    return {
        e["map-to"]: {"direction": e["io-direction"], "path": root / e["path"], "dims": e["dims"], "elem_size": e["elem-size"]}
        for e in entries
    }


# ─────────────────────────────────────────────────────────────────────────────
# Perf scripts.
# ─────────────────────────────────────────────────────────────────────────────
def write_perf_scripts(
    artifact_dir: Path,
    compile_cmd: List[str],
    io_json: Path,
    *,
    device_ids: List[int],
    num_iters: int = 30,
    profile_start_iter: int = 20,
    num_samples: int = 2,
    stats_level: int = 70,
    perf_metrics: bool = True,
    compiler_sdk: int = 1,
    compile_env: Optional[Dict[str, str]] = None,
) -> Path:
    perf_dir = artifact_dir / "perf_dump"
    # Own QPC dir, so the perf build never replaces the timing QPC and both can be built before any device step.
    qpc_dir = perf_dir / "qpc"
    stats_dir = perf_dir / "raw_device_stats"
    opstats_dir = perf_dir / "opstats"
    for d in (stats_dir, opstats_dir):
        d.mkdir(parents=True, exist_ok=True)
    perf_compile = [f"-aic-binary-dir={qpc_dir}" if c.startswith("-aic-binary-dir=") else c for c in compile_cmd]
    perf_compile += [f"-stats-level={stats_level}", "-ddr-stats", "-aic-pmu-recipe=KernelUtil"]
    perf_compile += ["-aic-perf-metrics"] if perf_metrics else []
    runner = [RUNNER, *runner_device_arg(device_ids), "-t", qpc_dir, "-n", num_iters,
              "--aic-profiling-type", "raw_device_stats",
              "--aic-profiling-start-iter", profile_start_iter,
              "--aic-profiling-num-samples", num_samples,
              "--aic-profiling-out-dir", stats_dir,
              "--aic-batch-json-input", io_json]
    opstats = [OPSTATS, "--qpc", qpc_dir / "programqpc.bin", "--input-dir", stats_dir,
               "--output-dir", opstats_dir, "--summary", "--trace"]
    join = lambda c: shlex.join(str(x) for x in c)  # noqa: E731
    write_script(perf_dir / "compile_perf.sh", f"rm -rf {qpc_dir}\n" + join(perf_compile), env_lines(compiler_sdk, compile_env))
    write_script(perf_dir / "run_perf.sh", f"rm -rf {stats_dir}/* {opstats_dir}/*\n" + join(runner))
    write_script(perf_dir / "decode_perf.sh", join(opstats))
    return perf_dir


# ─────────────────────────────────────────────────────────────────────────────
# Execution + measurement.
# ─────────────────────────────────────────────────────────────────────────────
def run_script(path: Path | str, log: Optional[Path] = None) -> str:
    proc = subprocess.run(["bash", str(path)], capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if log is not None:
        log.write_text(out)
    if proc.returncode != 0:
        raise RuntimeError(f"{path} failed ({proc.returncode}):\n{out[-4000:]}")
    return out


def run_perf(perf_dir: Path) -> Dict[str, Any]:
    for name in ("compile_perf.sh", "run_perf.sh", "decode_perf.sh"):
        print(f"Running {perf_dir / name}", flush=True)
        run_script(perf_dir / name, perf_dir / (name[:-3] + ".log"))
    return parse_opstats(perf_dir / "opstats")


def time_session(
    qpc_dir: Path,
    feed: Dict[str, np.ndarray],
    device_ids: List[int],
    *,
    skip: Iterable[str] = (),
    warmup: int = 5,
    iters: int = 50,
    prepare: Optional[Any] = None,
) -> Dict[str, Any]:
    """Time session.run(feed); ``prepare(session)`` runs untimed before each step."""
    session = QAICInferenceSession(qpc_dir, device_ids=device_ids)
    session.skip_buffers(skip)
    for _ in range(warmup):
        if prepare:
            prepare(session)
        session.run(feed)
    timings = []
    for _ in range(iters):
        if prepare:
            prepare(session)
        t0 = time.perf_counter()
        session.run(feed)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "warmup": warmup,
        "iterations": iters,
        "device_ids": device_ids,
    }


def feed_from_io(session: QAICInferenceSession, io_json: Path) -> Dict[str, np.ndarray]:
    """Load the dumped visible inputs with the QPC binding dtypes."""
    feed = {}
    for name, e in load_qaic_io(io_json).items():
        if e["direction"] == "in" and name in session.binding_index_map:
            feed[name] = np.fromfile(e["path"], dtype=session.binding_dtype(name)).reshape(e["dims"])
    return feed


def time_from_io(qpc_dir: Path, io_json: Path, device_ids: List[int], warmup: int = 5, iters: int = 50) -> Dict[str, Any]:
    """Uniform decode timing: feed dumped IO, skip every binding not in it."""
    probe = QAICInferenceSession(qpc_dir, device_ids=device_ids)
    feed = feed_from_io(probe, io_json)
    visible = set(load_qaic_io(io_json))
    skip = [b.name for b in probe.bindings if b.name not in visible]
    del probe
    return time_session(qpc_dir, feed, device_ids, skip=skip, warmup=warmup, iters=iters)


def runner_throughput(qpc_dir: Path, io_json: Path, device_ids: List[int], iters: int = 100) -> Dict[str, Any]:
    cmd = [RUNNER, *runner_device_arg(device_ids), "-t", str(qpc_dir), "-n", str(iters), "--aic-batch-json-input", str(io_json)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"TotalDuration\s+(\d+)us.*?Inf/Sec\s+([\d.]+)", proc.stdout + proc.stderr)
    if proc.returncode != 0 or not m:
        raise RuntimeError(f"qaic-runner failed:\n{(proc.stdout + proc.stderr)[-3000:]}")
    return {"inf_per_sec": float(m.group(2)), "avg_us": int(m.group(1)) / iters, "iters": iters}


def parse_opstats(opstats_dir: Path) -> Dict[str, Any]:
    """Aggregate qaic-opstats summaries: max per-core pcycles and total DDR MB."""
    per_slice = {}
    for f in sorted(Path(opstats_dir).glob("*.summary.txt")):
        text = f.read_text()
        pcycles = [int(x) for x in re.findall(r"Total inference pcycle is (\d+)", text)]
        ddr = [float(x) for x in re.findall(r"DDR traffic count ([\d.]+) MB", text)]
        m = re.search(r"Specialization: (\S+)", text)
        key = m.group(1) if m else f.name
        per_slice[key] = {"max_core_pcycles": max(pcycles, default=0), "ddr_mb": sum(ddr)}
    if not per_slice:
        return {}
    return {
        "max_core_pcycles": max(v["max_core_pcycles"] for v in per_slice.values()),
        "total_ddr_mb": sum(v["ddr_mb"] for v in per_slice.values()),
        "slices": per_slice,
    }


def mxint8_storage_shape(shape: Iterable[int]) -> tuple:
    dims = list(shape)
    dims[-1] = max(1, (dims[-1] + 1) // 2)
    return tuple(dims)


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
