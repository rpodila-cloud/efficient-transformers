# MoE CPU microbenchmark

Portable code-only bundle. Start in `source/` relative to this directory.
Use your own Python3.12+ environment; see
[requirements.txt](./source/layer_benchmarks/requirements.txt). No venv, model,
checkpoint, tensor data, ONNX or accelerator artifacts are included.

- [bench.py](./source/layer_benchmarks/moe/bench.py): implementation selection, inputs, routing/expert layout and CLI.
- [forwards.py](./source/layer_benchmarks/moe/forwards.py): expert/routing computation and local `Reference`.
- [ops.py](./source/layer_benchmarks/moe/ops.py): custom gather/scatter definitions.
- [parity.py](./source/layer_benchmarks/moe/parity.py): FP32/FP16 cases, ORT and optional HF helpers.

Safe startup check only:

```bash
cd source
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
python3 -m layer_benchmarks.moe.bench --help
```

Default execution/full self-test can export ONNX. Its FP16 execution smoke
coverage is not a justified precision-matched numerical acceptance test.
Compare actual shared expert weights, inputs, routing choices/weights and output
across the explicitly selected implementations. Optional HF helpers do not
establish official-checkpoint parity. No new accuracy result or accepted
precision budget is claimed; MSA-specific results do not apply. Accelerator
comparison is future, separately authorized work.
Copied hashes are in [COPY_MANIFEST.json](./COPY_MANIFEST.json).
