# HCA CPU microbenchmark

Portable code-only bundle. Start in `source/` relative to this directory.
Use your own Python3.12+ environment; see
[requirements.txt](./source/layer_benchmarks/requirements.txt). No venv, model,
checkpoint, tensor data, ONNX or accelerator artifacts are included.

- [bench.py](./source/layer_benchmarks/attention/hca/bench.py): HCA configuration, inputs and CLI.
- [forwards.py](./source/layer_benchmarks/attention/hca/forwards.py): compressed-attention computation; explicitly select the HCA mode when constructing cases.
- [cache.py](./source/layer_benchmarks/attention/hca/cache.py), [ops.py](./source/layer_benchmarks/attention/hca/ops.py): cache/eager/custom-op definitions.
- [parity.py](./source/layer_benchmarks/attention/hca/parity.py): CSA/HCA cases, ORT state-feedback and optional HF comparison helpers.

Safe startup check only:

```bash
cd source
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
python3 -m layer_benchmarks.attention.hca.bench --help
```

Default execution/self-test can export ONNX. Do not equate ORT tests with an
independent code-only numerical reference. Optional HF helpers need compatible
model code in your environment; no HF overlay/checkpoint is copied or downloaded.
Establish matched inputs, computation/precision scope and justified criteria
before accepting numerical comparisons. No new accuracy result is claimed;
MSA-specific results/tolerances do not apply. Accelerator work is not requested.
Copied hashes are in [COPY_MANIFEST.json](./COPY_MANIFEST.json).
