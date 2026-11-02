# GQA CPU microbenchmark

Portable code-only bundle. Start in `source/` relative to this directory.
Use your own Python3.12+ environment; dependencies are listed in
[requirements.txt](./source/layer_benchmarks/requirements.txt). No venv, model,
checkpoint, tensor data, ONNX or accelerator artifacts are included.

- [bench.py](./source/layer_benchmarks/attention/gqa/bench.py): construction, inputs and CLI.
- [forwards.py](./source/layer_benchmarks/attention/gqa/forwards.py): GQA decode/prefill implementations.
- [cache.py](./source/layer_benchmarks/attention/gqa/cache.py), [ops.py](./source/layer_benchmarks/attention/gqa/ops.py): cache/eager/custom-op definitions.
- [parity.py](./source/layer_benchmarks/attention/gqa/parity.py): local dense `mode="none"` comparison, state-feedback cases, ORT and HF helpers.

Safe startup check only:

```bash
cd source
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
python3 -m layer_benchmarks.attention.gqa.bench --help
```

Default benchmark execution and `--self-test` can export ONNX; do not invoke
them for a code-only task. Review the CPU/reference portions of `parity.py`
before numerical work. Match actual weights, inputs, positions and logical
caches across layouts. HF helpers do not establish official-checkpoint parity.
No new numerical accuracy result or precision acceptance limit is claimed by
copying/importing this bundle. MSA-specific results/tolerances do not apply here.
Future accelerator comparison needs separate validation and authorization.
Copied hashes are in [COPY_MANIFEST.json](./COPY_MANIFEST.json).
