# MLA CPU microbenchmark

Portable code-only bundle. Start in `source/` relative to this directory.
Use your own Python3.12+ environment; see
[requirements.txt](./source/layer_benchmarks/requirements.txt). No venv, model,
checkpoint, tensor data, ONNX or accelerator artifacts are included.

- [bench.py](./source/layer_benchmarks/attention/mla/bench.py): inputs, export interface and CLI.
- [forwards.py](./source/layer_benchmarks/attention/mla/forwards.py): MLA construction and blocking modes.
- [ref.py](./source/layer_benchmarks/attention/mla/ref.py): local `MLAReference`.
- [cache.py](./source/layer_benchmarks/attention/mla/cache.py), [ops.py](./source/layer_benchmarks/attention/mla/ops.py): cache/eager/custom-op definitions.
- [parity.py](./source/layer_benchmarks/attention/mla/parity.py): `_vs_reference` CPU checks and separate ORT/export checks.

Safe startup check only:

```bash
cd source
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
python3 -m layer_benchmarks.attention.mla.bench --help
```

Default benchmark execution and full self-test can export ONNX. The local
reference comparison is distinct from `_vs_ort`; do not run the combined
self-test for a code-only task. Match actual parameters, inputs and cache
contents when comparing supported modes. No new numerical accuracy result or
precision acceptance limit is claimed by copying/importing this bundle.
Do not transfer MSA-specific results/tolerances to MLA. Accelerator comparison
is future, separately authorized work.
Copied hashes are in [COPY_MANIFEST.json](./COPY_MANIFEST.json).
