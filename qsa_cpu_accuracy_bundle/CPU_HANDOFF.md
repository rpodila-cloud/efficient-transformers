# QSA CPU microbenchmark

Portable code-only bundle. Start in `source/` relative to this directory.
Use your own Python3.12+ environment; see
[requirements.txt](./source/layer_benchmarks/requirements.txt). No venv, model,
checkpoint, tensor data, ONNX or accelerator artifacts are included.

- [bench.py](./source/layer_benchmarks/attention/qsa/bench.py): QSA construction, inputs and CLI.
- [forwards.py](./source/layer_benchmarks/attention/qsa/forwards.py): QSA computation/configuration.
- [ref.py](./source/layer_benchmarks/attention/qsa/ref.py): local `dense_qsa_step` reference.
- [cache.py](./source/layer_benchmarks/attention/qsa/cache.py), [ops.py](./source/layer_benchmarks/attention/qsa/ops.py): cache/eager/custom-op definitions.
- [parity.py](./source/layer_benchmarks/attention/qsa/parity.py): recurrent/local-reference, ORT and optional HF checks.

Safe startup check only:

```bash
cd source
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
python3 -m layer_benchmarks.attention.qsa.bench --help
```

Default execution/self-test can create or require ONNX. Separate local
`dense_qsa_step` comparisons from ORT before CPU-only numerical work. Match
weights, inputs and state/position histories. Optional HF helpers need compatible
model code and are not proof of official-checkpoint parity. No new numerical
accuracy result or accepted precision budget is claimed; MSA-specific results
do not apply. Accelerator comparison is future, separately authorized work.
Copied hashes are in [COPY_MANIFEST.json](./COPY_MANIFEST.json).
