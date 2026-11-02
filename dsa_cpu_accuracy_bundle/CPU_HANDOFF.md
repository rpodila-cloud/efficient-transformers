# DSA CPU microbenchmark

Portable code-only bundle. Start in `source/` relative to this directory.
Use your own Python3.12+ environment; see
[requirements.txt](./source/layer_benchmarks/requirements.txt). No venv, model,
checkpoint, tensor data, ONNX or accelerator artifacts are included.

- [bench.py](./source/layer_benchmarks/attention/dsa/bench.py): model/input construction and CLI.
- [forwards.py](./source/layer_benchmarks/attention/dsa/forwards.py): GLM DSA indexer/attention paths and configuration.
- [cache.py](./source/layer_benchmarks/attention/dsa/cache.py), [ops.py](./source/layer_benchmarks/attention/dsa/ops.py): cache/eager/custom-op definitions.
- [parity.py](./source/layer_benchmarks/attention/dsa/parity.py): dense-equation reference, DP/CP layout, selection, cache and ORT checks.

Safe startup check only:

```bash
cd source
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
python3 -m layer_benchmarks.attention.dsa.bench --help
```

Default execution/full self-test can export ONNX. Review and separate the
CPU/reference portions before code-only numerical work. Equation-based local
reference code is not by itself official-checkpoint validation. Compare matched
weights/logical inputs, TopK sets, output and every cache across supported paths.
No new numerical accuracy result or accepted precision budget is claimed here;
MSA-specific results/tolerances do not apply. Accelerator comparison is future,
separately authorized work.
Copied hashes are in [COPY_MANIFEST.json](./COPY_MANIFEST.json).
