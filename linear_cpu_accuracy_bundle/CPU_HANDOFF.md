# Linear-attention CPU microbenchmark

Portable code-only bundle. Start in `source/` relative to this directory.
Use your own Python3.12+ environment; see
[requirements.txt](./source/layer_benchmarks/requirements.txt). No venv, model,
checkpoint, tensor data, ONNX or accelerator artifacts are included.

- [bench.py](./source/layer_benchmarks/attention/linear/bench.py): construction, inputs and CLI.
- [forwards.py](./source/layer_benchmarks/attention/linear/forwards.py): Qwen3.5 chunk/recurrent linear-attention paths.
- [cache.py](./source/layer_benchmarks/attention/linear/cache.py), [ops.py](./source/layer_benchmarks/attention/linear/ops.py): recurrent state and eager/custom-op definitions.
- [parity.py](./source/layer_benchmarks/attention/linear/parity.py): chunk/recurrent cases, ORT and optional HF helpers.

Safe startup check only:

```bash
cd source
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
python3 -m layer_benchmarks.attention.linear.bench --help
```

Default execution/self-test can export ONNX. Optional HF comparison code needs
compatible model implementations; no overlay/checkpoint is bundled or fetched.
For code-only work, establish the reference and match weights, activations,
convolution/recurrent state and sequence boundaries before comparing paths.
No new numerical accuracy result or precision acceptance limit is claimed;
MSA-specific results/tolerances do not apply. Accelerator comparison is future,
separately authorized work.
Copied hashes are in [COPY_MANIFEST.json](./COPY_MANIFEST.json).
