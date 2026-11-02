# DMA CPU microbenchmark

Portable code-only bundle. Start in `source/` relative to this directory.
Use your own Python3.12+ environment; see
[requirements.txt](./source/layer_benchmarks/requirements.txt). No venv, model,
checkpoint, tensor data, ONNX or accelerator artifacts are included.

- [bench.py](./source/layer_benchmarks/dma/bench.py): `Config`, `DMA`, construction/input generation and CLI.
- [ref.py](./source/layer_benchmarks/dma/ref.py): local `DMAReference` using the same actual constants.

Safe startup check only:

```bash
cd source
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
python3 -m layer_benchmarks.dma.bench --help
```

The CPU computation repeatedly adds FP16 constant tensors. It is not a CPU
measurement of accelerator DMA bandwidth. Match actual constants/input and
operation order for code comparisons. The full self-test also exports ONNX;
do not invoke it for a code-only task. No new numerical result is claimed by
copying/importing this bundle; MSA-specific results/tolerances do not apply.
Hardware bandwidth/accuracy work needs separate authorization.
Copied hashes are in [COPY_MANIFEST.json](./COPY_MANIFEST.json).
