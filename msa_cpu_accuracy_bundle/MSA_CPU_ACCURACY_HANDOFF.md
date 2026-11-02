# MSA CPU accuracy handoff

## Scope and paths

All paths are relative to the **bundle root**, wherever you copy/extract it.
Work in **`source/`**; no particular server directory is required.
This bundle contains the necessary source/import dependencies and CPU tests.
There is **no venv**, downloaded checkpoint, saved tensor/cache data, ONNX model,
compiler output or hardware campaign. Original source/run directories are unchanged.
Use your server's Python3.12+ environment with the benchmark dependencies.
[ENVIRONMENT.json](./ENVIRONMENT.json) records the versions used for the saved
test result; [requirements.txt](./source/layer_benchmarks/requirements.txt)
contains the upstream dependency list. No Python environment is bundled.

Target: `msa_indexer_attention_GP_unified`, non-paged single-token decode.
Boundary: indexer projections, normalization/RoPE, cache update, scoring and
block selection; main Q/K/V projections and cache updates; sparse attention,
merges and output projection. No standalone GQA/FFN/residual in this workload.

Production baseline: B1/S1, context1048576, position1048571, hidden6144,
main64Q/4KV heads of128, index4Q/1K heads of128;16 selected blocks of128 tokens
including current. IndexDP1/CP16, mainDP1/CP4,8 processing chunks,4 core groups.
The runnable tests below use smaller dimensions, not production coverage.

## Current accuracy strategy and truth

1. Compare our partitioned **FP32 CPU** implementation with the local
   unpartitioned **FP32 Reference**, using the same actual weights, inputs,
   positions/RoPE and nonzero logical history, correctly repacked for DP/CP.
2. Compare **FP16 CPU against that FP32 reference** to measure precision
   sensitivity separately from implementation agreement.
3. Vary supported DP/CP/chunks with matched logical data. Check outputs,
   selections, causality/current-block insertion, cache updates and exact
   untouched-cache preservation; investigate the first divergent stage.

Prior production-size CPU evidence: FP32 matched within4.66e-9; FP16 differed
by up to4.19e-4 and changed one historical TopK block near a small score margin.
Those are historical observations; their large fixtures are not included here.
They establish local consistency/precision sensitivity, not external MiniMax
validation or a hardware root cause.

The current Reference is local PyTorch code, **not an official HF checkpoint**.
The existing `hf_parity` helper is a separate tiny synthetic-weight
1GQA+1MSA comparison, not production unified-path/checkpoint validation.
Establishing a matching authoritative HF implementation is new work.

Existing `atol=1e-4, rtol=1e-3` is FP32 CPU-test provenance, not a justified
FP16/MXFP6 policy. Report error distributions/scales, normalized error,
finiteness and TopK margins; do not select limits just to pass existing failures.
CPU FP16 includes explicit FP32 computation. MXFP6/MXINT8 compiler options do
not implement their target arithmetic on CPU; no validated full-layer MX
emulator or numerical acceptance policy is established.

**Both CP values must exceed1 in this unified path.** A DP1/CP1 HF baseline
cannot be substituted into it without explicitly validating a different path.

## Necessary source

| File | Purpose |
|---|---|
| [forwards.py](./source/layer_benchmarks/attention/msa/forwards.py) | Target computation, profiles and configuration |
| [bench.py](./source/layer_benchmarks/attention/msa/bench.py) | Model/input construction; its ordinary inputs have zero caches |
| [parity.py](./source/layer_benchmarks/attention/msa/parity.py) | Local independent cache/attention reference and limited HF helper |
| [cache.py](./source/layer_benchmarks/attention/msa/cache.py), [ops.py](./source/layer_benchmarks/attention/msa/ops.py) | Cache operations and eager/custom-op definitions |
| [validation.py](./source/layer_benchmarks/attention/msa/configurator/validation.py), [test_numerical.py](./source/layer_benchmarks/attention/msa/configurator/tests/test_numerical.py) | Nonzero-history CPU layout/boundary/multi-step checks |
| [common/](./source/layer_benchmarks/common) | Required imports; QAIC/export helpers stay only because benchmark code imports them |

This is a minimal source subset, not the complete configurator/campaign tool.
Copied-file hashes are recorded in
[COPY_MANIFEST.json](./COPY_MANIFEST.json).

## Run only the CPU code checks

From the bundle root, using your chosen environment's `python3`:

```bash
cd source
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
python3 -m unittest \
  layer_benchmarks.attention.msa.configurator.tests.test_numerical.NumericalTests.test_repacking \
  layer_benchmarks.attention.msa.configurator.tests.test_numerical.NumericalTests.test_cp_chunks_and_boundaries \
  layer_benchmarks.attention.msa.configurator.tests.test_numerical.NumericalTests.test_batch_dp \
  -v
```

These are small **FP32** CPU/reference checks with nonzero history, selected
block/cache invariants and DP/CP/chunk variations; they do not establish FP16
accuracy. Do not run the whole test module: its separate decomposed-export test
generates ONNX. Do not invoke benchmark export/compile/hardware modes.
No accelerator SDK is required for these CPU checks. Do not copy a venv or
assume the original server's Python location; select a suitable environment
on the receiving server.

Future goal: compare accuracy on the compiled accelerator too, after trusted
input matching and justified precision-aware criteria. **No accelerator run
or performance optimization is part of this CPU handoff.**
