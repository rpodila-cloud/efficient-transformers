# CPU microbenchmark accuracy handoff

## Intent and structure

One portable **code-only** collection, preserving the original
`layer_benchmarks/` package layout with shared helpers once.
All paths below are relative to this bundle, wherever it is copied.
No venv, HF overlay/checkpoint, ONNX model, saved weights/tensors/caches,
compiler output or hardware campaign is included.

Use your server's Python3.12+ environment.
[requirements.txt](./layer_benchmarks/requirements.txt) is the upstream dependency
list; [ENVIRONMENT.json](./ENVIRONMENT.json) records the tested environment,
not a bundled interpreter. Original source/run directories were not modified.

| Benchmark | Source directory | CLI module |
|---|---|---|
| MSA | [msa/](./layer_benchmarks/attention/msa) | `layer_benchmarks.attention.msa.bench` |
| GQA | [gqa/](./layer_benchmarks/attention/gqa) | `layer_benchmarks.attention.gqa.bench` |
| MLA | [mla/](./layer_benchmarks/attention/mla) | `layer_benchmarks.attention.mla.bench` |
| DSA | [dsa/](./layer_benchmarks/attention/dsa) | `layer_benchmarks.attention.dsa.bench` |
| HCA | [hca/](./layer_benchmarks/attention/hca) | `layer_benchmarks.attention.hca.bench` |
| QSA | [qsa/](./layer_benchmarks/attention/qsa) | `layer_benchmarks.attention.qsa.bench` |
| CSA | [csa/](./layer_benchmarks/attention/csa) | `layer_benchmarks.attention.csa.bench` |
| Linear attention | [linear/](./layer_benchmarks/attention/linear) | `layer_benchmarks.attention.linear.bench` |
| MoE | [moe/](./layer_benchmarks/moe) | `layer_benchmarks.moe.bench` |
| DMA | [dma/](./layer_benchmarks/dma) | `layer_benchmarks.dma.bench` |

Within each family, start with `bench.py` for construction/inputs and
`forwards.py` for computation; DMA defines computation in `bench.py`.
`parity.py` and any `ref.py` contain reference/parity helpers. MoE's local
`Reference` is in `forwards.py`. [common/](./layer_benchmarks/common) contains
necessary imports; dormant QAIC/export code is retained to avoid altering
benchmark code, not to authorize accelerator use.

## Accuracy strategy and current truth

Match actual weights, activations, masks, positions/RoPE and nonzero logical
history, correctly repacked for the requested execution layout. Equal seeds
alone are insufficient. Separate:

1. Implementation agreement at matching precision.
2. Precision-induced differences against a stated higher-precision reference.
3. DP/CP/blocking equivalence with the same logical inputs.

Check output, selection/causal invariants and all updated state; untouched
cache data must be preserved. Investigate the first differing stage.
Fixed-selection comparisons can isolate attention errors but do not replace
the end-to-end own-selection comparison.

### MSA evidence and limitations

The studied target is `msa_indexer_attention_GP_unified`, one non-paged MSA
decode layer including indexer, projections, normalization/RoPE, block selection,
cache updates, sparse attention/merges and output projection; no standalone
GQA/FFN/residual. Baseline B1/S1, context1048576, position1048571, hidden6144,
main64Q/4KV heads of128, index4Q/1K heads of128,16 blocks of128 including current;
indexDP1/CP16, mainDP1/CP4,8 chunks,4 core groups.

Prior CPU evidence: FP32 matched the local FP32
[Reference](./layer_benchmarks/attention/msa/parity.py) within4.66e-9.
FP16 CPU versus that FP32 reference differed by up to4.19e-4 and changed one
historical TopK block near a small margin. These are historical observations;
the large fixture is not included. They are not external MiniMax validation
or a hardware root-cause finding.

The local Reference is not official HF checkpoint validation. MSA's separate
`hf_parity` helper uses tiny synthetic copied weights and1GQA+1MSA; establishing
a matching official implementation/checkpoint is new work. The unified path
requires **both CP values >1**; do not silently substitute DP1/CP1.

`atol=1e-4, rtol=1e-3` has FP32 CPU-test provenance, not a justified
FP16/MXFP6 policy. CPU FP16 includes explicit FP32 math. MX compiler options
do not emulate target arithmetic on CPU; no validated full-layer MX reference
or acceptance policy has been established. Do not select limits to pass failures.

For the other families, copied parity helpers and import readiness are **not
new numerical accuracy results**. HF helpers, where present, require compatible
model code and do not by themselves establish official-checkpoint parity.
Do not transfer MSA's results, layout restrictions or tolerances to other families.
DMA CPU additions are not accelerator DMA bandwidth measurements.

## CPU-only commands

From the bundle root, using your chosen environment's `python3`:

```bash
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4

# Safe CLI-readiness example; substitute another CLI module from the table.
python3 -m layer_benchmarks.attention.gqa.bench --help

# Small nonzero-history MSA FP32 CPU/reference/layout tests.
python3 -m unittest \
  layer_benchmarks.attention.msa.configurator.tests.test_numerical.NumericalTests.test_repacking \
  layer_benchmarks.attention.msa.configurator.tests.test_numerical.NumericalTests.test_cp_chunks_and_boundaries \
  layer_benchmarks.attention.msa.configurator.tests.test_numerical.NumericalTests.test_batch_dp \
  -v
```

Default benchmark execution and many full self-tests create/require ONNX.
Do not run them for this code-only task; review and separate their CPU portions.
The three MSA tests above do not establish FP16 accuracy or full production
coverage. No exports, downloads, compilation, accelerator sessions or timing
are requested. The MSA configurator directory is only a CPU-helper subset,
not the complete configuration-search/campaign tool.

Future goal: compare accuracy on the compiled accelerator too, after trusted
input matching and justified precision-aware criteria, with separate approval.
Source hashes and limited packaging/readiness/test scope are recorded in
[COPY_MANIFEST.json](./COPY_MANIFEST.json) and [VALIDATION.json](./VALIDATION.json).
