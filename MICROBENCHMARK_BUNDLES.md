# Portable code-only microbenchmark bundles

Each directory is independent: copy it to any server and start with its handoff.
Paths inside bundles are relative; no venv, checkpoint, ONNX model, saved tensors
or accelerator artifacts are included. Shared import dependencies are repeated
only so each bundle works without its siblings.

| Benchmark | Handoff |
|---|---|
| MSA | [MSA_CPU_ACCURACY_HANDOFF.md](./msa_cpu_accuracy_bundle/MSA_CPU_ACCURACY_HANDOFF.md) |
| GQA | [CPU_HANDOFF.md](./gqa_cpu_accuracy_bundle/CPU_HANDOFF.md) |
| MLA | [CPU_HANDOFF.md](./mla_cpu_accuracy_bundle/CPU_HANDOFF.md) |
| DSA | [CPU_HANDOFF.md](./dsa_cpu_accuracy_bundle/CPU_HANDOFF.md) |
| HCA | [CPU_HANDOFF.md](./hca_cpu_accuracy_bundle/CPU_HANDOFF.md) |
| QSA | [CPU_HANDOFF.md](./qsa_cpu_accuracy_bundle/CPU_HANDOFF.md) |
| CSA | [CPU_HANDOFF.md](./csa_cpu_accuracy_bundle/CPU_HANDOFF.md) |
| Linear attention | [CPU_HANDOFF.md](./linear_cpu_accuracy_bundle/CPU_HANDOFF.md) |
| MoE | [CPU_HANDOFF.md](./moe_cpu_accuracy_bundle/CPU_HANDOFF.md) |
| DMA | [CPU_HANDOFF.md](./dma_cpu_accuracy_bundle/CPU_HANDOFF.md) |

New bundles are source/import/CLI-readiness snapshots, not new numerical accuracy
certifications. Their existing self-tests often generate ONNX; handoffs explicitly
separate safe startup from numerical work. Do not transfer MSA-specific results
or error limits to other benchmarks. No accelerator execution or model downloads
are requested here.

All nine new bundles passed source-hash, renamed-directory import and CLI-help
checks with the accelerator SDK unavailable and network/export/session calls
blocked. Per-bundle `CPU_VALIDATION.json` records that limited readiness scope.
This is not a new numerical-accuracy result. The existing MSA bundle was not changed.
