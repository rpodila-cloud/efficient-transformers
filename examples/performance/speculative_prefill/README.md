Speculative Prefill with Continuous Batching (CB) — Quick Guide

Overview
This example demonstrates running Speculative Prefill on Cloud AI 100 with:
- A small speculator model producing prefill_queries and retained-state keys for host-side scoring and pruning.
- A base/target model compiled for continuous batching (CB) to run pruned prefill and decode efficiently.
- Centralized timing metrics collected via SpecPrefillEngine.prune_and_base_prefill().

Prerequisites
- Cloud AI 100 environment set up with the Platform SDK.
- Access to Hugging Face models used in the example.
- Python dependencies consistent with this repository.

Models and Compilations
1) Speculator (no CB required)
- Load with enable_speculative_prefill:
  spec_model = QEFFAutoModelForCausalLM.from_pretrained(
      "meta-llama/Llama-3.2-1B-Instruct",
      qaic_config={"enable_speculative_prefill": True}
  )
- Export/compile (example parameters):
  spec_model.compile(
      prefill_seq_len=128,
      ctx_len=4096,
      num_devices=4,
      num_cores=16,
      mxfp6_matmul=True,
      mxint8_kv_cache=True,
      allow_mxint8_mdp_io=True,
      split_retained_state_io=True,
      aic_enable_depth_first=True,
  )

2) Base/Target (CB required)
- Load with continuous_batching=True:
  base_model = QEFFAutoModelForCausalLM.from_pretrained(
      "meta-llama/Llama-3.2-3B-Instruct",
      continuous_batching=True
  )
- Export/compile with full_batch_size > 1:
  base_model.compile(
      prefill_seq_len=128,
      ctx_len=4096,
      full_batch_size=2,        # CB is engaged when FBS > 1
      kv_cache_batch_size=2,    # optional; defaults to FBS if omitted
      num_devices=4,
      num_cores=16,
      mxfp6_matmul=True,
      mxint8_kv_cache=True,
      allow_mxint8_mdp_io=True,
      aic_enable_depth_first=True,
  )

Verify CB is compiled
After base compile, confirm that the compiled QPC has full_batch_size > 1:
from QEfficient.generation.text_generation_inference import get_compilation_dims

_, _, fbs = get_compilation_dims(str(base_model.qpc_path))
print("Base full_batch_size (FBS):", fbs)  # Expect > 1 (e.g., 2)

Runtime execution
Use generate_speculative_prefill() to run the end-to-end flow:
- Speculator prefill, host-side scoring and selection
- Base prefill on pruned tokens
- Decode: continuous batching path is taken automatically when FBS > 1

Example (see examples/performance/speculative_prefill/speculative_prefill_app.py):
result = spec_model.generate_speculative_prefill(
    base_model=base_model,
    tokenizer=tokenizer,
    prompts=prompt_text,
    device_id=[8,9,10,11],        # example device IDs for speculator
    base_device_id=[12,13,14,15], # example device IDs for base
    keep_percentage=0.20,
)

- The engine prints a compact TTFT summary:
  [4.3] S=... kept=... TTFT(base_full)=...ms TTFT(spec_device)=...ms TTFT(host_scoring)=...ms TTFT(base_pruned_only)=...ms TTFT(speculative)=...ms
- The return dictionary includes metrics:
  - ttft_baseline_s
  - ttft_spec_device_s
  - ttft_host_scoring_s
  - ttft_base_pruned_only_s
  - ttft_speculative_s
  and selection stats:
  - S (original tokens), kept (kept tokens), keep_idx, importance

Verify CB is engaged at runtime
- A log is printed by the engine when CB decode path is taken:
  [spec] Using continuous batching decode with full_batch_size=<FBS>
- TTFT metrics are computed before decode and are valid regardless of CB, which is expected and correct.

Key notes
- Speculator QPC does not require CB; only the base/target model needs CB compilation.
- Ensure base compile includes a decode specialization (compile() default builds both prefill and decode when prefill_only is None).
- Memory footprint grows with full_batch_size and KV cache dtype; adjust FBS accordingly.
- Device IDs must match your hardware setup; the example shows separate device groups for speculator and base.

Reference script
See examples/performance/speculative_prefill/speculative_prefill_app.py for a runnable example that:
- Exports and compiles both models
- Compiles the base with CB (FBS=2)
- Verifies FBS from the base QPC
- Runs generate_speculative_prefill() and prints generated output

Further reading
- Speculative prefill algorithm and architecture details:
  - speculative_prefill/TODO_technical_summary.md
