Here’s a concise technical summary and TODO list capturing the current state of the speculative prefill integration work.

Speculative prefill (SPECPREFILL) and your integration
- High-level algorithm (look-ahead anchors supported; default look-ahead = 0):
  - Two-model pipeline:
    - Speculator (small model) runs full-prompt prefill to produce queries (Q) and per-layer keys (K).
      - Q capture path (flag-gated): `QEffLlamaAttention.forward` emits last-token per-head queries (`q_last`), collected in `QEffLlamaDecoderLayer` and stacked in `QEffLlamaModel` as `prefill_queries`, surfaced via `QEffLlamaForCausalLM`.
      - K capture path: retained-state `past_key.*_RetainedState` outputs enabled only on the final prefill chunk; harvested once from that chunk.
    - Importance computation:
      - Host NumPy computes logits = Q·Kᵀ/√D per head (handles head_dim padding), softmax over positions.
      - Head-wise max (or sum), then layer-wise max; look-ahead anchors averaged if enabled.
      - 1D smoothing (moving average).
      - Block top‑K by mean score; force-keep last token; selection is percentage-based.
    - Restore absolute position_ids for the kept tokens.
    - Base model runs prefill only on selected tokens; decode continues from original context length S (full position indexing preserved).
  - Host-side scoring:
    - Read Q and retained K (final chunk) to host; importance and selection computed on CPU to avoid modifying ONNX subgraphs; retained-state K may be head-dim padded and is sliced to S before scoring.
    - Note: reading K to host can be DMA-heavy for very long contexts but simplifies deployment.

- Integration into QEfficient (flag-gated, minimal disturbance to baseline):
  - SpeculativePrefillTransform:
    - Registered after KVCacheTransform in _pytorch_transforms.
    - Enables config.enable_speculative_prefill when qaic_config["enable_speculative_prefill"] is passed.
  - Llama modeling changes (flag-gated):
    - Attention captures the last-step queries when the flag is on.
    - Decoder aggregates per-layer queries into a prefill_queries tensor in outputs.
    - New QEff output dataclasses carry optional prefill_queries only when the flag is on; baseline outputs unchanged when off.
  - ONNX export:
    - Appends prefill_queries to output names and dynamic axes only when enable_speculative_prefill is True.
    - Retained-state KV outputs remain unchanged.
  - Runtime extension:
    - text_generation_inference.py adds prefill_from_ids:
      - Performs a prefill given explicit token ids and position_ids.
      - Supports optional batch_index to fill a specific CB slot.
      - Mirrors run_prefill chunking, position_ids semantics, and retained-state behavior.
  - New API path:
    - QEFFAutoModelForCausalLM.generate_speculative_prefill:
      - Requires qaic_config["enable_speculative_prefill"]=True and both models compiled.
      - Runs speculator prefill, performs host-side importance scoring, selects tokens, restores positions.
      - Base prefill:
        - Non-CB: serial prefill on pruned tokens, then decode.
        - CB: builds a pruned prompt queue; pads/duplicates as needed to fill full_batch_size; serially prefills each slot with batch_index; seeds decode inputs; then runs standard run_continuous_batching_decode on a queue (same CB execution semantics as baseline).
      - Guards against missing decode specialization.
      - Eliminates manual base execution by calling SpecPrefillEngine.prune_and_base_prefill(), centralizing timing and pruning logic and restoring TTFT metrics.
      - Returns:
        - generated_text_pruned
        - keep_idx
        - importance
        - ttft_baseline_s
        - ttft_spec_device_s
        - ttft_host_scoring_s
        - ttft_base_pruned_only_s
        - ttft_speculative_s

QEfficient/transformers/models/pytorch_transforms.py
QEfficient/transformers/models/modeling_auto.py
QEfficient/transformers/models/llama/modeling_llama.py
QEfficient/transformers/models/modeling_outputs_qeff.py
QEfficient/generation/speculative_prefill_engine.py
QEfficient/generation/text_generation_inference.py
QEfficient/__init__.py
speculative_prefill/speculative_prefill_integration_plan.md (updated plan)
Constraints:

Speculator: batch_size=1, no CB.
Base: ctx_len/prefill_seq_len must align with spec; CB supported when compiled with full_batch_size/batch_index; prompts queued/padded as per default CB logic.

- Constraints and compatibility:
  - Speculator runs with batch_size=1; no CB for speculator.
  - Base model must align ctx_len and prefill_seq_len with speculator’s assumptions so position restoration remains consistent.
  - CB is supported for the base when compiled appropriately (full_batch_size, batch_index wiring).
  - The integration is opt-in; baseline generate code path untouched when flag off.
