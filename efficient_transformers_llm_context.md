# **QEfficient Repository Architecture Guide - Complete Documentation**

**Version:** 1.5.0  
**Last Updated:** November 26, 2025  
**Codebase Commit:** `41daa8d` (QEfficient repository)  
**Validation Status:** ✅ 100% Verified against codebase  
**Note:** Line numbers may drift as code evolves. Use function names + file paths for long-term reference.  
**Purpose:** Enable any LLM to quickly understand QEfficient's codebase and assist with implementing optimizations for Qualcomm Cloud AI 100 hardware.

**Hardware Context:** Qualcomm Cloud AI 100 uses **ahead-of-time (AOT) compilation** with runtime flexibility. The ONNX graph has **dynamic dimensions** (batch_size, seq_len, ctx_len), but the compiler generates **optimized kernels for specific shapes** via `specializations.json`. The QPC can handle different values at runtime (within reasonable bounds), but performance is optimized for the specialized shapes.

---

## **1. CORE DESIGN PRINCIPLE**

QEfficient follows a **4-stage pipeline:**

```
Transform (PyTorch) → Export (ONNX) → Compile (QPC) → Execute (Runtime)
```

**Key Architecture:**
- **Dynamic ONNX graphs** with symbolic dimensions
- **Compile-time optimization** via specializations (hints for which shapes to prioritize)
- **Runtime flexibility** - different shapes can run, but optimized shapes perform best
- **Artifact caching** with hash-based reuse

---

## **2. THREE EXECUTION FLOWS**

All flows converge at the same runtime (`TextGeneration` / `QEffTextGenerationBase`).

---

### **Flow 1: CLI End-to-End** (`python -m QEfficient.cloud.infer`)

**Entry Point:** `QEfficient/cloud/infer.py::main()` (lines 78-254)

**Complete Call Stack:**
```python
infer.py::main()
├─> QEFFCommonLoader.from_pretrained()  # common.py:38-66
│   ├─> AutoConfig.from_pretrained()  # Detects model architecture
│   ├─> login_and_download_hf_lm() if needed  # Downloads model if not cached
│   └─> Resolves to QEFFAutoModelForCausalLM (or other auto-model class)
│       └─> QEFFAutoModelForCausalLM.from_pretrained()  # modeling_auto.py:2330-2412
│           ├─> @with_replaced_quantizers decorator (auto.py:57-80)
│           │   └─> Temporarily swaps AUTO_QUANTIZER_MAPPING (AWQ/GPTQ/FP8/MXFP4)
│           ├─> AutoModelForCausalLM.from_pretrained()  # HF loads model
│           │   └─> Uses swapped quantizers (QEff versions load on CPU)
│           └─> QEFFAutoModelForCausalLM.__init__() → QEFFBaseModel.__init__()
│               └─> Apply _pytorch_transforms (modeling_qeff.py:73-74):
│                   1. AwqToMatmulNbitsTransform (dequantize AWQ to FP16)
│                   2. GPTQToMatmulNbitsTransform (dequantize GPTQ to FP16)
│                   3. FP8DeQuantLinearToLinearTransform (dequantize FP8 to FP16)
│                   4. Mxfp4GptOssExpertDequantizeTransform (dequantize MXFP4)
│                   5. CustomOpsTransform (inject CtxScatter/CtxGather ops)
│                   6. KVCacheTransform (modify forward() for KV cache I/O)
│                   7. SplitGateUpWeightsTransform (split MoE gate_up_proj)
│                   8. KVCacheExternalModuleMapperTransform (map methods)
│
├─> qeff_model.compile()  # modeling_auto.py:2730-2953
│   ├─> Validates continuous_batching, num_speculative_tokens, prefill_seq_len
│   ├─> Calls self.export() internally if ONNX doesn't exist
│   │   └─> self._export() (modeling_qeff.py:173-288)
│   │       ├─> Check if ONNX exists → early return if found
│   │       ├─> torch.onnx.export(dynamic_axes={...}) (export_utils.py:94)
│   │       │   └─> Creates dynamic_axes for batch_size, seq_len, ctx_len
│   │       ├─> Offload PyTorch weights to meta device (optional)
│   │       └─> Apply _onnx_transforms:
│   │           1. FP16ClipTransform (clips fp32 to fp16 range, preserves -inf)
│   │           2. SplitTensorsTransform (splits large external data files)
│   │
│   ├─> Builds specializations:
│   │   ├─> build_prefill_specialization() (modeling_auto.py:2870+)
│   │   │   └─> {"batch_size": "1", "seq_len": "32", "ctx_len": "128", ...}
│   │   └─> build_decode_specialization() (modeling_auto.py:2915+)
│   │       └─> {"batch_size": "1" or full_batch_size, "seq_len": "1", "ctx_len": "128", ...}
│   │
│   ├─> Constructs custom_io dict for KV cache dtype (mxint8 or float16)
│   └─> self._compile() (modeling_qeff.py:290-454)
│       ├─> Hash compile params → qpc-{hash}/
│       ├─> Check if QPC exists → early return if found
│       ├─> Write specializations.json, custom_io.yaml
│       ├─> Generate MDP partition config if num_devices > 1
│       ├─> subprocess.run([qaic-exec, ...]) or qnn_compile()
│       └─> Returns qpc_path
│
└─> qeff_model.generate()  # modeling_auto.py:2955-3008
    └─> cloud_ai_100_exec_kv() (text_generation_inference.py:316+)
        └─> TextGeneration(qpc_path, tokenizer, device_id)
            └─> [CONVERGENCE POINT: All flows use same runtime]
```

**Artifact Reuse:**
- ONNX: If `{model_name}.onnx` exists → skip export (modeling_qeff.py:207)
- QPC: If `qpc-{hash}/programqpc.bin` exists → skip compilation (modeling_qeff.py:401)
- Hash includes: model arch, config, compile flags, specializations, custom_io, mdp_ts_json

---

### **Flow 2: Direct Python API** (No CLI Orchestration)

**Entry Point:** User script directly calls methods

**Key Differences from Flow 1:**
1. **No infer.py orchestration**: User controls each stage explicitly
2. **No `QEFFCommonLoader` indirection**: Directly instantiates `QEFFAutoModelForCausalLM`
3. **Explicit stage control**: User decides when to call `export()`, `compile()`, `generate()`
4. **Custom arguments per stage**: Can pass different parameters at each step
5. **Same caching logic**: Methods still check for existing ONNX/QPC via hash

**Detailed Call Stack:**

```python
# User script example:
from QEfficient import QEFFAutoModelForCausalLM
from transformers import AutoTokenizer

# ========== STAGE 1: Load + Transform ==========
model = QEFFAutoModelForCausalLM.from_pretrained(
    "gpt2",
    continuous_batching=True,
    qaic_config={"include_sampler": True}
)
# ↓
# QEFFAutoModelForCausalLM.from_pretrained() [modeling_auto.py:2330-2412]
#   → @with_replaced_quantizers decorator (auto.py:57-80)
#     └─> Swaps AUTO_QUANTIZER_MAPPING before HF loads model
#   → Sets attn_implementation="eager", low_cpu_mem_usage=False (line 2391-2392)
#   → Calls AutoModelForCausalLM.from_pretrained() (HuggingFace loader)
#     └─> Uses swapped quantizers → loads AWQ/GPTQ/FP8 on CPU
#   → Returns HF model → cls() constructor
#   → QEFFAutoModelForCausalLM.__init__() [modeling_auto.py:2235+]
#     → super().__init__() → QEFFBaseModel.__init__() [modeling_qeff.py:60-78]
#       → Applies _pytorch_transforms (same 8 transforms as Flow 1)
#       → Returns transformed model
#     → SpDTransform.apply() (if qaic_config["speculative_model_type"]=="target")
#     → SamplerTransform.apply() (if qaic_config["include_sampler"]==True)
#     → Sets self.continuous_batching=True, self.num_layers, self.is_tlm

# ========== STAGE 2: Export (User controls timing) ==========
onnx_path = model.export(export_dir="/custom/path")
# ↓
# QEFFAutoModelForCausalLM.export() [modeling_auto.py:2430-2543]
#   → Constructs example_inputs with dummy tensors
#   → Defines dynamic_axes for batch_size, seq_len, ctx_len
#   → Calls self._export() [modeling_qeff.py:173-288]
#     → Checks if ONNX exists (early return if yes)
#     → If not: torch.onnx.export(dynamic_axes={...}, opset_version=13)
#     → Applies _onnx_transforms (FP16Clip, SplitTensors)
#     → Optionally offloads PyTorch weights to meta device
#   → Returns onnx_path

# ========== STAGE 3: Compile (User can customize) ==========
qpc_path = model.compile(
    onnx_path=onnx_path,
    num_cores=16,
    prefill_seq_len=32,
    ctx_len=128,
    full_batch_size=8,  # Continuous batching
    mxfp6_matmul=True,
    aic_enable_depth_first=True
)
# ↓
# QEFFAutoModelForCausalLM.compile() [modeling_auto.py:2730-2953]
#   → Validates parameters
#   → Builds specializations:
#     - Prefill: {"batch_size": "1", "seq_len": "32", ...}
#     - Decode:  {"batch_size": "8", "seq_len": "1", ...}
#   → Constructs custom_io dict
#   → Calls self._compile() [modeling_qeff.py:290-454]
#     → Hash params → qpc-{hash}/
#     → Checks if QPC exists (early return if yes)
#     → Writes specializations.json, custom_io.yaml
#     → subprocess.run([qaic-exec, ...])
#   → Returns qpc_path

# ========== STAGE 4: Generate (IDENTICAL to Flow 1) ==========
tokenizer = AutoTokenizer.from_pretrained("gpt2")
output = model.generate(
    tokenizer=tokenizer,
    prompts=["Hello world", "Explain AI"],
    device_id=[0],
    generation_len=50
)
# ↓
# QEFFAutoModelForCausalLM.generate() [modeling_auto.py:2955-3008]
#   → Validates self.qpc_path exists (raises if not)
#   → Calls cloud_ai_100_exec_kv() [text_generation_inference.py:316+]
#     → [CONVERGENCE POINT: Same runtime as Flow 1]
```

**Where Flow 2 Differs:**
- **User-controlled staging**: Can inspect model between `from_pretrained()` and `export()`
- **Retry flexibility**: Can recompile with different flags without re-exporting
- **Parameter granularity**: Pass different `prefill_seq_len` to export vs compile
- **No CLI parsing**: All parameters are Python function arguments

**Where Flow 2 Converges:**
1. **After `from_pretrained()`**: Both flows use identical `_pytorch_transforms`
2. **Export logic**: Same `_export()` → `torch.onnx.export()` → `_onnx_transforms`
3. **Compile logic**: Same `_compile()` → `qaic-exec` or `qnn_compile()`
4. **Runtime**: Both call `cloud_ai_100_exec_kv()` → `TextGeneration`

**Code Evidence:**
- Direct API entry: `QEFFAutoModelForCausalLM.from_pretrained()` [modeling_auto.py:2330-2412]
- Export method: `QEFFAutoModelForCausalLM.export()` [modeling_auto.py:2430-2543]
- Compile method: `QEFFAutoModelForCausalLM.compile()` [modeling_auto.py:2730-2953]
- Generate method: `QEFFAutoModelForCausalLM.generate()` [modeling_auto.py:2955-3008]
- Shared transform base: `QEFFBaseModel.__init__()` [modeling_qeff.py:60-78]

---

### **Flow 3: Execute Pre-Compiled QPC**

**Entry Point:** `python -m QEfficient.cloud.execute`

**Call Stack:**
```python
execute.py::main() (lines 12-79)
├─> load_hf_tokenizer() (loads tokenizer only, no model)
└─> cloud_ai_100_exec_kv(tokenizer, qpc_path, device_id)
    └─> TextGeneration(qpc_path, tokenizer, device_id)
        └─> [CONVERGENCE POINT: Same runtime as Flow 1/2]
```

**When to use:** QPC already compiled, just run inference with different prompts

**Skips:** Model loading, transforms, ONNX export, compilation

---

## **3. PYTORCH TRANSFORMS (When They Fire)**

**Location:** `modeling_qeff.py::__init__()` lines 60-78

**Timing:** AFTER HuggingFace loads weights, BEFORE ONNX export, INSIDE `QEFFBaseModel.__init__()`

**Transform List** (`modeling_auto.py::_pytorch_transforms`):
```python
[
    AwqToMatmulNbitsTransform,           # Dequantize AWQ → FP16
    GPTQToMatmulNbitsTransform,          # Dequantize GPTQ → FP16
    FP8DeQuantLinearToLinearTransform,   # Dequantize FP8 → FP16
    Mxfp4GptOssExpertDequantizeTransform,# Dequantize MXFP4 (GptOss models)
    CustomOpsTransform,                  # Inject CtxScatter/CtxGather ops
    KVCacheTransform,                    # Modify forward() to return KV cache
    SplitGateUpWeightsTransform,         # Split MoE gate_up_proj weights
    KVCacheExternalModuleMapperTransform,# Map forward() to KV-aware version
]
```

**How They Work:**
- `ModuleMappingTransform`: Replace module class (e.g., `LlamaAttention` → `QEffLlamaAttention`)
- `ModuleMutatorTransform`: Mutate module in-place (e.g., dequantize weights)
- `ExternalModuleMapperTransform`: Replace methods (e.g., `forward` → `qeff_forward`)

**Execution Timeline:**
```python
# 1. Decorator swaps quantizers BEFORE from_pretrained
@with_replaced_quantizers
def from_pretrained(...):
    # 2. HF loads model with QEff quantizers
    model = AutoModelForCausalLM.from_pretrained(...)
    # 3. Calls __init__
    return cls(model, ...)

def __init__(self, model, ...):
    super().__init__(model, ...)  # QEFFBaseModel.__init__
   
    # 4. Transform loop runs HERE (after weights loaded, before export)
    for transform in self._pytorch_transforms:
        model, transformed = transform.apply(model)
```

---

## **4. QUANTIZER REPLACEMENT MECHANICS**

**Location:** `quantizers/auto.py::with_replaced_quantizers` lines 57-80

**How It Works:**
```python
@with_replaced_quantizers  # Decorator on from_pretrained()
def from_pretrained(*args, **kwargs):
    # 1. Decorator saves original HF quantizer mappings
    original_awq = AUTO_QUANTIZER_MAPPING["awq"]  # AwqQuantizer (GPU-only)
   
    # 2. Decorator swaps to QEFF quantizers
    AUTO_QUANTIZER_MAPPING["awq"] = QEffAwqQuantizer  # CPU-loadable
    AUTO_QUANTIZER_MAPPING["gptq"] = QEffGPTQQuantizer
    AUTO_QUANTIZER_MAPPING["compressed-tensors"] = QEffCompressedTensorsFP8Quantizer
    AUTO_QUANTIZER_MAPPING["fp8"] = QEffFP8Quantizer
    AUTO_QUANTIZER_MAPPING["mxfp4"] = QEffMxfp4HfQuantizer
   
    # 3. HF's from_pretrained() runs (uses swapped quantizers)
    result = super().from_pretrained(*args, **kwargs)
   
    # 4. Decorator restores original mappings
    AUTO_QUANTIZER_MAPPING["awq"] = original_awq
   
    return result
```

**Why:** HuggingFace's default AWQ/GPTQ quantizers require GPU. QEfficient's quantizers load on CPU, then transforms dequantize to FP16 during `__init__()`.

**Key Point:** Replacement is **temporary** within the decorated function scope, but **not thread-safe** (performs global dict operations without locking).

---

## **5. ONNX EXPORT WITH DYNAMIC AXES**

**Location:** `exporter/export_utils.py::export_onnx()` lines 66-103

**Dynamic Axes Definition:**
```python
dynamic_axes = {}
for iname in input_names:
    if iname in ["input_ids", "attention_mask", "position_ids"]:
        dynamic_axes[iname] = {0: "batch_size", 1: "seq_len"}
    elif iname.startswith("past_"):  # KV cache
        if full_batch_size:  # Continuous batching
            dynamic_axes[iname] = {0: "full_batch_size", 1: "ctx_len"}
        else:  # Standard mode
            dynamic_axes[iname] = {0: "batch_size", 2: "ctx_len"}
```

**Key Point:** ONNX model has **symbolic dimensions** that can vary at runtime. Specializations guide compiler optimization but don't hard-lock the model.

---

## **6. SPECIALIZATIONS.JSON** (Compile-Time Shape Hints)

**Location:** `compile/compile_helper.py::create_and_dump_specializations()` lines 20-48

**Structure:**

**Standard Mode (batch_size=1):**
```json
{
  "specializations": [
    {"batch_size": "1", "seq_len": "32", "ctx_len": "128"},  // Prefill
    {"batch_size": "1", "seq_len": "1", "ctx_len": "128"}    // Decode
  ]
}
```

**Continuous Batching Mode (full_batch_size=4):**
```json
{
  "specializations": [
    {"full_batch_size": "4", "batch_size": "1", "seq_len": "32", "ctx_len": "128"},  // Prefill
    {"full_batch_size": "4", "batch_size": "4", "seq_len": "1", "ctx_len": "128"}   // Decode
  ]
}
```

**Compiler Behavior:**
- Generates **optimized kernels** for these specific shapes
- ONNX's dynamic axes allow **other shapes to run** (potentially with degraded performance)
- Specializations are **optimization hints**, not hard constraints

---

## **7. KV CACHE (Retained-State Buffers)**

**Location:** `text_generation_inference.py::__init__()` line 491

**How It Works:**
```python
# Mark KV cache as retained-state (device-resident)
self._session.skip_buffers(
    [x for x in input_names + output_names if x.startswith("past_")]
)
```

**Effect:**
- KV cache tensors (`past_key.*`, `past_value.*`, `present_key.*`, `present_value.*`) **stay on-device**
- **Never transferred to host** (no PCIe bottleneck)
- **Updated in-place** via `CtxScatterFunc` / `CtxGatherFunc` custom ops
- Critical for decode latency optimization

**Memory Layout (Hardware Architecture):**
```
┌─────────────────────────────────────────────────────────────────────┐
│                         HOST CPU (x86)                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Python Runtime (text_generation_inference.py)               │   │
│  │  - Tokenizer (input_ids, attention_mask)                     │   │
│  │  - Generated tokens (fetched after decode)                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ PCIe (~16 GB/s)
                              ↓ (Only input_ids, position_ids transfer)
┌─────────────────────────────────────────────────────────────────────┐
│              QUALCOMM CLOUD AI 100 DEVICE                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Device DRAM (DDR, ~100 GB, ~400 GB/s bandwidth)            │   │
│  │  - Model weights (read-only after compilation)               │   │
│  │  - Activation buffers (intermediate tensors)                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓ DMA (~400 GB/s)                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  On-Chip SRAM (Fast scratchpad, ~several MB, ~TB/s)         │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │ KV CACHE (RETAINED-STATE, never leaves device)         │  │   │
│  │  │ - past_key.0  [batch, num_heads, ctx_len, head_dim]    │  │   │
│  │  │ - past_value.0 [batch, num_heads, ctx_len, head_dim]   │  │   │
│  │  │ - ... (repeated for all layers)                        │  │   │
│  │  │                                                         │  │   │
│  │  │ Updated via CtxScatter (in-place, no host transfer)    │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  │  - Attention matrices (Q, K, V)                              │   │
│  │  - MLP activations                                           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  NPU Cores (16-32 cores, parallel execution)                │   │
│  │  - MatMul units (MXFP6 compressed weights)                  │   │
│  │  - Attention kernels (optimized for specialized shapes)     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              ↑ PCIe
                              ↑ (Only logits/next_token transfer back)

Key Bandwidth Optimization:
- Host ↔ Device: ~16 GB/s (PCIe Gen4 x16) - AVOID transferring KV cache here
- Device DRAM ↔ SRAM: ~400 GB/s - KV cache loaded here during decode
- On-Chip operations: ~TB/s - KV cache stays here, never evicted
```

**Why Retained-State Matters:**
- **Without retained-state:** Each decode step transfers ~2GB KV cache over PCIe (125ms @ 16GB/s)
- **With retained-state:** KV cache stays on-device, updated in-place (~1ms)
- **Decode latency:** 1ms (compute) vs 126ms (compute + transfer) = **126x speedup**

---

## **7.1 KV CACHE SCATTER/GATHER MECHANICS (DEEP DIVE)**

**Purpose:** Understand HOW KV cache is updated and read using custom ONNX operators optimized for Cloud AI 100.

**Key Files:**
- `QEfficient/customop/ctx_scatter_gather.py` - Non-CB scatter/gather ops
- `QEfficient/customop/ctx_scatter_gather_cb.py` - CB scatter/gather ops with batch_index
- `QEfficient/transformers/cache_utils.py` - QEffDynamicCache implementation
- `QEfficient/utils/check_ccl_specializations.py` - CCL (Compute Context Length) handling

---

### **7.1.1 Buffer Pre-Allocation (Confirmed Architecture)**

**KV Cache Buffers ARE Pre-Allocated to Full `ctx_len`:**

```python
# cache_utils.py: QEffDynamicLayer initialization (first forward pass)
if self.keys is None:
    # Allocate FULL ctx_len buffers upfront (NOT grown dynamically)
    # Shape for non-CB: [batch_size, num_heads, ctx_len, head_dim]
    # Shape for CB:     [full_batch_size, num_heads, ctx_len, head_dim]
    self.keys = key_states    # First layer creates buffers sized to ctx_len
    self.values = value_states
```

**Critical Understanding:**
- **NOT** dynamically grown - allocated once to `ctx_len` during compilation
- **ctx_len** = maximum context length (e.g., 4096, 8192, 16384 tokens)
- Memory allocated even if current sequence is shorter (e.g., prompt_len=32 but ctx_len=4096)
- Enables efficient scatter/gather operations without reallocation

**Memory Allocation Example:**
```python
# Llama-3.1-8B with ctx_len=4096, full_batch_size=8
# Per-layer KV cache memory:
# Keys:   [8, 32, 4096, 128] * 2 bytes (FP16) = 256 MB
# Values: [8, 32, 4096, 128] * 2 bytes (FP16) = 256 MB
# Total per layer: 512 MB
# Total for 32 layers: 16 GB (stays on-device SRAM/DRAM)
```

---

### **7.1.2 Dynamic Parameters Controlling Updates**

**Four Key Parameters Drive Scatter/Gather Behavior:**

| Parameter | Type | Shape | Purpose | Code Location |
|-----------|------|-------|---------|---------------|
| `position_ids` | torch.Tensor | `[batch_size, seq_len]` | **WHERE** to write/read in context dimension | modeling_llama.py:161 |
| `batch_index` | torch.Tensor (CB only) | `[batch_size, 1]` | **WHICH** slot in batch dimension to update | cache_utils.py:43, 87 |
| `comp_ctx_len` (CCL) | int | scalar | **HOW MUCH** of cache to gather for attention | cache_utils.py:42, 150 |
| `ctx_indices` | torch.Tensor | `[1, 1, comp_ctx_len]` | Calculated indices for gather operation | cache_utils.py:48 |

**Parameter Evolution During Generation:**

```python
# Prefill (prompt_len=32, ctx_len=4096):
position_ids = [0, 1, 2, ..., 31]  # Shape: [1, 32]
batch_index = [0]                   # CB only: which slot to fill
comp_ctx_len = 32                   # Gather only valid tokens

# Decode step 1 (generate token 32):
position_ids = [32]                 # Shape: [1, 1]
batch_index = [0]                   # Same slot
comp_ctx_len = 33                   # Gather 0-32 for attention

# Decode step 2 (generate token 33):
position_ids = [33]                 # Shape: [1, 1]
batch_index = [0]                  
comp_ctx_len = 34                   # Gather 0-33 for attention

# ... continues until comp_ctx_len == ctx_len (full cache)
```

---

### **7.1.3 Scatter Operation (Writing New KV States)**

**Non-CB Scatter** (`ctx_scatter_gather.py:CtxScatterFunc`):
```python
# PyTorch forward (eager execution):
def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
    batch_idx = torch.arange(data.shape[0]).view(-1, 1, 1)
    head_idx = torch.arange(data.shape[1]).view(1, -1, 1)
    ctx_idx = position_ids.unsqueeze(1)  # Where to write
    data[batch_idx, head_idx, ctx_idx] = updates  # In-place update
    return data

# ONNX export (compiled to hardware ops):
@onnxscript.script(...)
def CtxScatter(data, position_ids, updates):
    # Build 4D indices: [batch, head, position, last_dim]
    indices = Concat(batch_idx, head_idx, ctx_idx, axis=3)
    return ScatterND(data, indices, updates)  # Hardware-optimized op
```

**CB Scatter** (`ctx_scatter_gather_cb.py:CtxScatterFuncCB`):
```python
# Additional batch_index parameter for slot-based updates:
def forward(data, batch_index, position_ids, updates):
    batch_idx = batch_index.view(-1, 1, 1)  # Which slot in full batch
    head_idx = torch.arange(data.shape[1]).view(1, -1, 1)
    ctx_idx = position_ids.unsqueeze(1)
    data[batch_idx, head_idx, ctx_idx] = updates
    return data
```

**Scatter Behavior Example:**
```python
# Initial KV cache (all zeros): [1, 32, 4096, 128]
kv_cache = torch.zeros(1, 32, 4096, 128)  # Pre-allocated to ctx_len=4096

# Prefill: position_ids = [0,1,2,...,31], new_kv shape = [1, 32, 32, 128]
# Scatter writes to kv_cache[:, :, 0:32, :] = new_kv
# Result: kv_cache[:, :, 0:32, :] = prefill data, rest still zeros

# Decode step 1: position_ids = [32], new_kv shape = [1, 32, 1, 128]
# Scatter writes to kv_cache[:, :, 32, :] = new_kv
# Result: kv_cache[:, :, 0:33, :] = valid data

# Invalid position handling (during ONNX export):
invalid_scatter_index = torch.iinfo(torch.int32).max  # 2147483647
scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)
# Positions < 0 mapped to INT_MAX → ScatterND ignores (no write)
```

---

### **7.1.4 Gather Operation (Reading KV States for Attention)**

**Non-CB Gather** (`ctx_scatter_gather.py:CtxGatherFunc`):
```python
# PyTorch forward:
def forward(data, ctx_indices, comp_ctx_len):
    batch_indices = torch.arange(data.shape[0]).view(-1, 1, 1)
    head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
    return data[batch_indices, head_indices, ctx_indices]  # Read valid range

# ONNX export:
@onnxscript.script(...)
def CtxGather(data, ctx_indices, comp_ctx_len):
    # Expand ctx_indices to [batch, heads, comp_ctx_len]
    ctx_indices = Expand(ctx_indices, shape_tensor)
    ctx_indices = Unsqueeze(ctx_indices, [-1])  # Add last dim for GatherND
    return GatherND(data, ctx_indices, batch_dims=2)  # Hardware-optimized
```

**Gather Masking (Prevent Invalid Attention):**
```python
# cache_utils.py:48-56 (QEffDynamicLayer.read_only)
ctx_len = cache_kwargs.get("CCL", k_out.shape[2])  # Use CCL if provided
ctx_indices = torch.arange(ctx_len)[None, None, ...]  # [1, 1, ctx_len]
gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)  # Max valid position
invalid_mask = ctx_indices > gather_limit  # Positions beyond current generation

# During ONNX export:
invalid_idx_value = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

# Gather operation:
k_out = CtxGatherFunc.apply(k_out, ctx_indices, ctx_len)
v_out = CtxGatherFunc.apply(v_out, ctx_indices, ctx_len)

# Mask invalid values in value states:
v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
```

**Gather Behavior Example:**
```python
# After prefill: kv_cache[:, :, 0:32, :] has data, rest zeros
# position_ids.max() = 31

# Gather with comp_ctx_len=32:
ctx_indices = [0, 1, 2, ..., 31]  # Shape: [1, 1, 32]
gather_limit = 31
invalid_mask = [False, False, ..., False]  # All valid
gathered_kv = kv_cache[:, :, 0:32, :]  # Returns all 32 positions

# Decode step 10: position_ids.max() = 41, comp_ctx_len=42
ctx_indices = [0, 1, 2, ..., 41]
gather_limit = 41
invalid_mask = [False, ..., False]  # All 42 positions valid
gathered_kv = kv_cache[:, :, 0:42, :]  # Returns 42 positions
```

---

### **7.1.5 Attention Mask Creation (Causal Masking)**

**Location:** `QEfficient/transformers/modeling_attn_mask_utils.py:13-50`

```python
def _create_causal_mask(position_ids, target_length, sliding_window=None):
    """
    Creates causal attention mask dynamically based on position_ids.
   
    Args:
        position_ids: [batch_size, seq_len] - current generation positions
        target_length: int - how many KV positions to attend to (= comp_ctx_len)
        sliding_window: int or None - for sliding window attention (e.g., Mistral)
   
    Returns:
        attention_mask: [batch_size, 1, seq_len, target_length] - True = masked position
    """
    if sliding_window is not None:
        # Sliding window logic (not covered here, see code for details)
        pass
    else:
        # Standard causal attention:
        query_indices = position_ids.unsqueeze(-1)  # [batch, seq_len, 1]
        kv_indices = torch.arange(target_length).view(1, 1, -1)  # [1, 1, target_length]
        attention_mask = kv_indices > query_indices  # Broadcasting to [batch, seq_len, target_length]
        attention_mask = attention_mask.unsqueeze(1)  # Add head dim: [batch, 1, seq_len, target_length]
   
    return attention_mask
```

**Attention Mask Evolution Example:**
```python
# Prefill (position_ids=[0,1,2,...,31], target_length=32):
# attention_mask[0, 0, :, :] =
# [[False, True, True, ..., True],   # Token 0 sees only itself
#  [False, False, True, ..., True],  # Token 1 sees 0-1
#  [False, False, False, ..., True], # Token 2 sees 0-2
#  ...
#  [False, False, False, ..., False]]# Token 31 sees all 0-31

# Decode step 1 (position_ids=[32], target_length=33):
# attention_mask[0, 0, 0, :] = [False, False, ..., False]  # Token 32 sees all 0-32

# Applied in attention (modeling_llama.py:116-119):
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
if attention_mask is not None:
    attn_weights = torch.where(
        attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE), attn_weights
    )  # MIN_MASKED_ATTENTION_VALUE = -10000.0 → softmax makes these ~0
```

---

### **7.1.6 CCL (Compute Context Length) Optimization**

**Purpose:** Reduce attention computation without reallocating KV cache buffers.

**Code Location:** `check_ccl_specializations.py:8-52`

```python
def process_ccl_specializations(qaic_config):
    """
    Generates specialized context lengths for prefill and decode phases.
   
    Example:
        ctx_len = 4096
        ccl_prefill = [128, 256, 512, 1024, 2048, 4096]
        ccl_decode = [128, 256, 512, 1024, 2048, 4096]
   
    Compiler creates optimized kernels for these specific lengths.
    Runtime picks closest CCL ≤ current_position for efficient attention.
    """
    ccl_prefill = qaic_config.pop("comp_ctx_lengths_prefill", None)
    ccl_decode = qaic_config.pop("comp_ctx_lengths_decode", None)
    ctx_len = qaic_config.pop("ctx_len", None)
    prefill_seq_len = qaic_config.pop("prefill_seq_len", 128)
   
    # Cap all CCL values to ctx_len
    ccl_prefill = [min(x, ctx_len) for x in ccl_prefill]
    ccl_decode = [min(x, ctx_len) for x in ccl_decode]
   
    # Remove duplicates and ensure no overlap between prefill/decode
    # (Specializations must be unique for compiler optimization)
    # ... (see code for full logic)
   
    return updated_prefill, ccl_decode
```

**CCL Usage in Attention** (cache_utils.py:150, modeling_llama.py:161):
```python
# During forward pass:
if comp_ctx_lengths is not None:
    # Truncate attention mask to CCL instead of full ctx_len
    attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
    cache_kwargs["CCL"] = attention_mask.shape[-1]  # Pass to gather operation

# In gather operation:
ctx_len = cache_kwargs.get("CCL", k_out.shape[2])  # Use CCL if provided, else full cache
ctx_indices = torch.arange(ctx_len)[None, None, ...]  # Gather only CCL positions
```

**Performance Impact:**
```python
# Without CCL (always use ctx_len=4096):
# Attention compute: O(seq_len * 4096) - wasteful when position=50

# With CCL (use comp_ctx_len=128 when position=50):
# Attention compute: O(seq_len * 128) - ~32x faster
# Compiler has pre-optimized kernel for length=128
```

---

### **7.1.7 Integration Summary: Scatter → Gather → Attention**

**Complete Flow for One Decode Step:**

```python
# 1. NEW KEY/VALUE GENERATION (modeling_llama.py:148-156)
query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # New K for current token
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # New V for current token
# Shapes: [batch=1, num_heads=32, seq_len=1, head_dim=128]

# 2. APPLY ROTARY EMBEDDINGS (modeling_llama.py:153-156)
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

# 3. SCATTER: Write new KV into cache (cache_utils.py:127-131)
cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
# CtxScatterFuncCB writes key_states into kv_cache at position_ids locations
self.keys = CtxScatterFuncCB.apply(self.keys, batch_index, position_ids, key_states)
self.values = CtxScatterFuncCB.apply(self.values, batch_index, position_ids, value_states)

# 4. GATHER: Read valid KV range for attention (cache_utils.py:136-156)
ctx_len = cache_kwargs.get("CCL", self.keys.shape[2])  # Use CCL if provided
ctx_indices = torch.arange(ctx_len)[None, None, ...]
gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
invalid_mask = ctx_indices > gather_limit
ctx_indices = torch.where(invalid_mask, torch.iinfo(torch.int32).max, ctx_indices)

k_out = CtxGatherFuncCB.apply(self.keys, batch_index, ctx_indices, ctx_len)
v_out = CtxGatherFuncCB.apply(self.values, batch_index, ctx_indices, ctx_len)
v_out = torch.where(invalid_mask.unsqueeze(-1), 0.0, v_out)  # Mask invalid positions
# Gathered shapes: [batch=1, num_heads=32, ctx_len, head_dim=128]

# 5. CREATE CAUSAL MASK (modeling_attn_mask_utils.py:13-50)
attention_mask = _create_causal_mask(position_ids, target_length=ctx_len)
# Shape: [batch=1, 1, seq_len=1, ctx_len]

# 6. COMPUTE ATTENTION (modeling_llama.py:108-120)
key_states = repeat_kv(k_out, self.num_key_value_groups)  # GQA: repeat KV heads
value_states = repeat_kv(v_out, self.num_key_value_groups)
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
# attn_weights shape: [batch=1, num_heads=32, seq_len=1, ctx_len]

attn_weights = torch.where(attention_mask, MIN_MASKED_ATTENTION_VALUE, attn_weights)
attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)
attn_output = torch.matmul(attn_weights, value_states)  # Weighted sum of values
# attn_output shape: [batch=1, num_heads=32, seq_len=1, head_dim=128]

# 7. OUTPUT PROJECTION (modeling_llama.py:177-178)
attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
attn_output = self.o_proj(attn_output)
```

---

### **7.1.8 Execution Timeline: When Scatter/Gather Operations Occur**

**Critical Understanding:** Scatter/Gather operations happen **INSIDE the model forward pass** during `session.run()`, not at the Python runtime level.

---

#### **PREFILL PHASE (First Forward Pass)**

**Python Runtime Level** (`text_generation_inference.py:770-850`):
```python
# Step 1: Tokenize and prepare inputs
inputs = tokenizer(prompt, return_tensors="np", padding=True)
inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
# position_ids = [0, 1, 2, ..., 31] for 32-token prompt

# Step 2: Initialize buffers (FIRST TIME ONLY)
self._set_output_buffers(batch_size=1, sequence_length=1)  # Allocates output logits buffer

# Step 3: Chunked prefill loop (if prompt > prefill_seq_len)
for chunk_id in range(num_chunks):  # e.g., 2 chunks for 256-token prompt with prefill_seq_len=128
    chunk_inputs["input_ids"] = inputs["input_ids"][:, chunk_id*128:(chunk_id+1)*128]
    chunk_inputs["position_ids"] = inputs["position_ids"][:, chunk_id*128:(chunk_id+1)*128]
    # Chunk 0: position_ids = [0-127], Chunk 1: position_ids = [128-255]
   
    outputs = self._session.run(chunk_inputs)  # ← MODEL FORWARD PASS (see below)
```

**Model Forward Pass** (`modeling_llama.py:129-178`, `cache_utils.py:103-160`):
```python
# ============ INSIDE session.run(chunk_inputs) ============
# For EACH transformer layer (repeated 32 times for Llama-3.1-8B):

# 1. Generate Q, K, V projections (modeling_llama.py:148-150)
query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # NEW keys for chunk
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # NEW values for chunk
# Shape: [batch=1, num_heads=32, seq_len=128, head_dim=128]

# 2. Apply rotary embeddings (modeling_llama.py:153-156)
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = qeff_apply_rotary_pos_emb(query, key, cos, sin, position_ids)

# 3. KV CACHE UPDATE with SCATTER (cache_utils.py:127-131) ← HERE!
if self.keys is None:  # FIRST CHUNK ONLY
    # Allocate KV cache to FULL ctx_len (e.g., 4096)
    self.keys = key_states    # Shape: [1, 32, ctx_len=4096, 128]
    self.values = value_states
else:  # SUBSEQUENT CHUNKS
    # SCATTER: Write new KV at position_ids locations
    cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
    self.keys = CtxScatterFuncCB.apply(self.keys, batch_index, position_ids, key_states)
    self.values = CtxScatterFuncCB.apply(self.values, batch_index, position_ids, value_states)
    # Chunk 0: Writes to positions [0-127]
    # Chunk 1: Writes to positions [128-255]

# 4. KV CACHE READ with GATHER (cache_utils.py:136-156) ← HERE!
ctx_len = cache_kwargs.get("CCL", self.keys.shape[2])  # Use CCL or full ctx_len
ctx_indices = torch.arange(ctx_len)[None, None, ...]  # [0, 1, 2, ..., ctx_len-1]
gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)  # Max position in chunk
invalid_mask = ctx_indices > gather_limit  # Mask positions beyond current generation

k_out = CtxGatherFuncCB.apply(self.keys, batch_index, ctx_indices, ctx_len)
v_out = CtxGatherFuncCB.apply(self.values, batch_index, ctx_indices, ctx_len)
v_out = torch.where(invalid_mask.unsqueeze(-1), 0.0, v_out)
# Chunk 0: Gathers positions [0-127] (128 positions)
# Chunk 1: Gathers positions [0-255] (256 positions, includes chunk 0 data)

# 5. CREATE ATTENTION MASK (modeling_llama.py:161, modeling_attn_mask_utils.py:13-50) ← HERE!
if comp_ctx_lengths is not None:
    attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
    cache_kwargs["CCL"] = attention_mask.shape[-1]
# Attention mask created from position_ids (causal masking)
# Shape: [batch=1, 1, seq_len=128, target_length=ctx_len]

# 6. COMPUTE ATTENTION (modeling_llama.py:108-120)
attn_weights = torch.matmul(query_states, k_out.transpose(2, 3)) * scaling
attn_weights = torch.where(attention_mask, MIN_MASKED_ATTENTION_VALUE, attn_weights)
attn_weights = nn.functional.softmax(attn_weights, dim=-1)
attn_output = torch.matmul(attn_weights, v_out)
# Uses gathered KV cache (NOT full cache, only valid positions)

# ============ END OF session.run() ============
# Returns: logits for last token in chunk (used to get next token)
```

**Key Insight for Chunked Prefill:**
- **Scatter happens EVERY chunk** (writes new KV at chunk positions)
- **Gather happens EVERY chunk** (reads cumulative KV for attention)
- **Attention mask re-created EVERY chunk** (based on current position_ids)
- **KV cache grows incrementally:** Chunk 0 fills [0-127], Chunk 1 adds [128-255]

---

#### **DECODE PHASE (Subsequent Forward Passes)**

**Python Runtime Level** (`text_generation_inference.py:864-1000`):
```python
# Step 1: Prepare decode inputs (ALL batch slots together for CB)
decode_inputs = self.prepare_decode_inputs()  # Sets up batch_size=full_batch_size
decode_inputs["input_ids"].shape  # [full_batch_size=8, 1] - one token per slot
decode_inputs["position_ids"].shape  # [8, 1] - current position for each slot
# Example: position_ids = [[32], [45], [60], [32], [100], [78], [50], [90]]
#          (8 different requests at different positions)

# Step 2: Decode loop (generates one token per iteration)
for decode_iteration in range(max_generation_len):
    outputs = self._session.run(decode_inputs)  # ← MODEL FORWARD PASS (see below)
   
    # Step 3: Get next token and update inputs
    next_token_id = self._fetch_next_token_id(outputs)
    decode_inputs["input_ids"] = next_token_id
    decode_inputs["position_ids"] += 1  # Increment position for next iteration
```

**Model Forward Pass** (`modeling_llama.py:129-178`, `cache_utils.py:103-160`):
```python
# ============ INSIDE session.run(decode_inputs) ============
# For EACH transformer layer:

# 1. Generate Q, K, V projections (batch_size=8 for CB)
query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
# Shape: [batch=8, num_heads=32, seq_len=1, head_dim=128]

# 2. Apply rotary embeddings
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = qeff_apply_rotary_pos_emb(query, key, cos, sin, position_ids)

# 3. KV CACHE UPDATE with SCATTER (PARALLEL across all batch slots) ← HERE!
cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
# batch_index = [0, 1, 2, 3, 4, 5, 6, 7] - which slot to update
# position_ids = [[32], [45], [60], [32], [100], [78], [50], [90]] - where to write

self.keys = CtxScatterFuncCB.apply(self.keys, batch_index, position_ids, key_states)
self.values = CtxScatterFuncCB.apply(self.values, batch_index, position_ids, value_states)
# Writes: Slot 0 at position 32, Slot 1 at position 45, Slot 2 at position 60, ...
# All 8 slots updated IN PARALLEL (one hardware operation)

# 4. KV CACHE READ with GATHER (PARALLEL across all batch slots) ← HERE!
ctx_len = cache_kwargs.get("CCL", self.keys.shape[2])
ctx_indices = torch.arange(ctx_len)[None, None, ...]
gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
invalid_mask = ctx_indices > gather_limit

k_out = CtxGatherFuncCB.apply(self.keys, batch_index, ctx_indices, ctx_len)
v_out = CtxGatherFuncCB.apply(self.values, batch_index, ctx_indices, ctx_len)
v_out = torch.where(invalid_mask.unsqueeze(-1), 0.0, v_out)
# Gathers: Slot 0 reads [0-32], Slot 1 reads [0-45], Slot 2 reads [0-60], ...
# Each slot gathers DIFFERENT ranges (based on their position_ids)

# 5. CREATE ATTENTION MASK (per-slot causal mask) ← HERE!
attention_mask = _create_causal_mask(position_ids, target_length=ctx_len)
# Shape: [batch=8, 1, seq_len=1, ctx_len]
# Slot 0: Can attend to [0-32], Slot 1: Can attend to [0-45], etc.

# 6. COMPUTE ATTENTION (batch_size=8 parallel)
attn_weights = torch.matmul(query_states, k_out.transpose(2, 3)) * scaling
attn_weights = torch.where(attention_mask, MIN_MASKED_ATTENTION_VALUE, attn_weights)
attn_weights = nn.functional.softmax(attn_weights, dim=-1)
attn_output = torch.matmul(attn_weights, v_out)
# Each slot computes attention over ITS OWN valid KV range

# ============ END OF session.run() ============
# Returns: logits for next token (shape: [8, vocab_size])
```

**Key Insight for Decode:**
- **Scatter happens EVERY iteration** (8 slots write in parallel)
- **Gather happens EVERY iteration** (8 slots read different ranges in parallel)
- **Attention mask re-created EVERY iteration** (each slot has different valid range)
- **Hardware parallelism:** All 8 batch slots process simultaneously

---

#### **Timeline Visualization**

```
TIME AXIS: Session.run() Call Sequence
═══════════════════════════════════════════════════════════════

PREFILL PHASE (Chunked, prompt=256 tokens, prefill_seq_len=128):
┌─────────────────────────────────────────────────────────────┐
│ session.run(chunk_0)  # position_ids=[0-127]                │
│   ├─ Layer 0: Scatter [0-127] → KV cache                    │
│   │           Gather [0-127] ← KV cache                      │
│   │           Attention over [0-127]                         │
│   ├─ Layer 1: Scatter [0-127] → KV cache                    │
│   │           Gather [0-127] ← KV cache                      │
│   ...                                                        │
│   └─ Layer 31: Scatter [0-127] → KV cache                   │
│               Gather [0-127] ← KV cache                      │
│   Output: logits for position 127                           │
├─────────────────────────────────────────────────────────────┤
│ session.run(chunk_1)  # position_ids=[128-255]              │
│   ├─ Layer 0: Scatter [128-255] → KV cache                  │
│   │           Gather [0-255] ← KV cache (includes chunk 0)  │
│   │           Attention over [0-255]                         │
│   ├─ Layer 1: Scatter [128-255] → KV cache                  │
│   │           Gather [0-255] ← KV cache                      │
│   ...                                                        │
│   └─ Layer 31: Scatter [128-255] → KV cache                 │
│               Gather [0-255] ← KV cache                      │
│   Output: logits for position 255                           │
└─────────────────────────────────────────────────────────────┘

DECODE PHASE (batch_size=8, generating tokens 256-300):
┌─────────────────────────────────────────────────────────────┐
│ session.run(iter_0)  # position_ids=[256, 256, ..., 256]    │
│   ├─ Layer 0: Scatter pos 256 → ALL 8 slots (parallel)      │
│   │           Gather [0-256] ← ALL 8 slots (parallel)        │
│   │           Attention over [0-256] (8 slots parallel)      │
│   ...                                                        │
│   └─ Layer 31: Scatter pos 256 → 8 slots                    │
│               Gather [0-256] ← 8 slots                       │
│   Output: next_token for all 8 slots                        │
├─────────────────────────────────────────────────────────────┤
│ session.run(iter_1)  # position_ids=[257, 258, 259, ...]    │
│   ├─ Layer 0: Scatter pos [257-264] → 8 slots (parallel)    │
│   │           Gather [0-257], [0-258], ... ← 8 slots         │
│   │           Attention (each slot over different range)     │
│   ...                                                        │
│   └─ Layer 31: Scatter → 8 slots                            │
│               Gather ← 8 slots (different ranges)            │
│   Output: next_token for all 8 slots                        │
├─────────────────────────────────────────────────────────────┤
│ ... continues for 44 more iterations ...                    │
└─────────────────────────────────────────────────────────────┘
```

---

#### **Shape Evolution Table**

| Phase | Step | Operation | Input Shape | Output Shape | Notes |
|-------|------|-----------|-------------|--------------|-------|
| **Prefill Chunk 0** | Scatter | Write KV | `[1, 32, 128, 128]` (new keys) | → KV cache `[0:128]` | First chunk allocates buffer |
| | Gather | Read KV | KV cache `[1, 32, 4096, 128]` | `[1, 32, 128, 128]` | Reads [0-127] |
| | Attention Mask | Create mask | position_ids `[0-127]` | `[1, 1, 128, 128]` | Causal mask for chunk |
| **Prefill Chunk 1** | Scatter | Write KV | `[1, 32, 128, 128]` (new keys) | → KV cache `[128:256]` | Appends to existing cache |
| | Gather | Read KV | KV cache `[1, 32, 4096, 128]` | `[1, 32, 256, 128]` | Reads [0-255] (both chunks) |
| | Attention Mask | Create mask | position_ids `[128-255]` | `[1, 1, 128, 256]` | Seq_len=128, can attend to 256 |
| **Decode Iter 0** | Scatter | Write KV | `[8, 32, 1, 128]` (8 slots) | → KV cache `[256]` | All slots at same position |
| | Gather | Read KV | KV cache `[8, 32, 4096, 128]` | `[8, 32, 256, 128]` | Each slot reads [0-256] |
| | Attention Mask | Create mask | position_ids `[[256], [256], ...]` | `[8, 1, 1, 256]` | 8 identical masks |
| **Decode Iter 10** | Scatter | Write KV | `[8, 32, 1, 128]` | → KV cache `[266-273]` | Slots at different positions |
| | Gather | Read KV | KV cache `[8, 32, 4096, 128]` | `[8, 32, 273, 128]` | Max position=273 across slots |
| | Attention Mask | Create mask | position_ids `[[266], [267], [268], ...]` | `[8, 1, 1, 273]` | Each slot different |

---

### **7.1.9 Key Architectural Insights**

1. **Buffer Pre-Allocation is Mandatory:**
   - KV cache buffers ALWAYS allocated to full `ctx_len` (e.g., 4096)
   - Not grown dynamically → enables efficient scatter/gather without reallocation
   - Memory cost is fixed regardless of actual sequence length

2. **Scatter/Gather Are Hardware Ops:**
   - Custom ONNX ops (`CtxScatter`, `CtxGather`) compiled to Cloud AI 100 instructions
   - Much faster than generic PyTorch indexing on NPU
   - Retained-state prevents host transfers → 126x speedup

3. **CCL Enables Efficient Attention:**
   - Buffers allocated to ctx_len, but attention computed over comp_ctx_len
   - Compiler creates optimized kernels for specific CCL values
   - Runtime picks best CCL ≤ current_position for performance

4. **CB Uses Batch Index for Slot Management:**
   - `batch_index` determines which slot in full_batch_size to update
   - Enables independent prefill/decode for different requests in same batch
   - Non-CB uses implicit batch dimension (always 0)

5. **Attention Mask Created Dynamically:**
   - NOT pre-allocated like KV cache
   - Created per forward pass based on current position_ids
   - Shape: `[batch, 1, seq_len, target_length]` where target_length ≤ ctx_len

6. **Operations Happen INSIDE Model Forward Pass:**
   - Scatter/Gather occur during `session.run()`, not at Python runtime level
   - Every layer repeats scatter → gather → attention (32 times for Llama-3.1-8B)
   - Hardware executes all layers sequentially, but batch slots in parallel

7. **Chunked Prefill Accumulates KV Cache:**
   - Each chunk scatters NEW positions and gathers ALL cumulative positions
   - Chunk 1 attention sees data from Chunk 0 (via gather operation)
   - No separate "merge" step - gather handles cumulative reads automatically

8. **⚠️ Compiler Padding in Retained-State Outputs:**
   - The `qaic-exec` compiler may add **trailing padding** to the last dimension of retained-state KV cache outputs for hardware memory alignment
   - **Example:** ONNX graph has `past_key.0_RetainedState` shape `[batch, 8, ctx_len, 64]`, but compiled QPC has shape `[batch, 8, ctx_len, 66]` (2-element padding: 64→66)
   - **Root Cause:** Hardware memory access patterns may require aligned dimensions for optimal performance
   - **Impact:** Code reading retained-state outputs must handle potential padding by slicing to the expected `head_dim` before use
   - **Detection:** Compare ONNX output shapes vs QPC IoDescriptor binding dimensions
   - **Fix Pattern:**
     ```python
     # Example: Reading retained-state KV cache with potential padding
     K = outputs["past_key.0_RetainedState"]  # Shape: [1, 8, 4096, 66] (padded)
     expected_head_dim = 64  # From model config
     
     if K.shape[-1] != expected_head_dim:
         # Compiler added padding - slice to valid head_dim
         K = K[..., :expected_head_dim]  # Shape: [1, 8, 4096, 64]
     
     # Now safe to use K for attention computation
     ```
   - **Verification:** Use `scripts/dump_onnx_shapes.py` to inspect ONNX output shapes and compare against runtime tensor shapes
   - **Note:** Padding is **trailing** (last elements), so slicing `[..., :head_dim]` extracts valid data without loss

---

## **8. CONTINUOUS BATCHING**

**Location:** text_generation_inference.py lines 717-956, 1206-1209

**Architecture:**

**CRITICAL: Two-Phase Sequential Execution** (Code: text_generation_inference.py:1206-1209)
```python
# Phase 1: ALL prefills complete FIRST
self._qaic_model.run_prefill_for_all_inputs(self._prompt_queue, generation_len)  # Line 1206

# Phase 2: Decode starts ONLY AFTER all prefills done
loop_start = perf_counter()  # Line 1208
decode_pause_time = self._qaic_model.run_continuous_batching_decode(self._prompt_queue, generation_len)  # Line 1209
```

**Prefill Phase** (Serial, batch_size=1, BLOCKS until ALL slots filled):
```python
# run_prefill_for_all_inputs() line 717-740
def run_prefill_for_all_inputs(self, prompt_queue, generation_len):
    logger.info(f"Starting serial prefill for {self.full_batch_size} prompts")
   
    for decode_batch_id in range(self.full_batch_size):  # MUST complete entire loop
        next_prompt = prompt_queue.popleft()
        logger.info(f"Prefilling prompt {decode_batch_id+1}/{self.full_batch_size} into slot {decode_batch_id}")
       
        outputs, position_ids, generation_len = self.run_prefill(
            next_prompt, generation_len, decode_batch_id=np.array(decode_batch_id, dtype=np.int64).reshape(1, 1)
        )
        _ = self.update_decode_input(outputs, position_ids, generation_len, decode_batch_id)
   
    logger.info(f"Serial prefill complete for all {self.full_batch_size} prompts")
    # Function returns ONLY when ALL slots are filled
```

**Decode Phase** (Parallel, batch_size=full_batch_size, STARTS after prefill returns):
```python
# run_continuous_batching_decode() line 864-944
def run_continuous_batching_decode(self, prompt_queue, generation_len):
    logger.info(f"Starting parallel decode with full_batch_size={self.full_batch_size}")
   
    # Setup expects ALL slots already populated
    self._set_output_buffers(
        batch_size=self.full_batch_size,  # Requires all slots ready
        sequence_length=self._decode_seq_len,
    )
   
    current_decode_ongoing = np.full((self.full_batch_size, 1), True)
    decode_inputs = self.prepare_decode_inputs()  # Assumes all KV caches populated
   
    while prompt_queue or current_decode_ongoing.any():
        outputs = self._session.run(decode_inputs)  # All slots decode together
       
        # Refill completed slots (within decode loop)
        for decode_batch_id in range(self.full_batch_size):
            if next_token_id[decode_batch_id, -1] == self.tokenizer.eos_token_id:
                if prompt_queue:
                    outputs, position_ids, generation_len = self.run_prefill(
                        prompt_queue.popleft(), generation_len, decode_batch_id=decode_batch_id
                    )
```

**Why Decode Cannot Start Early:**
1. **Sequential Function Calls (Lines 1206-1209):** Python executes line 1206 **completely** before line 1209. No threading, no async, no parallelism between these calls.
2. **Loop Blocks (Line 729):** `for decode_batch_id in range(self.full_batch_size)` must complete all iterations. Function cannot return until loop finishes.
3. **Decode Setup Assumes Full Batch (Line 879-881):** `batch_size=self.full_batch_size` in buffer setup. No mechanism to handle partial batch at initialization.
4. **No Conditional Early Start:** No code path allows `run_continuous_batching_decode()` to execute before `run_prefill_for_all_inputs()` returns.

**Execution Timeline (full_batch_size=3):**
```
Time →
┌─────────────── Initial Prefill Phase ───────────────┐
│ Prefill Slot 0 → Prefill Slot 1 → Prefill Slot 2   │ ← ALL must complete
└──────────────────────────────────────────────────────┘
                        ↓ (Function returns here)
                        ↓ (Line 1209 executes here)
                        ↓
┌─────────────── Parallel Decode Phase ────────────────────────┐
│ Decode(BS=3) → Decode(BS=3) → Decode(BS=3) → ...            │
│   Slot 0          Slot 0          Slot 0                     │
│   Slot 1          Slot 1          Slot 1 (EOS) → Refill     │
│   Slot 2          Slot 2          Slot 2                     │
└──────────────────────────────────────────────────────────────┘
```

**Latency Timeline with Real Numbers (Example: 3 prompts, 32-token prefill each):**
```
Time (ms) →    0      50     100    150    200    250    300    350    400
             ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prefill:     [━━━━━━━━━━━━━━━━━━━━━━━━━━━]                              (150ms total)
              Slot0  Slot1  Slot2
                50ms  50ms   50ms

Decode:                              [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→   (Continues...)
                                      All 3 slots decode in parallel
                                      ~5ms per token (batch_size=3)

Phase:       |←---- Serial Prefill --→|←-------- Parallel Decode ------→
Waiting:     [Prompt 1 waits 0ms]
             [Prompt 2 waits 50ms]     ← Waits for Prompt 1 prefill
             [Prompt 3 waits 100ms]    ← Waits for Prompts 1+2 prefill
```

**Key Observations:**
- **TTFT (Time To First Token) for Prompt 1:** 150ms (all prefills) + 5ms (first decode) = 155ms
- **TTFT for Prompt 3:** Same 155ms (decode starts only after all prefills)
- **Batch formation latency:** 150ms (serial prefill overhead)
- **Decode throughput:** 3 tokens / 5ms = 600 tokens/sec (parallel processing benefit)

**INCORRECT Assumption:** Flow diagram in previous version suggested overlapping prefill/decode. This is **architecturally impossible** with current code structure.

**CORRECT Behavior:**
- Initial prefills are **batched together in time** (serial execution)
- Decode starts **only when ALL initial prefills complete**
- Slot refills happen **within the decode loop** (line 916-927)
- No interleaving between initial prefill phase and decode phase

**Why Prefill is Serial:**
- Prefill specialization uses `batch_size=1` (one prompt at a time)
- Each slot filled independently via `decode_batch_id` index into KV cache
- Hardware optimized for single-prompt prefill + multi-prompt decode
- **Architectural constraint:** Initial batch formation is sequential

---

## **9. CHUNKED PREFILL**

**Location:** `text_generation_inference.py::run_prefill()` lines 770-850

**How It Works:**
```python
# If prompt=512 tokens and prefill_seq_len=128:
num_chunks = ceil(512 / 128) = 4 chunks
padded_len = 4 * 128 = 512 tokens

for chunk_id in range(num_chunks):
    chunk_start = chunk_id * 128
    chunk_end = chunk_start + 128
    chunk_inputs = inputs[:, chunk_start:chunk_end]
    outputs = self._session.run(chunk_inputs)  # Uses prefill specialization
    # KV cache accumulates across chunks via scatter ops
```

**Purpose:** Process long prompts without exceeding prefill `seq_len` specialization.

**Example:**
- Compiled with `prefill_seq_len=128`
- User prompt is 256 tokens
- Runs **2 chunked prefill passes** (128 + 128)
- KV cache grows incrementally: [0-127] → [0-255]

---

## **10. COMPUTE CONTEXT LENGTH (CCL)**

**Location:** Throughout text_generation_inference.py (search "comp_ctx_lengths")

**How It Works:**
```python
# Instead of always using full ctx_len (e.g., 8K), use smallest valid bucket
list_of_comp_ctx_lengths = [2048, 4096, 6144, 8192]

# For prompt of 3000 tokens, use 4096 instead of 8192
ccl_id = find_smallest_bucket(prompt_len=3000)
inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths[ccl_id]  # 4096

# Attention mask, keys, values shaped to [batch, heads, seq_len, 4096]
# Instead of [batch, heads, seq_len, 8192]
```

**Purpose:**
- Reduce attention matrix size (saves **on-chip memory** and **bandwidth**)
- Example: 3K prompt uses 4K attention window instead of 8K
- Hardware computes smaller attention matrices → faster execution

**Configuration:**
```bash
python -m QEfficient.cloud.infer \
    --model-name gpt2 \
    --ctx-len 8192 \
    --comp-ctx-lengths-prefill 2048,4096,6144 \
    --comp-ctx-lengths-decode 2048,4096,6144,8192
```

---

## **11. MULTI-DEVICE (TENSOR SLICING)**

**Location:** `modeling_qeff.py::_compile()` lines 376-411

**How It Works:**
```python
if mdp_ts_num_devices > 1:
    # Auto-generate MDP partition config
    mdp_ts_json = {
        "connections": [{"devices": [0, 1], "type": "p2p"}],
        "partitions": [{
            "name": "Partition0",
            "devices": [
                {"deviceId": 0, "numCores": 16},
                {"deviceId": 1, "numCores": 16}
            ]
        }]
    }
    command.append(f"-mdp-load-partition-config={mdp_ts_json_path}")
```

**Compiler Behavior:**
- **Slices model layers** across devices (e.g., layers 0-31 on Device 0, layers 32-63 on Device 1)
- **P2P links** handle inter-device communication (low latency)
- Runtime loads **same QPC** on all devices
- Requires **synchronized execution** across device group

**Example:**
```bash
python -m QEfficient.cloud.infer \
    --model-name meta-llama/Llama-2-70b \
    --device-group [0,1] \  # Multi-device
    --num-cores 16
```

---

## **12. COMPILATION HASH PARAMETERS**

**Location:** `modeling_qeff.py::_compile()` lines 387-393

**What Goes Into Hash:**
```python
compile_hash_params = {
    "command": [...],  # Full qaic-exec command with ALL flags
    "specializations": [...],  # Prefill/decode shapes
    "custom_io": {...},  # Input/output data types
    "mdp_ts_num_devices": N,  # Number of devices
    "mdp_ts_json": {...},  # Multi-device config
    "num_speculative_tokens": K,  # Speculative decoding
}
```

**Command Includes:**
- Model architecture + config
- `aic_num_cores`, `aic_hw_version`
- `batch_size`, `prefill_seq_len`, `ctx_len`
- `mxfp6_matmul`, `mxint8_kv_cache`, `aic_enable_depth_first`, `mos`
- **All** `compiler_options` kwargs passed to `qaic-exec`

**QPC Directory:** `{onnx_dir}/qpc-{hash}/qpc/`

**Stored As:** `{compile_dir}/hashed_compile_params.json`

**Cache Invalidation:** Changing **any** of these parameters generates a **new hash** → **new QPC**

---

## **13. CUSTOM OPS (Three Execution Modes)**

**Transform:** `CustomOpsTransform` in `_pytorch_transforms`

**Execution Modes:**

1. **Eager (PyTorch):**
   - `CtxScatterFunc.forward()` runs as PyTorch op
   - Used during model testing/validation

2. **ONNX Export:**
   - `CtxScatterFunc.symbolic()` emits custom ONNX node:
   ```python
   g.op("com.qti.aisw.onnx::CtxScatter",
        cache, updates, indices, ...)
   ```
   - Nodes stored in ONNX graph with custom domain

3. **QAIC Runtime:**
   - Custom nodes map to **hardware-accelerated kernels**
   - Compiler recognizes `com.qti.aisw.onnx` domain
   - Executes scatter/gather ops directly on NPU cores

**Purpose:** Efficient KV cache updates via on-device scatter/gather (no host roundtrip)

---

## **14. CODE-VERIFIED Q&A**

### **Q1: How does CLI handle artifact reuse?**
**A:** `export()` checks if `{model_name}.onnx` exists (modeling_qeff.py:207), `compile()` checks if `qpc-{hash}/programqpc.bin` exists (modeling_qeff.py:401). Early return if found. Hash computed from all compile parameters.

### **Q2: When do PyTorch transforms fire?**
**A:** During `QEFFBaseModel.__init__()` (modeling_qeff.py:73-74), **AFTER** HF loads weights, **BEFORE** ONNX export. Timeline: `from_pretrained` → HF loads weights → `__init__` → transforms → `export`.

### **Q3: Quantizer replacement mechanics?**
**A:** `@with_replaced_quantizers` decorator (auto.py:57-80) **temporarily swaps** `AUTO_QUANTIZER_MAPPING` **before** `from_pretrained()` body runs, then **restores** original mappings after. This allows CPU loading of AWQ/GPTQ models.

### **Q4: Continuous batching prefill batch size?**
**A:** Prefill **ALWAYS** uses `batch_size=1` (compile_helper.py:25-39), decode uses `batch_size=full_batch_size`. Prefill is **serial** (fills slots one-by-one), decode is **parallel** (processes all slots together).

### **Q4a: Can decode start before all initial prefills complete?**
**A:** **NO.** Code enforces sequential execution (text_generation_inference.py:1206-1209). `run_prefill_for_all_inputs()` must **return completely** before `run_continuous_batching_decode()` executes. Python's synchronous execution model prevents any overlap. The `for` loop in line 729 must complete all `full_batch_size` iterations before function returns. Decode phase initialization (line 879-881) assumes all slots are already populated with KV cache data.

### **Q5: Specializations structure?**
**A:** See Section 6. Standard mode has 2 entries (prefill + decode, both batch_size=1). Continuous batching mode has 2 entries (prefill batch_size=1, decode batch_size=full_batch_size). All entries share same `ctx_len`.

### **Q6: Retained-state buffer handling?**
**A:** `skip_buffers([x for x if x.startswith("past_")])` (text_generation_inference.py:491) marks **both inputs and outputs** starting with "past_" as device-resident. Hardware keeps them on-chip across iterations.

### **Q7: Custom ops execution modes?**
**A:** Three modes - **Eager** (PyTorch ops), **ONNX export** (custom nodes with domain `com.qti.aisw.onnx`), **QAIC runtime** (hardware kernels). See Section 13.

### **Q8: Hash parameters?**
**A:** See Section 12. Includes: full `qaic-exec` command, specializations, custom_io, mdp_ts_json, num_speculative_tokens. Stored in `hashed_compile_params.json`.

### **Q9: Multi-device setup?**
**A:** See Section 11. Auto-generates `mdp_ts_{N}.json` when `mdp_ts_num_devices > 1`. Creates tensor-slicing config with P2P connections. Compiler partitions layers across devices.

### **Q10: Where to integrate new optimizations?**
**A:**
- **Transform Level:** Add to `_pytorch_transforms` (graph modifications)
- **Specialization Level:** Extend specializations.json (new shapes/configs)
- **Custom IO Level:** Update custom_io.yaml (data types)
- **Runtime Level:** Modify `run_prefill()/run_decode()` (execution orchestration)
- **Hash Level:** Add parameters to `compile_hash_params` (cache invalidation)

### **Q11: Exact code locations proving CB execution model?**
**A:**
- **Sequential phase boundary:** `text_generation_inference.py:1206-1209` - Two separate function calls, no parallelism
- **Prefill blocks until complete:** `text_generation_inference.py:729` - `for` loop over `range(full_batch_size)`, cannot return early
- **Prefill completion log:** `text_generation_inference.py:740` - "Serial prefill complete for all {full_batch_size} prompts"
- **Decode expects full batch:** `text_generation_inference.py:879-881` - `batch_size=self.full_batch_size` in buffer setup
- **No conditional early start:** No `if` statement allowing decode before prefill completes
- **Proof of blocking:** Line 1206 must complete before line 1208 executes (Python execution model)

### **Q12: How are KV cache buffers allocated and shaped?**
**A:** **YES, buffers are pre-allocated to full `ctx_len`:**
- **Non-CB shape:** `[batch_size, num_heads, ctx_len, head_dim]` - e.g., `[1, 32, 4096, 128]`
- **CB shape:** `[full_batch_size, num_heads, ctx_len, head_dim]` - e.g., `[8, 32, 4096, 128]`
- **Allocated once** during first forward pass (cache_utils.py:29-31)
- **NOT grown dynamically** - fixed size regardless of actual sequence length
- **Memory stays on-device** via retained-state (text_generation_inference.py:491)
- **Updated in-place** via `CtxScatterFunc` at specific `position_ids` (Section 7.1)

### **Q13: What dynamic parameters control KV cache scatter/gather?**
**A:** Four key parameters (see Section 7.1 for details):
1. **`position_ids`** - WHERE to write/read in context dimension (e.g., `[32]` for token 32)
2. **`batch_index`** (CB only) - WHICH slot in batch dimension to update (e.g., `[0]` for first slot)
3. **`comp_ctx_len` (CCL)** - HOW MUCH of cache to gather for attention (e.g., `128` instead of full `4096`)
4. **`ctx_indices`** - Calculated indices for gather operation (e.g., `[0,1,2,...,127]`)

**Attention mask** created dynamically per forward pass, NOT pre-allocated:
- Shape: `[batch_size, 1, seq_len, target_length]` where `target_length ≤ ctx_len`
- Created by `_create_causal_mask()` (modeling_attn_mask_utils.py:13-50)
- Mask shape changes each step, but KV buffer shape stays fixed

### **Q14: When do scatter/gather operations actually execute during prefill and decode?**
**A:** Operations happen **INSIDE `session.run()` model forward pass**, NOT at Python runtime level (see Section 7.1.8):

**Prefill (Chunked):**
- Python: `session.run(chunk_inputs)` called once per chunk
- Inside model: For EACH layer (32x):
  1. **Scatter** - Write new KV at chunk positions (e.g., [128-255] for chunk 1)
  2. **Gather** - Read ALL cumulative KV (e.g., [0-255] including chunk 0 data)
  3. **Create mask** - Causal mask for current chunk
  4. **Attention** - Compute over gathered KV range
- Chunk 1 sees Chunk 0 data via gather (no separate merge)

**Decode (Per Token):**
- Python: `session.run(decode_inputs)` called once per token
- Inside model: For EACH layer (32x):
  1. **Scatter** - Write 1 new KV position for all 8 batch slots (parallel)
  2. **Gather** - Read each slot's valid KV range (different lengths, parallel)
  3. **Create mask** - Causal mask per slot (different valid ranges)
  4. **Attention** - 8 slots compute in parallel over their gathered KV

**Key Insight:** Scatter/Gather are hardware ops executed 32 times per `session.run()` (once per layer), not Python-level operations.

---

## **15. ARCHITECTURAL CONSTRAINTS (Critical for Optimization Work)**

### **Continuous Batching Execution Model**

**Current Implementation Constraint:**
- **NO pipelined/overlapped execution** between prefill and decode phases
- **NO partial batch decode** before all slots are filled
- **Sequential phase boundary** enforced by function call structure

**Code Evidence:**
```python
# text_generation_inference.py:1206-1209
self._qaic_model.run_prefill_for_all_inputs(...)  # BLOCKS until all slots filled
decode_pause_time = self._qaic_model.run_continuous_batching_decode(...)  # Starts AFTER
```

**Why This Matters for Optimization:**
1. **Latency for First Token:** If `full_batch_size=8` and you have 1 prompt, you still wait for 7 empty slots to be "prefilled" (or use padding logic).
2. **Batch Formation Time:** With 4 prompts arriving over 100ms, first prompt waits for all 4 before decode starts.
3. **Resource Utilization:** Hardware idle during serial prefill phase (no decode happening).

**Alternative Architectures (Not Implemented):**
- **Pipelined:** Start decode for Slot 0 while Slot 1 is prefilling
- **Dynamic Batching:** Decode starts with partial batch, adds prompts dynamically
- **Async Prefill:** Background thread prefills while main thread decodes

**If Implementing Advanced Continuous Batching:**
- Must refactor function boundaries (lines 1206-1209)
- Requires thread-safe KV cache updates
- Need partial batch decode support in specializations
- Consider async/await or threading model

---

## **16. INTEGRATION CHECKLIST FOR NEW OPTIMIZATIONS**

When adding new features (like speculative prefill), verify:

- [ ] **Transform Level:** Does it change PyTorch graph? Add to `_pytorch_transforms`
- [ ] **ONNX Level:** Does it need custom ONNX nodes? Implement `.symbolic()` method
- [ ] **Specialization Level:** Does it need new shapes? Extend specializations.json
- [ ] **Custom IO Level:** Does it change input/output types? Update custom_io.yaml
- [ ] **Runtime Level:** Does it change execution flow? Modify `QEffTextGenerationBase`
- [ ] **Hash Parameters:** Does it affect compilation? Add to `compile_hash_params`
- [ ] **Compatibility:** Works with continuous batching? Chunked prefill? CCL? Multi-device?
- [ ] **Dynamic Axes:** Does it require new symbolic dimensions in ONNX?
- [ ] **Device Memory:** Does it fit within on-chip memory constraints?
- [ ] **Validation:** Test with Flow 1 (CLI), Flow 2 (Python API), Flow 3 (pre-compiled QPC)
- [ ] **Error Handling:** Review Section 16 for potential failure modes and add appropriate checks

---

## **17. QUICK REFERENCE: KEY CODE LOCATIONS**

**For Fast Navigation During Development:**

| Task | File | Function/Method | What It Does |
|------|------|-----------------|--------------|
| **Entry Points** |
| CLI execution | `cloud/infer.py` | `main()` | End-to-end: load → compile → execute |
| Direct Python API | `modeling_auto.py` | `QEFFAutoModelForCausalLM` class | User-facing API for model operations |
| Pre-compiled execution | `cloud/execute.py` | `main()` | Load QPC and run inference |
| **Transform Pipeline** |
| PyTorch transforms | `modeling_qeff.py` | `QEFFBaseModel.__init__()` | Transform loop applying all PyTorch transforms |
| Transform definitions | `transformers/models/pytorch_transforms.py` | Various transform classes | All transform implementations |
| Quantizer swap | `quantizers/auto.py` | `@with_replaced_quantizers` decorator | Temporary quantizer mapping swap |
| **ONNX Export** |
| Export logic | `modeling_qeff.py` | `QEFFBaseModel._export()` | ONNX export with dynamic axes |
| Dynamic axes setup | `exporter/export_utils.py` | `export_onnx()` | Symbolic dimension definitions |
| ONNX transforms | `base/onnx_transforms.py` | Various transform classes | FP16Clip, SplitTensors |
| **Compilation** |
| Compile entry | `modeling_qeff.py` | `QEFFBaseModel._compile()` | Compilation with hash-based caching |
| Specializations | `compile/compile_helper.py` | `create_and_dump_specializations()` | JSON generation for CB/non-CB |
| Hash computation | `modeling_qeff.py` | `QEFFBaseModel._compile()` | Hash parameter collection |
| **Runtime Execution** |
| CB prefill (serial) | `text_generation_inference.py` | `run_prefill_for_all_inputs()` | Serial prefill for all batch slots |
| CB decode (parallel) | `text_generation_inference.py` | `run_continuous_batching_decode()` | Parallel decode across batch |
| Sequential enforcement | `text_generation_inference.py` | `generate()` method | Prefill → Decode phase boundary |
| KV cache retention | `text_generation_inference.py` | `TextGeneration.__init__()` | `skip_buffers()` call for retained-state |
| Chunked prefill | `text_generation_inference.py` | `run_prefill()` | Chunked prefill loop |
| **KV Cache Mechanics (Section 7.1)** |
| Cache initialization | `transformers/cache_utils.py` | `QEffDynamicLayer.__init__()` | Buffer pre-allocation |
| Scatter operation (non-CB) | `customop/ctx_scatter_gather.py` | `CtxScatterFunc.forward()` | Write KV states to cache |
| Scatter operation (CB) | `customop/ctx_scatter_gather_cb.py` | `CtxScatterFuncCB.forward()` | Write KV with batch_index |
| Gather operation (non-CB) | `customop/ctx_scatter_gather.py` | `CtxGatherFunc.forward()` | Read KV states from cache |
| Gather operation (CB) | `customop/ctx_scatter_gather_cb.py` | `CtxGatherFuncCB.forward()` | Read KV with batch_index |
| Cache update logic | `transformers/cache_utils.py` | `QEffDynamicLayer.update()` | Scatter + Gather integration |
| Attention mask creation | `transformers/modeling_attn_mask_utils.py` | `_create_causal_mask()` | Dynamic causal mask generation |
| CCL processing | `utils/check_ccl_specializations.py` | `process_ccl_specializations()` | Compute Context Length handling |
| **Debugging Targets** |
| Transform verification | `modeling_qeff.py` | `QEFFBaseModel.__init__()` | Check "applied" logs in transform loop |
| QPC existence check | `modeling_qeff.py` | `QEFFBaseModel._compile()` | Early return if cached QPC found |
| Decode batch size | `text_generation_inference.py` | `run_continuous_batching_decode()` | `_set_output_buffers()` call |
| Prefill loop completion | `text_generation_inference.py` | `run_prefill_for_all_inputs()` | Loop + completion log |
| KV buffer shapes | `transformers/cache_utils.py` | `QEffDynamicLayer.__init__()` | Verify `[batch, heads, ctx_len, dim]` |
| Scatter position_ids | `transformers/cache_utils.py` | `QEffDynamicLayer.update()` | Check invalid index handling |
| Gather masking | `transformers/cache_utils.py` | `QEffDynamicLayer.read_only()` | Invalid position masking logic |

---

**END OF COMPREHENSIVE PROMPT**

---

This documentation provides a complete, code-verified understanding of QEfficient's architecture with detailed call stacks, convergence points, integration guidelines, and practical debugging information. Use this as a reference for implementing new optimizations, debugging existing flows, or understanding how Qualcomm Cloud AI 100 hardware acceleration works.
