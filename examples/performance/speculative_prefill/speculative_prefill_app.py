from QEfficient import QEFFAutoModelForCausalLM
from transformers import AutoTokenizer
from pathlib import Path
import onnx
from QEfficient.generation.text_generation_inference import get_compilation_dims

spec_model = QEFFAutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    qaic_config={"enable_speculative_prefill": True}
)

export_dir = Path("export_spec_prefill")
onnx_path = spec_model.export(export_dir=export_dir)
print("Exported ONNX:", onnx_path)
m = onnx.load(str(onnx_path))
print("ONNX outputs:", [o.name for o in m.graph.output])

spec_model.compile(
    onnx_path=str(onnx_path),
    prefill_seq_len=128, 
    ctx_len=4096, 
    batch_size=1, 
    num_devices=4, 
    num_cores=16, 
    mxfp6_matmul=True, 
    mxint8_kv_cache=True, # Allows MXINT8 compression of MDP IO traffic
    allow_mxint8_mdp_io=True, 
    split_retained_state_io=True, 
    aic_enable_depth_first=True)


base_model = QEFFAutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
# base_model = QEFFAutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", continuous_batching=True)

export_dir = Path("export_base_prefill")
onnx_path = base_model.export(export_dir=export_dir)
print("Exported ONNX:", onnx_path)
m = onnx.load(str(onnx_path))
print("ONNX outputs:", [o.name for o in m.graph.output])

base_model.compile(
    onnx_path=str(onnx_path),
    prefill_seq_len=128, 
    ctx_len=4096, 
    batch_size=1, 
    # full_batch_size=2,
    num_devices=4, 
    num_cores=16, 
    mxfp6_matmul=True, 
    mxint8_kv_cache=True, 
    allow_mxint8_mdp_io=True, 
    aic_enable_depth_first=True)

# Verify continuous batching compile (FBS should be > 1)
_, _, fbs = get_compilation_dims(str(base_model.qpc_path))
print("Base full_batch_size (FBS):", fbs)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

prompt_text = Path("/local/mnt/p_drive/users/rpodila/dl_inference/speculative_prefill_gpu/speculative_prefill_demo/prompts/small_prompt.txt").read_text(encoding="utf-8")
# # ✅ Simple, clean, user-friendly
result = spec_model.generate_speculative_prefill(
    base_model=base_model,
    tokenizer=tokenizer,
    prompts=prompt_text,
    device_id=[8,9,10,11],
    base_device_id=[12,13,14,15],
    keep_percentage=0.20,
)

print(result["generated_text_pruned"])
