"""DeepSeek-V4 CSA/HCA parity loops.

* ``self_test`` – small multi-step decode cases with retained state fed back,
  checked against the decomposed ORT export (every output).
* ``hf_parity`` – fp32 decode loop vs HF ``DeepseekV4Attention`` prefill over
  the same tokens (sliding window + compressor + indexer top-k), weights
  copied from our module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from layer_benchmarks.common import custom_ops, hf, parity

from .forwards import CompressedAttention

# deepseek_v4_ref.PARITY_CASES (smoke profile).
CASES = (
    dict(name="csa_topk_smoke", mode="csa", ctx_len=64, ratio=4, topk=4, positions=(2, 3, 10, 11), cores=16),
    dict(name="csa_full_range_smoke", mode="csa", ctx_len=64, ratio=4, topk=16, positions=(6, 7), cores=16),
    dict(name="hca_ai100_smoke", mode="hca", ctx_len=2048, ratio=128, topk=4, positions=(254, 255), cores=16),
    dict(name="hca_ai200_smoke", mode="hca", ctx_len=512, ratio=128, topk=4, positions=(254, 255), cores=4),
    dict(name="hca_cp2_blk4_smoke", mode="hca", ctx_len=2048, ratio=128, topk=4, positions=(126, 127, 254, 255, 382, 383),
         cores=16, hca_cp=2, hca_blocks=4),
    dict(name="hca_cp4_blk2_smoke", mode="hca", ctx_len=2048, ratio=128, topk=4, positions=(126, 127, 510, 511),
         cores=4, hca_cp=4, hca_blocks=2),
)
BATCH, DP, CP, BLOCKS = 2, 2, 2, 2


def _ort_session(path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


@torch.no_grad()
def _vs_ort(case, out_dir: Path) -> dict:
    from .bench import build, export_onnx, inputs, io_names, make_config

    c = make_config("smoke", case["mode"], case["ctx_len"], case["ratio"], index_topk=case["topk"])
    model = build(c, torch.float32, attn_dp=DP, indexer_cp=CP, num_kv_blocks=BLOCKS, cores=case["cores"],
                  hca_compressed_kv_cp=case.get("hca_cp", 1), hca_attn_blocks=case.get("hca_blocks"))
    args = inputs(c, BATCH, case["positions"][0], torch.float32, attn_dp=DP, indexer_cp=CP)
    with custom_ops.enabled(False):
        path = export_onnx(model, args, out_dir / f"{case['name']}.onnx", custom_ops=False)
    session = _ort_session(path)
    input_names, output_names = io_names(c.attn_mode)
    hidden, state = args[0], list(args[2:])
    ort_state = [s.numpy() for s in state]
    checks = []
    for pos in case["positions"]:
        position_ids = torch.full((BATCH, 1), pos, dtype=torch.int64)
        want = model(hidden, position_ids, *state)
        got = session.run(None, dict(zip(input_names, [hidden.numpy(), position_ids.numpy(), *ort_state])))
        checks += [parity.compare(f"{case['name']}@{pos}:{n}", torch.from_numpy(a), e, atol=1e-4, rtol=1e-4)
                   for n, a, e in zip(output_names, got, want)]
        state, ort_state = list(want[1:]), got[1:]
    worst = max(checks, key=lambda r: r["max_abs"])
    return {"name": f"{case['name']}_vs_ort", "max_abs": worst["max_abs"],
            "mean_abs": sum(r["mean_abs"] for r in checks) / len(checks), "worst": worst["name"],
            "passed": all(r["passed"] for r in checks)}


def self_test(out_dir: Path, mode: str | None = None) -> dict:
    torch.set_num_threads(1)
    checks = []
    selected = [case for case in CASES if mode is None or case["mode"] == mode]
    for case in selected:
        checks.append(_vs_ort(case, out_dir))
    result = {"passed": all(r["passed"] for r in checks), "checks": checks}
    parity.require(result, "compressed attention self-test")
    return result


# ── HF ────────────────────────────────────────────────────────────────────────
# index_n_heads is raised so ReLU(q·k) is rarely zero on every head: exact-zero ties
# make top-k selection order-dependent and HF/ours could pick different blocks.
HF_CASES = (
    dict(name="csa_topk", mode="csa", ratio=4, topk=4, tokens=64, cores=16, index_n_heads=16),
    dict(name="csa_all", mode="csa", ratio=4, topk=64, tokens=40, cores=16, index_n_heads=16),
    dict(name="hca", mode="hca", ratio=16, topk=4, tokens=128, cores=4, index_n_heads=4),
)
HF_LAYER = {"csa": "compressed_sparse_attention", "hca": "heavily_compressed_attention"}


def _hf_attention(c, cfg_mod, mdl_mod):
    config = cfg_mod.DeepseekV4Config(
        hidden_size=c.hidden_size, num_attention_heads=c.num_heads, head_dim=c.head_dim, q_lora_rank=c.q_lora_rank,
        o_groups=c.o_groups, o_lora_rank=c.o_lora_rank, sliding_window=c.window_size, index_n_heads=c.index_n_heads,
        index_head_dim=c.index_head_dim, index_topk=c.index_topk, num_hidden_layers=1,
        layer_types=[HF_LAYER[c.attn_mode]], compress_rates={HF_LAYER["csa"]: c.compress_ratio, HF_LAYER["hca"]: c.compress_ratio},
        rope_theta=c.rope_theta, compress_rope_theta=c.compress_rope_theta, rms_norm_eps=c.norm_eps,
        partial_rotary_factor=c.rope_head_dim / c.head_dim, max_position_embeddings=c.ctx_len,
        rope_parameters={"rope_type": "yarn", "factor": c.yarn_factor, "beta_fast": c.yarn_beta_fast,
                         "beta_slow": c.yarn_beta_slow,
                         "original_max_position_embeddings": c.yarn_original_max_position_embeddings},
    )
    config._attn_implementation = "eager"
    return mdl_mod.DeepseekV4Attention(config, layer_idx=0).eval().float(), mdl_mod.DeepseekV4RotaryEmbedding(config)


def _copy_weights(ref, ours: CompressedAttention, c):
    pairs = {
        "q_a_proj.weight": "q_a_proj.weight", "q_a_norm.weight": "q_a_norm.weight", "q_b_proj.weight": "q_b_proj.weight",
        "kv_proj.weight": "kv_proj.weight", "kv_norm.weight": "kv_norm.weight", "o_b_proj.weight": "o_b_proj.weight",
        "sinks": "sinks", "compressor.kv_proj.weight": "comp_kv_proj.weight",
        "compressor.gate_proj.weight": "comp_gate_proj.weight", "compressor.position_bias": "comp_position_bias",
        "compressor.kv_norm.weight": "comp_norm.weight",
    }
    if c.attn_mode == "csa":
        pairs.update({
            "compressor.indexer.kv_proj.weight": "index_kv_proj.weight",
            "compressor.indexer.gate_proj.weight": "index_gate_proj.weight",
            "compressor.indexer.position_bias": "index_position_bias",
            "compressor.indexer.kv_norm.weight": "index_norm.weight",
            "compressor.indexer.q_b_proj.weight": "index_q_b_proj.weight",
            "compressor.indexer.scorer.weights_proj.weight": "index_weights_proj.weight",
        })
    ours_sd, ref_sd = ours.state_dict(), ref.state_dict()
    for hf_name, name in pairs.items():
        ref_sd[hf_name].copy_(ours_sd[name])
    ref_sd["o_a_proj.weight"].copy_(ours.o_a_weight.reshape(-1, ours.o_a_weight.shape[-1]))


@torch.no_grad()
def _hf_case(case, cfg_mod, mdl_mod) -> dict:
    from .bench import build, inputs, make_config

    tokens = case["tokens"]
    c = make_config("smoke", case["mode"], tokens, case["ratio"], index_topk=case["topk"],
                    index_n_heads=case["index_n_heads"])
    ours = build(c, torch.float32, attn_dp=DP, indexer_cp=CP, num_kv_blocks=BLOCKS, cores=case["cores"])
    # The original init leaves sinks / position biases at zero; randomise so they are exercised.
    g = torch.Generator().manual_seed(7)
    for name in ("sinks", "comp_position_bias", "index_position_bias", "q_a_norm.weight", "kv_norm.weight"):
        if name in ours.state_dict():
            p = ours.state_dict()[name]
            p.copy_(torch.randn(p.shape, generator=g) * (0.5 if "norm" not in name else 0.1) + ("norm" in name))
    ref, rope = _hf_attention(c, cfg_mod, mdl_mod)
    _copy_weights(ref, ours, c)
    yarn_error = float((rope.compress_inv_freq - ours.compress_inv_freq).abs().max())

    x = torch.randn(BATCH, tokens, c.hidden_size, generator=g)
    position_ids = torch.arange(tokens).view(1, -1).expand(BATCH, -1)
    embeddings = {k: rope(x, position_ids, layer_type=k) for k in ("main", "compress")}
    q_idx = torch.arange(tokens).view(-1, 1)
    k_idx = torch.arange(tokens).view(1, -1)
    visible = (k_idx <= q_idx) & (k_idx > q_idx - c.window_size)
    mask = torch.zeros(BATCH, 1, tokens, tokens).masked_fill(~visible, float("-inf"))
    expected, _ = ref(x, embeddings, position_ids, mask)

    state = list(inputs(c, BATCH, 0, torch.float32, attn_dp=DP, indexer_cp=CP)[2:])
    steps = []
    for pos in range(tokens):
        out = ours(x[:, pos : pos + 1], torch.full((BATCH, 1), pos, dtype=torch.int64), *state)
        steps.append(out[0])
        state = list(out[1:])
    result = parity.compare(f"hf_{case['name']}", torch.cat(steps, 1), expected, atol=5e-5, rtol=5e-5)
    result.update(tokens=tokens, compress_ratio=c.compress_ratio, index_topk=c.index_topk, yarn_inv_freq_error=yarn_error)
    return result


def hf_parity(mode: str | None = None) -> dict:
    torch.set_num_threads(1)
    cfg_mod = hf.import_hf("deepseek_v4.configuration_deepseek_v4")
    mdl_mod = hf.import_hf("deepseek_v4.modeling_deepseek_v4")
    selected = [case for case in HF_CASES if mode is None or case["mode"] == mode]
    checks = [_hf_case(case, cfg_mod, mdl_mod) for case in selected]
    result = {"passed": all(r["passed"] for r in checks), "checks": checks}
    parity.require(result, "compressed attention hf")
    return result
