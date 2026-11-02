"""Qwen3.5 linear-attention parity loops.

* ``self_test`` – the small chunk/recurrent cases, eager vs ORT on the export
  with the CustomRMSNorm op decomposed.
* ``hf_parity`` – HF ``Qwen3_5MoeGatedDeltaNet`` prefill (torch chunk kernel)
  vs our chunk path and vs our recurrent path stepped token by token from
  zero state; also chunk final state vs recurrent final state.
"""

from __future__ import annotations

from pathlib import Path

import torch

from layer_benchmarks.common import custom_ops, hf, parity

from .bench import build, export_onnx, inputs, io_names
from .forwards import Config

SMOKE = dict(hidden_size=256, num_k_heads=2, num_v_heads=4, head_k_dim=32, head_v_dim=32)
# qwen3_5_linear_attn_microbench_README.md "Smoke tests".
CASES = (
    dict(name="chunk_seq64", path="chunk", seq_len=64, pos=0, **SMOKE),
    dict(name="chunk_seq80", path="chunk", seq_len=80, pos=0, **SMOKE),
    dict(name="chunk_seq128", path="chunk", seq_len=128, pos=0, **SMOKE),
    dict(name="recurrent_hv_eq_hk", path="recurrent", seq_len=1, pos=0,
         hidden_size=128, num_k_heads=4, num_v_heads=4, head_k_dim=32, head_v_dim=32),
    dict(name="recurrent_pos40", path="recurrent", seq_len=1, pos=40, **SMOKE),
    dict(name="recurrent_hb2_channel_last", path="recurrent", seq_len=1, pos=40,
         conv_state_layout="channel_last", head_block_size=2, **SMOKE),
)
DIMS = ("hidden_size", "num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim")


def _config(case) -> Config:
    return Config(
        path=case["path"],
        **{k: case[k] for k in DIMS},
        **{k: case[k] for k in ("conv_state_layout", "head_block_size") if k in case},
    )


@torch.no_grad()
def _vs_ort(case, out_dir: Path) -> dict:
    c = _config(case)
    model = build(c)
    args = inputs(c, 1, case["seq_len"], case["pos"])
    with custom_ops.enabled(False):
        path = export_onnx(model, args, out_dir / f"{case['name']}.onnx", custom_ops=False)
    input_names, output_names = io_names(c.path)
    got = parity.ort_run(path, {n: t.numpy() for n, t in zip(input_names, args)}, output_names)
    want = model(*args)
    checks = [parity.compare(n, got[n], e, atol=1e-4, rtol=1e-4) for n, e in zip(output_names, want)]
    worst = max(checks, key=lambda r: r["max_abs"])
    return {"name": f"{case['name']}_vs_ort", "max_abs": worst["max_abs"],
            "mean_abs": sum(r["mean_abs"] for r in checks) / len(checks), "worst": worst["name"],
            "passed": all(r["passed"] for r in checks)}


def self_test(out_dir: Path) -> dict:
    torch.set_num_threads(1)
    checks = []
    for case in CASES:
        checks.append(_vs_ort(case, out_dir))
    result = {"passed": all(r["passed"] for r in checks), "checks": checks}
    parity.require(result, "linear self-test")
    return result


# ── HF ────────────────────────────────────────────────────────────────────────
# Qwen3.5-sized heads; tokens cross a chunk boundary so the padding path runs.
HF_CASES = (
    dict(name="qwen3_5_heads", tokens=80, hidden_size=512, num_k_heads=4, num_v_heads=8, head_k_dim=128, head_v_dim=128),
    dict(name="hv_eq_hk", tokens=64, hidden_size=256, num_k_heads=4, num_v_heads=4, head_k_dim=32, head_v_dim=32),
)


def _hf_layer(c: Config, cfg_mod, mdl_mod, ours):
    config = cfg_mod.Qwen3_5MoeTextConfig(
        hidden_size=c.hidden_size, linear_num_key_heads=c.num_k_heads, linear_num_value_heads=c.num_v_heads,
        linear_key_head_dim=c.head_k_dim, linear_value_head_dim=c.head_v_dim, linear_conv_kernel_dim=c.conv_kernel,
        rms_norm_eps=c.rms_norm_eps, num_hidden_layers=1, layer_types=["linear_attention"],
    )
    ref = mdl_mod.Qwen3_5MoeGatedDeltaNet(config, layer_idx=0).eval().float()
    sd = ours.state_dict()
    for name, p in ref.state_dict().items():
        p.copy_(sd[name])  # same parameter names; HF conv1d has no bias
    return ref


@torch.no_grad()
def _hf_case(case, cfg_mod, mdl_mod) -> dict:
    tokens = case["tokens"]
    chunk = _config(dict(case, path="chunk"))
    ours = build(chunk)
    g = torch.Generator().manual_seed(7)
    # HF has no conv bias: zero ours (and the split copies); randomise the norm so it is exercised.
    for name in ("conv1d.bias", "q_conv1d.bias", "k_conv1d.bias", "v_conv1d.bias"):
        ours.state_dict()[name].zero_()
    ours.norm.weight.copy_(1 + 0.1 * torch.randn(ours.norm.weight.shape, generator=g))
    ref = _hf_layer(chunk, cfg_mod, mdl_mod, ours)

    x = torch.randn(2, tokens, chunk.hidden_size, generator=g)
    position_ids = torch.arange(tokens, dtype=torch.int32).unsqueeze(0).expand(2, -1)
    expected = ref(x)
    out_chunk, state_chunk = ours(x, position_ids)

    recurrent = Config(
        **{
            **chunk.__dict__,
            "path": "recurrent",
            "conv_state_layout": "channel_last",
            "head_block_size": chunk.num_v_heads // 2,
        }
    )
    ours.c = recurrent
    conv_state = torch.zeros(2, chunk.conv_kernel, chunk.conv_dim)
    state = torch.zeros(2, chunk.num_v_heads, chunk.head_k_dim, chunk.head_v_dim)
    steps = []
    for t in range(tokens):
        out, conv_state, state = ours(x[:, t : t + 1], position_ids[:, t : t + 1], conv_state, state)
        steps.append(out)
    checks = [
        parity.compare(f"hf_{case['name']}_chunk", out_chunk, expected, atol=1e-4, rtol=1e-4),
        parity.compare(f"hf_{case['name']}_recurrent", torch.cat(steps, 1), expected, atol=1e-4, rtol=1e-4),
        parity.compare(f"{case['name']}_state_chunk_vs_recurrent", state_chunk, state, atol=1e-4, rtol=1e-4),
    ]
    return {"name": f"hf_{case['name']}", "tokens": tokens, "passed": all(r["passed"] for r in checks),
            "max_abs": max(r["max_abs"] for r in checks), "checks": checks}


def hf_parity() -> dict:
    torch.set_num_threads(1)
    cfg_mod = hf.import_hf("qwen3_5_moe.configuration_qwen3_5_moe")
    mdl_mod = hf.import_hf("qwen3_5_moe.modeling_qwen3_5_moe")
    checks = [r for case in HF_CASES for r in _hf_case(case, cfg_mod, mdl_mod)["checks"]]
    result = {"passed": all(r["passed"] for r in checks), "checks": checks}
    parity.require(result, "linear hf")
    return result
