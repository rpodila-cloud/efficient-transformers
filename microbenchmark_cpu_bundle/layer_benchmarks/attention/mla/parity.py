"""CPU parity checks for the MLA blocking paths against decomposed ORT."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from layer_benchmarks.common import custom_ops, parity

from .bench import export_onnx, inputs
from .forwards import Config, build

def _small() -> Config:
    return Config(hidden_size=64, num_attention_heads=8, q_lora_rank=32, qk_rope_head_dim=8,
                  kv_lora_rank=16, v_head_dim=8, qk_nope_head_dim=8, ctx_len=64,
                  num_kv_heads=2, num_kv_blocks=4, par_num_split=4)


@torch.no_grad()
def _vs_reference(c: Config) -> dict:
    from .ref import MLAReference

    torch.manual_seed(1234)
    ours = build(c)
    torch.manual_seed(1234)
    reference = MLAReference(c).eval()
    reference.load_state_dict({k: v for k, v in ours.state_dict().items() if k in reference.state_dict()}, strict=False)
    args = inputs(c, 1, 1, c.ctx_len - 2)
    got = ours(*args)
    want = reference(*args)
    checks = [parity.compare(name, a, b, atol=2e-3, rtol=2e-3)
              for name, a, b in zip(("attn_output", "compressed_kv", "k_pe"), got, want)]
    return {"name": f"vs_reference_{c.mode}_abs{int(c.mla_absorption)}_online{int(c.mla_online)}",
            "passed": all(x["passed"] for x in checks), "checks": checks,
            "max_abs": max(x["max_abs"] for x in checks)}


@torch.no_grad()
def _vs_ort(c: Config, out_dir: Path) -> dict:
    torch.manual_seed(1234)
    model = build(c)
    args = inputs(c, 1, 1, c.ctx_len - 2)
    with custom_ops.enabled(False):
        path = export_onnx(model, args, out_dir / "mla_small_decomposed.onnx", custom_ops=False)
    names = ("hidden_states", "position_ids", "compressed_kv.0", "k_pe.0")
    got = parity.ort_run(path, {n: t.numpy() for n, t in zip(names, args)},
                         ["attn_output", "compressed_kv.0_RetainedState", "k_pe.0_RetainedState"])
    want = model(*args)
    expected = {"attn_output": want[0], "compressed_kv.0_RetainedState": want[1], "k_pe.0_RetainedState": want[2]}
    checks = [parity.compare(n, got[n], expected[n], atol=1e-4, rtol=1e-4) for n in expected]
    return {"name": "vs_ort", "passed": all(x["passed"] for x in checks), "checks": checks,
            "max_abs": max(x.get("max_abs", 0.0) for x in checks),
            "mean_abs": sum(x.get("mean_abs", 0.0) for x in checks) / len(checks)}


def self_test(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    c = _small()
    checks = [_vs_reference(replace(c, mode=mode, mla_absorption=absorption, mla_online=online))
              for absorption in (True, False)
              for online in (False, True)
              for mode in ("none", "kv", "h", "par", "prefill_par", "prefill_par_online")]
    checks += [_vs_ort(replace(c, mode=mode, mla_absorption=absorption, mla_online=online), out_dir)
               for absorption in (True, False)
               for online in (False, True)
               for mode in ("none", "kv", "h", "par", "prefill_par", "prefill_par_online")]
    result = {"passed": all(x["passed"] for x in checks), "checks": checks}
    parity.require(result, "mla self-test")
    return result
