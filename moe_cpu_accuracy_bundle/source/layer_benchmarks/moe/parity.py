"""MoE parity loops.

* ``self_test`` – per impl, small shapes: eager/reference vs ORT on the export with custom ops
  decomposed (fp32). The attached ``CtxScatter3D*_v18`` bodies cannot run on ORT: their ScatterND
  sees the INT32_MAX padding rows, which QAIC treats as no-ops. The custom-op export is still
  built, and its histogram is checked.
* ``hf_parity`` – fp32 vs HF ``GptOssExperts`` (decode, alpha/limit set to ours) and
  ``Qwen3VLMoeTextSparseMoeBlock`` (routed and prefill paths), weights copied from our module.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch

from layer_benchmarks.common import export, hf, parity

from . import bench
from .forwards import (
    CumsumScatterGatherUpdate,
    CumsumScatterGatherUpdateWithRouter,
    GatherBmmDecode,
    GatherBmmLoopDecode,
    Reference,
    RoutedCumsumScatterGatherUpdate,
)

CASES = (
    dict(impls=bench.DECODE, batch_size=4, T=1, H=32, I=48, num_nsp=16, local_experts=1, num_experts_per_tok=3, shared_experts=0),
    dict(impls=bench.DECODE, batch_size=4, T=1, H=32, I=48, num_nsp=16, local_experts=1, num_experts_per_tok=3, shared_experts=2),
    dict(impls=bench.ROUTED, batch_size=8, T=2, H=32, I=48, num_nsp=8, local_experts=1, num_experts_per_tok=2, packed_chunk_size=4,
         num_devices=2),
    dict(impls=bench.ROUTED, batch_size=4, T=3, H=32, I=48, num_nsp=4, local_experts=2, num_experts_per_tok=3, packed_chunk_size=6),
    dict(impls=bench.PREFILL, batch_size=1, T=48, H=32, I=48, num_nsp=4, local_experts=2, num_experts_per_tok=4, packed_chunk_size=16),
    dict(impls=bench.PREFILL, batch_size=1, T=48, H=32, I=48, num_nsp=4, local_experts=2, num_experts_per_tok=4, packed_chunk_size=16,
         ffn_blocking_mode="token", ffn_token_block_size=5),
    dict(impls=bench.DECODE, batch_size=4, T=1, H=32, I=48, num_nsp=16, local_experts=1, num_experts_per_tok=3, shared_experts=1,
         ffn_blocking_mode="token", ffn_token_block_size=1),
    dict(impls=bench.DECODE_LOOP, batch_size=4, T=1, H=32, I=48, num_nsp=16, local_experts=1, num_experts_per_tok=3),
    dict(impls=bench.REFERENCE, batch_size=1, T=8, H=32, I=48, num_nsp=4, local_experts=2, num_experts_per_tok=2),
    dict(impls=bench.PREFILL_ROUTED, batch_size=2, T=8, H=32, I=48, num_nsp=4, local_experts=2, num_experts_per_tok=2, packed_chunk_size=4),
)
DEFAULTS = dict(shared_experts=0, packed_chunk_size=256, num_devices=1, experts_per_soc=None, seed=1234,
                ffn_blocking_mode="default", ffn_token_block_size=None)


def _args(case, dtype):
    return Namespace(**{**DEFAULTS, **case, "dtype": dtype})


def self_test(artifact_dir: Path) -> None:
    for case in CASES:
        name = f"{case['impls']}_N{case['num_nsp']}L{case['local_experts']}_S{case.get('shared_experts', 0)}_tb{case.get('ffn_token_block_size')}"
        for dtype in ("float32", "float16"):
            a = _args(case, dtype)
            ours, feed, dims, _ = bench.build(a)
            with torch.no_grad():
                out = ours(*feed.values())
            if dtype != "float32":
                continue
            for custom in (False, True):
                path = export.export(ours, tuple(feed.values()), artifact_dir / f"{name}_custom{int(custom)}.onnx",
                                     list(feed), ["O"], dim_params=dims, optimize=False, use_custom_ops=custom)
                hist = export.op_histogram(path)
                bad = {k: v for k, v in hist.items() if k in ("Sign", "NonZero")}
                if custom:
                    if bad:
                        raise SystemExit(f"{name}: AIC-hostile ops in export: {bad}")
                    print(f"  export ops ({name}): {hist}")
                    continue
                ort = parity.ort_run(path, {k: v.numpy() for k, v in feed.items()})["O"]
                parity.require(parity.compare("ORT O", ort, out, atol=1e-5, rtol=1e-5), f"{name} ORT (custom ops decomposed)")


def _hf_config(module_name, cls, **kw):
    cfg = getattr(hf.import_hf(module_name), cls)(**kw)
    cfg._experts_implementation = "eager"
    return cfg


def hf_parity() -> None:
    torch.manual_seed(0)
    gpt = hf.import_hf("gpt_oss.modeling_gpt_oss")
    qwen = hf.import_hf("qwen3_vl_moe.modeling_qwen3_vl_moe")
    H, I, N = 32, 48, 8

    # decode_gather_bmm ≡ GptOssExperts (interleaved gate/up, biases) with alpha=1, limit=127.
    m = GatherBmmDecode(N, 1, H, I, torch.float32, num_experts_per_tok=3)
    with torch.no_grad():
        for p in (m.b_g, m.b_u, m.b_d):
            p.normal_(0, 0.02)
    cfg = _hf_config("gpt_oss.configuration_gpt_oss", "GptOssConfig", hidden_size=H, intermediate_size=I,
                     num_local_experts=N, num_experts_per_tok=3)
    ref = gpt.GptOssExperts(cfg)
    ref.alpha, ref.limit = 1.0, 127.0
    with torch.no_grad():
        gu = torch.stack((m.W_g, m.W_u), dim=-1).reshape(N, H, 2 * I)
        ref.gate_up_proj.copy_(gu)
        ref.gate_up_proj_bias.copy_(torch.stack((m.b_g, m.b_u), dim=-1).reshape(N, 2 * I))
        ref.down_proj.copy_(m.W_d)
        ref.down_proj_bias.copy_(m.b_d)
        x = torch.randn(6, H)
        ri, rw = torch.stack([torch.randperm(N)[:3] for _ in range(6)]).int(), torch.softmax(torch.randn(6, 3), -1)
        res = parity.compare("decode_gather_bmm vs GptOssExperts", m(x, ri, rw), ref(x, ri.long(), rw), atol=1e-5, rtol=1e-5)
    parity.require(res, "HF gpt_oss")

    # Routed / prefill ≡ Qwen3VLMoeTextSparseMoeBlock (router + SwiGLU experts); e = le*N + n.
    for L, chunk, (B, S) in ((1, 4, (8, 1)), (2, 6, (2, 6))):
        E = N * L
        m = RoutedCumsumScatterGatherUpdate(N, L, H, I, torch.float32, chunk, num_experts_per_tok=2)
        cfg = _hf_config("qwen3_vl_moe.configuration_qwen3_vl_moe", "Qwen3VLMoeTextConfig", hidden_size=H,
                         moe_intermediate_size=I, num_experts=E, num_experts_per_tok=2, hidden_act="silu")
        ref = qwen.Qwen3VLMoeTextSparseMoeBlock(cfg)
        flat = lambda w: w.transpose(0, 1).reshape(E, *w.shape[2:])  # noqa: E731  [N, L, ...] → [E, ...]
        with torch.no_grad():
            ref.gate.weight.copy_(m.gate.weight)
            ref.experts.gate_up_proj.copy_(torch.cat((flat(m.W_g), flat(m.W_u)), dim=-1).transpose(1, 2))
            ref.experts.down_proj.copy_(flat(m.W_d).transpose(1, 2))
            x = torch.randn(B, S, H)
            routed = parity.compare(f"routed L={L} vs Qwen3VLMoe block", m(x), ref(x).reshape(B * S, H), atol=1e-5, rtol=1e-5)
            # Prefill: same experts, mask from the HF router, no router weights.
            p = CumsumScatterGatherUpdate(N, L, H, I, torch.float32, chunk)
            p.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("gate")})
            xt = x.reshape(B * S, H)
            _, _, idx = ref.gate(xt)
            e_ids = torch.arange(L).unsqueeze(0) * N + torch.arange(N).unsqueeze(1)
            t2e = (idx[None, None] == e_ids[..., None, None]).any(-1)
            ref_p = ref.experts(xt, idx, torch.ones_like(idx, dtype=torch.float32))
            prefill = parity.compare(f"prefill L={L} vs Qwen3VLMoe experts", p(xt, t2e), ref_p, atol=1e-5, rtol=1e-5)
        parity.require({"passed": routed["passed"] and prefill["passed"], "checks": [routed, prefill]}, f"HF qwen3_vl_moe L={L}")
