"""QSA parity loops.

* ``self_test`` – tiny config, every decode step through the full cache:
  eager vs ORT (all outputs incl. retained state) and eager vs the dense
  reference in ``ref.py``.
* ``hf_parity`` – the recurrent decode loop vs HF ``Qwen4ExpTextAttention``
  (prefill over the same tokens), weights copied from our module.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from layer_benchmarks.common import hf, parity

from .forwards import QSA, Config
from .ref import dense_qsa_step


@torch.no_grad()
def self_test(model: QSA, onnx_path, make_inputs) -> dict:
    import onnxruntime as ort

    from .bench import io_names

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    c = model.c
    dtype = next(model.parameters()).dtype
    input_names, _ = io_names(c)
    state = make_inputs(c, dtype)[2:]
    table = state[-1] if c.paged_kv else None
    if table is not None:
        state = state[:-1]
    ort_state = [s.numpy() for s in state]
    raw, keys, values = [], [], []
    max_ort = max_ref = 0.0
    for step in range(c.context):
        x = torch.randn(c.batch, 1, c.hidden, dtype=dtype)
        pos = torch.full((c.batch,), step, dtype=torch.long)
        feed = dict(zip(input_names, [x.numpy(), pos.numpy(), *ort_state]))
        model_args = (x, pos, *state)
        if table is not None:
            model_args = (*model_args, table)
            feed["block_table"] = table.numpy()
        out = model(*model_args)
        got = session.run(None, feed)
        for actual, expected in zip(got, out):
            np.testing.assert_allclose(actual, expected.numpy(), rtol=4e-4, atol=4e-5)
            max_ort = max(max_ort, float(np.abs(actual - expected.numpy()).max()))
        reference = dense_qsa_step(model, x, pos, raw, keys, values)
        torch.testing.assert_close(out[0], reference, rtol=4e-4, atol=4e-5)
        max_ref = max(max_ref, float((out[0] - reference).abs().max()))
        state, ort_state = out[1:], got[1:]
    return {"steps": c.context, "max_ort_error_all_states": max_ort, "max_dense_reference_error": max_ref, "passed": True}


@torch.no_grad()
def prefill_self_test(model: QSA, onnx_path, make_inputs, start_pos_id: int) -> dict:
    """Compare aligned prefill against ORT and recurrent decode."""
    import onnxruntime as ort

    from .bench import io_names

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    c = model.c
    dtype = next(model.parameters()).dtype
    args = make_inputs(c, dtype, start_pos_id)
    expected = model(*args)
    named_args = dict(zip(io_names(c)[0], args))
    feed = {
        item.name: named_args[item.name].numpy()
        for item in session.get_inputs()
    }
    actual = session.run(None, feed)
    max_ort = 0.0
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want.numpy(), rtol=4e-4, atol=4e-5)
        max_ort = max(max_ort, float(np.abs(got - want.numpy()).max()))

    decode = QSA(replace(c, seq_len=1)).eval().to(dtype)
    decode.load_state_dict(model.state_dict(), strict=True)
    state = [tensor.clone() for tensor in args[2:]]
    step_outputs = []
    for offset in range(c.seq_len):
        step = decode(
            args[0][:, offset : offset + 1],
            args[1][:, offset],
            *state,
        )
        step_outputs.append(step[0])
        state = list(step[1:])
    recurrent_output = torch.cat(step_outputs, dim=1)
    torch.testing.assert_close(expected[0], recurrent_output, rtol=4e-4, atol=4e-5)
    for prefill_state, recurrent_state in zip(expected[1:], state, strict=True):
        torch.testing.assert_close(prefill_state, recurrent_state, rtol=4e-4, atol=4e-5)
    max_decode = max(
        float((expected[0] - recurrent_output).abs().max()),
        *(
            float((a - b).abs().max())
            for a, b in zip(expected[1:], state, strict=True)
        ),
    )
    return {
        "query_len": c.seq_len,
        "max_ort_error_all_states": max_ort,
        "max_recurrent_decode_error": max_decode,
        "passed": True,
    }


HF_PROFILES = {
    "qwen38": dict(
        batch=2, hidden=2560, heads=24, kv_heads=2, dim=256, index_heads=4, index_dim=128, budget=2048, rotary_dim=64,
        rope={"rope_type": "default", "rope_theta": 1.0e7, "partial_rotary_factor": 0.25,
              "mrope_interleaved": True, "mrope_section": [11, 11, 10]},
    ),
    "tiny": dict(
        batch=2, hidden=64, heads=4, kv_heads=2, dim=16, index_heads=4, index_dim=8, budget=8, rotary_dim=4,
        rope={"rope_type": "default", "rope_theta": 1.0e7, "partial_rotary_factor": 0.25},
    ),
}


@torch.no_grad()
def hf_parity(tokens: int, seed: int, profile: str = "qwen38") -> dict:
    """Decode ``tokens`` steps one by one and compare with HF full-sequence attention."""
    from .bench import inputs

    if tokens < 4:
        raise ValueError("need at least one complete pooling group (4 tokens)")
    cfg_mod = hf.import_hf("qwen4_exp.configuration_qwen4_exp")
    mdl_mod = hf.import_hf("qwen4_exp.modeling_qwen4_exp")
    torch.manual_seed(seed)
    p = HF_PROFILES[profile]
    hf_config = cfg_mod.Qwen4ExpTextConfig(
        hidden_size=p["hidden"], num_hidden_layers=1, num_attention_heads=p["heads"], num_key_value_heads=p["kv_heads"],
        head_dim=p["dim"], max_position_embeddings=max(64, tokens), attention_bias=False, rms_norm_eps=1e-6,
        rope_parameters=p["rope"], indexer_n_heads=p["index_heads"], indexer_kv_heads=1,
        indexer_head_dim=p["index_dim"], indexer_budget=p["budget"], indexer_compress_ratio=4,
    )
    ref = mdl_mod.Qwen4ExpTextAttention(hf_config, layer_idx=0).eval().float()
    rope = mdl_mod.Qwen4ExpTextRotaryEmbedding(hf_config).eval()
    c = Config(
        batch=p["batch"], context=tokens, hidden=p["hidden"], heads=p["heads"], kv_heads=p["kv_heads"], dim=p["dim"],
        index_heads=p["index_heads"], index_dim=p["index_dim"], budget=p["budget"], rotary_dim=p["rotary_dim"],
        theta=1.0e7, indexer_dp=1, indexer_cp=1, attn_dp=1, num_devices=1,
    )
    local = QSA(c).eval().float()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        getattr(ref, name).weight.copy_(getattr(local, name).weight)
    for name in ("index_qk_proj", "q_layernorm", "k_layernorm"):
        getattr(ref.indexer, name).weight.copy_(getattr(local.indexer, name).weight)

    batch = c.batch
    x = torch.randn(batch, tokens, c.hidden)
    position_ids = torch.arange(tokens).view(1, 1, tokens).expand(3, batch, -1)
    causal = torch.ones(batch, 1, tokens, tokens, dtype=torch.bool).tril()
    mask = torch.zeros_like(causal, dtype=x.dtype).masked_fill(~causal, torch.finfo(x.dtype).min)
    expected, _ = ref(x, rope(x, position_ids), mask)

    state = inputs(c, torch.float32)[2:]
    steps = []
    for pos in range(tokens):
        out = local(x[:, pos : pos + 1], torch.full((batch,), pos, dtype=torch.long), *state)
        steps.append(out[0][:, 0])
        state = out[1:]
    result = parity.compare("attention_output", torch.stack(steps, 1), expected, atol=2e-5, rtol=2e-5)
    result.update(profile=profile, tokens=tokens, seed=seed)
    parity.require({"passed": result["passed"], "checks": [result]}, f"qsa hf {profile}")
    return result
