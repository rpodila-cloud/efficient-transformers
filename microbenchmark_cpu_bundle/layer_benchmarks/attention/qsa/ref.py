"""Independent dense QSA reference.

Does not import forwards/cache/ops: it rebuilds pooled index selection and
dense masked attention straight from the module parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


MASKED_LOGIT = -3.0e4


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    y = x.float()
    return (
        y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6)
        * (1 + weight.float())
    ).to(x.dtype)


def _rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    freq = position_ids.float()[:, None] * inv_freq.float()[None]
    freq = torch.cat((freq, freq), -1)[:, None]
    a, b = x[..., :rotary_dim].chunk(2, -1)
    rotated = x[..., :rotary_dim] * freq.cos().to(x.dtype) + torch.cat(
        (-b, a), -1
    ) * freq.sin().to(x.dtype)
    return torch.cat((rotated.to(x.dtype), x[..., rotary_dim:]), -1)


def _stable_topk(data: torch.Tensor, k: int):
    indices = torch.argsort(data, dim=-1, descending=True, stable=True)[..., :k]
    return data.gather(-1, indices), indices


def dense_qsa_step(
    module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    raw_history: list[torch.Tensor],
    key_history: list[torch.Tensor],
    value_history: list[torch.Tensor],
):
    """Run one dense full-history QSA step and append its history tensors."""
    c = module.c
    batch = c.batch
    ratio = c.ratio

    index_projection = F.linear(
        hidden_states,
        module.indexer.index_qk_proj.weight,
        module.indexer.index_qk_proj.bias,
    )
    index_query, raw = index_projection.split(
        [c.index_heads * c.index_dim, c.index_dim], -1
    )
    index_query = _rope(
        _rms_norm(
            index_query.reshape(batch, c.index_heads, c.index_dim),
            module.indexer.q_layernorm.weight,
        ),
        position_ids,
        module.inv_freq,
        c.rotary_dim,
    )
    raw_history.append(raw)

    q_gate = F.linear(hidden_states, module.q_proj.weight, module.q_proj.bias)
    query, gate = q_gate.reshape(batch, c.heads, 2 * c.dim).chunk(2, -1)
    query = _rope(
        _rms_norm(query, module.q_norm.weight),
        position_ids,
        module.inv_freq,
        c.rotary_dim,
    )
    key = _rope(
        _rms_norm(
            F.linear(hidden_states, module.k_proj.weight, module.k_proj.bias).reshape(
                batch, c.kv_heads, c.dim
            ),
            module.k_norm.weight,
        ),
        position_ids,
        module.inv_freq,
        c.rotary_dim,
    )
    value = F.linear(hidden_states, module.v_proj.weight, module.v_proj.bias).reshape(
        batch, c.kv_heads, c.dim
    )
    key_history.append(key)
    value_history.append(value)

    num_complete = (len(raw_history)) // ratio
    mask = torch.zeros(batch, len(raw_history), dtype=torch.bool)
    if num_complete:
        history = torch.cat(raw_history, 1)[:, : num_complete * ratio]
        pooled = _rms_norm(
            history.float()
            .reshape(batch, num_complete, ratio, c.index_dim)
            .mean(2)
            .to(hidden_states.dtype),
            module.indexer.k_layernorm.weight,
        )
        pooled = torch.stack(
            [
                _rope(
                    pooled[:, j : j + 1],
                    torch.full_like(position_ids, j * ratio),
                    module.inv_freq,
                    c.rotary_dim,
                )[:, 0]
                for j in range(num_complete)
            ],
            1,
        )
        scores = F.relu(
            index_query.float() @ pooled.float().transpose(-1, -2)
        ).sum(1) / math.sqrt(c.index_dim)
        padded = F.pad(
            scores, (0, c.context // ratio - num_complete), value=-torch.inf
        )
        local_context = c.context // (ratio * c.indexer_cp)
        k = min(c.budget // ratio, c.context // ratio)
        local_k = min(k, local_context)
        partitioned = padded.reshape(
            batch, local_context, c.indexer_cp
        ).transpose(1, 2)
        values, indices = _stable_topk(partitioned, local_k)
        indices = indices * c.indexer_cp + torch.arange(c.indexer_cp)[None, :, None]
        _, winners = _stable_topk(values.flatten(1), k)
        selected = indices.flatten(1).gather(1, winners)[:, : min(k, num_complete)]
        cutoff = scores.topk(min(k, num_complete), -1).values[:, -1:]
        if not torch.all(scores.gather(1, selected) >= cutoff - 1e-6):
            raise AssertionError("hierarchical index TopK is not globally valid")
        token = (
            selected[:, :, None] * ratio + torch.arange(ratio)[None, None]
        ).flatten(1)
        mask.scatter_(1, token, True)

    mask[:, num_complete * ratio :] = True
    keys = torch.stack(key_history, 2)
    values = torch.stack(value_history, 2)
    logits = (
        query.reshape(batch, c.kv_heads, c.heads // c.kv_heads, c.dim)
        @ keys.transpose(-1, -2)
        / math.sqrt(c.dim)
    )
    probabilities = (
        logits.masked_fill(~mask[:, None, None], -torch.inf)
        .float()
        .softmax(-1)
        .to(hidden_states.dtype)
    )
    attended = (probabilities @ values).reshape(batch, 1, -1)
    return F.linear(
        attended * gate.reshape(batch, 1, -1).sigmoid(),
        module.o_proj.weight,
        module.o_proj.bias,
    )
