"""Expert-blocked NSP MoE FFN (pure torch; parameter names kept from the original bench).

* ``GatherBmmDecode``        – decode: gather expert weights per (token, k) and bmm
                               (GPT-OSS style clamp/GLU, optional dense shared experts).
* ``GatherBmmLoopDecode``    – decode: one-token loop around the same gather/BMM body.
* ``CumsumScatterGatherUpdate`` – prefill: per NSP lane, pack the routed rows with a
                               cumsum + CtxScatter3DInt, then gather → SwiGLU → scatter-update
                               in ``packed_chunk_size`` chunks. Mask ``local_T2E`` is an input.
* ``CumsumScatterGatherUpdateWithRouter`` – same prefill body with the standalone
                               sigmoid/noaux_tc router in the graph.
* ``RoutedCumsumScatterGatherUpdate`` – same body with the Qwen3-VL TopK router in-graph
                               (x ``[B, S, H]``), router weights applied per packed row.
* ``Reference``              – explicit dense reference retained for parity/debugging.

Expert ``e = le * num_nsp + n`` lives in lane ``n``, local slot ``le``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_benchmarks.common import custom_ops

from .ops import CtxGather3D_v18, CtxScatter3D_v18, CtxScatter3DInt_v18

INT32_MAX = torch.iinfo(torch.int32).max


# ── custom-op wrappers (eager bodies = the original's dynamo-mode torch.library ops) ──
def _gather_eager(data, idx):
    safe = torch.where(idx == INT32_MAX, torch.zeros_like(idx), idx).long()
    return data.gather(1, safe.unsqueeze(-1).expand(data.shape[0], safe.shape[1], data.shape[2]))


def _scatter_eager(data, pos, updates):
    """Scatter-set via scatter_add (no boolean indexing); INT32_MAX positions are no-ops."""
    safe = torch.where(pos == INT32_MAX, torch.zeros_like(pos), pos).long()
    valid = (pos != INT32_MAX).to(updates.dtype)
    masked = updates * valid.unsqueeze(-1)
    zeros = torch.zeros_like(data)
    scattered = zeros.scatter_add(1, safe.unsqueeze(-1).expand_as(masked), masked)
    written = zeros[:, :, :1].to(torch.int32).scatter_add(1, safe.unsqueeze(-1), valid.to(torch.int32).unsqueeze(-1))
    return torch.where(written.bool(), scattered, data)


def ctx_gather(data, idx):
    return custom_ops.call(CtxGather3D_v18, (data, idx), _gather_eager, shape=(data.shape[0], idx.shape[1], data.shape[2]))


def ctx_scatter(data, pos, updates):
    fn = CtxScatter3DInt_v18 if data.dtype == torch.int32 else CtxScatter3D_v18
    return custom_ops.call(fn, (data, pos, updates), _scatter_eager, shape=data.shape)


# ── decode: gather-weights + bmm ─────────────────────────────────────────────────
def build_router_inputs(T, top_k, total_experts, dtype):
    router_indices = torch.stack([torch.randperm(total_experts)[:top_k].to(torch.int32) for _ in range(T)])
    router_weights = torch.softmax(torch.randn(T, top_k, dtype=dtype), dim=-1)
    return router_indices, router_weights


class GatherBmmDecode(nn.Module):
    """x [T, H], router_indices [T, K] int32, router_weights [T, K] → [T, H]."""

    def __init__(self, num_nsp, local_experts, hidden_size, inter_size, dtype, num_experts_per_tok=8,
                 shared_experts=0, limit=127.0, alpha=1.0, token_block=None):
        super().__init__()
        E, H, I, S = num_nsp * local_experts, hidden_size, inter_size, shared_experts
        self.top_k, self.num_shared_experts = num_experts_per_tok, S
        self.token_block = token_block
        self.register_buffer("gate_clamp_min", torch.tensor(torch.finfo(torch.float16).min, dtype=torch.float32))
        self.register_buffer("gate_clamp_max", torch.tensor(limit, dtype=torch.float32))
        self.register_buffer("up_clamp_min", torch.tensor(-limit, dtype=torch.float32))
        self.register_buffer("up_clamp_max", torch.tensor(limit, dtype=torch.float32))
        self.register_buffer("alpha_val", torch.tensor(alpha, dtype=torch.float32))
        self.W_g = nn.Parameter(torch.randn(E, H, I, dtype=dtype) * 0.02)
        self.b_g = nn.Parameter(torch.zeros(E, I, dtype=dtype))
        self.W_u = nn.Parameter(torch.randn(E, H, I, dtype=dtype) * 0.02)
        self.b_u = nn.Parameter(torch.zeros(E, I, dtype=dtype))
        self.W_d = nn.Parameter(torch.randn(E, I, H, dtype=dtype) * 0.02)
        self.b_d = nn.Parameter(torch.zeros(E, H, dtype=dtype))
        if S:
            self.W_g_shared = nn.Parameter(torch.randn(S, H, I, dtype=dtype) * 0.02)
            self.b_g_shared = nn.Parameter(torch.zeros(S, I, dtype=dtype))
            self.W_u_shared = nn.Parameter(torch.randn(S, H, I, dtype=dtype) * 0.02)
            self.b_u_shared = nn.Parameter(torch.zeros(S, I, dtype=dtype))
            self.W_d_shared = nn.Parameter(torch.randn(S, I, H, dtype=dtype) * 0.02)
            self.b_d_shared = nn.Parameter(torch.zeros(S, H, dtype=dtype))

    def _ffn(self, x, W_g, b_g, W_u, b_u, W_d, b_d):
        gate = (torch.bmm(x, W_g) + b_g.unsqueeze(1)).clamp(min=self.gate_clamp_min, max=self.gate_clamp_max)
        up = (torch.bmm(x, W_u) + b_u.unsqueeze(1)).clamp(min=self.up_clamp_min, max=self.up_clamp_max)
        glu = gate * torch.sigmoid(gate * self.alpha_val)
        return torch.bmm((up + 1) * glu, W_d) + b_d.unsqueeze(1)

    def forward(self, x, router_indices, router_weights):
        # Token blocking: gather/BMM over [tile*K] expert rows instead of all [T*K] at once.
        tb = self.token_block
        if tb is None or tb >= x.shape[0]:
            return self._forward_tile(x, router_indices, router_weights)
        return torch.cat([self._forward_tile(x[s:s + tb], router_indices[s:s + tb], router_weights[s:s + tb])
                          for s in range(0, x.shape[0], tb)], dim=0)

    def _forward_tile(self, x, router_indices, router_weights):
        T, H = x.shape
        K = self.top_k
        idx = router_indices.reshape(-1).long()
        expert_in = x.unsqueeze(1).expand(-1, K, -1).contiguous().view(-1, 1, H)
        out = self._ffn(expert_in, self.W_g[idx], self.b_g[idx], self.W_u[idx], self.b_u[idx], self.W_d[idx], self.b_d[idx])
        output = (out.view(T, K, H) * router_weights.unsqueeze(-1)).sum(dim=1)
        if self.num_shared_experts:
            # Dense shared experts: weights expanded over T so this stays a batched bmm.
            S = self.num_shared_experts
            I = self.W_g_shared.shape[2]
            rep = lambda w, *shape: w.unsqueeze(0).expand(T, *w.shape).reshape(T * S, *shape)  # noqa: E731
            shared = self._ffn(
                x.unsqueeze(1).expand(-1, S, -1).reshape(T * S, 1, H),
                rep(self.W_g_shared, H, I), rep(self.b_g_shared, I),
                rep(self.W_u_shared, H, I), rep(self.b_u_shared, I),
                rep(self.W_d_shared, I, H), rep(self.b_d_shared, H),
            )
            output = output + (torch.zeros(T, S, H, dtype=x.dtype) + shared.view(T, S, H)).sum(dim=1)
        return output


# ── prefill / routed: cumsum pack + chunked gather → SwiGLU → scatter-update ────────
def matched_idx_from_cumsum(T2Ei):
    """[N, T] bool → [N, T] int32: packed row p of lane n holds token matched[n, p] (INT32_MAX = empty)."""
    N, T = T2Ei.shape
    token_idx = torch.arange(T, dtype=torch.int32).unsqueeze(0).expand(N, -1)
    scatter_pos = torch.where(T2Ei, torch.cumsum(T2Ei.to(torch.int32), dim=1) - 1, torch.tensor(INT32_MAX, dtype=torch.int32))
    matched = torch.full_like(token_idx, INT32_MAX)
    return ctx_scatter(matched.unsqueeze(-1), scatter_pos, token_idx.unsqueeze(-1)).squeeze(-1)


def swiglu(x, W_g, W_u, W_d, token_block=None):
    """(x@W_u * silu(x@W_g)) @ W_d, optionally in token blocks along dim 1 (original ``--ffn-blocking-mode token``)."""
    if token_block is None or token_block >= x.shape[1]:
        return (x @ W_u * F.silu(x @ W_g)) @ W_d
    return torch.cat([swiglu(x[:, s:s + token_block], W_g, W_u, W_d) for s in range(0, x.shape[1], token_block)], dim=1)


def update_one_local_expert(x, T2Ei, W_g, W_u, W_d, expert_out, chunk, router_weights=None, token_block=None):
    N, T = T2Ei.shape
    chunk = max(1, min(chunk, T))
    matched = matched_idx_from_cumsum(T2Ei)
    valid_rows = T2Ei.to(torch.int32).sum(dim=1, keepdim=True)
    row_range = torch.arange(chunk, dtype=torch.int32).unsqueeze(0)
    x_n = x.unsqueeze(0).expand(N, -1, -1)
    for start in range(0, T, chunk):
        idx = matched[:, start:start + chunk]
        xc = ctx_gather(x_n, idx)
        down = swiglu(xc, W_g, W_u, W_d, token_block)
        prev = ctx_gather(expert_out, idx)
        if router_weights is not None:
            down *= ctx_gather(router_weights, idx)  # in place: keeps down's dtype
        live = row_range < torch.clamp(valid_rows - start, min=0, max=chunk)
        updated = torch.where(live.unsqueeze(-1), prev + down, torch.zeros_like(down))
        expert_out = ctx_scatter(expert_out, idx, updated)
    return expert_out


class CumsumScatterGatherUpdate(nn.Module):
    """x [T, H], local_T2E [N, L, T] bool → [T, H]."""

    def __init__(self, num_nsp, local_experts, hidden_size, inter_size, dtype, packed_chunk_size=256, token_block=None):
        super().__init__()
        self.num_nsp, self.local_experts, self.packed_chunk_size = num_nsp, local_experts, packed_chunk_size
        self.token_block = token_block
        shape_in = (num_nsp, local_experts, hidden_size, inter_size)
        self.W_g = nn.Parameter(torch.randn(*shape_in, dtype=dtype) * 0.02)
        self.W_u = nn.Parameter(torch.randn(*shape_in, dtype=dtype) * 0.02)
        self.W_d = nn.Parameter(torch.randn(num_nsp, local_experts, inter_size, hidden_size, dtype=dtype) * 0.02)

    def forward(self, x, local_T2E):
        out = torch.zeros(self.num_nsp, x.shape[0], x.shape[1], dtype=x.dtype)
        for le in range(self.local_experts):
            out = update_one_local_expert(x, local_T2E[:, le], self.W_g[:, le], self.W_u[:, le], self.W_d[:, le],
                                          out, self.packed_chunk_size, token_block=self.token_block)
        return out.sum(dim=0)


class MoEGate(nn.Module):
    """Sigmoid/noaux_tc gate used by the standalone prefill-router path."""

    def __init__(self, num_experts_per_tok, n_routed_experts, hidden_size, dtype,
                 routed_scaling_factor=2.827, n_group=1, topk_group=1):
        super().__init__()
        if n_routed_experts % n_group:
            raise ValueError("n_routed_experts must be divisible by n_group")
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.weight = nn.Parameter(torch.empty(n_routed_experts, hidden_size, dtype=dtype))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(n_routed_experts, dtype=dtype))
        with torch.no_grad():
            tmp = torch.empty_like(self.weight, dtype=torch.float32)
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight.copy_(tmp.to(dtype))

    def forward(self, hidden_states):
        bsz, seq_len, hidden = hidden_states.shape
        logits = F.linear(hidden_states.reshape(-1, hidden), self.weight)
        scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        experts_per_group = self.n_routed_experts // self.n_group
        grouped = scores_for_choice.view(-1, self.n_group, experts_per_group)
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = group_scores.topk(self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(
            -1, self.n_group, experts_per_group
        ).reshape(-1, self.n_routed_experts).bool()
        _, topk_idx = scores_for_choice.masked_fill(~score_mask, float("-inf")).topk(
            self.top_k, dim=-1, sorted=False
        )
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            topk_weight = topk_weight * self.routed_scaling_factor
        # Match the standalone gate ABI: flattened token dimension [B*S, K].
        return topk_idx, topk_weight


class CumsumScatterGatherUpdateWithRouter(nn.Module):
    """Prefill cumsum implementation with the standalone sigmoid router."""

    def __init__(self, num_nsp, local_experts, hidden_size, inter_size, dtype,
                 packed_chunk_size=256, num_experts_per_tok=8,
                 experts_per_soc=None, token_block=None):
        super().__init__()
        self.num_nsp, self.local_experts = num_nsp, local_experts
        self.packed_chunk_size = packed_chunk_size
        self.experts_per_soc = experts_per_soc
        self.token_block = token_block
        self.gate = MoEGate(num_experts_per_tok, num_nsp * local_experts,
                            hidden_size, dtype)
        shape_in = (num_nsp, local_experts, hidden_size, inter_size)
        self.W_g = nn.Parameter(torch.randn(*shape_in, dtype=dtype) * 0.02)
        self.W_u = nn.Parameter(torch.randn(*shape_in, dtype=dtype) * 0.02)
        self.W_d = nn.Parameter(torch.randn(num_nsp, local_experts, inter_size, hidden_size, dtype=dtype) * 0.02)

    def forward(self, x):
        topk_idx, _topk_weight = self.gate(x)
        tokens, hidden = x.shape[0] * x.shape[1], x.shape[2]
        x_flat = x.reshape(tokens, hidden)
        n, local = self.num_nsp, self.local_experts
        expert_ids = (
            torch.arange(local, device=x.device, dtype=topk_idx.dtype).view(1, local) * n
            + torch.arange(n, device=x.device, dtype=topk_idx.dtype).view(n, 1)
        )
        eq = topk_idx.unsqueeze(0).unsqueeze(0) == expert_ids.unsqueeze(-1).unsqueeze(-1)
        local_t2e = eq.to(topk_idx.dtype).sum(dim=-1) > 0
        out = torch.zeros(n, tokens, hidden, dtype=x_flat.dtype, device=x.device)
        for le in range(local):
            out = update_one_local_expert(
                x_flat, local_t2e[:, le], self.W_g[:, le], self.W_u[:, le],
                self.W_d[:, le], out, self.packed_chunk_size,
                None, self.token_block,
            )
        if self.experts_per_soc is not None:
            out = out.view(n // self.experts_per_soc, self.experts_per_soc, tokens, hidden).sum(dim=1)
        return out.sum(dim=0)


class Reference(nn.Module):
    """Explicit dense reference retained from the standalone benchmark."""

    def __init__(self, num_nsp, local_experts, hidden_size, inter_size, dtype, **kwargs):
        super().__init__()
        self.num_nsp, self.local_experts = num_nsp, local_experts
        self.W_g = nn.Parameter(torch.full((num_nsp, local_experts, hidden_size, inter_size), 0.02, dtype=dtype))
        self.W_u = nn.Parameter(torch.full((num_nsp, local_experts, hidden_size, inter_size), 0.03, dtype=dtype))
        self.W_d = nn.Parameter(torch.full((num_nsp, local_experts, inter_size, hidden_size), 0.04, dtype=dtype))

    def forward(self, x, local_T2E):
        out = torch.zeros_like(x)
        x_all = x.unsqueeze(0).expand(self.num_nsp, -1, -1)
        for le in range(self.local_experts):
            gate = x_all @ self.W_g[:, le]
            up = x_all @ self.W_u[:, le]
            down = (up * F.silu(gate)) @ self.W_d[:, le]
            out = out + (down * local_T2E[:, le].unsqueeze(-1)).sum(dim=0)
        return out


class QEffMoeTextTopKRouter(nn.Module):
    """Qwen3-VL MoE router (QEff forward: einsum normaliser)."""

    def __init__(self, top_k, num_experts, hidden_dim, dtype=torch.float16):
        super().__init__()
        self.top_k, self.num_experts, self.hidden_dim = top_k, num_experts, hidden_dim
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_dim), dtype=dtype))
        with torch.no_grad():
            tmp = torch.empty_like(self.weight, dtype=torch.float32)
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))  # fp32 init, then cast (half has no kaiming)
            self.weight.copy_(tmp.to(dtype))

    def forward(self, hidden_states):
        logits = F.softmax(F.linear(hidden_states.reshape(-1, self.hidden_dim), self.weight), dtype=torch.float, dim=-1)
        top, idx = torch.topk(logits, self.top_k, dim=-1)
        top = (top / torch.einsum("bk->b", top).unsqueeze(-1)).to(logits.dtype)
        return logits, top, idx


class RoutedCumsumScatterGatherUpdate(nn.Module):
    """x [B, S, H] → [B*S, H]; routing from the in-graph TopK router."""

    def __init__(self, num_nsp, local_experts, hidden_size, inter_size, dtype, packed_chunk_size=256,
                 num_experts_per_tok=8, experts_per_soc=None, token_block=None):
        super().__init__()
        self.num_nsp, self.local_experts, self.packed_chunk_size = num_nsp, local_experts, packed_chunk_size
        self.token_block = token_block
        self.experts_per_soc = experts_per_soc
        self.gate = QEffMoeTextTopKRouter(num_experts_per_tok, num_nsp * local_experts, hidden_size, dtype)
        shape_in = (num_nsp, local_experts, hidden_size, inter_size)
        self.W_g = nn.Parameter(torch.randn(*shape_in, dtype=dtype) * 0.02)
        self.W_u = nn.Parameter(torch.randn(*shape_in, dtype=dtype) * 0.02)
        self.W_d = nn.Parameter(torch.randn(num_nsp, local_experts, inter_size, hidden_size, dtype=dtype) * 0.02)

    def forward(self, x):
        _, topk_weight, topk_idx = self.gate(x)
        T, H = x.shape[0] * x.shape[1], x.shape[2]
        N, L = self.num_nsp, self.local_experts
        x = x.view(T, H)
        expert_ids = (torch.arange(L, dtype=topk_idx.dtype).unsqueeze(0) * N
                      + torch.arange(N, dtype=topk_idx.dtype).unsqueeze(1))  # [N, L]
        # Each lane derives its own mask from the top-k ids. Cast→ReduceSum→Greater, not .any():
        # AIC rejects ReduceMax over the rank-4 bool.
        eq = topk_idx.unsqueeze(0).unsqueeze(0) == expert_ids.unsqueeze(-1).unsqueeze(-1)  # [N, L, T, K]
        local_T2E = eq.to(topk_idx.dtype).sum(dim=-1) > 0
        routing_weights = (eq.to(topk_weight.dtype) * topk_weight.unsqueeze(0).unsqueeze(0)).sum(dim=-1).unsqueeze(-1)
        out = torch.zeros(N, T, H, dtype=x.dtype)
        for le in range(L):
            out = update_one_local_expert(x, local_T2E[:, le], self.W_g[:, le], self.W_u[:, le], self.W_d[:, le],
                                          out, self.packed_chunk_size, routing_weights[:, le], self.token_block)
        if self.experts_per_soc is not None:
            # Sum inside each SoC first so only [1, T, H] per SoC crosses PCIe.
            out = out.view(N // self.experts_per_soc, self.experts_per_soc, T, H).sum(dim=1)
        return out.sum(dim=0)


class GatherBmmLoopDecode(nn.Module):
    """Decode gather/BMM path with one-token slices, matching the standalone loop."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.inner = GatherBmmDecode(*args, **kwargs)

    def forward(self, x, router_indices, router_weights):
        return torch.cat(
            [self.inner(x[i:i + 1], router_indices[i:i + 1], router_weights[i:i + 1])
             for i in range(x.shape[0])], dim=0
        )
