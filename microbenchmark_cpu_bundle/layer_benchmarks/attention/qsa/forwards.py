"""Qwen3.8-Flash-Next QSA decode/prefill: pooled index, TopK, sparse GQA.

Cache mechanics live in ``cache.py``; the module helpers are also exposed by
``Qwen38QSAAttentionCache`` for consistency with the benchmark framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from . import cache
from .cache import Qwen38QSAAttentionCache
from .ops import stable_topk

MASKED_LOGIT = -3.0e4


def _merge_hierarchical_topk(
    scores: torch.Tensor,
    ids: torch.Tensor,
    local_k: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select locally per CP row, then merge only compact candidates."""
    batch_local, dp, cp, local_context = scores.shape
    if tuple(ids.shape) != (cp, local_context):
        raise ValueError("ids must have shape [cp, local_context]")
    values, selected_local = stable_topk(scores, local_k)
    global_ids = ids.view(1, 1, cp, local_context).expand(
        batch_local, dp, cp, local_context
    )
    selected_global = global_ids.gather(-1, selected_local)
    if cp == 1:
        values = values[:, :, 0]
        selected = selected_global[:, :, 0]
    else:
        candidate_values = values.reshape(batch_local, dp, cp * local_k)
        candidate_ids = selected_global.reshape(batch_local, dp, cp * local_k)
        values, winners = stable_topk(candidate_values, k)
        selected = candidate_ids.gather(-1, winners)
    values = values.reshape(batch_local * dp, k)
    selected = selected.reshape(batch_local * dp, k)
    return values, selected, values > MASKED_LOGIT


def _two_stage_attention_merge(
    query_group: torch.Tensor,
    gathered_keys: torch.Tensor,
    gathered_values: torch.Tensor,
    row_valid: torch.Tensor,
) -> torch.Tensor:
    """Merge owner-local GQA attention statistics across CP rows."""
    batch_local, dp, cp, kv_heads, gather_len, dim = gathered_keys.shape
    batch = query_group.shape[0]
    heads_per_kv = query_group.shape[2]
    if batch != batch_local * dp:
        raise ValueError("query batch does not match gathered cache layout")
    if gathered_values.shape != gathered_keys.shape:
        raise ValueError("gathered key/value shapes must match")
    if tuple(row_valid.shape) != (batch_local, dp, cp, gather_len):
        raise ValueError("row_valid does not match gathered cache shape")

    query_rows = query_group.reshape(
        batch_local, dp, kv_heads, heads_per_kv, dim
    ).unsqueeze(2)
    logits = (query_rows @ gathered_keys.transpose(-1, -2)) / math.sqrt(dim)
    mask = row_valid[:, :, :, None, None, :]
    logits = (
        logits * mask.to(logits.dtype)
        + (~mask).to(logits.dtype) * MASKED_LOGIT
    ).float()
    row_max = logits.max(dim=-1).values
    exp_logits = torch.exp(logits - row_max.unsqueeze(-1))
    exp_logits = torch.where(mask, exp_logits, torch.zeros_like(exp_logits))
    row_sum = exp_logits.sum(dim=-1)
    row_out = exp_logits @ gathered_values.float()

    global_max = row_max.max(dim=2).values
    cp_weights = torch.exp(row_max - global_max.unsqueeze(2))
    denominator = (cp_weights * row_sum).sum(dim=2)
    numerator = (cp_weights.unsqueeze(-1) * row_out).sum(dim=2)
    output = numerator / denominator.unsqueeze(-1)
    return output.to(query_group.dtype).reshape(batch, 1, -1)


@dataclass
class Config:
    batch: int = 16
    seq_len: int = 1
    context: int = 262144
    hidden: int = 2560
    heads: int = 24
    kv_heads: int = 2
    dim: int = 256
    index_heads: int = 4
    index_dim: int = 128
    ratio: int = 4
    budget: int = 2048
    rotary_dim: int = 64
    theta: float = 10000000.0
    indexer_dp: int = 16
    indexer_cp: int = 1
    attn_dp: int = 16
    attn_cp: int = 1
    num_devices: int = 16
    num_cores: int = 16
    indexer_num_blocks: int = 4
    indexer_ql_chunk: int = 512
    indexer_q_block_size: int = 512
    indexer_topk_blocking: int = 4
    sparse_q_block_size: int = 16
    sparse_kv_num_blocks: int = 8
    skip_kv: bool = False
    paged_kv: bool = False
    page_size: int = 4
    cache_layout: str = "dp_rows"  # or "batch_major"
    split_cache_cp: bool = False


TINY = dict(hidden=32, heads=4, kv_heads=2, dim=8, index_dim=8, budget=8, rotary_dim=4)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6) * (1 + self.weight.float())).to(x.dtype)


class QSA(Qwen38QSAAttentionCache, nn.Module):
    """One QSA layer: ``forward(hidden, position_ids, *cache_state)``."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.use_owner_local_attention_cp = True
        # None keeps parity dynamic; hardware exports specialise the
        # boundary (True) / non-boundary (False) residue.
        self.position_mode = None
        self.q_proj = nn.Linear(c.hidden, c.heads * c.dim * 2, bias=False)
        self.k_proj = nn.Linear(c.hidden, c.kv_heads * c.dim, bias=False)
        self.v_proj = nn.Linear(c.hidden, c.kv_heads * c.dim, bias=False)
        self.o_proj = nn.Linear(c.heads * c.dim, c.hidden, bias=False)
        self.q_norm, self.k_norm = RMSNorm(c.dim), RMSNorm(c.dim)
        self.indexer = nn.Module()
        self.indexer.index_qk_proj = nn.Linear(c.hidden, (c.index_heads + 1) * c.index_dim, bias=False)
        self.indexer.q_layernorm = RMSNorm(c.index_dim)
        self.indexer.k_layernorm = RMSNorm(c.index_dim)
        self.register_buffer("inv_freq", 1 / (c.theta ** (torch.arange(0, c.rotary_dim, 2).float() / c.rotary_dim)))

    def rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        c = self.c
        freq = pos.float().unsqueeze(-1) * self.inv_freq.float()
        freq = torch.cat((freq, freq), -1)
        if x.ndim == pos.ndim + 2:
            freq = freq.unsqueeze(-2)
        elif x.ndim != pos.ndim + 1:
            raise ValueError("RoPE input rank does not match position IDs")
        a, b = x[..., : c.rotary_dim].chunk(2, -1)
        rotated = x[..., : c.rotary_dim] * freq.cos().to(x.dtype) + torch.cat(
            (-b, a), -1
        ) * freq.sin().to(x.dtype)
        return torch.cat((rotated.to(x.dtype), x[..., c.rotary_dim :]), -1)

    def _build_prefill_topk_q_split(
        self,
        index_query: torch.Tensor,
        position_ids: torch.Tensor,
        index_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select pooled blocks with Q split and pooled K replicated."""
        c = self.c
        batch, query_len, index_heads, index_dim = index_query.shape
        pooled_context = c.context // c.ratio
        num_blocks = c.indexer_num_blocks
        num_cores = c.num_cores
        if pooled_context % num_blocks:
            raise ValueError("pooled context must divide evenly into indexer blocks")
        kv_block_size = pooled_context // num_blocks
        if (
            query_len % c.indexer_q_block_size
            or c.indexer_ql_chunk % c.indexer_q_block_size
            or c.indexer_q_block_size % num_cores
            or c.indexer_q_block_size % c.indexer_topk_blocking
        ):
            raise ValueError("invalid split-Q prefill indexer blocking")

        q_per_core = c.indexer_q_block_size // num_cores
        topk = min(c.budget // c.ratio, pooled_context)
        batch_local = batch // c.indexer_dp
        rows = c.indexer_dp
        mask_value = float(MASKED_LOGIT)
        complete = cache.trunc_div_pow2_nonnegative(position_ids + 1, c.ratio)
        local_key_offsets = torch.arange(
            kv_block_size,
            device=index_query.device,
            dtype=torch.int32,
        ).view(1, 1, 1, kv_block_size)
        topk_chunks: list[torch.Tensor] = []
        valid_chunks: list[torch.Tensor] = []

        for q_chunk_start in range(0, query_len, c.indexer_ql_chunk):
            q_chunk_end = min(q_chunk_start + c.indexer_ql_chunk, query_len)
            query_chunk = q_chunk_end - q_chunk_start
            if query_chunk % c.indexer_q_block_size:
                raise ValueError("every indexer Q chunk must divide into fixed Q blocks")
            query_chunk_tensor = index_query[:, q_chunk_start:q_chunk_end]
            complete_chunk = complete[:, q_chunk_start:q_chunk_end]
            q_block_ranges = list(
                range(q_chunk_start, q_chunk_end, c.indexer_q_block_size)
            )
            q_block_metadata: list[tuple[torch.Tensor, torch.Tensor]] = []
            for q_start in q_block_ranges:
                local_start = q_start - q_chunk_start
                local_end = local_start + c.indexer_q_block_size
                q_core = query_chunk_tensor[:, local_start:local_end].reshape(
                    batch,
                    num_cores,
                    q_per_core,
                    index_heads,
                    index_dim,
                )
                complete_core = complete_chunk[:, local_start:local_end].reshape(
                    batch,
                    num_cores,
                    q_per_core,
                )
                q_block_metadata.append((q_core, complete_core))

            score_blocks_by_q: list[list[torch.Tensor]] = [
                [] for _ in q_block_ranges
            ]
            index_blocks_by_q: list[list[torch.Tensor]] = [
                [] for _ in q_block_ranges
            ]

            for block_idx in range(num_blocks):
                start = block_idx * kv_block_size
                key_positions = torch.arange(
                    start,
                    start + kv_block_size,
                    device=index_query.device,
                    dtype=torch.int32,
                ).view(1, 1, kv_block_size)
                gather_limit = (
                    complete_chunk.max(dim=1).values - 1
                ).reshape(batch_local, rows, 1)
                gather_indices = torch.where(
                    key_positions.expand(batch_local, rows, kv_block_size)
                    > gather_limit,
                    torch.full(
                        (batch_local, rows, kv_block_size),
                        torch.iinfo(torch.int32).max,
                        device=index_query.device,
                        dtype=torch.int32,
                    ),
                    key_positions.expand(batch_local, rows, kv_block_size),
                )
                key_block = cache.gather_rows(
                    index_cache, gather_indices
                ).reshape(batch, kv_block_size, index_dim).float()
                key_core = key_block.unsqueeze(1).expand(
                    batch, num_cores, kv_block_size, index_dim
                )
                block_positions = start + local_key_offsets

                for q_block_idx, (q_core, complete_core) in enumerate(q_block_metadata):
                    scores_core = F.relu(
                        torch.matmul(
                            q_core.float(),
                            key_core[:, :, None].transpose(-1, -2),
                        )
                    ).sum(-2) / math.sqrt(c.index_dim)
                    causal_mask = block_positions >= complete_core.unsqueeze(-1)
                    scores_core = scores_core.masked_fill(causal_mask, mask_value)
                    score_blocks_by_q[q_block_idx].append(
                        scores_core.reshape(
                            batch, c.indexer_q_block_size, kv_block_size
                        )
                    )
                    index_blocks_by_q[q_block_idx].append(
                        block_positions.expand(
                            batch, num_cores, q_per_core, kv_block_size
                        ).reshape(batch, c.indexer_q_block_size, kv_block_size)
                    )

            selected_scores: list[torch.Tensor] = []
            selected_indices: list[torch.Tensor] = []
            q_sub_block_size = c.indexer_q_block_size // c.indexer_topk_blocking
            for score_parts, index_parts in zip(score_blocks_by_q, index_blocks_by_q):
                candidate_scores = torch.cat(score_parts, dim=-1)
                candidate_indices = torch.cat(index_parts, dim=-1)
                for sub_block in range(c.indexer_topk_blocking):
                    q_start = sub_block * q_sub_block_size
                    q_end = q_start + q_sub_block_size
                    values, order = stable_topk(
                        candidate_scores[:, q_start:q_end], topk
                    )
                    selected_scores.append(values)
                    selected_indices.append(
                        candidate_indices[:, q_start:q_end].gather(-1, order)
                    )

            scores = torch.cat(selected_scores, dim=1)
            topk_chunks.append(torch.cat(selected_indices, dim=1))
            valid_chunks.append(scores > mask_value)

        return torch.cat(topk_chunks, dim=1), torch.cat(valid_chunks, dim=1)

    def _gather_gqa_grouped_prefill(
        self,
        retained_cache: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Gather one KV-head row for each disjoint core group."""
        c = self.c
        if c.attn_cp != 1:
            raise ValueError("QSA grouped prefill attention requires attn_cp=1")
        if positions.ndim != 5 or valid.shape != positions.shape:
            raise ValueError("grouped positions must have shape [B,Hkv,G,Q/G,K]")
        batch, kv_heads, cores_per_kv_head, q_per_core, gather_len = positions.shape
        if kv_heads != c.kv_heads or batch % c.attn_dp:
            raise ValueError("grouped gather shape does not match QSA config")
        batch_local = batch // c.attn_dp
        rows = c.attn_dp * kv_heads
        if retained_cache.ndim != 4 or retained_cache.shape[:2] != (batch_local, rows):
            raise ValueError("cache must have shape [B/attn_dp,attn_dp*Hkv,T,D]")

        indices = positions.reshape(
            batch_local,
            c.attn_dp,
            kv_heads,
            cores_per_kv_head,
            q_per_core,
            gather_len,
        ).reshape(
            batch_local,
            rows,
            cores_per_kv_head,
            q_per_core,
            gather_len,
        )
        valid_rows = valid.reshape(
            batch_local,
            rows,
            cores_per_kv_head,
            q_per_core,
            gather_len,
        )
        indices = torch.where(
            valid_rows,
            indices,
            torch.full_like(indices, torch.iinfo(torch.int32).max),
        ).to(torch.int32)
        gathered = cache.gather_rows(retained_cache, indices).reshape(
            batch,
            kv_heads,
            cores_per_kv_head,
            q_per_core,
            gather_len,
            retained_cache.shape[-1],
        )
        return torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))

    def _sparse_prefill_attention_gqa_grouped(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        selected_blocks: torch.Tensor,
        selected_valid: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run GQA with one ``num_cores/kv_heads`` group per KV head."""
        c = self.c
        batch, query_len, kv_heads, heads_per_kv, dim = query.shape
        if kv_heads != c.kv_heads:
            raise ValueError("query KV-head grouping does not match config")
        if c.num_cores % kv_heads:
            raise ValueError("num_cores must be divisible by kv_heads")
        cores_per_kv_head = c.num_cores // kv_heads
        if (
            query_len % c.sparse_q_block_size
            or c.sparse_q_block_size % cores_per_kv_head
        ):
            raise ValueError("invalid KV-head-grouped Q/core blocking")

        selected_count = selected_blocks.shape[-1]
        logical_block_count = selected_count + 1
        if c.sparse_kv_num_blocks > logical_block_count:
            raise ValueError("too many sparse KV chunks for selected logical blocks")

        block_offsets = torch.arange(
            c.ratio, device=query.device, dtype=selected_blocks.dtype
        )
        complete = cache.trunc_div_pow2_nonnegative(position_ids + 1, c.ratio)
        pooled_context = c.context // c.ratio
        tail_in_range = complete < pooled_context
        tail_block = complete.clamp(max=pooled_context - 1)
        logical_blocks = torch.cat((selected_blocks, tail_block.unsqueeze(-1)), dim=-1)
        logical_valid = torch.cat((selected_valid, tail_in_range.unsqueeze(-1)), dim=-1)
        is_export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        output_chunks: list[torch.Tensor] = []

        for q_chunk_start in range(0, query_len, c.sparse_q_block_size):
            q_chunk_end = min(q_chunk_start + c.sparse_q_block_size, query_len)
            query_chunk = q_chunk_end - q_chunk_start
            q_per_core = query_chunk // cores_per_kv_head
            query_work = query[:, q_chunk_start:q_chunk_end].permute(
                0, 2, 1, 3, 4
            ).reshape(
                batch,
                kv_heads,
                cores_per_kv_head,
                q_per_core,
                heads_per_kv,
                dim,
            )
            block_ids_chunk = logical_blocks[:, q_chunk_start:q_chunk_end]
            block_valid_chunk = logical_valid[:, q_chunk_start:q_chunk_end]
            query_positions_chunk = position_ids[:, q_chunk_start:q_chunk_end]

            running_max = torch.full(
                query_work.shape[:-1],
                float(MASKED_LOGIT),
                dtype=query_work.dtype,
                device=query.device,
            )
            denominator = torch.zeros_like(running_max)
            output = torch.zeros(
                *running_max.shape,
                dim,
                dtype=query_work.dtype,
                device=query.device,
            )

            for kv_block_idx in range(c.sparse_kv_num_blocks):
                logical_start = logical_block_count * kv_block_idx // c.sparse_kv_num_blocks
                logical_end = logical_block_count * (kv_block_idx + 1) // c.sparse_kv_num_blocks
                block_ids = block_ids_chunk[..., logical_start:logical_end]
                block_valid = block_valid_chunk[..., logical_start:logical_end]
                positions_block = (
                    block_ids.unsqueeze(-1) * c.ratio + block_offsets
                ).flatten(-2)
                valid_block = block_valid.unsqueeze(-1).expand(
                    *block_valid.shape, c.ratio
                ).flatten(-2)
                valid_block = valid_block & (
                    positions_block <= query_positions_chunk.unsqueeze(-1)
                )
                token_count = positions_block.shape[-1]
                positions_group = positions_block.reshape(
                    batch,
                    cores_per_kv_head,
                    q_per_core,
                    token_count,
                ).unsqueeze(1).expand(
                    batch,
                    kv_heads,
                    cores_per_kv_head,
                    q_per_core,
                    token_count,
                )
                valid_group = valid_block.reshape(
                    batch,
                    cores_per_kv_head,
                    q_per_core,
                    token_count,
                ).unsqueeze(1).expand_as(positions_group)

                skip_future = None
                if c.skip_kv:
                    skip_future = ~valid_group.any(dim=-1)
                    if not is_export and bool(skip_future.all().item()):
                        continue

                key_block = self._gather_gqa_grouped_prefill(
                    key_cache, positions_group, valid_group
                )
                value_block = self._gather_gqa_grouped_prefill(
                    value_cache, positions_group, valid_group
                )
                work_rows = batch * kv_heads * cores_per_kv_head * q_per_core
                logits = torch.bmm(
                    query_work.reshape(work_rows, heads_per_kv, dim),
                    key_block.reshape(work_rows, token_count, dim).transpose(1, 2),
                ).reshape(
                    batch,
                    kv_heads,
                    cores_per_kv_head,
                    q_per_core,
                    heads_per_kv,
                    token_count,
                ) / math.sqrt(dim)
                mask = valid_group.unsqueeze(-2)
                logits = logits.masked_fill(~mask, float(MASKED_LOGIT))
                block_max = logits.max(dim=-1).values
                next_max = torch.maximum(running_max, block_max)
                previous_scale = torch.exp(running_max - next_max)
                probabilities = torch.exp(logits - next_max.unsqueeze(-1))
                probabilities = probabilities * mask.to(probabilities.dtype)
                next_denominator = (
                    denominator * previous_scale
                    + probabilities.sum(dim=-1, dtype=probabilities.dtype)
                )
                next_output = (
                    output * previous_scale.unsqueeze(-1)
                    + torch.matmul(probabilities, value_block)
                )
                if c.skip_kv and is_export:
                    skip_rows = skip_future.unsqueeze(-1)
                    running_max = torch.where(skip_rows, running_max, next_max)
                    denominator = torch.where(skip_rows, denominator, next_denominator)
                    output = torch.where(skip_rows.unsqueeze(-1), output, next_output)
                else:
                    running_max = next_max
                    denominator = next_denominator
                    output = next_output

            safe_denominator = torch.where(
                denominator > 0, denominator, torch.ones_like(denominator)
            )
            output = output / safe_denominator.unsqueeze(-1)
            output_chunks.append(
                output.to(query.dtype)
                .permute(0, 2, 3, 1, 4, 5)
                .reshape(batch, query_chunk, c.heads * dim)
            )

        return torch.cat(output_chunks, dim=1)

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        index_cache: torch.Tensor,
        index_partial: torch.Tensor,
    ):
        c = self.c
        batch, query_len, _ = hidden_states.shape
        if position_ids.shape != (batch, query_len):
            raise ValueError("prefill position_ids must have shape [B,Q]")
        if c.indexer_cp != 1 or c.attn_cp != 1:
            raise ValueError("QSA prefill currently requires indexer/attention CP=1")
        if query_len % c.ratio:
            raise ValueError("prefill Q length must divide by the pooling ratio")

        index_query, raw_index_key = self.indexer.index_qk_proj(
            hidden_states
        ).split([c.index_heads * c.index_dim, c.index_dim], -1)
        index_query = self.rope(
            self.indexer.q_layernorm(
                index_query.reshape(batch, query_len, c.index_heads, c.index_dim)
            ),
            position_ids,
        )

        pooled_count = query_len // c.ratio
        pooled_raw = raw_index_key.reshape(
            batch, pooled_count, c.ratio, c.index_dim
        ).float().mean(dim=2).to(hidden_states.dtype)
        pooled_positions = position_ids[:, ::c.ratio]
        pooled_keys = self.rope(
            self.indexer.k_layernorm(pooled_raw), pooled_positions
        )
        pooled_block_ids = cache.trunc_div_pow2_nonnegative(
            pooled_positions, c.ratio
        )
        index_cache = cache.scatter_sharded_cache_sequence(
            index_cache,
            pooled_block_ids,
            pooled_keys,
            dp=c.indexer_dp,
            cp=c.indexer_cp,
        )
        selected_blocks, valid_blocks = self._build_prefill_topk_q_split(
            index_query, position_ids, index_cache
        )

        query_and_gate = self.q_proj(hidden_states).reshape(
            batch, query_len, c.heads, 2 * c.dim
        )
        query, gate = query_and_gate.chunk(2, -1)
        query = self.rope(self.q_norm(query), position_ids)
        key = self.rope(
            self.k_norm(
                self.k_proj(hidden_states).reshape(
                    batch, query_len, c.kv_heads, c.dim
                )
            ),
            position_ids,
        )
        value = self.v_proj(hidden_states).reshape(
            batch, query_len, c.kv_heads, c.dim
        )
        key_cache = cache.scatter_sharded_cache_sequence(
            key_cache,
            position_ids,
            key,
            dp=c.attn_dp,
            cp=c.attn_cp,
            rows_per_dp_way=c.kv_heads,
        )
        value_cache = cache.scatter_sharded_cache_sequence(
            value_cache,
            position_ids,
            value,
            dp=c.attn_dp,
            cp=c.attn_cp,
            rows_per_dp_way=c.kv_heads,
        )

        output = self._sparse_prefill_attention_gqa_grouped(
            query.reshape(
                batch,
                query_len,
                c.kv_heads,
                c.heads // c.kv_heads,
                c.dim,
            ),
            key_cache,
            value_cache,
            selected_blocks,
            valid_blocks,
            position_ids,
        )
        output = self.o_proj(output * gate.reshape_as(output).sigmoid())
        # Aligned prefill starts with a cleared partial and completes every
        # pooling group. Pass it through so the retained input remains bound
        # to the output for direct decode handoff.
        return (
            output,
            key_cache,
            value_cache,
            index_cache,
            index_partial,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        *state,
    ):
        c = self.c
        if hidden_states.shape[1] != 1:
            if c.paged_kv or c.split_cache_cp or c.cache_layout != "dp_rows":
                raise ValueError("QSA prefill supports only non-paged dp_rows cache")
            if len(state) != 4:
                raise ValueError("QSA prefill requires key/value/index/partial inputs")
            return self._forward_prefill(hidden_states, position_ids, *state)
        if c.paged_kv:
            if c.split_cache_cp:
                raise ValueError("split-cache-cp is currently non-paged only")
            if len(state) != 5:
                raise ValueError("paged QSA requires block_table input")
            key_cache, value_cache, index_cache, index_partial, block_table = state
        else:
            block_table = None
            if c.split_cache_cp:
                cp = c.attn_cp
                if len(state) != 2 * cp + 2:
                    raise ValueError("invalid split QSA cache state tuple")
                key_cache = list(state[:cp])
                value_cache = list(state[cp : 2 * cp])
                index_cache, index_partial = state[2 * cp :]
            else:
                key_cache, value_cache, index_cache, index_partial = state
        batch, ratio, pooled_context = c.batch, c.ratio, c.context // c.ratio

        # Position IDs are nonnegative. This avoids Dynamo's integer Div+Sign
        # lowering, which QAIC does not support.
        block = cache.trunc_div_pow2_nonnegative(position_ids, ratio)
        complete = cache.trunc_div_pow2_nonnegative(position_ids + 1, ratio)
        if self.position_mode is None:
            boundary = position_ids + 1 == complete * ratio
        else:
            boundary = torch.full_like(
                position_ids, self.position_mode, dtype=torch.bool
            )

        # Indexer query and the raw key feature used by four-token pooling.
        index_query, raw_index_key = self.indexer.index_qk_proj(hidden_states).split(
            [c.index_heads * c.index_dim, c.index_dim], -1
        )
        index_query = self.rope(
            self.indexer.q_layernorm(
                index_query.reshape(batch, c.index_heads, c.index_dim)
            ),
            position_ids,
        )

        # index_partial carries the current incomplete group. The pooled value
        # is committed only on a boundary; non-boundary steps retain the sum.
        total = index_partial + raw_index_key.float()
        pooled = self.rope(
            self.indexer.k_layernorm((total / ratio).to(hidden_states.dtype)),
            block * ratio,
        )
        if self.position_mode is True:
            partial = torch.zeros_like(total)
        elif self.position_mode is False:
            partial = total
        else:
            partial = torch.where(
                boundary[:, None, None], torch.zeros_like(total), total
            )

        # Fold DP and CP into the index-cache row layout. A global pooled block
        # maps to cp_way = block % CP and local_slot = block // CP.
        local_context = pooled_context // c.indexer_cp
        batch_local = batch // c.indexer_dp
        if self.position_mode is not False:
            index_cache = cache.scatter_sharded_cache(
                index_cache,
                block,
                pooled,
                dp=c.indexer_dp,
                cp=c.indexer_cp,
                valid=boundary,
            )

        # Each CP partition scores its local index context. Local TopK results
        # are merged into one shared global selection.
        index_query = index_query.reshape(
            batch_local, c.indexer_dp, 1, c.index_heads, c.index_dim
        ).expand(-1, -1, c.indexer_cp, -1, -1)
        index_query = index_query.reshape(
            batch_local,
            c.indexer_dp * c.indexer_cp,
            c.index_heads,
            c.index_dim,
        )
        scores = F.relu(
            index_query.float() @ index_cache.float().transpose(-1, -2)
        ).sum(-2) / math.sqrt(c.index_dim)
        ids = (
            torch.arange(local_context, device=block.device)[None, :] * c.indexer_cp
            + torch.arange(c.indexer_cp, device=block.device)[:, None]
        )
        visible = ids[None, None] < complete.reshape(
            batch_local, c.indexer_dp, 1, 1
        )
        visible = visible.reshape(
            batch_local, c.indexer_dp * c.indexer_cp, local_context
        )
        scores = scores * visible.to(scores.dtype) + (~visible).to(scores.dtype) * MASKED_LOGIT
        scores = scores.reshape(
            batch_local, c.indexer_dp, c.indexer_cp, local_context
        )

        k = min(c.budget // ratio, pooled_context)
        local_k = min(k, local_context)
        _, selected, valid = _merge_hierarchical_topk(scores, ids, local_k, k)

        # Full K/V are retained; only index keys are compressed.
        query, gate = (
            self.q_proj(hidden_states)
            .reshape(batch, c.heads, 2 * c.dim)
            .chunk(2, -1)
        )
        query = self.rope(self.q_norm(query), position_ids)
        key = self.rope(
            self.k_norm(self.k_proj(hidden_states).reshape(batch, c.kv_heads, c.dim)),
            position_ids,
        )
        value = self.v_proj(hidden_states).reshape(batch, c.kv_heads, c.dim)

        # Retain K/V in folded rows: [1, CP*B, local_context, KV*D].  This
        # mirrors the folded-row layout used by the CSA/MSA benches.  DP and
        # CP ownership is represented by rows, so a BS16/1M cache is split
        # across MDP devices instead of exposing one 16 GiB dynamic map.
        cache_width = c.kv_heads * c.dim
        standard_folded_cache = (
            not c.paged_kv
            and not c.split_cache_cp
            and c.cache_layout == "dp_rows"
        )
        owner_local_attention = (
            standard_folded_cache
            and c.attn_cp > 1
            and self.use_owner_local_attention_cp
        )
        if c.paged_kv:
            if block_table is None:
                raise ValueError("paged QSA requires block_table")
            dp, cp, kv_heads = c.attn_dp, c.attn_cp, c.kv_heads
            batch_local = batch // dp
            pos_dp = position_ids.reshape(dp, batch_local).transpose(0, 1)
            logical_page = torch.div(pos_dp, ratio, rounding_mode="trunc")
            group_page = torch.div(logical_page, cp, rounding_mode="trunc")
            table_local = block_table.permute(1, 0, 2)
            physical_block = torch.gather(
                table_local, 2, group_page.unsqueeze(-1)
            ).squeeze(-1)
            owner_way = logical_page.remainder(cp)
            addr_dp = position_ids.reshape(dp, batch_local).transpose(0, 1).remainder(ratio)
            rows = dp * cp * kv_heads
            key_updates = key.reshape(batch_local, dp, kv_heads, 1, c.dim)
            value_updates = value.reshape(batch_local, dp, kv_heads, 1, c.dim)
            key_updates = key_updates[:, :, None].expand(
                batch_local, dp, cp, kv_heads, 1, c.dim
            ).reshape(batch_local, rows, 1, c.dim)
            value_updates = value_updates[:, :, None].expand(
                batch_local, dp, cp, kv_heads, 1, c.dim
            ).reshape(batch_local, rows, 1, c.dim)
            cp_ids = torch.arange(cp, device=position_ids.device).view(1, 1, cp, 1)
            live = (cp_ids == owner_way[:, :, None, None]).expand(
                batch_local, dp, cp, kv_heads
            ).reshape(batch_local, rows, 1)
            invalid = torch.iinfo(torch.int32).max
            block_ids = physical_block[:, :, None, None].expand(
                batch_local, dp, cp, kv_heads
            ).reshape(batch_local, rows, 1)
            block_ids = torch.where(live, block_ids, torch.full_like(block_ids, invalid))
            addr = addr_dp[:, :, None, None].expand(
                batch_local, dp, cp, kv_heads
            ).reshape(batch_local, rows, 1).to(torch.int32)
            key_cache = cache.scatter_cp(key_cache, block_ids, addr, key_updates)
            value_cache = cache.scatter_cp(value_cache, block_ids, addr, value_updates)
        if not c.paged_kv and c.split_cache_cp:
            # One retained tensor per CP way.  Each shard owns all DP/KV-head
            # rows for its local context, keeping BS16/1M state buffers below
            # the QAIC VA-map limit while retaining every K/V token.
            cp, dp, kv_heads = c.attn_cp, c.attn_dp, c.kv_heads
            batch_local = batch // dp
            pooled_pos = torch.div(position_ids, ratio, rounding_mode="trunc")
            cp_way = pooled_pos.remainder(cp).reshape(batch_local, dp)
            local_slots = (
                torch.div(pooled_pos, cp, rounding_mode="trunc") * ratio
                + position_ids.remainder(ratio)
            ).reshape(batch_local, dp)
            updates_key = key.reshape(batch_local, dp, kv_heads, 1, c.dim)
            updates_value = value.reshape(batch_local, dp, kv_heads, 1, c.dim)
            rows = dp * kv_heads
            invalid = torch.iinfo(torch.int32).max
            for way in range(cp):
                live = (cp_way == way).unsqueeze(-1).expand(
                    batch_local, dp, kv_heads
                ).reshape(batch_local, rows, 1)
                block_ids = torch.arange(batch_local, device=position_ids.device).view(
                    batch_local, 1, 1
                ).expand_as(live)
                block_ids = torch.where(
                    live, block_ids, torch.full_like(block_ids, invalid)
                )
                addr = local_slots[:, :, None].expand(
                    batch_local, dp, kv_heads
                ).reshape(batch_local, rows, 1).to(torch.int32)
                key_cache[way] = cache.scatter_cp(
                    key_cache[way], block_ids, addr,
                    updates_key.reshape(batch_local, rows, 1, c.dim),
                )
                value_cache[way] = cache.scatter_cp(
                    value_cache[way], block_ids, addr,
                    updates_value.reshape(batch_local, rows, 1, c.dim),
                )
        elif not c.paged_kv and c.cache_layout == "batch_major":
            # Batch-major retained rows: [B, CP*KVH, CL/CP, D].
            cp, kv_heads = c.attn_cp, c.kv_heads
            pooled_pos = torch.div(position_ids, ratio, rounding_mode="trunc")
            cp_way = pooled_pos.remainder(cp)
            local_slots = (
                torch.div(pooled_pos, cp, rounding_mode="trunc") * ratio
                + position_ids.remainder(ratio)
            )
            updates_key = key[:, None, :, None, :].expand(
                batch, cp, kv_heads, 1, c.dim
            )
            updates_value = value[:, None, :, None, :].expand(
                batch, cp, kv_heads, 1, c.dim
            )
            updates_key = updates_key.reshape(batch, cp * kv_heads, 1, c.dim)
            updates_value = updates_value.reshape(batch, cp * kv_heads, 1, c.dim)
            cp_ids = torch.arange(cp, device=position_ids.device).view(1, cp, 1)
            live = (cp_ids == cp_way[:, None, None]).expand(
                batch, cp, kv_heads
            ).reshape(batch, cp * kv_heads, 1)
            invalid = torch.iinfo(torch.int32).max
            block_ids = torch.arange(batch, device=position_ids.device).view(
                batch, 1, 1
            ).expand_as(live)
            block_ids = torch.where(live, block_ids, torch.full_like(block_ids, invalid))
            addr = local_slots[:, None, None].expand(
                batch, cp, kv_heads
            ).reshape(batch, cp * kv_heads, 1).to(torch.int32)
            key_cache = cache.scatter_cp(key_cache, block_ids, addr, updates_key)
            value_cache = cache.scatter_cp(value_cache, block_ids, addr, updates_value)
        elif standard_folded_cache and (
            c.attn_cp == 1 or self.use_owner_local_attention_cp
        ):
            # Token-striped folded rows let QAIC gather selected MXINT8 cache
            # entries before dequantization. With CP>1 each row gathers only
            # the tokens it owns and attention merges compact statistics.
            key_cache = cache.scatter_sharded_cache(
                key_cache,
                position_ids,
                key,
                dp=c.attn_dp,
                cp=c.attn_cp,
                rows_per_dp_way=c.kv_heads,
            )
            value_cache = cache.scatter_sharded_cache(
                value_cache,
                position_ids,
                value,
                dp=c.attn_dp,
                cp=c.attn_cp,
                rows_per_dp_way=c.kv_heads,
            )
        elif not c.paged_kv:
            # Explicit DP/CP/KV-head rows:
            # [B_local, DP*CP*Hkv, local_context, head_dim].  Only the CP
            # owner is live for the current pooled block; sentinel block IDs
            # leave the other rows untouched without reading the cache.
            cp = c.attn_cp
            dp = c.attn_dp
            kv_heads = c.kv_heads
            batch_local = batch // dp
            pooled_pos = torch.div(position_ids, ratio, rounding_mode="trunc")
            cp_way = pooled_pos.remainder(cp).reshape(batch_local, dp)
            local_slots = (
                torch.div(pooled_pos, cp, rounding_mode="trunc") * ratio
                + position_ids.remainder(ratio)
            ).reshape(batch_local, dp)
            updates_key = key.reshape(batch_local, dp, kv_heads, 1, c.dim)
            updates_value = value.reshape(batch_local, dp, kv_heads, 1, c.dim)
            updates_key = (
                updates_key[:, :, None]
                .expand(batch_local, dp, cp, kv_heads, 1, c.dim)
                .reshape(batch_local, dp * cp * kv_heads, 1, c.dim)
            )
            updates_value = (
                updates_value[:, :, None]
                .expand(batch_local, dp, cp, kv_heads, 1, c.dim)
                .reshape(batch_local, dp * cp * kv_heads, 1, c.dim)
            )
            cp_ids = torch.arange(cp, device=position_ids.device).view(1, 1, cp, 1)
            owner = (cp_ids == cp_way[:, :, None, None]).expand(
                batch_local, dp, cp, kv_heads
            ).reshape(batch_local, dp * cp * kv_heads, 1)
            invalid = torch.iinfo(torch.int32).max
            block_ids = torch.arange(batch_local, device=position_ids.device).view(
                batch_local, 1, 1
            ).expand_as(owner)
            block_ids = torch.where(owner, block_ids, torch.full_like(block_ids, invalid))
            addr = local_slots[:, :, None, None].expand(
                batch_local, dp, cp, kv_heads
            ).reshape(batch_local, dp * cp * kv_heads, 1).to(torch.int32)
            key_cache = cache.scatter_cp(key_cache, block_ids, addr, updates_key)
            value_cache = cache.scatter_cp(value_cache, block_ids, addr, updates_value)

        # Gather selected complete blocks plus one block for the incomplete
        # causal tail. tail_valid masks future entries in that final block.
        blocks = torch.cat(
            (selected, complete.clamp(max=pooled_context - 1)[:, None]), -1
        )
        tail_valid = (
            (complete * ratio)[:, None]
            + torch.arange(ratio, device=block.device)[None]
        ) <= position_ids[:, None]
        attention_mask = torch.cat(
            (valid[:, :, None].expand(-1, -1, ratio).flatten(1), tail_valid), -1
        )
        token_positions = (
            blocks[:, :, None] * ratio
            + torch.arange(ratio, device=blocks.device)[None, None, :]
        ).reshape(batch, (k + 1) * ratio)
        # With split attention-CP caches, keep each CP shard's gathered K/V
        # local.  Stacking the shards before attention forces QAIC to move the
        # full selected K/V payload to one graph slice.  The distributed path
        # below reduces only attention statistics/results across CP shards.
        distributed_gather = None
        row_gather_attention = False
        if c.paged_kv:
            dp, cp, kv_heads = c.attn_dp, c.attn_cp, c.kv_heads
            batch_local = batch // dp
            gather_len = blocks.shape[-1]
            table_local = block_table.permute(1, 0, 2)
            block_group = torch.div(blocks, cp, rounding_mode="trunc")
            block_way = blocks.remainder(cp)
            selected_key_groups = []
            selected_value_groups = []
            for local_idx in range(batch_local):
                physical = torch.gather(
                    table_local[local_idx],
                    1,
                    block_group.reshape(batch_local, dp, gather_len)[local_idx],
                ).transpose(0, 1)
                block_ids = physical[:, :, None, None].expand(
                    gather_len, dp, cp, kv_heads
                ).reshape(gather_len, dp * cp * kv_heads).to(torch.int32)
                gathered_keys = cache.gather_paged_kv(key_cache, block_ids)
                gathered_values = cache.gather_paged_kv(value_cache, block_ids)
                gathered_keys = gathered_keys.reshape(
                    dp, cp, kv_heads, gather_len, ratio, c.dim
                )
                gathered_values = gathered_values.reshape_as(gathered_keys)
                owner = block_way.reshape(batch_local, dp, gather_len)[local_idx]
                owner = owner[:, None, None, :, None, None].expand(
                    dp, 1, kv_heads, gather_len, ratio, c.dim
                )
                selected_key_groups.append(
                    torch.gather(gathered_keys, 1, owner).squeeze(1)
                )
                selected_value_groups.append(
                    torch.gather(gathered_values, 1, owner).squeeze(1)
                )
            selected_keys = torch.stack(selected_key_groups, dim=0)
            selected_values = torch.stack(selected_value_groups, dim=0)
            selected_keys = selected_keys.reshape(
                batch, kv_heads, gather_len, ratio, c.dim
            ).permute(0, 2, 3, 1, 4).reshape(
                batch, gather_len, ratio, cache_width
            )
            selected_values = selected_values.reshape(
                batch, kv_heads, gather_len, ratio, c.dim
            ).permute(0, 2, 3, 1, 4).reshape(
                batch, gather_len, ratio, cache_width
            )
        elif not c.paged_kv and c.cache_layout == "batch_major":
            cp, kv_heads = c.attn_cp, c.kv_heads
            gather_len = blocks.shape[-1]
            block_way = blocks.remainder(cp)
            block_local = torch.div(blocks, cp, rounding_mode="trunc")
            local_context = c.context // (ratio * cp)
            key_blocks = key_cache.reshape(
                batch, cp * kv_heads, local_context, ratio, c.dim
            )
            value_blocks = value_cache.reshape_as(key_blocks)
            gather_rows = block_local[:, None, :].expand(
                batch, cp * kv_heads, gather_len
            ).unsqueeze(-1).to(torch.int64)
            gathered_keys = cache.gather_blocks(key_blocks, gather_rows)
            gathered_values = cache.gather_blocks(value_blocks, gather_rows)
            gathered_keys = gathered_keys.reshape(
                batch, cp, kv_heads, gather_len, ratio, c.dim
            )
            gathered_values = gathered_values.reshape_as(gathered_keys)
            owner = block_way[:, None, :, None, None, None].expand(
                batch, 1, gather_len, kv_heads, ratio, c.dim
            ).permute(0, 1, 3, 2, 4, 5)
            selected_keys = torch.gather(gathered_keys, 1, owner).squeeze(1)
            selected_values = torch.gather(gathered_values, 1, owner).squeeze(1)
            selected_keys = selected_keys.permute(0, 2, 3, 1, 4).reshape(
                batch, gather_len, ratio, cache_width
            )
            selected_values = selected_values.permute(0, 2, 3, 1, 4).reshape(
                batch, gather_len, ratio, cache_width
            )
        elif not c.paged_kv and c.split_cache_cp:
            cp, dp, kv_heads = c.attn_cp, c.attn_dp, c.kv_heads
            batch_local = batch // dp
            gather_len = blocks.shape[-1]

            block_way = blocks.remainder(cp).reshape(batch_local, dp, gather_len)
            block_local = torch.div(blocks, cp, rounding_mode="trunc")
            local_context = c.context // (ratio * cp)

            gathered_key_ways = []
            gathered_value_ways = []
            gather_idx = (
                block_local.reshape(batch_local, dp, 1, gather_len, 1)
                .expand(batch_local, dp, kv_heads, gather_len, 1)
                .reshape(batch_local, dp * kv_heads, gather_len, 1)
                .to(torch.int64)
            )

            for way in range(cp):
                key_blocks = key_cache[way].reshape(
                    batch_local, dp * kv_heads, local_context, ratio, c.dim
                )
                value_blocks = value_cache[way].reshape_as(key_blocks)
                gathered_key_ways.append(
                    cache.gather_blocks(key_blocks, gather_idx).reshape(
                        batch_local, dp, kv_heads, gather_len, ratio, c.dim
                    )
                )
                gathered_value_ways.append(
                    cache.gather_blocks(value_blocks, gather_idx).reshape(
                        batch_local, dp, kv_heads, gather_len, ratio, c.dim
                    )
                )

            distributed_gather = (gathered_key_ways, gathered_value_ways, block_way)
        elif standard_folded_cache and (
            c.attn_cp == 1 or self.use_owner_local_attention_cp
        ):
            selected_keys, row_valid = cache.gather_sharded_cache_local(
                key_cache,
                token_positions,
                attention_mask,
                dp=c.attn_dp,
                cp=c.attn_cp,
                rows_per_dp_way=c.kv_heads,
            )
            selected_values, _ = cache.gather_sharded_cache_local(
                value_cache,
                token_positions,
                attention_mask,
                dp=c.attn_dp,
                cp=c.attn_cp,
                rows_per_dp_way=c.kv_heads,
            )
            if c.attn_cp == 1:
                selected_keys = selected_keys[:, :, 0].reshape(
                    batch, c.kv_heads, (k + 1) * ratio, c.dim
                )
                selected_values = selected_values[:, :, 0].reshape_as(selected_keys)
                row_gather_attention = True
        else:
            cp = c.attn_cp
            dp = c.attn_dp
            kv_heads = c.kv_heads
            batch_local = batch // dp
            gather_len = blocks.shape[-1]
            block_way = blocks.remainder(cp).to(torch.int32).reshape(
                batch_local, dp, gather_len
            )
            block_local = torch.div(blocks, cp, rounding_mode="trunc").to(torch.int32)
            local_context = c.context // (ratio * cp)
            key_blocks = key_cache.reshape(
                batch_local, dp * cp * kv_heads, local_context, ratio, c.dim
            )
            value_blocks = value_cache.reshape_as(key_blocks)
            gather_rows = block_local.reshape(batch_local, dp, 1, gather_len, 1).expand(
                batch_local, dp, cp * kv_heads, gather_len, 1
            ).reshape(batch_local, dp * cp * kv_heads, gather_len, 1).to(torch.int64)
            gathered_keys = cache.gather_blocks(key_blocks, gather_rows)
            gathered_values = cache.gather_blocks(value_blocks, gather_rows)
            gathered_keys = gathered_keys.reshape(
                batch_local, dp, cp, kv_heads, gather_len, ratio, c.dim
            )
            gathered_values = gathered_values.reshape_as(gathered_keys)
            owner = block_way[:, :, None, :, None, None, None].expand(
                batch_local, dp, 1, gather_len, kv_heads, ratio, c.dim
            ).permute(0, 1, 2, 4, 3, 5, 6)
            selected_keys = torch.gather(gathered_keys, 2, owner).squeeze(2)
            selected_values = torch.gather(gathered_values, 2, owner).squeeze(2)
            selected_keys = selected_keys.reshape(
                batch, kv_heads, gather_len, ratio, c.dim
            ).permute(0, 2, 3, 1, 4).reshape(
                batch, gather_len, ratio, cache_width
            )
            selected_values = selected_values.reshape(
                batch, kv_heads, gather_len, ratio, c.dim
            ).permute(0, 2, 3, 1, 4).reshape(
                batch, gather_len, ratio, cache_width
            )
        query = query.reshape(batch, c.kv_heads, c.heads // c.kv_heads, c.dim)
        if distributed_gather is not None:
            # Compute softmax attention as a CP reduction.  Each CP shard
            # retains and gathers only its own K/V cache; only max, sumexp,
            # and the small attended result cross the CP boundary.
            gathered_key_ways, gathered_value_ways, block_way = distributed_gather
            way_logits = []
            way_values = []
            way_masks = []
            global_max = None
            for way in range(c.attn_cp):
                way_keys = gathered_key_ways[way].reshape(
                    batch, c.kv_heads, (k + 1) * ratio, c.dim
                )
                way_values.append(
                    gathered_value_ways[way].reshape(
                        batch, c.kv_heads, (k + 1) * ratio, c.dim
                    )
                )
                owner = block_way.eq(way).reshape(batch, k + 1)
                way_mask = owner[:, :, None].expand(
                    batch, k + 1, ratio
                ).reshape(batch, (k + 1) * ratio)
                way_mask = way_mask & attention_mask
                way_masks.append(way_mask[:, None, None])
                logits = (query @ way_keys.transpose(-1, -2)) / math.sqrt(c.dim)
                masked_logits = (
                    logits * way_mask[:, None, None].to(logits.dtype)
                    + (~way_mask[:, None, None]).to(logits.dtype) * MASKED_LOGIT
                ).float()
                way_logits.append(masked_logits)
                way_max = masked_logits.amax(-1, keepdim=True)
                global_max = way_max if global_max is None else torch.maximum(
                    global_max, way_max
                )

            exp_ways = [torch.exp(logits - global_max) for logits in way_logits]
            denominator = None
            for exp_logits in exp_ways:
                term = exp_logits.sum(-1, keepdim=True)
                denominator = term if denominator is None else denominator + term

            attended = None
            for exp_logits, values, mask in zip(exp_ways, way_values, way_masks):
                probabilities = (exp_logits / denominator).to(query.dtype)
                partial_output = probabilities @ values
                attended = (
                    partial_output
                    if attended is None
                    else attended + partial_output
                )
            output = attended.reshape(batch, 1, c.heads * c.dim)
        elif owner_local_attention:
            output = _two_stage_attention_merge(
                query,
                selected_keys,
                selected_values,
                row_valid,
            )
        else:
            if not row_gather_attention:
                selected_keys = selected_keys.reshape(
                    batch, k + 1, ratio, c.kv_heads, c.dim
                ).permute(0, 3, 1, 2, 4).reshape(
                    batch, c.kv_heads, (k + 1) * ratio, c.dim
                )
                selected_values = selected_values.reshape(
                    batch, k + 1, ratio, c.kv_heads, c.dim
                ).permute(0, 3, 1, 2, 4).reshape(
                    batch, c.kv_heads, (k + 1) * ratio, c.dim
                )
            logits = (query @ selected_keys.transpose(-1, -2)) / math.sqrt(c.dim)
            mask = attention_mask[:, None, None]
            probabilities = (
                logits * mask.to(logits.dtype)
                + (~mask).to(logits.dtype) * MASKED_LOGIT
            ).float().softmax(-1).to(query.dtype)
            output = (probabilities @ selected_values).reshape(
                batch, 1, c.heads * c.dim
            )
        output = self.o_proj(output * gate.reshape_as(output).sigmoid())
        if c.split_cache_cp:
            return output, *key_cache, *value_cache, index_cache, partial
        return output, key_cache, value_cache, index_cache, partial
