"""MiniMax-M3 stacked GQA/MSA decode and prefill attention.

Ported from ``msa/minimax_m3_msa_decode_microbench.py`` (only the paths that
have a golden): MSA ``msa_attention_GP`` decode (dense and paged, CP=1/CP>1),
paged MSA prefill with the parallel indexer, and GQA decode/prefill. Cache
reads/writes go through ``MiniMaxM3MSAAttentionCache``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .cache import MiniMaxM3MSAAttentionCache
from .ops import rms_norm

MIN_MASKED_ATTENTION_VALUE = float("-inf")
MASKED_ATTENTION_LOGIT = -3.0e4
_FP16_MAX_VALUE = 65504.0


@dataclass
class Config:
    profile: str
    impl: str
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    index_n_heads: int
    indexer_n_head: int
    index_head_dim: int
    index_block_size: int
    index_topk_blocks: int
    index_local_blocks: int
    ctx_len: int
    norm_eps: float = 1.0e-6
    num_kv_blocks: int = 1
    msa_num_kv_blocks: int | None = None
    indexer_num_blocks: int | None = None
    q_block_size: int = 64
    q_block_chunk: int | None = None
    msa_prefill_q_chunk: int | None = None
    indexer_ql_chunk: int | None = None
    indexer_q_block_size: int | None = None
    q_head_block_chunk: int = 1
    attn_dp: int = 1
    attn_cp: int = 1
    paged_kv: bool = False
    random_slot_ids: bool = False
    page_block_size: int = 128
    kv_pool_gb: float | None = None
    num_physical_blocks: int = 0
    msa_attn_dp: int = 1
    msa_attn_cp: int = 1
    num_msa_attn_physical_blocks: int = 0
    msa_indexer_dp: int = 1
    msa_indexer_cp: int = 1
    msa_indexer_kv_pool_gb: float | None = None
    num_indexer_physical_blocks: int = 0
    indexer_prefill_parallel: bool = True
    indexer_only: bool = False
    num_cores_per_device: int = 16
    skip_kv: bool = True

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def indexer_kv_blocks(self) -> int:
        return self.num_kv_blocks if self.indexer_num_blocks is None else self.indexer_num_blocks

    @property
    def msa_kv_blocks(self) -> int:
        return self.num_kv_blocks if self.msa_num_kv_blocks is None else self.msa_num_kv_blocks

    @property
    def indexer_q_size(self) -> int:
        return self.q_block_size if self.indexer_q_block_size is None else self.indexer_q_block_size

    @property
    def indexer_q_chunk(self) -> int:
        if self.indexer_ql_chunk is not None:
            return self.indexer_ql_chunk
        return self.q_block_chunk or self.indexer_q_size

    @property
    def msa_q_chunk(self) -> int:
        if self.msa_prefill_q_chunk is not None:
            return self.msa_prefill_q_chunk
        return self.q_block_size

    @property
    def num_blocks(self) -> int:
        return (self.ctx_len + self.index_block_size - 1) // self.index_block_size

    @property
    def selected_blocks(self) -> int:
        return min(self.index_topk_blocks, self.num_blocks)

    @property
    def selected_tokens(self) -> int:
        return self.selected_blocks * self.index_block_size

    def with_pools(self, num_gqa_layers: int) -> "Config":
        """Fill the paged-MSA pool sizes from the GiB budgets (host fp16/mxint8: 2 bytes/elem)."""
        if not self.paged_kv:
            return self
        if num_gqa_layers > 0 or self.msa_indexer_kv_pool_gb is None:
            per_block = self.attn_dp * self.attn_cp * self.num_kv_heads * self.page_block_size * self.head_dim * 2
            self.num_physical_blocks = int(self.kv_pool_gb * 2**30) // per_block
        if self.msa_indexer_kv_pool_gb is None:
            self.num_indexer_physical_blocks = self.num_physical_blocks
        else:
            per_block = (
                self.msa_indexer_dp
                * self.msa_indexer_cp
                * self.indexer_n_head
                * self.page_block_size
                * self.index_head_dim
                * 2
            )
            self.num_indexer_physical_blocks = int(self.msa_indexer_kv_pool_gb * 2**30) // per_block
        if not self.indexer_only:
            self.num_msa_attn_physical_blocks = (
                self.num_physical_blocks if num_gqa_layers > 0 else self.num_indexer_physical_blocks
            )
        return self


PROFILES: dict[str, dict[str, Any]] = {
    "tiny": dict(
        profile="tiny",
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        rotary_dim=16,
        rope_theta=5_000_000.0,
        index_n_heads=2,
        indexer_n_head=1,
        index_head_dim=16,
        index_block_size=8,
        index_topk_blocks=4,
        index_local_blocks=1,
    ),
    "m3": dict(
        profile="m3",
        hidden_size=6144,
        num_heads=64,
        num_kv_heads=4,
        head_dim=128,
        rotary_dim=64,
        rope_theta=5_000_000.0,
        index_n_heads=4,
        indexer_n_head=1,
        index_head_dim=128,
        index_block_size=128,
        index_topk_blocks=16,
        index_local_blocks=1,
    ),
}


def _scalar_like(reference: torch.Tensor, value: int | float) -> torch.Tensor:
    return torch.tensor(value, dtype=reference.dtype, device=reference.device)


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def update_running_softmax(
    current_max: torch.Tensor,
    attn_weights_block: torch.Tensor,
    current_denominator: torch.Tensor,
    output: torch.Tensor,
    value_block: torch.Tensor,
    skip_kv: bool = False,
    skip_future: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    previous_max = current_max
    updated_max = torch.max(previous_max, attn_weights_block.max(dim=-1).values)
    delta_max = previous_max - updated_max
    current_exp = torch.exp(attn_weights_block - updated_max.unsqueeze(-1))
    updated_denominator = current_denominator * torch.exp(delta_max) + current_exp.sum(dim=-1)
    if value_block.ndim == current_exp.ndim + 1:
        queries_per_group = value_block.shape[-3]
        heads_per_query = current_exp.shape[-2] // queries_per_group
        current_exp_grouped = current_exp.reshape(
            *current_exp.shape[:-2], queries_per_group, heads_per_query, current_exp.shape[-1]
        )
        value_update = torch.matmul(current_exp_grouped, value_block).reshape(
            *current_exp.shape[:-2], current_exp.shape[-2], value_block.shape[-1]
        )
    else:
        value_update = torch.matmul(current_exp, value_block)
    updated_output = output * torch.exp(delta_max.unsqueeze(-1)) + value_update
    if skip_kv and (torch.compiler.is_exporting()):
        updated_max = torch.where(skip_future, previous_max, updated_max)
        updated_denominator = torch.where(skip_future, current_denominator, updated_denominator)
        updated_output = torch.where(skip_future.unsqueeze(-1), output, updated_output)
    return (updated_max, updated_denominator, updated_output)


class MiniMaxM3SparseDecodeAttention(MiniMaxM3MSAAttentionCache, nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = GemmaRMSNorm(config.head_dim, config.norm_eps)
        self.k_norm = GemmaRMSNorm(config.head_dim, config.norm_eps)
        indexer_dp = config.msa_indexer_dp
        index_q_out = config.index_n_heads * config.index_head_dim
        index_k_out = config.indexer_n_head * config.index_head_dim
        self.index_q_proj = nn.Linear(config.hidden_size, indexer_dp * index_q_out, bias=False)
        self.index_k_proj = nn.Linear(config.hidden_size, indexer_dp * index_k_out, bias=False)
        with torch.no_grad():
            q_weight = self.index_q_proj.weight[:index_q_out].clone()
            k_weight = self.index_k_proj.weight[:index_k_out].clone()
            self.index_q_proj.weight.copy_(q_weight.repeat(indexer_dp, 1))
            self.index_k_proj.weight.copy_(k_weight.repeat(indexer_dp, 1))
        self.index_q_norm = GemmaRMSNorm(config.index_head_dim, config.norm_eps)
        self.index_k_norm = GemmaRMSNorm(config.index_head_dim, config.norm_eps)
        self.register_buffer(
            "inv_freq",
            1.0 / config.rope_theta ** (torch.arange(0, config.rotary_dim, 2).float() / config.rotary_dim),
            persistent=False,
        )
        indexer_paged_local_pos = None
        if config.paged_kv:
            logical_kv_block_size = config.ctx_len // config.indexer_kv_blocks
            kv_block_size = logical_kv_block_size // config.msa_indexer_cp
            num_groups_blk = kv_block_size // config.page_block_size
            groups_per_core = num_groups_blk // config.num_cores_per_device
            T_h = groups_per_core * config.page_block_size
            rows = config.msa_indexer_dp * config.msa_indexer_cp * config.indexer_n_head
            t_idx = torch.arange(T_h).view(1, 1, 1, 1, T_h)
            core_idx = torch.arange(config.num_cores_per_device).view(1, 1, config.num_cores_per_device, 1, 1)
            cp_idx = torch.arange(config.msa_indexer_cp).view(1, config.msa_indexer_cp, 1, 1, 1)
            local_group = t_idx // config.page_block_size * config.num_cores_per_device + core_idx
            t_page = t_idx // config.page_block_size
            t_offset = t_idx - t_page * config.page_block_size
            local_pos = (
                local_group * (config.msa_indexer_cp * config.page_block_size)
                + cp_idx * config.page_block_size
                + t_offset
            ).to(torch.int32)
            indexer_paged_local_pos = (
                local_pos.view(1, 1, config.msa_indexer_cp, 1, config.num_cores_per_device, 1, T_h)
                .expand(
                    1,
                    config.msa_indexer_dp,
                    config.msa_indexer_cp,
                    config.indexer_n_head,
                    config.num_cores_per_device,
                    1,
                    T_h,
                )
                .reshape(1, rows, config.num_cores_per_device, 1, T_h)
            )
        self.register_buffer("indexer_paged_local_pos", indexer_paged_local_pos)
        self.last_block_indices: torch.Tensor | None = None

    def _rope(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.float()
        freqs = position_ids.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos().to(dtype), emb.sin().to(dtype))

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        rotary_dim = cos.shape[-1]
        rotated = x[..., :rotary_dim]
        passthrough = x[..., rotary_dim:]
        if cos.ndim != x.ndim:
            head_axis = x.ndim - 3
            cos = cos.unsqueeze(head_axis)
            sin = sin.unsqueeze(head_axis)
        rotated = rotated * cos + rotate_half(rotated) * sin
        return torch.cat((rotated, passthrough), dim=-1)

    def _project_index_q_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query_len = hidden_states.shape[1]
        q_proj_seq_chunk = 512
        num_q_chunks = (query_len + q_proj_seq_chunk - 1) // q_proj_seq_chunk
        base_q_chunk = query_len // num_q_chunks
        extra_q_tokens = query_len % num_q_chunks
        q_proj_chunks: list[torch.Tensor] = []
        q_start = 0
        for q_chunk_idx in range(num_q_chunks):
            q_chunk_len = base_q_chunk + (q_chunk_idx < extra_q_tokens)
            q_end = q_start + q_chunk_len
            q_proj_chunks.append(self.index_q_proj(hidden_states[:, q_start:q_end, :]))
            q_start = q_end
        return torch.cat(q_proj_chunks, dim=1)

    def _read_blocked_msa_k_dp(
        self, index_key_cache: torch.Tensor, position_ids_dp: torch.Tensor, start_index: int, end_index: int
    ) -> torch.Tensor:
        cfg = self.config
        batch_local, rows, _, _ = index_key_cache.shape
        dp = position_ids_dp.shape[1]
        cp = cfg.msa_indexer_cp
        hkv = cfg.indexer_n_head
        block_len = end_index - start_index
        pos_max = position_ids_dp.max(dim=-1).values
        block_size = cfg.index_block_size
        cycle_size = block_size * cp
        gather_limit = (
            (pos_max // cycle_size * block_size + block_size - 1)
            .view(batch_local, dp, 1)
            .expand(batch_local, dp, cp * hkv)
            .reshape(batch_local, rows, 1)
        )
        ctx_indices = torch.arange(start=start_index, end=end_index, device=index_key_cache.device)[None, None, :]
        invalid_mask = ctx_indices > gather_limit
        invalid_idx_value = torch.iinfo(torch.int32).max if torch.compiler.is_exporting() else 0
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices).to(torch.int32)
        ctx_indices = ctx_indices.expand(batch_local, rows, block_len)
        return self.gather_blocked_kv_dp(index_key_cache, ctx_indices)

    def _read_blocked_k_core(
        self,
        index_key_cache: torch.Tensor,
        position_ids_dp: torch.Tensor,
        start_index: int,
        end_index: int,
        num_cores: int,
        position_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather one non-paged indexer KV block as [B, C, KV/C, D]."""
        cfg = self.config
        batch_local, rows, _, head_dim = index_key_cache.shape
        if rows != 1 or position_ids_dp.shape[1] != 1:
            raise NotImplementedError(
                "Core-split MSA prefill gather requires one indexer KV row and DP=CP=1."
            )
        block_len = end_index - start_index
        if block_len < num_cores or block_len % num_cores:
            raise ValueError("Core-split MSA prefill KV block must divide evenly across cores.")
        tokens_per_core = block_len // num_cores
        block_size = cfg.index_block_size
        if start_index % block_size or end_index % block_size or tokens_per_core % block_size:
            raise ValueError("Core-split MSA prefill KV gather must align to index_block_size.")
        blocks_per_core = tokens_per_core // block_size
        core_block_ids = (
            torch.arange(num_cores, device=index_key_cache.device).view(1, num_cores, 1) * blocks_per_core
            + torch.arange(blocks_per_core, device=index_key_cache.device).view(1, 1, blocks_per_core)
            + start_index // block_size
        ).to(torch.int32)
        if position_max is None:
            position_max = position_ids_dp.max(dim=-1).values
        gather_block_limit = position_max[:, :, None] // block_size
        invalid_mask = core_block_ids > gather_block_limit
        invalid_idx_value = torch.iinfo(torch.int32).max if torch.compiler.is_exporting() else 0
        block_ids = torch.where(
            invalid_mask, _scalar_like(core_block_ids, invalid_idx_value), core_block_ids
        ).to(torch.int32)
        block_ids = block_ids.reshape(batch_local, 1, num_cores * blocks_per_core)
        key_block = self.gather_blocked_kv_blocks(index_key_cache, block_ids, block_size)
        return key_block.reshape(batch_local, 1, num_cores, tokens_per_core, head_dim).squeeze(1)

    def _read_index_paged_kv_dp(self, index_key_cache: torch.Tensor, block_ids_per_page: torch.Tensor) -> torch.Tensor:
        dp, num_pages = block_ids_per_page.shape
        rows = index_key_cache.shape[1]
        rows_per_dp = rows // dp
        block_ids_by_row = (
            block_ids_per_page.transpose(0, 1).unsqueeze(2).expand(num_pages, dp, rows_per_dp).reshape(num_pages, rows)
        )
        return self.gather_paged_kv_dp(index_key_cache, block_ids_by_row.to(torch.int32))

    def _read_msa_prefill_paged_block(
        self, index_key_cache: torch.Tensor, block_table: torch.Tensor, start_index: int, end_index: int
    ) -> torch.Tensor:
        cfg = self.config
        page_size = cfg.page_block_size
        page_start = start_index // page_size
        page_end = end_index // page_size
        page_ids = block_table[:, page_start:page_end].to(torch.int32)
        pages = self.gather_paged_kv_batch_with_heads(index_key_cache, page_ids)
        block_len = end_index - start_index
        return pages[:, :, :block_len]

    def _read_msa_prefill_paged_block_core(
        self, index_key_cache: torch.Tensor, block_table: torch.Tensor, start_index: int, end_index: int, num_cores: int
    ) -> torch.Tensor:
        block = self._read_msa_prefill_paged_block(index_key_cache, block_table, start_index, end_index)
        batch, _, block_len, head_dim = block.shape
        return block[:, 0].reshape(batch, num_cores, block_len // num_cores, head_dim)

    def _select_blocks_prefill_par(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        index_key_cache: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        paged_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        num_cores = cfg.num_cores_per_device
        num_index_heads = cfg.index_n_heads
        dim = cfg.index_head_dim
        num_kv_blocks = max(1, cfg.indexer_kv_blocks)
        q_block_size = cfg.indexer_q_size
        q_block_chunk = cfg.indexer_q_chunk
        cores_per_q_head = max(1, num_cores // num_index_heads)
        idx_q = self._project_index_q_prefill(hidden_states)[..., : num_index_heads * dim]
        idx_q = idx_q.view(batch, query_len, num_index_heads, dim).transpose(1, 2)
        idx_k = self.index_k_proj(hidden_states)[..., : cfg.indexer_n_head * dim]
        idx_k = idx_k.view(batch, query_len, cfg.indexer_n_head, dim).transpose(1, 2)
        idx_q = self.index_q_norm(idx_q)
        idx_k = self.index_k_norm(idx_k)
        idx_q = self._apply_rope(idx_q, cos[..., :dim], sin[..., :dim])
        idx_k = self._apply_rope(idx_k, cos[..., :dim], sin[..., :dim])
        index_key_cache = (
            self.ctx_scatter(index_key_cache, position_ids.to(torch.int32), idx_k)
            if paged_block_table is None
            else self._write_msa_paged_prefill_cache(
                index_key_cache, idx_k, position_ids, paged_block_table.unsqueeze(0)
            )
        )
        position_max = position_ids.unsqueeze(1).max(dim=-1).values
        kv_block_size = cfg.ctx_len // num_kv_blocks
        index_block_size = cfg.index_block_size
        tokens_per_q_head = kv_block_size // cores_per_q_head
        blocks_per_q_head = tokens_per_q_head // index_block_size
        num_blocks = cfg.ctx_len // index_block_size
        mask_value = MASKED_ATTENTION_LOGIT
        topk = min(cfg.selected_blocks, num_blocks)
        topk_indices_chunks: list[torch.Tensor] = []
        topk_valid_chunks: list[torch.Tensor] = []
        for q_chunk_start in range(0, query_len, q_block_chunk):
            q_chunk_end = min(q_chunk_start + q_block_chunk, query_len)
            query_chunk = q_chunk_end - q_chunk_start
            score_blocks: list[list[torch.Tensor]] = [[] for _ in range(query_chunk // q_block_size)]
            q_chunk = idx_q[:, :, q_chunk_start:q_chunk_end]
            query_positions = position_ids[:, q_chunk_start:q_chunk_end]
            q_chunk_position = query_positions.max(dim=-1).values
            is_export = torch.compiler.is_exporting()
            for block_idx in range(num_kv_blocks):
                start = block_idx * kv_block_size
                end = start + kv_block_size
                block_skip_future_chunk = torch.tensor(start, device=hidden_states.device) > q_chunk_position
                if cfg.skip_kv and (not is_export) and bool(block_skip_future_chunk.all().item()):
                    masked_block = torch.full(
                        (batch, num_index_heads, cores_per_q_head, q_block_size, blocks_per_q_head),
                        mask_value,
                        dtype=torch.float32,
                        device=hidden_states.device,
                    )
                    for q_block_idx in range(len(score_blocks)):
                        score_blocks[q_block_idx].extend(
                            (masked_block.clone() for _ in range(block_idx, num_kv_blocks))
                        )
                    break
                if paged_block_table is None:
                    key_block = self._read_blocked_k_core(
                        index_key_cache, position_ids.unsqueeze(1), start, end, num_cores, position_max
                    )
                else:
                    key_block = self._read_msa_prefill_paged_block_core(
                        index_key_cache, paged_block_table, start, end, num_cores
                    )
                key_block = key_block.float()
                tokens_per_core = kv_block_size // num_cores
                key_core = (
                    key_block.unsqueeze(1)
                    .expand(batch, num_index_heads, num_cores, tokens_per_core, dim)
                    .reshape(batch, num_index_heads, cores_per_q_head, kv_block_size // cores_per_q_head, dim)
                )
                for q_block_idx, q_start in enumerate(range(0, query_chunk, q_block_size)):
                    q_block = q_chunk[:, :, q_start : q_start + q_block_size]
                    q_core = q_block.unsqueeze(2).expand(batch, num_index_heads, cores_per_q_head, q_block_size, dim)
                    scores = torch.matmul(q_core.float(), key_core.transpose(-1, -2).float())
                    q_positions = query_positions[:, q_start : q_start + q_block_size]
                    block_skip_future = (
                        torch.tensor(start, device=hidden_states.device) > q_positions[:, None, None, :, None]
                    )
                    scores = scores.reshape(
                        batch, num_index_heads, cores_per_q_head, q_block_size, blocks_per_q_head, index_block_size
                    )
                    scores = scores.amax(dim=-1)
                    if cfg.skip_kv:
                        scores = torch.where(block_skip_future, _scalar_like(scores, mask_value), scores)
                    score_blocks[q_block_idx].append(scores)
            for q_block_idx in range(len(score_blocks)):
                scores = (
                    torch.stack(score_blocks[q_block_idx], dim=2)
                    .permute(0, 1, 4, 2, 3, 5)
                    .reshape(batch, num_index_heads, q_block_size, num_blocks)
                )
                q_positions = query_positions[:, q_block_idx * q_block_size : (q_block_idx + 1) * q_block_size]
                block_ids = torch.arange(num_blocks, device=hidden_states.device).view(1, 1, 1, num_blocks)
                mask_score = torch.tensor(mask_value, dtype=scores.dtype, device=scores.device)
                local_score = torch.tensor(-mask_value, dtype=scores.dtype, device=scores.device)
                block_skip_future = block_ids * index_block_size > q_positions[:, None, :, None]
                scores = torch.where(block_skip_future, mask_score, scores)
                for local_offset in range(cfg.index_local_blocks):
                    local_block = q_positions // index_block_size - local_offset
                    local_block = torch.where(local_block >= 0, local_block, torch.zeros_like(local_block))
                    scores = torch.where(block_ids == local_block[:, None, :, None], local_score, scores)
                top_scores, top_order = torch.topk(scores, k=topk, dim=-1)
                topk_indices_chunks.append(torch.gather(block_ids.expand_as(scores).to(torch.int32), -1, top_order))
                block_valid = top_scores > mask_value / 2
                if cfg.skip_kv:
                    top_block_skip_future = torch.gather(block_skip_future.expand_as(scores), -1, top_order)
                    block_valid = block_valid & ~top_block_skip_future
                topk_valid_chunks.append(block_valid)
        block_indices = torch.cat(topk_indices_chunks, dim=2)
        block_valid = torch.cat(topk_valid_chunks, dim=2)
        safe_indices = torch.where(block_valid, block_indices, torch.zeros_like(block_indices)).to(torch.int32)
        if not torch.compiler.is_exporting():
            self.last_block_indices = safe_indices.detach()
        return (safe_indices, block_valid, index_key_cache)

    def _select_blocks(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        index_key_cache: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        dp = cfg.msa_indexer_dp
        cp = cfg.msa_indexer_cp
        hkv = cfg.indexer_n_head
        num_index_heads = cfg.index_n_heads
        dim = cfg.index_head_dim
        num_cores = cfg.num_cores_per_device
        num_kv_blocks = max(1, cfg.indexer_kv_blocks)
        assert cfg.index_local_blocks == 1, "MSA GP decode requires index_local_blocks == 1."
        batch_local = batch // dp
        rows = dp * cp * hkv
        cache_slots = cfg.ctx_len // cp
        cache_block_size = cache_slots // num_kv_blocks
        tokens_per_core = cache_block_size // num_cores
        blocks_per_core = tokens_per_core // cfg.index_block_size
        position_ids_dp = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)
        cos_dp = cos.view(dp, batch_local, query_len, cos.shape[-1]).permute(1, 0, 2, 3)
        sin_dp = sin.view(dp, batch_local, query_len, sin.shape[-1]).permute(1, 0, 2, 3)
        idx_q_all = (
            self.index_q_proj(hidden_states)
            .view(dp, batch_local, query_len, dp, num_index_heads, dim)
            .permute(1, 0, 2, 3, 4, 5)
        )
        idx_k_all = (
            self.index_k_proj(hidden_states).view(dp, batch_local, query_len, dp, hkv, dim).permute(1, 0, 2, 3, 4, 5)
        )
        q_lane_idx = (
            torch.arange(dp, device=hidden_states.device)
            .view(1, dp, 1, 1, 1, 1)
            .expand(batch_local, dp, query_len, 1, num_index_heads, dim)
        )
        k_lane_idx = (
            torch.arange(dp, device=hidden_states.device)
            .view(1, dp, 1, 1, 1, 1)
            .expand(batch_local, dp, query_len, 1, hkv, dim)
        )
        idx_q = torch.gather(idx_q_all, 3, q_lane_idx).squeeze(3).transpose(2, 3)
        idx_k = torch.gather(idx_k_all, 3, k_lane_idx).squeeze(3).transpose(2, 3)
        idx_q = self.index_q_norm(idx_q)
        idx_k = self.index_k_norm(idx_k)
        idx_q = self._apply_rope(idx_q, cos_dp[..., : cfg.index_head_dim], sin_dp[..., : cfg.index_head_dim])
        idx_k = self._apply_rope(idx_k, cos_dp[..., : cfg.index_head_dim], sin_dp[..., : cfg.index_head_dim])
        k_updates = (
            idx_k.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, query_len, dim)
            .reshape(batch_local, rows, query_len, dim)
        )
        logical_index_block = position_ids_dp // cfg.index_block_size
        live_way = (
            (logical_index_block % cp)[:, :, None, None, :]
            .expand(batch_local, dp, cp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        row_idx = torch.arange(rows, device=hidden_states.device).view(1, rows, 1)
        row_way = row_idx.remainder(cp * hkv) // hkv
        row_live = row_way == live_way
        batch_id = (
            torch.arange(batch_local, device=hidden_states.device)
            .view(batch_local, 1, 1)
            .expand(batch_local, rows, query_len)
        )
        block_id = torch.where(row_live, batch_id, torch.iinfo(torch.int32).max).to(torch.int32)
        addr = (
            (logical_index_block // cp * cfg.index_block_size + position_ids_dp % cfg.index_block_size)
            .to(torch.int32)[:, :, None, None, :]
            .expand(batch_local, dp, cp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        index_key_cache = self.paged_scatter(index_key_cache, block_id, addr, k_updates)
        q_heads_per_kv = num_index_heads // hkv
        ql_eff = q_heads_per_kv * query_len
        q_rows = (
            idx_q.reshape(batch_local, dp, hkv, q_heads_per_kv, query_len, dim)
            .unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, dim)
            .reshape(batch_local, rows, ql_eff, dim)
        )
        q_pos_rows_all = (
            position_ids_dp[:, :, 0]
            .view(batch_local, dp, 1, 1)
            .expand(batch_local, dp, cp, hkv)
            .reshape(batch_local, rows)
        )
        block_ranges: list[tuple[int, int]] = []
        for block_idx in range(num_kv_blocks):
            start = block_idx * cache_block_size
            end = min(start + cache_block_size, cache_slots)
            if start < end:
                block_ranges.append((start, end))
        candidate_score_groups: list[torch.Tensor] = []
        candidate_block_groups: list[torch.Tensor] = []
        local_batch_size = 1
        torch.compiler.is_exporting()

        def make_block_ids_rows(start: int, local: int) -> torch.Tensor:
            block_start = start * cp // cfg.index_block_size
            block_ids_core = (
                block_start
                + torch.arange(cp, device=hidden_states.device).view(cp, 1, 1)
                + torch.arange(num_cores, device=hidden_states.device).view(1, num_cores, 1) * (cp * blocks_per_core)
                + torch.arange(blocks_per_core, device=hidden_states.device).view(1, 1, blocks_per_core) * cp
            )
            return (
                block_ids_core.view(1, 1, cp, 1, num_cores, blocks_per_core)
                .expand(local, dp, cp, hkv, num_cores, blocks_per_core)
                .reshape(local, rows, num_cores, blocks_per_core)
            )

        for batch_start in range(0, batch_local, local_batch_size):
            batch_end = min(batch_start + local_batch_size, batch_local)
            local = batch_end - batch_start
            q_local = q_rows[batch_start:batch_end]
            key_local = index_key_cache[batch_start:batch_end]
            pos_local = position_ids_dp[batch_start:batch_end]
            q_5d = q_local.unsqueeze(2).expand(local, rows, num_cores, ql_eff, dim)
            device_block_score_groups: list[torch.Tensor] = []
            device_block_id_groups: list[torch.Tensor] = []
            q_block_rows = (
                (position_ids_dp[batch_start:batch_end, :, 0] // cfg.index_block_size)
                .view(local, dp, 1, 1)
                .expand(local, dp, cp, hkv)
                .reshape(local, rows)
            )

            def compute_block_scores(block_skip_future: torch.Tensor, key_5d: torch.Tensor) -> torch.Tensor:
                score_block = torch.matmul(q_5d.float(), key_5d.transpose(-1, -2).float())
                score_block = score_block.view(local, rows, num_cores, ql_eff, blocks_per_core, cfg.index_block_size)
                block_scores_core = score_block.amax(dim=-1)
                return block_scores_core

            prologue_start, prologue_end = block_ranges[0]
            key_flat = self._read_blocked_msa_k_dp(key_local, pos_local, prologue_start, prologue_end)
            key_5d = key_flat.view(local, rows, num_cores, tokens_per_core, dim)
            for block_idx in range(len(block_ranges) - 1):
                start, end = block_ranges[block_idx]
                block_ids_rows = make_block_ids_rows(start, local)
                block_skip_future = (
                    torch.tensor(start * cp, device=hidden_states.device, dtype=q_pos_rows_all.dtype)
                    > q_pos_rows_all[batch_start:batch_end]
                )
                block_scores_core = compute_block_scores(block_skip_future, key_5d)
                if cfg.skip_kv:
                    block_scores_core = torch.where(
                        block_skip_future.view(local, rows, 1, 1, 1),
                        _scalar_like(block_scores_core, MASKED_ATTENTION_LOGIT),
                        block_scores_core,
                    )
                next_start, next_end = block_ranges[block_idx + 1]
                next_key_flat = self._read_blocked_msa_k_dp(key_local, pos_local, next_start, next_end)
                key_5d = next_key_flat.view(local, rows, num_cores, tokens_per_core, dim)
                device_block_score_groups.append(block_scores_core)
                device_block_id_groups.append(block_ids_rows)
            epilogue_idx = len(block_ranges) - 1
            epilogue_start, epilogue_end = block_ranges[epilogue_idx]
            epilogue_block_ids = make_block_ids_rows(epilogue_start, local)
            epilogue_skip_future = (
                torch.tensor(epilogue_start * cp, device=hidden_states.device, dtype=q_pos_rows_all.dtype)
                > q_pos_rows_all[batch_start:batch_end]
            )
            epilogue_scores = compute_block_scores(epilogue_skip_future, key_5d)
            device_block_score_groups.append(epilogue_scores)
            device_block_id_groups.append(epilogue_block_ids)
            device_block_scores = torch.cat(device_block_score_groups, dim=4)
            device_block_ids = (
                torch.cat(device_block_id_groups, dim=3)
                .unsqueeze(3)
                .expand(local, rows, num_cores, ql_eff, num_kv_blocks * blocks_per_core)
            )
            future_block = device_block_ids >= q_block_rows.view(local, rows, 1, 1, 1)
            device_block_scores = torch.where(
                future_block, _scalar_like(device_block_scores, MASKED_ATTENTION_LOGIT), device_block_scores
            )
            core_topk = min(cfg.selected_blocks, device_block_scores.shape[-1])
            core_topk_scores, core_topk_indices = torch.topk(device_block_scores, k=core_topk, dim=-1)
            core_topk_block_ids = torch.gather(device_block_ids, -1, core_topk_indices)
            core_candidate_scores = core_topk_scores.permute(0, 1, 3, 2, 4).reshape(
                local, rows, ql_eff, num_cores * core_topk
            )
            core_candidate_block_ids = core_topk_block_ids.permute(0, 1, 3, 2, 4).reshape(
                local, rows, ql_eff, num_cores * core_topk
            )
            device_topk = min(cfg.selected_blocks, core_candidate_scores.shape[-1])
            device_topk_scores, device_topk_indices = torch.topk(core_candidate_scores, k=device_topk, dim=-1)
            device_topk_block_ids = torch.gather(core_candidate_block_ids, -1, device_topk_indices)
            candidate_score_groups.append(device_topk_scores)
            candidate_block_groups.append(device_topk_block_ids)
        device_candidate_width = device_topk
        candidate_scores = (
            torch.cat(candidate_score_groups, dim=0)
            .view(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, device_candidate_width)
            .permute(0, 1, 3, 4, 5, 2, 6)
            .reshape(batch, num_index_heads, cp * device_candidate_width)
        )
        candidate_block_indices = (
            torch.cat(candidate_block_groups, dim=0)
            .view(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, device_candidate_width)
            .permute(0, 1, 3, 4, 5, 2, 6)
            .reshape(batch, num_index_heads, cp * device_candidate_width)
        )
        topk_scores, candidate_topk = torch.topk(candidate_scores, k=cfg.selected_blocks, dim=-1)
        block_indices = torch.gather(candidate_block_indices, -1, candidate_topk)
        topk_scores = topk_scores.view(batch_local, dp, hkv, q_heads_per_kv, query_len, cfg.selected_blocks)
        block_indices = block_indices.view(batch_local, dp, hkv, q_heads_per_kv, query_len, cfg.selected_blocks)
        current_block_ids = (
            (position_ids.view(dp, batch_local, query_len).permute(1, 0, 2) // cfg.index_block_size)
            .view(batch_local, dp, 1, 1, query_len, 1)
            .expand(batch_local, dp, hkv, q_heads_per_kv, query_len, 1)
            .to(block_indices.dtype)
        )
        block_indices = torch.cat([block_indices[..., :-1], current_block_ids], dim=-1)
        topk_scores = torch.cat([topk_scores[..., :-1], torch.zeros_like(topk_scores[..., -1:])], dim=-1)
        if cp > 1:
            topk_scores = (
                topk_scores.unsqueeze(2)
                .expand(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, cfg.selected_blocks)
                .permute(1, 0, 2, 3, 4, 5, 6)
                .reshape(batch, cp, num_index_heads, query_len, cfg.selected_blocks)
            )
            block_indices = (
                block_indices.unsqueeze(2)
                .expand(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, cfg.selected_blocks)
                .permute(1, 0, 2, 3, 4, 5, 6)
                .reshape(batch, cp, num_index_heads, query_len, cfg.selected_blocks)
            )
        block_valid = topk_scores > MASKED_ATTENTION_LOGIT / 2
        block_indices = block_indices[..., 0, :]
        block_valid = block_valid[..., 0, :]
        safe_indices = torch.where(block_valid, block_indices, torch.zeros_like(block_indices)).to(torch.int32)
        if not torch.compiler.is_exporting():
            self.last_block_indices = block_indices.detach()
        return (safe_indices, block_valid, index_key_cache)

    def _select_blocks_paged(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        index_key_cache: torch.Tensor,
        block_table: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        device = hidden_states.device
        batch, query_len, _ = hidden_states.shape
        dp = cfg.msa_indexer_dp
        cp = cfg.msa_indexer_cp
        hkv = cfg.indexer_n_head
        num_index_heads = cfg.index_n_heads
        dim = cfg.index_head_dim
        num_cores = cfg.num_cores_per_device
        num_kv_blocks = cfg.indexer_kv_blocks
        page_block_size = cfg.page_block_size
        logical_block_size = cfg.index_block_size
        assert cfg.index_local_blocks == 1, "MSA GP decode requires index_local_blocks == 1."
        batch_local = batch // dp
        rows = dp * cp * hkv
        logical_kv_block_size = cfg.ctx_len // num_kv_blocks
        kv_block_size = logical_kv_block_size // cp
        num_groups_blk = kv_block_size // page_block_size
        groups_per_core = num_groups_blk // num_cores
        tokens_per_core = groups_per_core * page_block_size
        (cfg.ctx_len // page_block_size + cp - 1) // cp
        position_ids_dp = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)
        cos_dp = cos.view(dp, batch_local, query_len, cos.shape[-1]).permute(1, 0, 2, 3)
        sin_dp = sin.view(dp, batch_local, query_len, sin.shape[-1]).permute(1, 0, 2, 3)
        idx_q_all = (
            self.index_q_proj(hidden_states)
            .view(dp, batch_local, query_len, dp, num_index_heads, dim)
            .permute(1, 0, 2, 3, 4, 5)
        )
        idx_k_all = (
            self.index_k_proj(hidden_states).view(dp, batch_local, query_len, dp, hkv, dim).permute(1, 0, 2, 3, 4, 5)
        )
        lane_idx = torch.arange(dp, device=device).view(1, dp, 1, 1, 1, 1)
        q_lane_idx = lane_idx.expand(batch_local, dp, query_len, 1, num_index_heads, dim)
        k_lane_idx = lane_idx.expand(batch_local, dp, query_len, 1, hkv, dim)
        idx_q = torch.gather(idx_q_all, 3, q_lane_idx).squeeze(3).transpose(2, 3)
        idx_k = torch.gather(idx_k_all, 3, k_lane_idx).squeeze(3).transpose(2, 3)
        idx_q = self.index_q_norm(idx_q)
        idx_k = self.index_k_norm(idx_k)
        idx_q = self._apply_rope(idx_q, cos_dp[..., :dim], sin_dp[..., :dim])
        idx_k = self._apply_rope(idx_k, cos_dp[..., :dim], sin_dp[..., :dim])
        row_shape = (batch_local, dp, cp, hkv, query_len)
        k_updates = idx_k.unsqueeze(2).expand(*row_shape, dim).reshape(batch_local, rows, query_len, dim)
        addr = (
            (position_ids_dp % page_block_size)
            .to(torch.int32)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(row_shape)
            .reshape(batch_local, rows, query_len)
        )
        logical_page = position_ids_dp // page_block_size
        logical_page_group = (logical_page // cp).to(torch.int64)
        block_id_dp = torch.gather(block_table.permute(1, 0, 2), 2, logical_page_group)
        block_id_dp = block_id_dp.to(torch.int32).unsqueeze(2).unsqueeze(3).expand(row_shape)
        live_way = (logical_page % cp).to(torch.int64)
        way_idx = torch.arange(cp, device=device).view(1, 1, cp, 1, 1)
        row_live = way_idx == live_way.unsqueeze(2).unsqueeze(3)
        row_live = row_live.expand(row_shape)
        block_id = torch.where(row_live, block_id_dp, torch.iinfo(torch.int32).max).reshape(
            batch_local, rows, query_len
        )
        index_key_cache = self.paged_scatter(index_key_cache, block_id, addr, k_updates)
        q_heads_per_kv = num_index_heads // hkv
        ql_eff = q_heads_per_kv * query_len
        q_rows = (
            idx_q.reshape(batch_local, dp, hkv, q_heads_per_kv, query_len, dim)
            .unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, dim)
            .reshape(batch_local, rows, ql_eff, dim)
        )
        group_order = (
            torch.arange(num_groups_blk, device=device).view(groups_per_core, num_cores).transpose(0, 1).reshape(-1)
        )
        pos_rows_all = (
            position_ids_dp.view(batch_local, dp, 1, 1, query_len)
            .expand(row_shape)
            .reshape(batch_local, rows, 1, query_len)
        )
        q_block_all = pos_rows_all // logical_kv_block_size
        q_block_offset_all = pos_rows_all - q_block_all * logical_kv_block_size
        q_page_all = q_block_offset_all // page_block_size
        q_rem_all = q_block_offset_all - q_page_all * page_block_size
        row_way_all = (torch.arange(rows, device=device).remainder(cp * hkv) // hkv).view(1, rows, 1, 1)
        way_rows = row_way_all.reshape(rows)
        pos_dtype = pos_rows_all.dtype
        mask_core_idx = torch.arange(num_cores, dtype=pos_dtype, device=device).view(1, 1, num_cores, 1)
        mask_pos = torch.arange(tokens_per_core, dtype=pos_dtype, device=device).view(1, 1, 1, tokens_per_core)
        block_idx_all = torch.arange(num_kv_blocks, dtype=q_block_all.dtype, device=device).view(1, 1, 1, num_kv_blocks)
        query_after_all = q_block_all > block_idx_all
        query_in_all = q_block_all == block_idx_all
        q_page_local_all = q_page_all // cp
        q_page_way_all = q_page_all - q_page_local_all * cp
        last_page_all = torch.where(
            row_way_all < q_page_way_all,
            q_page_local_all,
            torch.where(row_way_all == q_page_way_all, q_page_local_all, q_page_local_all - 1),
        )
        valid_page_all = last_page_all >= 0
        safe_page_all = torch.where(valid_page_all, last_page_all, torch.zeros_like(last_page_all))
        page_group_all = safe_page_all // num_cores
        page_core_all = safe_page_all - page_group_all * num_cores
        group_start_all = page_group_all * page_block_size
        threshold_before_all = group_start_all + page_block_size - 1
        threshold_current_all = group_start_all + q_rem_all
        partial_page_all = row_way_all == q_page_way_all
        threshold_after_all = torch.where(page_group_all > 0, group_start_all - 1, torch.full_like(group_start_all, -1))
        threshold_current_block_all = torch.where(
            valid_page_all & (mask_core_idx < page_core_all),
            threshold_before_all,
            torch.where(
                valid_page_all & (mask_core_idx == page_core_all),
                torch.where(partial_page_all, threshold_current_all, threshold_before_all),
                torch.where(valid_page_all, threshold_after_all, torch.full_like(threshold_after_all, -1)),
            ),
        )
        threshold_block = threshold_current_block_all.unsqueeze(-1)
        threshold_all = torch.where(
            query_after_all.unsqueeze(2),
            torch.full_like(threshold_block, tokens_per_core - 1),
            torch.where(query_in_all.unsqueeze(2), threshold_block, torch.full_like(threshold_block, -1)),
        )
        candidate_score_groups: list[torch.Tensor] = []
        candidate_block_groups: list[torch.Tensor] = []
        core_candidate_score_groups: list[torch.Tensor] = []
        core_candidate_block_groups: list[torch.Tensor] = []
        device_topk = cfg.selected_blocks
        is_export = torch.compiler.is_exporting()

        def make_paged_group_ids(block_idx: int) -> torch.Tensor:
            return block_idx * num_groups_blk + group_order

        def make_paged_block_ids_rows(block_idx: int) -> torch.Tensor:
            group_ids = make_paged_group_ids(block_idx)
            return group_ids.view(1, 1, num_cores, groups_per_core) * cp + way_rows.view(1, rows, 1, 1)

        for batch_idx in range(batch_local):
            q_local = q_rows[batch_idx : batch_idx + 1]
            q_5d = q_local.unsqueeze(2).expand(1, rows, num_cores, ql_eff, dim)
            bt_local = block_table[:, batch_idx]
            pos_rows_local = pos_rows_all[batch_idx : batch_idx + 1]
            device_block_score_groups: list[torch.Tensor] = []
            device_block_id_groups: list[torch.Tensor] = []

            def skip_future(block_idx: int) -> torch.Tensor:
                block_start = torch.tensor(block_idx * logical_kv_block_size, device=device, dtype=pos_dtype)
                return block_start > pos_rows_local

            def compute_block_scores(
                block_idx: int, block_skip_future: torch.Tensor, key_5d: torch.Tensor
            ) -> torch.Tensor:
                score_block = torch.matmul(q_5d.float(), key_5d.transpose(-1, -2).float())
                masked_logit = _scalar_like(score_block, MASKED_ATTENTION_LOGIT)
                threshold = threshold_all[batch_idx : batch_idx + 1, :, :, :, block_idx]
                causal_mask = mask_pos > threshold.unsqueeze(-1)
                score_block = score_block.masked_fill(causal_mask, float(MASKED_ATTENTION_LOGIT))
                if cfg.skip_kv:
                    skip_score = block_skip_future.reshape(1, rows, 1, 1, 1)
                    score_block = torch.where(skip_score, masked_logit, score_block)
                block_scores_core = score_block.view(1, rows, num_cores, ql_eff, groups_per_core, page_block_size).amax(
                    dim=-1
                )
                if cfg.skip_kv:
                    block_scores_core = torch.where(skip_score, masked_logit, block_scores_core)
                return block_scores_core

            def append_masked_blocks(start_idx: int) -> None:
                masked_block = torch.full(
                    (1, rows, num_cores, ql_eff, groups_per_core),
                    MASKED_ATTENTION_LOGIT,
                    dtype=torch.float32,
                    device=device,
                )
                for remaining_idx in range(start_idx, num_kv_blocks):
                    device_block_score_groups.append(masked_block)
                    device_block_id_groups.append(make_paged_block_ids_rows(remaining_idx))

            page_block_ids = bt_local[:, make_paged_group_ids(0)].reshape(dp, num_groups_blk)
            key_flat = self._read_index_paged_kv_dp(index_key_cache, page_block_ids)
            key_5d = key_flat.view(1, rows, num_cores, tokens_per_core, dim)
            next_page_block_ids = bt_local[:, make_paged_group_ids(1)].reshape(dp, num_groups_blk)
            stopped_on_future = False
            for block_idx in range(num_kv_blocks - 1):
                block_ids_rows = make_paged_block_ids_rows(block_idx)
                block_skip_future = skip_future(block_idx)
                if cfg.skip_kv and not is_export and bool(block_skip_future.all().item()):
                    append_masked_blocks(block_idx)
                    stopped_on_future = True
                    break
                block_scores_core = compute_block_scores(block_idx, block_skip_future, key_5d)
                next_key_flat = self._read_index_paged_kv_dp(index_key_cache, next_page_block_ids)
                key_5d = next_key_flat.view(1, rows, num_cores, tokens_per_core, dim)
                if block_idx + 2 < num_kv_blocks:
                    next_page_block_ids = bt_local[:, make_paged_group_ids(block_idx + 2)].reshape(dp, num_groups_blk)
                device_block_score_groups.append(block_scores_core)
                device_block_id_groups.append(block_ids_rows)
            if not stopped_on_future:
                epilogue_idx = num_kv_blocks - 1
                epilogue_block_ids = make_paged_block_ids_rows(epilogue_idx)
                epilogue_scores = compute_block_scores(epilogue_idx, skip_future(epilogue_idx), key_5d)
                device_block_score_groups.append(epilogue_scores)
                device_block_id_groups.append(epilogue_block_ids)
            device_block_scores = torch.cat(device_block_score_groups, dim=4)
            device_block_ids = (
                torch.cat(device_block_id_groups, dim=3)
                .unsqueeze(3)
                .expand(1, rows, num_cores, ql_eff, num_kv_blocks * groups_per_core)
            )
            q_block_rows = (
                (position_ids_dp[batch_idx : batch_idx + 1, :, 0] // logical_block_size)
                .view(1, dp, 1, 1)
                .expand(1, dp, cp, hkv)
                .reshape(1, rows)
            )
            future_block = device_block_ids >= q_block_rows.view(1, rows, 1, 1, 1)
            device_block_scores = torch.where(
                future_block, _scalar_like(device_block_scores, MASKED_ATTENTION_LOGIT), device_block_scores
            )
            core_topk = min(cfg.selected_blocks, device_block_scores.shape[-1])
            core_topk_scores, core_topk_indices = torch.topk(device_block_scores, k=core_topk, dim=-1)
            core_topk_block_ids = torch.gather(device_block_ids, -1, core_topk_indices)
            core_candidate_scores = core_topk_scores.permute(0, 1, 3, 2, 4).reshape(
                1, rows, ql_eff, num_cores * core_topk
            )
            core_candidate_block_ids = core_topk_block_ids.permute(0, 1, 3, 2, 4).reshape(
                1, rows, ql_eff, num_cores * core_topk
            )
            core_candidate_score_groups.append(core_candidate_scores)
            core_candidate_block_groups.append(core_candidate_block_ids)

        core_candidate_scores = torch.cat(core_candidate_score_groups, dim=0)
        core_candidate_block_ids = torch.cat(core_candidate_block_groups, dim=0)
        device_topk = min(cfg.selected_blocks, core_candidate_scores.shape[-1])
        device_topk_scores, device_topk_indices = torch.topk(
            core_candidate_scores, k=device_topk, dim=-1
        )
        device_topk_block_ids = torch.gather(core_candidate_block_ids, -1, device_topk_indices)
        candidate_score_groups.append(device_topk_scores)
        candidate_block_groups.append(device_topk_block_ids)

        def merge_cp_candidates(groups: list[torch.Tensor]) -> torch.Tensor:
            return (
                torch.cat(groups, dim=0)
                .view(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, device_topk)
                .permute(0, 1, 3, 4, 5, 2, 6)
                .reshape(batch_local, dp, hkv, q_heads_per_kv, query_len, cp * device_topk)
            )

        candidate_scores = merge_cp_candidates(candidate_score_groups)
        candidate_block_indices = merge_cp_candidates(candidate_block_groups)
        topk_scores, candidate_topk = torch.topk(candidate_scores, k=cfg.selected_blocks, dim=-1)
        block_indices = torch.gather(candidate_block_indices, -1, candidate_topk)
        out_shape = (batch, num_index_heads, query_len, cfg.selected_blocks)
        topk_scores = topk_scores.permute(1, 0, 2, 3, 4, 5).reshape(out_shape)
        block_indices = block_indices.permute(1, 0, 2, 3, 4, 5).reshape(out_shape)
        current_block_ids = (
            (position_ids // logical_block_size)
            .view(batch, 1, query_len, 1)
            .expand(batch, num_index_heads, query_len, 1)
            .to(block_indices.dtype)
        )
        block_indices = torch.cat([block_indices[..., :-1], current_block_ids], dim=-1)
        topk_scores = torch.cat([topk_scores[..., :-1], torch.zeros_like(topk_scores[..., -1:])], dim=-1)
        block_valid = topk_scores > MASKED_ATTENTION_LOGIT / 2
        block_indices = block_indices[:, :, 0]
        block_valid = block_valid[:, :, 0]
        safe_block_indices = torch.where(block_valid, block_indices, torch.zeros_like(block_indices)).to(torch.int32)
        if not torch.compiler.is_exporting():
            self.last_block_indices = block_indices.detach()
        return (safe_block_indices, block_valid, index_key_cache)

    def msa_attention_prefill(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        token_indices: torch.Tensor,
        token_valid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        paged_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        num_kv_heads = cfg.num_kv_heads
        n_rep = cfg.num_kv_groups
        num_cores = cfg.num_cores_per_device
        cores_per_kv_head = num_cores // num_kv_heads
        ql_chunk = cfg.msa_q_chunk
        kv_num_blocks = max(1, cfg.msa_kv_blocks)
        selected_blocks = int(token_indices.shape[-1])
        block_size = cfg.index_block_size
        mask_value = float(MIN_MASKED_ATTENTION_VALUE)
        q = self.q_norm(self.q_proj(hidden_states).view(batch, query_len, cfg.num_heads, cfg.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(batch, query_len, cfg.num_kv_heads, cfg.head_dim)).transpose(
            1, 2
        )
        v = self.v_proj(hidden_states).view(batch, query_len, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        if paged_block_table is None:
            key_cache = self.ctx_scatter(key_cache, position_ids.to(torch.int32), k)
            value_cache = self.ctx_scatter(value_cache, position_ids.to(torch.int32), v)
        q = q.reshape(batch, num_kv_heads, n_rep, query_len, cfg.head_dim)
        offsets = torch.arange(block_size, device=hidden_states.device).view(1, 1, 1, 1, 1, block_size)
        output_chunks: list[torch.Tensor] = []
        for q_chunk_start in range(0, query_len, ql_chunk):
            q_chunk_end = min(q_chunk_start + ql_chunk, query_len)
            query_chunk = q_chunk_end - q_chunk_start
            q_per_core = query_chunk // cores_per_kv_head
            q_core = (
                q[:, :, :, q_chunk_start:q_chunk_end]
                .reshape(batch, num_kv_heads, n_rep, cores_per_kv_head, q_per_core, cfg.head_dim)
                .permute(0, 1, 3, 4, 2, 5)
                .reshape(batch, num_kv_heads * cores_per_kv_head, q_per_core, n_rep, cfg.head_dim)
                .float()
            )
            block_core = token_indices[:, :, q_chunk_start:q_chunk_end].reshape(
                batch, num_kv_heads, cores_per_kv_head, q_per_core, selected_blocks
            )
            block_valid_core = token_valid[:, :, q_chunk_start:q_chunk_end].reshape(
                batch, num_kv_heads, cores_per_kv_head, q_per_core, selected_blocks
            )
            query_positions = position_ids[:, q_chunk_start:q_chunk_end].reshape(
                batch, 1, cores_per_kv_head, q_per_core, 1
            )
            q_head_width = q_per_core * n_rep
            q_score_work = q_core.reshape(batch * num_kv_heads * cores_per_kv_head * q_per_core, n_rep, cfg.head_dim)
            m_acc = torch.full(
                (batch, num_kv_heads * cores_per_kv_head, q_head_width),
                -_FP16_MAX_VALUE,
                dtype=q_core.dtype,
                device=q_core.device,
            )
            s_acc = torch.zeros_like(m_acc)
            o_acc = torch.zeros(
                batch,
                num_kv_heads * cores_per_kv_head,
                q_head_width,
                cfg.head_dim,
                dtype=q_core.dtype,
                device=q_core.device,
            )
            for kv_block_idx in range(kv_num_blocks):
                selected_start = selected_blocks * kv_block_idx // kv_num_blocks
                selected_end = selected_blocks * (kv_block_idx + 1) // kv_num_blocks
                selected_block_count = selected_end - selected_start
                token_count = selected_block_count * block_size
                selected_block_ids = block_core[..., selected_start:selected_end]
                selected_offsets = offsets.expand(1, 1, 1, 1, selected_block_count, block_size).reshape(
                    1, 1, 1, 1, token_count
                )
                local_query = (
                    (query_positions.unsqueeze(-2) - selected_block_ids.unsqueeze(-1) * block_size)
                    .expand(batch, num_kv_heads, cores_per_kv_head, q_per_core, selected_block_count, block_size)
                    .reshape(batch, num_kv_heads, cores_per_kv_head, q_per_core, token_count)
                )
                selected_block_valid = block_valid_core[..., selected_start:selected_end]
                valid_block = (
                    selected_block_valid
                    .unsqueeze(-1)
                    .expand(batch, num_kv_heads, cores_per_kv_head, q_per_core, selected_block_count, block_size)
                    .reshape(batch, num_kv_heads, cores_per_kv_head, q_per_core, token_count)
                )
                valid_block = valid_block & (selected_offsets <= local_query)
                selected_positions = (
                    selected_block_ids.unsqueeze(-1) * block_size + offsets
                ).reshape(
                    batch, num_kv_heads, cores_per_kv_head, q_per_core, token_count
                )
                invalid_idx_value = (
                    torch.iinfo(torch.int32).max
                    if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
                    else 0
                )
                safe_selected_positions = torch.where(
                    valid_block,
                    selected_positions,
                    _scalar_like(selected_positions, invalid_idx_value),
                )
                gather_positions = safe_selected_positions.reshape(
                    batch, num_kv_heads, cores_per_kv_head * q_per_core * token_count
                ).to(torch.int32)
                if paged_block_table is None:
                    key_block = self.gather_blocked_kv(
                        key_cache, gather_positions
                    ).reshape(
                        batch, num_kv_heads * cores_per_kv_head, q_per_core, token_count, cfg.head_dim
                    ).float()
                else:
                    page_ids = (
                        token_indices[:, :, q_chunk_start:q_chunk_end, selected_start:selected_end]
                        .to(torch.int64)
                        .reshape(batch, num_kv_heads, selected_block_count * query_chunk)
                    )
                    physical_ids = torch.gather(
                        paged_block_table.to(torch.int64), 1, page_ids.reshape(batch, -1)
                    ).reshape(batch, num_kv_heads, -1)
                    key_pages = self.gather_paged_kv_selected_heads(key_cache, physical_ids).view(
                        batch, num_kv_heads, query_chunk, token_count, cfg.head_dim
                    )
                    key_block = key_pages.reshape(
                        batch, num_kv_heads * cores_per_kv_head, q_per_core, token_count, cfg.head_dim
                    ).float()
                key_score_work = key_block.reshape(
                    batch * num_kv_heads * cores_per_kv_head * q_per_core, token_count, cfg.head_dim
                ).transpose(1, 2)
                scores_block = torch.bmm(q_score_work, key_score_work).reshape(
                    batch, num_kv_heads * cores_per_kv_head, q_per_core, n_rep, token_count
                ) * cfg.head_dim ** (-0.5)
                attn_block = torch.where(
                    ~valid_block.reshape(batch, num_kv_heads * cores_per_kv_head, q_per_core, token_count).unsqueeze(3),
                    _scalar_like(scores_block, mask_value),
                    scores_block,
                ).reshape(batch, num_kv_heads * cores_per_kv_head, q_head_width, token_count)

                if paged_block_table is None:
                    value_block = self.gather_blocked_kv(
                        value_cache, gather_positions
                    ).reshape(
                        batch, num_kv_heads * cores_per_kv_head, q_per_core, token_count, cfg.head_dim
                    ).float()
                else:
                    value_pages = self.gather_paged_kv_selected_heads(value_cache, physical_ids).view(
                        batch, num_kv_heads, query_chunk, token_count, cfg.head_dim
                    )
                    value_block = value_pages.reshape(
                        batch, num_kv_heads * cores_per_kv_head, q_per_core, token_count, cfg.head_dim
                    ).float()
                value_block = torch.where(
                    valid_block.reshape(batch, num_kv_heads * cores_per_kv_head, q_per_core, token_count).unsqueeze(-1),
                    value_block,
                    _scalar_like(value_block, 0.0),
                )
                m_acc, s_acc, o_acc = update_running_softmax(m_acc, attn_block, s_acc, o_acc, value_block)
            out_chunk = (
                (o_acc / s_acc.clamp_min(1.0).unsqueeze(-1))
                .reshape(batch, num_kv_heads, cores_per_kv_head, q_per_core, n_rep, cfg.head_dim)
                .permute(0, 1, 4, 2, 3, 5)
                .reshape(batch, cfg.num_heads, query_chunk, cfg.head_dim)
            )
            output_chunks.append(out_chunk)
        return (torch.cat(output_chunks, dim=2), key_cache, value_cache)

    def msa_attention_GP(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        token_indices: torch.Tensor,
        token_valid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        dp = cfg.msa_attn_dp
        cp = cfg.msa_attn_cp
        hkv = cfg.num_kv_heads
        n_rep = cfg.num_kv_groups
        rows = dp * hkv
        ql_eff = n_rep * query_len
        if cp > 1:
            return self._msa_attention_GP_cp(
                hidden_states, position_ids, key_cache, value_cache,
                token_indices, token_valid, cos, sin,
            )
        if token_indices.ndim == 4:
            token_indices = token_indices[:, 0]
            token_valid = token_valid[:, 0]
        batch_local = batch // dp
        is_export = torch.compiler.is_exporting()
        token_indices = (
            token_indices.view(dp, batch_local, hkv, cfg.selected_blocks)
            .permute(1, 0, 2, 3)
            .reshape(batch_local, rows, cfg.selected_blocks)
        )
        token_valid = (
            token_valid.view(dp, batch_local, hkv, cfg.selected_blocks)
            .permute(1, 0, 2, 3)
            .reshape(batch_local, rows, cfg.selected_blocks)
        )
        position_rows = (
            position_ids.view(dp, batch_local, query_len)
            .permute(1, 0, 2)
            .view(batch_local, dp, 1, query_len)
            .expand(batch_local, dp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        offsets = torch.arange(cfg.index_block_size, device=hidden_states.device).view(1, 1, 1, -1)
        token_indices_expanded = (
            token_indices.unsqueeze(-1).expand(
                batch_local, rows, cfg.selected_blocks, cfg.index_block_size
            ) * cfg.index_block_size
        )
        token_indices_expanded = token_indices_expanded + offsets.expand(
            batch_local, rows, cfg.selected_blocks, cfg.index_block_size
        )
        token_valid_expanded = token_valid.unsqueeze(-1).expand(
            batch_local, rows, cfg.selected_blocks, cfg.index_block_size
        ) & (
            token_indices_expanded <= position_rows.unsqueeze(-1).expand(
                batch_local, rows, cfg.selected_blocks, cfg.index_block_size
            )
        )
        selected_len = cfg.selected_tokens
        gather_idx = token_indices_expanded.reshape(
            batch_local, rows, selected_len
        ).to(torch.int32)
        valid_idx = token_valid_expanded.reshape(batch_local, rows, selected_len)
        num_cores = cfg.num_cores_per_device
        q_heads_per_core = n_rep // num_cores
        q = self.q_norm(
            self.q_proj(hidden_states)
            .view(dp, batch_local, query_len, cfg.num_heads, cfg.head_dim)
            .permute(1, 0, 2, 3, 4)
        ).transpose(2, 3)
        k = self.k_norm(
            self.k_proj(hidden_states)
            .view(dp, batch_local, query_len, hkv, cfg.head_dim)
            .permute(1, 0, 2, 3, 4)
        ).transpose(2, 3)
        v = (
            self.v_proj(hidden_states)
            .view(dp, batch_local, query_len, hkv, cfg.head_dim)
            .permute(1, 0, 2, 3, 4)
            .transpose(2, 3)
        )
        cos_dp = cos.view(dp, batch_local, query_len, cos.shape[-1]).permute(1, 0, 2, 3)
        sin_dp = sin.view(dp, batch_local, query_len, sin.shape[-1]).permute(1, 0, 2, 3)
        q = self._apply_rope(q, cos_dp, sin_dp)
        k = self._apply_rope(k, cos_dp, sin_dp)
        k_updates = k.reshape(batch_local, rows, query_len, cfg.head_dim)
        v_updates = v.reshape(batch_local, rows, query_len, cfg.head_dim)
        batch_idx = torch.arange(batch_local, device=hidden_states.device).view(batch_local, 1, 1)
        block_id = batch_idx.expand(batch_local, rows, query_len).to(torch.int32)
        addr = (
            position_ids.view(dp, batch_local, query_len)
            .permute(1, 0, 2)
            .to(torch.int32)[:, :, None, :]
            .expand(batch_local, dp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        key_cache = self.paged_scatter(key_cache, block_id, addr, k_updates)
        value_cache = self.paged_scatter(value_cache, block_id, addr, v_updates)
        q_rows = (
            q.view(batch_local, dp, hkv, n_rep, query_len, cfg.head_dim)
            .reshape(batch_local, rows, ql_eff, cfg.head_dim)
        )
        local_batch_size = 1
        out_groups: list[torch.Tensor] = []
        for batch_start in range(0, batch_local, local_batch_size):
            batch_end = min(batch_start + local_batch_size, batch_local)
            local = batch_end - batch_start
            key_local = key_cache[batch_start:batch_end]
            value_local = value_cache[batch_start:batch_end]
            gather_local = gather_idx[batch_start:batch_end]
            valid_local = valid_idx[batch_start:batch_end]
            invalid_idx_value = torch.iinfo(torch.int32).max if is_export else 0
            safe_gather_local = torch.where(
                valid_local, gather_local, _scalar_like(gather_local, invalid_idx_value)
            )
            q_local = q_rows[batch_start:batch_end]
            selected_k = self.gather_blocked_kv_dp(
                key_local, safe_gather_local
            ).unsqueeze(2).repeat(1, 1, num_cores, 1, 1)
            selected_v = self.gather_blocked_kv_dp(
                value_local, safe_gather_local
            ).unsqueeze(2).repeat(1, 1, num_cores, 1, 1)
            valid_core = valid_local.unsqueeze(2).repeat(1, 1, num_cores, 1)
            q_core = q_local.view(
                local, rows, num_cores, q_heads_per_core * query_len, cfg.head_dim
            )
            attn = torch.matmul(
                q_core.float(), selected_k.transpose(-1, -2).float()
            ) * cfg.head_dim ** (-0.5)
            attn = torch.where(
                ~valid_core.unsqueeze(3),
                _scalar_like(attn, MASKED_ATTENTION_LOGIT),
                attn,
            )
            max_core = attn.max(dim=-1).values
            exp_core = torch.exp(attn - max_core.unsqueeze(-1))
            exp_core = torch.where(
                valid_core.unsqueeze(3), exp_core, _scalar_like(exp_core, 0.0)
            )
            selected_v = torch.where(
                valid_core.unsqueeze(-1), selected_v, _scalar_like(selected_v, 0.0)
            )
            sum_core = exp_core.sum(dim=-1)
            sum_core = torch.where(sum_core > 0, sum_core, torch.ones_like(sum_core))
            out_groups.append(
                torch.matmul(exp_core, selected_v.float()) / sum_core.unsqueeze(-1)
            )
        out_all = torch.cat(out_groups, dim=0)
        out = (
            out_all.view(batch_local, dp, hkv, n_rep, query_len, cfg.head_dim)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(batch, cfg.num_heads, query_len, cfg.head_dim)
        )
        return (out, key_cache, value_cache)

    def msa_attention_GP_batched(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        token_indices: torch.Tensor,
        token_valid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        dp = cfg.msa_attn_dp
        cp = cfg.msa_attn_cp
        hkv = cfg.num_kv_heads
        n_rep = cfg.num_kv_groups
        rows = dp * cp * hkv
        ql_eff = n_rep * query_len
        if cfg.ctx_len % cp:
            raise ValueError("Non-paged MSA ctx_len must be divisible by msa_attn_cp.")
        cache_slots = cfg.ctx_len // cp
        expected_cache_shape = (batch // dp, rows, cache_slots, cfg.head_dim)
        if tuple(key_cache.shape) != expected_cache_shape or tuple(value_cache.shape) != expected_cache_shape:
            raise ValueError(f"Compact MSA key/value cache must both have shape {expected_cache_shape}.")
        num_cores = cfg.num_cores_per_device
        batch_local = batch // dp
        if token_indices.ndim == 5:
            token_indices = token_indices.reshape(batch, hkv, cfg.selected_blocks)
            token_valid = token_valid.reshape(batch, hkv, cfg.selected_blocks)
        elif token_indices.ndim == 4:
            selector_index_shape = (batch, cfg.msa_indexer_cp, hkv, cfg.selected_blocks)
            if tuple(token_indices.shape) != selector_index_shape or tuple(token_valid.shape) != selector_index_shape:
                raise ValueError(f"MSA token_indices/token_valid shape must be {selector_index_shape}.")
            token_indices = token_indices[:, 0]
            token_valid = token_valid[:, 0]
        compressed_block_shape = (batch, hkv, cfg.selected_blocks)
        if tuple(token_indices.shape) != compressed_block_shape or tuple(token_valid.shape) != compressed_block_shape:
            raise ValueError(f"MSA token_indices/token_valid shape must be {compressed_block_shape}.")
        if cfg.index_block_size % cp:
            raise ValueError("MSA index_block_size must be divisible by msa_attn_cp.")
        block_ids = token_indices.view(dp, batch_local, hkv, cfg.selected_blocks).permute(1, 0, 2, 3)
        block_valid = token_valid.view(dp, batch_local, hkv, cfg.selected_blocks).permute(1, 0, 2, 3)
        offsets = torch.arange(cfg.index_block_size, device=hidden_states.device).view(1, 1, 1, 1, -1)
        global_idx = block_ids.unsqueeze(-1) * cfg.index_block_size + offsets
        position_limit = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)[:, :, None, None, :]
        valid_base = block_valid.unsqueeze(-1) & (global_idx < cfg.ctx_len)
        local_q = position_limit - block_ids.unsqueeze(-1) * cfg.index_block_size
        valid_base = valid_base & (offsets <= local_q)
        local_offset = torch.tensor(
            [offset // cp for offset in range(cfg.index_block_size)], dtype=torch.int32, device=hidden_states.device
        ).view(1, 1, 1, 1, -1)
        offset_way = torch.tensor(
            [offset % cp for offset in range(cfg.index_block_size)], dtype=torch.int64, device=hidden_states.device
        ).view(1, 1, 1, 1, -1)
        selected_len = cfg.selected_tokens
        gather_local_base = (
            block_ids.unsqueeze(-1) * (cfg.index_block_size // cp) + local_offset
        ).reshape(batch_local, dp, hkv, selected_len)
        valid_base = valid_base.reshape(batch_local, dp, hkv, selected_len)
        gather_block_ids = (
            gather_local_base.unsqueeze(2).expand(batch_local, dp, cp, hkv, selected_len)
            .reshape(batch_local, rows, selected_len).to(torch.int32)
        )
        valid_idx = (
            valid_base.unsqueeze(2).expand(batch_local, dp, cp, hkv, selected_len)
            .reshape(batch_local, rows, selected_len)
        )
        offset_way = offset_way.expand(1, 1, 1, cfg.selected_blocks, cfg.index_block_size).reshape(
            1, 1, 1, 1, selected_len
        )
        way_idx = torch.arange(cp, device=hidden_states.device).view(1, 1, cp, 1, 1)
        gather_live = (
            (offset_way == way_idx).expand(batch_local, dp, cp, hkv, selected_len)
            .reshape(batch_local, rows, selected_len)
        )
        valid_idx = valid_idx & gather_live
        gather_block_ids = torch.where(
            valid_idx, gather_block_ids, _scalar_like(gather_block_ids, torch.iinfo(torch.int32).max)
        ).to(torch.int32)
        assert batch_local % num_cores == 0, "MSA batch_local must be divisible by the compile core count."
        local_batch_size = batch_local // num_cores
        q = self.q_norm(
            self.q_proj(hidden_states)
            .view(dp, batch_local, query_len, cfg.num_heads, cfg.head_dim)
            .permute(1, 0, 2, 3, 4)
        ).transpose(2, 3)
        k = self.k_norm(
            self.k_proj(hidden_states).view(dp, batch_local, query_len, hkv, cfg.head_dim).permute(1, 0, 2, 3, 4)
        ).transpose(2, 3)
        v = (
            self.v_proj(hidden_states)
            .view(dp, batch_local, query_len, hkv, cfg.head_dim)
            .permute(1, 0, 2, 3, 4)
            .transpose(2, 3)
        )
        cos_dp = cos.view(dp, batch_local, query_len, cos.shape[-1]).permute(1, 0, 2, 3)
        sin_dp = sin.view(dp, batch_local, query_len, sin.shape[-1]).permute(1, 0, 2, 3)
        q = self._apply_rope(q, cos_dp, sin_dp)
        k = self._apply_rope(k, cos_dp, sin_dp)
        k_updates = k.unsqueeze(2).expand(batch_local, dp, cp, hkv, query_len, cfg.head_dim).reshape(
            batch_local, rows, query_len, cfg.head_dim
        )
        v_updates = v.unsqueeze(2).expand(batch_local, dp, cp, hkv, query_len, cfg.head_dim).reshape(
            batch_local, rows, query_len, cfg.head_dim
        )
        position_ids_dp = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)
        row_way = torch.tensor(
            [way for _dp in range(dp) for way in range(cp) for _h in range(hkv)],
            dtype=torch.int64, device=hidden_states.device
        )
        addr_base = position_ids_dp // cp
        live_way = (
            (position_ids_dp - addr_base * cp).view(batch_local, dp, 1, 1, query_len)
            .expand(batch_local, dp, cp, hkv, query_len).reshape(batch_local, rows, query_len)
        )
        batch_idx = torch.arange(batch_local, device=hidden_states.device).view(batch_local, 1, 1)
        block_id = torch.where(
            row_way.view(1, rows, 1) == live_way,
            batch_idx.expand(batch_local, rows, query_len),
            torch.iinfo(torch.int32).max,
        ).to(torch.int32)
        addr = (
            addr_base.to(torch.int32)[:, :, None, None, :].expand(batch_local, dp, cp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        key_cache = self.paged_scatter(key_cache, block_id, addr, k_updates)
        value_cache = self.paged_scatter(value_cache, block_id, addr, v_updates)
        local_batch_size = batch_local // num_cores
        q_rows = (
            q.view(batch_local, dp, hkv, n_rep, query_len, cfg.head_dim)
            .unsqueeze(2).expand(batch_local, dp, cp, hkv, n_rep, query_len, cfg.head_dim)
            .reshape(batch_local, rows, ql_eff, cfg.head_dim)
        )
        selected_k = self.gather_blocked_kv_dp(key_cache, gather_block_ids).reshape(
            num_cores, local_batch_size, rows, selected_len, cfg.head_dim
        ).permute(0, 2, 1, 3, 4)
        selected_v = self.gather_blocked_kv_dp(value_cache, gather_block_ids).reshape(
            num_cores, local_batch_size, rows, selected_len, cfg.head_dim
        ).permute(0, 2, 1, 3, 4)
        valid_core = valid_idx.reshape(
            num_cores, local_batch_size, rows, selected_len
        ).permute(0, 2, 1, 3)
        q_core = q_rows.reshape(
            num_cores, local_batch_size, rows, ql_eff, cfg.head_dim
        ).permute(0, 2, 1, 3, 4)
        attn = torch.matmul(selected_k.float(), q_core.transpose(-1, -2).float()).transpose(-1, -2) * cfg.head_dim ** (-0.5)
        attn = torch.where(~valid_core.unsqueeze(3), _scalar_like(attn, MASKED_ATTENTION_LOGIT), attn)
        max_core = attn.max(dim=-1).values
        exp_core = torch.exp(attn - max_core.unsqueeze(-1))
        exp_core = torch.where(valid_core.unsqueeze(3), exp_core, _scalar_like(exp_core, 0.0))
        selected_v = torch.where(valid_core.unsqueeze(-1), selected_v, _scalar_like(selected_v, 0.0))
        sum_core = exp_core.sum(dim=-1)
        out_core = torch.matmul(exp_core, selected_v.float())
        max_cp = max_core.reshape(num_cores, dp, cp, hkv, local_batch_size, ql_eff)
        sum_cp = sum_core.reshape(num_cores, dp, cp, hkv, local_batch_size, ql_eff)
        out_cp = out_core.reshape(
            num_cores, dp, cp, hkv, local_batch_size, ql_eff, cfg.head_dim
        )
        max_all = max_cp.max(dim=2).values
        cp_weight = torch.exp(max_cp - max_all.unsqueeze(2))
        sum_all = (cp_weight * sum_cp).sum(dim=2)
        out_all = (cp_weight.unsqueeze(-1) * out_cp).sum(dim=2)
        out = (out_all / sum_all.unsqueeze(-1)).reshape(
            num_cores, dp, hkv, local_batch_size, n_rep, query_len, cfg.head_dim
        ).permute(0, 3, 1, 2, 4, 5, 6).reshape(
            batch_local, dp, hkv, n_rep, query_len, cfg.head_dim
        )
        out = out.permute(1, 0, 2, 3, 4, 5).reshape(batch, cfg.num_heads, query_len, cfg.head_dim)
        return (out, key_cache, value_cache)

    def _msa_attention_GP_cp(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        token_indices: torch.Tensor,
        token_valid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compact sparse MSA with non-paged cache rows ordered [DP, CP, Hkv]."""
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        dp, cp, hkv = cfg.msa_attn_dp, cfg.msa_attn_cp, cfg.num_kv_heads
        n_rep = cfg.num_kv_groups
        batch_local = batch // dp
        rows = dp * cp * hkv
        ql_eff = n_rep * query_len
        if cfg.ctx_len % cp:
            raise ValueError("Non-paged MSA ctx_len must be divisible by msa_attn_cp.")
        cache_slots = cfg.ctx_len // cp
        expected_cache_shape = (batch_local, rows, cache_slots, cfg.head_dim)
        if tuple(key_cache.shape) != expected_cache_shape or tuple(value_cache.shape) != expected_cache_shape:
            raise ValueError(
                "Compact MSA key/value cache must both have shape "
                f"{expected_cache_shape}."
            )
        num_cores = cfg.num_cores_per_device
        if n_rep % num_cores:
            raise ValueError(
                f"MSA Q-head core parallelism requires num_kv_groups ({n_rep}) "
                f"divisible by compile core count ({num_cores})."
            )
        q_heads_per_core = n_rep // num_cores

        if token_indices.ndim == 4:
            selector_index_shape = (
                batch,
                cfg.msa_indexer_cp,
                hkv,
                cfg.selected_blocks,
            )
            if tuple(token_indices.shape) != selector_index_shape:
                raise ValueError(
                    f"MSA token_indices shape {tuple(token_indices.shape)} != {selector_index_shape}."
                )
            if tuple(token_valid.shape) != selector_index_shape:
                raise ValueError(
                    f"MSA token_valid shape {tuple(token_valid.shape)} != {selector_index_shape}."
                )
            token_indices = token_indices[:, 0]
            token_valid = token_valid[:, 0]

        compressed_block_shape = (batch, hkv, cfg.selected_blocks)
        if tuple(token_indices.shape) != compressed_block_shape or tuple(token_valid.shape) != compressed_block_shape:
            raise ValueError(f"MSA CP>1 requires block indices/valid with shape {compressed_block_shape}.")
        if cfg.index_block_size % cp:
            raise ValueError("MSA index_block_size must be divisible by msa_attn_cp.")

        block_ids = token_indices.view(dp, batch_local, hkv, cfg.selected_blocks).permute(1, 0, 2, 3)
        block_valid = token_valid.view(dp, batch_local, hkv, cfg.selected_blocks).permute(1, 0, 2, 3)
        token_offset = torch.arange(cfg.index_block_size, device=hidden_states.device).view(1, 1, 1, 1, -1)
        global_idx = block_ids.unsqueeze(-1) * cfg.index_block_size + token_offset
        position_limit = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)[:, :, None, None, :]
        valid_base = block_valid.unsqueeze(-1) & (global_idx < cfg.ctx_len)
        local_q = position_limit - block_ids.unsqueeze(-1) * cfg.index_block_size
        valid_base = valid_base & (token_offset <= local_q)
        local_offset = torch.tensor(
            [offset // cp for offset in range(cfg.index_block_size)],
            dtype=torch.int32,
            device=hidden_states.device,
        ).view(1, 1, 1, 1, -1)
        offset_way = torch.tensor(
            [offset % cp for offset in range(cfg.index_block_size)],
            dtype=torch.int64,
            device=hidden_states.device,
        ).view(1, 1, 1, 1, -1)
        selected_len = cfg.selected_tokens
        gather_local_base = (
            block_ids.unsqueeze(-1) * (cfg.index_block_size // cp) + local_offset
        ).reshape(batch_local, dp, hkv, selected_len)
        valid_base = valid_base.reshape(batch_local, dp, hkv, selected_len)
        gather_idx = (
            gather_local_base.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, selected_len)
            .reshape(batch_local, rows, selected_len)
            .to(torch.int32)
        )
        valid_idx = (
            valid_base.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, selected_len)
            .reshape(batch_local, rows, selected_len)
        )
        offset_way = offset_way.expand(1, 1, 1, cfg.selected_blocks, cfg.index_block_size).reshape(
            1, 1, 1, 1, selected_len
        )
        way_idx = torch.arange(cp, device=hidden_states.device).view(1, 1, cp, 1, 1)
        gather_live = (
            (offset_way == way_idx)
            .expand(batch_local, dp, cp, hkv, selected_len)
            .reshape(batch_local, rows, selected_len)
        )
        valid_idx = valid_idx & gather_live
        gather_idx = torch.where(
            valid_idx,
            gather_idx,
            _scalar_like(gather_idx, torch.iinfo(torch.int32).max),
        ).to(torch.int32)

        q = self.q_norm(
            self.q_proj(hidden_states)
            .view(dp, batch_local, query_len, cfg.num_heads, cfg.head_dim)
            .permute(1, 0, 2, 3, 4)
        ).transpose(2, 3)
        k = self.k_norm(
            self.k_proj(hidden_states).view(dp, batch_local, query_len, hkv, cfg.head_dim).permute(1, 0, 2, 3, 4)
        ).transpose(2, 3)
        v = (
            self.v_proj(hidden_states)
            .view(dp, batch_local, query_len, hkv, cfg.head_dim)
            .permute(1, 0, 2, 3, 4)
            .transpose(2, 3)
        )
        cos_dp = cos.view(dp, batch_local, query_len, cos.shape[-1]).permute(1, 0, 2, 3)
        sin_dp = sin.view(dp, batch_local, query_len, sin.shape[-1]).permute(1, 0, 2, 3)
        q = self._apply_rope(q, cos_dp, sin_dp)
        k = self._apply_rope(k, cos_dp, sin_dp)

        k_updates = k.unsqueeze(2).expand(
            batch_local, dp, cp, hkv, query_len, cfg.head_dim
        ).reshape(batch_local, rows, query_len, cfg.head_dim)
        v_updates = v.unsqueeze(2).expand(
            batch_local, dp, cp, hkv, query_len, cfg.head_dim
        ).reshape(batch_local, rows, query_len, cfg.head_dim)
        position_ids_dp = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)
        row_way = torch.tensor(
            [way for _dp in range(dp) for way in range(cp) for _h in range(hkv)],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        addr_base = position_ids_dp // cp
        live_way = (
            (position_ids_dp - addr_base * cp)
            .view(batch_local, dp, 1, 1, query_len)
            .expand(batch_local, dp, cp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        row_way_scatter = row_way.view(1, rows, 1)
        batch_id = torch.arange(batch_local, device=hidden_states.device).view(batch_local, 1, 1).expand(
            batch_local, rows, query_len
        )
        block_id = torch.where(
            row_way_scatter == live_way,
            batch_id,
            torch.iinfo(torch.int32).max,
        ).to(torch.int32)
        addr = (
            addr_base.to(torch.int32)
            .view(batch_local, dp, 1, 1, query_len)
            .expand(batch_local, dp, cp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        key_cache = self.paged_scatter(key_cache, block_id, addr, k_updates)
        value_cache = self.paged_scatter(value_cache, block_id, addr, v_updates)

        q_rows = (
            q.view(batch_local, dp, hkv, n_rep, query_len, cfg.head_dim)
            .unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, n_rep, query_len, cfg.head_dim)
            .reshape(batch_local, rows, ql_eff, cfg.head_dim)
        )
        local_batch_size = 1
        out_groups: list[torch.Tensor] = []
        for batch_start in range(0, batch_local, local_batch_size):
            batch_end = min(batch_start + local_batch_size, batch_local)
            local = batch_end - batch_start
            gather_local = gather_idx[batch_start:batch_end]
            valid_local = valid_idx[batch_start:batch_end]
            selected_k = self.gather_blocked_kv_dp(
                key_cache[batch_start:batch_end], gather_local
            ).unsqueeze(2).repeat(1, 1, num_cores, 1, 1)
            selected_v = self.gather_blocked_kv_dp(
                value_cache[batch_start:batch_end], gather_local
            ).unsqueeze(2).repeat(1, 1, num_cores, 1, 1)
            valid_core = valid_local.unsqueeze(2).repeat(1, 1, num_cores, 1)
            q_core = q_rows[batch_start:batch_end].view(
                local, rows, num_cores, q_heads_per_core * query_len, cfg.head_dim
            )
            attn = torch.matmul(q_core.float(), selected_k.transpose(-1, -2).float()) * (cfg.head_dim ** -0.5)
            attn = torch.where(
                ~valid_core.unsqueeze(3), _scalar_like(attn, MASKED_ATTENTION_LOGIT), attn
            )
            max_core = attn.max(dim=-1).values
            exp_core = torch.exp(attn - max_core.unsqueeze(-1))
            exp_core = torch.where(valid_core.unsqueeze(3), exp_core, _scalar_like(exp_core, 0.0))
            selected_v = torch.where(valid_core.unsqueeze(-1), selected_v, _scalar_like(selected_v, 0.0))
            sum_core = exp_core.sum(dim=-1)
            out_core = torch.matmul(exp_core, selected_v.float())
            max_cp = max_core.view(local, dp, cp, hkv, num_cores, q_heads_per_core * query_len)
            sum_cp = sum_core.view(local, dp, cp, hkv, num_cores, q_heads_per_core * query_len)
            out_cp = out_core.view(
                local, dp, cp, hkv, num_cores,
                q_heads_per_core * query_len, cfg.head_dim,
            )
            max_all = max_cp.max(dim=2).values
            cp_weight = torch.exp(max_cp - max_all.unsqueeze(2))
            sum_all = (cp_weight * sum_cp).sum(dim=2)
            out_all = (cp_weight.unsqueeze(-1) * out_cp).sum(dim=2)
            sum_all = torch.where(sum_all > 0, sum_all, torch.ones_like(sum_all))
            out_groups.append(out_all / sum_all.unsqueeze(-1))

        out_all = torch.cat(out_groups, dim=0)
        out = (
            out_all.view(batch_local, dp, hkv, n_rep, query_len, cfg.head_dim)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(batch, cfg.num_heads, query_len, cfg.head_dim)
        )
        return out, key_cache, value_cache

    def _msa_attention_paged_GP_cp(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        token_indices: torch.Tensor,
        token_valid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_table: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        dp, cp, hkv = (cfg.msa_attn_dp, cfg.msa_attn_cp, cfg.num_kv_heads)
        n_rep = cfg.num_kv_groups
        batch_local = batch // dp
        rows = dp * cp * hkv
        ql_eff = n_rep * query_len
        selected_len = cfg.selected_tokens
        selected_blocks = cfg.selected_blocks
        (cfg.ctx_len // cfg.page_block_size + cp - 1) // cp
        num_cores = cfg.num_cores_per_device
        if n_rep % num_cores:
            raise ValueError(
                f"MSA Q-head core parallelism requires num_kv_groups ({n_rep}) "
                f"divisible by compile core count ({num_cores})."
            )
        q_heads_per_core = n_rep // num_cores
        position_ids_dp = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)
        block_ids_base = token_indices.view(dp, batch_local, hkv, selected_blocks).permute(1, 0, 2, 3)
        block_valid_base = token_valid.view(dp, batch_local, hkv, selected_blocks).permute(1, 0, 2, 3)
        offsets = torch.arange(cfg.index_block_size, device=hidden_states.device).view(1, 1, 1, 1, -1)
        global_idx = block_ids_base.unsqueeze(-1) * cfg.index_block_size + offsets
        valid_base = block_valid_base.unsqueeze(-1) & (global_idx < cfg.ctx_len)
        local_q = position_ids_dp[:, :, None, None, :] - block_ids_base.unsqueeze(-1) * cfg.index_block_size
        valid_base = valid_base & (offsets <= local_q)
        block_ids_rows = (
            block_ids_base.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, selected_blocks)
            .reshape(batch_local, rows, selected_blocks)
        )
        valid_idx = (
            valid_base.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, selected_blocks, cfg.index_block_size)
            .reshape(batch_local, rows, selected_len)
        )
        row_way = (torch.arange(rows, device=hidden_states.device).remainder(cp * hkv) // hkv).view(1, rows, 1)
        page_live_rows = block_ids_rows.remainder(cp) == row_way
        valid_idx = (
            valid_idx.view(batch_local, rows, selected_blocks, cfg.index_block_size) & page_live_rows.unsqueeze(-1)
        ).reshape(batch_local, rows, selected_len)
        q = self.q_norm(
            self.q_proj(hidden_states)
            .view(dp, batch_local, query_len, cfg.num_heads, cfg.head_dim)
            .permute(1, 0, 2, 3, 4)
        ).transpose(2, 3)
        k = self.k_norm(
            self.k_proj(hidden_states).view(dp, batch_local, query_len, hkv, cfg.head_dim).permute(1, 0, 2, 3, 4)
        ).transpose(2, 3)
        v = (
            self.v_proj(hidden_states)
            .view(dp, batch_local, query_len, hkv, cfg.head_dim)
            .permute(1, 0, 2, 3, 4)
            .transpose(2, 3)
        )
        cos_dp = cos.view(dp, batch_local, query_len, cos.shape[-1]).permute(1, 0, 2, 3)
        sin_dp = sin.view(dp, batch_local, query_len, sin.shape[-1]).permute(1, 0, 2, 3)
        q = self._apply_rope(q, cos_dp, sin_dp)
        k = self._apply_rope(k, cos_dp, sin_dp)
        k_updates = (
            k.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, query_len, cfg.head_dim)
            .reshape(batch_local, rows, query_len, cfg.head_dim)
        )
        v_updates = (
            v.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, query_len, cfg.head_dim)
            .reshape(batch_local, rows, query_len, cfg.head_dim)
        )
        addr = (
            (position_ids_dp % cfg.page_block_size)
            .to(torch.int32)
            .view(batch_local, dp, 1, 1, query_len)
            .expand(batch_local, dp, cp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        logical_page = position_ids_dp // cfg.page_block_size
        logical_page_group = (logical_page // cp).to(torch.int64)
        block_id_dp = (
            torch.gather(block_table.permute(1, 0, 2), 2, logical_page_group)
            .to(torch.int32)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(batch_local, dp, cp, hkv, query_len)
        )
        live_way = (logical_page % cp).to(torch.int64)
        way_idx = torch.arange(cp, device=hidden_states.device).view(1, 1, cp, 1, 1)
        row_live = way_idx == live_way.unsqueeze(2).unsqueeze(3)
        row_live = row_live.expand(batch_local, dp, cp, hkv, query_len)
        block_id = torch.where(row_live, block_id_dp, torch.iinfo(torch.int32).max).reshape(
            batch_local, rows, query_len
        )
        key_cache = self.paged_scatter(key_cache, block_id, addr, k_updates)
        value_cache = self.paged_scatter(value_cache, block_id, addr, v_updates)
        q_rows = (
            q.view(batch_local, dp, hkv, n_rep, query_len, cfg.head_dim)
            .unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, n_rep, query_len, cfg.head_dim)
            .reshape(batch_local, rows, ql_eff, cfg.head_dim)
        )
        row_dp = torch.tensor(
            [dp_idx for dp_idx in range(dp) for _way in range(cp) for _h in range(hkv)],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        invalid_block = torch.iinfo(torch.int32).max
        local_batch_size = 1
        out_groups: list[torch.Tensor] = []
        for batch_start in range(0, batch_local, local_batch_size):
            page_valid = (
                valid_idx[batch_start : batch_start + 1]
                .view(1, rows, selected_blocks, cfg.index_block_size)
                .float()
                .amax(dim=-1)
                > 0
            )
            logical_pages_local = block_ids_rows[batch_start].long()
            table_rows = block_table[:, batch_start].index_select(0, row_dp)
            physical_block_ids = torch.gather(table_rows, 1, logical_pages_local // cp)
            physical_block_ids = torch.where(
                page_valid[0], physical_block_ids, _scalar_like(physical_block_ids, invalid_block)
            )
            block_ids = physical_block_ids.transpose(0, 1).contiguous().to(torch.int32)
            selected_k = self.gather_paged_kv_dp(key_cache, block_ids).view(1, rows, selected_len, cfg.head_dim)
            selected_v = self.gather_paged_kv_dp(value_cache, block_ids).view(1, rows, selected_len, cfg.head_dim)
            selected_k = selected_k.unsqueeze(2).repeat(1, 1, num_cores, 1, 1)
            selected_v = selected_v.unsqueeze(2).repeat(1, 1, num_cores, 1, 1)
            valid_core = valid_idx[batch_start : batch_start + 1].unsqueeze(2).repeat(1, 1, num_cores, 1)
            q_core = q_rows[batch_start : batch_start + 1].view(
                1, rows, num_cores, q_heads_per_core * query_len, cfg.head_dim
            )
            attn = torch.matmul(q_core.float(), selected_k.transpose(-1, -2).float()) * cfg.head_dim ** (-0.5)
            attn = torch.where(~valid_core.unsqueeze(3), _scalar_like(attn, MASKED_ATTENTION_LOGIT), attn)
            max_core = attn.max(dim=-1).values
            exp_core = torch.exp(attn - max_core.unsqueeze(-1))
            exp_core = torch.where(valid_core.unsqueeze(3), exp_core, _scalar_like(exp_core, 0.0))
            selected_v = torch.where(valid_core.unsqueeze(-1), selected_v, _scalar_like(selected_v, 0.0))
            sum_core = exp_core.sum(dim=-1)
            out_core = torch.matmul(exp_core, selected_v.float())
            max_cp = max_core.view(1, dp, cp, hkv, num_cores, q_heads_per_core * query_len)
            sum_cp = sum_core.view(1, dp, cp, hkv, num_cores, q_heads_per_core * query_len)
            out_cp = out_core.view(
                1, dp, cp, hkv, num_cores,
                q_heads_per_core * query_len, cfg.head_dim,
            )
            max_all = max_cp.max(dim=2).values
            cp_weight = torch.exp(max_cp - max_all.unsqueeze(2))
            sum_all = (cp_weight * sum_cp).sum(dim=2)
            out_all = (cp_weight.unsqueeze(-1) * out_cp).sum(dim=2)
            sum_all = torch.where(sum_all > 0, sum_all, torch.ones_like(sum_all))
            out_groups.append(out_all / sum_all.unsqueeze(-1))
        out_all = torch.cat(out_groups, dim=0)
        out = (
            out_all.view(batch_local, dp, hkv, n_rep, query_len, cfg.head_dim)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(batch, cfg.num_heads, query_len, cfg.head_dim)
        )
        return (out, key_cache, value_cache)

    def msa_attention_paged_GP(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        token_indices: torch.Tensor,
        token_valid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        dp = cfg.msa_attn_dp
        cp = cfg.msa_attn_cp
        hkv = cfg.num_kv_heads
        n_rep = cfg.num_kv_groups
        rows = dp * hkv
        ql_eff = n_rep * query_len
        if cp > 1:
            return self._msa_attention_paged_GP_cp(
                hidden_states, position_ids, key_cache, value_cache, token_indices, token_valid, cos, sin, block_table
            )
        batch_local = batch // dp
        position_ids_dp = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)
        compressed_block_shape = (batch, hkv, cfg.selected_blocks)
        if token_indices.ndim == 3 and tuple(token_indices.shape) == compressed_block_shape:
            offsets = torch.arange(cfg.index_block_size, device=hidden_states.device).view(
                1, 1, 1, cfg.index_block_size
            )
            token_indices_expanded = token_indices.unsqueeze(-1) * cfg.index_block_size + offsets
            token_valid_expanded = token_valid.unsqueeze(-1)
            token_valid_expanded = token_valid_expanded & (token_indices_expanded < cfg.ctx_len)
            q_block_offset = position_ids[:, None, None, :] - token_indices.unsqueeze(-1) * cfg.index_block_size
            token_valid_expanded = token_valid_expanded & (offsets <= q_block_offset)
            token_indices_expanded = torch.where(
                token_valid_expanded, token_indices_expanded, torch.zeros_like(token_indices_expanded)
            )
            selected_len = cfg.selected_tokens
            gather_idx = (
                token_indices_expanded.view(dp, batch_local, hkv, selected_len)
                .permute(1, 0, 2, 3)
                .reshape(batch_local, rows, selected_len)
                .to(torch.int32)
            )
            valid_idx = (
                token_valid_expanded.view(dp, batch_local, hkv, selected_len)
                .permute(1, 0, 2, 3)
                .reshape(batch_local, rows, selected_len)
            )
        elif token_indices.ndim == 3:
            (batch_local, rows, token_indices.shape[-1])
            selected_len = token_indices.shape[-1]
            gather_idx = token_indices.to(torch.int32)
            valid_idx = token_valid
        elif token_indices.ndim == 4:
            selected_len = token_indices.flatten(-2).shape[-1]
            gather_idx = (
                token_indices.flatten(-2)
                .view(dp, batch_local, hkv, selected_len)
                .permute(1, 0, 2, 3)
                .reshape(batch_local, rows, selected_len)
                .to(torch.int32)
            )
            valid_idx = (
                token_valid.flatten(-2)
                .view(dp, batch_local, hkv, selected_len)
                .permute(1, 0, 2, 3)
                .reshape(batch_local, rows, selected_len)
            )
        else:
            raise ValueError(
                "MSA token_indices/token_valid must use either [B, Hkv, selected_blocks], [B_local, DP*Hkv, selected_len] or [B, Hkv, selected_blocks, index_block_size]."
            )
        selected_blocks = selected_len // cfg.index_block_size
        num_cores = cfg.num_cores_per_device
        q_heads_per_core = n_rep // num_cores
        (cfg.ctx_len // cfg.page_block_size + cp - 1) // cp
        q = self.q_norm(
            self.q_proj(hidden_states)
            .view(dp, batch_local, query_len, cfg.num_heads, cfg.head_dim)
            .permute(1, 0, 2, 3, 4)
        ).transpose(2, 3)
        k = self.k_norm(
            self.k_proj(hidden_states).view(dp, batch_local, query_len, hkv, cfg.head_dim).permute(1, 0, 2, 3, 4)
        ).transpose(2, 3)
        v = (
            self.v_proj(hidden_states)
            .view(dp, batch_local, query_len, hkv, cfg.head_dim)
            .permute(1, 0, 2, 3, 4)
            .transpose(2, 3)
        )
        cos_dp = cos.view(dp, batch_local, query_len, cos.shape[-1]).permute(1, 0, 2, 3)
        sin_dp = sin.view(dp, batch_local, query_len, sin.shape[-1]).permute(1, 0, 2, 3)
        q = self._apply_rope(q, cos_dp, sin_dp)
        k = self._apply_rope(k, cos_dp, sin_dp)
        k_updates = k.reshape(batch_local, rows, query_len, cfg.head_dim)
        v_updates = v.reshape(batch_local, rows, query_len, cfg.head_dim)
        logical_page = position_ids_dp // cfg.page_block_size
        logical_page_group = (logical_page // cp).to(torch.int64)
        block_id_dp = torch.gather(block_table.permute(1, 0, 2), 2, logical_page_group)
        block_id = (
            block_id_dp.unsqueeze(2)
            .expand(batch_local, dp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
            .to(torch.int32)
        )
        addr_dp = position_ids_dp % cfg.page_block_size
        addr = (
            addr_dp.to(torch.int32)[:, :, None, :]
            .expand(batch_local, dp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        key_cache = self.paged_scatter(key_cache, block_id, addr, k_updates)
        value_cache = self.paged_scatter(value_cache, block_id, addr, v_updates)
        q_rows = q.view(batch_local, dp, hkv, n_rep, query_len, cfg.head_dim).reshape(
            batch_local, rows, ql_eff, cfg.head_dim
        )
        local_batch_size = 1
        out_groups: list[torch.Tensor] = []
        for batch_start in range(0, batch_local, local_batch_size):
            batch_end = min(batch_start + local_batch_size, batch_local)
            local = batch_end - batch_start
            gather_blocks = gather_idx[batch_start:batch_end].view(local, rows, selected_blocks, cfg.index_block_size)
            valid_blocks = valid_idx[batch_start:batch_end].view(local, rows, selected_blocks, cfg.index_block_size)
            q_local = q_rows[batch_start:batch_end]
            logical_pages = (gather_blocks[..., 0] // cfg.page_block_size).long()
            valid_pages = valid_blocks.float().amax(dim=-1) > 0
            row_dp = torch.arange(rows, device=hidden_states.device) // hkv
            table_rows = block_table[:, batch_start].index_select(0, row_dp)
            physical_block_ids = torch.gather(table_rows, 1, logical_pages[0] // cp)
            physical_block_ids = torch.where(valid_pages[0], physical_block_ids, _scalar_like(physical_block_ids, 0))
            block_ids = physical_block_ids.transpose(0, 1).contiguous()
            selected_k = self.gather_paged_kv_dp(key_cache, block_ids.to(torch.int32))
            selected_k = (
                selected_k.view(local, rows, selected_len, cfg.head_dim).unsqueeze(2).repeat(1, 1, num_cores, 1, 1)
            )
            valid_core = valid_blocks.view(local, rows, selected_len).unsqueeze(2).repeat(1, 1, num_cores, 1)
            q_core = q_local.view(local, rows, num_cores, q_heads_per_core * query_len, cfg.head_dim)
            attn = torch.matmul(q_core.float(), selected_k.transpose(-1, -2).float()) * cfg.head_dim ** (-0.5)
            attn = torch.where(~valid_core.unsqueeze(3), _scalar_like(attn, MASKED_ATTENTION_LOGIT), attn)
            max_core = attn.max(dim=-1).values
            exp_core = torch.exp(attn - max_core.unsqueeze(-1))
            exp_core = torch.where(valid_core.unsqueeze(3), exp_core, _scalar_like(exp_core, 0.0))
            selected_v = (
                self.gather_paged_kv_dp(value_cache, block_ids.to(torch.int32))
                .view(local, rows, selected_len, cfg.head_dim)
                .unsqueeze(2)
                .repeat(1, 1, num_cores, 1, 1)
            )
            selected_v = torch.where(valid_core.unsqueeze(-1), selected_v, _scalar_like(selected_v, 0.0))
            sum_core = exp_core.sum(dim=-1)
            sum_core = torch.where(sum_core > 0, sum_core, torch.ones_like(sum_core))
            out_groups.append(torch.matmul(exp_core, selected_v.float()) / sum_core.unsqueeze(-1))
        out_all = torch.cat(out_groups, dim=0)
        out = (
            out_all.view(batch_local, dp, hkv, n_rep, query_len, cfg.head_dim)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(batch, cfg.num_heads, query_len, cfg.head_dim)
        )
        return (out, key_cache, value_cache)

    def msa_attention_paged_GP_batched(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        token_indices: torch.Tensor,
        token_valid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        dp = cfg.msa_attn_dp
        cp = cfg.msa_attn_cp
        hkv = cfg.num_kv_heads
        n_rep = cfg.num_kv_groups
        rows = dp * cp * hkv
        ql_eff = n_rep * query_len
        if block_table is None:
            raise ValueError("Paged MSA attention requires a block table.")
        batch_local = batch // dp
        position_ids_dp = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)
        selected_len = cfg.selected_tokens
        selected_blocks = selected_len // cfg.index_block_size
        num_cores = cfg.num_cores_per_device
        if selected_len % num_cores:
            raise ValueError("MSA selected token count must be divisible by the compile core count.")
        tokens_per_core = selected_len // num_cores
        block_ids_base = token_indices.view(dp, batch_local, hkv, selected_blocks).permute(1, 0, 2, 3)
        block_valid_base = token_valid.view(dp, batch_local, hkv, selected_blocks).permute(1, 0, 2, 3)
        offsets = torch.arange(cfg.index_block_size, device=hidden_states.device).view(1, 1, 1, 1, -1)
        global_idx = block_ids_base.unsqueeze(-1) * cfg.index_block_size + offsets
        valid_base = block_valid_base.unsqueeze(-1) & (global_idx < cfg.ctx_len)
        local_q = position_ids_dp[:, :, None, None, :] - block_ids_base.unsqueeze(-1) * cfg.index_block_size
        valid_base = valid_base & (offsets <= local_q)
        block_ids_rows = (
            block_ids_base.unsqueeze(2).expand(batch_local, dp, cp, hkv, selected_blocks)
            .reshape(batch_local, rows, selected_blocks)
        )
        valid_idx = (
            valid_base.unsqueeze(2).expand(batch_local, dp, cp, hkv, selected_blocks, cfg.index_block_size)
            .reshape(batch_local, rows, selected_len)
        )
        row_way = (torch.arange(rows, device=hidden_states.device).remainder(cp * hkv) // hkv).view(1, rows, 1)
        page_live_rows = block_ids_rows.remainder(cp) == row_way
        valid_idx = (
            valid_idx.view(batch_local, rows, selected_blocks, cfg.index_block_size) & page_live_rows.unsqueeze(-1)
        ).reshape(batch_local, rows, selected_len)
        q = self.q_norm(
            self.q_proj(hidden_states).view(dp, batch_local, query_len, cfg.num_heads, cfg.head_dim).permute(1, 0, 2, 3, 4)
        ).transpose(2, 3)
        k = self.k_norm(
            self.k_proj(hidden_states).view(dp, batch_local, query_len, hkv, cfg.head_dim).permute(1, 0, 2, 3, 4)
        ).transpose(2, 3)
        v = self.v_proj(hidden_states).view(dp, batch_local, query_len, hkv, cfg.head_dim).permute(1, 0, 2, 3, 4).transpose(2, 3)
        cos_dp = cos.view(dp, batch_local, query_len, cos.shape[-1]).permute(1, 0, 2, 3)
        sin_dp = sin.view(dp, batch_local, query_len, sin.shape[-1]).permute(1, 0, 2, 3)
        q = self._apply_rope(q, cos_dp, sin_dp)
        k = self._apply_rope(k, cos_dp, sin_dp)
        k_updates = k.unsqueeze(2).expand(
            batch_local, dp, cp, hkv, query_len, cfg.head_dim
        ).reshape(batch_local, rows, query_len, cfg.head_dim)
        v_updates = v.unsqueeze(2).expand(
            batch_local, dp, cp, hkv, query_len, cfg.head_dim
        ).reshape(batch_local, rows, query_len, cfg.head_dim)
        logical_page = position_ids_dp // cfg.page_block_size
        logical_page_group = (logical_page // cp).to(torch.int64)
        block_id_dp = torch.gather(block_table.permute(1, 0, 2), 2, logical_page_group)
        block_id_dp = block_id_dp.to(torch.int32).unsqueeze(2).unsqueeze(3).expand(
            batch_local, dp, cp, hkv, query_len
        )
        live_way = (logical_page % cp).to(torch.int64)
        way_idx = torch.arange(cp, device=hidden_states.device).view(1, 1, cp, 1, 1)
        row_live = way_idx == live_way.unsqueeze(2).unsqueeze(3)
        row_live = row_live.expand(batch_local, dp, cp, hkv, query_len)
        block_id = torch.where(
            row_live, block_id_dp, torch.iinfo(torch.int32).max
        ).reshape(batch_local, rows, query_len)
        addr_dp = position_ids_dp % cfg.page_block_size
        addr = addr_dp.to(torch.int32)[:, :, None, None, :].expand(
            batch_local, dp, cp, hkv, query_len
        ).reshape(batch_local, rows, query_len)
        key_cache = self.paged_scatter(key_cache, block_id, addr, k_updates)
        value_cache = self.paged_scatter(value_cache, block_id, addr, v_updates)
        q_rows = (
            q.view(batch_local, dp, hkv, n_rep, query_len, cfg.head_dim)
            .unsqueeze(2).expand(batch_local, dp, cp, hkv, n_rep, query_len, cfg.head_dim)
            .reshape(batch_local, rows, ql_eff, cfg.head_dim)
        )
        row_dp = torch.tensor(
            [dp_idx for dp_idx in range(dp) for _way in range(cp) for _h in range(hkv)],
            dtype=torch.int64, device=hidden_states.device
        )
        invalid_block = torch.iinfo(torch.int32).max
        local_batch_size = 1
        logical_pages = block_ids_rows.long()
        valid_blocks = valid_idx.view(batch_local, rows, selected_blocks, cfg.index_block_size)
        valid_pages = valid_blocks.float().amax(dim=-1) > 0
        table_rows = block_table.permute(1, 0, 2).index_select(1, row_dp)
        physical_block_ids = torch.gather(table_rows, 2, logical_pages // cp)
        physical_block_ids = torch.where(
            valid_pages, physical_block_ids, _scalar_like(physical_block_ids, invalid_block)
        ).to(torch.int32)
        max_core_groups: list[torch.Tensor] = []
        sum_core_groups: list[torch.Tensor] = []
        out_core_groups: list[torch.Tensor] = []
        for batch_start in range(0, batch_local, local_batch_size):
            batch_end = batch_start + local_batch_size
            selected_k_local = self.gather_paged_kv_selected_heads(key_cache, physical_block_ids[batch_start:batch_end]).reshape(1, rows, num_cores, tokens_per_core, cfg.head_dim)
            selected_v_local = self.gather_paged_kv_selected_heads(value_cache, physical_block_ids[batch_start:batch_end]).reshape(1, rows, num_cores, tokens_per_core, cfg.head_dim)
            valid_core = valid_idx[batch_start:batch_end].reshape(local_batch_size, rows, num_cores, tokens_per_core)
            q_core = q_rows[batch_start:batch_end].unsqueeze(2).expand(local_batch_size, rows, num_cores, ql_eff, cfg.head_dim)
            attn = torch.matmul(q_core.float(), selected_k_local.transpose(-1, -2).float()) * (cfg.head_dim ** -0.5)
            attn = torch.where(~valid_core.unsqueeze(3), _scalar_like(attn, MASKED_ATTENTION_LOGIT), attn)
            max_core = attn.max(dim=-1).values
            exp_core = torch.exp(attn - max_core.unsqueeze(-1))
            exp_core = torch.where(valid_core.unsqueeze(3), exp_core, _scalar_like(exp_core, 0.0))
            selected_v_local = torch.where(valid_core.unsqueeze(-1), selected_v_local, _scalar_like(selected_v_local, 0.0))
            sum_core = exp_core.sum(dim=-1)
            out_core = torch.matmul(exp_core, selected_v_local.float())
            max_core_groups.append(max_core)
            sum_core_groups.append(sum_core)
            out_core_groups.append(out_core)
        max_core = torch.cat(max_core_groups, dim=0)
        sum_core = torch.cat(sum_core_groups, dim=0)
        out_core = torch.cat(out_core_groups, dim=0)
        max_tokens = max_core.max(dim=2).values
        core_weight = torch.exp(max_core - max_tokens.unsqueeze(2))
        sum_tokens = (core_weight * sum_core).sum(dim=2)
        out_tokens = (core_weight.unsqueeze(-1) * out_core).sum(dim=2)
        max_cp = max_tokens.reshape(batch_local, dp, cp, hkv, ql_eff)
        sum_cp = sum_tokens.reshape(batch_local, dp, cp, hkv, ql_eff)
        out_cp = out_tokens.reshape(batch_local, dp, cp, hkv, ql_eff, cfg.head_dim)
        max_all = max_cp.max(dim=2).values
        cp_weight = torch.exp(max_cp - max_all.unsqueeze(2))
        sum_all = (cp_weight * sum_cp).sum(dim=2)
        out_all = (cp_weight.unsqueeze(-1) * out_cp).sum(dim=2)
        out = (out_all / sum_all.unsqueeze(-1)).reshape(
            batch_local, dp, hkv, n_rep, query_len, cfg.head_dim
        )
        out = out.permute(1, 0, 2, 3, 4, 5).reshape(batch, cfg.num_heads, query_len, cfg.head_dim)
        return (out, key_cache, value_cache)

    def msa_indexer_attention_GP_unified(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        index_key_cache: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Non-paged CP>1 decode: core top-k, MSA Q/K/V, global top-k, attention.

        Preserve the original GP layouts and cache operations. Buffer each
        indexer batch lane's core candidates without merging cores yet; the
        independent MSA projections run before one top-k across all cores and CP.
        Replicate head rows to cover the indexer rows for this final top-k.
        MSA replicates Q across cores and shards the selected K/V tokens, then
        merges the per-core and per-CP softmax statistics.
        """
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        if query_len != 1 or cfg.paged_kv or cfg.msa_attn_cp <= 1 or cfg.msa_indexer_cp <= 1:
            raise ValueError(
                "msa_indexer_attention_GP_unified requires non-paged decode "
                "(seq_len=1) with msa_attn_cp > 1 and msa_indexer_cp > 1."
            )
        if (cos is None) != (sin is None):
            raise ValueError("MSA decode requires both cos and sin inputs.")
        if cos is None:
            cos, sin = self._rope(position_ids, hidden_states.dtype)

        # 1. Indexer scoring and one top-k per core; keep the core axis.
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        dp = cfg.msa_indexer_dp
        cp = cfg.msa_indexer_cp
        hkv = cfg.indexer_n_head
        num_index_heads = cfg.index_n_heads
        dim = cfg.index_head_dim
        num_cores = cfg.num_cores_per_device
        num_kv_blocks = max(1, cfg.indexer_kv_blocks)
        assert cfg.index_local_blocks == 1, "MSA GP decode requires index_local_blocks == 1."
        batch_local = batch // dp
        rows = dp * cp * hkv
        cache_slots = cfg.ctx_len // cp
        cache_block_size = cache_slots // num_kv_blocks
        tokens_per_core = cache_block_size // num_cores
        blocks_per_core = tokens_per_core // cfg.index_block_size
        position_ids_dp = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)
        cos_dp = cos.view(dp, batch_local, query_len, cos.shape[-1]).permute(1, 0, 2, 3)
        sin_dp = sin.view(dp, batch_local, query_len, sin.shape[-1]).permute(1, 0, 2, 3)
        idx_q_all = (
            self.index_q_proj(hidden_states)
            .view(dp, batch_local, query_len, dp, num_index_heads, dim)
            .permute(1, 0, 2, 3, 4, 5)
        )
        idx_k_all = (
            self.index_k_proj(hidden_states).view(dp, batch_local, query_len, dp, hkv, dim).permute(1, 0, 2, 3, 4, 5)
        )
        q_lane_idx = (
            torch.arange(dp, device=hidden_states.device)
            .view(1, dp, 1, 1, 1, 1)
            .expand(batch_local, dp, query_len, 1, num_index_heads, dim)
        )
        k_lane_idx = (
            torch.arange(dp, device=hidden_states.device)
            .view(1, dp, 1, 1, 1, 1)
            .expand(batch_local, dp, query_len, 1, hkv, dim)
        )
        idx_q = torch.gather(idx_q_all, 3, q_lane_idx).squeeze(3).transpose(2, 3)
        idx_k = torch.gather(idx_k_all, 3, k_lane_idx).squeeze(3).transpose(2, 3)
        idx_q = self.index_q_norm(idx_q)
        idx_k = self.index_k_norm(idx_k)
        idx_q = self._apply_rope(idx_q, cos_dp[..., : cfg.index_head_dim], sin_dp[..., : cfg.index_head_dim])
        idx_k = self._apply_rope(idx_k, cos_dp[..., : cfg.index_head_dim], sin_dp[..., : cfg.index_head_dim])
        k_updates = (
            idx_k.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, query_len, dim)
            .reshape(batch_local, rows, query_len, dim)
        )
        logical_index_block = position_ids_dp // cfg.index_block_size
        live_way = (
            (logical_index_block % cp)[:, :, None, None, :]
            .expand(batch_local, dp, cp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        row_idx = torch.arange(rows, device=hidden_states.device).view(1, rows, 1)
        row_way = row_idx.remainder(cp * hkv) // hkv
        row_live = row_way == live_way
        batch_id = (
            torch.arange(batch_local, device=hidden_states.device)
            .view(batch_local, 1, 1)
            .expand(batch_local, rows, query_len)
        )
        block_id = torch.where(row_live, batch_id, torch.iinfo(torch.int32).max).to(torch.int32)
        addr = (
            (logical_index_block // cp * cfg.index_block_size + position_ids_dp % cfg.index_block_size)
            .to(torch.int32)[:, :, None, None, :]
            .expand(batch_local, dp, cp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        index_key_cache = self.paged_scatter(index_key_cache, block_id, addr, k_updates)
        q_heads_per_kv = num_index_heads // hkv
        ql_eff = q_heads_per_kv * query_len
        q_rows = (
            idx_q.reshape(batch_local, dp, hkv, q_heads_per_kv, query_len, dim)
            .unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, dim)
            .reshape(batch_local, rows, ql_eff, dim)
        )
        q_pos_rows_all = (
            position_ids_dp[:, :, 0]
            .view(batch_local, dp, 1, 1)
            .expand(batch_local, dp, cp, hkv)
            .reshape(batch_local, rows)
        )
        block_ranges: list[tuple[int, int]] = []
        for block_idx in range(num_kv_blocks):
            start = block_idx * cache_block_size
            end = min(start + cache_block_size, cache_slots)
            if start < end:
                block_ranges.append((start, end))
        core_score_groups: list[torch.Tensor] = []
        core_block_groups: list[torch.Tensor] = []
        candidate_score_groups: list[torch.Tensor] = []
        candidate_block_groups: list[torch.Tensor] = []
        local_batch_size = 1
        torch.compiler.is_exporting()

        def make_block_ids_rows(start: int, local: int) -> torch.Tensor:
            block_start = start * cp // cfg.index_block_size
            block_ids_core = (
                block_start
                + torch.arange(cp, device=hidden_states.device).view(cp, 1, 1)
                + torch.arange(num_cores, device=hidden_states.device).view(1, num_cores, 1) * (cp * blocks_per_core)
                + torch.arange(blocks_per_core, device=hidden_states.device).view(1, 1, blocks_per_core) * cp
            )
            return (
                block_ids_core.view(1, 1, cp, 1, num_cores, blocks_per_core)
                .expand(local, dp, cp, hkv, num_cores, blocks_per_core)
                .reshape(local, rows, num_cores, blocks_per_core)
            )

        for batch_start in range(0, batch_local, local_batch_size):
            batch_end = min(batch_start + local_batch_size, batch_local)
            local = batch_end - batch_start
            q_local = q_rows[batch_start:batch_end]
            key_local = index_key_cache[batch_start:batch_end]
            pos_local = position_ids_dp[batch_start:batch_end]
            q_5d = q_local.unsqueeze(2).expand(local, rows, num_cores, ql_eff, dim)
            device_block_score_groups: list[torch.Tensor] = []
            device_block_id_groups: list[torch.Tensor] = []
            q_block_rows = (
                (position_ids_dp[batch_start:batch_end, :, 0] // cfg.index_block_size)
                .view(local, dp, 1, 1)
                .expand(local, dp, cp, hkv)
                .reshape(local, rows)
            )

            def compute_block_scores(block_skip_future: torch.Tensor, key_5d: torch.Tensor) -> torch.Tensor:
                score_block = torch.matmul(q_5d.float(), key_5d.transpose(-1, -2).float())
                score_block = score_block.view(local, rows, num_cores, ql_eff, blocks_per_core, cfg.index_block_size)
                block_scores_core = score_block.amax(dim=-1)
                return block_scores_core

            prologue_start, prologue_end = block_ranges[0]
            key_flat = self._read_blocked_msa_k_dp(key_local, pos_local, prologue_start, prologue_end)
            key_5d = key_flat.view(local, rows, num_cores, tokens_per_core, dim)
            for block_idx in range(len(block_ranges) - 1):
                start, end = block_ranges[block_idx]
                block_ids_rows = make_block_ids_rows(start, local)
                block_skip_future = (
                    torch.tensor(start * cp, device=hidden_states.device, dtype=q_pos_rows_all.dtype)
                    > q_pos_rows_all[batch_start:batch_end]
                )
                block_scores_core = compute_block_scores(block_skip_future, key_5d)
                if cfg.skip_kv:
                    block_scores_core = torch.where(
                        block_skip_future.view(local, rows, 1, 1, 1),
                        _scalar_like(block_scores_core, MASKED_ATTENTION_LOGIT),
                        block_scores_core,
                    )
                next_start, next_end = block_ranges[block_idx + 1]
                next_key_flat = self._read_blocked_msa_k_dp(key_local, pos_local, next_start, next_end)
                key_5d = next_key_flat.view(local, rows, num_cores, tokens_per_core, dim)
                device_block_score_groups.append(block_scores_core)
                device_block_id_groups.append(block_ids_rows)
            epilogue_idx = len(block_ranges) - 1
            epilogue_start, epilogue_end = block_ranges[epilogue_idx]
            epilogue_block_ids = make_block_ids_rows(epilogue_start, local)
            epilogue_skip_future = (
                torch.tensor(epilogue_start * cp, device=hidden_states.device, dtype=q_pos_rows_all.dtype)
                > q_pos_rows_all[batch_start:batch_end]
            )
            epilogue_scores = compute_block_scores(epilogue_skip_future, key_5d)
            device_block_score_groups.append(epilogue_scores)
            device_block_id_groups.append(epilogue_block_ids)
            device_block_scores = torch.cat(device_block_score_groups, dim=4)
            device_block_ids = (
                torch.cat(device_block_id_groups, dim=3)
                .unsqueeze(3)
                .expand(local, rows, num_cores, ql_eff, num_kv_blocks * blocks_per_core)
            )
            future_block = device_block_ids >= q_block_rows.view(local, rows, 1, 1, 1)
            device_block_scores = torch.where(
                future_block, _scalar_like(device_block_scores, MASKED_ATTENTION_LOGIT), device_block_scores
            )
            core_topk = min(cfg.selected_blocks, device_block_scores.shape[-1])
            core_topk_scores, core_topk_indices = torch.topk(device_block_scores, k=core_topk, dim=-1)
            core_topk_block_ids = torch.gather(device_block_ids, -1, core_topk_indices)
            core_score_groups.append(core_topk_scores)
            core_block_groups.append(core_topk_block_ids)

        # 2. MSA Q/K/V do not depend on selected blocks. Keep their original layout.
        attn_dp, attn_hkv = cfg.msa_attn_dp, cfg.num_kv_heads
        attn_batch_local = batch // attn_dp
        q = self.q_norm(
            self.q_proj(hidden_states)
            .view(attn_dp, attn_batch_local, query_len, cfg.num_heads, cfg.head_dim)
            .permute(1, 0, 2, 3, 4)
        ).transpose(2, 3)
        k = self.k_norm(
            self.k_proj(hidden_states).view(attn_dp, attn_batch_local, query_len, attn_hkv, cfg.head_dim).permute(1, 0, 2, 3, 4)
        ).transpose(2, 3)
        v = (
            self.v_proj(hidden_states)
            .view(attn_dp, attn_batch_local, query_len, attn_hkv, cfg.head_dim)
            .permute(1, 0, 2, 3, 4)
            .transpose(2, 3)
        )
        attn_cos = cos.view(attn_dp, attn_batch_local, query_len, cos.shape[-1]).permute(1, 0, 2, 3)
        attn_sin = sin.view(attn_dp, attn_batch_local, query_len, sin.shape[-1]).permute(1, 0, 2, 3)
        q = self._apply_rope(q, attn_cos, attn_sin)
        k = self._apply_rope(k, attn_cos, attn_sin)

        # 3. Merge every core's candidates across CP, then select once globally.
        for core_topk_scores, core_topk_block_ids in zip(core_score_groups, core_block_groups):
            local = core_topk_scores.shape[0]
            core_candidate_scores = core_topk_scores.permute(0, 1, 3, 2, 4).reshape(
                local, rows, ql_eff, num_cores * core_topk
            )
            core_candidate_block_ids = core_topk_block_ids.permute(0, 1, 3, 2, 4).reshape(
                local, rows, ql_eff, num_cores * core_topk
            )
            candidate_score_groups.append(core_candidate_scores)
            candidate_block_groups.append(core_candidate_block_ids)
        core_candidate_width = num_cores * core_topk
        candidate_scores = (
            torch.cat(candidate_score_groups, dim=0)
            .view(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, core_candidate_width)
            .permute(0, 1, 3, 4, 5, 2, 6)
            .reshape(batch, num_index_heads, cp * core_candidate_width)
        )
        candidate_block_indices = (
            torch.cat(candidate_block_groups, dim=0)
            .view(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, core_candidate_width)
            .permute(0, 1, 3, 4, 5, 2, 6)
            .reshape(batch, num_index_heads, cp * core_candidate_width)
        )
        # Tile [head0, head1, ...] until the final TopK has enough device rows.
        # M3 DP1/CP16: [B, 4, 1024] -> [B, 16, 1024] -> [B, 16, 16].
        topk_rows = max(rows, num_index_heads)
        row_repeats = (topk_rows + num_index_heads - 1) // num_index_heads
        topk_input = candidate_scores
        if row_repeats > 1:
            topk_input = candidate_scores.repeat(1, row_repeats, 1)[:, :topk_rows, :]
        topk_scores, candidate_topk = torch.topk(topk_input, k=cfg.selected_blocks, dim=-1)
        if row_repeats > 1:
            # Each repeated row has the same candidates; restore the original
            # head layout before the block-ID gather and sparse attention.
            topk_scores = topk_scores[:, :num_index_heads, :]
            candidate_topk = candidate_topk[:, :num_index_heads, :]
        block_indices = torch.gather(candidate_block_indices, -1, candidate_topk)
        topk_scores = topk_scores.view(batch_local, dp, hkv, q_heads_per_kv, query_len, cfg.selected_blocks)
        block_indices = block_indices.view(batch_local, dp, hkv, q_heads_per_kv, query_len, cfg.selected_blocks)
        current_block_ids = (
            (position_ids.view(dp, batch_local, query_len).permute(1, 0, 2) // cfg.index_block_size)
            .view(batch_local, dp, 1, 1, query_len, 1)
            .expand(batch_local, dp, hkv, q_heads_per_kv, query_len, 1)
            .to(block_indices.dtype)
        )
        block_indices = torch.cat([block_indices[..., :-1], current_block_ids], dim=-1)
        topk_scores = torch.cat([topk_scores[..., :-1], torch.zeros_like(topk_scores[..., -1:])], dim=-1)
        topk_scores = (
            topk_scores.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, cfg.selected_blocks)
            .permute(1, 0, 2, 3, 4, 5, 6)
            .reshape(batch, cp, num_index_heads, query_len, cfg.selected_blocks)
        )
        block_indices = (
            block_indices.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, cfg.selected_blocks)
            .permute(1, 0, 2, 3, 4, 5, 6)
            .reshape(batch, cp, num_index_heads, query_len, cfg.selected_blocks)
        )
        block_valid = topk_scores > MASKED_ATTENTION_LOGIT / 2
        block_indices = block_indices[..., 0, :]
        block_valid = block_valid[..., 0, :]
        safe_indices = torch.where(block_valid, block_indices, torch.zeros_like(block_indices)).to(torch.int32)
        if not torch.compiler.is_exporting():
            self.last_block_indices = block_indices.detach()
        token_indices, token_valid = safe_indices, block_valid

        # 4. Keep the CP>1 cache/row layout; split selected K/V tokens across cores.
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        dp, cp, hkv = cfg.msa_attn_dp, cfg.msa_attn_cp, cfg.num_kv_heads
        n_rep = cfg.num_kv_groups
        batch_local = batch // dp
        rows = dp * cp * hkv
        ql_eff = n_rep * query_len
        if cfg.ctx_len % cp:
            raise ValueError("Non-paged MSA ctx_len must be divisible by msa_attn_cp.")
        cache_slots = cfg.ctx_len // cp
        expected_cache_shape = (batch_local, rows, cache_slots, cfg.head_dim)
        if tuple(key_cache.shape) != expected_cache_shape or tuple(value_cache.shape) != expected_cache_shape:
            raise ValueError(
                "Compact MSA key/value cache must both have shape "
                f"{expected_cache_shape}."
            )
        num_cores = cfg.num_cores_per_device
        if cfg.selected_tokens % num_cores:
            raise ValueError("MSA selected token count must be divisible by the compile core count.")
        tokens_per_core = cfg.selected_tokens // num_cores

        if token_indices.ndim == 4:
            selector_index_shape = (
                batch,
                cfg.msa_indexer_cp,
                hkv,
                cfg.selected_blocks,
            )
            if tuple(token_indices.shape) != selector_index_shape:
                raise ValueError(
                    f"MSA token_indices shape {tuple(token_indices.shape)} != {selector_index_shape}."
                )
            if tuple(token_valid.shape) != selector_index_shape:
                raise ValueError(
                    f"MSA token_valid shape {tuple(token_valid.shape)} != {selector_index_shape}."
                )
            token_indices = token_indices[:, 0]
            token_valid = token_valid[:, 0]

        compressed_block_shape = (batch, hkv, cfg.selected_blocks)
        if tuple(token_indices.shape) != compressed_block_shape or tuple(token_valid.shape) != compressed_block_shape:
            raise ValueError(f"MSA CP>1 requires block indices/valid with shape {compressed_block_shape}.")
        if cfg.index_block_size % cp:
            raise ValueError("MSA index_block_size must be divisible by msa_attn_cp.")

        block_ids = token_indices.view(dp, batch_local, hkv, cfg.selected_blocks).permute(1, 0, 2, 3)
        block_valid = token_valid.view(dp, batch_local, hkv, cfg.selected_blocks).permute(1, 0, 2, 3)
        token_offset = torch.arange(cfg.index_block_size, device=hidden_states.device).view(1, 1, 1, 1, -1)
        global_idx = block_ids.unsqueeze(-1) * cfg.index_block_size + token_offset
        position_limit = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)[:, :, None, None, :]
        valid_base = block_valid.unsqueeze(-1) & (global_idx < cfg.ctx_len)
        local_q = position_limit - block_ids.unsqueeze(-1) * cfg.index_block_size
        valid_base = valid_base & (token_offset <= local_q)
        local_offset = torch.tensor(
            [offset // cp for offset in range(cfg.index_block_size)],
            dtype=torch.int32,
            device=hidden_states.device,
        ).view(1, 1, 1, 1, -1)
        offset_way = torch.tensor(
            [offset % cp for offset in range(cfg.index_block_size)],
            dtype=torch.int64,
            device=hidden_states.device,
        ).view(1, 1, 1, 1, -1)
        selected_len = cfg.selected_tokens
        gather_local_base = (
            block_ids.unsqueeze(-1) * (cfg.index_block_size // cp) + local_offset
        ).reshape(batch_local, dp, hkv, selected_len)
        valid_base = valid_base.reshape(batch_local, dp, hkv, selected_len)
        gather_idx = (
            gather_local_base.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, selected_len)
            .reshape(batch_local, rows, selected_len)
            .to(torch.int32)
        )
        valid_idx = (
            valid_base.unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, selected_len)
            .reshape(batch_local, rows, selected_len)
        )
        offset_way = offset_way.expand(1, 1, 1, cfg.selected_blocks, cfg.index_block_size).reshape(
            1, 1, 1, 1, selected_len
        )
        way_idx = torch.arange(cp, device=hidden_states.device).view(1, 1, cp, 1, 1)
        gather_live = (
            (offset_way == way_idx)
            .expand(batch_local, dp, cp, hkv, selected_len)
            .reshape(batch_local, rows, selected_len)
        )
        valid_idx = valid_idx & gather_live
        gather_idx = torch.where(
            valid_idx,
            gather_idx,
            _scalar_like(gather_idx, torch.iinfo(torch.int32).max),
        ).to(torch.int32)

        k_updates = k.unsqueeze(2).expand(
            batch_local, dp, cp, hkv, query_len, cfg.head_dim
        ).reshape(batch_local, rows, query_len, cfg.head_dim)
        v_updates = v.unsqueeze(2).expand(
            batch_local, dp, cp, hkv, query_len, cfg.head_dim
        ).reshape(batch_local, rows, query_len, cfg.head_dim)
        position_ids_dp = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)
        row_way = torch.tensor(
            [way for _dp in range(dp) for way in range(cp) for _h in range(hkv)],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        addr_base = position_ids_dp // cp
        live_way = (
            (position_ids_dp - addr_base * cp)
            .view(batch_local, dp, 1, 1, query_len)
            .expand(batch_local, dp, cp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        row_way_scatter = row_way.view(1, rows, 1)
        batch_id = torch.arange(batch_local, device=hidden_states.device).view(batch_local, 1, 1).expand(
            batch_local, rows, query_len
        )
        block_id = torch.where(
            row_way_scatter == live_way,
            batch_id,
            torch.iinfo(torch.int32).max,
        ).to(torch.int32)
        addr = (
            addr_base.to(torch.int32)
            .view(batch_local, dp, 1, 1, query_len)
            .expand(batch_local, dp, cp, hkv, query_len)
            .reshape(batch_local, rows, query_len)
        )
        key_cache = self.paged_scatter(key_cache, block_id, addr, k_updates)
        value_cache = self.paged_scatter(value_cache, block_id, addr, v_updates)

        q_rows = (
            q.view(batch_local, dp, hkv, n_rep, query_len, cfg.head_dim)
            .unsqueeze(2)
            .expand(batch_local, dp, cp, hkv, n_rep, query_len, cfg.head_dim)
            .reshape(batch_local, rows, ql_eff, cfg.head_dim)
        )
        local_batch_size = 1
        out_groups: list[torch.Tensor] = []
        for batch_start in range(0, batch_local, local_batch_size):
            batch_end = min(batch_start + local_batch_size, batch_local)
            local = batch_end - batch_start
            gather_local = gather_idx[batch_start:batch_end]
            valid_local = valid_idx[batch_start:batch_end]
            selected_k = self.gather_blocked_kv_dp(
                key_cache[batch_start:batch_end], gather_local
            ).reshape(local, rows, num_cores, tokens_per_core, cfg.head_dim)
            selected_v = self.gather_blocked_kv_dp(
                value_cache[batch_start:batch_end], gather_local
            ).reshape(local, rows, num_cores, tokens_per_core, cfg.head_dim)
            valid_core = valid_local.reshape(local, rows, num_cores, tokens_per_core)
            q_core = q_rows[batch_start:batch_end].unsqueeze(2).expand(
                local, rows, num_cores, ql_eff, cfg.head_dim
            )
            attn = torch.matmul(q_core.float(), selected_k.transpose(-1, -2).float()) * (cfg.head_dim ** -0.5)
            attn = torch.where(
                ~valid_core.unsqueeze(3), _scalar_like(attn, MASKED_ATTENTION_LOGIT), attn
            )
            max_core = attn.max(dim=-1).values
            exp_core = torch.exp(attn - max_core.unsqueeze(-1))
            exp_core = torch.where(valid_core.unsqueeze(3), exp_core, _scalar_like(exp_core, 0.0))
            selected_v = torch.where(valid_core.unsqueeze(-1), selected_v, _scalar_like(selected_v, 0.0))
            sum_core = exp_core.sum(dim=-1)
            out_core = torch.matmul(exp_core, selected_v.float())
            # Flatten local and DP into one batch axis before either reduction.
            # This keeps the reduction inputs below the compiler's 7-D limit:
            #   [local, dp, cp, core, q] -> [local*dp, cp, core, q]
            # All cores handle the same Q heads over disjoint token slices.
            batch_dp = local * dp
            max_core = max_core.reshape(batch_dp, cp, hkv, num_cores, ql_eff)
            sum_core = sum_core.reshape(batch_dp, cp, hkv, num_cores, ql_eff)
            out_core = out_core.reshape(batch_dp, cp, hkv, num_cores, ql_eff, cfg.head_dim)

            # First reduce the core-sharded K/V slices locally on each device.
            max_tokens = max_core.max(dim=3).values
            core_weight = torch.exp(max_core - max_tokens.unsqueeze(3))
            sum_tokens = (core_weight * sum_core).sum(dim=3)
            out_tokens = (core_weight.unsqueeze(-1) * out_core).sum(dim=3)

            # Then reduce the CP/device-sharded results across devices.
            max_all = max_tokens.max(dim=1).values
            cp_weight = torch.exp(max_tokens - max_all.unsqueeze(1))
            sum_all = (cp_weight * sum_tokens).sum(dim=1)
            out_all = (cp_weight.unsqueeze(-1) * out_tokens).sum(dim=1)
            sum_all = torch.where(sum_all > 0, sum_all, torch.ones_like(sum_all))
            out_groups.append(out_all / sum_all.unsqueeze(-1))

        out_all = torch.cat(out_groups, dim=0)
        out = (
            out_all.view(batch_local, dp, hkv, n_rep, query_len, cfg.head_dim)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(batch, cfg.num_heads, query_len, cfg.head_dim)
        )

        out = (
            out.to(hidden_states.dtype)
            .reshape(batch, cfg.num_heads, query_len, cfg.head_dim)
            .transpose(1, 2)
            .reshape(batch, query_len, -1)
        )
        return (self.o_proj(out), key_cache, value_cache, index_key_cache)

    def _write_msa_paged_prefill_cache(
        self, cache: torch.Tensor, updates: torch.Tensor, position_ids: torch.Tensor, block_table: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        batch, query_len = position_ids.shape
        rows = updates.shape[1]
        page_size = cfg.page_block_size
        num_pages = query_len // page_size
        logical_pages = (position_ids[:, ::page_size] // page_size).to(torch.int64)
        physical_pages = torch.gather(block_table[0].to(torch.int64), 1, logical_pages).to(torch.int32)
        for batch_idx in range(batch):
            block_ids = physical_pages[batch_idx].view(num_pages, 1).expand(num_pages, rows)
            page_updates = updates[batch_idx].reshape(rows, num_pages, page_size, updates.shape[-1]).permute(1, 0, 2, 3)
            cache = self.paged_scatter_page(cache, block_ids, page_updates)
        return cache

    def _select_blocks_prefill_paged(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        index_key_cache: torch.Tensor,
        block_table: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        if not cfg.indexer_prefill_parallel:
            raise NotImplementedError("only --indexer-prefill-parallel is ported")
        selector = self._select_blocks_prefill_par
        token_indices, token_valid, index_key_cache = selector(
            hidden_states, position_ids, index_key_cache, cos, sin, paged_block_table=block_table[0]
        )
        return (token_indices, token_valid, index_key_cache)

    def _msa_attention_prefill_paged(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        token_indices: torch.Tensor,
        token_valid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch, query_len, _ = hidden_states.shape
        k = self.k_norm(
            self.k_proj(hidden_states).view(batch, query_len, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        )
        v = self.v_proj(hidden_states).view(batch, query_len, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        k = self._apply_rope(k, cos, sin)
        key_cache = self._write_msa_paged_prefill_cache(key_cache, k, position_ids, block_table)
        value_cache = self._write_msa_paged_prefill_cache(value_cache, v, position_ids, block_table)
        out, _, _ = self.msa_attention_prefill(
            hidden_states,
            position_ids,
            key_cache,
            value_cache,
            token_indices,
            token_valid,
            cos,
            sin,
            paged_block_table=block_table[0],
        )
        return (out, key_cache, value_cache)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        index_key_cache: torch.Tensor,
        indexer_block_table: torch.Tensor | None = None,
        attention_block_table: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        cfg = self.config
        if cfg.impl == "msa_indexer_attention_GP_unified":
            return self.msa_indexer_attention_GP_unified(
                hidden_states, position_ids, key_cache, value_cache, index_key_cache,
                cos, sin,
            )
        batch = hidden_states.shape[0]
        query_len = hidden_states.shape[1]
        if (cos is None) != (sin is None):
            raise ValueError("MSA attention requires both cos and sin inputs.")
        if cos is None:
            cos, sin = self._rope(position_ids, hidden_states.dtype)
        if query_len > 1 and cfg.paged_kv:
            token_indices, token_valid, index_key_cache = self._select_blocks_prefill_paged(
                hidden_states, position_ids, index_key_cache, indexer_block_table, cos, sin
            )
        elif query_len > 1:
            if not cfg.indexer_prefill_parallel:
                raise NotImplementedError("only --indexer-prefill-parallel is ported")
            token_indices, token_valid, index_key_cache = self._select_blocks_prefill_par(
                hidden_states, position_ids, index_key_cache, cos, sin
            )
        elif cfg.paged_kv:
            token_indices, token_valid, index_key_cache = self._select_blocks_paged(
                hidden_states, position_ids, index_key_cache, indexer_block_table, cos, sin
            )
        else:
            token_indices, token_valid, index_key_cache = self._select_blocks(
                hidden_states, position_ids, index_key_cache, cos, sin
            )
        if query_len > 1:
            if cfg.paged_kv:
                out, key_cache, value_cache = self._msa_attention_prefill_paged(
                    hidden_states,
                    position_ids,
                    key_cache,
                    value_cache,
                    token_indices,
                    token_valid,
                    cos,
                    sin,
                    attention_block_table,
                )
            else:
                out, key_cache, value_cache = self.msa_attention_prefill(
                    hidden_states,
                    position_ids,
                    key_cache,
                    value_cache,
                    token_indices,
                    token_valid,
                    cos,
                    sin,
                )
        elif cfg.impl in ("msa_attention_GP", "msa_attention_GP_batched"):
            if cfg.paged_kv:
                paged_gp = (
                    self.msa_attention_paged_GP_batched
                    if cfg.impl == "msa_attention_GP_batched"
                    else self.msa_attention_paged_GP
                )
                out, key_cache, value_cache = paged_gp(
                    hidden_states,
                    position_ids,
                    key_cache,
                    value_cache,
                    token_indices,
                    token_valid,
                    cos,
                    sin,
                    attention_block_table,
                )
            else:
                gp = (
                    self.msa_attention_GP_batched
                    if cfg.impl == "msa_attention_GP_batched"
                    else self.msa_attention_GP
                )
                out, key_cache, value_cache = gp(
                    hidden_states, position_ids, key_cache, value_cache, token_indices, token_valid, cos, sin
                )
        out = (
            out.to(hidden_states.dtype)
            .reshape(batch, cfg.num_heads, query_len, cfg.head_dim)
            .transpose(1, 2)
            .reshape(batch, query_len, -1)
        )
        return (self.o_proj(out), key_cache, value_cache, index_key_cache)


class MiniMaxM3GQADecodeAttention(MiniMaxM3MSAAttentionCache, nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.scaling = config.head_dim ** (-0.5)
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = GemmaRMSNorm(config.head_dim, config.norm_eps)
        self.k_norm = GemmaRMSNorm(config.head_dim, config.norm_eps)
        self.register_buffer(
            "inv_freq",
            1.0 / config.rope_theta ** (torch.arange(0, config.rotary_dim, 2).float() / config.rotary_dim),
            persistent=False,
        )
        paged_local_pos = None
        self.register_buffer("paged_local_pos", paged_local_pos)

    def _rope(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos().to(dtype), emb.sin().to(dtype))

    def _read_blocked_k_dp(
        self, key_cache: torch.Tensor, position_ids_dp: torch.Tensor, start_index: int, end_index: int
    ) -> torch.Tensor:
        cfg = self.config
        batch_local, rows, _, _ = key_cache.shape
        dp = position_ids_dp.shape[1]
        heads_per_dp = rows // dp
        hkv = heads_per_dp // cfg.attn_cp
        block_len = end_index - start_index
        pos_max = position_ids_dp.max(dim=-1).values
        way = torch.arange(rows, device=key_cache.device).remainder(cfg.attn_cp * hkv) // hkv
        gather_limit = ((pos_max[:, :, None] - way.view(1, dp, hkv * cfg.attn_cp)) // cfg.attn_cp).reshape(
            batch_local, rows, 1
        )
        ctx_indices = torch.arange(start=start_index, end=end_index, device=key_cache.device)[None, None, :]
        invalid_mask = ctx_indices > gather_limit
        invalid_idx_value = torch.iinfo(torch.int32).max if torch.compiler.is_exporting() else 0
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices).to(torch.int32)
        ctx_indices = ctx_indices.expand(batch_local, rows, block_len)
        return self.gather_blocked_kv_dp(key_cache, ctx_indices)

    def _read_blocked_v_dp(
        self, value_cache: torch.Tensor, position_ids_dp: torch.Tensor, start_index: int, end_index: int
    ) -> torch.Tensor:
        cfg = self.config
        batch_local, rows, _, _ = value_cache.shape
        dp = position_ids_dp.shape[1]
        heads_per_dp = rows // dp
        hkv = heads_per_dp // cfg.attn_cp
        block_len = end_index - start_index
        pos_max = position_ids_dp.max(dim=-1).values
        way = torch.arange(rows, device=value_cache.device).remainder(cfg.attn_cp * hkv) // hkv
        gather_limit = (pos_max[:, :, None] - way.view(1, dp, hkv * cfg.attn_cp)) // cfg.attn_cp
        gather_limit = gather_limit.reshape(batch_local, rows, 1)
        ctx_indices = torch.arange(start=start_index, end=end_index, device=value_cache.device)[None, None, :]
        invalid_mask = ctx_indices > gather_limit
        invalid_idx_value = torch.iinfo(torch.int32).max if torch.compiler.is_exporting() else 0
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices).to(torch.int32)
        ctx_indices = ctx_indices.expand(batch_local, rows, block_len)
        value_block = self.gather_blocked_kv_dp(value_cache, ctx_indices)
        return torch.where(
            invalid_mask.unsqueeze(-1),
            torch.tensor(0.0, dtype=value_block.dtype, device=value_block.device),
            value_block,
        )

    def _read_blocked_k(
        self, key_cache: torch.Tensor, position_ids: torch.Tensor, start_index: int, end_index: int
    ) -> torch.Tensor:
        batch, num_kv_heads, _, _ = key_cache.shape
        ctx_indices = torch.arange(start=start_index, end=end_index, device=key_cache.device)[None, None, :]
        gather_limit = position_ids.max(dim=1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit
        invalid_idx_value = torch.iinfo(torch.int32).max if torch.compiler.is_exporting() else 0
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices).to(torch.int32)
        ctx_indices = ctx_indices.expand(batch, num_kv_heads, end_index - start_index)
        return self.gather_blocked_kv(key_cache, ctx_indices)

    def _read_blocked_v(
        self, value_cache: torch.Tensor, position_ids: torch.Tensor, start_index: int, end_index: int
    ) -> torch.Tensor:
        batch, num_kv_heads, _, _ = value_cache.shape
        ctx_indices = torch.arange(start=start_index, end=end_index, device=value_cache.device)[None, None, :]
        gather_limit = position_ids.max(dim=1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit
        invalid_idx_value = torch.iinfo(torch.int32).max if torch.compiler.is_exporting() else 0
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices).to(torch.int32)
        ctx_indices = ctx_indices.expand(batch, num_kv_heads, end_index - start_index)
        value_block = self.gather_blocked_kv(value_cache, ctx_indices)
        return torch.where(
            invalid_mask.unsqueeze(-1),
            torch.tensor(0.0, dtype=value_block.dtype, device=value_block.device),
            value_block,
        )

    def _kv_blocked_attn_prefill(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        position_ids: torch.Tensor,
        ql_chunk: int = 64,
        q_block_chunk: int | None = None,
        n_rep_chunk: int = 1,
    ) -> torch.Tensor:
        cfg = self.config
        batch, num_heads, query_len, head_dim = q.shape
        num_kv_heads = cfg.num_kv_heads
        num_cores = cfg.num_cores_per_device
        kv_repeat = num_cores // num_kv_heads
        n_rep_per_core = num_heads // num_cores
        q_block_chunk = ql_chunk if q_block_chunk is None else q_block_chunk
        num_kv_blocks = max(1, cfg.num_kv_blocks)
        kv_block_size = -(-cfg.ctx_len // num_kv_blocks)
        q_fold = q.reshape(batch, num_cores, n_rep_per_core, query_len, head_dim)
        is_export = torch.compiler.is_exporting()
        rep_ranges = [
            (start, min(start + n_rep_chunk, n_rep_per_core)) for start in range(0, n_rep_per_core, n_rep_chunk)
        ]
        output_chunks: list[torch.Tensor] = []
        for q_chunk_start in range(0, query_len, q_block_chunk):
            q_chunk_end = min(q_chunk_start + q_block_chunk, query_len)
            q_chunk_position = position_ids[:, q_chunk_start:q_chunk_end].max(dim=-1).values
            q_states: dict[int, dict[str, Any]] = {}
            for q_block_start in range(q_chunk_start, q_chunk_end, ql_chunk):
                q_block_end = min(q_block_start + ql_chunk, q_chunk_end)
                q_block_len = q_block_end - q_block_start
                pos_sub = position_ids[:, q_block_start:q_block_end]
                accumulators: list[dict[str, Any]] = []
                for rep_start, rep_end in rep_ranges:
                    rep_count = rep_end - rep_start
                    accumulators.append(
                        {
                            "rep_count": rep_count,
                            "q": q_fold[:, :, rep_start:rep_end, q_block_start:q_block_end, :].reshape(
                                batch, num_cores, rep_count * q_block_len, head_dim
                            ),
                            "max": torch.full(
                                (batch, num_cores, rep_count * q_block_len),
                                MIN_MASKED_ATTENTION_VALUE,
                                device=q.device,
                                dtype=q.dtype,
                            ),
                            "sum": torch.zeros(
                                batch, num_cores, rep_count * q_block_len, device=q.device, dtype=q.dtype
                            ),
                            "out": torch.zeros(
                                batch, num_cores, rep_count * q_block_len, head_dim, device=q.device, dtype=q.dtype
                            ),
                        }
                    )
                q_states[q_block_start] = {
                    "block_len": q_block_len,
                    "positions": pos_sub,
                    "current_position": pos_sub.max(dim=-1).values,
                    "accumulators": accumulators,
                }
            for block_idx in range(num_kv_blocks):
                start_index = block_idx * kv_block_size
                end_index = min(start_index + kv_block_size, cfg.ctx_len)
                if (
                    cfg.skip_kv
                    and (not is_export)
                    and bool((torch.tensor(start_index, device=q.device) > q_chunk_position).all().item())
                ):
                    break
                key_block = self._read_blocked_k(key_cache, position_ids, start_index, end_index).repeat_interleave(
                    kv_repeat, dim=1
                )
                value_block = self._read_blocked_v(value_cache, position_ids, start_index, end_index).repeat_interleave(
                    kv_repeat, dim=1
                )
                key_positions = torch.arange(start_index, end_index, device=q.device)
                key_block_t = key_block.transpose(2, 3)
                for q_block_start in range(q_chunk_start, q_chunk_end, ql_chunk):
                    q_state = q_states[q_block_start]
                    # float amin: bool .all() exports as an int64 ReduceMin
                    skip_future = (q_state["current_position"].float() < start_index).float().amin(dim=0) > 0
                    q_block_len = q_state["block_len"]
                    causal = key_positions[None, None, None, :] > q_state["positions"][:, None, :, None]
                    for accumulator in q_state["accumulators"]:
                        rep_count = accumulator["rep_count"]
                        attn = torch.matmul(accumulator["q"], key_block_t) * self.scaling
                        attn = attn.view(batch, num_cores, rep_count, q_block_len, -1)
                        attn = torch.where(
                            causal.unsqueeze(2), _scalar_like(attn, MIN_MASKED_ATTENTION_VALUE), attn
                        ).view(batch, num_cores, rep_count * q_block_len, -1)
                        accumulator["max"], accumulator["sum"], accumulator["out"] = update_running_softmax(
                            accumulator["max"],
                            attn,
                            accumulator["sum"],
                            accumulator["out"],
                            value_block,
                            cfg.skip_kv,
                            skip_future,
                        )
            q_block_outputs: list[torch.Tensor] = []
            for q_block_start in range(q_chunk_start, q_chunk_end, ql_chunk):
                q_state = q_states[q_block_start]
                q_block_len = q_state["block_len"]
                rep_outputs: list[torch.Tensor] = []
                for accumulator in q_state["accumulators"]:
                    denominator = torch.where(
                        accumulator["sum"] > 0, accumulator["sum"], torch.ones_like(accumulator["sum"])
                    )
                    out = accumulator["out"] / denominator.unsqueeze(-1)
                    rep_outputs.append(out.view(batch, num_cores, accumulator["rep_count"], q_block_len, head_dim))
                q_block_outputs.append(torch.cat(rep_outputs, dim=2))
            output_chunks.append(torch.cat(q_block_outputs, dim=3))
        return torch.cat(output_chunks, dim=3).reshape(batch, num_heads, query_len, head_dim)

    def _kv_blocked_attn_decode(
        self, q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        B, q_rows, QL, D = q.shape
        DP = cfg.attn_dp
        N = cfg.attn_cp
        Hkv = cfg.num_kv_heads
        NQH = cfg.num_heads
        n_rep = cfg.num_kv_groups
        C = cfg.num_cores_per_device
        num_kv_blocks = max(1, cfg.num_kv_blocks)
        batch_local = B // DP
        ql_eff = n_rep * QL
        rows = DP * N * Hkv
        cache_slots = cfg.ctx_len // N
        cache_block_size = cache_slots // num_kv_blocks
        T_h = cache_block_size // C

        def compute_kv_len_block(block_idx: int, start_index: int) -> int:
            del block_idx
            return min(start_index + cache_block_size, cache_slots) - start_index

        def compute_kv_block_range(block_idx: int) -> tuple[int, int, int]:
            start_index = block_idx * cache_block_size
            kv_len_block = compute_kv_len_block(block_idx, start_index)
            end_index = start_index + kv_len_block
            return (start_index, end_index, kv_len_block)

        q_dp = q.view(DP, batch_local, NQH, N, QL, D).permute(1, 0, 2, 3, 4, 5)
        q_base = q_dp[:, :, :, 0]
        q_rows_flat = (
            q_base.reshape(batch_local, DP, Hkv, ql_eff, D)
            .unsqueeze(2)
            .expand(batch_local, DP, N, Hkv, ql_eff, D)
            .reshape(batch_local, rows, ql_eff, D)
        )
        pos_dp = position_ids.view(DP, batch_local, QL).permute(1, 0, 2)
        way_rows = torch.arange(rows, device=q.device).remainder(N * Hkv) // Hkv
        local_batch_size = 1
        q_pos_rows_all = pos_dp[:, :, 0].unsqueeze(-1).expand(batch_local, DP, N * Hkv).reshape(batch_local, rows)
        row_way_all = way_rows.view(1, rows, 1, 1)
        q_slot_rows_all = torch.where(
            q_pos_rows_all.unsqueeze(-1).unsqueeze(-1) >= row_way_all,
            (q_pos_rows_all.unsqueeze(-1).unsqueeze(-1) - row_way_all) // N,
            torch.full_like(q_pos_rows_all.unsqueeze(-1).unsqueeze(-1), -1),
        )
        block_idx_all = torch.arange(num_kv_blocks, dtype=q_pos_rows_all.dtype, device=q.device).view(
            1, 1, 1, num_kv_blocks
        )
        block_start_all = block_idx_all * cache_block_size
        relative_slot_all = q_slot_rows_all - block_start_all
        core_start_all = torch.arange(C, dtype=q_pos_rows_all.dtype, device=q.device).view(1, 1, C, 1) * T_h
        core_end_all = core_start_all + T_h - 1
        threshold_all = torch.where(
            relative_slot_all >= core_end_all,
            torch.full_like(relative_slot_all, T_h - 1),
            torch.where(
                relative_slot_all >= core_start_all,
                relative_slot_all - core_start_all,
                torch.full_like(relative_slot_all, -1),
            ),
        )
        max_groups: list[torch.Tensor] = []
        sum_groups: list[torch.Tensor] = []
        out_groups: list[torch.Tensor] = []
        for batch_start in range(0, batch_local, local_batch_size):
            batch_end = min(batch_start + local_batch_size, batch_local)
            local = batch_end - batch_start
            q_local = q_rows_flat[batch_start:batch_end]
            pos_local = pos_dp[batch_start:batch_end]
            key_local = key_cache[batch_start:batch_end]
            value_local = value_cache[batch_start:batch_end]
            q_flat = q_local.unsqueeze(2).expand(local, rows, C, ql_eff, D)
            max_buf: list[torch.Tensor] = []
            sum_buf: list[torch.Tensor] = []
            out_buf: list[torch.Tensor] = []

            def compute_block(
                start_index: int, T_orig: int, skip_future: torch.Tensor, K_flat: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                attn = torch.matmul(q_flat, K_flat.transpose(-1, -2)) * self.scaling
                block_idx = start_index // cache_block_size
                threshold = threshold_all[batch_start:batch_end, :, :, block_idx]
                causal_mask = torch.arange(T_orig, device=q.device).view(1, 1, 1, T_orig) > threshold.unsqueeze(-1)
                causal_mask = causal_mask.unsqueeze(3).repeat(1, 1, 1, n_rep, 1)
                attn = attn.masked_fill(causal_mask, float(-30000.0))
                m_blk = attn.max(dim=-1).values
                exp_blk = torch.exp(attn - m_blk.unsqueeze(-1))
                sum_blk = exp_blk.sum(dim=-1)
                if cfg.skip_kv:
                    m_blk = torch.where(skip_future, _scalar_like(m_blk, MASKED_ATTENTION_LOGIT), m_blk)
                    exp_blk = torch.where(skip_future.unsqueeze(-1), _scalar_like(exp_blk, 0.0), exp_blk)
                    sum_blk = torch.where(skip_future, _scalar_like(sum_blk, 0.0), sum_blk)
                return (m_blk, exp_blk, sum_blk)

            prologue_start_index, prologue_end_index, prologue_kv_len_block = compute_kv_block_range(0)
            K_flat = self._read_blocked_k_dp(key_local, pos_local, prologue_start_index, prologue_end_index).view(
                local, rows, C, prologue_kv_len_block // C, D
            )
            for j in range(num_kv_blocks - 1):
                start_index, end_index, kv_len_block = compute_kv_block_range(j)
                T_orig = kv_len_block // C
                skip_future = None
                if cfg.skip_kv:
                    threshold = threshold_all[batch_start:batch_end, :, :, j]
                    skip_future = (threshold < 0).unsqueeze(-1)
                    if not torch.compiler.is_exporting():
                        if skip_future.all().item():
                            break
                m_blk, exp_blk, sum_blk = compute_block(start_index, T_orig, skip_future, K_flat)
                V_flat = self._read_blocked_v_dp(value_local, pos_local, start_index, end_index).view(
                    local, rows, C, T_orig, D
                )
                next_start_index, next_end_index, next_kv_len_block = compute_kv_block_range(j + 1)
                K_flat = self._read_blocked_k_dp(key_local, pos_local, next_start_index, next_end_index)
                K_flat = K_flat.view(local, rows, C, next_kv_len_block // C, D)
                out_blk = torch.matmul(exp_blk, V_flat)
                if cfg.skip_kv and (torch.compiler.is_exporting()):
                    out_blk = torch.where(skip_future.unsqueeze(-1), _scalar_like(out_blk, 0.0), out_blk)
                max_buf.append(m_blk)
                sum_buf.append(sum_blk)
                out_buf.append(out_blk)
            epilogue_start_index, epilogue_end_index, epilogue_kv_len_block = compute_kv_block_range(num_kv_blocks - 1)
            epilogue_skip_future = None
            if cfg.skip_kv:
                epilogue_threshold = threshold_all[batch_start:batch_end, :, :, -1]
                epilogue_skip_future = (epilogue_threshold < 0).unsqueeze(-1)
            V_flat = self._read_blocked_v_dp(value_local, pos_local, epilogue_start_index, epilogue_end_index).view(
                local, rows, C, epilogue_kv_len_block // C, D
            )
            epi_m_blk, epi_exp_blk, epi_sum_blk = compute_block(
                epilogue_start_index, epilogue_kv_len_block // C, epilogue_skip_future, K_flat
            )
            epi_out_blk = torch.matmul(epi_exp_blk, V_flat)
            if cfg.skip_kv:
                epi_out_blk = torch.where(
                    epilogue_skip_future.unsqueeze(-1), _scalar_like(epi_out_blk, 0.0), epi_out_blk
                )
            max_buf.append(epi_m_blk)
            sum_buf.append(epi_sum_blk)
            out_buf.append(epi_out_blk)
            max_groups.append(torch.stack(max_buf, dim=0))
            sum_groups.append(torch.stack(sum_buf, dim=0))
            out_groups.append(torch.stack(out_buf, dim=0))
        max_stk = torch.cat(max_groups, dim=1)
        sum_stk = torch.cat(sum_groups, dim=1)
        out_stk = torch.cat(out_groups, dim=1)
        max_1 = max_stk.max(dim=0).values
        weight_1 = torch.exp(max_stk - max_1.unsqueeze(0))
        sum_1 = (weight_1 * sum_stk).sum(dim=0)
        out_1 = (weight_1.unsqueeze(-1) * out_stk).sum(dim=0)
        max_2 = max_1.max(dim=2).values
        weight_2 = torch.exp(max_1 - max_2.unsqueeze(2))
        sum_2 = (weight_2 * sum_1).sum(dim=2)
        out_2 = (weight_2.unsqueeze(-1) * out_1).sum(dim=2)
        max_n = max_2.view(batch_local, DP, N, Hkv, ql_eff)
        sum_n = sum_2.view(batch_local, DP, N, Hkv, ql_eff)
        out_n = out_2.view(batch_local, DP, N, Hkv, ql_eff, D)
        max_3 = max_n.max(dim=2).values
        weight_3 = torch.exp(max_n - max_3.unsqueeze(2))
        sum_3 = (weight_3 * sum_n).sum(dim=2)
        out_3 = (weight_3.unsqueeze(-1) * out_n).sum(dim=2)
        sum_3 = torch.where(sum_3 > 0, sum_3, torch.ones_like(sum_3))
        return (
            (out_3 / sum_3.unsqueeze(-1))
            .view(batch_local, DP, Hkv, n_rep, QL, D)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(B, NQH, QL, D)
        )

    def _read_paged_kv_dp(
        self,
        pool: torch.Tensor,
        block_ids_per_page: torch.Tensor,
    ) -> torch.Tensor:
        DP, num_pages = block_ids_per_page.shape
        rows = pool.shape[1]
        assert block_ids_per_page.ndim == 2 and rows % DP == 0
        Hkv = rows // DP

        block_ids_by_row = (
            block_ids_per_page.transpose(0, 1).unsqueeze(2).expand(num_pages, DP, Hkv).reshape(num_pages, rows)
        )
        return self.gather_paged_kv_dp(pool, block_ids_by_row.to(torch.int32))

    def _read_paged_v_dp(
        self,
        value_cache: torch.Tensor,
        block_ids: torch.Tensor,
        threshold: torch.Tensor,
        block_len: int,
    ) -> torch.Tensor:
        """Gather one paged value block and zero invalid interleave slots."""
        rows = value_cache.shape[1]
        C = self.config.num_cores_per_device
        D = self.config.head_dim
        invalid_v = (
            torch.arange(block_len, device=value_cache.device).view(1, 1, 1, block_len) > threshold.unsqueeze(-1)
        ).unsqueeze(-1)
        DP, num_pages = block_ids.shape
        assert block_ids.ndim == 2 and rows % DP == 0
        Hkv = rows // DP
        block_ids_by_row = block_ids.transpose(0, 1).unsqueeze(2).expand(num_pages, DP, Hkv).reshape(num_pages, rows)
        value_block = self.gather_paged_kv_dp(value_cache, block_ids_by_row.to(torch.int32))
        value_block = value_block.view(1, rows, C, block_len, D)
        return torch.where(
            invalid_v,
            _scalar_like(value_block, 0.0),
            value_block,
        )

    def _read_paged_kv_batch(
        self,
        pool: torch.Tensor,
        block_ids_per_batch_page: torch.Tensor,
    ) -> torch.Tensor:
        # Prefill page IDs are batch-major, but the compiler needs the KV-head
        # row to be explicit in the gather index for core splitting.
        return self.gather_paged_kv_batch_with_heads(pool, block_ids_per_batch_page)

    def _kv_blocked_attn_prefill_page(
        self,
        q: torch.Tensor,
        key_pool: torch.Tensor,
        value_pool: torch.Tensor,
        block_table: torch.Tensor,
        position_ids: torch.Tensor,
        ql_chunk: int = 64,
        q_block_chunk: int | None = None,
        n_rep_chunk: int = 1,
    ) -> torch.Tensor:
        """Paged GQA prefill, gathering one logical KV block per iteration."""
        cfg = self.config
        batch, num_heads, query_len, head_dim = q.shape
        num_kv_heads = cfg.num_kv_heads
        page_size = cfg.page_block_size
        num_cores = cfg.num_cores_per_device
        is_export = torch.compiler.is_exporting()
        if num_cores % num_kv_heads or num_heads % num_cores:
            raise ValueError(
                "Online prefill requires num_cores_per_device to be a multiple of num_kv_heads and to divide num_heads."
            )
        kv_repeat = num_cores // num_kv_heads
        n_rep_per_core = num_heads // num_cores
        if n_rep_per_core % n_rep_chunk:
            raise ValueError("q_head_block_chunk must divide num_heads // num_cores_per_device.")

        num_kv_blocks = max(1, cfg.num_kv_blocks)
        kv_block_size = -(-cfg.ctx_len // num_kv_blocks)
        q_block_chunk = ql_chunk if q_block_chunk is None else q_block_chunk
        if ql_chunk <= 0 or q_block_chunk < ql_chunk or q_block_chunk % ql_chunk:
            raise ValueError("q_block_size must be positive and q_block_chunk must be a multiple of q_block_size.")
        if tuple(key_pool.shape) != (
            cfg.num_physical_blocks,
            num_kv_heads,
            page_size,
            head_dim,
        ):
            raise ValueError("Paged prefill key-pool shape is incompatible with GQA.")
        if tuple(value_pool.shape) != tuple(key_pool.shape):
            raise ValueError("Paged prefill key/value pool shapes must match.")
        if block_table.ndim != 2 or tuple(block_table.shape) != (
            batch,
            cfg.ctx_len // page_size,
        ):
            raise ValueError("Paged prefill block_table must have shape [B, pages].")
        if query_len <= 1 or query_len % page_size:
            raise ValueError("Paged GQA prefill query length must be page aligned.")
        if not is_export and (
            bool((position_ids[:, 0] % page_size != 0).any().item())
            or not torch.equal(position_ids[:, 1:], position_ids[:, :-1] + 1)
        ):
            raise ValueError("Paged GQA prefill positions must be contiguous and page aligned.")

        q_fold = q.reshape(batch, num_cores, n_rep_per_core, query_len, head_dim)
        is_export = torch.compiler.is_exporting()
        rep_ranges = [
            (start, min(start + n_rep_chunk, n_rep_per_core)) for start in range(0, n_rep_per_core, n_rep_chunk)
        ]
        output_chunks: list[torch.Tensor] = []

        for q_chunk_start in range(0, query_len, q_block_chunk):
            q_chunk_end = min(q_chunk_start + q_block_chunk, query_len)
            q_chunk_position = position_ids[:, q_chunk_start:q_chunk_end].max(dim=-1).values
            q_states: dict[int, dict[str, Any]] = {}

            for q_block_start in range(q_chunk_start, q_chunk_end, ql_chunk):
                q_block_end = min(q_block_start + ql_chunk, q_chunk_end)
                q_block_len = q_block_end - q_block_start
                pos_sub = position_ids[:, q_block_start:q_block_end]
                accumulators: list[dict[str, Any]] = []
                for rep_start, rep_end in rep_ranges:
                    rep_count = rep_end - rep_start
                    accumulators.append(
                        {
                            "rep_count": rep_count,
                            "q": q_fold[:, :, rep_start:rep_end, q_block_start:q_block_end, :].reshape(
                                batch,
                                num_cores,
                                rep_count * q_block_len,
                                head_dim,
                            ),
                            "max": torch.full(
                                (batch, num_cores, rep_count * q_block_len),
                                MIN_MASKED_ATTENTION_VALUE,
                                device=q.device,
                                dtype=q.dtype,
                            ),
                            "sum": torch.zeros(
                                batch,
                                num_cores,
                                rep_count * q_block_len,
                                device=q.device,
                                dtype=q.dtype,
                            ),
                            "out": torch.zeros(
                                batch,
                                num_cores,
                                rep_count * q_block_len,
                                head_dim,
                                device=q.device,
                                dtype=q.dtype,
                            ),
                        }
                    )
                q_states[q_block_start] = {
                    "block_len": q_block_len,
                    "positions": pos_sub,
                    "current_position": pos_sub.max(dim=-1).values,
                    "accumulators": accumulators,
                }

            for block_idx in range(num_kv_blocks):
                start_index = block_idx * kv_block_size
                end_index = min(start_index + kv_block_size, cfg.ctx_len)
                if (
                    cfg.skip_kv
                    and not is_export
                    and bool((torch.tensor(start_index, device=q.device) > q_chunk_position).all().item())
                ):
                    break

                page_start = start_index // page_size
                page_end = end_index // page_size
                page_ids = block_table[:, page_start:page_end]

                # INT32_MAX is the paged gather sentinel.  The custom gather
                # consumes it without reading a physical page; keep it in
                # the original IDs so export does not materialize Equal/Where
                # sanitization around every KV block.
                key_block = self._read_paged_kv_batch(key_pool, page_ids)
                value_block = self._read_paged_kv_batch(value_pool, page_ids)
                key_block = key_block.repeat_interleave(kv_repeat, dim=1)
                value_block = value_block.repeat_interleave(kv_repeat, dim=1)
                key_positions = torch.arange(start_index, end_index, device=q.device)
                key_block_t = key_block.transpose(2, 3)
                invalid_value = key_positions[None, None, :] > position_ids.max(dim=1, keepdim=True).values.unsqueeze(1)
                value_block = torch.where(
                    invalid_value.unsqueeze(-1),
                    torch.tensor(0.0, dtype=value_block.dtype, device=value_block.device),
                    value_block,
                )

                for q_block_start in range(q_chunk_start, q_chunk_end, ql_chunk):
                    q_state = q_states[q_block_start]
                    skip_future = (torch.tensor(start_index, device=q.device) > q_state["current_position"]).all()
                    if cfg.skip_kv and not is_export and bool(skip_future.item()):
                        continue
                    q_block_len = q_state["block_len"]
                    causal = key_positions[None, None, None, :] > q_state["positions"][:, None, :, None]

                    for accumulator in q_state["accumulators"]:
                        rep_count = accumulator["rep_count"]
                        attn = torch.matmul(accumulator["q"], key_block_t) * self.scaling
                        attn = attn.view(
                            batch,
                            num_cores,
                            rep_count,
                            q_block_len,
                            -1,
                        )
                        attn = torch.where(
                            causal.unsqueeze(2),
                            _scalar_like(attn, MIN_MASKED_ATTENTION_VALUE),
                            attn,
                        ).view(
                            batch,
                            num_cores,
                            rep_count * q_block_len,
                            -1,
                        )
                        (
                            accumulator["max"],
                            accumulator["sum"],
                            accumulator["out"],
                        ) = update_running_softmax(
                            accumulator["max"],
                            attn,
                            accumulator["sum"],
                            accumulator["out"],
                            value_block,
                            cfg.skip_kv,
                            skip_future,
                        )

            q_block_outputs: list[torch.Tensor] = []
            for q_block_start in range(q_chunk_start, q_chunk_end, ql_chunk):
                q_state = q_states[q_block_start]
                q_block_len = q_state["block_len"]
                rep_outputs: list[torch.Tensor] = []
                for accumulator in q_state["accumulators"]:
                    denominator = torch.where(
                        accumulator["sum"] > 0,
                        accumulator["sum"],
                        torch.ones_like(accumulator["sum"]),
                    )
                    out = accumulator["out"] / denominator.unsqueeze(-1)
                    rep_outputs.append(
                        out.view(
                            batch,
                            num_cores,
                            accumulator["rep_count"],
                            q_block_len,
                            head_dim,
                        )
                    )
                q_block_outputs.append(torch.cat(rep_outputs, dim=2))
            output_chunks.append(torch.cat(q_block_outputs, dim=3))

        return torch.cat(output_chunks, dim=3).reshape(batch, num_heads, query_len, head_dim)

    def _kv_blocked_attn_decode_page(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Paged GP decode with physical rows ordered [DP, CP, Hkv]."""
        cfg = self.config
        B, q_rows, QL, D = q.shape
        DP = cfg.attn_dp
        CP = cfg.attn_cp
        Hkv = cfg.num_kv_heads
        NQH = cfg.num_heads
        C = cfg.num_cores_per_device
        PBS = cfg.page_block_size
        ctx_len = cfg.ctx_len
        num_kv_blocks = cfg.num_kv_blocks
        is_export = torch.compiler.is_exporting()

        assert (
            QL == 1
            and DP >= 1
            and cfg.paged_kv
            and B % DP == 0
            and NQH % Hkv == 0
            and q_rows == NQH
            and ctx_len % PBS == 0
            and ctx_len % num_kv_blocks == 0
        )
        B_local = B // DP
        n_rep = NQH // Hkv
        QL_eff = n_rep * QL
        rows = DP * CP * Hkv
        expected_paged_shape = (cfg.num_physical_blocks, rows, PBS, D)
        if is_export:
            assert (
                key_cache.shape[1] == rows
                and key_cache.shape[3] == D
                and value_cache.shape[1] == rows
                and value_cache.shape[3] == D
                and block_table.ndim == 3
                and tuple(block_table.shape[:2]) == (DP, B_local)
            )
        else:
            assert (
                tuple(key_cache.shape) == expected_paged_shape
                and tuple(value_cache.shape) == tuple(key_cache.shape)
                and block_table.ndim == 3
                and tuple(block_table.shape[:2]) == (DP, B_local)
            )

        logical_kv_block_size = ctx_len // num_kv_blocks
        assert logical_kv_block_size % CP == 0
        kv_block_size = logical_kv_block_size // CP
        num_groups_blk = kv_block_size // PBS
        assert kv_block_size % PBS == 0 and num_groups_blk % C == 0
        groups_per_core = num_groups_blk // C
        T_h = groups_per_core * PBS
        logical_page_groups = (ctx_len // PBS + CP - 1) // CP
        if not is_export:
            assert block_table.shape[2] >= logical_page_groups

        group_order = torch.arange(num_groups_blk, device=q.device).view(groups_per_core, C).transpose(0, 1).reshape(-1)
        q_dp = q.view(DP, B_local, NQH, QL, D).permute(1, 0, 2, 3, 4)
        pos_dp = position_ids.view(DP, B_local, QL).permute(1, 0, 2).to(torch.int32)
        pos_rows_all = pos_dp.view(B_local, DP, 1, 1, QL).expand(B_local, DP, CP, Hkv, QL).reshape(B_local, rows, 1, QL)
        q_block_all = pos_rows_all // logical_kv_block_size
        q_block_offset_all = pos_rows_all - q_block_all * logical_kv_block_size
        q_page_all = q_block_offset_all // PBS
        q_rem_all = q_block_offset_all - q_page_all * PBS
        # Compressed predicate coordinates.  The V data remains in the
        # interleaved [C, T_h] layout, while every core shares the same
        # contiguous local T_h coordinate range [0, T_h).
        mask_core_idx = torch.arange(C, dtype=pos_rows_all.dtype, device=q.device).view(1, 1, C, 1)
        row_group_all = (torch.arange(rows, dtype=pos_rows_all.dtype, device=q.device) // Hkv).view(1, rows, 1, 1)
        row_dp_all = row_group_all // CP
        row_way_all = row_group_all - row_dp_all * CP
        block_idx_all = torch.arange(num_kv_blocks, dtype=pos_rows_all.dtype, device=q.device).view(
            1, 1, 1, num_kv_blocks
        )
        query_after_all = (q_block_all > block_idx_all).squeeze(2)
        query_in_all = (q_block_all == block_idx_all).squeeze(2)
        # CP splits each global page group across ``CP`` row ways.  For a
        # query in page ``q_page_all``, the row way before the live way has
        # the whole local page, the live way has a partial page, and the row
        # ways after it stop at the previous local page.  Convert that
        # row-specific local page into the core-interleaved T_h coordinate.
        q_page_local_all = q_page_all // CP
        q_page_way_all = q_page_all - q_page_local_all * CP
        last_page_all = torch.where(
            row_way_all < q_page_way_all,
            q_page_local_all,
            torch.where(
                row_way_all == q_page_way_all,
                q_page_local_all,
                q_page_local_all - 1,
            ),
        )
        valid_page_all = last_page_all >= 0
        safe_page_all = torch.where(
            valid_page_all,
            last_page_all,
            torch.zeros_like(last_page_all),
        )
        page_group_all = safe_page_all // C
        page_core_all = safe_page_all - page_group_all * C
        group_start_all = page_group_all * PBS
        threshold_before_all = group_start_all + PBS - 1
        threshold_current_all = group_start_all + q_rem_all
        partial_page_all = row_way_all == q_page_way_all
        threshold_after_all = torch.where(
            page_group_all > 0,
            group_start_all - 1,
            torch.full_like(group_start_all, -1),
        )
        threshold_current_block_all = torch.where(
            valid_page_all & (mask_core_idx < page_core_all),
            threshold_before_all,
            torch.where(
                valid_page_all & (mask_core_idx == page_core_all),
                torch.where(
                    partial_page_all,
                    threshold_current_all,
                    threshold_before_all,
                ),
                torch.where(
                    valid_page_all,
                    threshold_after_all,
                    torch.full_like(threshold_after_all, -1),
                ),
            ),
        ).squeeze(-1)
        threshold_all = torch.where(
            query_after_all.unsqueeze(2),
            torch.full_like(threshold_current_block_all.unsqueeze(-1), T_h - 1),
            torch.where(
                query_in_all.unsqueeze(2),
                threshold_current_block_all.unsqueeze(-1),
                torch.full_like(threshold_current_block_all.unsqueeze(-1), -1),
            ),
        )

        batch_max_stk: list[torch.Tensor] = []
        batch_sum_stk: list[torch.Tensor] = []
        batch_out_stk: list[torch.Tensor] = []
        for b_local in range(B_local):
            q_local = q_dp[b_local]
            q_grouped = q_local.reshape(DP, Hkv, QL_eff, D)
            q_rows_gp = q_grouped.unsqueeze(1).expand(DP, CP, Hkv, QL_eff, D).reshape(1, rows, QL_eff, D)
            Q_5d = q_rows_gp.unsqueeze(2).expand(1, rows, C, QL_eff, D)
            bt_local = block_table[:, b_local]

            max_buf: list[torch.Tensor] = []
            sum_buf: list[torch.Tensor] = []
            out_buf: list[torch.Tensor] = []

            def compute_block(
                start_index: int,
                T_orig: int,
                skip_future: torch.Tensor,
                K_flat: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                block_idx = start_index // T_h
                threshold = threshold_all[b_local : b_local + 1, :, :, block_idx]
                causal_mask = torch.arange(T_orig, device=q.device).view(1, 1, 1, T_orig) > threshold.unsqueeze(-1)
                causal_mask = causal_mask.unsqueeze(3).repeat(1, 1, 1, n_rep, 1)
                attn = torch.matmul(Q_5d, K_flat.transpose(-1, -2)) * self.scaling
                attn = attn.masked_fill(causal_mask, float(-3.0e4))
                m_blk = attn.max(dim=-1).values
                exp_blk = torch.exp(attn - m_blk.unsqueeze(-1))
                sum_blk = exp_blk.sum(dim=-1)
                if cfg.skip_kv:
                    m_blk = torch.where(skip_future, _scalar_like(m_blk, MASKED_ATTENTION_LOGIT), m_blk)
                    exp_blk = torch.where(skip_future.unsqueeze(-1), _scalar_like(exp_blk, 0.0), exp_blk)
                    sum_blk = torch.where(skip_future, _scalar_like(sum_blk, 0.0), sum_blk)
                return m_blk, exp_blk, sum_blk

            prologue_block_ids = bt_local[:, group_order]
            K_flat = self._read_paged_kv_dp(key_cache, prologue_block_ids).view(1, rows, C, T_h, D)

            for j in range(num_kv_blocks - 1):
                start_index = j * T_h
                T_orig = T_h
                threshold = threshold_all[b_local : b_local + 1, :, :, j]
                skip_future = None
                if cfg.skip_kv:
                    skip_future = (threshold < 0).unsqueeze(-1)
                    if not torch.compiler.is_exporting():
                        if skip_future.all().item():
                            break

                m_blk, exp_blk, sum_blk = compute_block(start_index, T_orig, skip_future, K_flat)
                block_ids = bt_local[:, j * num_groups_blk + group_order]
                next_block_ids = bt_local[:, (j + 1) * num_groups_blk + group_order]

                V_5d = self._read_paged_v_dp(value_cache, block_ids, threshold, T_orig)

                K_flat = self._read_paged_kv_dp(key_cache, next_block_ids)
                out_blk = torch.matmul(exp_blk, V_5d)
                if cfg.skip_kv and is_export:
                    out_blk = torch.where(skip_future.unsqueeze(-1), _scalar_like(out_blk, 0.0), out_blk)
                max_buf.append(m_blk)
                sum_buf.append(sum_blk)
                out_buf.append(out_blk)
                K_flat = K_flat.view(1, rows, C, T_h, D)

            last_j = num_kv_blocks - 1
            last_start_index = last_j * T_h
            last_T_orig = T_h
            last_threshold = threshold_all[b_local : b_local + 1, :, :, last_j]
            last_skip_future = None
            if cfg.skip_kv:
                last_skip_future = (last_threshold < 0).unsqueeze(-1)
            last_block_ids = bt_local[:, last_j * num_groups_blk + group_order]
            V_5d = self._read_paged_v_dp(value_cache, last_block_ids, last_threshold, last_T_orig)
            epi_m_blk, epi_exp_blk, epi_sum_blk = compute_block(last_start_index, last_T_orig, last_skip_future, K_flat)
            epi_out_blk = torch.matmul(epi_exp_blk, V_5d)
            if cfg.skip_kv and is_export:
                epi_out_blk = torch.where(last_skip_future.unsqueeze(-1), _scalar_like(epi_out_blk, 0.0), epi_out_blk)
            max_buf.append(epi_m_blk)
            sum_buf.append(epi_sum_blk)
            out_buf.append(epi_out_blk)

            max_stk = torch.stack(max_buf)
            sum_stk = torch.stack(sum_buf)
            out_stk = torch.stack(out_buf)
            batch_max_stk.append(max_stk)
            batch_sum_stk.append(sum_stk)
            batch_out_stk.append(out_stk)

        max_stk = torch.cat(batch_max_stk, dim=1)
        sum_stk = torch.cat(batch_sum_stk, dim=1)
        out_stk = torch.cat(batch_out_stk, dim=1)
        m1 = max_stk.max(dim=0).values
        w1 = torch.exp(max_stk - m1.unsqueeze(0))
        s1 = (w1 * sum_stk).sum(dim=0)
        o1 = (w1.unsqueeze(-1) * out_stk).sum(dim=0)
        m2 = m1.max(dim=2).values
        w2 = torch.exp(m1 - m2.unsqueeze(2))
        s2 = (w2 * s1).sum(dim=2)
        o2 = (w2.unsqueeze(-1) * o1).sum(dim=2)

        m2_cp = m2.view(B_local, DP, CP, Hkv, QL_eff)
        s2_cp = s2.view(B_local, DP, CP, Hkv, QL_eff)
        o2_cp = o2.view(B_local, DP, CP, Hkv, QL_eff, D)
        m3 = m2_cp.max(dim=2).values
        w3 = torch.exp(m2_cp - m3.unsqueeze(2))
        s3 = (w3 * s2_cp).sum(dim=2)
        o3 = (w3.unsqueeze(-1) * o2_cp).sum(dim=2)
        s3 = torch.where(s3 > 0, s3, torch.ones_like(s3))
        output = (o3 / s3.unsqueeze(-1)).view(B_local, DP, NQH, QL, D)
        return output.permute(1, 0, 2, 3, 4).reshape(B, NQH, QL, D)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        # This benchmark exports a fixed [B, QL, hidden] graph.  Materialize
        # the dimensions as Python ints so QAIC receives a literal reshape
        # shape; symbolic ``-1`` is misinterpreted for long GQA prefills.
        batch = int(hidden_states.shape[0])
        seq_len = int(hidden_states.shape[1])
        is_prefill = seq_len > 1
        # Use the literal fixed prefill extent in the exported shape.  QAIC
        # does not reliably resolve a symbolic ``-1`` dimension here.
        # Keep the GQA projection graph in the same order as the standalone
        # benchmark: [B, QL, H, D] -> transpose -> RMSNorm.  RMSNorm is over
        # D, so the numerical result is unchanged, while the ONNX shape and
        # transpose placement are materially different to the compiler.
        q = self.q_proj(hidden_states).reshape(batch, seq_len, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_proj(hidden_states).reshape(batch, seq_len, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_norm(k)
        v = self.v_proj(hidden_states).reshape(batch, seq_len, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        if cfg.paged_kv and not is_prefill:
            k = (
                k.unsqueeze(2)
                .expand(
                    batch,
                    cfg.num_kv_heads,
                    cfg.attn_cp,
                    position_ids.shape[1],
                    cfg.head_dim,
                )
                .reshape(
                    batch,
                    cfg.num_kv_heads * cfg.attn_cp,
                    position_ids.shape[1],
                    cfg.head_dim,
                )
            )
            v = (
                v.unsqueeze(2)
                .expand(
                    batch,
                    cfg.num_kv_heads,
                    cfg.attn_cp,
                    position_ids.shape[1],
                    cfg.head_dim,
                )
                .reshape(
                    batch,
                    cfg.num_kv_heads * cfg.attn_cp,
                    position_ids.shape[1],
                    cfg.head_dim,
                )
            )
        if is_prefill:
            if (cos is None) != (sin is None):
                raise ValueError("GQA prefill requires both cos and sin inputs.")
            if cos is None:
                cos, sin = self._rope(position_ids, hidden_states.dtype)
        else:
            cos, sin = self._rope(position_ids, hidden_states.dtype)
        q = MiniMaxM3SparseDecodeAttention._apply_rope(q, cos, sin)
        k = MiniMaxM3SparseDecodeAttention._apply_rope(k, cos, sin)
        # Prefill and decode use separate cache layouts. GQA prefill follows
        # the page-level metadata/scatter contract; decode keeps its existing
        # DP/CP-folded path.
        if is_prefill and not cfg.paged_kv:
            # Keep the ordinary prefill cache update on the same explicit
            # scatter path as the other non-paged attention implementations.
            key_cache = self.ctx_scatter(key_cache, position_ids.to(torch.int32), k)
            value_cache = self.ctx_scatter(value_cache, position_ids.to(torch.int32), v)
            out = self._kv_blocked_attn_prefill(
                q,
                key_cache,
                value_cache,
                position_ids,
                ql_chunk=cfg.q_block_size,
                q_block_chunk=cfg.q_block_chunk,
                n_rep_chunk=cfg.q_head_block_chunk,
            )
        elif is_prefill and cfg.paged_kv:
            if cfg.attn_dp != 1 or cfg.attn_cp != 1:
                raise NotImplementedError("Paged GQA prefill requires attn_dp=1 and attn_cp=1.")
            if block_table is None:
                raise ValueError("Paged GQA prefill requires gqa_block_table.")
            num_query_pages = seq_len // cfg.page_block_size
            expected_pool_shape = (
                cfg.num_physical_blocks,
                cfg.num_kv_heads,
                cfg.page_block_size,
                cfg.head_dim,
            )
            if tuple(key_cache.shape) != expected_pool_shape:
                raise ValueError(
                    f"GQA paged prefill key_cache shape {tuple(key_cache.shape)} != {expected_pool_shape}."
                )
            if tuple(value_cache.shape) != expected_pool_shape:
                raise ValueError(
                    f"GQA paged prefill value_cache shape {tuple(value_cache.shape)} != {expected_pool_shape}."
                )
            if tuple(block_table.shape) != (
                batch,
                cfg.ctx_len // cfg.page_block_size,
            ):
                raise ValueError("gqa_block_table must have shape [B, pages] for prefill.")
            # The physical pages receiving this prefill are already encoded in
            # block_table.  Derive their IDs from the absolute query page
            # positions so a separate incoming-page graph input is unnecessary.
            query_page_positions = position_ids[:, :: cfg.page_block_size]
            query_page_ids = torch.gather(
                block_table,
                1,
                (query_page_positions // cfg.page_block_size).to(torch.int64),
            )
            for batch_idx in range(batch):
                block_ids = (
                    query_page_ids[batch_idx]
                    .to(torch.int32)
                    .view(num_query_pages, 1)
                    .expand(num_query_pages, cfg.num_kv_heads)
                )
                k_updates = (
                    k[batch_idx]
                    .reshape(
                        cfg.num_kv_heads,
                        num_query_pages,
                        cfg.page_block_size,
                        cfg.head_dim,
                    )
                    .permute(1, 0, 2, 3)
                )
                v_updates = (
                    v[batch_idx]
                    .reshape(
                        cfg.num_kv_heads,
                        num_query_pages,
                        cfg.page_block_size,
                        cfg.head_dim,
                    )
                    .permute(1, 0, 2, 3)
                )
                key_cache = self.paged_scatter_page(key_cache, block_ids, k_updates)
                value_cache = self.paged_scatter_page(value_cache, block_ids, v_updates)
            out = self._kv_blocked_attn_prefill_page(
                q,
                key_cache,
                value_cache,
                block_table,
                position_ids,
                ql_chunk=cfg.q_block_size,
                q_block_chunk=cfg.q_block_chunk,
                n_rep_chunk=cfg.q_head_block_chunk,
            )
        else:
            dp = cfg.attn_dp
            num_ways = cfg.attn_cp
            batch_local = batch // dp
            rows = dp * num_ways * cfg.num_kv_heads
            if cfg.paged_kv:
                if block_table is None:
                    raise ValueError("paged GQA requires gqa_block_table.")
                expected_paged_shape = (cfg.num_physical_blocks, rows, cfg.page_block_size, cfg.head_dim)
                if not torch.compiler.is_exporting() and tuple(key_cache.shape) != expected_paged_shape:
                    raise ValueError(f"GQA paged key_cache shape {tuple(key_cache.shape)} != {expected_paged_shape}.")
                if not torch.compiler.is_exporting() and tuple(value_cache.shape) != expected_paged_shape:
                    raise ValueError(
                        f"GQA paged value_cache shape {tuple(value_cache.shape)} != {expected_paged_shape}."
                    )
                if block_table.ndim != 3 or tuple(block_table.shape[:2]) != (
                    dp,
                    batch_local,
                ):
                    raise ValueError(
                        f"gqa_block_table must have shape [attn_dp, B_local, pages], got {tuple(block_table.shape)}."
                    )
                k_updates = (
                    k.view(
                        dp,
                        batch_local,
                        cfg.num_kv_heads,
                        num_ways,
                        seq_len,
                        cfg.head_dim,
                    )
                    .permute(1, 0, 3, 2, 4, 5)
                    .reshape(batch_local, rows, seq_len, cfg.head_dim)
                )
                v_updates = (
                    v.view(
                        dp,
                        batch_local,
                        cfg.num_kv_heads,
                        num_ways,
                        seq_len,
                        cfg.head_dim,
                    )
                    .permute(1, 0, 3, 2, 4, 5)
                    .reshape(batch_local, rows, seq_len, cfg.head_dim)
                )
                pos_dp = position_ids.view(dp, batch_local, seq_len).permute(1, 0, 2)
                logical_page = pos_dp // cfg.page_block_size
                page_offset = pos_dp - logical_page * cfg.page_block_size
                addr = (
                    page_offset.to(torch.int32)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .expand(batch_local, dp, num_ways, cfg.num_kv_heads, seq_len)
                    .reshape(batch_local, rows, seq_len)
                )
                logical_page_group_i32 = logical_page // num_ways
                logical_page_group = logical_page_group_i32.to(torch.int64)
                block_id_dp = (
                    torch.gather(block_table.permute(1, 0, 2), 2, logical_page_group)
                    .to(torch.int32)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .expand(batch_local, dp, num_ways, cfg.num_kv_heads, seq_len)
                )
                live_way = (logical_page - logical_page_group_i32 * num_ways).to(torch.int64)
                way_idx = torch.arange(num_ways, device=q.device).view(1, 1, num_ways, 1, 1)
                row_live = way_idx == live_way.unsqueeze(2).unsqueeze(3)
                row_live = row_live.expand(batch_local, dp, num_ways, cfg.num_kv_heads, seq_len)
                block_id = torch.where(
                    row_live,
                    block_id_dp,
                    torch.iinfo(torch.int32).max,
                ).reshape(batch_local, rows, seq_len)
                key_cache = self.paged_scatter(key_cache, block_id, addr, k_updates)
                value_cache = self.paged_scatter(value_cache, block_id, addr, v_updates)
                out = self._kv_blocked_attn_decode_page(
                    q,
                    key_cache,
                    value_cache,
                    block_table,
                    position_ids,
                )
            else:
                k_dp = k.view(dp, batch_local, cfg.num_kv_heads, seq_len, cfg.head_dim).permute(1, 0, 2, 3, 4)
                v_dp = v.view(dp, batch_local, cfg.num_kv_heads, seq_len, cfg.head_dim).permute(1, 0, 2, 3, 4)
                k_updates = (
                    k_dp.unsqueeze(2)
                    .expand(
                        batch_local,
                        dp,
                        num_ways,
                        cfg.num_kv_heads,
                        seq_len,
                        cfg.head_dim,
                    )
                    .reshape(batch_local, rows, seq_len, cfg.head_dim)
                )
                v_updates = (
                    v_dp.unsqueeze(2)
                    .expand(
                        batch_local,
                        dp,
                        num_ways,
                        cfg.num_kv_heads,
                        seq_len,
                        cfg.head_dim,
                    )
                    .reshape(batch_local, rows, seq_len, cfg.head_dim)
                )
                pos_dp = position_ids.view(dp, batch_local, seq_len).permute(1, 0, 2)
                live_way = (
                    (pos_dp % num_ways)[:, :, None, None, :]
                    .expand(
                        batch_local,
                        dp,
                        num_ways,
                        cfg.num_kv_heads,
                        seq_len,
                    )
                    .reshape(batch_local, rows, seq_len)
                )
                row_idx = torch.arange(rows, device=q.device).view(1, rows, 1)
                row_way = row_idx.remainder(num_ways * cfg.num_kv_heads) // cfg.num_kv_heads
                row_live = row_way == live_way
                batch_id = (
                    torch.arange(batch_local, device=q.device)
                    .view(batch_local, 1, 1)
                    .expand(batch_local, rows, seq_len)
                )
                block_id = torch.where(row_live, batch_id, torch.iinfo(torch.int32).max).to(torch.int32)
                addr = (
                    (pos_dp // num_ways)
                    .to(torch.int32)[:, :, None, None, :]
                    .expand(batch_local, dp, num_ways, cfg.num_kv_heads, seq_len)
                    .reshape(batch_local, rows, seq_len)
                )
                key_cache = self.paged_scatter(key_cache, block_id, addr, k_updates)
                value_cache = self.paged_scatter(value_cache, block_id, addr, v_updates)
                if is_prefill:
                    out = self._kv_blocked_attn_prefill(
                        q,
                        key_cache,
                        value_cache,
                        position_ids,
                        ql_chunk=cfg.q_block_size,
                        q_block_chunk=cfg.q_block_chunk,
                        n_rep_chunk=cfg.q_head_block_chunk,
                    )
                else:
                    q_gp = (
                        q.unsqueeze(2)
                        .expand(
                            batch,
                            cfg.num_heads,
                            num_ways,
                            seq_len,
                            cfg.head_dim,
                        )
                        .reshape(
                            batch,
                            cfg.num_heads * num_ways,
                            seq_len,
                            cfg.head_dim,
                        )
                    )
                    out = self._kv_blocked_attn_decode(q_gp, key_cache, value_cache, position_ids)
        out = (
            out.to(hidden_states.dtype)
            .reshape(batch, cfg.num_heads, seq_len, cfg.head_dim)
            .transpose(1, 2)
            .reshape(batch, seq_len, cfg.num_heads * cfg.head_dim)
        )
        return self.o_proj(out), key_cache, value_cache


class MiniMaxM3StackedDecodeModel(nn.Module):
    def __init__(
        self,
        config: Config,
        num_gqa_layers: int,
        num_msa_layers: int,
    ):
        super().__init__()
        self.config = config
        self.num_gqa_layers = num_gqa_layers
        self.num_msa_layers = num_msa_layers
        self.gqa_layers = nn.ModuleList(
            [MiniMaxM3GQADecodeAttention(config) for _ in range(num_gqa_layers)]
        )
        self.msa_layers = nn.ModuleList(
            [MiniMaxM3SparseDecodeAttention(config) for _ in range(num_msa_layers)]
        )
        self.last_block_indices: list[torch.Tensor] | None = None

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor, *caches: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        updated_caches: list[torch.Tensor] = []
        cache_idx = 0
        gqa_block_table = None
        gqa_cos = None
        gqa_sin = None
        msa_indexer_block_table = None
        msa_attn_block_table = None
        metadata_idx = 2 * self.num_gqa_layers + 3 * self.num_msa_layers
        if self.config.paged_kv:
            if self.num_gqa_layers:
                gqa_block_table = caches[metadata_idx]
                metadata_idx += 1
            if self.num_msa_layers:
                msa_indexer_block_table = caches[metadata_idx]
                metadata_idx += 1
                if not self.config.indexer_only:
                    msa_attn_block_table = caches[metadata_idx]
                    metadata_idx += 1
        if self.num_gqa_layers and int(position_ids.shape[1]) > 1:
            gqa_cos, gqa_sin = (caches[metadata_idx], caches[metadata_idx + 1])
            metadata_idx += 2
        msa_cos = msa_sin = None
        if self.num_msa_layers:
            msa_cos, msa_sin = (caches[metadata_idx], caches[metadata_idx + 1])
        for layer_idx, gqa in enumerate(self.gqa_layers):
            kc, vc = (caches[cache_idx], caches[cache_idx + 1])
            cache_idx += 2
            layer_block_table = gqa_block_table
            hidden_states, kc, vc = gqa(hidden_states, position_ids, kc, vc, layer_block_table, gqa_cos, gqa_sin)
            updated_caches += [kc, vc]
        block_indices: list[torch.Tensor] = []
        indexer_outputs: list[torch.Tensor] = []
        for msa in self.msa_layers:
            kc, vc, ikc = (caches[cache_idx], caches[cache_idx + 1], caches[cache_idx + 2])
            cache_idx += 3
            msa_outputs = msa(
                hidden_states,
                position_ids,
                kc,
                vc,
                ikc,
                msa_indexer_block_table,
                msa_attn_block_table,
                msa_cos,
                msa_sin,
            )
            hidden_states, kc, vc, ikc = msa_outputs[:4]
            updated_caches += [kc, vc, ikc]
            if msa.last_block_indices is not None:
                block_indices.append(msa.last_block_indices)
        if not torch.compiler.is_exporting():
            self.last_block_indices = block_indices or None
        return (hidden_states, *updated_caches, *indexer_outputs)
