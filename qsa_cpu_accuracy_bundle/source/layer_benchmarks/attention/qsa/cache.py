"""QSA retained-cache reads and writes.

Cache ABI (see README):

  attn_cp=1   key/value [B/attn_dp, attn_dp*kv_heads, context, head_dim]
  attn_cp>1   key/value [B/attn_dp, attn_dp*attn_cp*kv_heads, context/attn_cp, head_dim]
  paged       key/value [physical_blocks, attn_dp*attn_cp*kv_heads, 4, head_dim]
  index                 [B/indexer_dp, indexer_dp*indexer_cp, context/(4*indexer_cp), index_dim]
  partial               [B, 1, index_dim] fp32 (retained by decode only)

Each function runs eager PyTorch, or emits the QAIC custom op while exporting.
"""

from __future__ import annotations

import torch

from layer_benchmarks.common import custom_ops

from . import ops

INVALID = torch.iinfo(torch.int32).max


def trunc_div_pow2_nonnegative(value: torch.Tensor, divisor: int) -> torch.Tensor:
    """Divide nonnegative integers by a power of two without integer Div/Sign."""
    if divisor <= 0 or divisor & (divisor - 1):
        raise ValueError("divisor must be a positive power of two")
    reciprocal = 2.0 ** (-(divisor.bit_length() - 1))
    return (value.to(torch.float32) * reciprocal).to(value.dtype)


def scatter(data: torch.Tensor, positions: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """data[b, row, positions[b, row, s]] = updates[b, row, s]."""
    batch, rows, _, _ = updates.shape
    block_id = torch.arange(data.shape[0], device=data.device).view(-1, 1, 1).expand_as(positions)

    def eager(data, block_id, positions, updates):
        out = data.clone()
        out[torch.arange(batch).view(batch, 1, 1), torch.arange(rows).view(1, rows, 1), positions.long()] = updates
        return out

    return custom_ops.call(
        ops.CtxPagedScatter, (data, block_id.to(torch.int32), positions.to(torch.int32), updates), eager, shape=data.shape
    )


def scatter_cp(data: torch.Tensor, block_id: torch.Tensor, addr: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Owner-only write: rows whose block_id is INT32_MAX are left untouched."""

    def eager(data, block_id, addr, updates):
        # Direct masked assignment; a read-modify-write select would lower to
        # a full-cache GatherND under Dynamo.
        valid = block_id != INVALID
        rows = torch.arange(updates.shape[1], device=data.device).view(1, -1, 1)
        out = data.clone()
        out[block_id[valid].long(), rows.expand_as(block_id)[valid], addr[valid].long()] = updates[valid]
        return out

    return custom_ops.call(
        ops.CtxPagedScatter, (data, block_id.to(torch.int32), addr.to(torch.int32), updates), eager, shape=data.shape
    )


def gather_rows(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Gather context entries independently from each [batch, row]."""
    if ctx_indices.ndim < 3:
        raise ValueError("ctx_indices must have shape [B, rows, ...]")

    def eager(data, ctx_indices):
        batch_shape = (data.shape[0],) + (1,) * (ctx_indices.ndim - 1)
        row_shape = (1, data.shape[1]) + (1,) * (ctx_indices.ndim - 2)
        batch_indices = torch.arange(data.shape[0], device=data.device).view(*batch_shape)
        row_indices = torch.arange(data.shape[1], device=data.device).view(*row_shape)
        safe_indices = torch.where(ctx_indices == INVALID, torch.zeros_like(ctx_indices), ctx_indices)
        return data[batch_indices, row_indices, safe_indices.long()]

    return custom_ops.call(
        ops.V4CtxGatherRows,
        (data, ctx_indices.to(torch.int32)),
        eager,
        shape=(*ctx_indices.shape, data.shape[-1]),
    )


def scatter_sharded_cache(
    data: torch.Tensor,
    global_positions: torch.Tensor,
    updates: torch.Tensor,
    *,
    dp: int,
    cp: int,
    rows_per_dp_way: int = 1,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write only the folded DP/CP row that owns each logical position."""
    if dp < 1 or cp < 1 or rows_per_dp_way < 1:
        raise ValueError("dp, cp, and rows_per_dp_way must be positive")
    if updates.ndim == 2 and rows_per_dp_way == 1:
        updates = updates.unsqueeze(1)
    if updates.ndim != 3 or updates.shape[1] != rows_per_dp_way:
        raise ValueError("updates must have shape [B, rows_per_dp_way, D]")
    batch, _, dim = updates.shape
    if batch % dp:
        raise ValueError("batch size must be divisible by dp")
    batch_local = batch // dp
    rows = dp * cp * rows_per_dp_way
    if data.ndim != 4 or data.shape[:2] != (batch_local, rows):
        raise ValueError("cache must have shape [B/dp, dp*cp*rows_per_dp_way, T/cp, D]")
    if data.shape[-1] != dim:
        raise ValueError("cache and update feature dimensions must match")
    if tuple(global_positions.shape) != (batch,):
        raise ValueError("global_positions must have shape [B]")
    if valid is not None and tuple(valid.shape) != (batch,):
        raise ValueError("valid must have shape [B]")

    updates_dp = updates.reshape(batch_local, dp, rows_per_dp_way, dim)
    positions_dp = global_positions.reshape(batch_local, dp)
    updates_rows = (
        updates_dp.unsqueeze(2)
        .expand(batch_local, dp, cp, rows_per_dp_way, dim)
        .reshape(batch_local, rows, 1, dim)
    )
    valid_dp = (
        torch.ones_like(positions_dp, dtype=torch.bool)
        if valid is None
        else valid.reshape(batch_local, dp)
    )
    if cp == 1:
        live_way = torch.zeros_like(positions_dp)
        local_slots = positions_dp
    else:
        local_slots = trunc_div_pow2_nonnegative(positions_dp, cp)
        live_way = positions_dp - local_slots * cp

    cp_rows = torch.arange(cp, device=data.device).view(1, 1, cp, 1)
    row_live = (
        (live_way[:, :, None, None] == cp_rows) & valid_dp[:, :, None, None]
    ).expand(batch_local, dp, cp, rows_per_dp_way)
    row_owner_id = torch.arange(batch_local, device=data.device).view(
        batch_local, 1, 1, 1
    ).expand_as(row_live)
    row_owner_id = torch.where(
        row_live, row_owner_id, torch.full_like(row_owner_id, INVALID)
    ).reshape(batch_local, rows, 1).to(torch.int32)
    addr = (
        local_slots[:, :, None, None]
        .expand(batch_local, dp, cp, rows_per_dp_way)
        .reshape(batch_local, rows, 1)
        .to(torch.int32)
    )
    return scatter_cp(data, row_owner_id, addr, updates_rows)


def scatter_sharded_cache_sequence(
    data: torch.Tensor,
    global_positions: torch.Tensor,
    updates: torch.Tensor,
    *,
    dp: int,
    cp: int,
    rows_per_dp_way: int = 1,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write a complete sequence into folded DP/CP cache rows."""
    if dp < 1 or cp < 1 or rows_per_dp_way < 1:
        raise ValueError("dp, cp, and rows_per_dp_way must be positive")
    if global_positions.ndim != 2:
        raise ValueError("global_positions must have shape [B, Q]")
    if updates.ndim == 3 and rows_per_dp_way == 1:
        updates = updates.unsqueeze(2)
    if updates.ndim != 4 or updates.shape[2] != rows_per_dp_way:
        raise ValueError("updates must have shape [B, Q, rows_per_dp_way, D]")
    batch, seq_len = global_positions.shape
    if tuple(updates.shape[:2]) != (batch, seq_len):
        raise ValueError("positions and updates must share [B, Q]")
    if batch % dp:
        raise ValueError("batch size must be divisible by dp")
    batch_local = batch // dp
    rows = dp * cp * rows_per_dp_way
    if data.ndim != 4 or data.shape[:2] != (batch_local, rows):
        raise ValueError("cache must have shape [B/dp, dp*cp*rows_per_dp_way, T/cp, D]")
    if data.shape[-1] != updates.shape[-1]:
        raise ValueError("cache and update feature dimensions must match")
    if valid is not None and tuple(valid.shape) != (batch, seq_len):
        raise ValueError("valid must have shape [B, Q]")

    positions_dp = global_positions.reshape(batch_local, dp, seq_len)
    updates_dp = updates.reshape(
        batch_local, dp, seq_len, rows_per_dp_way, updates.shape[-1]
    )
    updates_rows = (
        updates_dp.unsqueeze(2)
        .expand(
            batch_local,
            dp,
            cp,
            seq_len,
            rows_per_dp_way,
            updates.shape[-1],
        )
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(batch_local, rows, seq_len, updates.shape[-1])
    )
    valid_dp = (
        torch.ones_like(positions_dp, dtype=torch.bool)
        if valid is None
        else valid.reshape(batch_local, dp, seq_len)
    )
    if cp == 1:
        live_way = torch.zeros_like(positions_dp)
        local_slots = positions_dp
    else:
        local_slots = trunc_div_pow2_nonnegative(positions_dp, cp)
        live_way = positions_dp - local_slots * cp

    cp_rows = torch.arange(cp, device=data.device).view(1, 1, cp, 1)
    row_live = (
        (live_way.unsqueeze(2) == cp_rows) & valid_dp.unsqueeze(2)
    ).unsqueeze(3).expand(batch_local, dp, cp, rows_per_dp_way, seq_len)
    row_owner_id = torch.arange(batch_local, device=data.device).view(
        batch_local, 1, 1, 1, 1
    ).expand_as(row_live)
    row_owner_id = torch.where(
        row_live, row_owner_id, torch.full_like(row_owner_id, INVALID)
    ).reshape(batch_local, rows, seq_len).to(torch.int32)
    addr = (
        local_slots[:, :, None, None, :]
        .expand(batch_local, dp, cp, rows_per_dp_way, seq_len)
        .reshape(batch_local, rows, seq_len)
        .to(torch.int32)
    )
    return scatter_cp(data, row_owner_id, addr, updates_rows)


def gather_sharded_cache_local(
    data: torch.Tensor,
    gather_positions: torch.Tensor,
    valid: torch.Tensor,
    *,
    dp: int,
    cp: int,
    rows_per_dp_way: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather logical positions only from their owner-local CP rows."""
    if dp < 1 or cp < 1 or rows_per_dp_way < 1:
        raise ValueError("dp, cp, and rows_per_dp_way must be positive")
    if gather_positions.ndim != 2 or valid.shape != gather_positions.shape:
        raise ValueError("gather_positions and valid must have shape [B, K]")
    batch, gather_len = gather_positions.shape
    if batch % dp:
        raise ValueError("batch size must be divisible by dp")
    batch_local = batch // dp
    rows = dp * cp * rows_per_dp_way
    if data.ndim != 4 or data.shape[:2] != (batch_local, rows):
        raise ValueError("cache must have shape [B/dp, dp*cp*rows_per_dp_way, T/cp, D]")

    positions_dp = gather_positions.reshape(batch_local, dp, gather_len)
    valid_dp = valid.reshape(batch_local, dp, gather_len)
    local_slots = int(data.shape[2])
    safe_positions = torch.where(valid_dp, positions_dp, torch.zeros_like(positions_dp))
    if cp == 1:
        physical_address = safe_positions.to(torch.int32)
        owner_cp = torch.zeros_like(physical_address, dtype=torch.int32)
    else:
        global_context = local_slots * cp
        physical_address_table = torch.tensor(
            [
                (position % cp) * local_slots + (position // cp)
                for position in range(global_context)
            ],
            device=data.device,
            dtype=torch.int32,
        ).view(1, 1, global_context)
        physical_address = torch.gather(
            physical_address_table,
            -1,
            safe_positions.to(torch.int64).reshape(1, 1, -1),
        ).reshape(batch_local, dp, gather_len)
        owner_cp = trunc_div_pow2_nonnegative(physical_address, local_slots).to(torch.int32)
    local_address = physical_address - owner_cp * local_slots

    cp_rows = torch.arange(cp, device=data.device, dtype=torch.int32).view(1, 1, cp, 1)
    row_valid = valid_dp.unsqueeze(2) & (owner_cp.unsqueeze(2) == cp_rows)
    row_valid_with_heads = row_valid.unsqueeze(3).expand(
        batch_local, dp, cp, rows_per_dp_way, gather_len
    )
    local_indices = local_address[:, :, None, None, :].expand(
        batch_local, dp, cp, rows_per_dp_way, gather_len
    )
    gather_indices = torch.where(
        row_valid_with_heads, local_indices, torch.full_like(local_indices, INVALID)
    ).reshape(batch_local, rows, gather_len).to(torch.int32)

    gathered = gather_rows(data, gather_indices).reshape(
        batch_local, dp, cp, rows_per_dp_way, gather_len, data.shape[-1]
    )
    gathered = torch.where(
        row_valid_with_heads.unsqueeze(-1), gathered, torch.zeros_like(gathered)
    )
    return gathered, row_valid


def gather_blocks(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """[B, rows, blocks, 4, D] gathered at indices [B, rows, K, 1] -> [B, rows, K, 4, D]."""
    block_ids = indices.squeeze(-1).to(torch.int32)
    return custom_ops.call(
        ops.CtxGatherBlockRangeKVDP,
        (data, block_ids),
        lambda data, _ids: ops.block_gather(data, indices),
        shape=(*block_ids.shape, data.shape[3], data.shape[4]),
    )


def gather_paged_kv(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
    """Physical pages [blocks, rows, 4, D] at block_ids [pages, rows] -> [1, rows, pages*4, D].

    Always the eager decomposition (as in the original bench).
    """
    block_ids = torch.where(block_ids == INVALID, torch.zeros_like(block_ids), block_ids)
    pages, rows = block_ids.shape
    gathered = data[block_ids.long(), torch.arange(rows, device=data.device).view(1, rows)]
    return gathered.permute(1, 0, 2, 3).reshape(1, rows, pages * data.shape[2], data.shape[3])


class Qwen38QSAAttentionCache:
    """Mixin exposing the QSA cache operations to the attention math."""

    trunc_div_pow2_nonnegative = staticmethod(trunc_div_pow2_nonnegative)
    scatter = staticmethod(scatter)
    scatter_cp = staticmethod(scatter_cp)
    gather_rows = staticmethod(gather_rows)
    scatter_sharded_cache = staticmethod(scatter_sharded_cache)
    scatter_sharded_cache_sequence = staticmethod(scatter_sharded_cache_sequence)
    gather_sharded_cache_local = staticmethod(gather_sharded_cache_local)
    gather_blocks = staticmethod(gather_blocks)
    gather_paged_kv = staticmethod(gather_paged_kv)
