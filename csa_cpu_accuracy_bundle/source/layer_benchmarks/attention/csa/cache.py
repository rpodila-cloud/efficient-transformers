"""DeepSeek-V4 retained-cache reads and writes (folded DP/CP decode).

Cache ABI (see README):

  local_kv_cache           [1, B, window, head_dim]            folded rows, one per batch item
  compressed_kv_cache      [B/dp, dp, comp_slots, head_dim]    DP-major
  compressor_{kv,score}    [1, B, state_len, width]            folded rows
  indexer_kv_cache (CSA)   [B/dp, dp, cp*slots, index_dim]     DP-major, CP way-major inside a row
  indexer_{kv,score}_state [B, 2*ratio, 2*index_dim]

Each method runs eager PyTorch, or emits the QAIC custom op while exporting.
"""

from __future__ import annotations

import torch

from layer_benchmarks.common import custom_ops

from . import ops

INVALID = torch.iinfo(torch.int32).max


class DeepseekV4FoldedAttentionCache:
    """Mixin exposing the folded DP/CP DeepSeek-V4 cache operations."""

    @staticmethod
    def scatter_1d(data: torch.Tensor, positions: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        """data[b, positions[b, s]] = updates[b, s]."""

        def eager(data, positions, updates):
            out = data.clone()
            out[torch.arange(data.shape[0], device=data.device).view(-1, 1), positions.long()] = updates
            return out

        return custom_ops.call(ops.V4CtxScatter1D, (data, positions.to(torch.int32), updates), eager, shape=data.shape)

    @staticmethod
    def scatter_folded(data: torch.Tensor, positions: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        """data [1, B, T, D][0, b, positions[b, s]] = updates [B, S, D]."""

        def eager(data, positions, updates):
            batch, _, seq_len, _ = updates.shape
            out = data.clone()
            batch_idx = torch.arange(batch, device=data.device).view(batch, 1).expand(batch, seq_len)
            out[0, batch_idx, positions.long()] = updates[:, 0]
            return out

        return custom_ops.call(
            ops.FoldedV4CtxFoldedScatter, (data, positions.to(torch.int32), updates.unsqueeze(1)), eager, shape=data.shape
        )

    @staticmethod
    def gather_folded(data: torch.Tensor, indices: torch.Tensor, valid: torch.Tensor, dp: int):
        """Rows of [1, B, T, D] at indices [B, L] -> ([B/dp, dp, L, D], valid [B/dp, dp, L]); invalid rows are zero."""
        safe = torch.where(valid, indices, torch.full_like(indices, INVALID)).to(torch.int32)
        batch, length = indices.shape

        def eager(data, idx):
            idx = torch.where(idx == INVALID, 0, idx)
            return data[0, torch.arange(data.shape[1], device=data.device).view(-1, 1), idx.long()].unsqueeze(0)

        gathered = custom_ops.call(
            ops.FoldedV4CtxGatherFoldedRows, (data, safe), eager, shape=(1, batch, length, data.shape[3])
        ).reshape(batch, length, -1)
        gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros(1, dtype=gathered.dtype))
        return gathered.view(batch // dp, dp, length, -1), valid.view(batch // dp, dp, length)

    @staticmethod
    def scatter_dp(data: torch.Tensor, positions: torch.Tensor, updates: torch.Tensor, dp: int) -> torch.Tensor:
        """data [B/dp, dp, T, D][l, p, positions] = updates [B, S, D] (masked paged scatter)."""
        batch, seq_len, dim = updates.shape
        batch_local = batch // dp
        rows = updates.view(batch_local, dp, seq_len, dim)
        block_id = torch.arange(batch_local).view(batch_local, 1, 1).expand(batch_local, dp, seq_len).to(torch.int32)
        addr = positions.view(batch_local, dp, seq_len).to(torch.int32)

        def eager(data, block_id, addr, updates):
            out = data.clone()
            b, r, s, _ = updates.shape
            row_idx = torch.arange(r, device=data.device).view(1, r, 1).expand(b, r, s)
            valid = block_id != INVALID
            out[block_id[valid].long(), row_idx[valid], addr[valid].long()] = updates[valid]
            return out

        return custom_ops.call(ops.V4CtxPagedScatter, (data, block_id, addr, rows), eager, shape=data.shape)

    @staticmethod
    def gather_dp(data: torch.Tensor, indices: torch.Tensor, valid: torch.Tensor, dp: int):
        """[B/dp, dp, T, D] at indices [B, L] -> ([B/dp, dp, L, D], valid [B/dp, dp, L]); invalid rows are zero."""
        batch, length = indices.shape
        idx = indices.view(batch // dp, dp, length).to(torch.int32)
        valid = valid.view(batch // dp, dp, length)
        safe = torch.where(valid, idx, torch.full_like(idx, INVALID)).to(torch.int32)

        def eager(data, idx):
            idx = torch.where(idx == INVALID, 0, idx)
            b = torch.arange(data.shape[0], device=data.device).view(-1, 1, 1)
            r = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
            return data[b, r, idx.long()]

        gathered = custom_ops.call(ops.V4CtxGatherRows, (data, safe), eager, shape=(*safe.shape, data.shape[3]))
        return torch.where(valid.unsqueeze(-1), gathered, torch.zeros(1, dtype=gathered.dtype)), valid

    @staticmethod
    def scatter_indexer(data: torch.Tensor, slots: torch.Tensor, updates: torch.Tensor, dp: int, cp: int) -> torch.Tensor:
        """Global compressed slot g -> data [B/dp, dp, cp*S, D] at (g % cp) * S + g // cp."""
        batch, seq_len, dim = updates.shape
        batch_local = batch // dp
        local_slots = data.shape[2] // cp
        global_dp = slots.view(batch_local, dp, seq_len).to(torch.int32)
        valid = global_dp != INVALID
        safe = torch.where(valid, global_dp, torch.zeros_like(global_dp))
        addr = torch.remainder(safe, cp) * local_slots + torch.div(safe, cp, rounding_mode="floor")
        block_id = torch.arange(batch_local).view(batch_local, 1, 1).expand(batch_local, dp, seq_len)
        block_id = torch.where(valid, block_id, torch.full_like(block_id, INVALID))

        def eager(data, block_id, addr, updates):
            b, r, s, _ = updates.shape
            row_idx = torch.arange(r, device=data.device).view(1, r, 1).expand(b, r, s)
            ok = block_id != INVALID
            safe_block = torch.where(ok, block_id, torch.zeros_like(block_id)).long()
            safe_addr = torch.where(ok, addr, torch.zeros_like(addr)).long()
            current = data[safe_block, row_idx, safe_addr]
            out = data.clone()
            out[safe_block, row_idx, safe_addr] = torch.where(ok.unsqueeze(-1), updates, current)
            return out

        return custom_ops.call(
            ops.FoldedV4CtxPagedScatter,
            (data, block_id.to(torch.int32), addr.to(torch.int32), updates.view(batch_local, dp, seq_len, dim)),
            eager,
            shape=data.shape,
        )
