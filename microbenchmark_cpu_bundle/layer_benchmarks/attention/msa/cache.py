"""MiniMax-M3 GQA/MSA retained-cache reads and writes.

Cache ABI (see README):

  gqa key/value   decode/paged  [B/attn_dp, attn_dp*attn_cp*kv_heads, ctx/attn_cp, head_dim]
                  prefill       [B, kv_heads, ctx, head_dim]
  msa key/value   contiguous    [B/msa_attn_dp, msa_attn_dp*msa_attn_cp*kv_heads, ctx/msa_attn_cp, head_dim]
                  paged         [attn_phys_blocks, msa_attn_dp*msa_attn_cp*kv_heads, page, head_dim]
  index_key       contiguous    [B/idx_dp, idx_dp*idx_cp*indexer_n_head, ctx/idx_cp, index_dim]
                  paged         [idx_phys_blocks, idx_dp*idx_cp*indexer_n_head, page, index_dim]

Each method runs eager PyTorch, or emits the QAIC custom op while exporting.
"""

from __future__ import annotations

import torch

from layer_benchmarks.common import custom_ops

from . import ops

INVALID = torch.iinfo(torch.int32).max


class MiniMaxM3MSAAttentionCache:
    """Mixin exposing the GQA/MSA cache operations to the attention math."""

    @staticmethod
    def ctx_scatter(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        """data [B, H, T, D][b, h, position_ids[b, s]] = updates."""

        def eager(data, position_ids, updates):
            out = data.clone()
            b = torch.arange(data.shape[0]).view(-1, 1, 1)
            h = torch.arange(data.shape[1]).view(1, -1, 1)
            out[b, h, position_ids.long().unsqueeze(1)] = updates
            return out

        return custom_ops.call(ops.M3CtxScatter, (data, position_ids, updates), eager, shape=data.shape)

    @staticmethod
    def paged_scatter(data, block_id, addr, updates) -> torch.Tensor:
        """data [P, R, T, D][block_id, r, addr] = updates [B, R, S, D]; INT32_MAX block ids are dropped."""

        def eager(data, block_id, addr, updates):
            out = data.clone()
            rows = torch.arange(updates.shape[1]).view(1, -1, 1).expand_as(block_id)
            valid = block_id != INVALID
            out[block_id[valid].long(), rows[valid], addr[valid].long()] = updates[valid]
            return out

        return custom_ops.call(ops.CtxPagedScatter, (data, block_id, addr, updates), eager, shape=data.shape)

    @staticmethod
    def paged_scatter_page(data, block_id, updates) -> torch.Tensor:
        """data [P, H, page, D][block_id[p, h], h] = updates [pages, H, page, D]."""

        def eager(data, block_id, updates):
            out = data.clone()
            heads = torch.arange(updates.shape[1]).view(1, -1).expand_as(block_id)
            valid = block_id != INVALID
            out[block_id[valid].long(), heads[valid]] = updates[valid]
            return out

        return custom_ops.call(ops.CtxPagedScatterPage, (data, block_id, updates), eager, shape=data.shape)

    @staticmethod
    def _gather_blocked(op, data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
        def eager(data, ctx_indices):
            idx = torch.where(ctx_indices == INVALID, torch.zeros_like(ctx_indices), ctx_indices).long()
            b = torch.arange(data.shape[0]).view(-1, 1, 1)
            r = torch.arange(data.shape[1]).view(1, -1, 1)
            return data[b, r, idx]

        return custom_ops.call(op, (data, ctx_indices), eager, shape=(*ctx_indices.shape, data.shape[-1]))

    @staticmethod
    def gather_blocked_kv_dp(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
        """data [B, R, T, D] at ctx_indices [B, R, L] -> [B, R, L, D]."""
        return MiniMaxM3MSAAttentionCache._gather_blocked(ops.CtxGatherBlockedKVDP, data, ctx_indices)

    @staticmethod
    def gather_blocked_kv(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
        """data [B, Hkv, T, D] at ctx_indices [B, Hkv, L] -> [B, Hkv, L, D]."""
        return MiniMaxM3MSAAttentionCache._gather_blocked(ops.CtxGatherBlockedKV, data, ctx_indices)

    @staticmethod
    def gather_blocked_kv_blocks(data: torch.Tensor, block_ids: torch.Tensor, block_size: int) -> torch.Tensor:
        """Gather complete contiguous blocks using one index per block."""
        if data.shape[2] % block_size:
            raise ValueError(f"cache length {data.shape[2]} must be divisible by block size {block_size}.")
        blocked = data.reshape(data.shape[0], data.shape[1], data.shape[2] // block_size, block_size, data.shape[3])

        def eager(blocked, block_ids):
            ids = torch.where(block_ids == INVALID, torch.zeros_like(block_ids), block_ids).long()
            b = torch.arange(blocked.shape[0], device=blocked.device).view(-1, 1, 1)
            r = torch.arange(blocked.shape[1], device=blocked.device).view(1, -1, 1)
            return blocked[b, r, ids]

        shape = (*block_ids.shape, block_size, data.shape[-1])
        return custom_ops.call(ops.CtxGatherBlockRangeKVDP, (blocked, block_ids), eager, shape=shape)

    @staticmethod
    def gather_paged_kv_dp(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        """Physical pages [P, R, page, D] at block_ids [pages, R] -> [1, R, pages*page, D]."""
        # Legacy TorchScript export represents shape values as trace tensors.
        # Freeze these static ABI dimensions before the eager callback closes
        # over them; otherwise torch.autograd.Function sees a value owned by a
        # different trace graph and fails with ``unordered_map::at``.
        pages, rows = int(block_ids.shape[0]), int(block_ids.shape[1])

        def eager(data, block_ids):
            ids = torch.where(block_ids == INVALID, torch.zeros_like(block_ids), block_ids).long()
            row_idx = torch.arange(rows, device=data.device).view(rows, 1)
            out = data[ids.transpose(0, 1), row_idx]
            return out.reshape(1, rows, pages * data.shape[2], data.shape[3])

        shape = (1, rows, pages * data.shape[2], data.shape[3])
        return custom_ops.call(ops.CtxGatherPagedKVDP, (data, block_ids), eager, shape=shape)

    @staticmethod
    def gather_paged_kv_heads(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        """Physical pages [P, H, page, D] at block_ids [pages, H] -> [1, H, pages*page, D]; INT32_MAX pages are 0."""
        pages, heads = int(block_ids.shape[0]), int(block_ids.shape[1])

        def eager(data, block_ids):
            valid = block_ids != INVALID
            ids = torch.where(valid, block_ids, torch.zeros_like(block_ids)).long()
            head_idx = torch.arange(heads, device=data.device).view(heads, 1)
            out = data[ids.transpose(0, 1), head_idx]
            valid = valid.transpose(0, 1)
            out = torch.where(valid[..., None, None], out, torch.zeros_like(out))
            return out.reshape(1, heads, pages * data.shape[2], data.shape[3])

        shape = (1, heads, pages * data.shape[2], data.shape[3])
        return custom_ops.call(ops.CtxGatherPagedKVHeads, (data, block_ids), eager, shape=shape)

    @staticmethod
    def gather_paged_kv_batch_with_heads(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        """block_ids [B, pages] shared by all heads -> [B, H, pages*page, D]."""
        batch, num_pages = int(block_ids.shape[0]), int(block_ids.shape[1])
        num_heads = data.shape[1]
        outputs = []
        for batch_idx in range(batch):
            ids = block_ids[batch_idx].to(torch.int32).view(num_pages, 1).expand(num_pages, num_heads)
            outputs.append(MiniMaxM3MSAAttentionCache.gather_paged_kv_heads(data, ids))
        return torch.cat(outputs, dim=0)

    @staticmethod
    def gather_paged_kv_selected_heads(data: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        """block_ids [B, H, pages] -> [B, H, pages*page, D]."""
        outputs = []
        for batch_idx in range(int(block_ids.shape[0])):
            ids = block_ids[batch_idx].transpose(0, 1).contiguous().to(torch.int32)
            outputs.append(MiniMaxM3MSAAttentionCache.gather_paged_kv_heads(data, ids))
        return torch.cat(outputs, dim=0)
