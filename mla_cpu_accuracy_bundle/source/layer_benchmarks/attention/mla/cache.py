"""Retained compressed-KV and rotary-KV cache operations for MLA."""

from __future__ import annotations

import torch

from layer_benchmarks.common import custom_ops

from . import ops

INVALID = torch.iinfo(torch.int32).max


class MLACompressedKVRopeCache:
    """Mixin exposing the compressed-KV / rotary-KV cache operations."""

    @staticmethod
    def scatter(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        """Write [B,H,S,D] updates into data [B,H,T,D] at [B,S] positions."""

        positions = position_ids.to(torch.int32)

        def eager(data, positions, updates):
            out = data.clone()
            b = torch.arange(data.shape[0], device=data.device).view(-1, 1, 1)
            h = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
            out[b, h, positions.long().unsqueeze(1)] = updates
            return out

        return custom_ops.call(ops.CtxScatter, (data, positions, updates), eager, shape=data.shape)

    @staticmethod
    def read_block(data: torch.Tensor, position_ids: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Gather a block and zero positions beyond the current decode position."""

        bsz, heads, _, _ = data.shape
        indices = torch.arange(start, end, device=data.device, dtype=torch.int32)[None, None, :]
        limit = position_ids.to(torch.float32).amax(dim=1, keepdim=True).to(torch.int32).unsqueeze(1)
        invalid = indices > limit
        safe = torch.where(invalid, torch.full_like(indices, INVALID), indices)
        safe = safe.expand(bsz, heads, end - start)

        def eager(data, safe):
            safe = torch.where(safe == INVALID, torch.zeros_like(safe), safe)
            b = torch.arange(data.shape[0], device=data.device).view(-1, 1, 1)
            h = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
            return data[b, h, safe.long()]

        gathered = custom_ops.call(ops.CtxGatherBlockedKV, (data, safe), eager, shape=(bsz, heads, end - start, data.shape[-1]))
        return torch.where(invalid.unsqueeze(-1), torch.zeros_like(gathered), gathered)
