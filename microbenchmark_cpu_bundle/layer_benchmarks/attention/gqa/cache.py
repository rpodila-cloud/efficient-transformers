"""Qwen3 GQA retained-cache reads and writes.

Cache ABI: ``k_cache`` / ``v_cache`` are [B, Hkv, ctx_len, D] at the graph boundary.
Batch-split decode folds them to [1, B*Hkv, ctx_len, D] only around its ops.

Each method runs eager PyTorch, or emits the QAIC custom op while exporting.
Gather slots past the last written position get INT32_MAX while exporting, so
AIC skips the read, and slot 0 in eager mode.
"""

from __future__ import annotations

import torch

from layer_benchmarks.common import custom_ops

from . import ops

INVALID = torch.iinfo(torch.int32).max


class Qwen3GQAKVCache:
    """Mixin exposing the GQA KV-cache operations to the attention math."""

    @staticmethod
    def pos_max(position_ids: torch.Tensor) -> torch.Tensor:
        """[B, QL] -> [B, 1]. Reduces in fp32: dynamo would emit an int32 ReduceMax (positions < 2**24 are exact)."""
        if position_ids.shape[1] == 1:
            return position_ids
        return position_ids.to(torch.float32).amax(1, keepdim=True).to(position_ids.dtype)

    @staticmethod
    def _invalid_value() -> int:
        return INVALID if torch.compiler.is_exporting() else 0

    @staticmethod
    def scatter(cache: torch.Tensor, position_ids: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """cache [B, H, T, D][b, h, position_ids[b, s]] = new [B, H, S, D] (plain index_put -> ScatterND)."""
        b_idx = torch.arange(cache.shape[0], device=cache.device).view(-1, 1, 1)
        h_idx = torch.arange(cache.shape[1], device=cache.device).view(1, -1, 1)
        cache = cache.clone()
        cache[b_idx, h_idx, position_ids.long().unsqueeze(1)] = new
        return cache

    @staticmethod
    def scatter_batch(cache: torch.Tensor, position_ids: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """Same write as ``scatter``, through CtxChunkScatterBatch on the [1, B*H, T, D] fold."""
        B, H, T, D = cache.shape

        def eager(data, position_ids, updates):
            batch, heads, seq_len, _ = updates.shape
            b_idx = torch.arange(batch, device=data.device).view(batch, 1, 1).expand(batch, heads, seq_len)
            h_idx = torch.arange(heads, device=data.device).view(1, heads, 1).expand(batch, heads, seq_len)
            out = data.clone()
            out[0, b_idx * heads + h_idx, position_ids.long().unsqueeze(1).expand(batch, heads, seq_len)] = updates
            return out

        folded = cache.reshape(1, B * H, T, D)
        out = custom_ops.call(ops.CtxChunkScatterBatch, (folded, position_ids, new), eager, shape=folded.shape)
        return out.reshape(B, H, T, D)

    @staticmethod
    def scatter_chunk(cache: torch.Tensor, position_ids: torch.Tensor, new: torch.Tensor,
                      *, chunk_n: int, chunk_size: int) -> torch.Tensor:
        """Write a plain [B,H,S,D] update into the packed chunk-KV cache."""
        B, rows, _, D = cache.shape
        H = new.shape[1]
        split = rows // (H * chunk_n)
        if chunk_size % split:
            raise ValueError(f"chunk_size ({chunk_size}) must be divisible by split ({split})")
        nc = chunk_n * split
        p = chunk_size // split
        pos = position_ids.long()
        group = (pos // p) % nc
        col = (pos // (p * nc)) * p + pos % p
        b = torch.arange(B, device=cache.device).view(B, 1, 1).expand(B, H, pos.shape[1])
        h = torch.arange(H, device=cache.device).view(1, H, 1).expand(B, H, pos.shape[1])
        out = cache.clone()

        def eager(data, positions, updates):
            result = data.clone()
            result[b, h * nc + group.unsqueeze(1).expand(B, H, pos.shape[1]), col.unsqueeze(1).expand(B, H, pos.shape[1])] = updates
            return result

        fn = ops.CtxChunkScatter if p == 1 else ops.CtxChunkScatterPacked
        attrs = {} if p == 1 else {"p": p}
        return custom_ops.call(fn, (out, position_ids, new), eager, shape=out.shape, attrs=attrs)

    @staticmethod
    def unpack_chunk(cache: torch.Tensor, *, ctx_len: int, chunk_n: int, chunk_size: int,
                    num_kv_heads: int) -> torch.Tensor:
        """Inverse packed chunk layout, used by the eager/reference reduction."""
        split = cache.shape[1] // (num_kv_heads * chunk_n)
        nc = chunk_n * split
        p = chunk_size // split
        pos = torch.arange(ctx_len, device=cache.device)
        group = (pos // p) % nc
        col = (pos // (p * nc)) * p + pos % p
        row = torch.arange(num_kv_heads, device=cache.device).view(-1, 1) * nc + group
        b = torch.arange(cache.shape[0], device=cache.device).view(-1, 1, 1)
        return cache[b, row.unsqueeze(0), col.view(1, 1, -1)]

    @staticmethod
    def read_block(cache: torch.Tensor, position_ids: torch.Tensor, start: int, end: int, *, zero_invalid: bool):
        """cache [B, H, T, D] slots [start, end) -> [B, H, end-start, D]; V zeroes slots past the position."""
        B, H, _, D = cache.shape
        idx = torch.arange(start, end, device=cache.device)[None, None, ...]
        invalid = idx > Qwen3GQAKVCache.pos_max(position_ids).unsqueeze(1)
        idx = torch.where(invalid, Qwen3GQAKVCache._invalid_value(), idx).to(torch.int32).expand(B, H, idx.shape[-1])

        def eager(data, ctx_indices):
            ctx_indices = torch.where(ctx_indices == INVALID, 0, ctx_indices).long()
            return data[torch.arange(data.shape[0]).view(-1, 1, 1), torch.arange(data.shape[1]).view(1, -1, 1), ctx_indices]

        out = custom_ops.call(ops.CtxGatherBlockedKV, (cache, idx), eager, shape=(B, H, end - start, D))
        if zero_invalid:
            out = torch.where(invalid.unsqueeze(-1), torch.tensor(0.0, dtype=out.dtype, device=out.device), out)
        return out

    @staticmethod
    def read_block_batch(folded: torch.Tensor, position_ids: torch.Tensor, start: int, end: int, *, zero_invalid: bool):
        """folded [1, BH, T, D] slots [start, end) -> [1, BH, end-start, D].

        The limit is folded to [1, BH, 1] on the tiny [B] side, so the index chain
        into the gather ends Where -> Cast -> Expand. The compiler's GatherNDRange fast
        path keys off that tail; a merge Reshape there falls back to a plain DDR GatherND.
        """
        B = position_ids.shape[0]
        BH, D = folded.shape[1], folded.shape[3]
        limit = Qwen3GQAKVCache.pos_max(position_ids).unsqueeze(1).expand(B, BH // B, 1).reshape(1, BH, 1)
        idx = torch.arange(start, end, device=folded.device)[None, None, ...]
        invalid = idx > limit
        idx = torch.where(invalid, Qwen3GQAKVCache._invalid_value(), idx).to(torch.int32).expand(1, BH, end - start)

        def eager(data, ctx_indices):
            ctx_indices = torch.where(ctx_indices == INVALID, 0, ctx_indices).long()
            return data[0, torch.arange(data.shape[1], device=data.device).view(-1, 1), ctx_indices[0]].unsqueeze(0)

        out = custom_ops.call(ops.CtxGatherBlockedKVBatch, (folded, idx), eager, shape=(1, BH, end - start, D))
        if zero_invalid:
            invalid = invalid.expand(1, BH, end - start)
            out = torch.where(invalid.unsqueeze(-1), torch.tensor(0.0, dtype=out.dtype, device=out.device), out)
        return out

    @staticmethod
    def read_chunk_block(
        cache: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
        end: int,
        *,
        chunk_n: int,
        chunk_size: int,
        split: int,
        zero_invalid: bool,
    ):
        """Read a block of slots from the packed chunk-KV cache.

        ``start``/``end`` are packed-cache slot indices, not logical token
        positions.  This is the cache ABI used by the standalone
        ``decode_headpar_chunk_kv_unroll`` path.
        """
        B, rows, _, D = cache.shape
        p = chunk_size // split
        nc = chunk_n * split
        pnc = p * nc
        pos_max = Qwen3GQAKVCache.pos_max(position_ids)
        # Positions are non-negative in this benchmark.  Keep this as a
        # floating-point floor division so ONNX lowering does not introduce a
        # Sign node for integer floor/trunc division.
        gather_limit = (
            torch.floor(pos_max.to(torch.float32) / float(pnc)) * p + p - 1
        ).unsqueeze(1)
        slot = torch.arange(start, end, device=cache.device)[None, None, ...]
        invalid = slot > gather_limit
        slot = torch.where(
            invalid,
            Qwen3GQAKVCache._invalid_value(),
            slot,
        ).to(torch.int32).expand(B, rows, end - start)

        def eager(data, ctx_indices):
            ctx_indices = torch.where(ctx_indices == INVALID, 0, ctx_indices).long()
            return data[
                torch.arange(data.shape[0], device=data.device).view(-1, 1, 1),
                torch.arange(data.shape[1], device=data.device).view(1, -1, 1),
                ctx_indices,
            ]

        out = custom_ops.call(
            ops.CtxGatherBlockedKV,
            (cache, slot),
            eager,
            shape=(B, rows, end - start, D),
        )
        if zero_invalid:
            out = torch.where(
                invalid.expand(B, rows, end - start).unsqueeze(-1),
                torch.zeros_like(out),
                out,
            )
        return out

    @staticmethod
    def causal_mask(position_ids: torch.Tensor, BH: int, start: int, end: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        B = position_ids.shape[0]
        limit = Qwen3GQAKVCache.pos_max(position_ids).unsqueeze(1).expand(B, BH // B, 1).reshape(1, BH, 1)
        idx = torch.arange(start, end, device=device)[None, None, ...]
        invalid = (idx > limit).bool()
        idx = torch.where(invalid, Qwen3GQAKVCache._invalid_value(), idx).to(torch.int32).expand(1, BH, end - start)
        return invalid, idx

    @staticmethod
    def read_block_batch_with_mask(folded: torch.Tensor, invalid: torch.Tensor, idx: torch.Tensor, start: int, end: int, zero_invalid: bool) -> torch.Tensor:
        BH, D = folded.shape[1], folded.shape[3]

        def eager(data, ctx_indices):
            ctx_indices = torch.where(ctx_indices == INVALID, 0, ctx_indices).long()
            return data[0, torch.arange(data.shape[1], device=data.device).view(-1, 1), ctx_indices[0]].unsqueeze(0)

        out = custom_ops.call(ops.CtxGatherBlockedKVBatch, (folded, idx), eager, shape=(1, BH, end - start, D))
        if zero_invalid:
            invalid = invalid.expand(1, BH, end - start)
            out = torch.where(invalid.unsqueeze(-1), torch.tensor(0.0, dtype=out.dtype, device=out.device), out)
        return out
