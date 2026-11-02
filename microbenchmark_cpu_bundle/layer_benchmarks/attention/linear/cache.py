"""Qwen3.5 linear-attention retained state.

  conv_state       [B, conv_dim, conv_kernel] or [B, conv_kernel, conv_dim]
                   last ``conv_kernel`` pre-conv qkv columns
  recurrent_state  [B, Hv, head_k_dim, head_v_dim]   delta-rule state S

Both are plain retained tensors (no custom op): the conv window is a static
slice and S is overwritten whole.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class Qwen35LinearAttentionCache:
    """Mixin exposing the linear-attention retained-state update."""

    @staticmethod
    def causal_conv1d_update(
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        position_ids: torch.Tensor,
        bias: Optional[torch.Tensor],
        state_layout: str = "channel_first",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Depthwise causal conv and retained-state update for decode.

        The original picks the kept columns with an argsort over position ids; for
        contiguous decode positions that is exactly the trailing ``state_len``
        columns, which a static Slice expresses without TopK/Sign. Inactive
        continuous-batching rows (negative position id) retain their old state.
        """
        _, hidden_size, seq_len = hidden_states.shape
        conv_state_cf = conv_state.transpose(1, 2) if state_layout == "channel_last" else conv_state
        window = torch.cat([conv_state_cf, hidden_states], dim=-1).to(weight.dtype)
        out = F.conv1d(window, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)

        if state_layout == "channel_last":
            shifted = torch.cat(
                [conv_state[:, 1:, :], hidden_states.transpose(1, 2).to(conv_state.dtype)],
                dim=1,
            )
        else:
            shifted = torch.cat(
                [conv_state[..., 1:], hidden_states.to(conv_state.dtype)],
                dim=-1,
            )
        valid = (position_ids[:, -1:] >= 0).to(conv_state.dtype).reshape(-1, 1, 1)
        new_state = shifted * valid + conv_state * (1 - valid)
        return F.silu(out[:, :, -seq_len:]).to(hidden_states.dtype), new_state
