"""Independent eager reference for the DMA arithmetic probe."""

from __future__ import annotations

import torch
from torch import nn


class DMAReference(nn.Module):
    def __init__(self, constants):
        super().__init__()
        self.constants = tuple(c.detach().clone() for c in constants)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for constant in self.constants:
            x = x + constant
        return x
