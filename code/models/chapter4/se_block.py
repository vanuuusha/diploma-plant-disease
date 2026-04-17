"""
SE-блок (Hu et al., 2018). Канальная самоатенция.
    s = GAP(F) ∈ R^C
    w = σ(W2 · ReLU(W1 s))   W1: C→C/r, W2: C/r→C
    F' = w ⊙ F               broadcast по (H, W)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.avg_pool(x).view(b, c)
        w = self.fc(s).view(b, c, 1, 1)
        return x * w
