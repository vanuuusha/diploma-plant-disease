"""
CBAM (Woo et al., 2018). Channel + Spatial attention.

Channel:
    w_c = σ( MLP(GAP(F)) + MLP(GMP(F)) )     shape [C]
    F1 = w_c ⊙ F
Spatial:
    pool = concat[ mean(F1, dim=1), max(F1, dim=1) ]    shape [2, H, W]
    w_s = σ( Conv7×7(pool) )                            shape [1, H, W]
    F2 = w_s ⊙ F1
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        a = self.mlp(self.avg(x).view(b, c))
        m = self.mlp(self.max(x).view(b, c))
        w = self.sig(a + m).view(b, c, 1, 1)
        return x * w


class _SpatialAttention(nn.Module):
    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.mean(dim=1, keepdim=True)
        m = x.max(dim=1, keepdim=True).values
        w = self.sig(self.conv(torch.cat([a, m], dim=1)))
        return x * w


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel = _ChannelAttention(channels, reduction)
        self.spatial = _SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))
