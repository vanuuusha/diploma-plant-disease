"""
Late-Fusion классификатор: ROI-фича + глобальный контекст → класс.
    roi:       [B, C, 7, 7]
    context_c: [B, context_dim]
    logits:    [B, num_classes]
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LateFusionClassifier(nn.Module):
    def __init__(self, roi_channels: int, roi_spatial: int = 7,
                 context_dim: int = 256, num_classes: int = 9,
                 hidden: int = 256):
        super().__init__()
        roi_flat = roi_channels * roi_spatial * roi_spatial
        self.roi_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(roi_flat, hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + context_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, roi: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        roi_feat = self.roi_mlp(roi)
        return self.head(torch.cat([roi_feat, context], dim=-1))
