"""
FiLM-слой — Feature-wise Linear Modulation (Perez et al., 2018).

    c  — контекстный вектор [B, context_dim]
    γ  = σ(W_γ c + b_γ)  ∈ [0, 1]    scaling   per-channel
    β  =  W_β c + b_β                shifting  per-channel
    F' = γ ⊙ F + β      broadcast по (H, W)

Инициализация: W_γ ~ N(0, 1e-4) так, что γ ≈ 0.5 в начале обучения
(близко к identity при β=0) — это стабилизирует warm-up.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    def __init__(self, context_dim: int, feature_channels: int):
        super().__init__()
        self.context_dim = context_dim
        self.feature_channels = feature_channels
        self.to_gamma = nn.Linear(context_dim, feature_channels)
        self.to_beta = nn.Linear(context_dim, feature_channels)
        # γ: инициализация близко к нулю → после sigmoid ≈ 0.5
        nn.init.normal_(self.to_gamma.weight, std=1e-4)
        nn.init.zeros_(self.to_gamma.bias)
        # β: нулевая инициализация → identity при γ=0.5 → F' = 0.5·F
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # features: [B, C, H, W]; context: [B, D]
        gamma = torch.sigmoid(self.to_gamma(context))  # [B, C], [0, 1]
        beta = self.to_beta(context)                   # [B, C]
        return gamma[..., None, None] * features + beta[..., None, None]

    def last_gamma(self, context: torch.Tensor) -> torch.Tensor:
        """Вернуть γ без модификации features (для γ-анализа / визуализации)."""
        return torch.sigmoid(self.to_gamma(context))
