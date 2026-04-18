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
        # γ = 1 + 0.5·tanh(Wc + b); инициализация W с небольшим шумом —
        # γ ≠ 1 сразу, чтобы модуляция влияла на forward pass c самого начала
        # (идентичная инициализация приводит к collapse: при γ=1 BN абсорбирует
        # модуляцию и модель обучается как baseline).
        # Масштаб 0.5 ограничивает γ ∈ (0.5, 1.5) — безопасная модуляция.
        nn.init.normal_(self.to_gamma.weight, std=0.05)
        nn.init.zeros_(self.to_gamma.bias)
        # β с малым шумом для выхода из identity-ловушки.
        nn.init.normal_(self.to_beta.weight, std=0.01)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # features: [B, C, H, W]; context: [B, D]
        gamma = 1.0 + 0.5 * torch.tanh(self.to_gamma(context))  # [B, C], (0.5, 1.5)
        beta = self.to_beta(context)                             # [B, C]
        return gamma[..., None, None] * features + beta[..., None, None]

    def last_gamma(self, context: torch.Tensor) -> torch.Tensor:
        """Вернуть γ без модификации features (для γ-анализа)."""
        return 1.0 + 0.5 * torch.tanh(self.to_gamma(context))
