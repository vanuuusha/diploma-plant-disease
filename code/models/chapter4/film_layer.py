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
    """
    Варианты:
      variant='default' (base v2):  γ = 1 + 0.5·tanh(Wc+b) ∈ (0.5, 1.5)
      variant='wide':               γ = 0.1 + 1.8·σ(Wc+b) ∈ (0.1, 1.9)
      variant='beta_noise':         как default, но init β c std=0.1
      residual=True:                F' = F + α·(γF + β - F),  α init=0
    """

    def __init__(self, context_dim: int, feature_channels: int,
                 variant: str = "default", residual: bool = False):
        super().__init__()
        self.context_dim = context_dim
        self.feature_channels = feature_channels
        self.variant = variant
        self.residual = residual

        self.to_gamma = nn.Linear(context_dim, feature_channels)
        self.to_beta = nn.Linear(context_dim, feature_channels)

        if variant == "wide":
            # γ = 0.1 + 1.8·σ(Wc+b): при W=0, b=0 → γ=1 (identity), но широкий диапазон
            nn.init.zeros_(self.to_gamma.weight)
            nn.init.zeros_(self.to_gamma.bias)
            nn.init.normal_(self.to_beta.weight, std=0.01)
            nn.init.zeros_(self.to_beta.bias)
        elif variant == "beta_noise":
            nn.init.normal_(self.to_gamma.weight, std=0.05)
            nn.init.zeros_(self.to_gamma.bias)
            nn.init.normal_(self.to_beta.weight, std=0.1)   # ← значительно больше
            nn.init.zeros_(self.to_beta.bias)
        else:  # default (v2)
            nn.init.normal_(self.to_gamma.weight, std=0.05)
            nn.init.zeros_(self.to_gamma.bias)
            nn.init.normal_(self.to_beta.weight, std=0.01)
            nn.init.zeros_(self.to_beta.bias)

        if residual:
            # F' = F + α·(modulated - F); при α=0 — точный identity
            self.alpha = nn.Parameter(torch.zeros(1))

    def _gamma(self, context: torch.Tensor) -> torch.Tensor:
        if self.variant == "wide":
            return 0.1 + 1.8 * torch.sigmoid(self.to_gamma(context))  # (0.1, 1.9)
        return 1.0 + 0.5 * torch.tanh(self.to_gamma(context))         # (0.5, 1.5)

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma = self._gamma(context)                   # [B, C]
        beta = self.to_beta(context)                   # [B, C]
        modulated = (gamma[..., None, None] * features
                     + beta[..., None, None])
        if self.residual:
            return features + self.alpha * (modulated - features)
        return modulated

    def last_gamma(self, context: torch.Tensor) -> torch.Tensor:
        return self._gamma(context)
