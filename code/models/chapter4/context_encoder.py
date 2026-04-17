"""
Контекстный энкодер сцены (E_ctx).
Изображение 224×224 → d-мерный вектор c.

Backbone-варианты (timm):
  mobilenetv3_small_100  ≈ 1.5M  feat=576   (default, fast)
  efficientnet_b0        ≈ 5.3M  feat=1280  (mid)
  vit_tiny_patch16_224   ≈ 5.7M  feat=192   (transformer)

После backbone — GAP (для CNN) → LayerNorm → Linear(→out_dim) → ReLU → Linear(→out_dim).
"""
from __future__ import annotations

import torch
import torch.nn as nn


SUPPORTED = ("mobilenetv3_small_100", "efficientnet_b0", "vit_tiny_patch16_224")


class ContextEncoder(nn.Module):
    def __init__(self, backbone: str = "mobilenetv3_small_100",
                 out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        if backbone not in SUPPORTED:
            raise ValueError(f"unsupported backbone: {backbone}")
        import timm
        self.backbone_name = backbone
        self.out_dim = out_dim
        # num_classes=0 + global_pool='avg' → [B, feat_dim]
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        # Динамически определяем фактический выход через dry-run
        # (некоторые timm-модели имеют conv_head, увеличивающий размерность
        # поверх backbone num_features — тогда fallback `num_features` неверен).
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[-1]
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        feat = self.backbone(x)            # [B, feat_dim]
        return self.proj(feat)             # [B, out_dim]

    @staticmethod
    def input_size() -> int:
        return 224
