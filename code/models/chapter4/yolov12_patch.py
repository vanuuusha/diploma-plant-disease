"""
Интеграция SE/CBAM/FiLM в neck Ultralytics YOLO (v8/v11/v12).

Общая идея:
  1. `YOLO('yolo12m.pt').model` — это `DetectionModel(nn.Module)`, у которого
     `.model` — `nn.Sequential`-подобный список модулей. Последний модуль
     — `Detect`, у него `.f = [p3_idx, p4_idx, p5_idx]` — индексы слоёв neck,
     чьи выходы идут в детектирующую голову.
  2. Оборачиваем эти слои в `ModulatedLayer(orig, block)`: после обычного
     forward добавляем SE/CBAM (self-attention) или FiLM (context).
  3. Для FiLM — патчим `DetectionModel.forward`, чтобы при каждом forward
     генерировать контекст $c$ из входа через `ContextEncoder` и
     инжектировать в все FiLM-слои.

Универсальность: атрибуты Ultralytics (`.f`, `.i`, `.type`, `.np`) делегируются.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulatedLayer(nn.Module):
    """Оборачивает слой Ultralytics и прикладывает блок модуляции к выходу."""

    def __init__(self, orig: nn.Module, block: nn.Module, kind: str = "self"):
        super().__init__()
        if kind not in ("self", "film"):
            raise ValueError(f"unknown kind: {kind}")
        self.orig = orig
        self.block = block
        self.kind = kind
        # Контекст инжектируется извне перед forward (для kind='film')
        self._context: Optional[torch.Tensor] = None
        # Делегирование Ultralytics-атрибутов — чтобы Sequential-проход работал
        if hasattr(orig, "f"):
            self.f = orig.f
        if hasattr(orig, "i"):
            self.i = orig.i
        self.type = getattr(orig, "type", type(orig).__name__)
        self.np = getattr(orig, "np", sum(p.numel() for p in orig.parameters()))
        # Для γ-логирования (обновляется в forward, если kind='film')
        self.last_gamma: Optional[torch.Tensor] = None

    def set_context(self, c: Optional[torch.Tensor]) -> None:
        self._context = c

    def forward(self, x):
        out = self.orig(x)
        if self.kind == "self":
            return self.block(out)
        # kind == "film"
        if self._context is None:
            # sanity: не упасть при dry-run без контекста
            return out
        # сохранить γ для последующего анализа (по возможности)
        if hasattr(self.block, "last_gamma"):
            with torch.no_grad():
                self.last_gamma = self.block.last_gamma(self._context).detach()
        return self.block(out, self._context)


def _find_detect_layer(model: nn.Module) -> nn.Module:
    """Находит Detect-голову (у Ultralytics она всегда последняя)."""
    layers = list(model.model)  # DetectionModel.model — Sequential
    for m in reversed(layers):
        cls = type(m).__name__
        if cls in ("Detect", "v10Detect", "v8Detect", "Segment", "Pose", "OBB"):
            return m
    return layers[-1]


def _infer_neck_channels(det_model: nn.Module) -> list[int]:
    """
    Определяет каналы выходных уровней P3/P4/P5 neck через dry-run.
    Возвращает список в порядке `detect.f`.
    """
    detect = _find_detect_layer(det_model)
    # Способ 1 — через атрибут Detect: часто у него есть .cv2/.cv3 модули
    # со входами из neck; разворачиваем вход Conv.
    try:
        chs = []
        # detect.cv2[i][0].conv.in_channels — вход свёртки на уровне i
        for m in detect.cv2:
            chs.append(m[0].conv.in_channels)
        if len(chs) == len(detect.f):
            return chs
    except Exception:
        pass
    # Способ 2 — dry-run
    device = next(det_model.parameters()).device
    dummy = torch.zeros(1, 3, 640, 640, device=device)
    feats = []

    def hook(mod, inp, out):
        feats.append(out.shape[1])

    handles = [det_model.model[i].register_forward_hook(hook) for i in detect.f]
    with torch.no_grad():
        det_model(dummy)
    for h in handles:
        h.remove()
    return feats


def wrap_neck_with(
    det_model: nn.Module,
    block_factory: Callable[[int], nn.Module],
    context_encoder: Optional[nn.Module] = None,
    context_dim: int = 256,
    levels: Optional[list[str]] = None,
) -> dict:
    """
    Оборачивает выходы neck в блоки модуляции.

    Args:
        det_model: `yolo.model` (DetectionModel).
        block_factory: `callable(channels) -> nn.Module`.
          Для self-attention (SE/CBAM): `lambda c: SEBlock(c)`.
          Для FiLM: `lambda c: FiLMLayer(context_dim, c)`.
        context_encoder: если задан — kind='film', контекст инжектируется
          патчем `forward`.
        context_dim: размерность контекстного вектора.
        levels: подмножество {'P3','P4','P5'} для модуляции (по умолчанию все).
          Используется для аблации (task_16).

    Returns:
        dict с рабочими ссылками: {'modulated_layers': [...],
        'context_encoder': ..., 'levels': [...]}.
    """
    if levels is None:
        levels = ["P3", "P4", "P5"]
    detect = _find_detect_layer(det_model)
    feat_channels = _infer_neck_channels(det_model)
    # detect.f — список индексов в порядке [P3, P4, P5] (от мелкого к крупному
    # receptive field в Ultralytics yaml-конфигурациях v8/v11/v12)
    all_level_names = ["P3", "P4", "P5"]
    kind = "film" if context_encoder is not None else "self"

    modulated_layers: list[ModulatedLayer] = []
    seq = det_model.model  # список модулей (ModuleList / Sequential)
    for lvl_name, idx, chs in zip(all_level_names, detect.f, feat_channels):
        if lvl_name not in levels:
            continue
        orig = seq[idx]
        if kind == "film":
            block = block_factory(chs)  # FiLMLayer(context_dim, chs)
        else:
            block = block_factory(chs)  # SEBlock(chs) или CBAMBlock(chs)
        wrapped = ModulatedLayer(orig, block, kind=kind)
        seq[idx] = wrapped
        modulated_layers.append(wrapped)

    # Для FiLM: патчим DetectionModel.forward — на входе генерируем контекст
    if context_encoder is not None:
        _patch_forward_with_context(det_model, context_encoder, modulated_layers)
        # прикрепляем энкодер к модели, чтобы он участвовал в .to(), .train()
        det_model.context_encoder = context_encoder

    return {
        "modulated_layers": modulated_layers,
        "context_encoder": context_encoder,
        "levels": levels,
        "feat_channels": feat_channels,
    }


def _patch_forward_with_context(det_model: nn.Module,
                                context_encoder: nn.Module,
                                film_layers: list[ModulatedLayer]) -> None:
    """Оборачивает `.forward` so что контекст генерируется и инжектируется."""
    orig_forward = det_model.forward

    def new_forward(x, *args, **kwargs):
        # Генерируем контекст, только если x — тензор изображений
        if isinstance(x, torch.Tensor) and x.ndim == 4 and x.shape[1] == 3:
            # downsample до 224×224 для контекстного энкодера
            x_ctx = F.interpolate(x.float(), size=(224, 224),
                                  mode="bilinear", align_corners=False)
            # нормировка: Ultralytics подаёт уже [0,1], ImageNet-норма не нужна
            # для начальной реализации (backbone timm примет такие данные,
            # хотя оптимально было бы стандартизовать — но это одинаково
            # применяется и в train, и в val, поэтому не вносит утечки)
            c = context_encoder(x_ctx)
            for fl in film_layers:
                fl.set_context(c)
        return orig_forward(x, *args, **kwargs)

    det_model.forward = new_forward  # заменяем bound method


def describe_wrap(wrap_info: dict) -> str:
    """Краткое описание — для RESULT.md."""
    ml = wrap_info["modulated_layers"]
    if not ml:
        return "no modulation"
    kind = ml[0].kind
    lvls = ",".join(wrap_info["levels"])
    chs = wrap_info["feat_channels"]
    return f"kind={kind} levels={lvls} channels={chs}"
