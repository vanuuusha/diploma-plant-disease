# Модули главы 4 — CGFM и бейзлайны

Пакет содержит компоненты для экспериментов из `code/docs/chapter4_protocol.md`.

## Модули

| Файл | Класс | Назначение |
|---|---|---|
| `film_layer.py` | `FiLMLayer` | Feature-wise Linear Modulation: $F' = \gamma \odot F + \beta$ с $\gamma, \beta$, предсказанными из внешнего контекста |
| `context_encoder.py` | `ContextEncoder` | Лёгкий энкодер сцены `[B,3,224,224] → [B, out_dim]` поверх `timm`-бэкбонов |
| `se_block.py` | `SEBlock` | Squeeze-and-Excitation (Hu et al., 2018) — channel self-attention |
| `cbam_block.py` | `CBAMBlock` | CBAM (Woo et al., 2018) — channel + spatial self-attention |
| `late_fusion_head.py` | `LateFusionClassifier` | ROI-feature + контекст → логиты класса (используется в Late Fusion и CGFM+Late) |
| `yolov12_patch.py` | `wrap_neck_with` | Обёртка выходов neck Ultralytics YOLO (v8/v11/v12) указанным блоком |

## Интеграция в YOLOv12

Ultralytics-модель устроена так, что `yolo.model.model[-1]` — это `Detect`, и у неё
`.f = [p3_idx, p4_idx, p5_idx]` — индексы слоёв с выходами P3/P4/P5. `wrap_neck_with`
оборачивает эти слои в `ModulatedLayer`, делегируя оригинальные атрибуты
(`.f`, `.i`, `.type`) чтобы Sequential-проход не ломался.

Для FiLM-варианта дополнительно патчится `yolo.model.forward`: перед каждым
прямым проходом вход 640×640 сжимается до 224×224, прогоняется через
`ContextEncoder` и полученный вектор $c$ инжектируется во все обёрнутые слои.

```python
from ultralytics import YOLO
from models.chapter4 import FiLMLayer, SEBlock, ContextEncoder
from models.chapter4.yolov12_patch import wrap_neck_with

# вариант A: SE-neck (self-attention)
y = YOLO("yolo12m.pt")
wrap_neck_with(y.model, block_factory=SEBlock)

# вариант B: CGFM (context FiLM)
y = YOLO("yolo12m.pt")
enc = ContextEncoder("mobilenetv3_small_100", out_dim=256, pretrained=True)
wrap_neck_with(
    y.model,
    block_factory=lambda ch: FiLMLayer(context_dim=256, feature_channels=ch),
    context_encoder=enc,
)

y.train(data="code/data/dataset_final/data.yaml", epochs=100, ...)
```

## Unit-тесты

```bash
cd code
python tests/chapter4/test_film_layer.py
python tests/chapter4/test_context_encoder.py
python tests/chapter4/test_se_cbam.py
python tests/chapter4/test_late_fusion.py
python tests/chapter4/test_yolov12_patch.py
```
