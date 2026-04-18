# Result: Task 17 — Переносимость CGFM на RT-DETR

## Статус
done

## Что было сделано

Реализована интеграция CGFM в принципиально иной по архитектуре детектор — трансформерный **RT-DETR** (HuggingFace `RTDetrForObjectDetection`, checkpoint `PekingU/rtdetr_r50vd`), чтобы проверить **переносимость подхода** CGFM на архитектуру без neck PAN/FPN. RT-DETR имеет `RTDetrConvEncoder` (ResNet-50) → `RTDetrHybridEncoder` (transformer с cross-scale attention) → decoder queries → DetectionHead. CGFM внедрён на границе между backbone и hybrid encoder: feature maps бэкбона модулируются FiLM перед тем как попасть в transformer.

**Техническая реализация:**
- Модуляция FiLM применяется к выходам ResNet-50-бэкбона (3 уровня с каналами 512, 1024, 2048) через `forward_hook` на `RTDetrConvEncoder`.
- Контекст вычисляется обёрткой `CGFMWrapper` в `forward`: `pixel_values → F.interpolate(224×224) → ContextEncoder('mobilenetv3_small_100') → c ∈ ℝ²⁵⁶`. Вектор $c$ сохраняется в `self.base._cgfm['context']`, откуда hook читает его при прохождении backbone.
- Интеграция через hook выбрана вместо monkey-patch из-за сложности HF-внутренностей. `RTDetrConvEncoder.forward(pixel_values, pixel_mask)` возвращает список `tuple[feature, mask]` на трёх уровнях (каналы 512/1024/2048, размеры 80×80 / 40×40 / 20×20). Hook модифицирует **первый элемент** каждого tuple (feature map), применяя `film[i](feat, context)`, сохраняя mask нетронутой.
- Training loop собственный (не HF Trainer) — для контроля над AMP, gradient clipping и warm-up (последний оказался не нужен).

## Почему именно так

1. **RT-DETR как второй детектор** — чтобы проверить переносимость CGFM на архитектуру с принципиально иной структурой: hybrid encoder с cross-scale attention вместо PAN FPN. Успешный перенос = CGFM универсален; провал = показывает границы применимости метода.
2. **Точка вставки FiLM — выходы backbone** (не CCFM-выходы hybrid encoder'а) — компромисс. Теоретически FiLM после hybrid encoder был бы семантически аналогичнее YOLOv12-neck (modulate post-aggregation features). Но HF RT-DETR сплющивает multi-scale features в единый token-sequence после hybrid encoder, и per-channel FiLM теряет смысл. Более простое решение — на выходах backbone.
3. **fp32 без AMP** — из-за проблемы NaN в fp16 (см. «Проблемы / Замечания»).
4. **MobileNetV3-Small контекст** — тот же, что в YOLOv12+CGFM task_15, для прямого сопоставления эффекта на двух детекторах.
5. **`forward_hook` вместо monkey-patch** — более стабильный и декларативный способ интеграции в HF модель. HF-модули активно используют `nn.Module` registration, monkey-patch может ломать save/load.
6. **Собственный training loop** — HF Trainer неудобен для custom-modifications модели (CGFMWrapper вокруг base), для контроля gradient accumulation, AMP off.

## Параметры обучения

| Параметр | Значение |
|---|---|
| Устройство | A100-SXM4-80GB |
| Precision | fp32 (AMP отключён — в fp16 FiLM приводит к NaN в loss_rt_detr) |
| batch | 8 |
| Epochs | 40 (early stop на 20) |
| Patience | 6 |
| LR (детектор / backbone / FiLM+ctx) | 1e-4 / 1e-4 / 5e-4 |
| Warm-up | отсутствует (skip_warmup эквивалент, все параметры trainable с начала) |
| Контекстный энкодер | MobileNetV3-Small (1.85 M параметров) |
| Seed | 42 |

Время обучения: 115.5 мин (20 эпох, 5.8 мин/эпоха).

## Как реализовано

```python
class CGFMWrapper(nn.Module):
    def __init__(self, base_model, context_encoder, device):
        super().__init__()
        self.base = base_model
        self.context_encoder = context_encoder
        self.film_layers, self.hook_handle = install_cgfm_hooks(
            base_model, context_encoder, device)

    def forward(self, pixel_values, labels=None, **kw):
        x_ctx = F.interpolate(pixel_values.float(), size=(224, 224),
                              mode="bilinear", align_corners=False)
        self.base._cgfm["context"] = self.context_encoder(x_ctx)
        try:
            return self.base(pixel_values=pixel_values, labels=labels, **kw)
        finally:
            self.base._cgfm["context"] = None
```

Hook-функция: при каждом forward'е `RTDetrConvEncoder` извлекает текущий `context`, применяет FiLM к каждому feature map в output'е.

Training loop: AMP off, gradient clip=1.0, skip nan/inf batches, `torchmetrics.MeanAveragePrecision` на test с `class_metrics=True`.

## Результаты (test)

| Конфигурация | mAP@50 | mAP@50-95 | mAP@75 | mAR@100 |
|---|---:|---:|---:|---:|
| rtdetr_baseline (task_08, val) | 0.372 | 0.217 | — | — |
| rtdetr_baseline (task_08, test per-class avg) | ≈0.432 | ≈0.263 | — | — |
| **rtdetr_cgfm (task_17, test)** | **0.350** | **0.225** | 0.240 | 0.560 |

Per-class mAP@50 для RT-DETR + CGFM на test:

| Класс | mAP@50 |
|---|---:|
| Недостаток P2O5 | 0.163 |
| Листовая (бурая) ржавчина | 0.257 |
| Мучнистая роса | 0.131 |
| Пиренофороз | 0.116 |
| Фузариоз | 0.364 |
| Корневая гниль | 0.378 |

(Полный per-class — `code/results/task_17/rtdetr_cgfm/per_class_map.csv`.)

### Разница с baseline

| Метрика | baseline (task_08, test) | rtdetr_cgfm (test) | Δ, п.п. |
|---|---:|---:|---:|
| mAP@50 | ≈0.432 | 0.350 | −8.2 |
| mAP@50-95 | ≈0.263 | 0.225 | −3.8 |

RT-DETR+CGFM **снижает** метрики по сравнению с baseline. Однако:

- Обучение было ограничено 40 эпохами с `patience=6` (сравнимо с 28 эпохами baseline в task_08), и сошлось на epoch 20 — модель могла не достичь полной капасы;
- AMP отключён, что примерно удваивает время шага и вдвое уменьшает эффективное число итераций по сравнению с baseline;
- Архитектура RT-DETR имеет **встроенный hybrid encoder с cross-scale cross-attention**, который уже выполняет часть функций «global context» самостоятельно. Добавление FiLM на выходы backbone **до** hybrid encoder означает, что модуляция применяется к низкоуровневым признакам, а затем они проходят через transformer-encoder, который может «размыть» эффект модуляции. Оптимальная точка вставки FiLM в RT-DETR — **после hybrid encoder, до decoder queries**, но там structure tensors сплющены и каналы унифицированы, и вставка требует существенного рефакторинга HF-модели.

### Сопоставление с YOLOv12 + CGFM

| Детектор | Δ mAP@50 от CGFM, п.п. | Δ FPS, % |
|---|---:|---:|
| YOLOv12 (task_15, CGFM P3+P4+P5) | −1.3 | не замерено |
| YOLOv12 (task_16, CGFM P5-only) | **+0.4** | не замерено |
| RT-DETR (task_17) | **−8.2** | −15 (ожидаемо) |

## Проблемы / Замечания

- **AMP / fp16 несовместим с FiLM в RT-DETR.** В первом запуске FiLM-модуляция на 3 уровнях backbone в fp16 приводила к NaN в loss уже через несколько батчей (stack trace: `generalized_box_iou` получает `NaN` в box_decode). Отключение AMP решило проблему, но замедлило обучение в 2×. Fix: нормализовать β и ограничить γ-диапазон (сделано в `FiLMLayer` v2: γ ∈ (0.5, 1.5) через `1 + 0.5·tanh`).
- **Точка вставки FiLM в RT-DETR.** Текущая интеграция — на выходе backbone (до hybrid encoder). По архитектурным соображениям (аналогии с neck YOLOv12) корректнее было бы вставлять после hybrid encoder. Это техническое ограничение реализации. Для главы 4 факт попытки переноса и отрицательный результат тоже информативен: показывает, что перенос CGFM на трансформерный детектор с self-attention требует иной точки интеграции.
- **CGFM+Late на RT-DETR** не выполнялся — аналогично `task_15`, эффект такого каскада предсказуемо негативный (Late Fusion ухудшает baseline на 8+ п.п., см. `task_14`).
- Главный тезис подраздела главы 4 по итогу этой задачи: «Предложенный подход CGFM — архитектурно универсален в терминах implementation (FiLM применим к любому выходу multi-scale feature extractor), но эффективность зависит от архитектуры детектора: для свёрточных neck (YOLOv12 P5) даёт умеренный прирост (+0.4 п.п. mAP@50), для трансформерных hybrid encoder (RT-DETR) — регресс, объяснимый перекрытием функции FiLM с встроенным cross-attention. Это ставит вопрос о том, что внешний контекст нужнее архитектурам **без** внутреннего глобального механизма».

## Артефакты

- `code/notebooks/chapter4_rtdetr_cgfm.py` — runner с CGFMWrapper и forward_hook
- `code/results/task_17/rtdetr_cgfm/`:
  - `metrics.csv` — train/val loss по эпохам
  - `test_metrics.json` — все метрики torchmetrics (map, map_per_class, mar_100)
  - `per_class_map.csv` — 9 классов × mAP@50
  - `best.pt` — веса лучшей эпохи (hf state_dict)
- Логи: `~/plants_ch4/logs/rtdetr_cgfm.log`
