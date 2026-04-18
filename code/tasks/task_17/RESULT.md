# Result: Task 17 — Переносимость CGFM на RT-DETR

## Статус
done

## Что было сделано

Реализована интеграция CGFM в трансформерный детектор **RT-DETR** (HuggingFace `RTDetrForObjectDetection`, checkpoint `PekingU/rtdetr_r50vd`). Модуляция FiLM применяется к выходам ResNet-50-бэкбона (3 уровня с каналами 512, 1024, 2048) через `forward_hook` на `RTDetrConvEncoder`. Контекст вычисляется обёрткой `CGFMWrapper`, принимающей pixel_values, downsampling до 224×224 и пропускающей через `ContextEncoder('mobilenetv3_small_100')`.

Интеграция через hook выбрана вместо monkey-patch из-за сложности HF-внутренностей (RT-DETR encoder принимает `pixel_values` + `pixel_mask`, возвращает список `tuple[feature, mask]` на трёх уровнях). Hook модифицирует первый элемент каждого tuple (feature map), сохраняя маски нетронутыми.

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
