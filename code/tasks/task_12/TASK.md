# Task 12: Инфраструктура главы 4 — модули CGFM/SE/CBAM/Late и фиксация baseline

## Статус
pending

## Цель

Подготовить общий код для всех экспериментов главы 4: реализовать и протестировать модули (FiLM-слой, контекстный энкодер, SE-блок, CBAM-блок, Late-Fusion-классификатор), описать способ интеграции в neck YOLOv12, зафиксировать baseline (результат `task_07/yolov12_aug_diffusion`) как reference-строку для сводной таблицы главы 4. Без этого фундамента последующие задачи `task_13`–`task_17` не могут быть запущены в единообразных условиях.

## Общий протокол

Все решения по гиперпараметрам, датасетам, формату артефактов, именам конфигураций, замеру FPS — в `code/docs/chapter4_protocol.md`. Не отклоняться от него без явной договорённости.

## Шаги

1. **Создать директорию** `code/models/chapter4/` и инициализировать пакет (`__init__.py`).
2. **Реализовать FiLM-слой** — `code/models/chapter4/film_layer.py`:
   - Класс `FiLMLayer(context_dim: int, feature_channels: int)`.
   - Формула $F'_l = \sigma(W_\gamma c) \odot F_l + W_\beta c$ (γ через сигмоиду для ограничения масштабирования в $[0,1]$, β — линейно).
   - Инициализация: $W_\gamma$ — ближе к нулю (чтобы сначала модуляция ≈ 0.5, близка к identity), $W_\beta$ — нулевой.
   - Прямой проход поддерживает batch-форму `[B, C, H, W]` для признаков и `[B, context_dim]` для контекста.
3. **Реализовать контекстный энкодер** — `code/models/chapter4/context_encoder.py`:
   - Класс `ContextEncoder(backbone: str, out_dim: int = 256, pretrained: bool = True)`.
   - Поддержать 3 backbone через `timm.create_model`: `mobilenetv3_small_100`, `efficientnet_b0`, `vit_tiny_patch16_224`.
   - После backbone — GlobalAvgPool (если CNN), LayerNorm, Linear → `out_dim`.
   - Вход: стандартный `[B, 3, 224, 224]` (изображение масштабируется до 224 специально для контекстной ветви; при этом детектор продолжает работать с `640×640`).
4. **Реализовать SE-блок** — `code/models/chapter4/se_block.py`:
   - Класс `SEBlock(channels: int, reduction: int = 16)`.
   - Стандарт Hu et al., 2018: GAP → FC(C→C/r) → ReLU → FC(C/r→C) → Sigmoid → scale.
5. **Реализовать CBAM-блок** — `code/models/chapter4/cbam_block.py`:
   - Класс `CBAMBlock(channels: int, reduction: int = 16, spatial_kernel: int = 7)`.
   - Channel Attention: GAP + GMP → общий MLP → sigmoid → scale.
   - Spatial Attention: channel-wise pool (avg + max, concat) → conv 7×7 → sigmoid → scale.
6. **Реализовать Late-Fusion-классификатор** — `code/models/chapter4/late_fusion_head.py`:
   - Класс `LateFusionClassifier(roi_dim: int, context_dim: int, num_classes: int = 9)`.
   - Вход: ROI-фича (можно получить через `torchvision.ops.roi_align` из feature map P4 по предсказанным bbox), контекстный вектор c.
   - Архитектура: ROI → flatten → FC → concat с c → FC → logits.
7. **Описать способ интеграции в neck YOLOv12** — `code/models/chapter4/yolov12_patch.py`:
   - Функция `wrap_neck_with(model: YOLO, block_factory: Callable, context_encoder: Optional[ContextEncoder] = None)`.
   - Для каждого выхода P3/P4/P5 neck (в Ultralytics YOLOv12 это индексы модулей из `model.model.model`, найти их можно по типу — последние три `C2f` перед `Detect`) вставлять дополнительный блок.
   - Если `context_encoder` задан — использовать FiLMLayer с внешним контекстом; иначе — SE/CBAM (самореферентные).
   - Сохранить возможность загружать исходные веса Ultralytics: новые слои — only-new, не ломать `load_state_dict(strict=False)`.
8. **Написать unit-тесты** — `code/tests/chapter4/`:
   - `test_film_layer.py` — проверка форм, identity при нулевых β и γ=1, градиенты по всем параметрам.
   - `test_context_encoder.py` — форма выхода для каждого backbone, что `timm`-веса корректно подгружаются.
   - `test_se_cbam.py` — формы, identity при инициализации масштабирования в 1.
   - `test_late_fusion.py` — форма logits, прогон одного ROI.
   - `test_yolov12_patch.py` — проверка, что обёрнутая модель: (а) принимает вход `[B, 3, 640, 640]`, (б) возвращает корректные форматы Detect-головы, (в) градиенты текут во все новые параметры, (г) загрузка COCO-весов работает через `strict=False`.
   - Все тесты — через `pytest`; прогоняются командой `pytest code/tests/chapter4/ -v`.
9. **Зафиксировать baseline для таблицы главы 4**:
   - Скопировать (или симлинк) `code/results/task_07/yolov12_aug_diffusion/` → `code/results/task_12/yolov12_baseline/`.
   - Убедиться, что для baseline присутствуют `metrics.csv`, `per_class_map.csv`, `fps_measurement.json`, `predictions_examples/`.
   - Дополнительно подсчитать `param_count.json` для baseline (число параметров и MACs через `thop`/`fvcore`) — он нужен как точка сравнения со всеми модифицированными конфигурациями.
   - Если каких-то артефактов не хватает (например, `per_class_map.csv` не был сохранён в task_07) — дочинить: загрузить `best.pt`, прогнать `model.val(..., split='test')` и сохранить недостающее.
10. **Извлечь контекстные эмбеддинги baseline** (нужны для t-SNE в `task_18`):
    - Запустить `ContextEncoder(backbone='mobilenetv3_small_100')` на всём test-срезе (изображения ресайзятся до 224).
    - Сохранить `[N_test, 256]` в `code/results/task_12/yolov12_baseline/context_embeddings.npy`.
    - Это даёт «контекстный эмбеддинг без дообучения» — референс для сравнения с CGFM-версией после обучения.
11. **Коммит** — `chapter4 infra: modules + tests + baseline fixed` + `git push`.

## Входные данные

- `code/data/dataset_final/` (только как reference, обучение в этой задаче не производится)
- `code/results/task_07/yolov12_aug_diffusion/` (источник baseline)
- `code/results/task_08/rtdetr_aug_diffusion/` (источник baseline RT-DETR — пригодится в task_17)
- Предобученные веса backbone-ов (`mobilenetv3`, `efficientnet_b0`, `vit_tiny`) скачиваются `timm` автоматически

## Ожидаемый результат

- Пакет `code/models/chapter4/` с рабочими модулями (все тесты зелёные).
- Документ-памятка `code/models/chapter4/README.md` — краткое описание модулей и способа интеграции в YOLOv12 (со схемой).
- `code/results/task_12/yolov12_baseline/` — полный набор артефактов baseline (скопирован из task_07), плюс `param_count.json` и `context_embeddings.npy`.
- Заполненный `RESULT.md` по шаблону ниже.

## Результат записать в

`code/tasks/task_12/RESULT.md` по следующему шаблону:

```markdown
# Result: Task 12 — Инфраструктура главы 4

## Статус
done | partial | failed

## Что было сделано
Краткое описание: какие модули созданы, результат прогона unit-тестов, какой backbone выбран для пробного эмбеддинга.

## Результаты тестов

| Тест | Статус | Покрытие |
|---|---|---|
| test_film_layer.py | pass | 100% |
| test_context_encoder.py | pass | ... |
| test_se_cbam.py | pass | ... |
| test_late_fusion.py | pass | ... |
| test_yolov12_patch.py | pass | ... |

## Baseline для сводной таблицы главы 4

| Метрика | Значение | Источник |
|---|---:|---|
| mAP@50 | ... | task_07/yolov12_aug_diffusion |
| mAP@50-95 | ... | task_07/yolov12_aug_diffusion |
| Precision | ... | task_07/yolov12_aug_diffusion |
| Recall | ... | task_07/yolov12_aug_diffusion |
| FPS | ... | task_07/yolov12_aug_diffusion |
| Параметров, M | ... | (пересчитано) |
| GFLOPs | ... | (пересчитано через thop) |

## Проблемы / Замечания
- Какие-либо расхождения при подмене neck
- Недостающие артефакты baseline (если пришлось дочинивать)

## Артефакты
- `code/models/chapter4/film_layer.py`
- `code/models/chapter4/context_encoder.py`
- `code/models/chapter4/se_block.py`
- `code/models/chapter4/cbam_block.py`
- `code/models/chapter4/late_fusion_head.py`
- `code/models/chapter4/yolov12_patch.py`
- `code/models/chapter4/README.md`
- `code/tests/chapter4/*.py`
- `code/results/task_12/yolov12_baseline/` (полный набор)
```
