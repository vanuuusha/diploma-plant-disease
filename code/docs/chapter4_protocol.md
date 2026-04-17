# Протокол экспериментов главы 4: контекстная модуляция признаков (CGFM)

Документ описывает единый экспериментальный протокол для задач `task_12`–`task_18` главы 4. Все TASK.md этой группы ссылаются сюда, чтобы исключить расхождения в гиперпараметрах, наборе данных, метриках и формате артефактов между конфигурациями. Методология метрик и формат артефактов наследуются от `code/docs/chapter3_protocol.md` (далее — «протокол гл. 3») — в явных отклонениях указано иное.

---

## 1. Цель блока задач

Количественно оценить, насколько предложенный модуль контекстной модуляции признаков **CGFM** (Context-Gated Feature Modulation, описан в `диплом/artifacts/ART_001_CGFM.md`) превосходит существующие альтернативы — самореферентные механизмы внимания (SE-Net, CBAM) и позднее слияние контекста (Late Fusion) — при задаче детекции болезней пшеницы.

Основная серия — на YOLOv12; дополнительная серия — на RT-DETR для проверки переносимости подхода на трансформерные архитектуры с гибридным энкодером. В обоих случаях сравнение ведётся относительно одной и той же лучшей конфигурации данных (`aug_diffusion`, `dataset_final/`), обоснованной в главе 3.

Серия состоит из **шести базовых конфигураций на YOLOv12 + аблация CGFM + две конфигурации на RT-DETR**:

| # | Конфигурация | Что проверяем | Задача |
|---|--------------|---------------|--------|
| 1 | YOLOv12 Baseline | Нижняя граница | Уже зафиксирована в `task_07/yolov12_aug_diffusion` |
| 2 | YOLOv12 + SE-Neck | Самореферентная канальная самоатенция | `task_13` |
| 3 | YOLOv12 + CBAM-Neck | Канальная + пространственная самоатенция | `task_13` |
| 4 | YOLOv12 + Late Fusion | Контекст подключается после детекции | `task_14` |
| 5 | YOLOv12 + CGFM (основной) | FiLM от внешнего энкодера в neck | `task_15` |
| 6 | YOLOv12 + CGFM + Late | Максимальный эффект (контекст на двух этапах) | `task_15` |
| — | CGFM-аблация: уровни × энкодер | Выбор лучшей конфигурации | `task_16` |
| 7 | RT-DETR Baseline | Из `task_08/rtdetr_aug_diffusion` | `task_17` (referенс) |
| 8 | RT-DETR + CGFM | Переносимость на гибридный энкодер | `task_17` |

Финальная сводка, статистические тесты и визуализации — `task_18`.

---

## 2. Исходные данные — единственный вариант

В отличие от главы 3 (4 варианта датасета), глава 4 использует **только `aug_diffusion`** — лучший вариант по результатам главы 3. Это оправдано тем, что цель главы 4 — не оценка данных, а оценка архитектурной модификации при фиксированных данных.

| Параметр | Значение |
|---|---|
| Директория данных | `code/data/dataset_final/` |
| `data.yaml` | `code/data/dataset_final/data.yaml` |
| Формат | YOLO (images + labels), 9 классов |
| Split | train / val / test — те же, что в главе 3 |

Классы (id 0–8) и splits полностью соответствуют протоколу гл. 3, §2.

---

## 3. Унифицированные гиперпараметры

Все запуски главы 4 используют **идентичные** гиперпараметры, совпадающие с `task_07/yolov12_aug_diffusion` (или `task_08/rtdetr_aug_diffusion` для RT-DETR-ветви). Это гарантирует, что разница в метриках объясняется исключительно архитектурной модификацией.

| Параметр | YOLOv12 | RT-DETR | Комментарий |
|---|---|---|---|
| `epochs` | 100 | 100 | Максимум |
| `patience` | 15 | 15 | Early stopping |
| `imgsz` | 640 | 640 | |
| `batch` | 16 | 8 | Как в гл. 3; снизить при OOM |
| `optimizer` | default Ultralytics (AdamW) | AdamW, lr=1e-4 (head) / 1e-5 (backbone) | |
| `seed` | 42 | 42 | Фиксированный |
| `device` | `cuda:0` | `cuda:0` | RTX 5070 Ti |
| `pretrained` | True (COCO) | True (COCO) | |
| Встроенные аугментации | **отключены** (см. §3.2 протокола гл. 3) | отключены | |

### 3.1 Бюджет обучения для новых ветвей

Контекстный энкодер и FiLM-параметры инициализируются случайно (кроме предобученных весов самого энкодера). Для стабилизации:

- **Warm-up контекстного энкодера**: первые 3 эпохи обучать только γ, β и контекстный энкодер (основной YOLOv12 заморожен), затем полный end-to-end. Дополнительный lr множитель для новых параметров (`lr_new = 5 × lr_base`) в течение первых 10 эпох.
- **Late fusion-ветвь**: обучать классификатор контекста после завершения основного обучения детектора (этап 2), на предсказанных боксах — см. §4.4.

### 3.2 Фиксация baseline

YOLOv12-baseline повторно не обучается — используются результаты `task_07/yolov12_aug_diffusion`:
- веса `best.pt` загружаются из этой директории
- метрики (mAP@50, mAP@50-95, Precision, Recall, FPS) копируются в таблицу главы 4 как строка «Baseline»
- confusion matrix, per-class mAP, predictions_examples — переиспользуются напрямую

Аналогично RT-DETR-baseline — из `task_08/rtdetr_aug_diffusion`.

---

## 4. Реализация модулей

Все модули главы 4 размещаются в `code/models/chapter4/`. Каждый модуль — отдельный `.py` файл, покрыт unit-тестом в `code/tests/chapter4/`.

### 4.1 FiLM-слой (основа CGFM)

```python
# code/models/chapter4/film_layer.py
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (Perez et al., 2018).
    F'_l = γ_l ⊙ F_l + β_l
    """
    def __init__(self, context_dim: int, feature_channels: int):
        super().__init__()
        self.to_gamma = nn.Linear(context_dim, feature_channels)
        self.to_beta  = nn.Linear(context_dim, feature_channels)

    def forward(self, features, context):
        gamma = torch.sigmoid(self.to_gamma(context))      # [B, C], ∈ [0, 1]
        beta  = self.to_beta(context)                      # [B, C]
        # broadcast: [B, C, 1, 1] на карту [B, C, H, W]
        return gamma[..., None, None] * features + beta[..., None, None]
```

### 4.2 Context Encoder

```python
# code/models/chapter4/context_encoder.py
class ContextEncoder(nn.Module):
    """
    Отдельный лёгкий энкодер всей сцены → d-мерный вектор c.
    Backbone варианты (выбираются флагом):
        'mobilenetv3_small' — ≈ 1.5M, d=576 → projection → 256
        'efficientnet_b0'   — ≈ 5.3M, d=1280 → projection → 256
        'vit_tiny_patch16'  — ≈ 5.7M, d=192 → projection → 256
    """
    def __init__(self, backbone: str = 'mobilenetv3_small',
                 out_dim: int = 256, pretrained: bool = True): ...
    def forward(self, x) -> torch.Tensor: ...  # [B, out_dim]
```

Реализуется через `timm` (присутствует в `requirements.txt`).

### 4.3 Конкуренты — SE-блок и CBAM

```python
# code/models/chapter4/se_block.py
class SEBlock(nn.Module):
    """SE-Net (Hu et al., 2018). Самореферентная канальная самоатенция."""
    def __init__(self, channels: int, reduction: int = 16): ...

# code/models/chapter4/cbam_block.py
class CBAMBlock(nn.Module):
    """CBAM (Woo et al., 2018). Channel + Spatial самоатенция."""
    def __init__(self, channels: int, reduction: int = 16,
                 spatial_kernel: int = 7): ...
```

Реализации следуют референсным статьям; использовать существующие открытые реализации (например, `timm`) недопустимо — требуется контроль над местом вставки в neck.

### 4.4 Late Fusion-классификатор

```python
# code/models/chapter4/late_fusion_head.py
class LateFusionClassifier(nn.Module):
    """
    На вход: ROI-фича из боксов YOLOv12 + контекстный вектор c.
    Переклассифицирует класс каждого предсказанного bbox с учётом контекста.
    """
    def __init__(self, roi_dim: int, context_dim: int,
                 num_classes: int = 9): ...
```

Обучается этапом 2 — на предсказаниях замороженного YOLOv12-baseline (таким образом, изменяется только класс, но не сам bbox).

### 4.5 CGFM-интеграция в neck YOLOv12

Модификация происходит на выходах P3, P4, P5 neck PAN. В `Ultralytics` neck реализован через `C2f` и `Concat`; нужно добавить FiLM-модуляцию **после** последнего свёрточного блока каждого уровня, **до** Detect-головы.

Конкретное место вставки и алгоритм подмены — см. `task_12`.

---

## 5. Единый лог-контракт: что сохранять на каждом запуске

Каждый прогон обучения производит фиксированный набор артефактов. Пути — относительно корня проекта. Формат артефактов полностью совпадает с §4 протокола гл. 3 (`metrics.csv`, `learning_curves.png`, `confusion_matrix.png`, `per_class_map.csv`, `predictions_examples/`, `fps_measurement.json`, `best.pt`, `train.log`).

Директория: `code/results/task_NN/<config_name>/`, где `<config_name>` из §6.

Дополнительно для CGFM-конфигураций:
- `gamma_maps/` — 6–10 карт значений γ (тепловые карты по каналам) на тех же тестовых файлах из `code/docs/chapter3_qualitative_sample.txt`
- `context_embeddings.npy` — shape `[N_test, d]`, все контекстные эмбеддинги на test для последующего t-SNE
- `param_count.json` — количество параметров и MACs для сравнения сложности

### 5.1 Качественная выборка

Используется тот же `code/docs/chapter3_qualitative_sample.txt` (10 тестовых изображений), что и в гл. 3. Это делает качественную галерею сквозной для глав 3 и 4.

---

## 6. Имена конфигураций (`<config_name>`)

Строгие имена директорий (snake_case):

| Код | Конфигурация |
|---|---|
| `yolov12_baseline` | Baseline (симлинк на `task_07/yolov12_aug_diffusion`) |
| `yolov12_se_neck` | YOLOv12 + SE в neck (task_13) |
| `yolov12_cbam_neck` | YOLOv12 + CBAM в neck (task_13) |
| `yolov12_late_fusion` | YOLOv12 + Late Fusion (task_14) |
| `yolov12_cgfm` | YOLOv12 + CGFM — основной метод (task_15) |
| `yolov12_cgfm_late` | CGFM + Late Fusion (task_15) |
| `yolov12_cgfm_abl_p5only` | Аблация: CGFM только на P5 (task_16) |
| `yolov12_cgfm_abl_p3only` | Аблация: CGFM только на P3 (task_16) |
| `yolov12_cgfm_abl_mobilenet` | Аблация: MobileNetV3-Small как энкодер (task_16) |
| `yolov12_cgfm_abl_effb0` | Аблация: EfficientNet-B0 как энкодер (task_16) |
| `yolov12_cgfm_abl_vittiny` | Аблация: ViT-Tiny как энкодер (task_16) |
| `rtdetr_baseline` | Baseline (симлинк на `task_08/rtdetr_aug_diffusion`) |
| `rtdetr_cgfm` | RT-DETR + CGFM на выходах гибридного энкодера (task_17) |

Аблация по уровням фиксирует лучший энкодер из аблации по энкодерам (и наоборот), чтобы избежать факториального взрыва.

---

## 7. Протокол замера FPS

Полностью наследуется из §4.5 протокола гл. 3 (100 прогонов, warm-up 20, `batch=1`, `imgsz=640`, `fp32`). Дополнительно для CGFM-конфигураций фиксируется отдельный замер латентности **контекстного энкодера** как доли общего времени инференса:

```json
{
  "detector_fps": 54.3,
  "context_encoder_ms": 2.1,
  "film_layers_ms": 0.2,
  "total_overhead_percent": 12.7
}
```

---

## 8. Статистические тесты (в task_18)

Для честного сравнения ключевых конфигураций:
1. **Bootstrap 95 % доверительных интервалов** mAP@50 (как в task_11): для каждой конфигурации — 1000 прогонов на случайном подмножестве test (80 %), построение распределения mAP, вычисление CI.
2. **Permutation test** разницы mAP@50 между CGFM и лучшим конкурентом (SE / CBAM / Late): переставить метки «моя модель / конкурент» на уровне изображений и пересчитать разницу. p-value как доля перестановок, в которых разница превзошла наблюдаемую.

---

## 9. Чек-лист перед пушем в git

Для каждого эксперимента:
- [ ] Веса `best.pt` **не** в git (в `.gitignore`)
- [ ] `metrics.csv`, `learning_curves.png`, `confusion_matrix.png`, `per_class_map.csv`, `predictions_examples/`, `fps_measurement.json` — присутствуют
- [ ] Для CGFM-конфигураций — дополнительно `gamma_maps/`, `context_embeddings.npy`, `param_count.json`
- [ ] `RESULT.md` соответствующей задачи заполнен по шаблону
- [ ] Коммит `chapter4_<config>: done` + `git push`

---

## 10. Порядок выполнения задач

```
task_12 (инфраструктура) ← подготовка модулей, интеграционные тесты
        ↓
task_13 (SE + CBAM)      ← 2 самореферентных бейзлайна
        ↓
task_14 (Late Fusion)    ← бейзлайн позднего слияния
        ↓
task_15 (CGFM + CGFM+Late) ← основной метод + максимальная конфигурация
        ↓
task_16 (аблация CGFM)   ← уровни × энкодер, выбор оптимума
        ↓
task_17 (RT-DETR + CGFM) ← переносимость, 1–2 конфигурации
        ↓
task_18 (сводка)         ← таблицы, γ-визуализация, t-SNE, FP/FN, стат-тесты
```

Порядок задач 13 → 14 → 15 может исполняться параллельно при наличии второй GPU: все они обучают YOLOv12 на одном и том же датасете с разными модификациями neck. task_16 зависит от task_15 (нужен рабочий CGFM). task_17 не зависит ни от чего, кроме task_12.
