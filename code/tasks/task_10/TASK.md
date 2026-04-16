# Task 10: Обучение DETR на 4 вариантах датасета

## Статус
pending

## Цель

Обучить трансформерный детектор DETR (DEtection TRansformer) на четырёх последовательных вариантах подготовки данных (baseline → aug_geom → aug_oversample → aug_diffusion). DETR в главе 3 — представитель классического трансформерного подхода к детекции: отсутствие anchor-free/NMS, обработка как set prediction. Сопоставление с RT-DETR (модифицированный DETR, оптимизированный под real-time) показывает, какой прирост дают доработки real-time-варианта по сравнению с базовым трансформером.

## Общий протокол

Все гиперпараметры, пути к датасетам, формат артефактов и структура ноутбука соответствуют `code/docs/chapter3_protocol.md`. Специфика — см. ниже.

## Шаги

1. **Создать ноутбук** `code/notebooks/chapter3_detr.ipynb` по структуре §5 протокола.
2. **Использовать ту же качественную выборку** из `code/docs/chapter3_qualitative_sample.txt`.
3. **Подготовить DataLoader** в формате, совместимом с HuggingFace `DetrImageProcessor` / transformers: `(pixel_values, labels)`, где `labels = {"class_labels": Tensor[N], "boxes": Tensor[N, 4] в формате (cx, cy, w, h) нормализованных}`. Именно такой формат ожидает `DetrForObjectDetection.forward(...)`.
4. **Вариант baseline** — обучить на `code/data/dataset/`, артефакты → `code/results/task_10/detr_baseline/`.
5. **Вариант aug_geom** — `code/data/dataset_augmented/`, артефакты → `detr_aug_geom/`.
6. **Вариант aug_oversample** — `code/data/dataset_balanced/`, артефакты → `detr_aug_oversample/`.
7. **Вариант aug_diffusion** — `code/data/dataset_final/`, артефакты → `detr_aug_diffusion/`.
8. **Сводка** — таблица 4 × 7 → `code/results/task_10/summary.csv`.

После каждого варианта — git-коммит `chapter3_detr: <variant> done` + `git push`.

## Специфика DETR

- **Фреймворк**: HuggingFace `transformers` — `DetrForObjectDetection`, `DetrImageProcessor`.
- **Backbone**: ResNet-50, чекпойнт `facebook/detr-resnet-50`.
- **Адаптация головы**: через `AutoConfig` переопределить `num_labels=9`, `id2label`, `label2id`; использовать `ignore_mismatched_sizes=True` при `from_pretrained(...)` чтобы переинициализировать классификационный слой под 9 классов.
- **batch**: 4 (при OOM снизить до 2).
- **Optimizer**: AdamW, lr=1e-4 для всех параметров, кроме backbone — у backbone lr=1e-5.
- **Weight decay**: 1e-4.
- **Scheduler**: фиксированный lr на весь цикл (оригинальная статья DETR использует константный lr со скачком в конце, но в рамках 100 эпох с patience=15 этого избегаем).
- **Лосс**: Hungarian matching + классификация + bbox + GIoU — считается внутри модели (`outputs.loss`).
- **Метрики**: как в task_09 — через `torchmetrics.MeanAveragePrecision`, после denormalization предсказанных bbox.
- **Особенность сходимости**: DETR знаменит медленной сходимостью. Оригинальная статья использует 150–300 эпох. В нашем протоколе 100 эпох с patience=15 — компромисс; если модель на 100 эпохах явно не сошлась, зафиксировать это в `RESULT.md`. Не продлевать без явной договорённости — задача главы 3 состоит в сравнении детекторов в одинаковых условиях, а не в выжимании максимума из каждой архитектуры.
- **Встроенные аугментации HuggingFace processor**: отключить (никаких RandomHorizontalFlip, ColorJitter и т.д.) — оставить только resize и нормализацию.

### Скелет запуска

```python
from transformers import DetrForObjectDetection, DetrImageProcessor, AutoConfig

id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
label2id = {v: k for k, v in id2label.items()}

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
config = AutoConfig.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=9,
    id2label=id2label,
    label2id=label2id,
)
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    config=config,
    ignore_mismatched_sizes=True,
).to("cuda")

param_dicts = [
    {"params": [p for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad], "lr": 1e-4},
    {"params": [p for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad], "lr": 1e-5},
]
optimizer = torch.optim.AdamW(param_dicts, weight_decay=1e-4)
```

## Входные данные

- `code/data/dataset/`
- `code/data/dataset_augmented/`
- `code/data/dataset_balanced/`
- `code/data/dataset_final/`

## Ожидаемый результат

- Ноутбук `code/notebooks/chapter3_detr.ipynb` с выводами 4 прогонов.
- `code/results/task_10/` с 4 поддиректориями `detr_<variant>/` и `summary.csv`.
- Заполненный `RESULT.md`.

### Ориентиры

- mAP@50 на baseline — порядка 0.55–0.68 (DETR обычно уступает YOLO и RT-DETR при ограниченном бюджете эпох).
- FPS — самый низкий из четырёх детекторов (ожидается 8–18 FPS на batch=1, imgsz=640).
- Большой прирост от aug_diffusion возможен (DETR особенно чувствителен к объёму и разнообразию обучающей выборки из-за set-prediction лосса).

## Результат записать в

`code/tasks/task_10/RESULT.md` по шаблону:

```markdown
# Result: Task 10 — Обучение DETR на 4 вариантах датасета

## Статус
done | partial | failed

## Что было сделано
Краткое описание пайплайна, использованный чекпойнт, конфигурация оптимизатора, фактический batch.

## Результаты

| Вариант | n_train | mAP@50 | mAP@50-95 | Precision | Recall | FPS | Эпох |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | | | | | | | |
| aug_geom | | | | | | | |
| aug_oversample | | | | | | | |
| aug_diffusion | | | | | | | |

### Per-class mAP@50 на лучшем варианте

## Проблемы / Замечания
- Факт сходимости за 100 эпох (сошлась / не сошлась)
- Фактический batch
- Любые отклонения от протокола

## Артефакты
- `code/notebooks/chapter3_detr.ipynb`
- `code/results/task_10/detr_baseline/`
- `code/results/task_10/detr_aug_geom/`
- `code/results/task_10/detr_aug_oversample/`
- `code/results/task_10/detr_aug_diffusion/`
- `code/results/task_10/summary.csv`
```
