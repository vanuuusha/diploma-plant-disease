# Task 09: Обучение Faster R-CNN на 4 вариантах датасета

## Статус
pending

## Цель

Обучить классический two-stage детектор Faster R-CNN (ResNet-50 FPN) на четырёх последовательных вариантах подготовки данных (baseline → aug_geom → aug_oversample → aug_diffusion). Faster R-CNN в главе 3 играет роль «справочной архитектуры»: двухэтапный подход с RPN известен стабильностью и даёт верхнюю границу точности при низкой скорости инференса. Сопоставление с YOLOv12 и RT-DETR показывает, насколько современные одностадийные модели сократили разрыв по точности, сохранив преимущество по скорости.

## Общий протокол

Все гиперпараметры, пути к датасетам, формат артефактов и структура ноутбука соответствуют `code/docs/chapter3_protocol.md`. Отличие от task_07/task_08: фреймворк — torchvision, поэтому формат разметки YOLO-txt нужно конвертировать в COCO-JSON «на лету» (внутри DataLoader) либо заранее.

## Шаги

1. **Создать ноутбук** `code/notebooks/chapter3_faster_rcnn.ipynb` по структуре §5 протокола.
2. **Использовать ту же качественную выборку** из `code/docs/chapter3_qualitative_sample.txt`.
3. **Подготовить DataLoader**, который принимает YOLO-разметку (`.txt`) и возвращает `(image, target)` в формате torchvision (`target = {"boxes": Tensor[N, 4] (x1, y1, x2, y2, абсолютные), "labels": Tensor[N]}`). Класс `0` в torchvision зарезервирован под background, поэтому id классов YOLO `0..8` отображаются в torchvision `1..9`.
4. **Вариант baseline** — обучить на `code/data/dataset/`, артефакты → `code/results/task_09/faster_rcnn_baseline/`.
5. **Вариант aug_geom** — `code/data/dataset_augmented/`, артефакты → `faster_rcnn_aug_geom/`.
6. **Вариант aug_oversample** — `code/data/dataset_balanced/`, артефакты → `faster_rcnn_aug_oversample/`.
7. **Вариант aug_diffusion** — `code/data/dataset_final/`, артефакты → `faster_rcnn_aug_diffusion/`.
8. **Сводка** — таблица 4 × 7 → `code/results/task_09/summary.csv`.

После каждого варианта — git-коммит `chapter3_faster_rcnn: <variant> done` + `git push`.

## Специфика Faster R-CNN

- **Фреймворк**: `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`.
- **Предобучение**: веса COCO (`weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`).
- **Адаптация головы**: заменить `box_predictor` на новый `FastRCNNPredictor(in_features, num_classes=10)` (9 классов + background).
- **batch**: 4 (при OOM снизить до 2).
- **Optimizer**: SGD, lr=0.005, momentum=0.9, weight_decay=0.0005 (стандарт torchvision).
- **Scheduler**: `StepLR(step_size=3, gamma=0.1)` или `MultiStepLR(milestones=[60, 80], gamma=0.1)`.
- **Лосс**: комбинированный (classification + box regression + RPN) — считается автоматически.
- **Аугментации DataLoader**: только resize (имитирующий 640×640 с сохранением пропорций + padding) и ToTensor. Никаких случайных флипов/кропов — все стадии аугментации уже внесены в `dataset_*` директории.
- **Метрики**: считать mAP@50 и mAP@50-95 через `torchmetrics.detection.MeanAveragePrecision` или через экспорт в COCO-формат + pycocotools. Для единообразия с task_07/task_08 лучше torchmetrics.
- **Early stopping**: патерн patience=15 реализовать вручную (torchvision не даёт коробочного ES) — отслеживать val mAP@50, сохранять `best.pt` при улучшении, прерывать цикл после 15 эпох без улучшения.

### Скелет запуска

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection import MeanAveragePrecision

NUM_CLASSES = 10  # 9 + background
weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn_v2(weights=weights)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.to("cuda")

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
```

## Входные данные

- `code/data/dataset/` (+ `data.yaml`)
- `code/data/dataset_augmented/`
- `code/data/dataset_balanced/`
- `code/data/dataset_final/`

## Ожидаемый результат

- Ноутбук `code/notebooks/chapter3_faster_rcnn.ipynb` с выводами 4 прогонов.
- `code/results/task_09/` с 4 поддиректориями `faster_rcnn_<variant>/` и `summary.csv`.
- Заполненный `RESULT.md`.

### Ориентиры

- mAP@50 на baseline — порядка 0.60–0.72 (Faster R-CNN обычно чуть уступает современным YOLO на малых датасетах).
- FPS — заметно ниже всех остальных детекторов (ожидается 6–15 FPS на RTX 5070 Ti при batch=1, imgsz=640).
- Характер прироста по стадиям: та же тенденция, что в task_07 — aug_geom умеренный прирост, aug_oversample — на редких, aug_diffusion — на сложных классах.

## Результат записать в

`code/tasks/task_09/RESULT.md` по шаблону:

```markdown
# Result: Task 09 — Обучение Faster R-CNN на 4 вариантах датасета

## Статус
done | partial | failed

## Что было сделано
Краткое описание пайплайна, конфигурация оптимизатора/scheduler, фактический batch.

## Результаты

| Вариант | n_train | mAP@50 | mAP@50-95 | Precision | Recall | FPS | Эпох |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | | | | | | | |
| aug_geom | | | | | | | |
| aug_oversample | | | | | | | |
| aug_diffusion | | | | | | | |

### Per-class mAP@50 на лучшем варианте

## Проблемы / Замечания
- Фактический batch
- Специфика torchvision-обёртки (реализация early stopping, использованные метрики)
- Любые отклонения от протокола

## Артефакты
- `code/notebooks/chapter3_faster_rcnn.ipynb`
- `code/results/task_09/faster_rcnn_baseline/`
- `code/results/task_09/faster_rcnn_aug_geom/`
- `code/results/task_09/faster_rcnn_aug_oversample/`
- `code/results/task_09/faster_rcnn_aug_diffusion/`
- `code/results/task_09/summary.csv`
```
