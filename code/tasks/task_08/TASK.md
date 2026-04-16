# Task 08: Обучение RT-DETR на 4 вариантах датасета

## Статус
pending

## Цель

Обучить детектор RT-DETR на четырёх последовательных вариантах подготовки данных (baseline → aug_geom → aug_oversample → aug_diffusion). RT-DETR — главный конкурент YOLOv12 по сочетанию скорости и точности; в главе 3 сопоставление этих двух архитектур критично для аргументации выбора baseline-архитектуры для главы 4.

## Общий протокол

Все гиперпараметры, пути к датасетам, формат артефактов и структура ноутбука строго соответствуют документу `code/docs/chapter3_protocol.md`. Ниже — только специфика RT-DETR.

## Шаги

1. **Создать ноутбук** `code/notebooks/chapter3_rtdetr.ipynb` по структуре §5 протокола.
2. **Использовать ту же качественную выборку** из `code/docs/chapter3_qualitative_sample.txt`, что и task_07 — это гарантирует прямое визуальное сопоставление предсказаний разных детекторов на одних и тех же изображениях.
3. **Вариант baseline** — обучить на `code/data/dataset/data.yaml`, артефакты → `code/results/task_08/rtdetr_baseline/`.
4. **Вариант aug_geom** — `code/data/dataset_augmented/data.yaml`, артефакты → `rtdetr_aug_geom/`.
5. **Вариант aug_oversample** — `code/data/dataset_balanced/data.yaml`, артефакты → `rtdetr_aug_oversample/`.
6. **Вариант aug_diffusion** — `code/data/dataset_final/data.yaml`, артефакты → `rtdetr_aug_diffusion/`.
7. **Сводка** — таблица 4 × 7 в финальной ячейке → `code/results/task_08/summary.csv`.

После каждого варианта — git-коммит `chapter3_rtdetr: <variant> done` + `git push`.

## Специфика RT-DETR

- **Фреймворк**: Ultralytics (у RT-DETR интерфейс совместим с YOLO-классом).
- **Модель**: `rtdetr-l.pt` (large). Если памяти недостаточно — `rtdetr-l.pt` остаётся приоритетным; уменьшить batch перед заменой на `rtdetr-x`. Фактический выбор зафиксировать в `RESULT.md`.
- **batch**: 8 (при OOM снизить до 4).
- **Встроенные аугментации Ultralytics**: отключить полностью (см. §3.2 протокола).
- **Предупреждение о сходимости**: DETR-подобные модели, включая RT-DETR, сходятся медленнее one-stage YOLO. Лимит 100 эпох с patience=15 оставить, но если на 100 эпохах обучение ещё не стабилизировалось — отметить это в `RESULT.md` (возможно, потребуется дообучение в рамках будущей итерации).
- **Скелет запуска**:

```python
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")
results = model.train(
    data="/home/vanusha/diplom/diploma-plant-disease/code/data/dataset/data.yaml",
    epochs=100,
    patience=15,
    imgsz=640,
    batch=8,
    seed=42,
    device=0,
    workers=8,
    project="code/results/task_08",
    name="rtdetr_baseline",
    exist_ok=True,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
    verbose=True,
)
```

- Confusion matrix, per-class mAP и тестовые метрики — через `model.val(data=..., split="test")`.
- FPS-замер — по §4.5 протокола (ручной цикл, fp32, batch=1).

## Входные данные

- `code/data/dataset/data.yaml`
- `code/data/dataset_augmented/data.yaml`
- `code/data/dataset_balanced/data.yaml`
- `code/data/dataset_final/data.yaml`
- Предобученные веса RT-DETR (скачиваются автоматически Ultralytics).

## Ожидаемый результат

- Ноутбук `code/notebooks/chapter3_rtdetr.ipynb` с сохранёнными выводами 4 прогонов.
- `code/results/task_08/` с 4 поддиректориями `rtdetr_<variant>/` и `summary.csv`.
- Заполненный `RESULT.md` по шаблону из §«Результат записать в».

### Ориентиры

- mAP@50 на baseline — порядка 0.70–0.78 (близко к YOLOv12, RT-DETR обычно чуть уступает по скорости, выигрывает по сходимости на сложных сценах).
- FPS — заметно ниже YOLOv12 (ожидается 25–45 FPS против 50–90 у YOLOv12).
- Поведение по стадиям аугментации — аналогично task_07: aug_geom даёт умеренный прирост, aug_oversample — прирост на редких классах, aug_diffusion — на сложных.

## Результат записать в

`code/tasks/task_08/RESULT.md` по шаблону:

```markdown
# Result: Task 08 — Обучение RT-DETR на 4 вариантах датасета

## Статус
done | partial | failed

## Что было сделано
Краткое описание пайплайна, использованная модель (rtdetr-l / rtdetr-x), фактический batch.

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
- Замечания по сходимости, если модель не стабилизировалась за 100 эпох
- Любые отклонения от протокола

## Артефакты
- `code/notebooks/chapter3_rtdetr.ipynb`
- `code/results/task_08/rtdetr_baseline/`
- `code/results/task_08/rtdetr_aug_geom/`
- `code/results/task_08/rtdetr_aug_oversample/`
- `code/results/task_08/rtdetr_aug_diffusion/`
- `code/results/task_08/summary.csv`
```
