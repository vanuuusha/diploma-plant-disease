# Task 07: Обучение YOLOv12 на 4 вариантах датасета

## Статус
pending

## Цель

Обучить детектор YOLOv12 на четырёх последовательных вариантах подготовки данных (baseline → aug_geom → aug_oversample → aug_diffusion) и замерить вклад каждой стадии аугментации в качество детекции. YOLOv12 в этой серии — приоритетный детектор: по его результатам будет выбран базовый вариант датасета для главы 4 (CGFM).

## Общий протокол

Все гиперпараметры, пути к датасетам, формат артефактов и структура ноутбука строго соответствуют документу `code/docs/chapter3_protocol.md`. Не отклоняться от него без явной договорённости. Ниже — только уточнения, специфичные для YOLOv12.

## Шаги

1. **Создать ноутбук** `code/notebooks/chapter3_yolov12.ipynb` по структуре из §5 протокола (7 блоков: импорты, утилиты, 4 варианта, сводка).
2. **Подготовить качественную выборку.** Если ещё не сделано в рамках главы 3 — отобрать 10 тестовых изображений (разной сложности), записать их имена в `code/docs/chapter3_qualitative_sample.txt`. Та же выборка будет использоваться в task_08–task_10.
3. **Вариант baseline** — обучить YOLOv12-m на `code/data/dataset/data.yaml`, сохранить артефакты в `code/results/task_07/yolov12_baseline/`.
4. **Вариант aug_geom** — обучить на `code/data/dataset_augmented/data.yaml`, артефакты → `yolov12_aug_geom/`.
5. **Вариант aug_oversample** — обучить на `code/data/dataset_balanced/data.yaml`, артефакты → `yolov12_aug_oversample/`.
6. **Вариант aug_diffusion** — обучить на `code/data/dataset_final/data.yaml`, артефакты → `yolov12_aug_diffusion/`.
7. **Сводка** — финальная ячейка ноутбука строит таблицу 4 × 7 по шаблону из §6 протокола и сохраняет в `code/results/task_07/summary.csv`.

После каждого варианта — git-коммит по шаблону `chapter3_yolov12: <variant> done`, затем `git push`.

## Специфика YOLOv12

- **Фреймворк**: Ultralytics (`ultralytics>=8.3.0`).
- **Модель**: `yolo12m.pt` (medium). Если недоступна — использовать `yolo12s.pt` и зафиксировать это в `RESULT.md` с обоснованием.
- **batch**: 16 (скорректировать при OOM, записать фактический в `RESULT.md`).
- **Встроенные аугментации Ultralytics**: отключить полностью (см. §3.2 протокола).
- **Инициализация**: веса COCO (`pretrained=True` — значение по умолчанию Ultralytics).
- **Скелет запуска**:

```python
from ultralytics import YOLO

model = YOLO("yolo12m.pt")
results = model.train(
    data="/home/vanusha/diplom/diploma-plant-disease/code/data/dataset/data.yaml",
    epochs=100,
    patience=15,
    imgsz=640,
    batch=16,
    seed=42,
    device=0,
    workers=8,
    project="code/results/task_07",
    name="yolov12_baseline",
    exist_ok=True,
    # аугментации Ultralytics полностью отключены
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
    verbose=True,
)
```

- Для confusion matrix, per-class mAP и test-метрик использовать `model.val(data=..., split="test")`.
- Для FPS-замера — ручной цикл по §4.5 протокола (не использовать встроенный `speed` Ultralytics, так как там другой режим).

## Входные данные

- `code/data/dataset/data.yaml`
- `code/data/dataset_augmented/data.yaml`
- `code/data/dataset_balanced/data.yaml`
- `code/data/dataset_final/data.yaml`
- Предобученные веса YOLOv12 (скачиваются автоматически).

## Ожидаемый результат

- Ноутбук `code/notebooks/chapter3_yolov12.ipynb` с сохранёнными выводами всех 4 прогонов.
- `code/results/task_07/` с 4 поддиректориями (`yolov12_<variant>/`) и `summary.csv`.
- Заполненный `RESULT.md` по шаблону ниже.
- Все артефакты, кроме весов, закоммичены в git.

### Ожидаемое качественное поведение

Это ориентиры для быстрой проверки адекватности результатов (не требования):
- mAP@50 на baseline — порядка 0.72–0.80 (исходя из итогов предшествующих экспериментов по детекции болезней пшеницы).
- aug_geom должен давать умеренный прирост (+1–3 п.п. mAP@50) либо нейтральный эффект.
- aug_oversample — прирост преимущественно на Recall редких классов (Недостаток N, Повреждение заморозками, Септориоз).
- aug_diffusion — прирост на mAP@50-95 и/или на сложных классах (Фузариоз, Корневая гниль), но возможно падение Precision.

Если результаты сильно отличаются (например, aug_geom даёт −10 п.п.) — зафиксировать в `RESULT.md` и не пытаться «починить» настройки: задача — оценить эффект стадий как они есть.

## Результат записать в

`code/tasks/task_07/RESULT.md` по следующему шаблону:

```markdown
# Result: Task 07 — Обучение YOLOv12 на 4 вариантах датасета

## Статус
done | partial | failed

## Что было сделано
Краткое описание пайплайна, использованная модель (yolo12m / yolo12s), фактический batch.

## Результаты

| Вариант | n_train | mAP@50 | mAP@50-95 | Precision | Recall | FPS | Эпох |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | | | | | | | |
| aug_geom | | | | | | | |
| aug_oversample | | | | | | | |
| aug_diffusion | | | | | | | |

### Per-class mAP@50 на лучшем варианте
(таблица по 9 классам — из per_class_map.csv)

## Проблемы / Замечания
- Фактический batch, если отличался от плана
- Любые отклонения от протокола

## Артефакты
- `code/notebooks/chapter3_yolov12.ipynb`
- `code/results/task_07/yolov12_baseline/`
- `code/results/task_07/yolov12_aug_geom/`
- `code/results/task_07/yolov12_aug_oversample/`
- `code/results/task_07/yolov12_aug_diffusion/`
- `code/results/task_07/summary.csv`
```
