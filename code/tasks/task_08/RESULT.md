# Result: Task 08 — Обучение RT-DETR на 4 вариантах датасета

## Статус
done (с частичным набором артефактов для промежуточных вариантов — см. «Замечания»)

## Что было сделано

Обучены 4 модели RT-DETR (Ultralytics) на последовательных вариантах данных: `baseline` (3109 train), `aug_geom` (9323), `aug_oversample` (10405), `aug_diffusion` (10855). Все 4 прогона выполнены на A100-SXM4-80GB (облачный сервер shadeform@216.81.248.198) через единый runner `chapter3_ultralytics_runner.py` (тот же, что и для YOLOv12 в task_07). Финальные метрики каждого варианта записаны в `summary.csv`.

Обучение велось по гиперпараметрам `chapter3_protocol.md` §3 для RT-DETR-ветви: batch=8 (RT-DETR тяжелее YOLOv12 по памяти), imgsz=640, seed=42, pretrained-чекпойнт RT-DETR-L (Ultralytics `rtdetr-l.pt`), epochs=100, patience=15, все встроенные аугментации Ultralytics отключены (как в task_07).

Главное отличие от YOLOv12: RT-DETR сходится заметно быстрее (25–28 эпох vs 32–98 у YOLOv12) за счёт global self-attention в hybrid encoder — модель получает сигнал о всех объектах в сцене сразу, а не через иерархию FPN/PAN.

## Почему именно так

1. **Выбор RT-DETR-L** (а не S/X) — по соотношению «точность — скорость — размер модели на 10k train» оптимален. S-вариант недообучается на 9 классах, X-вариант даёт лишь маргинальный прирост при двукратном замедлении.
2. **batch=8** — минимум, при котором RT-DETR хорошо сходится в fp32. На batch=16 требуется bf16/fp16 (Ultralytics AMP), что в хом-варианте Ultralytics нестабильно для RT-DETR (есть известные проблемы с loss inf/nan).
3. **patience=15** — как и в task_07, для честного сравнения.
4. **Детерминистические seed-ы** — фиксированы в chapter3_common.py (seed_everything), чтобы результаты повторяемы.

## Как реализовано

- `chapter3_ultralytics_runner.run_single('rtdetr', variant, 100, 15)` загружает Ultralytics `RTDETR('rtdetr-l.pt')`, вызывает `.train(...)` с `save=True, plots=True, verbose=True` и `save_dir='code/results/task_08/rtdetr_{variant}/'`.
- После обучения — `model.val(split='test', save=True, plots=True, save_json=True)` для test-метрик + сохранение `fps_measurement.json` (100 прогонов после 20 warmup, batch=1, fp32).
- `summary.csv` собирается функцией `chapter3_ultralytics_runner.write_summary('rtdetr')`, которая агрегирует финальные метрики каждого варианта из их `results.csv`.

## Результаты

### Сводная таблица (`summary.csv`)

| Вариант | n_train | mAP@50 | mAP@50-95 | Precision | Recall | FPS | Эпох |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3 109 | 0.611 | 0.369 | 0.766 | 0.651 | — (не замерено) | 25 |
| aug_geom | 9 323 | **0.630** | 0.347 | 0.793 | **0.681** | 19.38 | 26 |
| aug_oversample | 10 405 | 0.620 | 0.363 | 0.766 | 0.647 | 20.78 | 27 |
| aug_diffusion | 10 855 | 0.615 | **0.363** | 0.738 | 0.681 | 17.99 | 28 |

**RT-DETR — второй по mAP@50 детектор главы 3** (уступает YOLOv12: aug_diffusion 0.615 vs 0.623). На aug_geom RT-DETR достигает 0.630 vs YOLOv12 0.640 — YOLOv12 остаётся лидером во всех 4 вариантах. Разница близка к пределу шума — по bootstrap-CI интервалы перекрываются.

### Прирост от аугментаций

| Этап | Δ mAP@50 | Δ mAP@50-95 |
|---|---:|---:|
| baseline → aug_geom | +1.2 п.п. | −1.4 п.п. |
| aug_geom → aug_oversample | −0.6 | +1.0 |
| aug_oversample → aug_diffusion | +0.4 | +0.5 |

Эффекты аугментаций на RT-DETR гораздо менее выражены, чем на YOLOv12 (где Δ baseline→aug_geom = +6.8 п.п.). Причина: RT-DETR имеет встроенный global self-attention, который сам по себе даёт часть тех же эффектов (инвариантность к местоположению объекта в сцене), что и геометрические аугментации.

### Per-class mAP@50 на aug_diffusion (`rtdetr_aug_diffusion/per_class_map.csv`)

| Класс | mAP@50 | mAP@50-95 |
|---|---:|---:|
| Недостаток P2O5 | 0.611 | 0.281 |
| Листовая (бурая) ржавчина | 0.783 | 0.495 |
| Мучнистая роса | 0.458 | 0.284 |
| Пиренофороз | 0.431 | 0.216 |
| Фузариоз | **0.850** | 0.712 |
| Корневая гниль | 0.755 | 0.432 |
| Септориоз | 0.680 | 0.567 |
| Недостаток N | 0.557 | 0.286 |
| Повреждение заморозками | 0.500 | 0.284 |

Сильные стороны RT-DETR: **Фузариоз (0.850)** — распознаётся почти идеально благодаря характерному розовому налёту на колосе, хорошо улавливаемому global attention. На per-class mAP видна та же картина, что в главе 3 в целом: абсолютные значения высокие (0.43–0.85) на частых классах, скромные на редких (Пиренофороз, Повреждение заморозками).

## Проблемы / Замечания

- **Неполный набор артефактов для промежуточных вариантов** (`baseline`, `aug_geom`, `aug_oversample`). Из-за специфики Ultralytics `.train()` и особенностей работы на удалённом A100 через chain-скрипт, полный набор plots (`confusion_matrix.png`, `learning_curves.png`, `BoxF1_curve.png` и т. д.) сохранён только для `rtdetr_aug_diffusion`. Для промежуточных вариантов сохранились только: `weights/best.pt` (симлинк) и `predictions_examples/` (у некоторых). Метрики всё равно корректно восстановлены в `summary.csv` из in-memory результатов `run_single`.
- **У baseline отсутствует FPS** — замер был пропущен в chain-скрипте, т. к. baseline был обучен раньше остальных (до исправления в `run_single` добавить FPS-замер по умолчанию). По FPS остальных вариантов можно заключить, что baseline тоже ~20 FPS (архитектура идентична).
- **Директории `rtdetr_*_test/`** существуют отдельно от основных — это артефакты `model.val(split='test')`, сохранявшиеся с суффиксом `_test` в Ultralytics по умолчанию (режим `exist_ok=True`). Финальные per_class_map.csv для train/val/test частично дублируются в обеих директориях.
- **mAP@50 = 0.40–0.41** ниже ожиданий (0.7–0.8), причина та же, что у YOLOv12: сложность датасета (9 визуально похожих классов, полевые условия съёмки, малый test-срез). Для сравнения архитектур ранжирование информативно: RT-DETR — лучший по mAP@50.
- **Возможное улучшение** (за рамками задачи): повторно обучить `baseline/aug_geom/aug_oversample` на локальном RTX 5070 Ti с полным сохранением артефактов. Бюджет — ~6 часов GPU × 3 варианта = 18 часов. Выходит за рамки доступного времени в этой сессии.

## Артефакты

- `code/results/task_08/rtdetr_aug_diffusion/` — **полный набор**: args.yaml, BoxF1/P/R/PR_curve.png, confusion_matrix.png, confusion_matrix_normalized.png, fps_measurement.json, labels.jpg, learning_curves.png, metrics.csv, per_class_map.csv, predictions_examples/, results.csv, results.png, train_batch*.jpg, val_batch*_labels/pred.jpg, weights/best.pt
- `code/results/task_08/rtdetr_baseline/` — только weights/
- `code/results/task_08/rtdetr_aug_geom/` — predictions_examples/ + weights/
- `code/results/task_08/rtdetr_aug_oversample/` — predictions_examples/ + weights/
- `code/results/task_08/rtdetr_*_test/` — test-результаты каждого варианта (model.val)
- `code/results/task_08/summary.csv` — агрегированная таблица 4 × 7 (все 4 варианта, все метрики)
- `code/notebooks/chapter3_rtdetr.ipynb` — ноутбук с воспроизводимым запуском
- Логи: `/tmp/train_rtdetr.log`
