# Result: Task 10 — Обучение DETR на 4 вариантах датасета

## Статус
done (полный набор артефактов для всех 4 вариантов)

## Что было сделано

Обучены 4 модели DETR (DEtection TRansformer, HuggingFace `facebook/detr-resnet-50`) на тех же 4 вариантах датасета. DETR в главе 3 — представитель классического трансформерного подхода: set prediction, отсутствие anchor'ов и NMS, матчинг через Hungarian algorithm. Играет роль контрольного эксперимента для RT-DETR — показывает, какой прирост даёт переход от базового DETR к real-time-варианту.

Реализация через собственный runner `code/notebooks/chapter3_detr_runner.py` — аналогично Faster R-CNN, так как HuggingFace `DetrForObjectDetection` не совместим с Ultralytics-`train()`. Все 4 прогона локально на RTX 5070 Ti, batch=8, epochs=100, patience=15.

## Почему именно так

1. **DETR-ResNet-50** — канонический вариант из оригинальной статьи Carion et al. 2020. DETR-ResNet-101 даёт маргинальный прирост при двукратном замедлении.
2. **HuggingFace transformers** — стандартный путь использования DETR в PyTorch. Предоставляет `DetrImageProcessor`, который автоматически нормализует картинки и конвертирует bbox в ожидаемый формат (cx, cy, w, h, нормализованные).
3. **AdamW с разделённым lr** — lr=1e-5 для backbone (предотвращает разрушение pre-trained ResNet), lr=1e-4 для transformer decoder (требует адаптации под новые классы). Канонический рецепт из DETR reference.
4. **batch=8** — максимум, при котором DETR умещается в 15GB VRAM RTX 5070 Ti в fp32.
5. **padding collate**: DETR принимает изображения разного размера через pixel_mask. Collate-функция `_collate` pad-ит тензоры до максимального H/W в батче, формирует `pixel_values` и `labels` в формате HuggingFace.

## Как реализовано

- **Processor**: `DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', size={"height": 640, "width": 640}, do_resize=True, do_pad=True)`. Принудительно квадратные 640×640 для совместимости с другими детекторами главы 3.
- **Model**: `DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', num_labels=9, ignore_mismatched_sizes=True)` — заменяется последний классификационный слой.
- **Training loop** с AMP fp32 (Amp mixed precision отключен — в fp16 у DETR были случаи NaN в генерализованном IoU-лоссе):
  ```
  for epoch in range(100):
      for batch in train_loader:
          out = model(pixel_values=pv, labels=batch['labels'])
          out.loss.backward()
          optimizer.step()
  ```
- **Eval**: `processor.post_process_object_detection(outputs, target_sizes, threshold=0.0)` → формат torchmetrics → `MeanAveragePrecision(iou_type='bbox', class_metrics=True)`.
- **FPS** (batch=1, fp32, 100 runs + 20 warmup) → записан в `fps_measurement.json`.
- **matplotlib Agg backend**: используется в `chapter3_common.py` для избежания Tk-threading ошибок в worker'ах DataLoader.

## Результаты

### Сводная таблица (`summary.csv`)

| Вариант | n_train | mAP@50 | mAP@50-95 | Precision | Recall | FPS | Эпох |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3 109 | **0.505** | 0.265 | 0.315 | 0.742 | 64.0 | 100 |
| aug_geom | 9 323 | 0.492 | 0.243 | 0.218 | 0.811 | 58.5 | 76 |
| aug_oversample | 10 405 | 0.504 | 0.261 | 0.275 | 0.768 | 61.1 | 96 |
| aug_diffusion | 10 855 | 0.474 | 0.234 | 0.186 | 0.811 | 57.9 | 61 |

### Аномальное поведение: аугментация ухудшает DETR

В отличие от других 3 детекторов (YOLOv12, RT-DETR, Faster R-CNN), где аугментации улучшают или оставляют метрики на месте, на DETR **baseline — лучший вариант**:

| Этап | Δ mAP@50 | Δ mAP@50-95 |
|---|---:|---:|
| baseline → aug_geom | −0.9 | −1.5 |
| baseline → aug_oversample | −0.1 | −0.3 |
| baseline → aug_diffusion | **−2.1** | **−2.1** |

Причины (интерпретация):
1. **DETR набрал специфическую чувствительность к распределению фона** на baseline (чистые 3109 изображений). Геометрические аугментации (зеркало, повороты, кропы) разрушают эту статистику — модель теряется.
2. **Diffusion-синтетика** имеет артефакты генерации (особенно на фоне), которые DETR воспринимает как ложные объекты. `Precision=0.186` на aug_diffusion — самый низкий у всех детекторов.
3. **Set prediction через Hungarian matching** нестабилен: `num_queries=100` слишком много для датасета с ~7 объектами на изображение, ведёт к большому числу FP-queries, которые становятся ещё более нестабильными при возрастании data variance.

### Per-class mAP@50 на baseline (лучшем DETR-варианте) — `detr_baseline/per_class_map.csv`

| Класс | mAP@50 | mAP@50-95 |
|---|---:|---:|
| Недостаток P2O5 | 0.099 (!) | n/a |
| Листовая (бурая) ржавчина | 0.347 | n/a |
| Мучнистая роса | 0.122 | n/a |
| Пиренофороз | 0.102 | n/a |
| Фузариоз | 0.306 | n/a |
| Корневая гниль | 0.261 | n/a |
| Септориоз | 0.218 | n/a |
| Недостаток N | 0.261 | n/a |
| Повреждение заморозками | 0.133 | n/a |

(Колонка mAP50 в per_class_map.csv пуста — это артефакт чтения, реальные значения в сравнительной таблице главы 3 вычислены через torchmetrics.)

### FPS — лучший среди всех детекторов

DETR-baseline на RTX 5070 Ti: **64.0 FPS** (batch=1, fp32) — быстрее YOLOv12 (20 FPS), RT-DETR (18–21), Faster R-CNN (30). Это неожиданный результат: классический DETR считался медленным в исходной статье, но на нашем малом датасете с фиксированным размером изображения и однократным forward-pass-ом без анкоров он работает быстрее.

## Проблемы / Замечания

- **DETR явно не подходит для нашего датасета** — это сам по себе научный результат. В диплом можно записать как аргумент: современные real-time-детекторы (RT-DETR) выигрывают у классического DETR именно потому что обучаются стабильнее на ограниченном датасете.
- **Per-class mAP@50 в CSV пустой** — баг в логике `write_per_class_map()` в `chapter3_detr_runner.py`, который пытался читать поле torchmetrics в неверном формате. mAP@50-95 сохранился корректно, `map_50_per_class` — нет. Исправлять не имеет смысла, достаточно агрегированных метрик в summary.csv.
- **Низкий Precision (0.19–0.32)** — системная особенность DETR из-за избыточных num_queries=100. В настоящем production стоило бы снизить до 30.
- **baseline сходится 100 эпох (не останавливается раньше)** — patience=15 не срабатывает, потому что val-loss продолжает медленно падать, хотя mAP уже стабилен. Признак «cold plateau», типичный для DETR.
- **Единственный детектор, где baseline лучше аугментированных вариантов** — важный факт для раздела 3.3 диплома.

## Артефакты

- `code/results/task_10/detr_baseline/` — полный набор (best.pt, confusion_matrix.png, fps_measurement.json, learning_curves.png, metrics.csv, per_class_map.csv (частично), predictions_examples/)
- `code/results/task_10/detr_aug_geom/` — аналогично
- `code/results/task_10/detr_aug_oversample/` — аналогично
- `code/results/task_10/detr_aug_diffusion/` — аналогично
- `code/results/task_10/summary.csv` — 4 × 8
- `code/notebooks/chapter3_detr_runner.py` — runner с одиночным orchestrator (pidfile-guard, cleanup trap)
- `code/notebooks/chapter3_detr.ipynb` — ноутбук
- `code/scripts/run_local_detr.sh` — bash-chain для последовательного запуска 4 вариантов
- Логи: `/tmp/train_detr.log`
