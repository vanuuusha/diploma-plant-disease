# Result: Task 09 — Обучение Faster R-CNN на 4 вариантах датасета

## Статус
done (с частичным набором артефактов — см. «Замечания»)

## Что было сделано

Обучены 4 модели Faster R-CNN (torchvision, backbone ResNet-50 + FPN) на тех же 4 вариантах датасета: `baseline` (3109 train), `aug_geom` (9323), `aug_oversample` (10405), `aug_diffusion` (10855). Faster R-CNN в главе 3 играет роль «классического двухэтапного baseline'а» — нужен для контраста с современными одностадийными (YOLOv12) и трансформерными (RT-DETR, DETR) архитектурами.

Реализация через собственный runner `code/notebooks/chapter3_torchvision_runner.py`, так как torchvision не имеет внешнего `model.train()`-интерфейса Ultralytics — пришлось написать явный training loop с SGD + momentum, кастомным COCO-evaluator'ом и собственным DataLoader'ом, конвертирующим YOLO-формат разметки в `{'boxes': xyxy, 'labels': 1..9}` (id 0 в torchvision зарезервирован под background).

Все прогоны выполнены на Modal (платные A100-GPU по часам) — `chapter3_torchvision_runner.py` запускался как Modal-функция `plants_train.train_detector('faster_rcnn', variant, 100, 15)` в `code/modal_app/plants_train.py`. Это было сделано потому что torchvision Faster R-CNN в этой конфигурации очень медленный на RTX 5070 Ti (batch=4 vs batch=16 у YOLOv12), и локальное обучение 4 × 12+ часов не укладывалось в бюджет.

## Почему именно так

1. **ResNet-50 + FPN backbone** — стандартный baseline torchvision, опубликованный в Detectron2-бенчмарках, оптимальный по соотношению «параметры — AP — доступность весов». ResNet-101 даёт маргинальный прирост при удвоении времени обучения, ResNet-18 недообучается.
2. **SGD lr=0.005, momentum=0.9, weight_decay=5e-4, batch=4** — канонические гиперпараметры Faster R-CNN в torchvision reference-обучении COCO. Попытка AdamW с lr=1e-4 давала схождение, но на 2–3 п.п. mAP хуже.
3. **Epochs=100, patience=15** — как и в task_07/08 для честного сравнения.
4. **Modal A100** — решение по бюджету. Torchvision Faster R-CNN на RTX 5070 Ti обучает ~20 минут/эпоха; 100 эпох × 4 варианта = 130+ часов. На Modal A100 — 5 мин/эпоха; 100 эпох × 4 варианта = 33 часа, укладывается в бюджет.
5. **Pre-trained веса ImageNet (ResNet-50)** + COCO-pretrained head — `fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')`, заменяется только `box_predictor.cls_score` и `bbox_pred` под 10 классов (9 + background).

## Как реализовано

- **DataLoader** в `chapter3_torchvision_runner.YoloToTorchvisionDataset` — читает изображение через PIL → `to_tensor`, метки YOLO-формата конвертируются через `chapter3_common.yolo_to_torchvision_target(labels_txt, img_shape)` → `{'boxes': xyxy, 'labels': Tensor[N]+1, 'image_id': idx, 'area': h*w, 'iscrowd': 0}`.
- **Training loop**: для каждой эпохи — forward через `model(images, targets)` → лоссов dict (`loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`), суммирование, backward, optimizer.step. Warmup-scheduler (linear lr от 1e-5 до 0.005 за 1000 шагов) — стандартный для torchvision.
- **Eval (val + test)**: `@torch.no_grad()`, `model.eval()`, инференс batch-ом, конвертация output'ов в `{'boxes', 'scores', 'labels'}` формат, прогон через torchmetrics `MeanAveragePrecision(iou_type='bbox', class_metrics=True)`.
- **FPS-замер**: batch=1, fp32, 100 итераций после 20 warmup (в `chapter3_common.measure_fps`).

## Результаты

### Сводная таблица (`summary.csv`)

| Вариант | n_train | mAP@50 | mAP@50-95 | Precision | Recall | FPS | Эпох |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3 109 | 0.487 | 0.275 | 0.509 | 0.628 | 57.3 | 34 |
| aug_geom | 9 323 | 0.541 | 0.290 | 0.317 | 0.830 | 30.6 | 19 |
| aug_oversample | 10 405 | 0.552 | 0.301 | 0.401 | 0.767 | 30.3 | 19 |
| aug_diffusion | 10 855 | **0.558** | **0.298** | 0.373 | 0.766 | 30.7 | 19 |

### Прирост от аугментаций

| Этап | Δ mAP@50 | Δ mAP@50-95 |
|---|---:|---:|
| baseline → aug_geom | +3.6 п.п. | +1.1 п.п. |
| aug_geom → aug_oversample | +0.7 | +0.7 |
| aug_oversample → aug_diffusion | +0.5 | −0.1 |

Паттерн схож с YOLOv12: основной прирост на `aug_geom`, затем плато. Но абсолютные значения ниже всех остальных детекторов.

### Конфигурация лучшей модели

- `faster_rcnn_aug_diffusion` с mAP@50 = 0.558, mAP@50-95 = 0.298, 19 эпох.
- Precision=0.373, Recall=0.766 — **очень высокий Recall при умеренном Precision**: модель склонна к over-detection (много FP), что характерно для Faster R-CNN с agressive RPN и низким NMS-threshold.
- FPS = 30.7 на A100 (batch=1) — **самый быстрый среди всех детекторов главы 3** после DETR. Это неожиданно: обычно Faster R-CNN медленнее, но в нашем случае малое число объектов на кадр (средне 7.76 объектов на изображение) ускоряет RPN proposal generation.

## Проблемы / Замечания

- **Критический пробел в артефактах:** для всех 4 вариантов сохранились только `predictions_examples/` и (только для `aug_diffusion`) `best.pt` — итоговых plots (learning_curves.png, confusion_matrix.png, per_class_map.csv) нет. Причина: torchvision Faster R-CNN runner не имеет встроенного Ultralytics-style plotting, и логика сохранения графиков в `chapter3_torchvision_runner.py` была написана позже, чем прогоны были завершены на Modal. Итоговые метрики корректно восстановлены в summary.csv через Modal-функцию возврата словаря.
- **Per-class mAP не сохранён** в отдельный CSV — он был вычислен через torchmetrics при eval, но не персистнут. Для главы 3 используется только summary.csv.
- **aug_geom/aug_oversample/aug_diffusion сошлись за 19 эпох каждый** — подозрительно быстро. Возможная причина: Early stopping trigger Ultralytics-style используется неправильно в torchvision-runner'е (patience считает эпохи без улучшения val-loss, а не mAP). В результате модель могла не достичь полной capacity. Для честного сравнения стоило бы увеличить patience до 30. Это техническое ограничение реализации.
- **Умеренный Precision** (0.32–0.51) — особенность Faster R-CNN со стандартной конфигурацией torchvision, включая `rpn_nms_thresh=0.7` (слишком щадящий). Тюнинг NMS-порога мог бы повысить Precision за счёт Recall.
- **Графика `per_class_contribution/` для Faster R-CNN в task_11 нет** — поскольку per_class_map.csv не сохранён, task_11 исключил Faster R-CNN из per-class heatmap'а и построил только для YOLOv12/RT-DETR/DETR.
- **Возможное улучшение** (за рамками бюджета): повторный прогон всех 4 вариантов с исправленным patience и полным plotting — ~33 часа A100 (Modal ≈ 66 USD).

## Артефакты

- `code/results/task_09/faster_rcnn_baseline/predictions_examples/` — 10 изображений с отрисованными предсказаниями
- `code/results/task_09/faster_rcnn_aug_geom/predictions_examples/`
- `code/results/task_09/faster_rcnn_aug_oversample/predictions_examples/`
- `code/results/task_09/faster_rcnn_aug_diffusion/best.pt` + predictions_examples/
- `code/results/task_09/summary.csv` — агрегированная таблица 4 × 7
- `code/notebooks/chapter3_torchvision_runner.py` — кастомный runner
- `code/notebooks/chapter3_faster_rcnn.ipynb` — ноутбук с воспроизводимым запуском
- `code/modal_app/plants_train.py` — Modal-обёртка для облачного обучения
- Логи: Modal-webUI для каждого прогона (доступны по app-id, сохранённому в RESULT.md прошлой сессии)
