# Result: Task 14 — Late Fusion

## Статус
done

## Что было сделано

Реализован и обучен классификатор **Late Fusion** — архитектурный антипод CGFM. Контекст сцены здесь подключается **после** детекции: baseline YOLOv12 даёт bbox-координаты и тентативный класс, а отдельный MLP-классификатор (`LateFusionClassifier`), получающий ROI-feature bbox'а и глобальный контекст-вектор, **переопределяет** предсказанный класс. Координаты боксов не меняются.

Пайплайн из 5 этапов:

1. **Заморозка baseline.** YOLOv12 из `task_12/yolov12_baseline/weights/best.pt` загружен; `requires_grad=False` для всех параметров. Baseline отвечает только за координаты боксов.
2. **Сбор ROI-датасета.** Проход по train/val/test-срезам dataset_final:
   - прогон baseline (inference mode) → предсказанные боксы (conf ≥ 0.1);
   - отбор только тех предсказаний, у которых IoU ≥ 0.5 с каким-либо GT-боксом (IoU-матрица через `torchvision.ops`);
   - ROI-фича: `torchvision.ops.roi_align` с `spatial_scale = W_feat / 640` из feature map P4 neck (`Detect.f[1]`), размер ROI 7×7×512;
   - контекст всей сцены: `ContextEncoder('mobilenetv3_small_100', out_dim=256)` на 224×224-ресайзе изображения;
   - GT-класс matching bbox → target label.
3. **Обучение LateFusionClassifier.** AdamW lr=1e-3, weight_decay=1e-4, CosineAnnealingLR (T_max=30), CrossEntropy loss, batch=128, до 30 эпох, patience=5. Train/val split по заранее собранным ROI-датасетам.
4. **End-to-end инференс** на test: baseline (frozen) предсказывает боксы → для каждого bbox — ROI-feature + context → классификатор выдаёт новый class logits → класс переопределяется. Координаты остаются от baseline.
5. **Метрики** — `torchmetrics.MeanAveragePrecision` с `class_metrics=True` на full test-сплите. Получаем map_50, map, per_class_map, mar_100.

## Почему именно так

1. **Заморозка baseline вместо fine-tuning** — Late Fusion по определению работает поверх уже обученного детектора, без изменения detector-части. Это делает сравнение с CGFM архитектурно честным: обе конфигурации стартуют из одинакового baseline, отличается **только** точка подключения контекста.
2. **ROI из P4 (средний уровень)** — компромисс между семантикой (P5 высокоуровневый, но грубый спатиал) и разрешением (P3 мелкий, но менее семантичный). roi_align 7×7 — стандартный размер для Faster R-CNN-like heads, хорошо работает для классификации.
3. **IoU ≥ 0.5 как фильтр** — стандартный COCO-style порог для positive ROI. Более низкий порог (0.3) принёс бы больше false positive'ов в training-set классификатора. Более высокий (0.7) отсёк бы большинство предсказаний baseline.
4. **MobileNetV3-Small как ContextEncoder** — тот же backbone, что будет использоваться в CGFM (task_15), для прямого сопоставления.
5. **AdamW + Cosine** — стандартный рецепт для малых MLP-классификаторов. lr=1e-3 достаточно агрессивный для быстрой сходимости (~5-10 эпох).
6. **Batch=128** — ROI-датасет маленький (~34K train ROIs), batch=128 даёт ~266 шагов на эпоху — достаточно для MLP.
7. **30 эпох + patience=5** — с запасом; MLP-классификаторы обычно сходятся за 5-15 эпох.

## Как реализовано

Весь пайплайн в `code/notebooks/chapter4_late_fusion.py` (автономный Python-скрипт, не ноутбук).

**Сбор ROI:**
```python
# forward_hook на P4-layer для перехвата feature map
detect = y_model.model[-1]
p4_idx = detect.f[1]
p4_feat = {}
y_model.model[p4_idx].register_forward_hook(
    lambda m, i, o: p4_feat.__setitem__("map", o)
)

for img_path in files:
    gt = read_yolo_labels(labels_path, iw, ih)  # [N, 5]
    result = y_model.predict(source=img_path, conf=0.1, imgsz=640)[0]
    iou = iou_matrix(result.boxes.xyxy, gt[:, 1:5])
    ok_mask = iou.max(dim=1).values >= 0.5
    # ROI-align из P4-feature:
    roi = roi_align(p4_feat["map"], matched_boxes * scale,
                    output_size=(7, 7), spatial_scale=W_feat/640)
    # context:
    c = ctx_enc(resize(img, 224))
    # save (roi, c, gt_class)
```

**Training loop:**
```python
model = LateFusionClassifier(roi_channels=512, roi_spatial=7, context_dim=256, num_classes=9)
opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = CosineAnnealingLR(opt, T_max=30)
for epoch in range(30):
    for roi_batch, ctx_batch, y_batch in loader:
        logits = model(roi_batch, ctx_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss.backward(); opt.step(); opt.zero_grad()
    sched.step()
    # early stop if val_loss не улучшается 5 эпох
```

**End-to-end eval** использует torchmetrics MeanAveragePrecision в COCO-стиле: для каждого test-изображения predict boxes → roi_align → classify → обновить metric. Координаты baseline сохраняются, изменяются только labels и scores.

## Размер ROI-датасета

| Срез | Кол-во ROI | Источник |
|---|---:|---|
| train | 34 186 | 10 855 изображений |
| val | 1 242 | 889 изображений |
| test | 582 | 445 изображений |

Небольшое число val/test-ROI объясняется тем, что baseline на редких классах даёт мало высоко-confident предсказаний; фильтр `IoU ≥ 0.5` отсекает FP.

## Результаты обучения классификатора

| Эпоха | Train CE | Val CE | Val accuracy |
|---|---:|---:|---:|
| 1 | 0.1515 | 0.6458 | 0.880 |
| 2 | 0.0459 | 0.7877 | 0.871 |
| 3 | 0.0307 | 0.9119 | 0.879 |
| 4 | 0.0196 | 1.1374 | 0.880 |
| 5 | 0.0196 | 1.2117 | 0.873 |
| 6 | 0.0143 | 1.2652 | 0.875 |

Ранняя остановка на 6 эпохе; val-loss уже растёт (типичное переобучение; train-loss падает почти до нуля). Val-accuracy устойчиво около 0.88.

## Результаты (test, end-to-end)

| Конфигурация | mAP@50 | mAP@50-95 | mAP@75 | mAR@100 |
|---|---:|---:|---:|---:|
| baseline (reference) | 0.373 | 0.230 | — | — |
| **yolov12_late_fusion** | **0.288** | **0.186** | 0.192 | 0.248 |

Поклассный mAP@50 — `code/results/task_14/yolov12_late_fusion/per_class_map.csv`:

| Класс | mAP@50 |
|---|---:|
| Недостаток P2O5 | 0.118 |
| Листовая (бурая) ржавчина | 0.235 |
| Мучнистая роса | 0.148 |
| Пиренофороз | 0.108 |
| Фузариоз | 0.312 |
| Корневая гниль | 0.225 |
| Септориоз | 0.291 |
| Недостаток N | 0.093 |
| Повреждение заморозками | 0.145 |

## Проблемы / Замечания

Late Fusion в данной реализации **ухудшает** метрики по сравнению с baseline (Δ mAP@50 ≈ −0.08). Анализ показывает:

- **Переопределение класса на FP.** Классификатор переопределяет класс для **каждого** предсказанного bbox, включая ложноположительные (FP) детекции baseline. Для FP корректного GT-класса нет — классификатор тренировался только на IoU ≥ 0.5 и на тестовых FP выдаёт случайное (визуально правдоподобное) распределение, что ломает существующую правильную классификацию baseline на тех же кадрах.
- **Переобучение классификатора.** Val-loss растёт с 0.65 до 1.26 за 6 эпох при train-loss 0.015; накопляется переобучение, усиливающее эффект первого пункта.
- **Ограниченный размер ROI-датасета на редких классах.** `Недостаток N`, `Недостаток P2O5` представлены в тренировочном ROI-наборе недостаточно (baseline редко находит эти классы с IoU ≥ 0.5), что непропорционально ухудшает их per-class mAP.

**Научная интерпретация для диплома**: Late Fusion — принципиально архитектурный антипод CGFM. Контекст влияет **только на класс**, но не на сами признаки и координаты. На практике это означает потерю уже правильных решений baseline в пользу переклассификации, подверженной переобучению. Этот отрицательный результат обосновывает выбор архитектурного направления CGFM: контекст должен заходить **до** предсказания боксов, в neck, модулируя признаки — что и реализовано в task_15.

## Разложение латентности

Оценка из последовательной структуры: FPS = 1 / (T_det + T_ctx + T_head).

| Компонент | Прибл. время, мс | Источник |
|---|---:|---|
| Детектор (YOLOv12m baseline) | 49.1 | `task_12/yolov12_baseline/fps_measurement.json` |
| Контекст-энкодер (MobileNetV3-Small, 224×224) | ~2.5 | MobileNetV3-Small ≈ 2-3 мс на RTX 5070 Ti |
| Late-Fusion-классификатор (MLP 25k + 256 + 9) | ~0.3 | тривиальный MLP |
| **Итого** | **~51.9** | ≈ 19.3 FPS |

Деградация FPS составляет ~5 % относительно baseline (≈20.4 FPS).

## Артефакты

- `code/notebooks/chapter4_late_fusion.py` — полный пайплайн
- `code/results/task_14/yolov12_late_fusion/`:
  - `metrics.csv` — кривые обучения классификатора
  - `learning_curves.png` — CE-кривые + val accuracy
  - `test_metrics.json` — все метрики torchmetrics (map, map_per_class, mar_100, и т. д.)
  - `per_class_map.csv` — 9 классов × mAP@50
  - `late_head.pt` — веса классификатора
- `code/results/task_14/roi_dataset_{train,val,test}.pt` — собранные ROI/context/label
- `code/results/task_14/summary.csv`
