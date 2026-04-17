# Task 14: Late Fusion — контекст в классификаторе после детекции

## Статус
pending (после `task_12`)

## Цель

Реализовать и обучить бейзлайн «Late Fusion» — контекстный сигнал подключается **после** детекции, на уровне переклассификации уже предсказанных боксов. Такая схема соответствует базовому пониманию контекста из постановочных тезисов и является архитектурным антиподом CGFM (где контекст влияет на признаки **до** детектирующей головы). Сравнение показывает, что позднее слияние не влияет на сами координаты боксов, только на предсказанный класс — и демонстрирует, почему контекст должен заходить раньше (в neck), как в CGFM.

## Общий протокол

Все гиперпараметры, датасет, формат артефактов — в `code/docs/chapter4_protocol.md`. Используется `code/data/dataset_final/`. Модули Late-Fusion-классификатора и контекстного энкодера — из `code/models/chapter4/`.

## Шаги

1. **Создать ноутбук** `code/notebooks/chapter4_late_fusion.ipynb` со структурой:
   - §0. Импорты, seed, проверка GPU.
   - §1. Загрузка baseline YOLOv12 (из `code/results/task_12/yolov12_baseline/best.pt`) и его заморозка.
   - §2. Сбор датасета ROI-фичей.
   - §3. Обучение Late-Fusion-классификатора.
   - §4. Интеграция: end-to-end инференс (детектор → классификатор) и замер метрик на test.
   - §5. Сводка + экспорт в `code/results/task_14/summary.csv`.

2. **Этап A — заморозка baseline**:
   - Загрузить `best.pt` baseline YOLOv12 (ссылаясь на артефакты из `task_07/yolov12_aug_diffusion` через `task_12/yolov12_baseline/`).
   - Все параметры детектора заморозить (`requires_grad = False`).
   - Baseline в этом эксперименте отвечает **только за предсказание боксов**. Их класс будет переопределён Late-Fusion-головой.

3. **Этап B — сбор датасета ROI-фичей**:
   - Пройти по train-срезу `code/data/dataset_final/`, для каждого изображения:
     - Прогнать через baseline → получить предсказанные bbox + confidence + класс.
     - Для каждого предсказанного bbox, пересечение которого (IoU) с каким-либо ground-truth bbox ≥ 0.5 — сохранить:
       - ROI-фичу (через `torchvision.ops.roi_align` из карты P4 neck, размер 7×7×C) — это представление региона.
       - Контекстный вектор всего изображения: прогон через `ContextEncoder(backbone='mobilenetv3_small_100')` (или тот, который был выбран в task_12 по умолчанию).
       - Ground-truth класс matched bbox.
     - Сохранить в `code/results/task_14/roi_dataset_train.pt` (torch tensor dict: `{roi: [N, C, 7, 7], context: [N, 256], label: [N]}`).
   - Повторить для val и test → `roi_dataset_val.pt`, `roi_dataset_test.pt`.
   - Это даёт тренировочный набор «ROI + контекст → класс», на котором обучается Late-Fusion-классификатор.

4. **Этап C — обучение Late-Fusion-классификатора**:
   - Модель: `LateFusionClassifier(roi_dim=C*7*7, context_dim=256, num_classes=9)` из `code/models/chapter4/late_fusion_head.py`.
   - Оптимизатор: AdamW, lr=1e-3, weight_decay=1e-4.
   - Лосс: CrossEntropy по ground-truth классу matched bbox.
   - Epochs=30, patience=5 (классификатор обучается быстро).
   - Сохранить веса в `code/results/task_14/yolov12_late_fusion/late_head.pt`.

5. **Этап D — end-to-end инференс и метрики**:
   - На test-срезе: для каждого изображения прогнать baseline → получить bbox → для каждого bbox прогнать Late-Fusion-классификатор (используя ROI-фичу и контекст всей сцены) → переопределить класс bbox.
   - Пересчитать метрики через тот же инструментарий, что и в гл. 3 (`torchmetrics.MeanAveragePrecision` или внутренний `model.val`).
   - Координаты bbox при этом не меняются — только класс. Это ожидаемое поведение Late Fusion.

6. **Артефакты** — собрать полный набор согласно §5 протокола главы 4:
   - `metrics.csv` (по эпохам — для классификатора; для детектора — фиксированные значения из baseline)
   - `learning_curves.png` (кривые только для классификатора: train/val CrossEntropy)
   - `confusion_matrix.png` на test (после end-to-end)
   - `per_class_map.csv` на test
   - `predictions_examples/` — те же 10 файлов из `code/docs/chapter3_qualitative_sample.txt`, с предсказаниями после переклассификации
   - `fps_measurement.json` — теперь латентность = время детектора + время контекст-энкодера + время Late-Fusion-классификатора
   - `param_count.json` (детектор + контекст-энкодер + классификатор)

7. **Коммит** — `chapter4 late_fusion: done` + `git push`.

## Специфика

- **Координаты bbox не меняются** от Late Fusion. Следовательно, метрики mAP могут измениться **только** за счёт изменения предсказанного класса. Это означает:
  - mAP@50-95 и Precision могут вырасти за счёт исправления класса в FP-случаях.
  - Recall не может вырасти (Late Fusion не создаёт новые bbox).
  - На редких классах возможен заметный прирост Precision (Late-Fusion-классификатор может исправить частые путаницы).
- **ROI-фича извлекается из P4** (средний уровень FPN) как компромисс между семантикой и пространственным разрешением. Размер ROI после `roi_align` — 7×7. Если OOM — уменьшить до 5×5, зафиксировать в RESULT.md.
- **Bbox matcher**: простой IoU ≥ 0.5. Если для предсказанного bbox нет matching ground-truth — не включать в train-набор (это FP, для него нет правильной метки класса).
- **Баланс классов в ROI-датасете**: может сильно отличаться от оригинального распределения (т.к. baseline склонен находить более частые классы). Оставить как есть — это честное отражение данных, на которых будет работать классификатор в инференсе.

## Входные данные

- `code/data/dataset_final/data.yaml`
- `code/results/task_12/yolov12_baseline/best.pt` (замороженный детектор)
- `code/models/chapter4/late_fusion_head.py`, `context_encoder.py` (из task_12)

## Ожидаемый результат

- Ноутбук `code/notebooks/chapter4_late_fusion.ipynb` с полным пайплайном.
- `code/results/task_14/yolov12_late_fusion/` с полным набором артефактов.
- `code/results/task_14/summary.csv` — одна строка (yolov12_late_fusion) + reference-строка baseline.
- Заполненный `RESULT.md`.

### Ожидаемое качественное поведение

- mAP@50 прирост: +0.5–1.5 п.п. (исправление класса без сдвига bbox).
- mAP@50-95 прирост: слабее (высокие IoU-пороги чувствительны к точности координат, которые не меняются).
- Recall: неизменный или в пределах шума (±0.001).
- Precision: прирост +1–2 п.п. на частых классах.
- Confusion matrix: ожидается снижение путаницы между визуально похожими классами (Пиренофороз ↔ Септориоз).

## Результат записать в

`code/tasks/task_14/RESULT.md` по шаблону:

```markdown
# Result: Task 14 — Late Fusion

## Статус
done | partial | failed

## Что было сделано
Заморозка baseline, сбор ROI-датасета, обучение Late-Fusion-классификатора, end-to-end инференс.

## Размер ROI-датасета

| Срез | Кол-во ROI |
|---|---:|
| train | |
| val | |
| test | |

## Результаты

| Конфигурация | mAP@50 | mAP@50-95 | Precision | Recall | FPS | Params_M (total) | Эпох classifier |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (reference) | | | | | | | — |
| yolov12_late_fusion | | | | | | | |

### Per-class mAP@50 (сравнение)

| Класс | baseline | late_fusion | Δ |
|---|---:|---:|---:|
| ... | | | |

### Разложение латентности (Late Fusion)

| Компонент | Время, мс | Доля, % |
|---|---:|---:|
| Детектор | | |
| Контекст-энкодер | | |
| Late-Fusion-классификатор | | |
| Итого | | 100 |

## Проблемы / Замечания

## Артефакты
- `code/notebooks/chapter4_late_fusion.ipynb`
- `code/results/task_14/yolov12_late_fusion/`
- `code/results/task_14/roi_dataset_{train,val,test}.pt`
- `code/results/task_14/summary.csv`
```
