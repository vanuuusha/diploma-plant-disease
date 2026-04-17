# Task 18: Сводный анализ главы 4 — таблицы, визуализации, статистические тесты

## Статус
pending (стартует после завершения `task_13`, `task_14`, `task_15`, `task_16`, `task_17`)

## Цель

Собрать результаты всех экспериментов главы 4 в единую аналитическую картину, провести статистические тесты значимости прироста CGFM над ближайшими конкурентами и подготовить весь числовой и визуальный материал для разделов 4.3 «Аблационное исследование» и 4.4 «Результаты и анализ» дипломной работы. Без этой задачи глава 4 не может быть написана — для текста нужны сводные таблицы, картинки и доверительные интервалы.

## Общий протокол

Формат, имена конфигураций, методология метрик — в `code/docs/chapter4_protocol.md`. Статистические процедуры — §8 этого протокола.

## Шаги

1. **Создать ноутбук** `code/notebooks/chapter4_summary.ipynb`.

2. **Сборка данных** — прочитать `summary.csv` из всех предшествующих задач и объединить:
   - `code/results/task_12/yolov12_baseline/` (baseline YOLOv12)
   - `code/results/task_13/summary.csv` (SE, CBAM)
   - `code/results/task_14/summary.csv` (Late Fusion)
   - `code/results/task_15/summary.csv` (CGFM, CGFM+Late)
   - `code/results/task_16/summary.csv` (аблация CGFM)
   - `code/results/task_17/summary.csv` (RT-DETR baseline, RT-DETR+CGFM)
   - Выход: `code/results/task_18/chapter4_grand_summary.csv` — одна строка на конфигурацию, колонки: `config, detector, variant, mAP@50, mAP@50-95, Precision, Recall, FPS, Params_M, GFLOPs, Epochs, Task`.

3. **Главная сводная таблица** для раздела 4.4 — `code/results/task_18/main_results_table.csv` + `main_results_table.png` (визуальный рендер):

| Конфигурация | mAP@50 | mAP@50-95 | Precision | Recall | FPS | Params_M | Δ mAP@50 vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv12 baseline | | | | | | | 0 (ref) |
| YOLOv12 + SE-Neck | | | | | | | |
| YOLOv12 + CBAM-Neck | | | | | | | |
| YOLOv12 + Late Fusion | | | | | | | |
| YOLOv12 + **CGFM** | | | | | | | |
| YOLOv12 + CGFM + Late | | | | | | | |

4. **Таблица аблации CGFM** — `code/results/task_18/ablation_table.csv` (5 строк из task_15 + task_16):

| Уровни | Энкодер | mAP@50 | mAP@50-95 | FPS | Params_M |
|---|---|---:|---:|---:|---:|
| P3+P4+P5 | MobileNetV3-Small | | | | |
| P5 only | MobileNetV3-Small | | | | |
| P3 only | MobileNetV3-Small | | | | |
| P3+P4+P5 | EfficientNet-B0 | | | | |
| P3+P4+P5 | ViT-Tiny | | | | |

5. **Таблица переносимости** — `code/results/task_18/transferability_table.csv`:

| Детектор | Baseline mAP@50 | + CGFM mAP@50 | Δ, п.п. | Δ FPS, % |
|---|---:|---:|---:|---:|
| YOLOv12 | | | | |
| RT-DETR | | | | |

6. **Bar-plot сравнения конфигураций** — `code/results/task_18/configs_barplot.png`:
   - 6 столбиков (baseline, SE, CBAM, Late, CGFM, CGFM+Late).
   - Две оси Y: mAP@50 (левая) и mAP@50-95 (правая).
   - Значения над столбиками; baseline — полоса reference-линия пунктиром.

7. **Pareto scatter «FPS vs mAP@50-95»** — `code/results/task_18/pareto_scatter.png`:
   - Все конфигурации главы 4 как точки (включая RT-DETR-ветвь, другим цветом).
   - Подписи конфигураций; линия Pareto-фронта.

8. **Per-class mAP heatmap** — `code/results/task_18/per_class_heatmap.png`:
   - Строки — 9 классов (болезни + дефициты).
   - Колонки — конфигурации (baseline, SE, CBAM, Late, CGFM).
   - Цвет — mAP@50 (от тёмного к светлому); значения внутри ячеек.
   - Даёт наглядную картину, на каких классах CGFM даёт максимальный прирост.

9. **γ-визуализации** (ключевое для раздела 4.4) — `code/results/task_18/gamma_analysis/`:
   - **γ-тепловые карты по уровням P3/P4/P5** для 10 тестовых файлов — скомпоновать из `code/results/task_15/yolov12_cgfm/gamma_maps/` в единую фигуру 10×3 для диплома (файл: `gamma_heatmaps_grid.png`).
   - **Гистограммы значений γ по классам** — 9 классов × 3 уровня. Фигура `gamma_histograms_by_class.png`: показывает, что разные болезни активируют разные подмножества каналов.
   - **Корреляция «сложность сцены ↔ γ-дисперсия»** — для test-среза вычислить прокси-сложности (например, число объектов на кадре) и построить scatter с γ-дисперсией. Фигура `gamma_vs_scene_complexity.png`.

10. **t-SNE контекстных эмбеддингов** — `code/results/task_18/tsne_context.png`:
    - Загрузить `context_embeddings.npy` из `yolov12_cgfm/` (после обучения), `yolov12_baseline/context_embeddings.npy` (до обучения), и эмбеддингов других аблационных энкодеров (EfficientNet, ViT-Tiny).
    - Прогнать sklearn t-SNE (perplexity=30, n_iter=1000).
    - Покрасить точки по преобладающему классу болезни на изображении (или по условиям съёмки — близкий/общий план).
    - Сравнить: до обучения (без CGFM) vs после (CGFM) — ожидается, что обученные контекстные эмбеддинги кластеризуются по визуальным характеристикам сцены лучше, чем pretrained.

11. **Качественная сводная галерея (FP/FN анализ)** — `code/results/task_18/qualitative_fpfn.png`:
    - Выбрать 3–5 тестовых изображений, на которых baseline делает ошибку (FP или FN), а CGFM предсказывает правильно (или наоборот, если такие случаи есть).
    - Сетка: img × (true boxes, baseline prediction, CGFM prediction).
    - Для диплома — самые наглядные примеры.

12. **Статистические тесты** (раздел 4.4 — доверительные интервалы и значимость):

    **12.1. Bootstrap CI для всех ключевых конфигураций**:
    - Для каждой из 6 YOLOv12-конфигураций: 1000 bootstrap-прогонов, каждый раз семплирование 80 % test-изображений с возвратом, пересчёт mAP@50.
    - Распределение → 95 % CI (2.5-я и 97.5-я перцентили).
    - Выход: `code/results/task_18/bootstrap_ci.json`.

    **12.2. Permutation test значимости прироста CGFM над лучшим конкурентом**:
    - Пара: CGFM vs argmax(SE, CBAM, Late) по mAP@50.
    - Для каждого test-изображения есть mAP@50 CGFM и mAP@50 конкурента.
    - Переставить метки «CGFM / конкурент» на уровне изображений 10 000 раз, пересчитать разницу средних mAP.
    - p-value = доля перестановок, где разница превзошла наблюдаемую.
    - Выход: `code/results/task_18/permutation_test.json`.

    **12.3. Аналогичный тест для RT-DETR + CGFM vs RT-DETR baseline**.

    **12.4. Визуализация** — `code/results/task_18/ci_forestplot.png`: лесной график с точкой mAP@50 и горизонтальным CI для каждой конфигурации.

13. **Финальные таблицы для диплома** — единый файл `code/results/task_18/for_diploma.md` с готовыми к вставке markdown-таблицами:
    - Таблица 4.1. Главные результаты (6 строк).
    - Таблица 4.2. Аблация CGFM (5 строк).
    - Таблица 4.3. Переносимость (2 строки).
    - Таблица 4.4. Bootstrap 95 % CI (6 строк).
    - Таблица 4.5. Per-class mAP@50 baseline vs CGFM (9 строк).

14. **Коммит** — `chapter4 summary: done` + `git push`.

## Специфика

- **Статистическая методология должна быть корректной**:
  - Bootstrap: семплирование **с возвратом** из test-изображений, не из bbox (bbox одного и того же изображения коррелируют).
  - Permutation test: перестановка только между двумя сравниваемыми конфигурациями; смешивание с третьим разрушит распределение под null-гипотезой.
- **Шрифты и размеры фигур** — согласовать со стилем главы 3 (чтобы визуально сочеталось). Размеры: 8×6 дюймов для одиночных фигур, 12×8 для больших; DPI=150.
- **Цветовая кодировка конфигураций**:
  - baseline — серый
  - SE-Neck — оранжевый
  - CBAM-Neck — жёлтый
  - Late Fusion — голубой
  - **CGFM — тёмно-зелёный (акцентный)**
  - CGFM + Late — зелёный более светлый
  - RT-DETR-ветвь — красноватые оттенки
- **Для γ-визуализаций** использовать `matplotlib.colormaps['RdBu_r']` (красный — усиление канала, синий — подавление, белый — identity), нормировка симметрично относительно γ=1.
- **Все числа** в финальных таблицах — до 3 значащих цифр после запятой (mAP); FPS — до одной цифры; параметры — до 2.

## Входные данные

- `code/results/task_12/...` (baseline reference)
- `code/results/task_13/summary.csv`
- `code/results/task_14/summary.csv`
- `code/results/task_15/` (с γ-картами и эмбеддингами)
- `code/results/task_16/summary.csv`
- `code/results/task_17/summary.csv`

## Ожидаемый результат

- Ноутбук `code/notebooks/chapter4_summary.ipynb` со всеми вычислениями и фигурами.
- `code/results/task_18/`:
  ```
  chapter4_grand_summary.csv
  main_results_table.csv
  main_results_table.png
  ablation_table.csv
  transferability_table.csv
  configs_barplot.png
  pareto_scatter.png
  per_class_heatmap.png
  gamma_analysis/
      gamma_heatmaps_grid.png
      gamma_histograms_by_class.png
      gamma_vs_scene_complexity.png
  tsne_context.png
  qualitative_fpfn.png
  bootstrap_ci.json
  permutation_test.json
  ci_forestplot.png
  for_diploma.md
  ```
- Заполненный `RESULT.md` с готовыми к вставке в диплом выводами.

## Результат записать в

`code/tasks/task_18/RESULT.md` по шаблону:

```markdown
# Result: Task 18 — Сводный анализ главы 4

## Статус
done | partial | failed

## Что было сделано
Агрегация, статистические тесты, визуализации.

## Главные результаты

### Таблица сравнения конфигураций (YOLOv12)

| Конфигурация | mAP@50 | mAP@50-95 | Precision | Recall | FPS | 95 % CI mAP@50 |
|---|---:|---:|---:|---:|---:|---|
| baseline | | | | | | [ , ] |
| SE-Neck | | | | | | [ , ] |
| CBAM-Neck | | | | | | [ , ] |
| Late Fusion | | | | | | [ , ] |
| **CGFM** | | | | | | [ , ] |
| CGFM + Late | | | | | | [ , ] |

### Значимость прироста CGFM над лучшим конкурентом

- Лучший конкурент: (SE / CBAM / Late) с mAP@50 = ...
- Прирост CGFM: Δ = ... п.п.
- Permutation test p-value: ...
- Значимо на уровне α=0.05: да / нет

### Аблация — оптимальная конфигурация CGFM

Победитель: ... (уровни: ..., энкодер: ...)
Обоснование: ...

### Переносимость

- RT-DETR + CGFM vs RT-DETR baseline: Δ mAP@50 = ... п.п. (p = ...)
- Соответствие прироста YOLOv12 + CGFM: да/нет → тезис о переносимости подтверждён/опровергнут

### Анализ γ-распределений

- Ключевые наблюдения по γ-тепловым картам (2–3 предложения)
- Кластеризация t-SNE контекстных эмбеддингов: видимая структура по ... (условиям съёмки / классам болезней / ...)

## Готовые таблицы и фигуры для диплома

Перечень артефактов, готовых к прямой вставке в `диплом/chapter_4.md` (см. `for_diploma.md`).

## Проблемы / Замечания

## Артефакты
(полный перечень директории code/results/task_18/)
```
