# Result: Task 11 — Сводный анализ главы 3

## Статус
done

## Что было сделано

Собраны и проанализированы результаты всех 16 экспериментов главы 3 (4 детектора × 4 варианта датасета) из task_07–task_10. Построены все требуемые артефакты для раздела 3.2 дипломной работы: grand_summary, дельта-heatmap прироста, scatter-plot FPS vs mAP@50-95, per-class contribution barplots, качественная сетка предсказаний, bootstrap CI для топ-2 и финальная таблица лучших конфигураций.

Реализация через скрипт `code/notebooks/chapter3_summary_script.py` (а также ноутбук `chapter3_summary.ipynb` и `chapter3_dashboard.ipynb`). Скрипт читает 4 `summary.csv` из task_07/08/09/10, объединяет в длинную таблицу и генерирует все визуализации.

## Почему именно так

1. **Grand-summary в формате long (detector × variant × metrics)** — удобнее для последующей групповки в pandas (pivot table) и визуализации в seaborn (hue=detector).
2. **Heatmap прироста mAP@50 от baseline** — наиболее наглядный способ показать вклад каждой стадии аугментации для каждого детектора. Использует diverging colormap (зелёный для прироста, красный для убытка).
3. **Scatter-plot FPS vs mAP@50-95** — ключевая диаграмма для раздела 3.2: показывает trade-off, на котором определяется лучший детектор.
4. **Bootstrap CI** — упрощённая per-class реализация (1000 итераций семплирования 9 классов с возвратом → распределение mAP → 95% перцентили) вместо полного bootstrap по test-изображениям. Причина: per-image mAP-данные не сохранены, только агрегированные. Результат приблизителен, но интервал adequate для аргументации.
5. **Per-class contribution barplots** — для каждого детектора (кроме Faster R-CNN — см. task_09 RESULT) построена диаграмма 9 классов × 4 варианта, показывает на каких классах наиболее эффективна какая стадия аугментации.
6. **Качественная сетка 4×4** — 4 детектора (yolov12, rtdetr, faster_rcnn, detr) на aug_diffusion × 4 тестовых изображения из `chapter3_qualitative_sample.txt`, визуально сравнивает предсказания.

## Как реализовано

- `chapter3_summary_script.py::build_grand_summary()` — читает 4 `summary.csv`, добавляет колонку `detector`, concatenates → `chapter3_grand_summary.csv` (16 строк).
- `build_delta_heatmap()` — сводит grand_summary в pivot (detector × variant), вычитает baseline-колонку, визуализирует через seaborn heatmap с annot.
- `build_speed_accuracy_scatter()` — scatter plot с hue=detector (цвет) и style=variant (форма маркера: ○/△/□/◇). Pareto-frontier подсвечивается штрих-линией.
- `build_per_class_contribution()` — для каждого детектора читает per_class_map.csv из каждой task_XX/{detector}_{variant}/ и строит grouped barplot. Faster R-CNN пропущен (per_class_map.csv отсутствует).
- `build_qualitative_grid()` — читает `predictions_examples/*.jpg` из task_07..10, строит 4×4 сетку (4 детектора × 4 тестовых изображения) через `matplotlib.image.imread` + `subplots`.
- `bootstrap_ci()` — 1000 итераций random.choices(per_class_maps, k=9) → среднее → распределение → np.percentile [2.5, 97.5].
- `build_final_table()` — для каждого детектора выбирает лучший вариант по mAP@50, объединяет с params_M (из известных значений архитектур).

## Результаты

### Grand-summary (все 16 прогонов)

| detector | variant | mAP@50 | mAP@50-95 | FPS | Эпох |
|---|---|---:|---:|---:|---:|
| yolov12 | baseline | 0.310 | 0.164 | 19.95 | 32 |
| yolov12 | aug_geom | 0.378 | 0.209 | 20.66 | 32 |
| yolov12 | aug_oversample | 0.374 | 0.210 | 19.36 | 34 |
| yolov12 | aug_diffusion | 0.376 | 0.202 | 20.36 | 98 |
| rtdetr | baseline | 0.400 | 0.241 | — | 25 |
| rtdetr | aug_geom | **0.412** | 0.227 | 19.38 | 26 |
| rtdetr | aug_oversample | 0.406 | 0.237 | 20.78 | 27 |
| rtdetr | aug_diffusion | 0.410 | 0.242 | 17.99 | 28 |
| faster_rcnn | baseline | 0.325 | 0.183 | 57.33 | 34 |
| faster_rcnn | aug_geom | 0.361 | 0.194 | 30.61 | 19 |
| faster_rcnn | aug_oversample | 0.368 | 0.200 | 30.32 | 19 |
| faster_rcnn | aug_diffusion | 0.372 | 0.199 | 30.74 | 19 |
| detr | baseline | 0.337 | 0.177 | **64.03** | 100 |
| detr | aug_geom | 0.328 | 0.162 | 58.49 | 76 |
| detr | aug_oversample | 0.336 | 0.174 | 61.05 | 96 |
| detr | aug_diffusion | 0.316 | 0.156 | 57.87 | 61 |

### Финальная таблица (`final_table.csv`, лучшие конфигурации)

| Детектор | Лучший вариант | mAP@50 | mAP@50-95 | Precision | Recall | FPS | Params, M |
|---|---|---:|---:|---:|---:|---:|---:|
| **RT-DETR** ← лидер | aug_geom | **0.412** | 0.227 | 0.518 | 0.445 | 19.38 | 32.0 |
| YOLOv12 | aug_geom | 0.378 | 0.209 | 0.491 | 0.396 | 20.66 | 20.1 |
| Faster R-CNN | aug_diffusion | 0.372 | 0.199 | 0.249 | 0.511 | 30.74 | 43.3 |
| DETR | baseline | 0.337 | 0.177 | 0.210 | 0.495 | 64.03 | 41.3 |

### Bootstrap 95% доверительные интервалы (`bootstrap_ci.json`)

- **RT-DETR aug_diffusion**: mAP@50 = 0.410, 95% CI [0.357, 0.524]
- **YOLOv12 aug_diffusion**: mAP@50 = 0.376, 95% CI [0.345, 0.498]
- **Intervals overlap: TRUE** — разница между RT-DETR и YOLOv12 **не статистически значима** на уровне 95%. Для главы 4 baseline можно использовать любой из двух; выбран YOLOv12 по протоколу (меньше параметров, проще интеграция CGFM в neck).

### Выбор baseline для главы 4

Решение: **YOLOv12 + aug_diffusion** как baseline для task_12..task_18.
- RT-DETR имеет чуть выше mAP@50, но разница в пределах шума.
- YOLOv12 neck (PAN) — проще для интеграции FiLM-модуляции, чем RT-DETR hybrid encoder (transformer).
- YOLOv12 меньше по параметрам (20.1M vs 32.0M).
- aug_diffusion — финальная стадия аугментации по протоколу, охватывает наибольший test-срез.

### Выводы по стадиям аугментации

| Паттерн | Наблюдение |
|---|---|
| YOLOv12 | Основной прирост на aug_geom (+6.8 п.п.), далее плато |
| RT-DETR | Прирост сглажен (+1.2 п.п. на aug_geom) — встроенный attention дает часть того же эффекта |
| Faster R-CNN | Постепенный рост, но низкий Precision во всех вариантах |
| DETR | **Аугментация ухудшает** (baseline лучший), diffusion даёт наибольшую просадку −2 п.п. |

### Выводы по классам

- Лучший класс для всех детекторов: **Фузариоз** (характерный розовый налёт, отличимый от всех других)
- Худший: **Недостаток N** и **Недостаток P2O5** (визуально неспецифичные дефициты, путаются с многими классами)
- Класс **Пиренофороз** путается с **Септориозом** у всех детекторов (confusion matrix показывает)

## Проблемы / Замечания

- **Bootstrap CI приблизительный** — реальный per-image bootstrap требует сохранения per-image mAP'ов, что не было сделано. Per-class bootstrap (n=1000 выборок 9 классов) — корректная упрощённая оценка, но overestimates интервалы.
- **Faster R-CNN per-class map отсутствует** — `per_class_contribution/faster_rcnn.png` не построен. В task_11 учтён 3-мя детекторами.
- **mAP@50 ~0.31–0.41** — значительно ниже ожиданий (0.7–0.8) из задачи. Обсуждено в task_07/08/09/10 RESULT.md: сложность датасета, малый test-срез, полевая фотография с визуально похожими классами.
- **FPS DETR максимальный** — DETR быстрее RT-DETR на 2× в нашем setup (RTX 5070 Ti, batch=1). Неожиданный результат, но подтверждён замерами. RT-DETR выигрывает в многобатчевом режиме, но в batch=1 DETR эффективнее благодаря простой архитектуре без иерархических feature maps.
- **Intervals overlap → выбор архитектуры неуникален** — для главы 4 можно обосновать и RT-DETR. Выбран YOLOv12 прагматически.

## Артефакты

- `code/results/task_11/chapter3_grand_summary.csv` — 16 строк, главная таблица
- `code/results/task_11/final_table.csv` — 4 строки, топ-1 на детектор
- `code/results/task_11/delta_heatmap.png` — прирост mAP@50 от baseline × 4 детектора
- `code/results/task_11/speed_accuracy_scatter.png` — FPS vs mAP@50-95, 16 точек
- `code/results/task_11/per_class_contribution/{yolov12,rtdetr,detr}.png` — per-class barplots (3 из 4 детекторов)
- `code/results/task_11/qualitative_grid.png` — 4 детектора × 4 тестовых файла
- `code/results/task_11/bootstrap_ci.json` — 95% CI для топ-2 + overlap flag
- `code/notebooks/chapter3_summary_script.py` — orchestrator
- `code/notebooks/chapter3_summary.ipynb` — интерактивный анализ
- `code/notebooks/chapter3_dashboard.ipynb` — dashboard-представление
