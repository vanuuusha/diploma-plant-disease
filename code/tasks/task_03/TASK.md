# Task 03: EDA датасета — полная статистика и визуализации

## Статус
pending

## Цель
Провести exploratory data analysis (EDA) датасета: собрать все статистики, построить визуализации, сохранить графики для использования в тексте дипломной работы.

## Зависимости
Выполнить после task_02 (конвертация в YOLO-формат).

## Входные данные
- `code/data/dataset/` — датасет в YOLO-формате (результат task_02)
- `code/data/dataset/data.yaml` — маппинг классов

## Шаги

### 1. Общая статистика датасета
Собрать и вывести:
- Общее число изображений (total и по сплитам)
- Общее число аннотаций (bounding boxes)
- Число классов
- Среднее/медианное/min/max число bbox на изображение
- Разрешения изображений: min, max, среднее, распределение

### 2. Распределение классов (дисбаланс)
- Подсчитать число аннотаций для каждого класса (на всём датасете и по сплитам)
- Вычислить imbalance ratio = max_class / min_class
- Вычислить коэффициент вариации числа аннотаций

### 3. Визуализации — сохранить каждую как PNG (300 dpi)

**3.1 Гистограмма распределения классов**
- Горизонтальная столбчатая диаграмма
- Подписи: название класса + количество аннотаций
- Отсортировать по убыванию
- Сохранить: `code/results/task_03/class_distribution.png`

**3.2 Гистограмма распределения классов по сплитам**
- Grouped bar chart: train / val / test рядом для каждого класса
- Сохранить: `code/results/task_03/class_distribution_by_split.png`

**3.3 Распределение числа bbox на изображение**
- Гистограмма (binned)
- Сохранить: `code/results/task_03/bbox_per_image.png`

**3.4 Распределение размеров bbox**
- Scatter plot: width vs height (нормализованные)
- Цвет точек по классу
- Сохранить: `code/results/task_03/bbox_sizes.png`

**3.5 Heatmap позиций bbox на изображении**
- Тепловая карта центров bbox на нормализованном холсте [0,1]×[0,1]
- Сохранить: `code/results/task_03/bbox_heatmap.png`

**3.6 Распределение разрешений изображений**
- Scatter plot: width vs height в пикселях
- Сохранить: `code/results/task_03/image_resolutions.png`

**3.7 Примеры изображений с bbox для каждого класса**
- Для каждого класса выбрать 1 репрезентативное изображение
- Нарисовать bounding boxes на изображении с подписями
- Сохранить grid: `code/results/task_03/class_examples_grid.png`
- Также сохранить отдельно каждый пример: `code/results/task_03/examples/class_{name}.png`

**3.8 Boxplot площадей bbox по классам**
- Сохранить: `code/results/task_03/bbox_area_by_class.png`

### 4. Сводная таблица
Создать CSV-файл со всеми числовыми результатами:
- `code/results/task_03/dataset_summary.csv`

### 5. Оформление графиков
- Все графики: figsize не менее (10, 6), шрифт 12+
- Заголовки на русском
- Легенда при необходимости
- Цветовая палитра: использовать seaborn "Set2" или "tab10"
- Все графики сохранять в 300 dpi

## Ожидаемый результат
```
code/results/task_03/
├── class_distribution.png
├── class_distribution_by_split.png
├── bbox_per_image.png
├── bbox_sizes.png
├── bbox_heatmap.png
├── image_resolutions.png
├── class_examples_grid.png
├── bbox_area_by_class.png
├── examples/
│   ├── class_пиренофороз.png
│   ├── class_септориоз.png
│   └── ... (по одному на класс)
└── dataset_summary.csv
```

- Скрипт EDA сохранить в `code/scripts/eda_dataset.py`
- Все числа и факты записать в RESULT.md — они будут использоваться в тексте главы 2 диплома

## Результат записать в
`code/tasks/task_03/RESULT.md`
