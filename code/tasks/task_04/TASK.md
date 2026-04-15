# Task 04: Геометрические и фотометрические аугментации (Albumentations)

## Статус
pending

## Цель
Реализовать пайплайн классических аугментаций через Albumentations для увеличения объёма обучающей выборки. Применить аугментации к **train** сплиту, сохранить аугментированный датасет, визуализировать примеры до/после.

## Зависимости
Выполнить после task_02 и task_03.

## Входные данные
- `code/data/dataset/train/` — тренировочный сплит в YOLO-формате
- `code/data/dataset/data.yaml`

## Контекст
Аугментации выполняют две задачи:
1. **Увеличение объёма данных** — больше разнообразия для лучшей генерализации
2. **Имитация полевых условий** — модель должна быть робастной к освещению, ракурсу, масштабу

Аугментации применяются **только к train**, val/test остаются нетронутыми.

## Шаги

### 1. Реализовать пайплайн аугментаций
Использовать `albumentations` с `BboxParams(format='yolo')`.

**Набор трансформаций (применять случайную комбинацию к каждому изображению):**

Геометрические:
- HorizontalFlip (p=0.5)
- VerticalFlip (p=0.3)
- RandomRotate90 (p=0.3)
- ShiftScaleRotate (shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.5)
- RandomResizedCrop (height=640, width=640, scale=(0.7, 1.0), p=0.4)

Фотометрические:
- RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, p=0.5)
- HueSaturationValue (hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4)
- GaussianBlur (blur_limit=(3, 7), p=0.3)
- GaussNoise (var_limit=(10, 50), p=0.3)
- CLAHE (clip_limit=4.0, p=0.3)
- RandomShadow (p=0.2) — имитация теней на поле
- RandomFog (p=0.15) — имитация утренней дымки
- ImageCompression (quality_lower=70, quality_upper=100, p=0.2) — имитация jpeg-артефактов

### 2. Стратегия аугментации
- Для каждого исходного изображения из train генерировать **2 аугментированных копии**
- Итого train будет ~×3 от оригинала (оригинал + 2 аугментации)
- Имена файлов: `{orig_name}_aug1.jpg`, `{orig_name}_aug2.jpg`
- Соответствующие .txt файлы с пересчитанными bbox

### 3. Валидация аугментированных данных
- Проверить, что все bbox остались в [0, 1] после трансформаций
- Удалить bbox с площадью < 0.001 (слишком мелкие после кропа)
- Удалить изображения, где все bbox были отфильтрованы
- Вывести: сколько изображений было, стало; сколько bbox потеряно

### 4. Визуализации

**4.1 Примеры до/после**
- Для 10 случайных изображений показать: оригинал (с bbox) + 2 аугментации (с bbox)
- Grid 10×3
- Сохранить: `code/results/task_04/augmentation_examples.png`

**4.2 Сравнение распределения классов до/после**
- Bar chart: оригинальный train vs аугментированный train
- Сохранить: `code/results/task_04/class_distribution_after_aug.png`

**4.3 Сравнение размеров bbox до/после**
- Два scatter plots рядом
- Сохранить: `code/results/task_04/bbox_sizes_comparison.png`

### 5. Сохранить результат
```
code/data/dataset_augmented/
├── data.yaml          # обновлённый (train путь → augmented train)
├── train/
│   ├── images/        # оригиналы + аугментации
│   └── labels/
├── val/               # БЕЗ ИЗМЕНЕНИЙ (скопировать/симлинк из dataset/)
│   ├── images/
│   └── labels/
└── test/              # БЕЗ ИЗМЕНЕНИЙ
    ├── images/
    └── labels/
```

## Ожидаемый результат
- Аугментированный датасет в `code/data/dataset_augmented/`
- Скрипт: `code/scripts/augment_classic.py`
- Визуализации в `code/results/task_04/`
- Все числа (было/стало) записать в RESULT.md

## Результат записать в
`code/tasks/task_04/RESULT.md`
