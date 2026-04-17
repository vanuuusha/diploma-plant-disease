# Task 13: Самореферентные бейзлайны — SE-Neck и CBAM-Neck YOLOv12

## Статус
pending (после `task_12`)

## Цель

Обучить два бейзлайна с самореферентными механизмами внимания в neck YOLOv12:
1. **SE-Neck** — канальная самоатенция (Hu et al., 2018).
2. **CBAM-Neck** — канальная + пространственная самоатенция (Woo et al., 2018).

Оба метода берут сигнал модуляции из тех же признаков, которые модулируют (внутренний контекст). Их назначение в главе 4 — показать, что самореферентная атенция не решает задачу контекстной адаптации, и обосновать необходимость внешнего сигнала, реализованного в CGFM (`task_15`).

## Общий протокол

Все гиперпараметры, датасет, формат артефактов, имена конфигураций — в `code/docs/chapter4_protocol.md`. Используется единственный датасет `code/data/dataset_final/` (aug_diffusion).

## Шаги

1. **Создать ноутбук** `code/notebooks/chapter4_self_attention.ipynb` со структурой:
   - §0. Импорты, seed, проверка GPU, загрузка модулей из `code/models/chapter4/`.
   - §1. Общие утилиты (логирование, FPS-замер, рендер предсказаний, сохранение артефактов в `code/results/task_13/`).
   - §2. Вариант **SE-Neck**.
   - §3. Вариант **CBAM-Neck**.
   - §4. Сводная таблица 2 × 7 + экспорт в `code/results/task_13/summary.csv`.
2. **Подготовить модель SE-Neck**:
   - Загрузить `yolo12m.pt` (Ultralytics), заморозить ничего не нужно — обучение end-to-end.
   - Через `wrap_neck_with(model, block_factory=SEBlock, context_encoder=None)` вставить SE-блоки после каждого уровня P3/P4/P5.
   - Проверить `print(model.model)` — убедиться, что SE-блоки встали в правильные места.
3. **Обучить SE-Neck**:
   - `data=code/data/dataset_final/data.yaml`
   - гиперпараметры из `chapter4_protocol.md` §3 (epochs=100, patience=15, imgsz=640, batch=16, seed=42)
   - встроенные аугментации Ultralytics полностью отключены (hsv_*, degrees, translate, scale, shear, perspective, flipud, fliplr, mosaic, mixup, copy_paste = 0)
   - `project='code/results/task_13'`, `name='yolov12_se_neck'`
   - `verbose=True`, прогресс по эпохам виден в ноутбуке
4. **Артефакты для SE-Neck** — собрать все по §5 протокола главы 4:
   - `metrics.csv`, `learning_curves.png`, `confusion_matrix.png`, `per_class_map.csv`, `predictions_examples/` (те же 10 файлов из `code/docs/chapter3_qualitative_sample.txt`), `fps_measurement.json` (100 прогонов после 20 warm-up), `best.pt`, `train.log`, `param_count.json`.
5. **Коммит после SE-Neck** — `chapter4 self_attention: se_neck done` + `git push`.
6. **Повторить шаги 2–5 для CBAM-Neck**:
   - Свежая загрузка `yolo12m.pt` (чтобы не тащить SE-состояние).
   - `wrap_neck_with(model, block_factory=CBAMBlock, context_encoder=None)`.
   - `name='yolov12_cbam_neck'`.
   - Коммит `chapter4 self_attention: cbam_neck done` + `git push`.
7. **Построить сводную таблицу** в последней ячейке ноутбука — сравнение SE-Neck vs CBAM-Neck + строка baseline (загружается из `code/results/task_12/yolov12_baseline/`).
   - Колонки: `config, mAP@50, mAP@50-95, Precision, Recall, FPS, Params_M, GFLOPs, Epochs`
   - Экспорт в `code/results/task_13/summary.csv`.
8. **Качественная галерея** — `code/results/task_13/qualitative_comparison.png`: 10 тестовых изображений × 3 колонки (baseline / SE / CBAM), прямое сравнение предсказаний.

## Специфика

- **SE-Neck и CBAM-Neck никогда не сочетаются.** Это два независимых эксперимента, каждый — свой прогон с нуля (ни pretraining SE → CBAM, ни ансамбль).
- **Оверхед по параметрам должен быть малым.** SE добавляет ~0.1–0.3 % параметров на уровень, CBAM — до 0.5 %. Если увидите, что число параметров выросло заметно больше — искать ошибку интеграции.
- **Сходимость самоатенции быстрая** — обычно 30–50 эпох, patience=15 должен срабатывать раньше 100. Если реально нужно 100+ эпох — записать замечание в RESULT.md.
- **FPS-деградация ожидается минимальной** (1–3 %): SE/CBAM легковесны.

## Входные данные

- `code/data/dataset_final/data.yaml`
- `code/models/chapter4/se_block.py`, `cbam_block.py`, `yolov12_patch.py` (из task_12)
- `code/results/task_12/yolov12_baseline/` (для справочной строки в сводной таблице)
- Предобученные веса `yolo12m.pt` (скачиваются автоматически)

## Ожидаемый результат

- Ноутбук `code/notebooks/chapter4_self_attention.ipynb` с сохранёнными выводами двух прогонов.
- `code/results/task_13/yolov12_se_neck/` и `code/results/task_13/yolov12_cbam_neck/` с полным набором артефактов из §5 протокола.
- `code/results/task_13/summary.csv` и `code/results/task_13/qualitative_comparison.png`.
- Заполненный `RESULT.md`.

### Ожидаемое качественное поведение

- SE-Neck даёт умеренный прирост mAP@50 (+0.5–1.5 п.п. к baseline) — самоатенция помогает, но ограниченно.
- CBAM-Neck — обычно немного лучше SE (+0.3–0.8 п.п. к SE), за счёт пространственной ветви.
- Precision может вырасти сильнее Recall (самоатенция подавляет неинформативные каналы).
- На per-class mAP: возможен прирост на частых классах (Листовая ржавчина, Пиренофороз) и близкая к нулю дельта на редких (Повреждение заморозками, Недостаток N) — именно этот паттерн указывает на неспособность самоатенции адаптироваться к контексту сложных случаев.

Если результаты сильно отличаются (например, SE/CBAM дают −2 п.п. mAP@50) — зафиксировать в RESULT.md. Задача — честная оценка методов как есть, не подгонка под ожидания.

## Результат записать в

`code/tasks/task_13/RESULT.md` по следующему шаблону:

```markdown
# Result: Task 13 — Самореферентные бейзлайны SE-Neck и CBAM-Neck

## Статус
done | partial | failed

## Что было сделано
Пайплайн, места вставки блоков в neck, фактический batch, время обучения.

## Результаты

| Конфигурация | mAP@50 | mAP@50-95 | Precision | Recall | FPS | Params_M | GFLOPs | Эпох |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (из task_12) | | | | | | | | — |
| yolov12_se_neck | | | | | | | | |
| yolov12_cbam_neck | | | | | | | | |

### Прирост относительно baseline

| Конфигурация | Δ mAP@50, п.п. | Δ mAP@50-95, п.п. | Δ FPS, % |
|---|---:|---:|---:|
| SE-Neck | | | |
| CBAM-Neck | | | |

### Per-class mAP@50 на лучшем из двух
(таблица по 9 классам)

## Проблемы / Замечания

## Артефакты
- `code/notebooks/chapter4_self_attention.ipynb`
- `code/results/task_13/yolov12_se_neck/`
- `code/results/task_13/yolov12_cbam_neck/`
- `code/results/task_13/summary.csv`
- `code/results/task_13/qualitative_comparison.png`
```
