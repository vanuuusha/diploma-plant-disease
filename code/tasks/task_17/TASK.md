# Task 17: Переносимость CGFM на RT-DETR

## Статус
pending (после `task_12` и `task_15`; не зависит от task_13/14/16)

## Цель

Проверить, что предложенный подход CGFM не привязан к архитектуре YOLOv12 и переносится на принципиально иной детектор — трансформерный RT-DETR с гибридным энкодером. Это подраздел «4.3 Переносимость подхода» в главе 4 дипломной работы. Цель — **не** оптимизировать RT-DETR до максимума, а продемонстрировать, что FiLM-модуляция от внешнего контекстного энкодера даёт прирост и в другой архитектуре → метод общий, а не частный приём для YOLO.

Согласно рекомендации научного руководителя, ограничить подраздел **1–2 конфигурациями RT-DETR**:
1. **RT-DETR Baseline** — reference, результат уже есть в `task_08/rtdetr_aug_diffusion` (новое обучение не требуется).
2. **RT-DETR + CGFM** — FiLM-модуляция на выходах гибридного энкодера (S3, S4, S5 после CCFM).

## Общий протокол

Гиперпараметры, датасет, формат артефактов — в `code/docs/chapter4_protocol.md`. RT-DETR-ветви соответствует §3 того же документа. Используется `code/data/dataset_final/`.

## Шаги

1. **Создать ноутбук** `code/notebooks/chapter4_rtdetr_cgfm.ipynb`:
   - §0. Импорты, seed, проверка GPU.
   - §1. Загрузка reference baseline из `task_08/rtdetr_aug_diffusion/` (артефакты копируются/симлинкаются в `code/results/task_17/rtdetr_baseline/`).
   - §2. Интеграция CGFM в RT-DETR.
   - §3. Обучение RT-DETR + CGFM.
   - §4. Сводная таблица + экспорт в `code/results/task_17/summary.csv`.

2. **Этап A — baseline reference**:
   - Скопировать полный набор артефактов `code/results/task_08/rtdetr_aug_diffusion/*` в `code/results/task_17/rtdetr_baseline/`.
   - Если не хватает каких-то артефактов (`per_class_map.csv`, `context_embeddings.npy` из MobileNetV3-Small) — дочинить: загрузить `best.pt`, пересчитать метрики и прогнать контекст-энкодер на test.

3. **Этап B — интеграция CGFM в RT-DETR**:
   - Фреймворк: **HuggingFace `transformers.RTDetrForObjectDetection`** (как в task_08).
   - Архитектура RT-DETR (Zhao et al., 2023):
     - Backbone (ResNet-50 или HGNet) → многоуровневые признаки S3, S4, S5.
     - **Hybrid Encoder** — Transformer-энкодер с cross-scale feature fusion (CCFM): объединяет S3/S4/S5 → выдаёт уточнённые $\tilde{S}_3, \tilde{S}_4, \tilde{S}_5$.
     - Decoder — decoder queries → финальные боксы.
   - Точка вставки FiLM — **выходы CCFM** (до decoder queries). Это аналог neck в YOLO.
   - Реализовать `code/models/chapter4/rtdetr_patch.py`:
     ```python
     def wrap_rtdetr_encoder_with_cgfm(model, context_encoder):
         """
         Оборачивает CCFM-выходы RT-DETR FiLM-слоями, использующими
         вектор контекста от отдельного энкодера всей сцены.
         """
     ```
   - Предусмотреть обработку того, что в HuggingFace RT-DETR модули имеют разные имена — найти CCFM-выходы можно по `model.model.decoder.encoder_hidden_states` или аналогичному полю (уточнить по docstring модели).

4. **Этап C — обучение RT-DETR + CGFM**:
   - Контекстный энкодер — `MobileNetV3-Small` (та же конфигурация, что в task_15), для прямого сопоставления с YOLOv12-CGFM.
   - Гиперпараметры — как в `task_08`/`chapter4_protocol.md` §3: `epochs=100`, `patience=15`, `imgsz=640`, `batch=8`, AdamW lr=1e-4 (head) / 1e-5 (backbone + ResNet) / 5e-4 (FiLM + контекст-энкодер, свежие слои), `seed=42`.
   - Warm-up: первые 3 эпохи — заморозить RT-DETR (только FiLM и контекст-энкодер обучаются), затем end-to-end.
   - Встроенные аугментации HuggingFace processor — отключены (только resize и нормализация).
   - `project='code/results/task_17'`, `name='rtdetr_cgfm'`.

5. **Артефакты для RT-DETR + CGFM** — стандартный набор из §5 протокола главы 4:
   - `metrics.csv`, `learning_curves.png`, `confusion_matrix.png`, `per_class_map.csv`, `predictions_examples/`, `fps_measurement.json` (с декомпозицией), `best.pt`, `train.log`, `param_count.json`, `context_embeddings.npy`.

6. **Коммит** — `chapter4 rtdetr_cgfm: done` + `git push`.

7. **Сводная таблица** (§4 ноутбука):
   - Строки: `rtdetr_baseline`, `rtdetr_cgfm`.
   - Колонки: `mAP@50`, `mAP@50-95`, `Precision`, `Recall`, `FPS`, `Params_M`, `Δ mAP@50`, `Δ FPS %`.
   - Экспорт в `code/results/task_17/summary.csv`.

8. **Качественное сравнение** — `code/results/task_17/qualitative_comparison.png`:
   - 10 тестовых файлов × 2 колонки (baseline RT-DETR, RT-DETR + CGFM).
   - Подчеркнуть случаи, где CGFM исправляет ошибки baseline.

## Специфика

- **Гибридный энкодер RT-DETR — не то же самое, что neck YOLO.** В YOLO neck — чисто свёрточная структура FPN/PAN. В RT-DETR CCFM — Transformer с cross-attention между уровнями. Но выходы CCFM — это всё те же многоуровневые карты признаков, к которым FiLM применим без изменений. Это ключевой архитектурный аргумент для диплома: FiLM модулирует **выходы слоя**, не предполагая его внутреннюю структуру.
- **Сходимость RT-DETR медленнее, чем YOLOv12**, особенно на ограниченном бюджете эпох. Если на 100 эпохах с `patience=15` явно не сошлась — зафиксировать в RESULT.md, не продлять.
- **FPS-просадка** ожидается немного больше, чем у YOLOv12+CGFM, т.к. RT-DETR сам по себе тяжелее (8–18 FPS baseline, контекст-энкодер добавляет ещё ~2 мс).
- **Ограниченность эксперимента**: это не полная аблация RT-DETR (только 1 конфигурация CGFM), а демонстрация переносимости. Полная аблация на RT-DETR вынесена в «перспективы» заключения дипломной работы.
- **Не обучать RT-DETR baseline повторно.** Использовать веса и метрики из task_08. Единственная новая обучаемая модель — RT-DETR + CGFM.

## Входные данные

- `code/data/dataset_final/data.yaml`
- `code/results/task_08/rtdetr_aug_diffusion/` (baseline reference)
- `code/models/chapter4/film_layer.py`, `context_encoder.py` (из task_12)
- Чекпойнт HuggingFace RT-DETR (тот же, что использовался в task_08; имя указать в RESULT.md).

## Ожидаемый результат

- Ноутбук `code/notebooks/chapter4_rtdetr_cgfm.ipynb` с выводами одного прогона.
- `code/models/chapter4/rtdetr_patch.py` — реализация интеграции CGFM в RT-DETR.
- `code/results/task_17/rtdetr_baseline/` (скопированный из task_08) и `code/results/task_17/rtdetr_cgfm/` (новый).
- `code/results/task_17/summary.csv` (2 строки) и `code/results/task_17/qualitative_comparison.png`.
- Заполненный `RESULT.md` с акцентом: «метод переносится, прирост сохраняется».

### Ожидаемое качественное поведение

- Прирост mAP@50 от CGFM: +1.5–3 п.п. (возможно меньше, чем на YOLOv12, т.к. RT-DETR уже имеет встроенный global-attention и часть эффекта «контекста» уже присутствует в архитектуре).
- Прирост mAP@50-95 — аналогичный порядок.
- FPS-деградация 15–25 %.
- На per-class mAP — прирост сохраняется на редких классах (как и в YOLOv12), что подтверждает универсальность механизма.

**Главный тезис раздела главы 4 по итогу этой задачи**: «Предложенный подход не зависит от архитектуры детектора. Модуляция выходов feature-extraction-ступени внешним контекстом даёт прирост как в свёрточной FPN/PAN (YOLOv12), так и в трансформерном гибридном энкодере (RT-DETR), что говорит о фундаментальной природе выявленной проблемы — отсутствия внешнего контекстного сигнала в детекторах — а не о частной уязвимости конкретной архитектуры».

## Результат записать в

`code/tasks/task_17/RESULT.md` по шаблону:

```markdown
# Result: Task 17 — Переносимость CGFM на RT-DETR

## Статус
done | partial | failed

## Что было сделано
Способ интеграции FiLM на выходы CCFM HuggingFace RT-DETR. Warm-up. Фактический batch. Сходимость.

## Результаты

| Конфигурация | mAP@50 | mAP@50-95 | Precision | Recall | FPS | Params_M | Эпох |
|---|---:|---:|---:|---:|---:|---:|---:|
| rtdetr_baseline (task_08 ref) | | | | | | | — |
| rtdetr_cgfm | | | | | | | |

### Прирост от CGFM на RT-DETR

| Метрика | Значение |
|---|---:|
| Δ mAP@50, п.п. | |
| Δ mAP@50-95, п.п. | |
| Δ Precision, п.п. | |
| Δ Recall, п.п. | |
| Δ FPS, % | |

### Сопоставление с YOLOv12-CGFM

| Детектор | Δ mAP@50 от CGFM, п.п. | Δ FPS, % |
|---|---:|---:|
| YOLOv12 (task_15) | | |
| RT-DETR (task_17) | | |

### Per-class mAP@50

## Проблемы / Замечания
- Особенности интеграции FiLM в HuggingFace RT-DETR.
- Сходимость за 100 эпох (сошлась / не сошлась).

## Артефакты
- `code/notebooks/chapter4_rtdetr_cgfm.ipynb`
- `code/models/chapter4/rtdetr_patch.py`
- `code/results/task_17/rtdetr_baseline/`
- `code/results/task_17/rtdetr_cgfm/`
- `code/results/task_17/summary.csv`
- `code/results/task_17/qualitative_comparison.png`
```
