# Task 16: Аблация CGFM — уровни FPN и контекстный энкодер

## Статус
pending (после `task_15`)

## Цель

Провести аблационное исследование основного метода CGFM, чтобы понять:
1. **На каких уровнях FPN** CGFM даёт наибольший вклад: только P5 (глобальный семантический уровень) vs только P3 (мелкие детали) vs все три (базовая конфигурация из task_15).
2. **Какой контекстный энкодер** оптимален по трейдоффу «качество ↔ скорость»: MobileNetV3-Small (~1.5M, быстрый) vs EfficientNet-B0 (~5.3M, средний) vs ViT-Tiny (~5.7M, трансформерный).

Базовая конфигурация task_15 — CGFM(P3+P4+P5) с MobileNetV3-Small. Аблация отвечает на вопросы:
- Нужны ли все три уровня, или достаточно одного?
- Стоит ли более тяжёлый/экспрессивный энкодер дополнительного прироста?

## Общий протокол

Гиперпараметры, датасет, формат артефактов — в `code/docs/chapter4_protocol.md`. Базовая стратегия warm-up и формат артефактов — как в task_15.

Чтобы избежать факториального взрыва (3 уровней × 3 энкодера = 9 прогонов), аблация разделена на **две независимые ветви**:

**Ветвь A — уровни FPN** (энкодер фиксирован = MobileNetV3-Small):
- `yolov12_cgfm_abl_p5only` — FiLM только на P5
- `yolov12_cgfm_abl_p3only` — FiLM только на P3
- (Базовая конфигурация P3+P4+P5 — уже обучена в task_15, не переобучается)

**Ветвь B — контекстный энкодер** (уровни фиксированы = все три, результат task_15 — как отсчёт):
- `yolov12_cgfm_abl_effb0` — EfficientNet-B0 как энкодер
- `yolov12_cgfm_abl_vittiny` — ViT-Tiny как энкодер
- (MobileNetV3-Small — уже обучена в task_15)

**Итого в этой задаче: 4 новых прогона**.

## Шаги

1. **Создать ноутбук** `code/notebooks/chapter4_cgfm_ablation.ipynb`:
   - §0. Импорты.
   - §1. Общие утилиты (переиспользуются из task_15).
   - §2. Ветвь A — уровни FPN (2 прогона).
   - §3. Ветвь B — контекстный энкодер (2 прогона).
   - §4. Сводная таблица аблации + экспорт в `code/results/task_16/summary.csv`.

2. **Ветвь A — P5-only**:
   - Загрузить `yolo12m.pt`.
   - `ContextEncoder(backbone='mobilenetv3_small_100')`.
   - `wrap_neck_with(..., levels=['P5'])` — вставить FiLM **только** на самый глубокий уровень.
   - Гиперпараметры и warm-up — как в task_15.
   - Артефакты → `code/results/task_16/yolov12_cgfm_abl_p5only/`.
   - Коммит `chapter4 cgfm_abl: p5only done` + `git push`.

3. **Ветвь A — P3-only**:
   - Аналогично, но `wrap_neck_with(..., levels=['P3'])`.
   - Артефакты → `code/results/task_16/yolov12_cgfm_abl_p3only/`.
   - Коммит `chapter4 cgfm_abl: p3only done` + `git push`.

4. **Ветвь B — EfficientNet-B0**:
   - Загрузить `yolo12m.pt`.
   - `ContextEncoder(backbone='efficientnet_b0', out_dim=256, pretrained=True)`.
   - `wrap_neck_with(..., levels=['P3', 'P4', 'P5'])`.
   - Артефакты → `code/results/task_16/yolov12_cgfm_abl_effb0/`.
   - Коммит `chapter4 cgfm_abl: effb0 done` + `git push`.

5. **Ветвь B — ViT-Tiny**:
   - `ContextEncoder(backbone='vit_tiny_patch16_224', out_dim=256, pretrained=True)`.
   - Артефакты → `code/results/task_16/yolov12_cgfm_abl_vittiny/`.
   - Коммит `chapter4 cgfm_abl: vittiny done` + `git push`.

6. **Артефакты для каждого прогона** — стандартный набор из §5 протокола главы 4 (без `gamma_maps/` и `gamma_histograms/` — это слишком много визуализаций, достаточно одного набора из task_15 для основного метода). Но сохранить:
   - `context_embeddings.npy` (нужны для t-SNE в task_18 — сравнение эмбеддингов разных энкодеров)
   - `fps_measurement.json` с декомпозицией латентности
   - `param_count.json`

7. **Сводная таблица аблации** — одна таблица 5 × N (5 конфигураций × метрики), включая reference-строку `yolov12_cgfm` (MobileNetV3 + P3+P4+P5) из task_15:

```
| Конфиг | Уровни | Энкодер | mAP@50 | mAP@50-95 | Precision | Recall | FPS | Params_M |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cgfm (task_15) | P3+P4+P5 | MobileNetV3-Small | | | | | | |
| cgfm_abl_p5only | P5 | MobileNetV3-Small | | | | | | |
| cgfm_abl_p3only | P3 | MobileNetV3-Small | | | | | | |
| cgfm_abl_effb0 | P3+P4+P5 | EfficientNet-B0 | | | | | | |
| cgfm_abl_vittiny | P3+P4+P5 | ViT-Tiny | | | | | | |
```

8. **Выбор оптимальной конфигурации** — в RESULT.md обосновать выбор (mAP vs FPS Pareto-optimality). Итоговая «рекомендуемая конфигурация» будет использоваться в главе 5 (программная реализация) и упомянута в главе 4.4 дипломной работы как итог аблации.

## Специфика

- **Все 4 прогона обучаются с warm-up** (этапы 0–2 + 3–99), как в task_15.
- **ViT-Tiny требует входа 224×224** (фиксированный размер patch). MobileNetV3 и EfficientNet-B0 могут работать и с другими размерами, но для унификации всем энкодерам подаётся вход 224×224.
- **При переключении энкодера** пересоздаётся и FiLM-слой: его вход (`context_dim`) должен соответствовать `out_dim` контекстного энкодера. Если `out_dim` везде 256 — FiLM-слои идентичны.
- **Ожидаемая асимметрия**:
  - P5-only часто даёт худший Recall на мелких объектах (модулируется только глобальный масштаб).
  - P3-only — худшую Precision на больших объектах (глобальный контекст не доходит до них).
  - Все три уровня — лучший результат, но дороже по параметрам и чуть медленнее.
- **Не делать полное 3×3** — факториал дорог по времени (8–12 часов на прогон × 9 = 3–5 дней GPU), и научно оправданна только диагональная аблация «уровни при фиксированном энкодере» и «энкодер при фиксированных уровнях».

## Входные данные

- `code/data/dataset_final/data.yaml`
- `code/models/chapter4/` (из task_12)
- `code/results/task_15/yolov12_cgfm/` (для reference-строки в сводной таблице)

## Ожидаемый результат

- Ноутбук `code/notebooks/chapter4_cgfm_ablation.ipynb` с выводами 4 прогонов.
- `code/results/task_16/` с 4 поддиректориями и `summary.csv` (5 строк с учётом reference).
- `code/results/task_16/pareto_plot.png` — scatter plot «FPS vs mAP@50-95», 5 точек, подписи конфигураций.
- Заполненный `RESULT.md` с обоснованием выбора оптимальной конфигурации.

### Ожидаемое качественное поведение

- **P3+P4+P5 > P5-only > P3-only** по mAP@50-95 (ожидаемый порядок; P3-only обычно сильно хуже, т.к. глобальный контекст должен модулировать именно глубокие уровни).
- **MobileNetV3 ≈ EfficientNet-B0** по mAP, но EfficientNet медленнее → MobileNetV3 на Pareto-фронте.
- **ViT-Tiny** — теоретически более выразительный, но на данном размере train-сета (несколько тысяч) может недообучиться; ожидание — паритет с MobileNetV3 по mAP, но проигрыш по FPS.
- Pareto-optimum ожидается: **P3+P4+P5 + MobileNetV3-Small** (т.е. конфигурация task_15). Цель аблации — это подтвердить.

## Результат записать в

`code/tasks/task_16/RESULT.md` по шаблону:

```markdown
# Result: Task 16 — Аблация CGFM

## Статус
done | partial | failed

## Что было сделано
Краткое описание 4 прогонов. Фактический batch, сходимость.

## Результаты аблации

| Конфиг | Уровни | Энкодер | mAP@50 | mAP@50-95 | Precision | Recall | FPS | Params_M |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cgfm (reference) | P3+P4+P5 | MobileNetV3-Small | | | | | | |
| cgfm_abl_p5only | P5 | MobileNetV3-Small | | | | | | |
| cgfm_abl_p3only | P3 | MobileNetV3-Small | | | | | | |
| cgfm_abl_effb0 | P3+P4+P5 | EfficientNet-B0 | | | | | | |
| cgfm_abl_vittiny | P3+P4+P5 | ViT-Tiny | | | | | | |

### Выбор оптимальной конфигурации

Обоснование (2–3 абзаца): по mAP@50-95, по FPS, по Pareto-optimality. Итоговая рекомендация для главы 5.

## Проблемы / Замечания

## Артефакты
- `code/notebooks/chapter4_cgfm_ablation.ipynb`
- `code/results/task_16/yolov12_cgfm_abl_p5only/`
- `code/results/task_16/yolov12_cgfm_abl_p3only/`
- `code/results/task_16/yolov12_cgfm_abl_effb0/`
- `code/results/task_16/yolov12_cgfm_abl_vittiny/`
- `code/results/task_16/summary.csv`
- `code/results/task_16/pareto_plot.png`
```
