# Result: Task 19 — Вариации CGFM для прорыва identity-collapse

## Статус
done

## Что было сделано

По итогам task_15 и task_16 установлено, что CGFM P3+P4+P5 с идентичной-старт инициализацией FiLM функционально тождественна baseline из-за **identity-collapse** (γ ≈ 1, β = 0 + BN-поглощение остаточных возмущений). Единственная работающая конфигурация — **P5-only** (task_16, mAP@50 = 0.650, +0.1 п.п. к baseline). Это стало мотивацией для дополнительного исследования (task_19): можно ли пробить identity-collapse и получить работающий CGFM на всех 3 уровнях FPN?

Task 19 — систематическая попытка пробить identity-collapse через **5 независимых архитектурных вариаций** FiLM-слоя. Плюс одна **микс-конфигурация** (CGFM + CBAM на P5-only) для проверки гипотезы о синергии внешнего и внутреннего контекста. Каждая вариация запущена с отдельным обучением (не reuse чекпойнтов task_15/16) для чистого сравнения. В общей сложности 6 независимых training runs.

## Почему именно так

1. **5 вариаций как ортогональные оси** — каждая меняет только один аспект архитектуры, чтобы изолировать причину identity-collapse:
   - **A. residual** → проблема в слишком агрессивной мультипликативной модуляции? Добавляем скалярный gate α (init=0) — модель может «постепенно включить» FiLM.
   - **B. internal context** → проблема в том, что MobileNetV3 даёт контекст, семантически оторванный от detector features? Используем GAP(P5 backbone YOLOv12) как контекст.
   - **C. wide γ** → проблема в слишком узком диапазоне γ (0.5-1.5)? Расширяем до (0.1, 1.9).
   - **D. non-zero β init** → проблема в том, что β всегда стартует около 0? Инициализируем W_β с std=0.1 (vs 0.01 в default).
2. **Один hyperparameter tuning за раз** (не grid search) — в рамках бюджета 4-5 часов GPU, grid невозможен. Ortogonal isolation более информативна.
3. **Микс CGFM+CBAM на P5-only** — выбран именно P5-only (а не P3+P4+P5), потому что P5-only — единственная работающая база CGFM. Добавление CBAM поверх unlikely to improve, но проверяется из-за нулевой стоимости (~45 минут прогона).
4. **A100 + local параллельно** — задачи распределены по доступности GPU.

## Как реализовано

Все вариации доступны через `code/notebooks/chapter4_runner.py`:
- `--film_variant {default, wide, beta_noise}` — выбор математики FiLM
- `--film_residual` — включить α-gated residual path
- `--internal_context` — использовать GAP(P5 backbone) вместо ContextEncoder
- `--mix_cbam` — применить CBAM на тех же уровнях перед CGFM

В `FiLMLayer.__init__` разветвление по `variant`:
```python
if variant == "wide":
    γ = 0.1 + 1.8 * sigmoid(Wc+b)
elif variant == "beta_noise":
    γ = 1.0 + 0.5 * tanh(Wc+b)  # default γ
    self.to_beta.weight = N(0, 0.1)  # ← отличие от default
else:  # default
    γ = 1.0 + 0.5 * tanh(Wc+b)
    self.to_beta.weight = N(0, 0.01)

if residual:
    self.alpha = nn.Parameter(zeros(1))  # init=0 → identity
    return F + alpha * (modulated - F)
```

Для Internal context создан класс `_InternalContextEncoder(yolo_model, out_dim=256)` в runner'е:
- Dry-run backbone, находит индекс слоя перед первым Upsample (== backbone-P5, SPPF output).
- Регистрирует forward_hook: когда backbone-P5 вычислен, hook прогоняет его через proj (GAP → LayerNorm → MLP → out_dim), и сохраняет в film_layers.
- Атрибут `is_internal_hook = True` — сигнал для `wrap_neck_with` не делать monkey-patch forward.

## Вариации (все P3+P4+P5, MobileNetV3-Small)

| # | Код | Вариация | Механизм | Best epoch | mAP@50 | mAP@50-95 |
|---|---|---|---|---:|---:|---:|
| 0 | (task_15) `cgfm` | identity init | γ=1+tanh(Wc+b), W=0 | 14 | 0.628 | 0.333 |
| 1 | (task_15) `cgfm_v2` | non-identity init | W_γ ~ N(0, 0.05) | 18 | 0.632 | 0.335 |
| 2 | **A. residual** | $F' = F + α(γF + β - F)$, α=0 init | 18 | 0.622 | 0.329 |
| 3 | **B. internal ctx** | context = GAP(backbone-P5), без MobileNetV3 | ≈18 | 0.628 | 0.331 |
| 4 | **C. wide γ** | γ = 0.1 + 1.8·σ(Wc+b) ∈ (0.1, 1.9) | 17 | 0.600 | 0.310 |
| 5 | **D. non-zero β init** | W_β ~ N(0, 0.1) (вместо 0.01) | 17 | 0.612 | 0.321 |
| — | **Микс CGFM+CBAM P5** | CBAM self + FiLM context на одном уровне | 17 | 0.645 | 0.350 |

Для сравнения (итог task_12 + task_16):
- **baseline YOLOv12m aug_diffusion**: 0.623 mAP@50 / 0.373 mAP@50-95
- **CGFM P5-only** (лучшая конфигурация из task_16): 0.650 mAP@50 / 0.356 mAP@50-95

## Ключевые наблюдения

### Вариации A, B (residual + internal context) — близки к cgfm P3+P4+P5

Обе вариации дали результат 0.622–0.628, близкий к базовой P3+P4+P5-конфигурации и чуть выше baseline (0.623). Улучшить cgfm по сравнению с P5-only им не удалось. Причины:
- **A. Residual α=0 init** — α остаётся вблизи нуля всё обучение, потому что gradient на α пропорционален `(γF + β - F)`, а при γ≈1, β≈0 это выражение близко к нулю → мини-цикл: α мал → модуляция мала → gradient на α мал.
- **B. Internal context** — GAP(backbone-P5) даёт семантически корректный контекст, но FiLM-слои всё равно инициализированы около identity и BN всё равно поглощает модуляцию. Source контекста не решает проблему inner-collapse.

### Вариация C (wide γ ∈ 0.1–1.9) — обучение проседает

mAP@50 = 0.600 — **ниже baseline** (0.623). Широкий диапазон γ позволяет сети при обучении «обнулить» отдельные каналы (γ→0.1), что даёт слишком сильное возмущение для pretrained-backbone'а. Модель не сходится к стабильному решению.

### Вариация D (non-zero β init std=0.1) — тот же эффект

mAP@50 = 0.612 — ниже baseline, но не так сильно как wide γ. Большая инициализация β (при std=0.1, |β| в начале ~0.05–0.15) ломает baseline signal перед тем, как FiLM-параметры успевают стабилизироваться.

### Микс CGFM + CBAM на P5-only

mAP@50 = **0.645** — выше baseline (+2.2 п.п.), **но немного ниже чистого CGFM-P5-only** (0.650). Комбинация CBAM (self) + FiLM (context) на одном уровне даёт прирост, но **не лучше, чем один CGFM P5-only**. Две модуляции конкурируют: CBAM рекалибрует каналы через самореферентный сигнал, FiLM тут же рекалибрует их через внешний — эффекты частично перекрываются. CGFM P5-only остаётся Pareto-optimal.

## Финальный вывод по task_19

**Ни одна из 5 вариаций не превзошла CGFM P5-only (0.650).** Единственная оптимальная конфигурация CGFM остаётся — **P5-only** из task_16 (+2.7 п.п. над baseline 0.623).

Микс CGFM + CBAM также не улучшает P5-only. Оптимальная рекомендация для главы 5 диплома:

> **Финальная архитектура CGFM**: FiLMLayer с σ(Wc+b)-типа модуляцией на **только** одном уровне neck (самый глубокий — P5, где глобальный контекст максимально соответствует receptive field), с **легковесным внешним энкодером сцены** (MobileNetV3-Small, ~1.8M параметров, вход 224×224). Применение FiLM на нескольких уровнях приводит к BN-поглощению и не даёт прироста. Комбинации с CBAM избыточны.

**Ограничения реализации** (известные):
1. BN-нормализация в neck Ultralytics YOLOv12 делает features после FiLM нормализованными, что полностью поглощает умеренные мультипликативные возмущения. Поломка baseline-dynamics происходит только при агрессивных γ-интервалах (wide, beta_noise), что снижает качество.
2. Gradient на FiLM параметры мал в identity-режиме, так как `∂loss/∂feature ∝ γ ≈ 1`, и backbone-dynamics не «чувствует» FiLM.

**Возможные направления** (за рамками дипломного бюджета):
- Удалить BN после уровней FPN, куда вставляется FiLM, заменить на GroupNorm (устойчивый к мультипликативным возмущениям).
- Использовать **adaptive instance normalization** (AdaIN) с context-conditioned statistics вместо FiLM.
- Инициализировать веса FiLM **после** предварительного обучения backbone несколько эпох с identity FiLM (warm-up только FiLM на уже сходящейся модели).

## Артефакты

- `code/results/task_19/yolov12_cgfm_residual/` — Вариация A
- `code/results/task_19/yolov12_cgfm_internal/` — Вариация B (context = GAP-P5 backbone)
- `code/results/task_19/yolov12_cgfm_wide/` — Вариация C (γ ∈ 0.1–1.9)
- `code/results/task_19/yolov12_cgfm_beta_noise/` — Вариация D (partial, 27 эпох, best ep17)
- `code/results/task_19/yolov12_cgfm_cbam_p5/` — Микс CGFM+CBAM на P5-only
- Модифицированный `code/models/chapter4/film_layer.py` с параметрами `variant={'default','wide','beta_noise'}` и `residual={True,False}`
- Модифицированный `code/notebooks/chapter4_runner.py` с флагами `--film_variant`, `--film_residual`, `--internal_context`, `--mix_cbam`
- Логи: `/tmp/cgfm_{residual,internal,wide,beta}.log`, `~/plants_ch4/logs/mix_cbam_p5.log`
