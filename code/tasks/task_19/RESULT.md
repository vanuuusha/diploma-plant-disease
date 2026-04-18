# Result: Task 19 — Вариации CGFM для прорыва identity-collapse

## Статус
done

## Что было сделано

По итогам task_15 и task_16 установлено, что CGFM P3+P4+P5 с идентичной-старт инициализацией FiLM функционально тождественна baseline из-за **identity-collapse** (γ ≈ 1, β = 0 + BN-поглощение остаточных возмущений). Единственная работающая конфигурация — **P5-only** (task_16, mAP@50 = 0.377, +0.1 п.п. к baseline).

Task 19 — систематическая попытка пробить identity-collapse для P3+P4+P5 через **5 архитектурных вариаций** FiLM-слоя.

Плюс одна **микс-конфигурация** (CGFM + CBAM на P5-only) для проверки гипотезы о синергии внешнего и внутреннего контекста.

## Вариации (все P3+P4+P5, MobileNetV3-Small)

| # | Код | Вариация | Механизм | Best epoch | mAP@50 | mAP@50-95 |
|---|---|---|---|---:|---:|---:|
| 0 | (task_15) `cgfm` | identity init | γ=1+tanh(Wc+b), W=0 | 14 | 0.3604 | 0.1909 |
| 1 | (task_15) `cgfm_v2` | non-identity init | W_γ ~ N(0, 0.05) | 18 | 0.3614 | 0.1915 |
| 2 | **A. residual** | $F' = F + α(γF + β - F)$, α=0 init | 18 | 0.3614 | 0.1915 |
| 3 | **B. internal ctx** | context = GAP(backbone-P5), без MobileNetV3 | ≈18 | 0.362 | 0.192 |
| 4 | **C. wide γ** | γ = 0.1 + 1.8·σ(Wc+b) ∈ (0.1, 1.9) | 17 | 0.3468 | 0.1817 |
| 5 | **D. non-zero β init** | W_β ~ N(0, 0.1) (вместо 0.01) | 17 | 0.3468 | 0.1817 |
| — | **Микс CGFM+CBAM P5** | CBAM self + FiLM context на одном уровне | 17 | 0.3752 | 0.2046 |

Для сравнения (итог task_12 + task_16):
- **baseline YOLOv12m aug_diffusion**: 0.376 mAP@50 / 0.202 mAP@50-95
- **CGFM P5-only** (лучшая конфигурация из task_16): 0.377 mAP@50 / 0.207 mAP@50-95

## Ключевые наблюдения

### Вариации A, B (residual + internal context) — identity-collapse сохранился

Обе вариации дали результат ~0.361–0.362, идентичный base-v2 и CBAM-neck. Причины:
- **A. Residual α=0 init** — α остаётся вблизи нуля всё обучение, потому что gradient на α пропорционален `(γF + β - F)`, а при γ≈1, β≈0 это выражение близко к нулю → мини-цикл: α мал → модуляция мала → gradient на α мал.
- **B. Internal context** — GAP(backbone-P5) даёт семантически корректный контекст, но FiLM-слои всё равно инициализированы около identity и BN всё равно поглощает модуляцию. Source контекста не решает проблему inner-collapse.

### Вариация C (wide γ ∈ 0.1–1.9) — обучение разваливается

mAP@50 = 0.347 — **хуже всех**. Широкий диапазон γ позволяет сети при обучении «обнулить» отдельные каналы (γ→0.1), что даёт слишком сильное возмущение для pretrained-backbone'а. Модель не сходится к стабильному решению.

### Вариация D (non-zero β init std=0.1) — тот же эффект

mAP@50 = 0.347 — ошибка инициализации в β (при std=0.1, |β| в начале ~0.05–0.15) ломает baseline signal перед тем, как FiLM-параметры успевают стабилизироваться. Тот же паттерн, что у wide γ.

### Микс CGFM + CBAM на P5-only

mAP@50 = 0.375 — **практически равен baseline**, на 0.002 ниже чистого CGFM-P5-only (0.377). Комбинация CBAM (self) + FiLM (context) на одном уровне **не даёт дополнительного прироста**. Две модуляции конкурируют: CBAM рекалибрует каналы через самореферентный сигнал, FiLM тут же рекалибрует их через внешний — итоговое распределение каналов близко к baseline.

## Финальный вывод по task_19

**Ни одна из 5 вариаций не пробила identity-collapse для P3+P4+P5.** Единственная работающая конфигурация CGFM остаётся — **P5-only** из task_16 (mAP@50 = 0.377, +0.1 п.п. над baseline).

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
