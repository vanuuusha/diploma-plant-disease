# Task 19: Вариации CGFM для прорыва identity-collapse

## Статус
in_progress

## Цель

По итогам task_15 обнаружен **identity-collapse**: FiLM-слои с γ≈1, β≈0 функционально тождественны identity, а BatchNorm перед Detect-головой поглощает малые возмущения. Training-динамика детектора становится идентичной baseline → mAP@50 одинаков у SE-Neck, CBAM, CGFM-identity, CGFM-EffB0, CGFM-ViT-Tiny (все 0.3604).

**Единственный положительный результат** (CGFM P5-only = 0.377) получен за счёт того, что модуляция одного уровня меньше поглощается BN. Цель task_19 — проверить **4 архитектурные вариации** FiLM, которые принудительно выводят модуляцию из identity-режима и потенциально дают прирост на всех трёх уровнях (P3+P4+P5):

| Вариант | Механизм | Ожидание |
|---|---|---|
| A. **Residual gating** | $F' = F + \alpha \cdot (\gamma F + \beta - F)$, $\alpha$ — обучаемый скаляр, init=0 | Плавное включение модуляции, не ломая baseline начало |
| B. **Internal context** | Контекст $c$ = GAP(P5-baseline), без отдельного MobileNetV3 | Контекст семантически согласован с features детектора |
| C. **Wide-range γ** | $\gamma = 0.1 + 1.8\sigma(Wc+b) \in (0.1, 1.9)$ | Более агрессивная модуляция, труднее поглощается BN |
| D. **Non-zero β init** | $W_\beta \sim \mathcal{N}(0, 0.1)$ (вместо 0.01) | β не может collapsed в 0 в начале |

## Общий протокол

Те же гиперпараметры что task_15 (P3+P4+P5, batch=16, seed=42, epochs=100, patience=15 или 10). Все встроенные Ultralytics аугментации отключены.

## Шаги

1. **Реализовать вариант A (residual gating)** в `code/models/chapter4/film_layer.py` как опциональный параметр `residual=True`:
   ```python
   class FiLMLayer:
       def __init__(..., residual=False):
           ...
           if residual:
               self.alpha = nn.Parameter(torch.zeros(1))  # init=0 → identity
       def forward(...):
           y = gamma * features + beta
           if self.residual:
               return features + self.alpha * (y - features)
           return y
   ```

2. **Реализовать вариант B (internal context)** в `code/notebooks/chapter4_runner.py`:
   - Новый флаг `--internal_context` — вместо ContextEncoder регистрируется hook на P5-слое neck, берётся GAP на forward, используется как c.
   - Никакого MobileNetV3, экономия 1.85 M параметров.

3. **Реализовать вариант C (wide-range γ)** — формула `γ = 0.1 + 1.8 * sigmoid(Wc+b)`:
   - Инициализация $W_\gamma = 0$, $b = 0$ → $\sigma(0) = 0.5$ → $\gamma = 1.0$ (identity при инициализации, но диапазон шире при обучении).

4. **Реализовать вариант D (non-zero β init)**:
   - `std=0.1` для $W_\beta$ — при типичных pretrained-контекстах MobileNetV3 даст $|β|$ ~ 0.05–0.15 c начала.

5. **Запустить параллельно:**
   - A100: B (internal context) — самый принципиальный вариант
   - A100 затем: A (residual gating)
   - Локально: C (wide-range γ)
   - Локально затем: D (non-zero β)

6. **Сравнить все 4 с baseline + task_15 (P3+P4+P5 identity) + task_16 (P5-only)**. Выбрать лучший.

## Ожидаемый результат

- Минимум одна из четырёх вариаций даёт mAP@50 ≥ 0.377 (на уровне P5-only) на P3+P4+P5.
- Идеально: вариация B (internal context) превзойдёт P5-only за счёт семантически корректного контекста.
- Если все 4 вариации не улучшают → окончательный вывод: FiLM на P3+P4+P5 фундаментально не работает с BN, рекомендуемая конфигурация — P5-only.

## Результат записать в

`code/tasks/task_19/RESULT.md` — таблица 4 конфигураций × метрики + анализ.
