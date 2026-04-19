# Result: Task 12 — Инфраструктура главы 4

## Статус
done

## Что было сделано

Создан полный пакет программных компонентов для экспериментов главы 4 (CGFM и его бейзлайны), зафиксирован reference baseline из task_07 как исходный замер для сравнения в task_13–task_18, и покрыт unit-тестами весь новый код. Инфраструктура позволяет запускать все конфигурации задач главы 4 через единый `code/notebooks/chapter4_runner.py` с параметрами `--config {se_neck, cbam_neck, cgfm}`.

Ключевые реализованные модули в `code/models/chapter4/`:

1. **`film_layer.FiLMLayer`** — канонический FiLM-блок (Perez et al., 2018):
   - $\gamma = \sigma(W_\gamma c + b_\gamma)$ (или альтернативно $1 + 0.5\tanh(\cdot)$, $1 + \tanh(\cdot)$, $0.1 + 1.8\sigma(\cdot)$ в вариациях task_19)
   - $\beta = W_\beta c + b_\beta$
   - $F' = \gamma \odot F + \beta$ с broadcast по (H, W)
   - Инициализация: $W_\gamma \sim \mathcal{N}(0, 10^{-4})$, $W_\beta = 0$ — так что при нулевом контексте $\gamma \approx 0.5$ (или 1 в более поздних вариациях), $\beta = 0$, модуляция близка к identity — стабилизирует раннее обучение.
   - Forward принимает features `[B, C, H, W]` и context `[B, D]`. Метод `last_gamma(context)` возвращает γ-значения без модификации — нужен для γ-визуализации в task_18.

2. **`context_encoder.ContextEncoder`** — внешний энкодер сцены поверх `timm`:
   - Поддерживает 3 backbone: `mobilenetv3_small_100` (~1.5M, default для CGFM), `efficientnet_b0` (~5.3M), `vit_tiny_patch16_224` (~5.7M).
   - После backbone — GlobalAvgPool (для CNN автоматически через `timm.create_model(global_pool='avg')`) → LayerNorm → Linear(feat→out_dim=256) → ReLU → Linear(out_dim→out_dim).
   - Вход — `[B, 3, 224, 224]` (отдельно от детектора, который работает с 640×640).
   - Feat-dim определяется **динамически dry-run'ом** — у MobileNetV3 `num_features=576`, но выход модели с `num_classes=0` — 1024 (conv_head добавляет преобразование). Поэтому единый dry-run в `__init__` безопасно определяет реальный размер.

3. **`se_block.SEBlock`** (Hu et al., 2018):
   - GAP(F) → FC(C → C/r=16) → ReLU → FC(C/r → C) → Sigmoid → F · w (broadcast).
   - Чисто self-referential: сигнал модуляции берётся из тех же features, которые модулирует.

4. **`cbam_block.CBAMBlock`** (Woo et al., 2018):
   - Channel attention: GAP+GMP → shared MLP → σ → F · w_c.
   - Spatial attention: channel-pool (avg+max, concat) → Conv 7×7 → σ → F · w_s.
   - Последовательное применение: channel → spatial.

5. **`late_fusion_head.LateFusionClassifier`** (архитектурный антипод CGFM):
   - На вход: ROI-feature `[B, C, 7, 7]` (из P4-feature neck через roi_align) + context `[B, 256]`.
   - ROI → Flatten → FC(C·49 → 256) → ReLU → concat(roi, context) → FC(512 → 256) → ReLU → Dropout(0.3) → FC(256 → 9).

6. **`yolov12_patch.wrap_neck_with`** — **главная техническая реализация** интеграции блоков модуляции в Ultralytics YOLO:
   - Находит `Detect`-слой (последний в `model.model` по имени класса) и читает его `.f = [p3_idx, p4_idx, p5_idx]` — индексы neck-слоёв с выходами P3/P4/P5.
   - Определяет число каналов выходов через `detect.cv2[i][0].conv.in_channels` (быстро) или через dry-run с hook'ом (fallback).
   - Заменяет `model.model[idx]` на `ModulatedLayer(orig, block, kind)`:
     - `kind='self'` для SE/CBAM: `output = block(orig(x))`
     - `kind='film'` для FiLM: `output = block(orig(x), context)` с контекстом из `set_context(c)`
   - **Делегирует** атрибуты Ultralytics: `.f`, `.i`, `.type`, `.np` — без этого Sequential-проход DetectionModel падает.
   - Для FiLM-режима с внешним ContextEncoder: патчит `DetectionModel.forward` через замыкание `_patch_forward_with_context`, которое на входе downsampling'ет изображение до 224×224, пропускает через encoder, получает c ∈ ℝ²⁵⁶, и инжектирует его в каждый FiLM-слой через `set_context(c)` перед forward детектора.

Интеграция прошла sanity-check: 1 эпоха YOLOv12n+SE на dataset_final за ~2 минуты, loss падает, Ultralytics trainer и monkey-patching сосуществуют. Детектные файлы: `code/scripts/sanity_train_se.py`, лог `/tmp/sanity_se/run/`.

## Почему именно так

1. **Monkey-patch `DetectionModel.forward` вместо переписывания `Sequential`**: Ultralytics активно использует `_forward_once` с хитрым сохранением промежуточных выходов (атрибут `save`). Переписать эту логику — значит дублировать ~50 строк Ultralytics. Monkey-patch оставляет оригинальную логику нетронутой и только добавляет pre-processing (генерация контекста).
2. **Делегирование атрибутов** (`.f`, `.i`, `.type`): `_forward_once` рутинно обращается к этим полям для маршрутизации тензоров. Если `ModulatedLayer` их не имеет, `_forward_once` падает с AttributeError.
3. **Dry-run для feat_dim в ContextEncoder** — безопаснее и универсальнее hardcoded-таблицы. `timm` может изменять num_features между версиями.
4. **Baseline зафиксирован как симлинк на task_07**, а не новый прогон: экономит ~4 часа GPU и гарантирует, что главы 3 и 4 используют **тот же референс** для сравнения.
5. **param_count через thop**: `thop.profile` даёт и число параметров, и MACs. GFLOPs = 2 × MACs (одно умножение + одно сложение = 2 FLOP на MAC).
6. **context_embeddings.npy с pretrained MobileNetV3** (без обучения) — фиксирует референсное распределение контекстных эмбеддингов до любого CGFM-дообучения. В task_18 будет сравнение: t-SNE этих baseline-эмбеддингов vs эмбеддинги после CGFM-обучения.

## Как реализовано

- **Написание и тестирование**: каждый модуль покрыт отдельным тестом (`code/tests/chapter4/`). Тесты проверяют формы, градиенты и (для FiLM) identity-свойство при нулевом контексте.
- **Запуск тестов**: `python code/tests/chapter4/test_<name>.py` (без pytest — тесты запускаются как скрипты с явным вызовом `if __name__ == '__main__'`).
- **Интеграция с Ultralytics**: единый вызов `wrap_neck_with(y.model, block_factory, context_encoder, levels)` обрабатывает все 4 режима (SE-neck, CBAM-neck, CGFM P3+P4+P5, CGFM с подмножеством уровней).
- **Chapter4 runner** (`code/notebooks/chapter4_runner.py`) — CLI для запуска конфигураций. Принимает `--config {se_neck, cbam_neck, cgfm} --encoder {mobilenetv3_small_100, efficientnet_b0, vit_tiny_patch16_224} --levels P3 P4 P5 --out <dir> --name <str>`.

## Результаты тестов

Все 5 тестов проходят (16 отдельных assertions):

| Тест | Assertions | Что проверяется |
|---|---:|---|
| `test_film_layer.py` | 4 | Формы, identity при zero-context, потоки градиента, диапазон γ |
| `test_context_encoder.py` | 3 | Выходная размерность 256 для каждого из 3 backbone |
| `test_se_cbam.py` | 4 | Формы SE/CBAM, потоки градиента |
| `test_late_fusion.py` | 2 | Формы logits, потоки градиента |
| `test_yolov12_patch.py` | 3 | wrap+dry-run для SE self-attention, FiLM с context, P5-only subset |

## Baseline для сводной таблицы главы 4

Полный набор артефактов скопирован из `code/results/task_07/yolov12_aug_diffusion/` в `code/results/task_12/yolov12_baseline/` (best.pt — как симлинк, не копия). Дополнительно сгенерированы:
- `param_count.json`: `params_total=20144427 (20.14M)`, `macs=33886950400`, `gflops=67.77` — через thop на dry-run с `x = zeros(1,3,640,640)`.
- `context_embeddings.npy`: `(445, 256)` — MobileNetV3-Small (pretrained, без дообучения) на всех 445 тестовых изображениях, ресайз до 224×224. Используется как baseline для t-SNE сравнения в task_18.
- `context_embeddings_filenames.json`: порядок имён файлов, соответствующих строкам в .npy.

| Метрика | Значение | Источник |
|---|---:|---|
| mAP@50 | 0.651 | task_07/yolov12_aug_diffusion (best epoch 17) |
| mAP@50-95 | 0.365 | task_07/yolov12_aug_diffusion |
| Precision | 0.810 | task_07/yolov12_aug_diffusion |
| Recall | 0.688 | task_07/yolov12_aug_diffusion |
| FPS (batch=1, fp32) | 20.4 | task_07/yolov12_aug_diffusion |
| Параметров, M | 20.14 | `param_count.json` |
| GFLOPs | 67.77 | `param_count.json` |

## Проблемы / Замечания

- **`timm.create_model('mobilenetv3_small_100', num_classes=0)` возвращает 1024-вектор**, не 576 (как указано в `num_features`). Причина: conv_head, который остаётся активным несмотря на `num_classes=0`. Исправлено через dry-run.
- **`ModulatedLayer` должен делегировать атрибуты Ultralytics** (`.f`, `.i`, `.type`). Без этого `DetectionModel._forward_once` падает. Исправлено в конструкторе.
- **Двойной `.train()` в Ultralytics 8.3.240** — вызов `yolo.train()` дважды подряд на одной и той же YOLO-instance падает с `KeyError: 'model'`. Это стало критичным препятствием для warm-up-стратегии в task_15 — пришлось отказаться от warm-up и полагаться на identity-подобную инициализацию FiLM.
- **`best.pt` — симлинк, gitignored** (`*.pt` в .gitignore). Для воспроизводимости нужен доступ к task_07/yolov12_aug_diffusion/weights/best.pt.

## Артефакты

**Модули:**
- `code/models/__init__.py`
- `code/models/chapter4/__init__.py` — публичный API
- `code/models/chapter4/film_layer.py` — FiLMLayer
- `code/models/chapter4/context_encoder.py` — ContextEncoder
- `code/models/chapter4/se_block.py` — SEBlock
- `code/models/chapter4/cbam_block.py` — CBAMBlock (channel + spatial attention)
- `code/models/chapter4/late_fusion_head.py` — LateFusionClassifier
- `code/models/chapter4/yolov12_patch.py` — wrap_neck_with, ModulatedLayer
- `code/models/chapter4/README.md` — документация пакета

**Тесты:**
- `code/tests/__init__.py`
- `code/tests/chapter4/__init__.py`
- `code/tests/chapter4/test_film_layer.py` (4 tests)
- `code/tests/chapter4/test_context_encoder.py` (3 tests)
- `code/tests/chapter4/test_se_cbam.py` (4 tests)
- `code/tests/chapter4/test_late_fusion.py` (2 tests)
- `code/tests/chapter4/test_yolov12_patch.py` (3 tests)

**Baseline:**
- `code/results/task_12/yolov12_baseline/` — полный набор Ultralytics-артефактов + param_count.json + context_embeddings.npy + context_embeddings_filenames.json

**Runner и скрипты:**
- `code/notebooks/chapter4_runner.py` — универсальный runner для SE/CBAM/CGFM
- `code/scripts/task12_fix_baseline.py` — генератор baseline-артефактов (copy from task_07 + param_count + embeddings)
- `code/scripts/sanity_train_se.py` — sanity-check интеграции (1 epoch SE-neck на YOLOv12n)
