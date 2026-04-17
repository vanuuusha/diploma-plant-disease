# Result: Task 12 — Инфраструктура главы 4

## Статус
done

## Что было сделано

Создан пакет `code/models/chapter4/` со всеми модулями, необходимыми для экспериментов главы 4:

- `film_layer.FiLMLayer` — Feature-wise Linear Modulation с инициализацией $W_\gamma \sim \mathcal{N}(0, 10^{-4})$, $W_\beta = 0$ (γ ≈ 0.5, модуляция ≈ 0.5·F при нулевом контексте — стабилизирует warm-up).
- `context_encoder.ContextEncoder` — обёртка над timm-бэкбонами; поддерживаются `mobilenetv3_small_100`, `efficientnet_b0`, `vit_tiny_patch16_224`; выход проектируется через LayerNorm+MLP в 256-мерный вектор.
- `se_block.SEBlock` и `cbam_block.CBAMBlock` — референсные реализации самоатенции согласно Hu et al. 2018 и Woo et al. 2018.
- `late_fusion_head.LateFusionClassifier` — ROI + контекст → 9-класс logits.
- `yolov12_patch.wrap_neck_with` — обёртка выходов neck Ultralytics YOLO: находит `Detect.f = [p3,p4,p5]`, оборачивает соответствующие слои в `ModulatedLayer`, делегируя атрибуты `.f`, `.i`, `.type`. Для FiLM-режима дополнительно патчит `DetectionModel.forward`, инжектируя контекст $c$ из 224×224-даунсемпла входа.

Интеграция проверена sanity-прогоном 1 эпохи YOLOv12n+SE на `dataset_final`: ~2 минуты, `12.8 it/s`, loss падает — Ultralytics trainer и монки-патчинг корректно сосуществуют.

## Результаты тестов

| Тест | Статус |
|---|---|
| test_film_layer.py | pass (4/4) |
| test_context_encoder.py | pass (3/3) |
| test_se_cbam.py | pass (4/4) |
| test_late_fusion.py | pass (2/2) |
| test_yolov12_patch.py | pass (3/3) |

Запуск:
```
python code/tests/chapter4/test_film_layer.py
python code/tests/chapter4/test_context_encoder.py
python code/tests/chapter4/test_se_cbam.py
python code/tests/chapter4/test_late_fusion.py
python code/tests/chapter4/test_yolov12_patch.py
```

## Baseline для сводной таблицы главы 4

Артефакты скопированы из `task_07/yolov12_aug_diffusion` в `code/results/task_12/yolov12_baseline/`. Дополнительно:

- `param_count.json` — посчитан через `thop`.
- `context_embeddings.npy` — `(445, 256)`, MobileNetV3-Small без обучения, ImageNet-pretrained.

| Метрика | Значение | Источник |
|---|---:|---|
| mAP@50 | 0.373 | task_07/yolov12_aug_diffusion |
| mAP@50-95 | 0.230 | task_07/yolov12_aug_diffusion |
| Precision | 0.508 | task_07/yolov12_aug_diffusion |
| Recall | 0.329 | task_07/yolov12_aug_diffusion |
| FPS (batch=1, fp32) | 20.4 | task_07/yolov12_aug_diffusion |
| Параметров, M | 20.14 | пересчитано (thop) |
| GFLOPs | 67.77 | пересчитано (thop) |

(точные значения мAP — из `metrics.csv` в референс-директории).

## Проблемы / Замечания

- `timm.create_model('mobilenetv3_small_100', num_classes=0)` возвращает вектор размерности 1024 (не 576 как в `num_features`), т.к. в архитектуре остаётся `conv_head`. Размер feature определяется через единый dry-run на этапе инициализации энкодера.
- `ModulatedLayer` обязательно делегирует `.f`, `.i` от оригинального слоя — иначе `DetectionModel._forward_once` ломается при проходе по Sequential. Добавлен явный копирующий код в конструкторе.

## Артефакты

- `code/models/chapter4/film_layer.py`
- `code/models/chapter4/context_encoder.py`
- `code/models/chapter4/se_block.py`
- `code/models/chapter4/cbam_block.py`
- `code/models/chapter4/late_fusion_head.py`
- `code/models/chapter4/yolov12_patch.py`
- `code/models/chapter4/README.md`
- `code/tests/chapter4/{test_film_layer,test_context_encoder,test_se_cbam,test_late_fusion,test_yolov12_patch}.py`
- `code/results/task_12/yolov12_baseline/` — полный набор артефактов baseline (метрики, per-class, confusion matrix, predictions, fps, weights/best.pt (симлинк), param_count.json, context_embeddings.npy).
- `code/notebooks/chapter4_runner.py` — универсальный runner для экспериментов главы 4 (SE/CBAM/CGFM).
- `code/scripts/task12_fix_baseline.py` — генератор baseline-артефактов.
- `code/scripts/sanity_train_se.py` — sanity-check интеграции (1 эпоха SE-neck).
