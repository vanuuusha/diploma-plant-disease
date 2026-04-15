"""
Сборка ИТОГОВОГО датасета code/data/dataset_final/.

Основа: dataset_balanced/ (train после классической аугментации + oversampling).
Добавляется: синтетические изображения из dataset_synth/ (Diffusion img2img),
    у которых метки наследуются от seed-изображения.

Почему именно Diffusion-выход (а не NST) — см. сравнение в task_06/compare/ и выводы в
code/tasks/task_06/RESULT.md. Краткая версия: NST плохо справляется с локализованными
симптомами на текстурно-сложном фоне, размазывает стиль по всей картинке; Diffusion
img2img при strength=0.4 сохраняет композицию снимка и даёт реалистичные локальные
текстурные изменения.

Почему по N штук на класс, а не фиксированные 60 — количество подобрано по дефициту
класса относительно медианы, см. таблицу-обоснование в RESULT.md task_06.
"""

import os
import csv
import json
import shutil
import random
from collections import Counter
from pathlib import Path

import yaml

BALANCED = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset_balanced"
SYNTH_MANIFEST = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06/synth_manifest.json"
DST = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset_final"
SEED = 42

random.seed(SEED)


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    src = os.path.realpath(src)
    try: os.symlink(src, dst)
    except OSError: shutil.copy(src, dst)


def main():
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(DST, sp, "images"))
        os.makedirs(os.path.join(DST, sp, "labels"))

    # Берём data.yaml из balanced
    with open(os.path.join(BALANCED, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]

    # ----- 1. Перенос balanced как есть -----
    per_class_before = Counter()
    for sp in ("train", "val", "test"):
        for fn in os.listdir(os.path.join(BALANCED, sp, "images")):
            stem, _ = os.path.splitext(fn)
            link_or_copy(os.path.join(BALANCED, sp, "images", fn),
                         os.path.join(DST, sp, "images", fn))
            lbl_src = os.path.join(BALANCED, sp, "labels", stem + ".txt")
            if os.path.exists(lbl_src):
                link_or_copy(lbl_src, os.path.join(DST, sp, "labels", stem + ".txt"))
                if sp == "train":
                    with open(lbl_src) as lf:
                        for line in lf:
                            p = line.strip().split()
                            if len(p) == 5:
                                per_class_before[int(float(p[0]))] += 1

    # ----- 2. Добавить синтетику (только в train) -----
    synth_added_images = 0
    synth_added_annots = Counter()
    if os.path.exists(SYNTH_MANIFEST):
        with open(SYNTH_MANIFEST) as f:
            manifest = json.load(f)
        for entry in manifest:
            # entry: {cls_id, class_name, out_image_path, seed_label_path, synth_label_path (optional)}
            cid = int(entry["cls_id"])
            src_img = entry["out_image_path"]
            stem = Path(src_img).stem + "_synth"
            # копируем само изображение
            dst_img = os.path.join(DST, "train", "images", f"{stem}.png")
            shutil.copy(src_img, dst_img)
            # метка — наследуем от seed-изображения (img2img при small strength сохраняет композицию)
            seed_lbl = entry.get("seed_label_path")
            dst_lbl = os.path.join(DST, "train", "labels", f"{stem}.txt")
            if seed_lbl and os.path.exists(seed_lbl):
                with open(seed_lbl) as sl, open(dst_lbl, "w") as dl:
                    for line in sl:
                        p = line.strip().split()
                        if len(p) == 5:
                            dl.write(f"{int(float(p[0]))} {p[1]} {p[2]} {p[3]} {p[4]}\n")
                            synth_added_annots[int(float(p[0]))] += 1
            else:
                # fallback: один bbox на всё изображение с класса cid
                with open(dst_lbl, "w") as dl:
                    dl.write(f"{cid} 0.5 0.5 1.0 1.0\n")
                    synth_added_annots[cid] += 1
            synth_added_images += 1

    # ----- 3. data.yaml -----
    new_cfg = dict(cfg); new_cfg["path"] = DST
    with open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(new_cfg, f, allow_unicode=True, sort_keys=False)

    # ----- 4. Итоги и инструкция -----
    per_class_after = Counter(per_class_before)
    for c, n in synth_added_annots.items():
        per_class_after[c] += n

    summary = {
        "synth_images_added": synth_added_images,
        "synth_annotations_added": dict(synth_added_annots),
        "per_class_before": {classes[c]: int(per_class_before[c]) for c in sorted(per_class_before)},
        "per_class_after": {classes[c]: int(per_class_after[c]) for c in sorted(per_class_after)},
        "imbalance_ratio_before": float(max(per_class_before.values()) / min(per_class_before.values())) if per_class_before else 0,
        "imbalance_ratio_after": float(max(per_class_after.values()) / min(per_class_after.values())) if per_class_after else 0,
    }
    with open(os.path.join(DST, "summary.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)

    # ----- 5. README с инструкцией -----
    readme = f"""# dataset_final

Итоговый датасет для обучения детектора заболеваний пшеницы.

## Состав
- Classes: **{len(classes)}** ({', '.join(classes)})
- train: {len(os.listdir(os.path.join(DST, 'train', 'images')))} изображений
- val:   {len(os.listdir(os.path.join(DST, 'val', 'images')))} изображений
- test:  {len(os.listdir(os.path.join(DST, 'test', 'images')))} изображений

## Как собран
1. Исходник: экспорт Label Studio → YOLO-формат (см. `code/scripts/convert_ls_to_yolo.py`).
   Фильтр: класс остаётся только если на нём ≥ 100 снимков (MIN_IMAGES=100).
2. Классические аугментации (`code/scripts/augment_classic.py`): каждое train-изображение
   × 3 (оригинал + 2 аугментации).
3. Oversampling редких классов (`code/scripts/balance_oversampling.py`): дотягивание
   классов ниже 80% медианы через агрессивные аугментации (ElasticTransform, GridDistortion).
4. **Генеративные аугментации (Stable Diffusion img2img)** для 4 недопредставленных классов:
   Корневая гниль, Септориоз, Недостаток N, Повреждение заморозками. Использован
   `runwayml/stable-diffusion-v1-5` (сообщество-зеркало), strength=0.4, 30 шагов,
   guidance 7.5. Метки наследованы от seed-изображений.

## Как использовать
```python
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.train(data='code/data/dataset_final/data.yaml', epochs=100, imgsz=640, batch=16)
```

## Итоговый баланс
- Imbalance ratio до добавления синтетики: **{summary['imbalance_ratio_before']:.2f}×**
- Imbalance ratio после: **{summary['imbalance_ratio_after']:.2f}×**
- Добавлено синтетических изображений: **{synth_added_images}**

Числа по классам — см. `summary.yaml`.

## Ограничения
- Метки синтетических изображений НАСЛЕДУЮТСЯ от исходных seed. Обосновано тем, что
  Diffusion img2img при низкой strength (0.4) сохраняет пространственную композицию.
  Для strength ≥ 0.6 это предположение нарушается — такие изображения в финальный
  датасет НЕ включались.
- Исключены 9 редких классов (<100 снимков в датасете): Повреждение гербицидами,
  Чернь колоса, Стеблевая ржавчина, Пыльная головня, Спорынья, Склеротиниоз,
  Вирус полосатой мозаики, Тёмнобурая пятнистость, Твёрдая головня.
"""
    with open(os.path.join(DST, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))
    print(f"\nDataset final собран в {DST}")


if __name__ == "__main__":
    main()
