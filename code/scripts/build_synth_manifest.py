"""
Из diffusion_log.csv строит synth_manifest.json для build_dataset_final.py.
Метки наследуются от seed-изображения (strength=0.4–0.6 сохраняет композицию).
"""

import csv
import json
import os

LOG = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06/diffusion_log.csv"
DATASET_LABELS = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset/train/labels"
OUT_MANIFEST = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06/synth_manifest.json"

STRENGTH_THRESHOLD = 0.55  # отсекаем агрессивные варианты, где композиция ломается

CLASS_NAMES = [
    "Недостаток P2O5", "Листовая (бурая) ржавчина", "Мучнистая роса",
    "Пиренофороз", "Фузариоз", "Корневая гниль", "Септориоз",
    "Недостаток N", "Повреждение заморозками",
]


def main():
    manifest = []
    skipped_high_strength = 0
    with open(LOG, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            strength = float(row["strength"])
            if strength > STRENGTH_THRESHOLD:
                skipped_high_strength += 1
                continue
            cls_name = row["class_name"]
            cls_id = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else -1
            if cls_id < 0:
                continue
            seed_img = row["seed_path"]
            stem = os.path.splitext(os.path.basename(seed_img))[0]
            seed_lbl = os.path.join(DATASET_LABELS, stem + ".txt")
            if not os.path.exists(seed_lbl):
                continue
            manifest.append({
                "cls_id": cls_id,
                "class_name": cls_name,
                "seed_image_path": seed_img,
                "seed_label_path": seed_lbl,
                "out_image_path": row["out_path"],
                "strength": strength,
                "prompt": row["prompt"],
            })
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Manifest: {len(manifest)} записей (отфильтровано с strength>{STRENGTH_THRESHOLD}: {skipped_high_strength})")
    per_class = {}
    for m in manifest:
        per_class[m["class_name"]] = per_class.get(m["class_name"], 0) + 1
    for k, v in per_class.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
