"""
Конвертация Label Studio JSON (полигоны) в YOLO-формат (bbox)
+ стратифицированное разбиение train/val/test 70/20/10.

Структура входа (хардкод):
- /home/vanusha/diplom/all_dieseas_class/from_ls.json
- /home/vanusha/diplom/all_dieseas_class/origins/<russian_class>/<task_id>.jpg

Каждая Label Studio task связана с одним изображением через id (filename = <id>.jpg).
В origins/ присутствуют 9 классов-папок (по основной патологии съёмки), но в разметке
встречаются 18 классов лейблов (некоторые изображения содержат несколько типов поражений).
"""

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

# ---- Конфиг (хардкод) ----
JSON_PATH = "/home/vanusha/diplom/all_dieseas_class/from_ls.json"
ORIGINS_DIR = "/home/vanusha/diplom/all_dieseas_class/origins"
PROJECT_ROOT = "/home/vanusha/diplom/diploma-plant-disease"
OUT_DATASET = os.path.join(PROJECT_ROOT, "code/data/dataset")
CLASSES_YAML = os.path.join(PROJECT_ROOT, "code/configs/classes.yaml")
SEED = 42
SPLIT = (0.70, 0.20, 0.10)
# Класс попадает в датасет только если для него есть >= MIN_IMAGES снимков,
# на которых он присутствует хотя бы одной аннотацией. Критерий именно по фото,
# а не по числу bbox — потому что один снимок может содержать несколько bbox.
MIN_IMAGES = 100


def build_image_index():
    """task_id (int) -> abs path. Сканируем origins/<class>/<id>.jpg."""
    idx = {}
    for cls_dir in os.listdir(ORIGINS_DIR):
        full_dir = os.path.join(ORIGINS_DIR, cls_dir)
        if not os.path.isdir(full_dir):
            continue
        for fn in os.listdir(full_dir):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                tid = int(stem)
            except ValueError:
                continue
            idx[tid] = os.path.join(full_dir, fn)
    return idx


def polygon_to_yolo_bbox(points, img_w, img_h):
    """LS points в %, нужны абсолютные → нормализованный bbox xc,yc,w,h ∈ [0,1]."""
    xs = [p[0] / 100.0 for p in points]
    ys = [p[1] / 100.0 for p in points]
    x_min, x_max = max(0.0, min(xs)), min(1.0, max(xs))
    y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))
    if x_max <= x_min or y_max <= y_min:
        return None
    return (
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        x_max - x_min,
        y_max - y_min,
    )


def main():
    print("[1/5] Загрузка JSON…")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    print(f"  Tasks total: {len(tasks)}")

    image_idx = build_image_index()
    print(f"  Image files in origins/: {len(image_idx)}")

    # ---- 2. Разведка лейблов ----
    label_counter = Counter()
    for t in tasks:
        for ann in t["annotations"]:
            for r in ann["result"]:
                for lab in r["value"].get("polygonlabels", []):
                    label_counter[lab] += 1
    # Подсчёт числа УНИКАЛЬНЫХ ИЗОБРАЖЕНИЙ, на которых присутствует каждый класс
    # (с учётом наличия файла в origins/ — иначе изображение всё равно будет отброшено).
    images_per_class = Counter()
    for t in tasks:
        tid = t["id"]
        if tid not in image_idx:
            continue
        seen = set()
        for ann in t["annotations"]:
            for r in ann["result"]:
                for lab in r["value"].get("polygonlabels", []):
                    seen.add(lab)
        for lab in seen:
            images_per_class[lab] += 1

    all_classes = [c for c, _ in label_counter.most_common()]
    classes = [c for c in all_classes if images_per_class[c] >= MIN_IMAGES]
    dropped = [c for c in all_classes if images_per_class[c] < MIN_IMAGES]
    cls2id = {c: i for i, c in enumerate(classes)}
    print(f"[2/5] Классов в JSON: {len(all_classes)}; оставлено (>= {MIN_IMAGES} снимков с классом): {len(classes)}")
    print(f"  id  imgs  anns   класс")
    for c in classes:
        print(f"  {cls2id[c]:2d}  {images_per_class[c]:5d}  {label_counter[c]:5d}   {c}")
    if dropped:
        print(f"  Отброшено классов с <{MIN_IMAGES} снимками: {len(dropped)}")
        for c in dropped:
            print(f"    -- {images_per_class[c]:3d} img / {label_counter[c]:3d} ann   {c}")

    # ---- 3. Сбор образцов ----
    samples = []  # (image_path, [(cls_id, xc, yc, w, h), ...], dominant_cls)
    skipped_no_image = 0
    skipped_no_ann = 0

    for t in tqdm(tasks, desc="[3/5] Подготовка"):
        tid = t["id"]
        img_path = image_idx.get(tid)
        if img_path is None:
            skipped_no_image += 1
            continue
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            skipped_no_image += 1
            continue

        bboxes = []
        per_cls = Counter()
        for ann in t["annotations"]:
            for r in ann["result"]:
                pls = r["value"].get("polygonlabels", [])
                pts = r["value"].get("points")
                if not pls or not pts:
                    continue
                lab = pls[0]
                if lab not in cls2id:
                    continue
                cls_id = cls2id[lab]
                bb = polygon_to_yolo_bbox(pts, img_w, img_h)
                if bb is None:
                    continue
                bboxes.append((cls_id, *bb))
                per_cls[cls_id] += 1
        if not bboxes:
            skipped_no_ann += 1
            continue
        dominant = per_cls.most_common(1)[0][0]
        samples.append((img_path, bboxes, dominant))

    print(f"  Готово: {len(samples)} изображений с разметкой")
    print(f"  Пропущено без изображения: {skipped_no_image}")
    print(f"  Пропущено без аннотаций (после фильтрации): {skipped_no_ann}")

    # ---- 4. Стратифицированное разбиение ----
    print("[4/5] Стратифицированное разбиение 70/20/10 по доминантному классу")
    y = [s[2] for s in samples]
    # Если в каком-то классе < 3 образцов — стратификация невозможна для test_split
    cls_counts = Counter(y)
    too_few = [c for c, n in cls_counts.items() if n < 3]
    if too_few:
        print(f"  Внимание: классы с <3 изображениями (исключены из стратификации): {too_few}")
        keep_mask = [c not in too_few for c in y]
        rest = [s for s, k in zip(samples, keep_mask) if not k]
        samples = [s for s, k in zip(samples, keep_mask) if k]
        y = [s[2] for s in samples]
    else:
        rest = []

    test_frac = SPLIT[2]
    val_frac = SPLIT[1] / (SPLIT[0] + SPLIT[1])
    train_val, test = train_test_split(
        samples, test_size=test_frac, stratify=y, random_state=SEED
    )
    y_tv = [s[2] for s in train_val]
    train, val = train_test_split(
        train_val, test_size=val_frac, stratify=y_tv, random_state=SEED
    )
    # Маленькие классы — все в train
    train.extend(rest)
    print(f"  train: {len(train)}, val: {len(val)}, test: {len(test)}")

    # ---- 5. Запись на диск ----
    print("[5/5] Запись YOLO-датасета")
    if os.path.isdir(OUT_DATASET):
        shutil.rmtree(OUT_DATASET)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT_DATASET, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DATASET, split, "labels"), exist_ok=True)

    def write_split(split_name, items):
        per_cls = Counter()
        for img_path, bboxes, _ in tqdm(items, desc=f"  {split_name}"):
            stem = Path(img_path).stem
            ext = Path(img_path).suffix
            dst_img = os.path.join(OUT_DATASET, split_name, "images", f"{stem}{ext}")
            dst_lbl = os.path.join(OUT_DATASET, split_name, "labels", f"{stem}.txt")
            if not os.path.exists(dst_img):
                try:
                    os.symlink(img_path, dst_img)
                except OSError:
                    shutil.copy(img_path, dst_img)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                for cls_id, xc, yc, w, h in bboxes:
                    # Зажать в [0, 1]
                    xc = min(max(xc, 0.0), 1.0)
                    yc = min(max(yc, 0.0), 1.0)
                    w = min(max(w, 0.0), 1.0)
                    h = min(max(h, 0.0), 1.0)
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    per_cls[cls_id] += 1
        return per_cls

    train_stat = write_split("train", train)
    val_stat = write_split("val", val)
    test_stat = write_split("test", test)

    # data.yaml
    data_yaml = {
        "path": OUT_DATASET,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(classes),
        "names": classes,
    }
    with open(os.path.join(OUT_DATASET, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    os.makedirs(os.path.dirname(CLASSES_YAML), exist_ok=True)
    with open(CLASSES_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump({"names": classes, "nc": len(classes)}, f, allow_unicode=True, sort_keys=False)

    # ---- Валидация ----
    print("\n=== Валидация ===")
    bad = 0
    for split in ("train", "val", "test"):
        lbl_dir = os.path.join(OUT_DATASET, split, "labels")
        img_dir = os.path.join(OUT_DATASET, split, "images")
        n_imgs = len(os.listdir(img_dir))
        n_lbls = len(os.listdir(lbl_dir))
        for fn in os.listdir(lbl_dir):
            with open(os.path.join(lbl_dir, fn), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        bad += 1
                        continue
                    cid = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    if cid < 0 or cid >= len(classes):
                        bad += 1
                    if any(c < 0 or c > 1 for c in coords):
                        bad += 1
        print(f"  {split}: {n_imgs} images, {n_lbls} labels")
    print(f"  Невалидных строк bbox: {bad}")

    # ---- Сводка ----
    print("\n=== Сводка по классам (train / val / test) ===")
    for c in classes:
        cid = cls2id[c]
        print(f"  {c:35s}  {train_stat.get(cid,0):5d} / {val_stat.get(cid,0):4d} / {test_stat.get(cid,0):4d}")

    # Сохраняем сводку для RESULT.md
    summary = {
        "total_tasks": len(tasks),
        "total_images_with_annotations": len(train) + len(val) + len(test),
        "skipped_no_image": skipped_no_image,
        "skipped_no_annotation": skipped_no_ann,
        "num_classes": len(classes),
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "classes": [
            {
                "id": cls2id[c],
                "name": c,
                "train": train_stat.get(cls2id[c], 0),
                "val": val_stat.get(cls2id[c], 0),
                "test": test_stat.get(cls2id[c], 0),
                "total_annotations": label_counter[c],
            }
            for c in classes
        ],
    }
    summary_path = os.path.join(PROJECT_ROOT, "code/results/task_02_summary.yaml")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)
    print(f"\nСводка сохранена: {summary_path}")
    print(f"data.yaml: {os.path.join(OUT_DATASET, 'data.yaml')}")


if __name__ == "__main__":
    main()
