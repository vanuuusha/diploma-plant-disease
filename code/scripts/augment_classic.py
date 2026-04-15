"""
Классические аугментации (Albumentations) для train сплита.
- Для каждого исходного изображения генерируется 2 аугментированные копии.
- Оригиналы сохраняются как симлинки.
- val/test копируются как симлинки без изменений.
- Аугментированные сохраняются с уменьшением до max-side ≤ 1024 для экономии диска.
"""

import os
import shutil
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import albumentations as A
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml
from tqdm import tqdm

cv2.setNumThreads(0)

SRC = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset"
DST = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset_augmented"
RES = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_04"
SEED = 42
N_AUG_PER_IMAGE = 2
MAX_SIDE = 1024
MIN_BBOX_AREA = 0.001  # норм., < — отбрасываем

random.seed(SEED)
np.random.seed(SEED)


def get_pipeline():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Affine(translate_percent=0.1, scale=(0.85, 1.15), rotate=(-20, 20), p=0.5),
            A.RandomResizedCrop(size=(640, 640), scale=(0.7, 1.0), p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(std_range=(0.04, 0.2), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.3),
            A.RandomShadow(p=0.2),
            A.RandomFog(p=0.15),
            A.ImageCompression(quality_range=(70, 100), p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_area=0,
            min_visibility=0.1,
            clip=True,
        ),
        seed=SEED,
    )


def load_yolo_labels(lbl_path):
    bboxes, labels = [], []
    if not os.path.exists(lbl_path):
        return bboxes, labels
    with open(lbl_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:
                continue
            labels.append(int(float(p[0])))
            bboxes.append(list(map(float, p[1:])))
    return bboxes, labels


def save_yolo_labels(lbl_path, bboxes, labels):
    with open(lbl_path, "w") as f:
        for cid, (xc, yc, w, h) in zip(labels, bboxes):
            xc, yc, w, h = (min(max(v, 0.0), 1.0) for v in (xc, yc, w, h))
            f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy(src, dst)


def resize_max_side(img, max_side):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    k = max_side / s
    return cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)


def main():
    with open(os.path.join(SRC, "data.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]

    if os.path.isdir(DST):
        shutil.rmtree(DST)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(DST, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(DST, sp, "labels"), exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    # ---- val/test (без изменений, симлинки) ----
    for sp in ("val", "test"):
        src_img = os.path.join(SRC, sp, "images")
        src_lbl = os.path.join(SRC, sp, "labels")
        for fn in os.listdir(src_img):
            sp_path = os.path.join(src_img, fn)
            real = os.path.realpath(sp_path)
            link_or_copy(real, os.path.join(DST, sp, "images", fn))
        for fn in os.listdir(src_lbl):
            link_or_copy(os.path.join(src_lbl, fn), os.path.join(DST, sp, "labels", fn))

    # ---- train: оригиналы + 2 аугментации ----
    src_img = os.path.join(SRC, "train", "images")
    src_lbl = os.path.join(SRC, "train", "labels")
    dst_img = os.path.join(DST, "train", "images")
    dst_lbl = os.path.join(DST, "train", "labels")

    pipeline = get_pipeline()

    orig_class_count = Counter()
    aug_class_count = Counter()
    orig_size_pairs = []
    aug_size_pairs = []

    img_files = sorted(os.listdir(src_img))

    n_orig = 0
    n_aug = 0
    n_aug_skipped_no_bbox = 0
    n_bbox_lost = 0

    for fn in tqdm(img_files, desc="augment train"):
        stem, ext = os.path.splitext(fn)
        src_img_path = os.path.realpath(os.path.join(src_img, fn))
        src_lbl_path = os.path.join(src_lbl, stem + ".txt")
        bboxes, labels = load_yolo_labels(src_lbl_path)
        for cid in labels:
            orig_class_count[cid] += 1
        for bx in bboxes:
            orig_size_pairs.append((bx[2], bx[3]))

        # 1) Оригинал — симлинк на исходник, метка копируется
        link_or_copy(src_img_path, os.path.join(dst_img, fn))
        link_or_copy(src_lbl_path, os.path.join(dst_lbl, stem + ".txt"))
        n_orig += 1

        # 2) N аугментаций
        try:
            img = cv2.imread(src_img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = resize_max_side(img, MAX_SIDE)
        except Exception:
            continue

        for i in range(1, N_AUG_PER_IMAGE + 1):
            try:
                out = pipeline(image=img, bboxes=bboxes, class_labels=labels)
            except Exception:
                continue
            new_bboxes = out["bboxes"]
            new_labels = out["class_labels"]
            # фильтр мелких bbox
            kept = [
                (cid, bx)
                for cid, bx in zip(new_labels, new_bboxes)
                if bx[2] * bx[3] >= MIN_BBOX_AREA
            ]
            n_bbox_lost += (len(new_bboxes) - len(kept))
            if not kept:
                n_aug_skipped_no_bbox += 1
                continue
            aug_img = out["image"]
            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            aug_name = f"{stem}_aug{i}.jpg"
            cv2.imwrite(os.path.join(dst_img, aug_name), aug_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            kept_labels = [c for c, _ in kept]
            kept_bboxes = [list(bx) for _, bx in kept]
            save_yolo_labels(os.path.join(dst_lbl, f"{stem}_aug{i}.txt"), kept_bboxes, kept_labels)
            for cid in kept_labels:
                aug_class_count[cid] += 1
            for bx in kept_bboxes:
                aug_size_pairs.append((bx[2], bx[3]))
            n_aug += 1

    # ---- data.yaml ----
    new_cfg = dict(cfg)
    new_cfg["path"] = DST
    with open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(new_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"\noriginals: {n_orig}, augmented saved: {n_aug}, aug skipped (no bbox left): {n_aug_skipped_no_bbox}")
    print(f"bbox dropped due to area<{MIN_BBOX_AREA}: {n_bbox_lost}")

    # ---- Визуализация 4.1: до/после ----
    sample = random.sample(img_files, min(10, len(img_files)))
    fig, axes = plt.subplots(10, 3, figsize=(15, 50))
    for row, fn in enumerate(sample):
        stem, _ = os.path.splitext(fn)
        orig_p = os.path.realpath(os.path.join(src_img, fn))
        bboxes, labels = load_yolo_labels(os.path.join(src_lbl, stem + ".txt"))
        # original
        img = cv2.cvtColor(cv2.imread(orig_p), cv2.COLOR_BGR2RGB)
        img = resize_max_side(img, MAX_SIDE)
        axes[row, 0].imshow(img)
        h, w = img.shape[:2]
        for cid, (xc, yc, bw, bh) in zip(labels, bboxes):
            x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
            axes[row, 0].add_patch(Rectangle((x1, y1), bw * w, bh * h, fill=False, edgecolor="lime", linewidth=2))
        axes[row, 0].set_title("оригинал" if row == 0 else "")
        axes[row, 0].axis("off")
        # 2 аугментации
        for col in (1, 2):
            try:
                out = pipeline(image=img, bboxes=bboxes, class_labels=labels)
                aug = out["image"]
                axes[row, col].imshow(aug)
                hh, ww = aug.shape[:2]
                for cid, (xc, yc, bw, bh) in zip(out["class_labels"], out["bboxes"]):
                    x1, y1 = (xc - bw / 2) * ww, (yc - bh / 2) * hh
                    axes[row, col].add_patch(Rectangle((x1, y1), bw * ww, bh * hh, fill=False, edgecolor="red", linewidth=2))
                if row == 0:
                    axes[row, col].set_title(f"аугментация {col}")
            except Exception:
                pass
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "augmentation_examples.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Визуализация 4.2: распределение классов до/после ----
    items = sorted(orig_class_count.items())
    cls_ids = [c for c, _ in items if c < len(classes)]
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(cls_ids))
    width = 0.4
    ax.bar(x - width / 2, [orig_class_count.get(c, 0) for c in cls_ids], width, label="оригинал (train)")
    ax.bar(x + width / 2, [orig_class_count.get(c, 0) + aug_class_count.get(c, 0) for c in cls_ids], width, label="после аугментации")
    ax.set_xticks(x)
    ax.set_xticklabels([classes[c] for c in cls_ids], rotation=45, ha="right")
    ax.set_ylabel("Число аннотаций")
    ax.set_title("Распределение классов: оригинал vs аугментированный train")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "class_distribution_after_aug.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Визуализация 4.3: размеры bbox ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if orig_size_pairs:
        ow, oh = zip(*orig_size_pairs)
        axes[0].scatter(ow, oh, s=3, alpha=0.3)
    axes[0].set_title(f"Оригинал (n={len(orig_size_pairs)})")
    axes[0].set_xlabel("width (норм.)"); axes[0].set_ylabel("height (норм.)")
    if aug_size_pairs:
        aw, ah = zip(*aug_size_pairs)
        axes[1].scatter(aw, ah, s=3, alpha=0.3, color="orange")
    axes[1].set_title(f"После аугментации (n={len(aug_size_pairs)})")
    axes[1].set_xlabel("width (норм.)"); axes[1].set_ylabel("height (норм.)")
    fig.suptitle("Размеры bbox: до и после аугментации")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "bbox_sizes_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Сводка ----
    summary = {
        "originals": n_orig,
        "augmented": n_aug,
        "skipped_no_bbox_after_aug": n_aug_skipped_no_bbox,
        "bbox_dropped_small": n_bbox_lost,
        "train_total_after": n_orig + n_aug,
        "orig_annotations": int(sum(orig_class_count.values())),
        "aug_annotations": int(sum(aug_class_count.values())),
        "total_annotations_after": int(sum(orig_class_count.values()) + sum(aug_class_count.values())),
        "class_orig": {classes[c]: int(orig_class_count[c]) for c in cls_ids},
        "class_after": {classes[c]: int(orig_class_count[c] + aug_class_count[c]) for c in cls_ids},
    }
    with open(os.path.join(RES, "summary.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
