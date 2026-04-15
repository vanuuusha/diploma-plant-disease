"""
Oversampling редких классов через агрессивные аугментации.
Работает поверх dataset_augmented/train. Создаёт dataset_balanced/train, val и test
полностью копируются (симлинками) из dataset_augmented.
"""

import os
import shutil
import random
from collections import Counter, defaultdict
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

SRC = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset_augmented"
DST = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset_balanced"
RES = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_05"
SEED = 42
MAX_REPEAT_PER_IMAGE = 10
TARGET_FRACTION_OF_MEDIAN = 0.8  # итоговая цель (task_05 + task_06) — 80% медианы
BALANCE_FRACTION_OF_GAP = 0.7    # oversampling закрывает 70% дефицита; остальные 30% — Diffusion (task_06)
MAX_SIDE = 1024
MIN_BBOX_AREA = 0.001

random.seed(SEED)
np.random.seed(SEED)


def get_aggressive_pipeline():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.4),
            A.RandomRotate90(p=0.4),
            A.Affine(translate_percent=0.15, scale=(0.8, 1.2), rotate=(-30, 30), p=0.7),
            A.RandomResizedCrop(size=(640, 640), scale=(0.6, 1.0), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.5),
            A.ElasticTransform(alpha=30, sigma=5, p=0.3),
            A.GridDistortion(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(std_range=(0.04, 0.2), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.3),
            A.RandomShadow(p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"],
            min_area=0, min_visibility=0.1, clip=True,
        ),
        seed=SEED,
    )


def load_yolo(lbl_path):
    bboxes, labels = [], []
    with open(lbl_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5: continue
            labels.append(int(float(p[0])))
            bboxes.append(list(map(float, p[1:])))
    return bboxes, labels


def save_yolo(lbl_path, bboxes, labels):
    with open(lbl_path, "w") as f:
        for cid, (xc, yc, w, h) in zip(labels, bboxes):
            xc, yc, w, h = (min(max(v, 0.0), 1.0) for v in (xc, yc, w, h))
            f.write(f"{int(cid)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def link_or_copy(src, dst):
    if os.path.exists(dst): return
    src = os.path.realpath(src)
    try: os.symlink(src, dst)
    except OSError: shutil.copy(src, dst)


def resize_max_side(img, max_side):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side: return img
    k = max_side / s
    return cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)


def main():
    with open(os.path.join(SRC, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]

    if os.path.isdir(DST): shutil.rmtree(DST)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(DST, sp, "images"))
        os.makedirs(os.path.join(DST, sp, "labels"))
    os.makedirs(RES, exist_ok=True)

    # val/test — симлинки
    for sp in ("val", "test"):
        for fn in os.listdir(os.path.join(SRC, sp, "images")):
            link_or_copy(os.path.join(SRC, sp, "images", fn), os.path.join(DST, sp, "images", fn))
        for fn in os.listdir(os.path.join(SRC, sp, "labels")):
            link_or_copy(os.path.join(SRC, sp, "labels", fn), os.path.join(DST, sp, "labels", fn))

    # train — симлинки на всё, что было в augmented
    src_img = os.path.join(SRC, "train", "images")
    src_lbl = os.path.join(SRC, "train", "labels")
    dst_img = os.path.join(DST, "train", "images")
    dst_lbl = os.path.join(DST, "train", "labels")

    cls_to_imgs = defaultdict(list)  # cid -> [stem,...]
    class_count_before = Counter()

    for fn in sorted(os.listdir(src_img)):
        stem, ext = os.path.splitext(fn)
        link_or_copy(os.path.join(src_img, fn), os.path.join(dst_img, fn))
        lbl_path = os.path.join(src_lbl, stem + ".txt")
        if os.path.exists(lbl_path):
            link_or_copy(lbl_path, os.path.join(dst_lbl, stem + ".txt"))
            _, labs = load_yolo(lbl_path)
            for c in labs:
                class_count_before[c] += 1
            for c in set(labs):
                cls_to_imgs[c].append((stem, ext))

    # ---- Анализ дисбаланса ----
    counts = {c: class_count_before[c] for c in range(len(classes)) if class_count_before[c] > 0}
    median_n = float(np.median(list(counts.values())))
    final_target = int(median_n * TARGET_FRACTION_OF_MEDIAN)
    # На этап oversampling закрываем только BALANCE_FRACTION_OF_GAP от дефицита.
    # Остаток закроет Diffusion (task_06). Цель по классам — индивидуальная.
    per_class_targets = {c: int(n + BALANCE_FRACTION_OF_GAP * max(0, final_target - n)) for c, n in counts.items()}
    print(f"Медиана: {median_n:.0f}, финальная цель (после task_06) ≥ {final_target}")
    print(f"Oversampling закрывает {BALANCE_FRACTION_OF_GAP * 100:.0f}% дефицита; остальное — Diffusion.")
    rare = [c for c in counts if counts[c] < per_class_targets[c]]
    print(f"Классы для oversampling: {[(classes[c], counts[c], per_class_targets[c]) for c in rare]}")

    pipeline = get_aggressive_pipeline()
    class_count_after = Counter(class_count_before)
    repeats = defaultdict(int)  # stem -> count
    rare_examples = defaultdict(list)  # cid -> [(orig_path, [aug_paths...])]

    for cid in rare:
        stems = cls_to_imgs[cid][:]
        random.Random(SEED + cid).shuffle(stems)
        target = per_class_targets[cid]
        deficit = target - class_count_after[cid]
        idx = 0
        # цикл по изображениям, прибавляя ann этого класса
        while class_count_after[cid] < target and stems:
            stem, ext = stems[idx % len(stems)]
            idx += 1
            if repeats[stem] >= MAX_REPEAT_PER_IMAGE:
                # удалить из списка, если все исчерпаны — выйти
                stems = [s for s in stems if repeats[s[0]] < MAX_REPEAT_PER_IMAGE]
                if not stems: break
                continue
            src_img_path = os.path.realpath(os.path.join(src_img, stem + ext))
            src_lbl_path = os.path.join(src_lbl, stem + ".txt")
            try:
                img = cv2.imread(src_img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = resize_max_side(img, MAX_SIDE)
            except Exception:
                continue
            bboxes, labels = load_yolo(src_lbl_path)
            try:
                out = pipeline(image=img, bboxes=bboxes, class_labels=labels)
            except Exception:
                continue
            new_bboxes = [list(b) for b in out["bboxes"]]
            new_labels = list(out["class_labels"])
            kept = [(c, b) for c, b in zip(new_labels, new_bboxes) if b[2] * b[3] >= MIN_BBOX_AREA]
            if not kept: continue
            i = repeats[stem] + 1
            new_stem = f"{stem}_bal{i}"
            cv2.imwrite(os.path.join(dst_img, new_stem + ".jpg"),
                        cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            kl = [c for c, _ in kept]; kb = [b for _, b in kept]
            save_yolo(os.path.join(dst_lbl, new_stem + ".txt"), kb, kl)
            for c in kl:
                class_count_after[c] += 1
            repeats[stem] += 1
            if len(rare_examples[cid]) < 1:
                rare_examples[cid].append((src_img_path, []))
            if len(rare_examples[cid][0][1]) < 3:
                rare_examples[cid][0][1].append(os.path.join(dst_img, new_stem + ".jpg"))

        print(f"  {classes[cid]}: было {counts[cid]} → стало {class_count_after[cid]}")

    # ---- data.yaml ----
    new_cfg = dict(cfg); new_cfg["path"] = DST
    with open(os.path.join(DST, "data.yaml"), "w") as f:
        yaml.safe_dump(new_cfg, f, allow_unicode=True, sort_keys=False)

    # ---- Визуализации ----
    cls_ids_present = sorted(counts.keys())
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(cls_ids_present)); w = 0.4
    before = [class_count_before[c] for c in cls_ids_present]
    after = [class_count_after[c] for c in cls_ids_present]
    ax.bar(x - w/2, before, w, label="до балансировки")
    ax.bar(x + w/2, after, w, label="после балансировки")
    ax.set_xticks(x)
    ax.set_xticklabels([classes[c] for c in cls_ids_present], rotation=45, ha="right")
    ax.set_ylabel("Аннотации")
    ax.set_title("Балансировка редких классов: до vs после")
    for i, (b, a) in enumerate(zip(before, after)):
        if b > 0 and a != b:
            ax.text(i + w/2, a, f"+{(a/b-1)*100:.0f}%", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RES, "balance_comparison.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    # imbalance ratio
    ir_before = max(before) / min(before)
    ir_after = max(after) / min(after)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["До", "После"], [ir_before, ir_after], color=["#ff7f7f", "#7fbf7f"])
    for i, v in enumerate([ir_before, ir_after]):
        ax.text(i, v, f"{v:.2f}×", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_ylabel("Imbalance ratio (max/min)")
    ax.set_title("Дисбаланс классов: до и после балансировки")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "imbalance_ratio.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    # примеры oversampled
    if rare_examples:
        n_rows = len(rare_examples)
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
        if n_rows == 1: axes = axes[None, :]
        for r, (cid, lst) in enumerate(rare_examples.items()):
            orig_p, augs = lst[0]
            try:
                im = cv2.cvtColor(cv2.imread(orig_p), cv2.COLOR_BGR2RGB)
                im = resize_max_side(im, MAX_SIDE)
                axes[r, 0].imshow(im); axes[r, 0].set_title(f"{classes[cid]} (ориг.)"); axes[r, 0].axis("off")
            except Exception:
                axes[r, 0].axis("off")
            for j, p in enumerate(augs[:3], 1):
                try:
                    im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                    axes[r, j].imshow(im); axes[r, j].set_title(f"aug {j}"); axes[r, j].axis("off")
                except Exception:
                    axes[r, j].axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(RES, "oversampling_examples.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    summary = {
        "median": median_n,
        "target": target,
        "imbalance_ratio_before": float(ir_before),
        "imbalance_ratio_after": float(ir_after),
        "classes": {
            classes[c]: {
                "before": int(class_count_before[c]),
                "after": int(class_count_after[c]),
                "delta": int(class_count_after[c] - class_count_before[c]),
            } for c in cls_ids_present
        },
    }
    with open(os.path.join(RES, "summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
