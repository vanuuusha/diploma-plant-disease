"""
EDA датасета в YOLO-формате.
Собирает статистики, рисует все визуализации, описанные в TASK_03.
"""

import os
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import yaml
from PIL import Image
from tqdm import tqdm

# Хардкод-конфиг
DATASET = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset"
OUT_DIR = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_03"
EX_DIR = os.path.join(OUT_DIR, "examples")
SPLITS = ("train", "val", "test")
RNG = random.Random(42)

plt.rcParams.update({"font.size": 12, "figure.dpi": 100, "savefig.dpi": 300})
sns.set_palette("Set2")


def load_data():
    with open(os.path.join(DATASET, "data.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]
    return classes


def collect():
    """Собрать все аннотации и метаданные изображений."""
    classes = load_data()
    rows = []  # split, img_path, w, h, cls_id, xc, yc, bw, bh
    img_meta = []  # split, img_path, w, h, n_bbox
    for split in SPLITS:
        img_dir = os.path.join(DATASET, split, "images")
        lbl_dir = os.path.join(DATASET, split, "labels")
        for fn in tqdm(sorted(os.listdir(img_dir)), desc=f"scan {split}"):
            stem = os.path.splitext(fn)[0]
            img_path = os.path.join(img_dir, fn)
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                continue
            n_bbox = 0
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) != 5:
                            continue
                        cid = int(p[0])
                        xc, yc, bw, bh = map(float, p[1:])
                        rows.append((split, img_path, w, h, cid, xc, yc, bw, bh))
                        n_bbox += 1
            img_meta.append((split, img_path, w, h, n_bbox))
    return classes, rows, img_meta


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EX_DIR, exist_ok=True)
    classes, rows, img_meta = collect()
    arr = np.array([(c, xc, yc, bw, bh) for _, _, _, _, c, xc, yc, bw, bh in rows], dtype=np.float64)
    splits = np.array([r[0] for r in rows])
    img_split = np.array([m[0] for m in img_meta])
    n_bbox_per_img = np.array([m[4] for m in img_meta])
    img_w = np.array([m[2] for m in img_meta])
    img_h = np.array([m[3] for m in img_meta])

    # ---- 1. Общая статистика ----
    summary = {}
    summary["num_classes"] = len(classes)
    summary["num_images_total"] = len(img_meta)
    summary["num_annotations_total"] = len(rows)
    for sp in SPLITS:
        summary[f"images_{sp}"] = int((img_split == sp).sum())
        summary[f"annotations_{sp}"] = int((splits == sp).sum())
    summary["bbox_per_image_mean"] = float(np.mean(n_bbox_per_img))
    summary["bbox_per_image_median"] = float(np.median(n_bbox_per_img))
    summary["bbox_per_image_min"] = int(np.min(n_bbox_per_img))
    summary["bbox_per_image_max"] = int(np.max(n_bbox_per_img))
    summary["img_width_min"] = int(np.min(img_w))
    summary["img_width_max"] = int(np.max(img_w))
    summary["img_height_min"] = int(np.min(img_h))
    summary["img_height_max"] = int(np.max(img_h))
    summary["img_width_mean"] = float(np.mean(img_w))
    summary["img_height_mean"] = float(np.mean(img_h))

    # ---- 2. Распределение классов ----
    class_counts_total = Counter(arr[:, 0].astype(int).tolist())
    class_counts_split = {sp: Counter() for sp in SPLITS}
    for sp, cid in zip(splits, arr[:, 0].astype(int)):
        class_counts_split[sp][int(cid)] += 1

    nz = [v for v in class_counts_total.values() if v > 0]
    summary["imbalance_ratio"] = float(max(nz) / min(nz)) if nz else 0
    summary["coef_variation_classes"] = float(np.std(nz) / np.mean(nz)) if nz else 0

    # ---- CSV сводка ----
    csv_path = os.path.join(OUT_DIR, "dataset_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["class_id", "class_name", "train", "val", "test", "total"])
        for cid, name in enumerate(classes):
            tr = class_counts_split["train"][cid]
            va = class_counts_split["val"][cid]
            te = class_counts_split["test"][cid]
            w.writerow([cid, name, tr, va, te, tr + va + te])
    print(f"CSV: {csv_path}")

    # ---- Plot helpers ----
    def save(fig, name):
        path = os.path.join(OUT_DIR, name)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {name}")

    # 3.1 Гистограмма распределения классов
    items = sorted(class_counts_total.items(), key=lambda x: x[1], reverse=True)
    items = [(cid, n) for cid, n in items if n > 0]
    fig, ax = plt.subplots(figsize=(12, 7))
    names = [classes[c] for c, _ in items]
    vals = [n for _, n in items]
    bars = ax.barh(names[::-1], vals[::-1], color=sns.color_palette("Set2", len(items))[::-1])
    ax.set_xlabel("Число аннотаций")
    ax.set_title("Распределение классов в датасете")
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + max(vals) * 0.005, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=10)
    save(fig, "class_distribution.png")

    # 3.2 По сплитам
    fig, ax = plt.subplots(figsize=(14, 7))
    width = 0.27
    cls_names_present = [c for c, _ in items]
    x = np.arange(len(cls_names_present))
    for i, sp in enumerate(SPLITS):
        ys = [class_counts_split[sp][c] for c in cls_names_present]
        ax.bar(x + (i - 1) * width, ys, width, label=sp)
    ax.set_xticks(x)
    ax.set_xticklabels([classes[c] for c in cls_names_present], rotation=45, ha="right")
    ax.set_ylabel("Число аннотаций")
    ax.set_title("Распределение классов по сплитам")
    ax.legend()
    save(fig, "class_distribution_by_split.png")

    # 3.3 Bbox per image
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(0, n_bbox_per_img.max() + 2) - 0.5
    ax.hist(n_bbox_per_img, bins=bins, color=sns.color_palette("Set2")[0], edgecolor="black")
    ax.set_xlabel("Число bbox на изображение")
    ax.set_ylabel("Число изображений")
    ax.set_title("Распределение числа bbox на изображение")
    save(fig, "bbox_per_image.png")

    # 3.4 Размеры bbox scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    cids = arr[:, 0].astype(int)
    palette = sns.color_palette("tab20", len(classes))
    for cid in sorted(set(cids.tolist())):
        m = cids == cid
        ax.scatter(arr[m, 3], arr[m, 4], s=4, alpha=0.4, label=classes[cid], color=palette[cid])
    ax.set_xlabel("Ширина bbox (норм.)")
    ax.set_ylabel("Высота bbox (норм.)")
    ax.set_title("Размеры bbox (нормализованные)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=2)
    save(fig, "bbox_sizes.png")

    # 3.5 Heatmap позиций
    H, xedges, yedges = np.histogram2d(arr[:, 1], arr[:, 2], bins=50, range=[[0, 1], [0, 1]])
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(H.T, origin="lower", extent=[0, 1, 0, 1], cmap="hot", aspect="equal")
    ax.set_xlabel("x_center (норм.)")
    ax.set_ylabel("y_center (норм.)")
    ax.set_title("Тепловая карта центров bbox")
    plt.colorbar(im, ax=ax, label="Число bbox")
    save(fig, "bbox_heatmap.png")

    # 3.6 Разрешения
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(img_w, img_h, s=10, alpha=0.5, color=sns.color_palette("Set2")[1])
    ax.set_xlabel("Ширина (пиксели)")
    ax.set_ylabel("Высота (пиксели)")
    ax.set_title(f"Разрешения изображений (n={len(img_w)})")
    save(fig, "image_resolutions.png")

    # 3.7 Примеры по классам
    cls_to_imgs = defaultdict(list)
    for split, img_path, w, h, cid, xc, yc, bw, bh in rows:
        cls_to_imgs[cid].append((img_path, w, h, xc, yc, bw, bh))

    chosen = {}
    for cid, lst in cls_to_imgs.items():
        if lst:
            chosen[cid] = RNG.choice(lst)

    def draw_bboxes(ax, img_path, all_bboxes, classes_map):
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            arr_im = np.array(im)
            ih, iw = arr_im.shape[:2]
        ax.imshow(arr_im)
        for cid, xc, yc, bw, bh in all_bboxes:
            x1 = (xc - bw / 2) * iw
            y1 = (yc - bh / 2) * ih
            ax.add_patch(Rectangle((x1, y1), bw * iw, bh * ih, fill=False, edgecolor="red", linewidth=2))
        ax.set_axis_off()

    # отдельные примеры
    img_to_bboxes = defaultdict(list)
    for split, img_path, w, h, cid, xc, yc, bw, bh in rows:
        img_to_bboxes[img_path].append((cid, xc, yc, bw, bh))

    sorted_cids = sorted(chosen.keys())
    cols = 4
    rows_n = (len(sorted_cids) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 5, rows_n * 5))
    axes = np.array(axes).reshape(-1)
    for i, cid in enumerate(sorted_cids):
        img_path, *_ = chosen[cid]
        draw_bboxes(axes[i], img_path, img_to_bboxes[img_path], classes)
        axes[i].set_title(classes[cid], fontsize=11)
        # Per-class single example
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        draw_bboxes(ax2, img_path, img_to_bboxes[img_path], classes)
        ax2.set_title(classes[cid])
        safe = classes[cid].replace(" ", "_").replace("/", "_")
        save(fig2, f"examples/class_{safe}.png")
    for j in range(len(sorted_cids), len(axes)):
        axes[j].set_axis_off()
    save(fig, "class_examples_grid.png")

    # 3.8 Boxplot площадей
    fig, ax = plt.subplots(figsize=(14, 7))
    data_box = []
    labels_box = []
    for cid in sorted(set(arr[:, 0].astype(int).tolist())):
        m = arr[:, 0].astype(int) == cid
        if m.sum() > 0:
            data_box.append(arr[m, 3] * arr[m, 4])
            labels_box.append(classes[cid])
    ax.boxplot(data_box, labels=labels_box)
    ax.set_yscale("log")
    ax.set_ylabel("Площадь bbox (норм., log)")
    ax.set_title("Распределение площадей bbox по классам")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, "bbox_area_by_class.png")

    print("\n=== СВОДКА ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\n=== Аннотаций по классам (всего) ===")
    for cid in sorted(set(arr[:, 0].astype(int).tolist())):
        print(f"  {cid:2d}  {classes[cid]:35s}  {class_counts_total[cid]}")


if __name__ == "__main__":
    main()
