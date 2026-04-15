"""
Гистограмма распределения классов в dataset_final
(сравнение: dataset / dataset_augmented / dataset_balanced / dataset_final).
Используется для иллюстрации эффекта всего пайплайна подготовки данных.
"""

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

BASE = "/home/vanusha/diplom/diploma-plant-disease/code/data"
OUT = "/home/vanusha/diplom/diploma-plant-disease/code/results/dataset_final_hist.png"
OUT_COMPARE = "/home/vanusha/diplom/diploma-plant-disease/code/results/dataset_pipeline_comparison.png"

DATASETS = [
    ("dataset",           "1. Исходный"),
    ("dataset_augmented", "2. + Классическая аугментация (×3)"),
    ("dataset_balanced",  "3. + Oversampling (task_05)"),
    ("dataset_final",     "4. + Diffusion (итог)"),
]

sns.set_palette("Set2")


def count_train(path):
    cls_count = Counter()
    lbl_dir = os.path.join(path, "train", "labels")
    if not os.path.isdir(lbl_dir):
        return None, cls_count
    with open(os.path.join(path, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    for fn in os.listdir(lbl_dir):
        with open(os.path.join(lbl_dir, fn)) as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 5:
                    cls_count[int(float(p[0]))] += 1
    return cfg["names"], cls_count


def main():
    stages = []
    names = None
    for folder, label in DATASETS:
        full = os.path.join(BASE, folder)
        n, cc = count_train(full)
        if n is None:
            continue
        if names is None:
            names = n
        stages.append((label, cc))

    # ---- Финальная гистограмма: dataset_final ----
    final_label, final_cc = stages[-1]
    cls_ids = sorted([c for c in final_cc if final_cc[c] > 0])
    vals = [final_cc[c] for c in cls_ids]
    labels = [names[c] for c in cls_ids]

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(labels[::-1], vals[::-1], color=sns.color_palette("Set2", len(cls_ids))[::-1])
    ax.set_xlabel("Число аннотаций (train)", fontsize=12)
    ax.set_title(f"Распределение классов в dataset_final ({final_label.split('. ', 1)[1]})", fontsize=13)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + max(vals) * 0.005, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=10)
    # Подпись: imbalance ratio
    if vals:
        ir = max(vals) / min(vals)
        ax.text(0.02, 0.02, f"Imbalance ratio (max/min) = {ir:.2f}×", transform=ax.transAxes,
                fontsize=12, bbox=dict(facecolor="lightyellow", edgecolor="gray", alpha=0.8))
    fig.tight_layout(); fig.savefig(OUT, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT}")

    # ---- Grouped bar chart: все 4 стадии пайплайна рядом ----
    fig, ax = plt.subplots(figsize=(16, 8))
    n_stages = len(stages)
    n_cls = len(cls_ids)
    width = 0.2
    x = np.arange(n_cls)
    palette = sns.color_palette("Set2", n_stages)
    for i, (label, cc) in enumerate(stages):
        ys = [cc[c] for c in cls_ids]
        offset = (i - (n_stages - 1) / 2) * width
        ax.bar(x + offset, ys, width, label=label, color=palette[i])
    ax.set_xticks(x)
    ax.set_xticklabels([names[c] for c in cls_ids], rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Число аннотаций (train)", fontsize=12)
    ax.set_title("Эволюция распределения классов по этапам пайплайна", fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout(); fig.savefig(OUT_COMPARE, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_COMPARE}")

    # Таблица в консоль
    print(f"\n{'Класс':<30s} " + " ".join([f"{s[0]:>25s}" for s in stages]))
    for c in cls_ids:
        row = [f"{names[c]:<30s}"] + [f"{s[1][c]:>25d}" for s in stages]
        print(" ".join(row))


if __name__ == "__main__":
    main()
