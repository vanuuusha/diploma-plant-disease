"""
Shared utilities for chapter 3 detector experiments.

Provides functions to build protocol-compliant artifacts in
code/results/task_NN/<detector>_<variant>/:
    - metrics.csv (epoch, train_loss, val_loss, val_precision, val_recall, val_mAP50, val_mAP5095, lr, time_sec)
    - learning_curves.png (2x2 subplots)
    - confusion_matrix.png (9x9 normalized)
    - per_class_map.csv
    - predictions_examples/ (rendered on fixed qualitative sample)
    - fps_measurement.json (100 runs, batch=1, imgsz=640)
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/vanusha/diplom/diploma-plant-disease")
QUALITATIVE_SAMPLE = ROOT / "code/docs/chapter3_qualitative_sample.txt"
TEST_IMG_DIR = ROOT / "code/data/dataset/test/images"
TEST_LBL_DIR = ROOT / "code/data/dataset/test/labels"

CLASS_NAMES = [
    "Недостаток P2O5",
    "Листовая (бурая) ржавчина",
    "Мучнистая роса",
    "Пиренофороз",
    "Фузариоз",
    "Корневая гниль",
    "Септориоз",
    "Недостаток N",
    "Повреждение заморозками",
]

DATASET_VARIANTS = {
    "baseline": ROOT / "code/data/dataset/data.yaml",
    "aug_geom": ROOT / "code/data/dataset_augmented/data.yaml",
    "aug_oversample": ROOT / "code/data/dataset_balanced/data.yaml",
    "aug_diffusion": ROOT / "code/data/dataset_final/data.yaml",
}


def qualitative_filenames() -> list[str]:
    """Read qualitative sample filenames (no comments)."""
    out = []
    for ln in QUALITATIVE_SAMPLE.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_metrics_csv(
    out_path: Path,
    rows: Sequence[dict],
) -> None:
    """
    Save metrics.csv in the protocol-compliant format.

    Each row dict keys expected:
        epoch, train_loss, val_loss, val_precision, val_recall,
        val_mAP50, val_mAP5095, lr, time_sec
    """
    cols = [
        "epoch", "train_loss", "val_loss", "val_precision", "val_recall",
        "val_mAP50", "val_mAP5095", "lr", "time_sec",
    ]
    ensure_dir(out_path.parent)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def plot_learning_curves(metrics_csv: Path, out_png: Path, title: str = "") -> None:
    """Read metrics.csv, draw 2x2 subplots, save PNG."""
    import pandas as pd

    df = pd.read_csv(metrics_csv)
    if df.empty:
        return

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=120)
    axs = axs.flatten()

    # Loss panel
    ax = axs[0]
    if "train_loss" in df.columns and df["train_loss"].notna().any():
        ax.plot(df["epoch"], df["train_loss"], label="train_loss", color="tab:blue")
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        ax.plot(df["epoch"], df["val_loss"], label="val_loss", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Precision / Recall
    ax = axs[1]
    if "val_precision" in df.columns:
        ax.plot(df["epoch"], df["val_precision"], label="val_precision", color="tab:green")
    if "val_recall" in df.columns:
        ax.plot(df["epoch"], df["val_recall"], label="val_recall", color="tab:red")
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title("Precision / Recall (val)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # mAP@50
    ax = axs[2]
    if "val_mAP50" in df.columns:
        ax.plot(df["epoch"], df["val_mAP50"], color="tab:purple")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP@50")
    ax.set_title("mAP@50 (val)")
    ax.grid(True, alpha=0.3)

    # mAP@50-95
    ax = axs[3]
    if "val_mAP5095" in df.columns:
        ax.plot(df["epoch"], df["val_mAP5095"], color="tab:brown")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP@50-95")
    ax.set_title("mAP@50-95 (val)")
    ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    out_png: Path,
    title: str = "Confusion matrix (row-normalized)",
) -> None:
    """9x9 or (N+1) confusion matrix. Rows — true, cols — pred. Normalize per row."""
    m = np.asarray(matrix, dtype=float)
    row_sum = m.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    m = m / row_sum

    n = m.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(7, n * 0.8)), dpi=120)
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = list(class_names)
    if n == len(class_names) + 1:
        labels = labels + ["background"]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            if m[i, j] > 0.01:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        color="white" if m[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_per_class_map(out_csv: Path, rows: Sequence[dict]) -> None:
    """rows: list of dicts with keys class_id, class_name, n_test_instances, mAP50, mAP50_95."""
    cols = ["class_id", "class_name", "n_test_instances", "mAP50", "mAP50_95"]
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def save_fps(out_json: Path, detector: str, variant: str, latencies_ms: Sequence[float]) -> None:
    import torch

    lat = np.asarray(latencies_ms, dtype=float)
    data = {
        "detector": detector,
        "variant": variant,
        "imgsz": 640,
        "batch": 1,
        "precision": "fp32",
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "warmup_runs": 20,
        "measure_runs": int(len(lat)),
        "mean_ms": float(lat.mean()),
        "std_ms": float(lat.std()),
        "median_ms": float(np.median(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "fps": float(1000.0 / lat.mean()),
    }
    ensure_dir(out_json.parent)
    out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def render_predictions_examples(
    out_dir: Path,
    predict_fn,
    conf_thres: float = 0.25,
) -> None:
    """
    Render predictions on the qualitative sample images.

    predict_fn(image_path: str) -> list[dict(box=[x1,y1,x2,y2], cls=int, score=float)]
    Drawing: GT boxes — thin green, PRED boxes — red with class:score.
    """
    import cv2

    ensure_dir(out_dir)
    for fname in qualitative_filenames():
        img_path = TEST_IMG_DIR / fname
        lbl_path = TEST_LBL_DIR / (Path(fname).stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        vis = img.copy()

        # GT (green thin)
        if lbl_path.exists():
            for ln in lbl_path.read_text().splitlines():
                parts = ln.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 1)

        # Pred (red)
        try:
            preds = predict_fn(str(img_path))
        except Exception as e:
            preds = []
            cv2.putText(vis, f"pred err: {e}"[:60], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for p in preds:
            if p.get("score", 1.0) < conf_thres:
                continue
            x1, y1, x2, y2 = map(int, p["box"])
            cls = p["cls"]
            score = p.get("score", 1.0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{cls}:{score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 2, y1), (0, 0, 255), -1)
            cv2.putText(vis, label, (x1 + 1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(str(out_dir / fname), vis)


def write_summary_csv(out_csv: Path, rows: Sequence[dict]) -> None:
    """Summary across 4 variants per detector."""
    cols = ["variant", "n_train", "mAP50", "mAP5095", "precision", "recall", "fps", "epochs"]
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def count_train_images(variant_key: str) -> int:
    """Count train images for a given dataset variant."""
    p = DATASET_VARIANTS[variant_key].parent / "train" / "images"
    if not p.exists():
        return 0
    return sum(1 for _ in p.iterdir())
