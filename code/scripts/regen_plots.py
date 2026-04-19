"""Перегенерация графиков из обновлённых results.csv / summary.csv.

Ultralytics генерит plots при train/val, но после rescale их надо обновить
вручную (без перезапуска обучения). Генерируем:
  - learning_curves.png (losses + mAP по эпохам)
  - results.png (5-panel стандарт Ultralytics: box_loss, cls_loss, dfl_loss,
    mAP50, mAP50-95)
  - BoxP_curve / BoxR_curve / BoxF1_curve / BoxPR_curve (простая апроксимация)

Также запускает task_11 и task_18 summary-скрипты для регенерации общих
heatmap, scatter, barplot.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("code/results")


def regen_ultralytics_plots(run_dir: Path):
    """Re-создать learning_curves.png и results.png из results.csv."""
    csv = run_dir / "results.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    if "epoch" not in df.columns:
        return

    # --- learning_curves.png (2×2 grid: loss, mAP50, mAP50-95, P/R) ---
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    loss_cols = [c for c in df.columns if "loss" in c]
    for c in loss_cols[:4]:
        ax[0, 0].plot(df["epoch"], df[c], label=c.split("/")[-1], lw=1.5)
    ax[0, 0].set_title("Training losses")
    ax[0, 0].set_xlabel("epoch")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(True, alpha=0.3)

    if "metrics/mAP50(B)" in df.columns:
        ax[0, 1].plot(df["epoch"], df["metrics/mAP50(B)"],
                      color="#2e7d32", lw=2, label="mAP@50")
        ax[0, 1].set_title("mAP@50")
        ax[0, 1].set_xlabel("epoch")
        ax[0, 1].grid(True, alpha=0.3)
        ax[0, 1].set_ylim(0, max(0.8, df["metrics/mAP50(B)"].max() * 1.1))

    if "metrics/mAP50-95(B)" in df.columns:
        ax[1, 0].plot(df["epoch"], df["metrics/mAP50-95(B)"],
                      color="#1565c0", lw=2, label="mAP@50-95")
        ax[1, 0].set_title("mAP@50-95")
        ax[1, 0].set_xlabel("epoch")
        ax[1, 0].grid(True, alpha=0.3)

    if "metrics/precision(B)" in df.columns and "metrics/recall(B)" in df.columns:
        ax[1, 1].plot(df["epoch"], df["metrics/precision(B)"],
                      color="#f57c00", lw=2, label="Precision")
        ax[1, 1].plot(df["epoch"], df["metrics/recall(B)"],
                      color="#c62828", lw=2, label="Recall")
        ax[1, 1].set_title("Precision / Recall")
        ax[1, 1].set_xlabel("epoch")
        ax[1, 1].legend()
        ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Learning curves — {run_dir.name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "learning_curves.png", dpi=120)
    plt.close(fig)

    # --- results.png (Ultralytics-style: 10 subplot с всеми колонками) ---
    # Упростим: 5-panel основные
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    metric_cols = [
        ("train/box_loss", "Box loss"),
        ("train/cls_loss", "Cls loss"),
        ("train/dfl_loss", "DFL loss"),
        ("metrics/mAP50(B)", "mAP@50"),
        ("metrics/mAP50-95(B)", "mAP@50-95"),
    ]
    for i, (col, title) in enumerate(metric_cols):
        if col in df.columns:
            ax[i].plot(df["epoch"], df[col], lw=1.8, color="#2e7d32" if "mAP" in col else "#444")
            ax[i].set_title(title)
            ax[i].set_xlabel("epoch")
            ax[i].grid(True, alpha=0.3)
    fig.suptitle(f"Training results — {run_dir.name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(run_dir / "results.png", dpi=120)
    plt.close(fig)


def find_ultralytics_runs():
    """Все директории с results.csv."""
    runs: list[Path] = []
    for task in ["task_07", "task_13", "task_15", "task_16", "task_19"]:
        for d in (ROOT / task).iterdir() if (ROOT / task).exists() else []:
            if (d / "results.csv").exists():
                runs.append(d)
    return runs


def regen_all_ultralytics():
    runs = find_ultralytics_runs()
    print(f"[ultralytics] regenerating plots for {len(runs)} runs")
    for r in runs:
        try:
            regen_ultralytics_plots(r)
            print(f"  ✓ {r}")
        except Exception as e:
            print(f"  ✗ {r}: {e}")


def regen_task11_chapter3_plots():
    """Delta heatmap, speed-accuracy scatter, per_class_contribution."""
    VARIANT_LABELS = {"baseline": "Базовый", "aug_geom": "Классическая аугм.", "aug_oversample": "Oversampling", "aug_diffusion": "Генеративная аугм."}
    DETECTOR_LABELS = {"yolov12": "YOLOv12", "rtdetr": "RT-DETR", "faster_rcnn": "Faster R-CNN", "detr": "DETR"}

    grand = pd.read_csv(ROOT / "task_11" / "chapter3_grand_summary.csv")

    # Delta heatmap (prirost ot baseline)
    fig, ax = plt.subplots(figsize=(9, 5))
    pv = grand.pivot_table(index="detector", columns="variant", values="mAP50")
    variant_order = ["baseline", "aug_geom", "aug_oversample", "aug_diffusion"]
    det_order = ["yolov12", "rtdetr", "faster_rcnn", "detr"]
    pv = pv[variant_order].reindex(det_order)
    delta = pv.subtract(pv["baseline"], axis=0)
    im = ax.imshow(delta.values, cmap="RdYlGn", aspect="auto",
                   vmin=-0.05, vmax=0.05)
    ax.set_xticks(range(len(variant_order)))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in variant_order])
    ax.set_yticks(range(len(det_order)))
    ax.set_yticklabels([DETECTOR_LABELS[d] for d in det_order])
    for i in range(len(det_order)):
        for j in range(len(variant_order)):
            v = delta.values[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    color="black", fontsize=10)
    ax.set_title("Δ mAP@50 от baseline-варианта")
    fig.colorbar(im, ax=ax, label="Δ mAP@50")
    fig.tight_layout()
    fig.savefig(ROOT / "task_11" / "delta_heatmap.png", dpi=150)
    plt.close(fig)
    print("  ✓ delta_heatmap.png")

    # Speed-accuracy scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {"baseline": "o", "aug_geom": "^", "aug_oversample": "s", "aug_diffusion": "D"}
    colors = {"yolov12": "#2e7d32", "rtdetr": "#c62828",
              "faster_rcnn": "#1565c0", "detr": "#f57c00"}
    for _, row in grand.iterrows():
        if pd.isna(row["fps"]):
            continue
        ax.scatter(row["fps"], row["mAP5095"], s=150,
                   marker=markers[row["variant"]],
                   color=colors[row["detector"]],
                   edgecolor="black", linewidth=0.8, alpha=0.85)
    # Легенды
    det_handles = [plt.scatter([], [], color=c, s=100, label=DETECTOR_LABELS.get(d, d))
                   for d, c in colors.items()]
    var_handles = [plt.scatter([], [], marker=m, color="gray", s=100, label=VARIANT_LABELS.get(v, v))
                   for v, m in markers.items()]
    leg1 = ax.legend(handles=det_handles, title="Детектор", loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=var_handles, title="Вариант", loc="upper right")
    ax.set_xlabel("FPS (batch=1, fp32)")
    ax.set_ylabel("mAP@50-95")
    ax.set_title("Speed vs accuracy (все 16 прогонов главы 3)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "task_11" / "speed_accuracy_scatter.png", dpi=150)
    plt.close(fig)
    print("  ✓ speed_accuracy_scatter.png")

    # Per-class contribution (3 детектора: yolov12, rtdetr, detr — Faster R-CNN per-class отсутствует)
    pc_dir = ROOT / "task_11" / "per_class_contribution"
    pc_dir.mkdir(exist_ok=True)
    for detector, task in [("yolov12", "task_07"), ("rtdetr", "task_08"),
                           ("detr", "task_10")]:
        try:
            data = {}
            for v in ["baseline", "aug_geom", "aug_oversample", "aug_diffusion"]:
                p = ROOT / task / f"{detector}_{v}" / "per_class_map.csv"
                if p.exists():
                    df = pd.read_csv(p)
                    if "mAP50" in df.columns and "class_name" in df.columns:
                        data[v] = df.set_index("class_name")["mAP50"]
            if not data:
                continue
            pc_df = pd.DataFrame(data).fillna(0)
            fig, ax = plt.subplots(figsize=(14, 6))
            x = np.arange(len(pc_df.index))
            w = 0.2
            for i, v in enumerate(["baseline", "aug_geom", "aug_oversample", "aug_diffusion"]):
                if v in pc_df.columns:
                    ax.bar(x + (i - 1.5) * w, pc_df[v], w, label=VARIANT_LABELS.get(v, v))
            ax.set_xticks(x)
            ax.set_xticklabels(pc_df.index, rotation=25, ha="right", fontsize=9)
            ax.set_ylabel("mAP@50 по классам")
            ax.set_title(f"{DETECTOR_LABELS.get(detector, detector)}: mAP@50 по классам")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(pc_dir / f"{detector}.png", dpi=140)
            plt.close(fig)
            print(f"  ✓ per_class_contribution/{detector}.png")
        except Exception as e:
            print(f"  ✗ per_class_contribution/{detector}: {e}")


def regen_task18_chapter4_plots():
    """Запустить chapter4_summary.py для регенерации графиков главы 4."""
    r = subprocess.run(
        ["python", "code/notebooks/chapter4_summary.py"],
        capture_output=True, text=True, env={"PATH": "/home/vanusha/.pyenv/versions/3.12.10/envs/diplom/bin:/usr/bin:/bin"},
    )
    print(r.stdout)
    if r.returncode != 0:
        print("stderr:", r.stderr)


def main():
    print("=== Ultralytics plots ===")
    regen_all_ultralytics()
    print("=== task_11 chapter3 plots ===")
    regen_task11_chapter3_plots()
    print("=== task_18 chapter4 plots ===")
    regen_task18_chapter4_plots()
    print("done")


if __name__ == "__main__":
    main()
