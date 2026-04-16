"""
Task 11 summary aggregation per chapter3_protocol §7–8.

Reads 4 summary.csv from task_07..task_10, builds:
    - chapter3_grand_summary.csv (16 rows)
    - delta_heatmap.png
    - speed_accuracy_scatter.png
    - per_class_contribution/<detector>.png (4 files)
    - qualitative_grid.png
    - bootstrap_ci.json
    - final_table.csv

Called by: python chapter3_summary_script.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chapter3_common as cc

ROOT = cc.ROOT
OUT_DIR = ROOT / "code/results/task_11"
DETECTORS = ["yolov12", "rtdetr", "faster_rcnn", "detr"]
DETECTOR_TASK = {
    "yolov12": "task_07",
    "rtdetr": "task_08",
    "faster_rcnn": "task_09",
    "detr": "task_10",
}
VARIANTS = ["baseline", "aug_geom", "aug_oversample", "aug_diffusion"]


def load_grand_summary() -> pd.DataFrame:
    rows = []
    for det in DETECTORS:
        sp = ROOT / "code/results" / DETECTOR_TASK[det] / "summary.csv"
        if not sp.exists():
            continue
        df = pd.read_csv(sp)
        for _, r in df.iterrows():
            rows.append({
                "detector": det,
                "variant": r.get("variant", ""),
                "n_train": r.get("n_train", ""),
                "mAP50": r.get("mAP50", np.nan),
                "mAP5095": r.get("mAP5095", np.nan),
                "precision": r.get("precision", np.nan),
                "recall": r.get("recall", np.nan),
                "fps": r.get("fps", np.nan),
                "epochs": r.get("epochs", np.nan),
            })
    out = pd.DataFrame(rows)
    cc.ensure_dir(OUT_DIR)
    out.to_csv(OUT_DIR / "chapter3_grand_summary.csv", index=False)
    print(f"wrote {OUT_DIR}/chapter3_grand_summary.csv  ({len(out)} rows)")
    return out


def delta_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(index="detector", columns="variant", values="mAP50", aggfunc="first")
    pivot = pivot.reindex(DETECTORS).reindex(columns=VARIANTS)
    base = pivot["baseline"].astype(float)
    delta = pivot.sub(base, axis=0) * 100.0  # percentage points

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    im = ax.imshow(delta.values, cmap="RdYlGn", vmin=-10, vmax=10, aspect="auto")
    ax.set_xticks(range(len(VARIANTS))); ax.set_xticklabels(VARIANTS, rotation=20)
    ax.set_yticks(range(len(DETECTORS))); ax.set_yticklabels(DETECTORS)
    for i in range(len(DETECTORS)):
        for j in range(len(VARIANTS)):
            v = delta.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        color="black" if abs(v) < 7 else "white", fontsize=10)
    fig.colorbar(im, ax=ax, label="Δ mAP@50, п.п. к baseline")
    ax.set_title("Прирост mAP@50 относительно baseline")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "delta_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR}/delta_heatmap.png")


def speed_accuracy_scatter(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    colors = {"yolov12": "tab:blue", "rtdetr": "tab:orange",
              "faster_rcnn": "tab:green", "detr": "tab:red"}
    markers = {"baseline": "o", "aug_geom": "s",
               "aug_oversample": "^", "aug_diffusion": "D"}

    for _, r in df.iterrows():
        try:
            x = float(r["fps"]); y = float(r["mAP5095"])
        except Exception:
            continue
        if np.isnan(x) or np.isnan(y):
            continue
        ax.scatter(x, y, s=140, c=colors.get(r["detector"], "black"),
                   marker=markers.get(r["variant"], "x"), edgecolors="black", linewidths=0.5)
    ax.set_xlabel("FPS (batch=1, imgsz=640, fp32)")
    ax.set_ylabel("mAP@50-95 (test)")
    ax.set_title("Трейдофф скорость / точность — все 16 экспериментов")
    ax.grid(True, alpha=0.3)
    # Legends
    det_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                               markersize=10, label=d) for d, c in colors.items()]
    var_handles = [plt.Line2D([0], [0], marker=m, color="black", markerfacecolor="white",
                               markersize=10, label=v) for v, m in markers.items()]
    l1 = ax.legend(handles=det_handles, loc="lower right", title="Детектор", fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=var_handles, loc="upper left", title="Вариант", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "speed_accuracy_scatter.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR}/speed_accuracy_scatter.png")


def per_class_contribution():
    """For each detector — grouped bar chart 9 classes x 4 variants (mAP@50)."""
    pc_dir = OUT_DIR / "per_class_contribution"
    cc.ensure_dir(pc_dir)

    for det in DETECTORS:
        data = {v: [np.nan] * 9 for v in VARIANTS}
        any_data = False
        for v in VARIANTS:
            pc = ROOT / "code/results" / DETECTOR_TASK[det] / f"{det}_{v}" / "per_class_map.csv"
            if not pc.exists():
                continue
            df = pd.read_csv(pc)
            any_data = True
            for _, r in df.iterrows():
                cid = int(r["class_id"])
                try:
                    data[v][cid] = float(r["mAP50"])
                except Exception:
                    pass
        if not any_data:
            continue
        fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
        x = np.arange(9)
        w = 0.2
        for i, v in enumerate(VARIANTS):
            ax.bar(x + (i - 1.5) * w, data[v], w, label=v)
        ax.set_xticks(x)
        ax.set_xticklabels([n[:20] for n in cc.CLASS_NAMES], rotation=25, ha="right")
        ax.set_ylabel("mAP@50")
        ax.set_title(f"{det} — mAP@50 по классам × вариантам подготовки")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()
        fig.tight_layout()
        fig.savefig(pc_dir / f"{det}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {pc_dir}/{det}.png")


def qualitative_grid():
    """Grid: 9 rows (images) × 4 cols (detectors), variant=aug_diffusion."""
    qfile = ROOT / "code/docs/chapter3_qualitative_sample.txt"
    qnames = [l.strip() for l in qfile.read_text().splitlines()
              if l.strip() and not l.startswith("#")]
    rows, cols = len(qnames), len(DETECTORS)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), dpi=100)
    for i, fname in enumerate(qnames):
        for j, det in enumerate(DETECTORS):
            ax = axs[i, j] if rows > 1 else axs[j]
            p = (ROOT / "code/results" / DETECTOR_TASK[det]
                 / f"{det}_aug_diffusion" / "predictions_examples" / fname)
            if p.exists():
                ax.imshow(Image.open(p))
            else:
                ax.text(0.5, 0.5, "нет", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(det, fontsize=11)
            if j == 0:
                ax.set_ylabel(fname, fontsize=9, rotation=0, labelpad=35, ha="right")
    fig.suptitle("Качественное сравнение: 9 тестовых изображений × 4 детектора (aug_diffusion)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qualitative_grid.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR}/qualitative_grid.png")


def bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000):
    """Bootstrap 95% CI on aug_diffusion mAP@50 for top-2 detectors."""
    top = df[df["variant"] == "aug_diffusion"].dropna(subset=["mAP50"])
    if len(top) < 2:
        print("not enough data for bootstrap_ci")
        return
    top = top.sort_values("mAP50", ascending=False).head(2)
    # For CI we need per-image AP — which we don't have. Approximate via
    # resampling from the per-class mAP distribution (9 classes × 4 variants).
    # For a proper CI, one would use per-image predictions; here we provide a
    # rough estimate using per-class mAP@50 as the resampling unit.
    out = {}
    for _, r in top.iterrows():
        det = r["detector"]
        pc = ROOT / "code/results" / DETECTOR_TASK[det] / f"{det}_aug_diffusion" / "per_class_map.csv"
        if not pc.exists():
            continue
        dfp = pd.read_csv(pc)
        try:
            vals = dfp["mAP50"].astype(float).values
        except Exception:
            continue
        rng = np.random.default_rng(42)
        bootstrap = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(vals), len(vals))
            bootstrap.append(float(np.mean(vals[idx])))
        bs = np.asarray(bootstrap)
        out[det] = {
            "mAP50": float(r["mAP50"]),
            "ci_low": float(np.percentile(bs, 2.5)),
            "ci_high": float(np.percentile(bs, 97.5)),
            "method": "per-class bootstrap (n=1000, 9 classes) — приблизительная оценка",
        }
    # Overlap check
    dets = list(out.keys())
    overlap = False
    if len(dets) == 2:
        a = (out[dets[0]]["ci_low"], out[dets[0]]["ci_high"])
        b = (out[dets[1]]["ci_low"], out[dets[1]]["ci_high"])
        overlap = not (a[1] < b[0] or b[1] < a[0])
    (OUT_DIR / "bootstrap_ci.json").write_text(json.dumps({
        "top2": out,
        "intervals_overlap": overlap,
    }, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_DIR}/bootstrap_ci.json  (overlap={overlap})")


def final_table(df: pd.DataFrame):
    """Best variant per detector — for chapter 3 final table."""
    rows = []
    for det in DETECTORS:
        sub = df[df["detector"] == det]
        if sub.empty:
            continue
        try:
            sub_numeric = sub.copy()
            sub_numeric["mAP50_num"] = pd.to_numeric(sub_numeric["mAP50"], errors="coerce")
            best = sub_numeric.loc[sub_numeric["mAP50_num"].idxmax()]
        except Exception:
            best = sub.iloc[0]
        # Param counts (approximate; exact values depend on weights)
        params = {"yolov12": 20.1, "rtdetr": 32.0, "faster_rcnn": 43.3, "detr": 41.3}
        rows.append({
            "detector": det,
            "best_variant": best.get("variant", ""),
            "mAP50": best.get("mAP50", ""),
            "mAP5095": best.get("mAP5095", ""),
            "precision": best.get("precision", ""),
            "recall": best.get("recall", ""),
            "fps": best.get("fps", ""),
            "params_M": params.get(det, ""),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "final_table.csv", index=False)
    print(f"wrote {OUT_DIR}/final_table.csv")


def main():
    cc.ensure_dir(OUT_DIR)
    df = load_grand_summary()
    if df.empty:
        print("no summary data yet — nothing to aggregate")
        return
    try:
        delta_heatmap(df)
    except Exception as e:
        print(f"delta_heatmap failed: {e}")
    try:
        speed_accuracy_scatter(df)
    except Exception as e:
        print(f"speed_accuracy_scatter failed: {e}")
    try:
        per_class_contribution()
    except Exception as e:
        print(f"per_class_contribution failed: {e}")
    try:
        qualitative_grid()
    except Exception as e:
        print(f"qualitative_grid failed: {e}")
    try:
        bootstrap_ci(df)
    except Exception as e:
        print(f"bootstrap_ci failed: {e}")
    try:
        final_table(df)
    except Exception as e:
        print(f"final_table failed: {e}")
    print("\ntask_11 aggregation done")


if __name__ == "__main__":
    main()
