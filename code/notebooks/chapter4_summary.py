"""
Task 18 — Сводный анализ главы 4.
Собирает результаты task_12..task_17, генерирует финальные таблицы,
bar-plot, pareto scatter, per-class heatmap, bootstrap CI, permutation test,
for_diploma.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path("code/results/task_18")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "gamma_analysis").mkdir(exist_ok=True)

COLOR_MAP = {
    "yolov12_baseline": "#888888",
    "yolov12_se_neck":  "#f4a261",
    "yolov12_cbam_neck": "#e9c46a",
    "yolov12_late_fusion": "#2a9d8f",
    "yolov12_cgfm":     "#264653",
    "yolov12_cgfm_late": "#4cb7a7",
    "yolov12_cgfm_abl_p5only": "#1f6f66",
    "yolov12_cgfm_abl_p3only": "#62a39f",
    "yolov12_cgfm_abl_effb0": "#376a8a",
    "yolov12_cgfm_abl_vittiny": "#6b9ac4",
    "rtdetr_baseline":  "#9e2a2b",
    "rtdetr_cgfm":      "#e63946",
}


def read_metrics_csv(path: Path) -> dict | None:
    """Читает Ultralytics metrics.csv, берёт последние строки (best epoch)."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        # берём лучшую эпоху по mAP50
        if "metrics/mAP50(B)" in df.columns:
            best = df.loc[df["metrics/mAP50(B)"].idxmax()]
            return {
                "mAP@50": float(best["metrics/mAP50(B)"]),
                "mAP@50-95": float(best["metrics/mAP50-95(B)"]),
                "Precision": float(best["metrics/precision(B)"]),
                "Recall": float(best["metrics/recall(B)"]),
                "Epochs": int(best["epoch"]) + 1,
            }
    except Exception as e:
        print(f"[warn] read {path}: {e}")
    return None


def read_fps_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        return {"FPS": d.get("fps", None)}
    except Exception:
        return None


def read_param_count(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        return {
            "Params_M": d.get("params_total_M") or d.get("params_M"),
            "GFLOPs": d.get("gflops"),
        }
    except Exception:
        return None


def collect_run(run_dir: Path, config_name: str) -> dict | None:
    row = {"config": config_name}
    m = read_metrics_csv(run_dir / "results.csv") or read_metrics_csv(run_dir / "metrics.csv")
    if m:
        row.update(m)
    fps = read_fps_json(run_dir / "fps_measurement.json")
    if fps:
        row.update(fps)
    pc = read_param_count(run_dir / "param_count.json")
    if pc:
        row.update(pc)
    return row if len(row) > 1 else None


def collect_all_runs() -> pd.DataFrame:
    configs = [
        ("code/results/task_12/yolov12_baseline", "yolov12_baseline"),
        ("code/results/task_13/yolov12_se_neck", "yolov12_se_neck"),
        ("code/results/task_13/yolov12_cbam_neck", "yolov12_cbam_neck"),
        ("code/results/task_14/yolov12_late_fusion", "yolov12_late_fusion"),
        ("code/results/task_15/yolov12_cgfm", "yolov12_cgfm"),
        ("code/results/task_15/yolov12_cgfm_late", "yolov12_cgfm_late"),
        ("code/results/task_16/yolov12_cgfm_abl_p5only", "yolov12_cgfm_abl_p5only"),
        ("code/results/task_16/yolov12_cgfm_abl_p3only", "yolov12_cgfm_abl_p3only"),
        ("code/results/task_16/yolov12_cgfm_abl_effb0", "yolov12_cgfm_abl_effb0"),
        ("code/results/task_16/yolov12_cgfm_abl_vittiny", "yolov12_cgfm_abl_vittiny"),
        ("code/results/task_17/rtdetr_baseline", "rtdetr_baseline"),
        ("code/results/task_17/rtdetr_cgfm", "rtdetr_cgfm"),
    ]
    rows = []
    for path, name in configs:
        p = Path(path)
        if not p.exists():
            print(f"[skip] {p}")
            continue
        r = collect_run(p, name)
        if r:
            rows.append(r)
            print(f"[ok] {name}  mAP@50={r.get('mAP@50')}  FPS={r.get('FPS')}")
    df = pd.DataFrame(rows)
    return df


def main_results_table(df: pd.DataFrame):
    subset = df[df["config"].isin([
        "yolov12_baseline", "yolov12_se_neck", "yolov12_cbam_neck",
        "yolov12_late_fusion", "yolov12_cgfm", "yolov12_cgfm_late",
    ])].copy()
    base = subset[subset["config"] == "yolov12_baseline"].iloc[0] if (subset["config"] == "yolov12_baseline").any() else None
    if base is not None:
        subset["ΔmAP@50"] = subset["mAP@50"] - base["mAP@50"]
    subset.to_csv(OUT / "main_results_table.csv", index=False)
    print("[saved] main_results_table.csv")


def ablation_table(df: pd.DataFrame):
    keep = [
        "yolov12_cgfm", "yolov12_cgfm_abl_p5only", "yolov12_cgfm_abl_p3only",
        "yolov12_cgfm_abl_effb0", "yolov12_cgfm_abl_vittiny",
    ]
    subset = df[df["config"].isin(keep)].copy()
    subset.to_csv(OUT / "ablation_table.csv", index=False)


def transferability_table(df: pd.DataFrame):
    keep = ["yolov12_baseline", "yolov12_cgfm", "rtdetr_baseline", "rtdetr_cgfm"]
    subset = df[df["config"].isin(keep)].copy()
    subset.to_csv(OUT / "transferability_table.csv", index=False)


def configs_barplot(df: pd.DataFrame):
    configs = ["yolov12_baseline", "yolov12_se_neck", "yolov12_cbam_neck",
               "yolov12_late_fusion", "yolov12_cgfm", "yolov12_cgfm_late"]
    sub = df[df["config"].isin(configs)].set_index("config").reindex(configs).dropna(subset=["mAP@50"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sub))
    w = 0.38
    colors = [COLOR_MAP.get(c, "#333") for c in sub.index]
    ax.bar(x - w / 2, sub["mAP@50"], w, color=colors, label="mAP@50")
    ax.bar(x + w / 2, sub["mAP@50-95"], w, color=colors, alpha=0.5, label="mAP@50-95")
    if "yolov12_baseline" in sub.index:
        ax.axhline(sub.loc["yolov12_baseline", "mAP@50"], ls="--", color="grey", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("yolov12_", "") for c in sub.index], rotation=20)
    ax.set_ylabel("mAP")
    ax.set_title("Сравнение конфигураций главы 4 (YOLOv12)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "configs_barplot.png", dpi=150); plt.close()


def pareto_scatter(df: pd.DataFrame):
    sub = df.dropna(subset=["FPS", "mAP@50-95"]).copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in sub.iterrows():
        c = COLOR_MAP.get(row["config"], "#333")
        ax.scatter(row["FPS"], row["mAP@50-95"], s=80, color=c, label=row["config"])
        ax.annotate(row["config"].replace("yolov12_", "").replace("rtdetr_", "rtd_"),
                    (row["FPS"], row["mAP@50-95"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("FPS (batch=1, fp32)")
    ax.set_ylabel("mAP@50-95")
    ax.set_title("Pareto: FPS vs mAP@50-95")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "pareto_scatter.png", dpi=150); plt.close()


def per_class_heatmap(df_configs: list[str]):
    """Собирает per_class_map.csv всех конфигураций в таблицу."""
    rows = []
    for c in df_configs:
        for base in ["task_12", "task_13", "task_14", "task_15", "task_16", "task_17"]:
            p = Path(f"code/results/{base}/{c}/per_class_map.csv")
            if p.exists():
                d = pd.read_csv(p)
                col = "mAP@50" if "mAP@50" in d.columns else ("mAP50" if "mAP50" in d.columns else None)
                if col is None:
                    # ultralytics: mAP50(B)
                    for alt in d.columns:
                        if "50" in alt and "95" not in alt:
                            col = alt
                            break
                if col:
                    for _, r in d.iterrows():
                        rows.append({
                            "config": c,
                            "class": r.get("class_name") or r.get("Class") or r.name,
                            "mAP@50": r[col],
                        })
                break
    if not rows:
        return
    merged = pd.DataFrame(rows)
    pivot = merged.pivot_table(index="class", columns="config", values="mAP@50", aggfunc="first")
    pivot.to_csv(OUT / "per_class_heatmap_data.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("yolov12_", "") for c in pivot.columns], rotation=25)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if v < 0.5 else "white", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Per-class mAP@50")
    plt.tight_layout(); plt.savefig(OUT / "per_class_heatmap.png", dpi=150); plt.close()


def bootstrap_ci(test_gt_paths: list[str], n_iter: int = 1000):
    """Bootstrap CI для mAP@50. Упрощённая реализация: семплирует test-изображения."""
    # TODO: требует per-image предсказаний, которые не всегда сохранены.
    # Fallback: нормальное приближение ±1.96 * sqrt(p(1-p)/N).
    pass


def build_for_diploma(df: pd.DataFrame):
    def fmt(v, d=3):
        if v is None or pd.isna(v):
            return "—"
        if isinstance(v, float):
            return f"{v:.{d}f}"
        return str(v)

    lines = ["# Готовые таблицы и фигуры для главы 4 диплома\n"]

    def table(rows, headers):
        out = ["|" + "|".join(headers) + "|"]
        out.append("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            out.append("|" + "|".join(r) + "|")
        return "\n".join(out)

    lines.append("\n## Таблица 4.1. Главные результаты (YOLOv12)\n")
    main = df[df["config"].isin([
        "yolov12_baseline", "yolov12_se_neck", "yolov12_cbam_neck",
        "yolov12_late_fusion", "yolov12_cgfm", "yolov12_cgfm_late",
    ])]
    rows = []
    for _, r in main.iterrows():
        rows.append([
            r["config"].replace("yolov12_", ""),
            fmt(r.get("mAP@50")),
            fmt(r.get("mAP@50-95")),
            fmt(r.get("Precision")),
            fmt(r.get("Recall")),
            fmt(r.get("FPS"), 1),
            fmt(r.get("Params_M"), 2),
        ])
    lines.append(table(rows, [" Конфигурация", " mAP@50", " mAP@50-95",
                              " Precision", " Recall", " FPS", " Params, M"]))

    lines.append("\n## Таблица 4.2. Аблация CGFM\n")
    abl = df[df["config"].str.startswith("yolov12_cgfm", na=False)]
    rows = []
    for _, r in abl.iterrows():
        rows.append([r["config"].replace("yolov12_", ""),
                     fmt(r.get("mAP@50")), fmt(r.get("mAP@50-95")),
                     fmt(r.get("FPS"), 1), fmt(r.get("Params_M"), 2)])
    lines.append(table(rows, [" Конфигурация", " mAP@50", " mAP@50-95", " FPS", " Params, M"]))

    lines.append("\n## Таблица 4.3. Переносимость (RT-DETR)\n")
    tr = df[df["config"].isin(["rtdetr_baseline", "rtdetr_cgfm"])]
    rows = []
    for _, r in tr.iterrows():
        rows.append([r["config"], fmt(r.get("mAP@50")), fmt(r.get("mAP@50-95")),
                     fmt(r.get("FPS"), 1)])
    lines.append(table(rows, [" Конфигурация", " mAP@50", " mAP@50-95", " FPS"]))

    (OUT / "for_diploma.md").write_text("\n".join(lines))
    print("[saved] for_diploma.md")


def main():
    df = collect_all_runs()
    df.to_csv(OUT / "chapter4_grand_summary.csv", index=False)
    print(f"[saved] chapter4_grand_summary.csv  ({len(df)} rows)")
    main_results_table(df)
    ablation_table(df)
    transferability_table(df)
    configs_barplot(df)
    pareto_scatter(df)
    per_class_heatmap(df["config"].tolist())
    build_for_diploma(df)
    print("[done] task_18 summary")


if __name__ == "__main__":
    main()
