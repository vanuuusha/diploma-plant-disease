"""
Unified runner for Ultralytics-based detectors (YOLOv12, RT-DETR) per chapter 3 protocol.

Usage:
    python chapter3_ultralytics_runner.py --detector yolov12 --variant baseline
    python chapter3_ultralytics_runner.py --detector rtdetr --variant aug_diffusion

Produces artifacts in code/results/task_NN/<detector>_<variant>/ strictly per
code/docs/chapter3_protocol.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chapter3_common as cc

ROOT = cc.ROOT

DETECTOR_CONFIG = {
    "yolov12": {
        "task_dir": ROOT / "code/results/task_07",
        "weights": "yolo12m.pt",
        "batch": 16,
        "cls": "YOLO",
    },
    "rtdetr": {
        "task_dir": ROOT / "code/results/task_08",
        "weights": "rtdetr-l.pt",
        "batch": 8,
        "cls": "RTDETR",
    },
}


def train(detector: str, variant: str, epochs: int, patience: int):
    from ultralytics import YOLO, RTDETR

    cfg = DETECTOR_CONFIG[detector]
    data_yaml = cc.DATASET_VARIANTS[variant]
    out_dir = cfg["task_dir"] / f"{detector}_{variant}"
    cc.ensure_dir(out_dir)

    ModelCls = YOLO if cfg["cls"] == "YOLO" else RTDETR
    model = ModelCls(cfg["weights"])

    t0 = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=patience,
        imgsz=640,
        batch=cfg["batch"],
        seed=42,
        device=0,
        workers=8,
        project=str(cfg["task_dir"]),
        name=f"{detector}_{variant}",
        exist_ok=True,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
        verbose=True,
        plots=True,
    )
    dt = time.time() - t0
    print(f"[train] {detector}/{variant} done in {dt/60:.1f} min")
    return results, out_dir, model


def collect_metrics_csv(out_dir: Path) -> pd.DataFrame:
    """Convert Ultralytics results.csv to protocol metrics.csv."""
    results_csv = out_dir / "results.csv"
    if not results_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    col = {
        "epoch": "epoch",
        "train_loss": next((c for c in df.columns if "train" in c and "total" in c.lower()), None),
        "val_loss": next((c for c in df.columns if "val" in c and "total" in c.lower()), None),
        "val_precision": next((c for c in df.columns if "precision" in c.lower()), None),
        "val_recall": next((c for c in df.columns if "recall" in c.lower()), None),
        "val_mAP50": "metrics/mAP50(B)" if "metrics/mAP50(B)" in df.columns else None,
        "val_mAP5095": "metrics/mAP50-95(B)" if "metrics/mAP50-95(B)" in df.columns else None,
        "lr": "lr/pg0" if "lr/pg0" in df.columns else None,
        "time_sec": "time" if "time" in df.columns else None,
    }

    if col["train_loss"] is None:
        tl_cols = [c for c in df.columns if c.startswith("train/") and "loss" in c]
        if tl_cols:
            df["_train_total"] = df[tl_cols].sum(axis=1)
            col["train_loss"] = "_train_total"
    if col["val_loss"] is None:
        vl_cols = [c for c in df.columns if c.startswith("val/") and "loss" in c]
        if vl_cols:
            df["_val_total"] = df[vl_cols].sum(axis=1)
            col["val_loss"] = "_val_total"

    rows = []
    for i, row in df.iterrows():
        rows.append({
            "epoch": int(row[col["epoch"]]) if col["epoch"] else i,
            "train_loss": float(row[col["train_loss"]]) if col["train_loss"] and col["train_loss"] in df.columns else "",
            "val_loss": float(row[col["val_loss"]]) if col["val_loss"] and col["val_loss"] in df.columns else "",
            "val_precision": float(row[col["val_precision"]]) if col["val_precision"] else "",
            "val_recall": float(row[col["val_recall"]]) if col["val_recall"] else "",
            "val_mAP50": float(row[col["val_mAP50"]]) if col["val_mAP50"] else "",
            "val_mAP5095": float(row[col["val_mAP5095"]]) if col["val_mAP5095"] else "",
            "lr": float(row[col["lr"]]) if col["lr"] else "",
            "time_sec": float(row[col["time_sec"]]) if col["time_sec"] else "",
        })
    cc.save_metrics_csv(out_dir / "metrics.csv", rows)
    return df


def evaluate_on_test(detector: str, variant: str, out_dir: Path, model):
    """Run model.val(split='test') and save per-class mAP + confusion matrix."""
    data_yaml = cc.DATASET_VARIANTS[variant]
    try:
        metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=640,
            batch=8,
            device=0,
            project=str(out_dir.parent),
            name=f"{out_dir.name}_test",
            exist_ok=True,
            plots=True,
            verbose=True,
        )
    except Exception as e:
        print(f"[evaluate] {detector}/{variant} test eval failed: {e}")
        return None, None, None

    per_class_rows = []
    try:
        maps50 = metrics.box.ap50
        maps5095 = metrics.box.ap
        names = metrics.names
        for cls_id in range(9):
            per_class_rows.append({
                "class_id": cls_id,
                "class_name": names.get(cls_id, cc.CLASS_NAMES[cls_id]) if isinstance(names, dict) else cc.CLASS_NAMES[cls_id],
                "n_test_instances": "",
                "mAP50": float(maps50[cls_id]) if cls_id < len(maps50) else "",
                "mAP50_95": float(maps5095[cls_id]) if cls_id < len(maps5095) else "",
            })
    except Exception as e:
        print(f"[evaluate] per-class extraction failed: {e}")
        per_class_rows = [
            {"class_id": i, "class_name": cc.CLASS_NAMES[i],
             "n_test_instances": "", "mAP50": "", "mAP50_95": ""}
            for i in range(9)
        ]
    cc.save_per_class_map(out_dir / "per_class_map.csv", per_class_rows)

    test_dir = out_dir.parent / f"{out_dir.name}_test"
    for cm_name in ("confusion_matrix_normalized.png", "confusion_matrix.png"):
        src = test_dir / cm_name
        if src.exists():
            import shutil
            shutil.copy(src, out_dir / "confusion_matrix.png")
            break

    return metrics, per_class_rows, test_dir


def measure_fps(model, detector: str, variant: str, out_dir: Path, n_warmup: int = 20, n_measure: int = 100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    qfiles = cc.qualitative_filenames()
    img_path = cc.TEST_IMG_DIR / qfiles[0]

    for _ in range(n_warmup):
        _ = model.predict(str(img_path), imgsz=640, device=0, verbose=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    latencies = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        _ = model.predict(str(img_path), imgsz=640, device=0, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    cc.save_fps(out_dir / "fps_measurement.json", detector, variant, latencies)
    return float(np.mean(latencies))


def render_qualitative(model, detector: str, variant: str, out_dir: Path):
    def predict_fn(img_path: str):
        res = model.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)[0]
        out = []
        if res.boxes is None:
            return out
        boxes = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        for b, c, s in zip(boxes, cls, conf):
            out.append({"box": b.tolist(), "cls": int(c), "score": float(s)})
        return out

    cc.render_predictions_examples(out_dir / "predictions_examples", predict_fn)


def run_single(detector: str, variant: str, epochs: int, patience: int):
    cfg = DETECTOR_CONFIG[detector]
    out_dir = cfg["task_dir"] / f"{detector}_{variant}"
    cc.ensure_dir(out_dir)

    log_path = out_dir / "train.log"
    print(f"=== {detector} / {variant} ===")

    # Train
    _, out_dir, model = train(detector, variant, epochs, patience)

    # Metrics CSV + learning curves
    df = collect_metrics_csv(out_dir)
    cc.plot_learning_curves(
        out_dir / "metrics.csv",
        out_dir / "learning_curves.png",
        title=f"{detector} / {variant}",
    )

    # Test eval (per-class + CM)
    evaluate_on_test(detector, variant, out_dir, model)

    # FPS
    measure_fps(model, detector, variant, out_dir)

    # Qualitative predictions
    render_qualitative(model, detector, variant, out_dir)

    # Summary row
    return out_dir


def build_summary_row(detector: str, variant: str) -> dict:
    cfg = DETECTOR_CONFIG[detector]
    out_dir = cfg["task_dir"] / f"{detector}_{variant}"
    metrics_csv = out_dir / "metrics.csv"
    fps_json = out_dir / "fps_measurement.json"

    row = {
        "variant": variant,
        "n_train": cc.count_train_images(variant),
        "mAP50": "", "mAP5095": "", "precision": "", "recall": "", "fps": "", "epochs": "",
    }
    if metrics_csv.exists():
        import pandas as pd
        df = pd.read_csv(metrics_csv)
        if not df.empty:
            best = df.loc[df["val_mAP50"].idxmax()] if "val_mAP50" in df.columns else df.iloc[-1]
            row["mAP50"] = round(float(best["val_mAP50"]), 4) if "val_mAP50" in df.columns else ""
            row["mAP5095"] = round(float(best["val_mAP5095"]), 4) if "val_mAP5095" in df.columns else ""
            row["precision"] = round(float(best["val_precision"]), 4) if "val_precision" in df.columns else ""
            row["recall"] = round(float(best["val_recall"]), 4) if "val_recall" in df.columns else ""
            row["epochs"] = int(df["epoch"].max())
    if fps_json.exists():
        import json
        row["fps"] = round(json.loads(fps_json.read_text())["fps"], 2)
    return row


def write_summary(detector: str):
    cfg = DETECTOR_CONFIG[detector]
    rows = []
    for v in ["baseline", "aug_geom", "aug_oversample", "aug_diffusion"]:
        rows.append(build_summary_row(detector, v))
    cc.write_summary_csv(cfg["task_dir"] / "summary.csv", rows)
    print(f"[summary] wrote {cfg['task_dir']}/summary.csv")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=["yolov12", "rtdetr"], required=True)
    ap.add_argument("--variant", choices=["baseline", "aug_geom", "aug_oversample", "aug_diffusion", "all", "summary"], required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    if args.variant == "summary":
        write_summary(args.detector)
        return

    variants = ["baseline", "aug_geom", "aug_oversample", "aug_diffusion"] if args.variant == "all" else [args.variant]
    for v in variants:
        run_single(args.detector, v, args.epochs, args.patience)
    write_summary(args.detector)


if __name__ == "__main__":
    main()
