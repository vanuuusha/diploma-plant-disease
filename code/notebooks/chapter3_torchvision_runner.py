"""
Runner for torchvision-based detectors (Faster R-CNN) for chapter 3 protocol.

Usage:
    python chapter3_torchvision_runner.py --variant baseline
    python chapter3_torchvision_runner.py --variant all
    python chapter3_torchvision_runner.py --variant summary

Produces artifacts in code/results/task_09/faster_rcnn_<variant>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chapter3_common as cc

ROOT = cc.ROOT
TASK_DIR = ROOT / "code/results/task_09"
NUM_CLASSES = 10  # 9 disease classes + background


class YoloDataset(torch.utils.data.Dataset):
    """Read YOLO format labels, return (image, target) for torchvision detectors."""

    def __init__(self, images_dir: Path, labels_dir: Path, imgsz: int = 640):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.imgsz = imgsz
        self.files = sorted(
            [p.name for p in self.images_dir.iterdir()
             if p.suffix.lower() in (".jpg", ".png", ".jpeg")]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = self.images_dir / fname
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        lbl_path = self.labels_dir / (Path(fname).stem + ".txt")
        boxes = []
        labels = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                c = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = (cx - bw / 2) * W
                y1 = (cy - bh / 2) * H
                x2 = (cx + bw / 2) * W
                y2 = (cy + bh / 2) * H
                x1, x2 = max(0, min(W, x1)), max(0, min(W, x2))
                y1, y2 = max(0, min(H, y1)), max(0, min(H, y2))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(c + 1)  # shift by 1 for torchvision background=0

        # Resize to imgsz x imgsz (letterbox replaced by simple resize for simplicity)
        img = img.resize((self.imgsz, self.imgsz))
        sx = self.imgsz / W
        sy = self.imgsz / H
        if boxes:
            boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        import torchvision.transforms.functional as TF
        img_t = TF.to_tensor(img)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "_filename": fname,
        }
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


def load_dataset(variant: str, split: str):
    ds_dir = cc.DATASET_VARIANTS[variant].parent
    return YoloDataset(ds_dir / split / "images", ds_dir / split / "labels")


def build_model():
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def evaluate_map(model, loader, device) -> dict:
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            preds_cpu = [{"boxes": p["boxes"].cpu(),
                          "scores": p["scores"].cpu(),
                          "labels": p["labels"].cpu()} for p in preds]
            tgts_cpu = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
            metric.update(preds_cpu, tgts_cpu)
    out = metric.compute()
    return {k: (v.cpu().numpy().tolist() if hasattr(v, "cpu") else v) for k, v in out.items()}


def scalar(v, default=0.0):
    if isinstance(v, (list, tuple)) and v:
        return float(v[0])
    try:
        fv = float(v)
        return fv if fv >= 0 else default
    except Exception:
        return default


def compute_pr(model, loader, device, score_thres=0.25):
    """Rough precision/recall at a fixed score threshold — for parity with Ultralytics metrics.csv."""
    tp = fp = fn = 0
    model.eval()
    iou_t = 0.5
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            for p, t in zip(preds, targets):
                pb = p["boxes"].cpu().numpy()
                ps = p["scores"].cpu().numpy()
                pl = p["labels"].cpu().numpy()
                keep = ps >= score_thres
                pb, ps, pl = pb[keep], ps[keep], pl[keep]
                gb = t["boxes"].numpy()
                gl = t["labels"].numpy()
                used_gt = np.zeros(len(gb), dtype=bool)
                for i in range(len(pb)):
                    x1, y1, x2, y2 = pb[i]
                    best_iou = 0.0
                    best_j = -1
                    for j in range(len(gb)):
                        if used_gt[j] or gl[j] != pl[i]:
                            continue
                        ix1 = max(x1, gb[j, 0])
                        iy1 = max(y1, gb[j, 1])
                        ix2 = min(x2, gb[j, 2])
                        iy2 = min(y2, gb[j, 3])
                        iw = max(0.0, ix2 - ix1)
                        ih = max(0.0, iy2 - iy1)
                        inter = iw * ih
                        area_p = (x2 - x1) * (y2 - y1)
                        area_g = (gb[j, 2] - gb[j, 0]) * (gb[j, 3] - gb[j, 1])
                        union = area_p + area_g - inter + 1e-9
                        iou = inter / union
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j
                    if best_iou >= iou_t and best_j >= 0:
                        tp += 1
                        used_gt[best_j] = True
                    else:
                        fp += 1
                fn += int((~used_gt).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return prec, rec


def train_variant(variant: str, epochs: int, patience: int, batch: int = 4):
    device = torch.device("cuda:0")
    out_dir = TASK_DIR / f"faster_rcnn_{variant}"
    cc.ensure_dir(out_dir)

    train_ds = load_dataset(variant, "train")
    val_ds = load_dataset(variant, "val")
    print(f"[faster_rcnn/{variant}] n_train={len(train_ds)} n_val={len(val_ds)} batch={batch}")

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    best_map50 = -1.0
    best_epoch = -1
    no_improve = 0
    rows = []

    for ep in range(1, epochs + 1):
        model.train()
        ep_t0 = time.time()
        train_loss_sum = 0.0
        n_batches = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets_dev = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
                           for t in targets]
            loss_dict = model(images, targets_dev)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            n_batches += 1
        scheduler.step()

        train_loss = train_loss_sum / max(n_batches, 1)

        # Val
        val_metrics = evaluate_map(model, val_loader, device)
        m50 = scalar(val_metrics.get("map_50", 0.0))
        m5095 = scalar(val_metrics.get("map", 0.0))
        prec, rec = compute_pr(model, val_loader, device)

        ep_time = time.time() - ep_t0
        row = {
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "val_loss": "",
            "val_precision": round(prec, 4),
            "val_recall": round(rec, 4),
            "val_mAP50": round(m50, 4),
            "val_mAP5095": round(m5095, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(ep_time, 1),
        }
        rows.append(row)
        cc.save_metrics_csv(out_dir / "metrics.csv", rows)
        print(f"  ep{ep:3d} tl={train_loss:.3f} mAP50={m50:.3f} mAP50-95={m5095:.3f} "
              f"P={prec:.3f} R={rec:.3f} t={ep_time:.1f}s")

        if m50 > best_map50 + 1e-4:
            best_map50 = m50
            best_epoch = ep
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            no_improve += 1

        cc.plot_learning_curves(
            out_dir / "metrics.csv",
            out_dir / "learning_curves.png",
            title=f"Faster R-CNN / {variant}",
        )

        if no_improve >= patience:
            print(f"  early stop at ep{ep} (best={best_epoch} mAP50={best_map50:.3f})")
            break

    return model, out_dir, best_epoch


def evaluate_on_test(model, variant: str, out_dir: Path):
    device = torch.device("cuda:0")
    # Load best weights
    best_path = out_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_ds = load_dataset(variant, "test")
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                             collate_fn=collate_fn, num_workers=4)

    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    model.eval()
    conf_matrix = np.zeros((10, 10), dtype=np.int64)  # 9 classes + bg
    n_instances = np.zeros(9, dtype=int)

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            preds_cpu = [{"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(),
                          "labels": p["labels"].cpu()} for p in preds]
            tgts_cpu = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
            metric.update(preds_cpu, tgts_cpu)

            # Confusion matrix (GT x best-matched Pred class at iou>=0.5)
            for p, t in zip(preds_cpu, tgts_cpu):
                gb = t["boxes"].numpy()
                gl = t["labels"].numpy()
                pb = p["boxes"].numpy()
                pl = p["labels"].numpy()
                ps = p["scores"].numpy()
                for cls in gl:
                    if 1 <= cls <= 9:
                        n_instances[cls - 1] += 1
                keep = ps >= 0.25
                pb, pl, ps = pb[keep], pl[keep], ps[keep]
                used_pred = np.zeros(len(pb), dtype=bool)
                for j in range(len(gb)):
                    gtcls = gl[j]
                    best_iou = 0.0
                    best_i = -1
                    for i in range(len(pb)):
                        if used_pred[i]:
                            continue
                        x1, y1, x2, y2 = pb[i]
                        ix1 = max(x1, gb[j, 0]); iy1 = max(y1, gb[j, 1])
                        ix2 = min(x2, gb[j, 2]); iy2 = min(y2, gb[j, 3])
                        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
                        inter = iw * ih
                        area_p = (x2 - x1) * (y2 - y1)
                        area_g = (gb[j, 2] - gb[j, 0]) * (gb[j, 3] - gb[j, 1])
                        iou = inter / (area_p + area_g - inter + 1e-9)
                        if iou > best_iou:
                            best_iou = iou; best_i = i
                    if best_iou >= 0.5 and best_i >= 0:
                        conf_matrix[gtcls - 1, pl[best_i] - 1] += 1
                        used_pred[best_i] = True
                    else:
                        conf_matrix[gtcls - 1, 9] += 1  # missed → background

    mdict = metric.compute()
    # per-class
    per_class_m50 = mdict.get("map_50_per_class", [])
    per_class_m = mdict.get("map_per_class", [])
    classes_order = mdict.get("classes", [])
    if hasattr(classes_order, "cpu"):
        classes_order = classes_order.cpu().numpy().tolist()
    per_class_rows = []
    for cid in range(9):
        try:
            idx = classes_order.index(cid + 1) if (cid + 1) in classes_order else -1
            m50 = float(per_class_m50[idx]) if idx >= 0 and idx < len(per_class_m50) else ""
            m = float(per_class_m[idx]) if idx >= 0 and idx < len(per_class_m) else ""
        except Exception:
            m50 = ""
            m = ""
        per_class_rows.append({
            "class_id": cid,
            "class_name": cc.CLASS_NAMES[cid],
            "n_test_instances": int(n_instances[cid]),
            "mAP50": round(m50, 4) if isinstance(m50, float) else "",
            "mAP50_95": round(m, 4) if isinstance(m, float) else "",
        })
    cc.save_per_class_map(out_dir / "per_class_map.csv", per_class_rows)
    cc.plot_confusion_matrix(conf_matrix, cc.CLASS_NAMES, out_dir / "confusion_matrix.png",
                             title=f"Faster R-CNN / {variant}")

    return mdict


def measure_fps(model, out_dir: Path, variant: str):
    device = torch.device("cuda:0")
    model.eval()
    import torchvision.transforms.functional as TF
    qfiles = cc.qualitative_filenames()
    img = Image.open(cc.TEST_IMG_DIR / qfiles[0]).convert("RGB").resize((640, 640))
    t = TF.to_tensor(img).to(device)

    with torch.no_grad():
        for _ in range(20):
            _ = model([t])
        torch.cuda.synchronize()
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = model([t])
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
    cc.save_fps(out_dir / "fps_measurement.json", "faster_rcnn", variant, latencies)


def render_qualitative(model, variant: str, out_dir: Path):
    device = torch.device("cuda:0")
    model.eval()
    import torchvision.transforms.functional as TF

    def predict_fn(img_path: str):
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        img_r = img.resize((640, 640))
        t = TF.to_tensor(img_r).to(device)
        with torch.no_grad():
            p = model([t])[0]
        boxes = p["boxes"].cpu().numpy()
        scores = p["scores"].cpu().numpy()
        labels = p["labels"].cpu().numpy()
        sx, sy = W / 640, H / 640
        out = []
        for b, s, l in zip(boxes, scores, labels):
            if s < 0.25:
                continue
            x1, y1, x2, y2 = b
            out.append({
                "box": [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                "cls": int(l) - 1,
                "score": float(s),
            })
        return out

    cc.render_predictions_examples(out_dir / "predictions_examples", predict_fn)


def run_single(variant: str, epochs: int, patience: int, batch: int):
    out_dir = TASK_DIR / f"faster_rcnn_{variant}"
    cc.ensure_dir(out_dir)
    print(f"=== faster_rcnn / {variant} ===")
    model, out_dir, best_epoch = train_variant(variant, epochs, patience, batch)
    evaluate_on_test(model, variant, out_dir)
    measure_fps(model, out_dir, variant)
    render_qualitative(model, variant, out_dir)
    print(f"[faster_rcnn/{variant}] done, best_epoch={best_epoch}")
    return out_dir


def build_summary_row(variant: str) -> dict:
    out_dir = TASK_DIR / f"faster_rcnn_{variant}"
    metrics_csv = out_dir / "metrics.csv"
    fps_json = out_dir / "fps_measurement.json"
    per_class_csv = out_dir / "per_class_map.csv"

    row = {"variant": variant, "n_train": cc.count_train_images(variant),
           "mAP50": "", "mAP5095": "", "precision": "", "recall": "", "fps": "", "epochs": ""}
    if metrics_csv.exists():
        import pandas as pd
        df = pd.read_csv(metrics_csv)
        if not df.empty:
            best = df.loc[df["val_mAP50"].idxmax()]
            row["mAP50"] = round(float(best["val_mAP50"]), 4)
            row["mAP5095"] = round(float(best["val_mAP5095"]), 4)
            row["precision"] = round(float(best["val_precision"]), 4)
            row["recall"] = round(float(best["val_recall"]), 4)
            row["epochs"] = int(df["epoch"].max())
    if fps_json.exists():
        row["fps"] = round(json.loads(fps_json.read_text())["fps"], 2)
    # Prefer test mAP if available (from per_class average)
    if per_class_csv.exists():
        import pandas as pd
        dfp = pd.read_csv(per_class_csv)
        if "mAP50" in dfp.columns and dfp["mAP50"].notna().any():
            try:
                row["mAP50"] = round(float(dfp["mAP50"].astype(float).mean()), 4)
                row["mAP5095"] = round(float(dfp["mAP50_95"].astype(float).mean()), 4)
            except Exception:
                pass
    return row


def write_summary():
    rows = [build_summary_row(v) for v in
            ["baseline", "aug_geom", "aug_oversample", "aug_diffusion"]]
    cc.write_summary_csv(TASK_DIR / "summary.csv", rows)
    print(f"[summary] {TASK_DIR}/summary.csv written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "aug_geom", "aug_oversample", "aug_diffusion", "all", "summary"], required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    if args.variant == "summary":
        write_summary()
        return

    variants = ["baseline", "aug_geom", "aug_oversample", "aug_diffusion"] if args.variant == "all" else [args.variant]
    for v in variants:
        run_single(v, args.epochs, args.patience, args.batch)
    write_summary()


if __name__ == "__main__":
    main()
