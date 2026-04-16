"""
Runner for HuggingFace DETR (facebook/detr-resnet-50) per chapter 3 protocol.

Usage:
    python chapter3_detr_runner.py --variant baseline
    python chapter3_detr_runner.py --variant all
    python chapter3_detr_runner.py --variant summary

Produces artifacts in code/results/task_10/detr_<variant>/.

Requires: HF_HUB_OFFLINE=1 if weights are already cached (avoids SYN_SENT hang).
"""
from __future__ import annotations

import argparse
import json
import os
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
TASK_DIR = ROOT / "code/results/task_10"
CHECKPOINT = "facebook/detr-resnet-50"


def make_processor():
    from transformers import DetrImageProcessor
    return DetrImageProcessor.from_pretrained(CHECKPOINT, do_resize=True, size={"shortest_edge": 640, "longest_edge": 640})


def make_model():
    from transformers import DetrForObjectDetection, AutoConfig

    id2label = {i: name for i, name in enumerate(cc.CLASS_NAMES)}
    label2id = {v: k for k, v in id2label.items()}
    config = AutoConfig.from_pretrained(
        CHECKPOINT, num_labels=9, id2label=id2label, label2id=label2id,
    )
    model = DetrForObjectDetection.from_pretrained(
        CHECKPOINT, config=config, ignore_mismatched_sizes=True,
    )
    return model


class DetrYoloDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, processor):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.processor = processor
        self.files = sorted(
            [p.name for p in self.images_dir.iterdir()
             if p.suffix.lower() in (".jpg", ".png", ".jpeg")]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(self.images_dir / fname).convert("RGB")
        lbl_path = self.labels_dir / (Path(fname).stem + ".txt")
        anns = []
        if lbl_path.exists():
            for ln in lbl_path.read_text().splitlines():
                parts = ln.strip().split()
                if len(parts) != 5:
                    continue
                c = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:])
                W, H = img.size
                x = (cx - bw / 2) * W
                y = (cy - bh / 2) * H
                w = bw * W
                h = bh * H
                if w < 1 or h < 1:
                    continue
                anns.append({
                    "bbox": [x, y, w, h],
                    "category_id": c,
                    "area": w * h,
                    "iscrowd": 0,
                })
        coco = {"image_id": idx, "annotations": anns}
        enc = self.processor(images=img, annotations=coco, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)
        target = enc["labels"][0]
        return {
            "pixel_values": pixel_values,
            "labels": target,
            "_filename": fname,
            "_raw_image": img,
        }


def collate_fn_factory(processor):
    def _collate(batch):
        pv = [b["pixel_values"] for b in batch]
        pv = torch.stack([p for p in pv])  # same size since we resize uniformly
        labels = [b["labels"] for b in batch]
        return {"pixel_values": pv, "labels": labels}
    return _collate


def convert_detr_preds_to_torchmetrics(outputs, target_sizes, id_map=None):
    """DETR raw outputs → torchmetrics list of dicts in original image coords."""
    from transformers.models.detr.image_processing_detr import DetrImageProcessor
    proc = make_processor()
    # processor.post_process_object_detection returns list of dicts
    results = proc.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.0
    )
    preds = []
    for r in results:
        preds.append({
            "boxes": r["boxes"].cpu(),
            "scores": r["scores"].cpu(),
            "labels": r["labels"].cpu(),
        })
    return preds


def evaluate_map(model, loader, device, processor):
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pv = batch["pixel_values"].to(device)
            labels = batch["labels"]
            outputs = model(pixel_values=pv)
            # target size: labels[i]["orig_size"] is (H, W); tensor
            target_sizes = torch.stack([lbl["orig_size"] for lbl in labels]).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.0
            )
            preds = [{"boxes": r["boxes"].cpu(), "scores": r["scores"].cpu(),
                      "labels": r["labels"].cpu()} for r in results]
            # Build GT in original image coords
            gts = []
            for lbl in labels:
                bx = lbl["boxes"]  # cx, cy, w, h normalized
                cls = lbl["class_labels"]
                H, W = lbl["orig_size"].tolist()
                # convert to x1,y1,x2,y2 absolute
                if bx.numel() > 0:
                    cx, cy, bw, bh = bx.unbind(-1)
                    x1 = (cx - bw / 2) * W
                    y1 = (cy - bh / 2) * H
                    x2 = (cx + bw / 2) * W
                    y2 = (cy + bh / 2) * H
                    abs_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                else:
                    abs_boxes = torch.zeros((0, 4))
                gts.append({"boxes": abs_boxes.cpu(), "labels": cls.cpu()})
            metric.update(preds, gts)
    out = metric.compute()
    return {k: (v.cpu().numpy().tolist() if hasattr(v, "cpu") else v) for k, v in out.items()}


def compute_pr_detr(model, loader, device, processor, score_thres=0.25):
    tp = fp = fn = 0
    iou_t = 0.5
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pv = batch["pixel_values"].to(device)
            labels = batch["labels"]
            outputs = model(pixel_values=pv)
            target_sizes = torch.stack([lbl["orig_size"] for lbl in labels]).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=score_thres
            )
            for r, lbl in zip(results, labels):
                pb = r["boxes"].cpu().numpy()
                pl = r["labels"].cpu().numpy()
                bx = lbl["boxes"].cpu().numpy()
                cls = lbl["class_labels"].cpu().numpy()
                H, W = lbl["orig_size"].tolist()
                if bx.shape[0] > 0:
                    cx, cy, bw, bh = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
                    gx1 = (cx - bw / 2) * W; gy1 = (cy - bh / 2) * H
                    gx2 = (cx + bw / 2) * W; gy2 = (cy + bh / 2) * H
                    gb = np.stack([gx1, gy1, gx2, gy2], axis=-1)
                else:
                    gb = np.zeros((0, 4))
                used_gt = np.zeros(len(gb), dtype=bool)
                for i in range(len(pb)):
                    x1, y1, x2, y2 = pb[i]
                    best_iou = 0.0; best_j = -1
                    for j in range(len(gb)):
                        if used_gt[j] or cls[j] != pl[i]:
                            continue
                        ix1 = max(x1, gb[j, 0]); iy1 = max(y1, gb[j, 1])
                        ix2 = min(x2, gb[j, 2]); iy2 = min(y2, gb[j, 3])
                        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
                        inter = iw * ih
                        area_p = (x2 - x1) * (y2 - y1)
                        area_g = (gb[j, 2] - gb[j, 0]) * (gb[j, 3] - gb[j, 1])
                        iou = inter / (area_p + area_g - inter + 1e-9)
                        if iou > best_iou:
                            best_iou = iou; best_j = j
                    if best_iou >= iou_t and best_j >= 0:
                        tp += 1; used_gt[best_j] = True
                    else:
                        fp += 1
                fn += int((~used_gt).sum())
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def scalar(v, default=0.0):
    if isinstance(v, (list, tuple)) and v:
        return float(v[0])
    try:
        fv = float(v)
        return fv if fv >= 0 else default
    except Exception:
        return default


def train_variant(variant: str, epochs: int, patience: int, batch: int = 4):
    device = torch.device("cuda:0")
    out_dir = TASK_DIR / f"detr_{variant}"
    cc.ensure_dir(out_dir)

    processor = make_processor()
    ds_dir = cc.DATASET_VARIANTS[variant].parent
    train_ds = DetrYoloDataset(ds_dir / "train/images", ds_dir / "train/labels", processor)
    val_ds = DetrYoloDataset(ds_dir / "val/images", ds_dir / "val/labels", processor)
    print(f"[detr/{variant}] n_train={len(train_ds)} n_val={len(val_ds)} batch={batch}")

    collate = collate_fn_factory(processor)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              collate_fn=collate, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            collate_fn=collate, num_workers=4, pin_memory=True)

    model = make_model().to(device)
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad], "lr": 1e-5},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)

    best_map50 = -1.0
    best_epoch = -1
    no_improve = 0
    rows = []

    for ep in range(1, epochs + 1):
        model.train()
        ep_t0 = time.time()
        tl_sum = 0.0
        n_batches = 0
        for batch_data in train_loader:
            pv = batch_data["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in lbl.items() if torch.is_tensor(v)}
                      for lbl in batch_data["labels"]]
            outputs = model(pixel_values=pv, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            tl_sum += float(loss.item())
            n_batches += 1

        train_loss = tl_sum / max(n_batches, 1)
        val_metrics = evaluate_map(model, val_loader, device, processor)
        m50 = scalar(val_metrics.get("map_50", 0.0))
        m5095 = scalar(val_metrics.get("map", 0.0))
        prec, rec = compute_pr_detr(model, val_loader, device, processor)
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
            best_map50 = m50; best_epoch = ep; no_improve = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            no_improve += 1

        cc.plot_learning_curves(
            out_dir / "metrics.csv", out_dir / "learning_curves.png",
            title=f"DETR / {variant}",
        )
        if no_improve >= patience:
            print(f"  early stop at ep{ep} (best={best_epoch} mAP50={best_map50:.3f})")
            break

    return model, out_dir, best_epoch, processor


def evaluate_on_test(model, variant: str, out_dir: Path, processor):
    device = torch.device("cuda:0")
    best_path = out_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    ds_dir = cc.DATASET_VARIANTS[variant].parent
    test_ds = DetrYoloDataset(ds_dir / "test/images", ds_dir / "test/labels", processor)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                             collate_fn=collate_fn_factory(processor), num_workers=4)

    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    model.eval()
    conf_matrix = np.zeros((10, 10), dtype=np.int64)
    n_instances = np.zeros(9, dtype=int)

    with torch.no_grad():
        for batch in test_loader:
            pv = batch["pixel_values"].to(device)
            labels = batch["labels"]
            outputs = model(pixel_values=pv)
            target_sizes = torch.stack([lbl["orig_size"] for lbl in labels]).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.0
            )
            preds = [{"boxes": r["boxes"].cpu(), "scores": r["scores"].cpu(),
                      "labels": r["labels"].cpu()} for r in results]
            gts = []
            for lbl in labels:
                bx = lbl["boxes"]
                cls = lbl["class_labels"]
                H, W = lbl["orig_size"].tolist()
                if bx.numel() > 0:
                    cx, cy, bw, bh = bx.unbind(-1)
                    x1 = (cx - bw / 2) * W; y1 = (cy - bh / 2) * H
                    x2 = (cx + bw / 2) * W; y2 = (cy + bh / 2) * H
                    abs_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                else:
                    abs_boxes = torch.zeros((0, 4))
                gts.append({"boxes": abs_boxes.cpu(), "labels": cls.cpu()})
            metric.update(preds, gts)

            for r, gt in zip(results, gts):
                gl = gt["labels"].numpy()
                gb = gt["boxes"].numpy()
                pl = r["labels"].cpu().numpy()
                ps = r["scores"].cpu().numpy()
                pb = r["boxes"].cpu().numpy()
                for c in gl:
                    if 0 <= c < 9:
                        n_instances[c] += 1
                keep = ps >= 0.25
                pb, pl = pb[keep], pl[keep]
                used_pred = np.zeros(len(pb), dtype=bool)
                for j in range(len(gb)):
                    gtcls = int(gl[j])
                    best_iou = 0.0; best_i = -1
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
                        conf_matrix[gtcls, int(pl[best_i])] += 1
                        used_pred[best_i] = True
                    else:
                        conf_matrix[gtcls, 9] += 1  # bg column

    mdict = metric.compute()
    per_class_m50 = mdict.get("map_50_per_class", [])
    per_class_m = mdict.get("map_per_class", [])
    classes_order = mdict.get("classes", [])
    if hasattr(classes_order, "cpu"):
        classes_order = classes_order.cpu().numpy().tolist()
    per_class_rows = []
    for cid in range(9):
        try:
            idx = classes_order.index(cid) if cid in classes_order else -1
            m50 = float(per_class_m50[idx]) if idx >= 0 and idx < len(per_class_m50) else ""
            m = float(per_class_m[idx]) if idx >= 0 and idx < len(per_class_m) else ""
        except Exception:
            m50 = ""; m = ""
        per_class_rows.append({
            "class_id": cid,
            "class_name": cc.CLASS_NAMES[cid],
            "n_test_instances": int(n_instances[cid]),
            "mAP50": round(m50, 4) if isinstance(m50, float) else "",
            "mAP50_95": round(m, 4) if isinstance(m, float) else "",
        })
    cc.save_per_class_map(out_dir / "per_class_map.csv", per_class_rows)
    cc.plot_confusion_matrix(conf_matrix, cc.CLASS_NAMES, out_dir / "confusion_matrix.png",
                             title=f"DETR / {variant}")
    return mdict


def measure_fps(model, processor, out_dir: Path, variant: str):
    device = torch.device("cuda:0")
    model.eval()
    qfiles = cc.qualitative_filenames()
    img = Image.open(cc.TEST_IMG_DIR / qfiles[0]).convert("RGB")
    enc = processor(images=img, return_tensors="pt")
    pv = enc["pixel_values"].to(device)
    with torch.no_grad():
        for _ in range(20):
            _ = model(pixel_values=pv)
        torch.cuda.synchronize()
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = model(pixel_values=pv)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
    cc.save_fps(out_dir / "fps_measurement.json", "detr", variant, latencies)


def render_qualitative(model, processor, variant: str, out_dir: Path):
    device = torch.device("cuda:0")
    model.eval()

    def predict_fn(img_path: str):
        img = Image.open(img_path).convert("RGB")
        H, W = img.size[1], img.size[0]
        enc = processor(images=img, return_tensors="pt")
        pv = enc["pixel_values"].to(device)
        with torch.no_grad():
            out = model(pixel_values=pv)
        r = processor.post_process_object_detection(
            out, target_sizes=torch.tensor([[H, W]]).to(device), threshold=0.25
        )[0]
        boxes = r["boxes"].cpu().numpy()
        scores = r["scores"].cpu().numpy()
        labels = r["labels"].cpu().numpy()
        return [{"box": b.tolist(), "cls": int(l), "score": float(s)}
                for b, l, s in zip(boxes, labels, scores)]

    cc.render_predictions_examples(out_dir / "predictions_examples", predict_fn)


def run_single(variant: str, epochs: int, patience: int, batch: int):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    out_dir = TASK_DIR / f"detr_{variant}"
    cc.ensure_dir(out_dir)
    print(f"=== detr / {variant} ===")
    model, out_dir, best_epoch, processor = train_variant(variant, epochs, patience, batch)
    evaluate_on_test(model, variant, out_dir, processor)
    measure_fps(model, processor, out_dir, variant)
    render_qualitative(model, processor, variant, out_dir)
    print(f"[detr/{variant}] done, best_epoch={best_epoch}")
    return out_dir


def build_summary_row(variant: str) -> dict:
    out_dir = TASK_DIR / f"detr_{variant}"
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
