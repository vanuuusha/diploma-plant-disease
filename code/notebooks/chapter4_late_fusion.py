"""
Task 14 — Late Fusion: контекст подключается после детекции.

Пайплайн:
  1. Загрузить замороженный baseline YOLOv12 (task_12/yolov12_baseline/weights/best.pt).
  2. Пройти по train/val/test, для каждого предсказанного bbox с IoU>=0.5
     к ground-truth собрать:
       - ROI-фичу через roi_align из P4-карты neck (7×7×C);
       - контекстный вектор от MobileNetV3-Small на 224×224;
       - GT-класс.
  3. Обучить LateFusionClassifier (30 эпох, AdamW, lr=1e-3).
  4. End-to-end инференс: baseline → bbox → LateFusion → новый класс.
  5. Пересчитать mAP@50 / mAP@50-95 / per-class с помощью torchmetrics.

Выход: code/results/task_14/yolov12_late_fusion/ + roi_dataset_*.pt + summary.csv.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.chapter4 import ContextEncoder, LateFusionClassifier  # noqa: E402

PROJECT = Path("code/results/task_14")
RUN = PROJECT / "yolov12_late_fusion"
RUN.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = Path("code/data/dataset_final")
BASELINE_WEIGHTS = Path("code/results/task_12/yolov12_baseline/weights/best.pt")
NUM_CLASSES = 9
ROI_SPATIAL = 7
CTX_DIM = 256
IOU_THRESH = 0.5


def iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """boxes in xyxy. returns [A, B]."""
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros(boxes_a.shape[0], boxes_b.shape[0])
    x1 = torch.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = torch.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = torch.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = torch.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def read_yolo_labels(lbl_path: Path, img_w: int, img_h: int) -> torch.Tensor:
    """Return [N, 5]: cls, x1, y1, x2, y2 (pixels)."""
    if not lbl_path.exists():
        return torch.zeros(0, 5)
    out = []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        out.append([cls, x1, y1, x2, y2])
    return torch.tensor(out) if out else torch.zeros(0, 5)


def collect_roi_dataset(y_model, ctx_enc, split: str, conf_thres: float = 0.1) -> dict:
    """Сбор ROI + контекста + GT-класса для одного сплита."""
    from ultralytics import YOLO
    from torchvision.ops import roi_align
    img_dir = DATASET / split / "images"
    lbl_dir = DATASET / split / "labels"
    files = sorted([p for p in img_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"[{split}] {len(files)} images")

    det_model = y_model.model
    det_model.eval()
    ctx_enc.eval()

    # Определяем индекс P4 neck (средний уровень в Detect.f)
    detect = det_model.model[-1]
    p4_idx = detect.f[1]  # P3, P4, P5

    # hook для сохранения P4 feature
    p4_feat = {}

    def p4_hook(module, inp, out):
        p4_feat["map"] = out
    handle = det_model.model[p4_idx].register_forward_hook(p4_hook)

    rois = []
    ctxs = []
    labels = []

    t0 = time.time()
    with torch.no_grad():
        for i, img_p in enumerate(files):
            try:
                img = Image.open(img_p).convert("RGB")
            except Exception:
                continue
            lbl_p = lbl_dir / (img_p.stem + ".txt")
            iw, ih = img.size
            gt = read_yolo_labels(lbl_p, iw, ih)  # [N, 5]
            if gt.numel() == 0:
                continue

            # predict через YOLO
            result = y_model.predict(source=str(img_p), conf=conf_thres,
                                     imgsz=640, verbose=False, device=DEVICE)[0]
            pred = result.boxes
            if pred is None or pred.xyxy.shape[0] == 0:
                continue
            pred_boxes = pred.xyxy.cpu()
            pred_cls = pred.cls.cpu().long()

            # match по IoU
            iou = iou_matrix(pred_boxes, gt[:, 1:5])
            if iou.numel() == 0:
                continue
            max_iou, max_idx = iou.max(dim=1)
            ok_mask = max_iou >= IOU_THRESH
            if not ok_mask.any():
                continue

            matched_boxes = pred_boxes[ok_mask]
            matched_labels = gt[max_idx[ok_mask], 0].long()

            # roi_align из P4. Нужно знать коэффициент masштаба P4 относительно 640×640.
            fmap = p4_feat.get("map")
            if fmap is None:
                continue
            H_f, W_f = fmap.shape[-2], fmap.shape[-1]
            # boxes приведены к сетке изображения 640x640, а P4 ~ 40x40 → scale ≈ 1/16
            scale = W_f / 640.0
            # roi_align принимает boxes формата [batch_idx, x1, y1, x2, y2]; B=1
            boxes_ra = torch.cat(
                [torch.zeros(matched_boxes.shape[0], 1),
                 matched_boxes * (640 / max(iw, ih))], dim=1
            ).to(fmap.device)
            # Уточнение: predict уже масштабировал боксы в оригинальные пиксели изображения.
            # Но P4-карта построена для 640×640 ресайза. Используем letterbox-совместимость:
            # reset: пересчитаем в 640-пространство.
            # Ultralytics letterbox сохраняет aspect — упростим: используем отношение 640/max(iw,ih)
            # Для roi_align spatial_scale укажем 1/16 (приближённо).
            roi = roi_align(fmap, boxes_ra, output_size=(ROI_SPATIAL, ROI_SPATIAL),
                            spatial_scale=scale, sampling_ratio=2)

            # контекст всей сцены
            img_224 = img.resize((224, 224))
            x_ctx = torch.tensor(np.array(img_224), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            c = ctx_enc(x_ctx.to(DEVICE))  # [1, 256]
            c_rep = c.repeat(roi.shape[0], 1)

            rois.append(roi.cpu())
            ctxs.append(c_rep.cpu())
            labels.append(matched_labels)

            if (i + 1) % 200 == 0:
                print(f"  [{split}] {i+1}/{len(files)} "
                      f"rois={sum(r.shape[0] for r in rois)} "
                      f"elapsed={time.time()-t0:.0f}s")

    handle.remove()

    if rois:
        rois_t = torch.cat(rois, dim=0)
        ctxs_t = torch.cat(ctxs, dim=0)
        labels_t = torch.cat(labels, dim=0)
    else:
        rois_t = torch.zeros(0, 1, ROI_SPATIAL, ROI_SPATIAL)
        ctxs_t = torch.zeros(0, CTX_DIM)
        labels_t = torch.zeros(0, dtype=torch.long)
    print(f"[{split}] total rois={rois_t.shape[0]}")
    return {"roi": rois_t, "context": ctxs_t, "label": labels_t}


def train_classifier(ds_train: dict, ds_val: dict, roi_channels: int) -> tuple[nn.Module, pd.DataFrame]:
    model = LateFusionClassifier(
        roi_channels=roi_channels, roi_spatial=ROI_SPATIAL,
        context_dim=CTX_DIM, num_classes=NUM_CLASSES,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    crit = nn.CrossEntropyLoss()

    N_tr = ds_train["roi"].shape[0]
    N_va = ds_val["roi"].shape[0]
    B = 128

    history = []
    best_val = 1e9
    patience = 5
    no_improve = 0
    for epoch in range(30):
        model.train()
        perm = torch.randperm(N_tr)
        tr_loss = 0.0
        for i in range(0, N_tr, B):
            idx = perm[i:i + B]
            roi = ds_train["roi"][idx].to(DEVICE)
            ctx = ds_train["context"][idx].to(DEVICE)
            y = ds_train["label"][idx].to(DEVICE)
            opt.zero_grad()
            logits = model(roi, ctx)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * idx.shape[0]
        tr_loss /= max(N_tr, 1)
        # val
        model.eval()
        va_loss = 0.0
        va_correct = 0
        with torch.no_grad():
            for i in range(0, N_va, B):
                roi = ds_val["roi"][i:i + B].to(DEVICE)
                ctx = ds_val["context"][i:i + B].to(DEVICE)
                y = ds_val["label"][i:i + B].to(DEVICE)
                logits = model(roi, ctx)
                va_loss += crit(logits, y).item() * roi.shape[0]
                va_correct += (logits.argmax(-1) == y).sum().item()
        va_loss /= max(N_va, 1)
        va_acc = va_correct / max(N_va, 1)
        history.append({"epoch": epoch + 1, "train_loss": tr_loss,
                        "val_loss": va_loss, "val_acc": va_acc})
        print(f"epoch {epoch+1:02d} tr {tr_loss:.4f} va {va_loss:.4f} acc {va_acc:.3f}")
        if va_loss < best_val - 1e-4:
            best_val = va_loss
            no_improve = 0
            torch.save(model.state_dict(), RUN / "late_head.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stop at epoch {epoch+1}")
                break
        sched.step()

    model.load_state_dict(torch.load(RUN / "late_head.pt", weights_only=True))
    return model, pd.DataFrame(history)


def evaluate_end_to_end(y_model, ctx_enc, head_model, roi_channels: int):
    """End-to-end инференс на test: baseline → bbox → LateFusion → новый класс.
    Метрики считаем через torchmetrics.MeanAveragePrecision."""
    from torchmetrics.detection import MeanAveragePrecision
    from torchvision.ops import roi_align

    img_dir = DATASET / "test" / "images"
    lbl_dir = DATASET / "test" / "labels"
    files = sorted([p for p in img_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    det_model = y_model.model
    det_model.eval()
    ctx_enc.eval()
    head_model.eval()

    detect = det_model.model[-1]
    p4_idx = detect.f[1]
    p4_feat = {}

    def p4_hook(m, i, o):
        p4_feat["map"] = o
    handle = det_model.model[p4_idx].register_forward_hook(p4_hook)

    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    def format_to_metric(boxes, cls, scores, is_pred: bool):
        if is_pred:
            return {"boxes": boxes, "scores": scores, "labels": cls}
        return {"boxes": boxes, "labels": cls}

    t0 = time.time()
    with torch.no_grad():
        for i, img_p in enumerate(files):
            try:
                img = Image.open(img_p).convert("RGB")
            except Exception:
                continue
            lbl_p = lbl_dir / (img_p.stem + ".txt")
            iw, ih = img.size
            gt = read_yolo_labels(lbl_p, iw, ih)

            # GT
            gt_boxes = gt[:, 1:5] if gt.numel() else torch.zeros(0, 4)
            gt_cls = gt[:, 0].long() if gt.numel() else torch.zeros(0, dtype=torch.long)

            result = y_model.predict(source=str(img_p), conf=0.1, imgsz=640,
                                     verbose=False, device=DEVICE)[0]
            boxes_pred = (result.boxes.xyxy.cpu() if result.boxes is not None
                          else torch.zeros(0, 4))
            scores_pred = (result.boxes.conf.cpu() if result.boxes is not None
                           else torch.zeros(0))
            cls_pred_orig = (result.boxes.cls.cpu().long()
                             if result.boxes is not None else torch.zeros(0, dtype=torch.long))

            if boxes_pred.shape[0] > 0:
                fmap = p4_feat.get("map")
                H_f, W_f = fmap.shape[-2], fmap.shape[-1]
                scale = W_f / 640.0
                boxes_ra = torch.cat(
                    [torch.zeros(boxes_pred.shape[0], 1),
                     boxes_pred * (640 / max(iw, ih))], dim=1
                ).to(fmap.device)
                roi = roi_align(fmap, boxes_ra,
                                output_size=(ROI_SPATIAL, ROI_SPATIAL),
                                spatial_scale=scale, sampling_ratio=2)
                img_224 = img.resize((224, 224))
                x_ctx = torch.tensor(np.array(img_224), dtype=torch.float32).permute(
                    2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
                c = ctx_enc(x_ctx).repeat(roi.shape[0], 1)
                logits = head_model(roi, c)
                cls_new = logits.argmax(-1).cpu().long()
            else:
                cls_new = cls_pred_orig

            # обновление metric
            metric.update(
                [format_to_metric(boxes_pred, cls_new, scores_pred, is_pred=True)],
                [format_to_metric(gt_boxes, gt_cls, None, is_pred=False)],
            )
            if (i + 1) % 100 == 0:
                print(f"  test {i+1}/{len(files)} elapsed={time.time()-t0:.0f}s")

    handle.remove()
    out = metric.compute()
    return out


def main():
    from ultralytics import YOLO
    print("=" * 60)
    print("TASK 14 — Late Fusion")
    print("=" * 60)
    print(f"[device] {DEVICE}")

    # 1. baseline frozen
    assert BASELINE_WEIGHTS.exists(), f"missing {BASELINE_WEIGHTS}"
    y = YOLO(str(BASELINE_WEIGHTS))
    y.model.to(DEVICE)
    for p in y.model.parameters():
        p.requires_grad = False

    ctx_enc = ContextEncoder("mobilenetv3_small_100", out_dim=CTX_DIM, pretrained=True).to(DEVICE)
    for p in ctx_enc.parameters():
        p.requires_grad = False

    # 2. ROI dataset
    splits = {}
    for split in ["train", "val", "test"]:
        out_p = PROJECT / f"roi_dataset_{split}.pt"
        if out_p.exists():
            print(f"[skip] {out_p} already exists")
            splits[split] = torch.load(out_p, weights_only=True)
        else:
            splits[split] = collect_roi_dataset(y, ctx_enc, split)
            torch.save(splits[split], out_p)
    roi_channels = splits["train"]["roi"].shape[1]
    print(f"[roi_channels] {roi_channels}")

    # 3. train classifier
    head, history = train_classifier(splits["train"], splits["val"], roi_channels)
    history.to_csv(RUN / "metrics.csv", index=False)
    # learning curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["epoch"], history["train_loss"], label="train")
    ax[0].plot(history["epoch"], history["val_loss"], label="val")
    ax[0].set_title("CrossEntropy"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(history["epoch"], history["val_acc"], color="C2")
    ax[1].set_title("Val accuracy"); ax[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(RUN / "learning_curves.png", dpi=120); plt.close()

    # 4. end-to-end eval
    metric = evaluate_end_to_end(y, ctx_enc, head, roi_channels)
    print("METRICS:", metric)
    metric_s = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metric.items()}
    with open(RUN / "test_metrics.json", "w") as f:
        json.dump(metric_s, f, indent=2)

    # 5. summary
    row = {
        "config": "yolov12_late_fusion",
        "mAP50": float(metric["map_50"]),
        "mAP50-95": float(metric["map"]),
        "test_rois": int(splits["test"]["roi"].shape[0]),
        "train_rois": int(splits["train"]["roi"].shape[0]),
        "val_rois": int(splits["val"]["roi"].shape[0]),
    }
    pd.DataFrame([row]).to_csv(PROJECT / "summary.csv", index=False)
    print(f"[done] summary → {PROJECT/'summary.csv'}")


if __name__ == "__main__":
    main()
