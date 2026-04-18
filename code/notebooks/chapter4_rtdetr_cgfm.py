"""
Task 17 — RT-DETR + CGFM. FiLM-модуляция backbone feature maps через forward_hook.

Пайплайн:
  1. HF RTDetrForObjectDetection на класс 9.
  2. Регистрируем forward_hook на RTDetrConvEncoder: hook модифицирует
     feature_maps в output, применяя FiLM(features, context).
  3. Контекст вычисляется в обёртке forward на основе pixel_values.
  4. Training loop простой: собственный, без HF Trainer.

Usage:
  python chapter4_rtdetr_cgfm.py --out code/results/task_17 --data ~/dataset_final \
      --epochs 50 --batch 8 --variant cgfm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.chapter4 import ContextEncoder, FiLMLayer  # noqa: E402


CLASS_NAMES = [
    "Недостаток P2O5", "Листовая (бурая) ржавчина", "Мучнистая роса",
    "Пиренофороз", "Фузариоз", "Корневая гниль", "Септориоз",
    "Недостаток N", "Повреждение заморозками",
]
NUM_CLASSES = len(CLASS_NAMES)
CTX_DIM = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--variant", default="cgfm", choices=["cgfm", "baseline"])
    p.add_argument("--name", default=None)
    p.add_argument("--data", default="code/data/dataset_final")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder", default="mobilenetv3_small_100")
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--checkpoint", default="PekingU/rtdetr_r50vd")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, split: str, processor):
        self.img_dir = root / split / "images"
        self.lbl_dir = root / split / "labels"
        self.files = sorted([p for p in self.img_dir.iterdir()
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        self.processor = processor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        img_p = self.files[i]
        img = Image.open(img_p).convert("RGB")
        iw, ih = img.size
        lbl_p = self.lbl_dir / (img_p.stem + ".txt")
        ann = []
        if lbl_p.exists():
            for line in lbl_p.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                x1 = (cx - w / 2) * iw
                y1 = (cy - h / 2) * ih
                ww = w * iw
                hh = h * ih
                ann.append({"bbox": [x1, y1, ww, hh],
                            "category_id": cls,
                            "area": ww * hh, "iscrowd": 0})
        target = {"image_id": i, "annotations": ann}
        enc = self.processor(images=img, annotations=target, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "labels": enc["labels"][0],
            "orig_h": ih, "orig_w": iw,
            "file": str(img_p),
        }


def collate(batch):
    pv = [b["pixel_values"] for b in batch]
    max_h = max(t.shape[1] for t in pv)
    max_w = max(t.shape[2] for t in pv)
    padded = []
    for t in pv:
        c, h, w = t.shape
        if h == max_h and w == max_w:
            padded.append(t)
        else:
            pad = torch.zeros((c, max_h, max_w), dtype=t.dtype)
            pad[:, :h, :w] = t
            padded.append(pad)
    return {
        "pixel_values": torch.stack(padded),
        "labels": [b["labels"] for b in batch],
        "orig_sizes": torch.tensor([[b["orig_h"], b["orig_w"]] for b in batch]),
        "files": [b["file"] for b in batch],
    }


def install_cgfm_hooks(model, context_encoder, device):
    """Регистрирует forward_hook на RTDetrConvEncoder, модифицирующий feature maps.
    Контекст сохраняется в model._ctx_holder через обёртку forward."""

    # Dry-run backbone для определения каналов
    bb = model.model.backbone
    # HF RTDetrConvEncoder.forward(pixel_values, pixel_mask)
    # Создадим dummy pixel_mask
    pm = torch.ones(1, 640, 640, dtype=torch.long, device=device)
    pv = torch.randn(1, 3, 640, 640, device=device)
    model.eval()
    with torch.no_grad():
        try:
            bb_out = bb(pv, pm)
        except Exception as e:
            print(f"[warn] backbone probe with pixel_mask failed: {e}")
            try:
                bb_out = bb(pv)
            except Exception as e2:
                raise RuntimeError(f"backbone probe failed: {e2}")

    # bb_out — обычно список tuples (features, mask)
    channels = []
    if isinstance(bb_out, (list, tuple)):
        for item in bb_out:
            if isinstance(item, (list, tuple)) and len(item) > 0 and hasattr(item[0], "shape"):
                channels.append(item[0].shape[1])
            elif hasattr(item, "shape"):
                channels.append(item.shape[1])
    elif hasattr(bb_out, "feature_maps"):
        channels = [f.shape[1] for f in bb_out.feature_maps]

    print(f"[cgfm] backbone channels: {channels}")
    if not channels:
        raise RuntimeError(f"cannot determine channels; bb_out type {type(bb_out)}")

    # Создаём FiLM-слои
    film_layers = nn.ModuleList([
        FiLMLayer(context_dim=CTX_DIM, feature_channels=c) for c in channels
    ]).to(device)

    # holder для контекста
    model._cgfm = {"context": None, "films": film_layers,
                   "context_encoder": context_encoder}

    def hook(module, inputs, output):
        c = model._cgfm["context"]
        if c is None:
            return output
        films = model._cgfm["films"]
        # output — список tuples (features, mask)
        if isinstance(output, (list, tuple)):
            new_out = []
            for i, item in enumerate(output):
                if isinstance(item, (list, tuple)) and len(item) >= 1 and hasattr(item[0], "shape") and item[0].ndim == 4:
                    mod = films[i](item[0], c)
                    rest = item[1:]
                    new_out.append((mod, *rest))
                elif hasattr(item, "shape") and item.ndim == 4:
                    new_out.append(films[i](item, c))
                else:
                    new_out.append(item)
            # сохранить тип
            if isinstance(output, tuple):
                return tuple(new_out)
            return new_out
        return output

    handle = bb.register_forward_hook(hook)
    return film_layers, handle


class CGFMWrapper(nn.Module):
    def __init__(self, base_model, context_encoder, device):
        super().__init__()
        self.base = base_model
        self.context_encoder = context_encoder
        self.film_layers, self.hook_handle = install_cgfm_hooks(base_model, context_encoder, device)
        # register in parameters
        for m in [context_encoder, self.film_layers]:
            for n, p in m.named_parameters():
                self.register_parameter("__" + n.replace(".", "_"), p) if False else None
        # nn.Module уже регистрирует submodules — ok

    def forward(self, pixel_values, labels=None, **kw):
        # контекст на 224x224
        x_ctx = F.interpolate(pixel_values.float(), size=(224, 224),
                              mode="bilinear", align_corners=False)
        self.base._cgfm["context"] = self.context_encoder(x_ctx)
        try:
            return self.base(pixel_values=pixel_values, labels=labels, **kw)
        finally:
            self.base._cgfm["context"] = None


def compute_map(preds, targets):
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric.update(preds, targets)
    return metric.compute()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.name or f"rtdetr_{args.variant}"
    run_dir = Path(args.out) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}  device={device}")

    from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
    processor = RTDetrImageProcessor.from_pretrained(args.checkpoint)
    base = RTDetrForObjectDetection.from_pretrained(
        args.checkpoint,
        num_labels=NUM_CLASSES,
        id2label={i: n for i, n in enumerate(CLASS_NAMES)},
        label2id={n: i for i, n in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,
    ).to(device)

    if args.variant == "cgfm":
        ctx_enc = ContextEncoder(args.encoder, out_dim=CTX_DIM, pretrained=True).to(device)
        model = CGFMWrapper(base, ctx_enc, device).to(device)
    else:
        model = base

    # Dataset
    root = Path(args.data)
    ds_train = YoloDataset(root, "train", processor)
    ds_val = YoloDataset(root, "val", processor)
    ds_test = YoloDataset(root, "test", processor)
    print(f"[data] train {len(ds_train)}  val {len(ds_val)}  test {len(ds_test)}")

    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batch, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers,
    )
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size=args.batch, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers,
    )

    # Optimizer
    if args.variant == "cgfm":
        new_params = [p for n, p in model.named_parameters()
                      if "context_encoder" in n or "film_layers" in n]
        old_params = [p for n, p in model.named_parameters()
                      if "context_encoder" not in n and "film_layers" not in n]
        optimizer = torch.optim.AdamW([
            {"params": old_params, "lr": 1e-4},
            {"params": new_params, "lr": 5e-4},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Training
    history = []
    best_val = 1e9
    no_improve = 0
    t0 = time.time()
    # Отключаем AMP — FiLM модуляция в fp16 приводит к NaN.
    scaler = None

    for epoch in range(args.epochs):
        # warmup: первые warmup_epochs эпох — только film + context_encoder
        if args.variant == "cgfm" and epoch < args.warmup_epochs:
            for n, p in model.named_parameters():
                p.requires_grad = ("context_encoder" in n or "film_layers" in n)
        elif epoch == args.warmup_epochs and args.variant == "cgfm":
            for p in model.parameters():
                p.requires_grad = True

        model.train()
        tr_loss = 0.0
        n_b = 0
        for batch in dl_train:
            pv = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in l.items()} for l in batch["labels"]]
            optimizer.zero_grad()
            out = model(pixel_values=pv, labels=labels)
            loss = out.loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[warn] nan/inf loss at epoch {epoch+1} batch {n_b}, skipping")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
            n_b += 1
        tr_loss /= max(n_b, 1)

        # Val loss
        model.eval()
        va_loss = 0.0
        n_vb = 0
        with torch.no_grad():
            for batch in dl_val:
                pv = batch["pixel_values"].to(device)
                labels = [{k: v.to(device) for k, v in l.items()} for l in batch["labels"]]
                out = model(pixel_values=pv, labels=labels)
                va_loss += out.loss.item()
                n_vb += 1
        va_loss /= max(n_vb, 1)

        elapsed = (time.time() - t0) / 60
        print(f"epoch {epoch+1:3d}/{args.epochs}  tr {tr_loss:.4f}  va {va_loss:.4f}  "
              f"elapsed {elapsed:.1f}m", flush=True)
        history.append({"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": va_loss})
        pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            no_improve = 0
            # save only state_dict (weights saved as best.pt for consistency)
            torch.save({"state_dict": model.state_dict(),
                        "epoch": epoch + 1, "val_loss": va_loss},
                       run_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stop at epoch {epoch+1}", flush=True)
                break

    # Final eval on test
    print("[eval] test set", flush=True)
    if (run_dir / "best.pt").exists():
        ckpt = torch.load(run_dir / "best.pt", weights_only=False, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    model.eval()

    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    with torch.no_grad():
        for batch in dl_test:
            pv = batch["pixel_values"].to(device)
            orig_sizes = batch["orig_sizes"].to(device)
            out = model(pixel_values=pv)
            results = processor.post_process_object_detection(
                out, target_sizes=orig_sizes.float(), threshold=0.0,
            )
            # targets конвертируем из HF-формата
            targets = []
            for l, sz in zip(batch["labels"], orig_sizes):
                h, w = sz.cpu().tolist()
                cxcywh = l["boxes"].cpu()
                if cxcywh.numel():
                    x1 = (cxcywh[:, 0] - cxcywh[:, 2] / 2) * w
                    y1 = (cxcywh[:, 1] - cxcywh[:, 3] / 2) * h
                    x2 = (cxcywh[:, 0] + cxcywh[:, 2] / 2) * w
                    y2 = (cxcywh[:, 1] + cxcywh[:, 3] / 2) * h
                    bxyxy = torch.stack([x1, y1, x2, y2], dim=-1)
                else:
                    bxyxy = torch.zeros(0, 4)
                targets.append({"boxes": bxyxy, "labels": l["class_labels"].cpu()})
            preds = [{"boxes": r["boxes"].cpu(), "scores": r["scores"].cpu(),
                      "labels": r["labels"].cpu()} for r in results]
            metric.update(preds, targets)
    res = metric.compute()
    print(f"[eval] mAP@50={res['map_50']:.4f}  mAP@50-95={res['map']:.4f}", flush=True)
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items()},
                  f, indent=2)

    # per_class
    pc = res.get("map_per_class", None)
    if pc is not None and hasattr(pc, "tolist"):
        pcl = pc.tolist()
        pd.DataFrame({
            "class_id": list(range(len(pcl))),
            "class_name": CLASS_NAMES[:len(pcl)],
            "mAP@50": pcl,
        }).to_csv(run_dir / "per_class_map.csv", index=False)

    row = {
        "config": run_name,
        "detector": "rtdetr",
        "mAP@50": float(res["map_50"]),
        "mAP@50-95": float(res["map"]),
    }
    summary_path = Path(args.out) / "summary.csv"
    pd.DataFrame([row]).to_csv(summary_path, index=False, mode="a",
                                header=not summary_path.exists())
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
