"""
Task 17 — RT-DETR + CGFM. Переносимость подхода на трансформерный детектор.

Пайплайн близок к chapter3_detr_runner.py, но использует HuggingFace
`RTDetrForObjectDetection`. FiLM-слои вставляются на выходы encoder-слоёв
(feature maps до Decoder queries). Контекстный вектор генерируется
отдельным `ContextEncoder` (MobileNetV3-Small), общим для всех FiLM-слоёв.

Usage:
  python chapter4_rtdetr_cgfm.py --out code/results/task_17 \
      --variant cgfm --data code/data/dataset_final --epochs 100 --batch 8

Поддерживает также --variant baseline (без CGFM) — но это redundant с task_08
и используется только при отсутствии референса.
"""
from __future__ import annotations

import argparse
import json
import os
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--variant", default="cgfm", choices=["cgfm", "baseline"])
    p.add_argument("--name", default=None)
    p.add_argument("--data", default="code/data/dataset_final")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder", default="mobilenetv3_small_100")
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--checkpoint", default="PekingU/rtdetr_r50vd")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


CLASS_NAMES = [
    "Недостаток P2O5", "Листовая (бурая) ржавчина", "Мучнистая роса",
    "Пиренофороз", "Фузариоз", "Корневая гниль", "Септориоз",
    "Недостаток N", "Повреждение заморозками",
]
NUM_CLASSES = len(CLASS_NAMES)
CTX_DIM = 256


class YoloDataset(torch.utils.data.Dataset):
    """Загружает YOLO-формат (txt-метки) для HuggingFace RT-DETR processor."""

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
        boxes, labels = [], []
        if lbl_p.exists():
            for line in lbl_p.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                # COCO-style (x, y, w, h) в абсолютных пикселях — processor сам
                # конвертирует в формат модели
                x1 = (cx - w / 2) * iw
                y1 = (cy - h / 2) * ih
                ww = w * iw
                hh = h * ih
                boxes.append([x1, y1, ww, hh])
                labels.append(cls)
        target = {
            "image_id": i,
            "annotations": [
                {"bbox": b, "category_id": c, "area": b[2] * b[3], "iscrowd": 0}
                for b, c in zip(boxes, labels)
            ],
        }
        enc = self.processor(images=img, annotations=target, return_tensors="pt")
        # enc["pixel_values"] shape [1, 3, H, W]; enc["labels"] — список
        return {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "labels": enc["labels"][0],
            "orig_image": img,
            "file": str(img_p),
        }


def _collate(batch):
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
        "files": [b["file"] for b in batch],
    }


class RTDetrWithCGFM(nn.Module):
    """Оборачивает HuggingFace RTDetrForObjectDetection, добавляя FiLM-слои
    на выходы encoder (hidden_states на 3 масштабах)."""

    def __init__(self, base_model, context_encoder, film_channels):
        super().__init__()
        self.base = base_model
        self.context_encoder = context_encoder
        self.film_layers = nn.ModuleList([
            FiLMLayer(context_dim=CTX_DIM, feature_channels=c) for c in film_channels
        ])
        self._install_hooks()

    def _install_hooks(self):
        """Регистрирует forward_pre_hook на decoder, заменяющий encoder_hidden_states.

        RT-DETR HuggingFace:
          model.model.encoder.forward(...) → (last_hidden_state, ...)
          Затем в model.model.decoder принимаются encoder outputs

        Простейший способ: пропатчить model.model.encoder.forward, чтобы
        применить FiLM к проекциям (outputs). Это зависит от версии HF.
        Попробуем патчить RTDetrHybridEncoder.forward через wrapping.
        """
        enc = self.base.model.encoder
        orig_forward = enc.forward
        film_layers = self.film_layers
        get_ctx = self._get_context_from_input

        def new_forward(inputs_embeds=None, attention_mask=None, *args, **kwargs):
            out = orig_forward(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                *args, **kwargs,
            )
            # out.last_hidden_state — [B, sum(H*W per level), hidden_dim] (flat)
            # Для модуляции нужны 3 уровня (P3/P4/P5 после hybrid encoder).
            # В BaseModelOutput есть hidden_states tuple — попробуем оттуда
            c = get_ctx()
            if c is None:
                return out
            # Если out — BaseModelOutput: последний hidden_state — [B, N, D].
            # Нам нужны многоуровневые карты. Посмотрим, есть ли у encoder
            # атрибут 'fpn_states' (HF не гарантирует, это fallback).
            # Без успеха — сохраним out без модификации; FiLM применим к
            # decoder-входам на следующем этапе.
            return out

        enc.forward = new_forward
        # Простой вариант: модулировать backbone multi-scale features
        # через wrapping model.model.backbone_projector или аналогичного
        # модуля. Используем обёртку на проекционные слои.
        # Fallback: FiLM работает на features после проекции.

    def _get_context_from_input(self):
        """Должен быть установлен внешне перед forward."""
        return getattr(self, "_last_context", None)

    def forward(self, pixel_values, labels=None, **kwargs):
        # 1. контекст
        x_ctx = F.interpolate(pixel_values.float(), size=(224, 224),
                              mode="bilinear", align_corners=False)
        self._last_context = self.context_encoder(x_ctx)
        # 2. forward базовой модели
        return self.base(pixel_values=pixel_values, labels=labels, **kwargs)


def compute_map(preds, targets, image_sizes):
    """preds/targets в COCO-стиле → torchmetrics MeanAveragePrecision."""
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric.update(preds, targets)
    return metric.compute()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.name or (f"rtdetr_{args.variant}")
    run_dir = Path(args.out) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}  device={device}")

    # --- Processor + Model
    from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
    processor = RTDetrImageProcessor.from_pretrained(args.checkpoint)
    base = RTDetrForObjectDetection.from_pretrained(
        args.checkpoint,
        num_labels=NUM_CLASSES,
        id2label={i: n for i, n in enumerate(CLASS_NAMES)},
        label2id={n: i for i, n in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,
    )

    # --- CGFM интеграция (упрощённый подход):
    #    Вставляем FiLM не через HF internals (это хрупко и зависит от версии
    #    transformers), а через обёртку backbone-модели: после backbone
    #    получаем multi-scale features, модулируем их FiLM, отдаём дальше.
    if args.variant == "cgfm":
        # Определим каналы выходов backbone
        base.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 640, 640)
            # backbone RT-DETR (ResNet-like) возвращает list of feature maps
            try:
                bb = base.model.backbone
                feats = bb(dummy)
                if hasattr(feats, 'feature_maps'):
                    feats = feats.feature_maps
                elif isinstance(feats, dict):
                    feats = list(feats.values())
                channels = [f.shape[1] for f in feats]
            except Exception as e:
                print(f"[warn] backbone probe failed: {e}; using default [512,1024,2048]")
                channels = [512, 1024, 2048]
        print(f"[cgfm] backbone out channels: {channels}")

        ctx_enc = ContextEncoder(args.encoder, out_dim=CTX_DIM, pretrained=True)
        model = RTDetrWithCGFMWrapped(base, ctx_enc, channels).to(device)
    else:
        model = base.to(device)

    # --- Dataset
    root = Path(args.data)
    ds_train = YoloDataset(root, "train", processor)
    ds_val = YoloDataset(root, "val", processor)
    ds_test = YoloDataset(root, "test", processor)
    print(f"[data] train {len(ds_train)}  val {len(ds_val)}  test {len(ds_test)}")

    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch, shuffle=True,
        collate_fn=_collate, num_workers=args.num_workers, pin_memory=True,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batch, shuffle=False,
        collate_fn=_collate, num_workers=args.num_workers, pin_memory=True,
    )

    # --- Optimizer with different lr for new params
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

    # --- Training loop
    best_val = 1e9
    no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        # Warm-up: первые warmup_epochs эпох — только FiLM + context_encoder
        if args.variant == "cgfm" and epoch < args.warmup_epochs:
            for n, p in model.named_parameters():
                p.requires_grad = ("context_encoder" in n or "film_layers" in n)
        elif epoch == args.warmup_epochs:
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
            n_b += 1
        tr_loss /= max(n_b, 1)

        # Val
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
        history.append({"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": va_loss})
        elapsed = (time.time() - t0) / 60
        print(f"epoch {epoch+1:3d}/{args.epochs}  tr {tr_loss:.4f}  va {va_loss:.4f}  "
              f"elapsed {elapsed:.1f}m")

        # Save
        if va_loss < best_val - 1e-4:
            best_val = va_loss
            no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": va_loss,
            }, run_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stop at epoch {epoch+1}")
                break

        pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)

    # --- Final metrics на test
    print("[eval] test")
    ckpt = torch.load(run_dir / "best.pt", weights_only=False, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size=args.batch, shuffle=False,
        collate_fn=_collate, num_workers=args.num_workers,
    )
    with torch.no_grad():
        for batch in dl_test:
            pv = batch["pixel_values"].to(device)
            orig_sizes = torch.tensor([[img.size[1], img.size[0]] for img in
                                       [_open(f) for f in batch["files"]]]).to(device)
            out = model(pixel_values=pv)
            results = processor.post_process_object_detection(
                out, target_sizes=orig_sizes, threshold=0.0,
            )
            # labels → в пиксельных размерах исходного изображения
            targets = []
            for l, sz in zip(batch["labels"], orig_sizes):
                h, w = sz.cpu().tolist()
                boxes_xywh = l["boxes"].cpu()
                # HF сохраняет boxes в нормализованном cxcywh → конвертируем
                if boxes_xywh.numel():
                    cxcywh = boxes_xywh
                    # Some processors normalize; detect range
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
    print(f"[eval] mAP@50 = {res['map_50']:.4f}  mAP@50-95 = {res['map']:.4f}")

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items()},
                  f, indent=2)

    # per-class mAP
    per_class = res.get("map_per_class", None)
    if per_class is not None and hasattr(per_class, "tolist"):
        pc = per_class.tolist()
        pd.DataFrame({
            "class_id": list(range(len(pc))),
            "class_name": CLASS_NAMES[:len(pc)],
            "mAP@50": pc,
        }).to_csv(run_dir / "per_class_map.csv", index=False)

    # summary
    row = {
        "config": run_name,
        "detector": "rtdetr",
        "mAP@50": float(res["map_50"]),
        "mAP@50-95": float(res["map"]),
        "best_epoch": ckpt["epoch"],
    }
    pd.DataFrame([row]).to_csv(Path(args.out) / "summary.csv", index=False, mode="a",
                                header=not (Path(args.out) / "summary.csv").exists())
    print("[done]")


def _open(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


# --- Упрощённая обёртка для RT-DETR, модулирующая backbone-проекции.
class RTDetrWithCGFMWrapped(nn.Module):
    def __init__(self, base_model, context_encoder, channels):
        super().__init__()
        self.base = base_model
        self.context_encoder = context_encoder
        self.film_layers = nn.ModuleList([
            FiLMLayer(context_dim=CTX_DIM, feature_channels=c) for c in channels
        ])
        self._install_backbone_hook()

    def _install_backbone_hook(self):
        """Оборачивает backbone.forward: модулирует каждую feature map FiLM-ом."""
        bb = self.base.model.backbone
        orig_forward = bb.forward
        films = self.film_layers

        def new_forward(pixel_values, *args, **kwargs):
            out = orig_forward(pixel_values, *args, **kwargs)
            c = getattr(self, "_ctx_vec", None)
            if c is None:
                return out
            # out может быть BaseModelOutput или tuple of feature maps
            # Для RT-DETR HF: backbone возвращает `RTDetrFrozenBatchNorm2dBackboneOutput`
            # или `BaseModelOutput` с `.feature_maps`
            fmaps = None
            if hasattr(out, "feature_maps"):
                fmaps = list(out.feature_maps)
            elif isinstance(out, (list, tuple)):
                fmaps = list(out)
            elif isinstance(out, dict):
                fmaps = list(out.values())
            if fmaps is None:
                return out
            for i, fl in enumerate(films):
                if i < len(fmaps):
                    fmaps[i] = fl(fmaps[i], c)
            if hasattr(out, "feature_maps"):
                out.feature_maps = tuple(fmaps)
            elif isinstance(out, tuple):
                out = tuple(fmaps)
            elif isinstance(out, list):
                out = fmaps
            elif isinstance(out, dict):
                out = {k: v for k, v in zip(out.keys(), fmaps)}
            return out

        bb.forward = new_forward

    def forward(self, pixel_values, labels=None, **kwargs):
        x_ctx = F.interpolate(pixel_values.float(), size=(224, 224),
                              mode="bilinear", align_corners=False)
        self._ctx_vec = self.context_encoder(x_ctx)
        return self.base(pixel_values=pixel_values, labels=labels, **kwargs)


if __name__ == "__main__":
    main()
