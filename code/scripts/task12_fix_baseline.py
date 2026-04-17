"""
task_12: Фиксация baseline главы 4 — копирует артефакты YOLOv12 aug_diffusion
из task_07 в task_12/yolov12_baseline/, дополнительно сохраняет:
  - param_count.json (число параметров и GFLOPs)
  - context_embeddings.npy (MobileNetV3-Small на test-срезе, без обучения)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.chapter4 import ContextEncoder  # noqa: E402


SRC = Path("code/results/task_07/yolov12_aug_diffusion")
DST = Path("code/results/task_12/yolov12_baseline")


def copy_reference():
    DST.mkdir(parents=True, exist_ok=True)
    keep = [
        "metrics.csv", "per_class_map.csv", "fps_measurement.json",
        "confusion_matrix.png", "confusion_matrix_normalized.png",
        "learning_curves.png", "results.csv", "results.png",
        "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png", "BoxPR_curve.png",
        "labels.jpg", "args.yaml",
    ]
    for name in keep:
        src = SRC / name
        if src.exists():
            shutil.copy2(src, DST / name)
    # predictions_examples — копируем всю папку
    pe_src = SRC / "predictions_examples"
    pe_dst = DST / "predictions_examples"
    if pe_src.exists() and not pe_dst.exists():
        shutil.copytree(pe_src, pe_dst)
    # best.pt → symlink (слишком большой для git, но нужен для downstream)
    best_src = SRC / "weights" / "best.pt"
    weights_dst = DST / "weights"
    weights_dst.mkdir(exist_ok=True)
    best_dst = weights_dst / "best.pt"
    if best_src.exists() and not best_dst.exists():
        best_dst.symlink_to(best_src.resolve())
    print(f"[copy] → {DST}")


def compute_param_count():
    from ultralytics import YOLO
    try:
        from thop import profile
    except ImportError:
        profile = None
    best = DST / "weights" / "best.pt"
    if not best.exists():
        print("skip param_count: best.pt absent")
        return
    y = YOLO(str(best))
    # числа параметров
    n_total = sum(p.numel() for p in y.model.parameters())
    n_train = sum(p.numel() for p in y.model.parameters() if p.requires_grad)
    info = {
        "params_total": n_total,
        "params_total_M": round(n_total / 1e6, 3),
        "params_trainable": n_train,
    }
    if profile is not None:
        try:
            y.model.eval()
            x = torch.zeros(1, 3, 640, 640)
            macs, _ = profile(y.model, inputs=(x,), verbose=False)
            info["macs"] = int(macs)
            info["gflops"] = round(2 * macs / 1e9, 3)
        except Exception as e:
            info["macs"] = None
            info["gflops_error"] = str(e)
    with open(DST / "param_count.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[param_count] {info}")


def compute_context_embeddings():
    """Прогон MobileNetV3-Small (без обучения) на test-срезе → [N,256]."""
    test_dir = Path("code/data/dataset_final/test/images")
    if not test_dir.exists():
        print("skip embeddings: test dir absent")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = ContextEncoder("mobilenetv3_small_100", out_dim=256, pretrained=True).to(device)
    enc.eval()
    from PIL import Image
    import torchvision.transforms.functional as TF

    files = sorted([p for p in test_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"[embeddings] {len(files)} test images on {device}")
    feats = []
    names = []
    batch_imgs = []
    B = 16
    with torch.no_grad():
        for i, p in enumerate(files):
            try:
                img = Image.open(p).convert("RGB").resize((224, 224))
            except Exception:
                continue
            t = TF.to_tensor(img)
            batch_imgs.append(t)
            names.append(p.name)
            if len(batch_imgs) == B or i == len(files) - 1:
                x = torch.stack(batch_imgs).to(device)
                c = enc(x).cpu().numpy()
                feats.append(c)
                batch_imgs = []
    feats = np.concatenate(feats, axis=0) if feats else np.zeros((0, 256))
    np.save(DST / "context_embeddings.npy", feats)
    with open(DST / "context_embeddings_filenames.json", "w") as f:
        json.dump(names, f, indent=2)
    print(f"[embeddings] saved {feats.shape}")


if __name__ == "__main__":
    copy_reference()
    compute_param_count()
    compute_context_embeddings()
    print("task_12 baseline: done")
