"""
Универсальный runner для экспериментов главы 4 (YOLOv12-ветвь).

CLI:
  python chapter4_runner.py --config se_neck --out code/results/task_13 --batch 16 --epochs 100
  python chapter4_runner.py --config cbam_neck --out code/results/task_13
  python chapter4_runner.py --config cgfm --encoder mobilenetv3_small_100 --levels P3 P4 P5 --out code/results/task_15
  python chapter4_runner.py --config cgfm --encoder efficientnet_b0 --levels P3 P4 P5 --out code/results/task_16 --name yolov12_cgfm_abl_effb0
  python chapter4_runner.py --config cgfm --encoder mobilenetv3_small_100 --levels P5 --out code/results/task_16 --name yolov12_cgfm_abl_p5only

Поддерживаемые конфигурации:
  - se_neck         : SE блоки на P3/P4/P5 (kind='self')
  - cbam_neck       : CBAM блоки на P3/P4/P5 (kind='self')
  - cgfm            : FiLM + контекстный энкодер (kind='film'), уровни и энкодер настраиваются

Выход: результаты Ultralytics в <out>/<name>/ + доп. артефакты (param_count.json,
context_embeddings.npy для cgfm).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.chapter4.film_layer import FiLMLayer
from models.chapter4.se_block import SEBlock
from models.chapter4.cbam_block import CBAMBlock
from models.chapter4.context_encoder import ContextEncoder
from models.chapter4.yolov12_patch import wrap_neck_with, describe_wrap

import torch
import torch.nn as nn
import torch.nn.functional as F

CTX_DIM = 256


class _InternalContextEncoder(nn.Module):
    """Контекст из самой YOLOv12: GAP на backbone-P5 (SPPF output) + MLP → out_dim.

    Реализация: forward_hook на backbone-P5 слое (последний слой до первого
    Upsample) сразу инжектирует контекст во все film_layers. Detector forward
    последовательный: backbone → P5 hook срабатывает → context готов →
    neck (P3/P4/P5) с FiLM использует контекст. Всё в одном проходе.

    Атрибут `is_internal_hook = True` — сигнал для `wrap_neck_with` не
    делать monkey-patch `forward` (контекст поставляется hook'ом).
    """

    is_internal_hook = True

    def __init__(self, yolo_model, out_dim: int = 256):
        super().__init__()
        # Находим backbone-P5: слой перед первым Upsample в model.model
        from torch.nn.modules.upsampling import Upsample
        bp5_idx = None
        for i, layer in enumerate(yolo_model.model):
            if isinstance(layer, Upsample):
                bp5_idx = i - 1
                break
        if bp5_idx is None:
            bp5_idx = 9  # fallback для YOLOv12m
        # Dry-run для определения числа каналов
        device = next(yolo_model.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640, device=device)
            cache = {}
            h = yolo_model.model[bp5_idx].register_forward_hook(
                lambda m, i, o: cache.__setitem__("out", o))
            yolo_model(dummy)
            h.remove()
            p5_channels = cache["out"].shape[1]
        self.bp5_idx = bp5_idx
        self.p5_channels = p5_channels
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(p5_channels),
            nn.Linear(p5_channels, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.film_layers: list = []  # заполняется wrap_neck_with
        # Регистрируем hook — при прохождении backbone-P5 инжектирует
        # context во все film_layers
        def _inject(module, inputs, output):
            c = self.proj(output)  # [B, out_dim]
            for fl in self.film_layers:
                fl.set_context(c)
        self._hook = yolo_model.model[bp5_idx].register_forward_hook(_inject)
        print(f"[InternalContextEncoder] bp5_idx={bp5_idx}  channels={p5_channels}")

    def forward(self, x):
        # Неиспользуется напрямую — hook делает всю работу
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   choices=["se_neck", "cbam_neck", "cgfm"])
    p.add_argument("--out", required=True, help="project directory")
    p.add_argument("--name", default=None, help="run name (subdir); default auto")
    p.add_argument("--data", default="code/data/dataset_final/data.yaml")
    p.add_argument("--weights", default="yolo12m.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder", default="mobilenetv3_small_100",
                   choices=["mobilenetv3_small_100", "efficientnet_b0",
                            "vit_tiny_patch16_224"])
    p.add_argument("--levels", nargs="+", default=["P3", "P4", "P5"])
    p.add_argument("--warmup_epochs", type=int, default=3,
                   help="для CGFM: сколько эпох фриза детектора")
    p.add_argument("--skip_warmup", action="store_true",
                   help="обучать end-to-end сразу (без warm-up)")
    p.add_argument("--film_variant", default="default",
                   choices=["default", "wide", "beta_noise"])
    p.add_argument("--film_residual", action="store_true",
                   help="FiLM с residual α-gated, F' = F + α·(γF + β - F)")
    p.add_argument("--internal_context", action="store_true",
                   help="контекст = GAP(P5) от detector'а, без ContextEncoder")
    p.add_argument("--mix_cbam", action="store_true",
                   help="перед CGFM применить CBAM на тех же уровнях (микс)")
    return p.parse_args()


def build_model(args):
    from ultralytics import YOLO
    y = YOLO(args.weights)
    film_layers = []
    context_encoder = None
    if args.config == "se_neck":
        info = wrap_neck_with(y.model, block_factory=SEBlock,
                              context_encoder=None, levels=args.levels)
    elif args.config == "cbam_neck":
        info = wrap_neck_with(y.model, block_factory=CBAMBlock,
                              context_encoder=None, levels=args.levels)
    else:  # cgfm
        # Сначала применяем CBAM (если микс), потом FiLM — два wrap подряд
        if args.mix_cbam:
            wrap_neck_with(y.model, block_factory=CBAMBlock,
                           context_encoder=None, levels=args.levels)
        factory = lambda ch: FiLMLayer(
            context_dim=CTX_DIM, feature_channels=ch,
            variant=args.film_variant, residual=args.film_residual,
        )
        if args.internal_context:
            # внутренний контекст: GAP(P5) от самой YOLOv12 (baseline-features)
            # FiLM получит 512-мерный вектор (channels P5 для YOLOv12m = 512)
            # После эмбеддинг-проекции в 256 (как у MobileNetV3 context)
            context_encoder = _InternalContextEncoder(y.model, out_dim=CTX_DIM)
            factory = lambda ch: FiLMLayer(
                context_dim=CTX_DIM, feature_channels=ch,
                variant=args.film_variant, residual=args.film_residual,
            )
        else:
            context_encoder = ContextEncoder(args.encoder, out_dim=CTX_DIM, pretrained=True)
        info = wrap_neck_with(
            y.model, block_factory=factory,
            context_encoder=context_encoder, levels=args.levels,
        )
        film_layers = info["modulated_layers"]
    print(f"[model] config={args.config} wrap={describe_wrap(info)}")
    return y, info, context_encoder, film_layers


def freeze_detector_except_modulation(y, info, context_encoder):
    """Этап warm-up: заморозить всё кроме FiLM + context_encoder."""
    modulated_layers = info["modulated_layers"]
    mod_params = set()
    for ml in modulated_layers:
        for p in ml.block.parameters():
            mod_params.add(id(p))
    ctx_params = set(id(p) for p in context_encoder.parameters()) if context_encoder else set()
    for p in y.model.parameters():
        if id(p) in mod_params or id(p) in ctx_params:
            p.requires_grad = True
        else:
            p.requires_grad = False
    n_tr = sum(p.numel() for p in y.model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in y.model.parameters())
    print(f"[warmup] trainable {n_tr/1e6:.2f}M / {n_tot/1e6:.2f}M")


def unfreeze_all(y):
    for p in y.model.parameters():
        p.requires_grad = True


def train(args, y, info, context_encoder):
    run_name = args.name or _default_name(args)
    # Общие аргументы
    common = dict(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        project=args.out,
        name=run_name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        # Все встроенные аугментации отключены (как в протоколе гл. 3)
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
        close_mosaic=0, erasing=0.0, crop_fraction=1.0,
    )

    # Warm-up: через Ultralytics callback on_train_epoch_start
    # (двойной .train() ломается в Ultralytics 8.3.240 — KeyError 'model').
    if args.config == "cgfm" and not args.skip_warmup and args.warmup_epochs > 0:
        print(f"[stage-1→2] warm-up {args.warmup_epochs} эпох (FiLM+ctx only), "
              f"затем end-to-end до {args.epochs} эпох")
        # заморозим сразу
        freeze_detector_except_modulation(y, info, context_encoder)
        _wu = args.warmup_epochs

        def _unfreeze_cb(trainer):
            if trainer.epoch == _wu:
                print(f"[stage-2] unfreezing detector at epoch {trainer.epoch}", flush=True)
                for p in trainer.model.parameters():
                    p.requires_grad = True
                # пересобрать optimizer с новыми params
                trainer.optimizer = trainer.build_optimizer(
                    model=trainer.model, name=trainer.args.optimizer,
                    lr=trainer.args.lr0, momentum=trainer.args.momentum,
                    decay=trainer.args.weight_decay, iterations=trainer.args.epochs,
                ) if hasattr(trainer, "build_optimizer") else trainer.optimizer

        y.add_callback("on_train_epoch_start", _unfreeze_cb)
    y.train(epochs=args.epochs, patience=args.patience, **common)

    return Path(args.out) / run_name


def save_extra_artifacts(run_dir: Path, args, y, info, context_encoder):
    # param_count.json
    try:
        from thop import profile
        n_total = sum(p.numel() for p in y.model.parameters())
        info_pc = {
            "params_total": n_total,
            "params_total_M": round(n_total / 1e6, 3),
        }
        if context_encoder is not None:
            info_pc["context_encoder_params_M"] = round(
                sum(p.numel() for p in context_encoder.parameters()) / 1e6, 3)
        try:
            y.model.eval()
            x = torch.zeros(1, 3, 640, 640, device=next(y.model.parameters()).device)
            macs, _ = profile(y.model, inputs=(x,), verbose=False)
            info_pc["gflops"] = round(2 * macs / 1e9, 3)
        except Exception as e:
            info_pc["gflops_error"] = str(e)
        with open(run_dir / "param_count.json", "w") as f:
            json.dump(info_pc, f, indent=2)
        print(f"[param_count] {info_pc}")
    except Exception as e:
        print(f"[param_count] skipped: {e}")

    # context_embeddings.npy для CGFM
    if args.config == "cgfm" and context_encoder is not None:
        test_dir = Path("code/data/dataset_final/test/images")
        # учтём что на A100 путь другой
        if not test_dir.exists():
            alt = Path(args.data).parent / "test" / "images"
            if alt.exists():
                test_dir = alt
        if test_dir.exists():
            _dump_embeddings(run_dir, context_encoder, test_dir)


def _dump_embeddings(run_dir: Path, encoder, test_dir: Path):
    from PIL import Image
    import torchvision.transforms.functional as TF
    device = next(encoder.parameters()).device
    encoder.eval()
    files = sorted([p for p in test_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    B = 16
    feats = []
    names = []
    batch = []
    with torch.no_grad():
        for i, p in enumerate(files):
            try:
                img = Image.open(p).convert("RGB").resize((224, 224))
            except Exception:
                continue
            t = TF.to_tensor(img)
            batch.append(t)
            names.append(p.name)
            if len(batch) == B or i == len(files) - 1:
                x = torch.stack(batch).to(device)
                feats.append(encoder(x).cpu().numpy())
                batch = []
    if feats:
        feats = np.concatenate(feats, axis=0)
    else:
        feats = np.zeros((0, 256))
    np.save(run_dir / "context_embeddings.npy", feats)
    with open(run_dir / "context_embeddings_filenames.json", "w") as f:
        json.dump(names, f, indent=2)
    print(f"[embeddings] saved {feats.shape}")


def _default_name(args) -> str:
    if args.config == "se_neck":
        return "yolov12_se_neck"
    if args.config == "cbam_neck":
        return "yolov12_cbam_neck"
    # cgfm
    enc_abbr = {"mobilenetv3_small_100": "mnv3",
                "efficientnet_b0": "effb0",
                "vit_tiny_patch16_224": "vittiny"}[args.encoder]
    lvl_abbr = "".join(l.lower() for l in args.levels)
    return f"yolov12_cgfm_{enc_abbr}_{lvl_abbr}"


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    y, info, ctx_enc, film_layers = build_model(args)
    t0 = time.time()
    run_dir = train(args, y, info, ctx_enc)
    print(f"[train] finished in {(time.time() - t0)/60:.1f} min")
    save_extra_artifacts(run_dir, args, y, info, ctx_enc)


if __name__ == "__main__":
    main()
