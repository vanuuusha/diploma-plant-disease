"""
Улучшенный Neural Style Transfer для редких классов.
Ключевые отличия от augment_nst.py:
- Оптимизатор **L-BFGS** (классика Gatys 2016) вместо Adam — качественно сильнее.
- Разрешение **512×512** вместо 384×384.
- **300 итераций** (быстрее сходится на L-BFGS, чем 250 на Adam).
- **Color preservation** — перенос только яркостной компоненты (YUV), чтобы сохранить зелёный цвет листьев content.
- Инициализация target из content + небольшой гауссов шум (лучшая стабильность).

Используется для генерации и для сравнения с Diffusion.
"""

import os
import csv
import random
import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.models import vgg19, VGG19_Weights
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import yaml

DATASET = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset"
OUT_ROOT = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 512
STEPS = 300
SEED = 42
PRESERVE_COLOR = True  # сохранить хроматические каналы content

random.seed(SEED)
torch.manual_seed(SEED)


def load_image(path, size=SIZE):
    img = Image.open(path).convert("RGB")
    return T.Compose([T.Resize((size, size)), T.ToTensor()])(img).unsqueeze(0).to(DEVICE)


def tensor_to_pil(t):
    return T.ToPILImage()(t.detach().clamp(0, 1).cpu().squeeze(0))


def rgb_to_yuv(img):
    """img: (1,3,H,W) in [0,1]. Returns Y, UV separately."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return y, torch.cat([u, v], dim=1)


def yuv_to_rgb(y, uv):
    u, v = uv[:, 0:1], uv[:, 1:2]
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u
    return torch.cat([r, g, b], dim=1)


class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.style_layers = {0, 5, 10, 19, 28}
        self.content_layer = 21
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.style_layers:
                out[("s", i)] = x
            if i == self.content_layer:
                out[("c", i)] = x
            if i > max(max(self.style_layers), self.content_layer):
                break
        return out


def gram(x):
    b, c, h, w = x.shape
    f = x.view(c, h * w)
    return (f @ f.t()) / (c * h * w)


def stylize(content, style, steps=STEPS, content_w=1.0, style_w=1e6, tv_w=1e-4):
    extractor = VGGFeatures().to(DEVICE)
    # init: content + small noise
    target = (content.clone() + 0.01 * torch.randn_like(content)).clamp(0, 1)
    target.requires_grad_(True)

    with torch.no_grad():
        sf = extractor(style)
        cf = extractor(content)
        style_grams = {k: gram(v) for k, v in sf.items() if k[0] == "s"}
        content_feats = {k: v.clone() for k, v in cf.items() if k[0] == "c"}

    optimizer = optim.LBFGS([target], max_iter=20, history_size=50, line_search_fn="strong_wolfe")

    step = [0]

    def closure():
        optimizer.zero_grad()
        target.data.clamp_(0, 1)
        tf = extractor(target)
        c_loss = sum(((tf[k] - content_feats[k]) ** 2).mean() for k in content_feats)
        s_loss = 0
        for k in style_grams:
            s_loss = s_loss + ((gram(tf[k]) - style_grams[k]) ** 2).mean()
        # total variation for smoothness
        diff_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        diff_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        tv = (diff_x.pow(2).mean() + diff_y.pow(2).mean())
        loss = content_w * c_loss + style_w * s_loss + tv_w * tv
        loss.backward()
        step[0] += 1
        return loss

    # 15 внешних проходов × до 20 внутренних = до 300 итераций
    outer = steps // 20
    for _ in range(outer):
        optimizer.step(closure)
        target.data.clamp_(0, 1)

    if PRESERVE_COLOR:
        y_target, _ = rgb_to_yuv(target.detach())
        _, uv_content = rgb_to_yuv(content.detach())
        target = yuv_to_rgb(y_target, uv_content).clamp(0, 1)

    return target.detach()


def list_class_images(classes):
    train_img = os.path.join(DATASET, "train", "images")
    train_lbl = os.path.join(DATASET, "train", "labels")
    cls_imgs = defaultdict(list)
    for fn in os.listdir(train_img):
        stem, _ = os.path.splitext(fn)
        lbl = os.path.join(train_lbl, stem + ".txt")
        if not os.path.exists(lbl): continue
        with open(lbl) as f:
            cls_set = set()
            for line in f:
                p = line.strip().split()
                if len(p) == 5:
                    cls_set.add(int(float(p[0])))
        for c in cls_set:
            cls_imgs[c].append(os.path.join(train_img, fn))
    return cls_imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_subdir", default="nst_v2", help="подпапка в task_06/ для результатов")
    parser.add_argument("--per_class", type=int, default=10, help="сколько изображений на редкий класс")
    parser.add_argument("--content_pool_size", type=int, default=30, help="из скольких content-снимков выбирать")
    parser.add_argument("--seeds_per_class", type=int, default=4, help="сколько разных style-снимков использовать")
    args = parser.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(DATASET, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]

    cls_imgs = list_class_images(classes)
    rare_class_ids = [5, 6, 7, 8]  # Корневая гниль, Септориоз, Недостаток N, Заморозки

    content_pool = []
    for c in (1, 2, 3):
        content_pool.extend(cls_imgs[c][: args.content_pool_size])
    random.Random(SEED).shuffle(content_pool)

    log_rows = [["method", "class_id", "class_name", "content_path", "style_path", "out_path"]]

    for cid in rare_class_ids:
        if not cls_imgs[cid]:
            continue
        styles = cls_imgs[cid][: args.seeds_per_class]
        safe = classes[cid].replace(" ", "_").replace("/", "_")
        class_contents = list(content_pool)
        random.Random(SEED + cid).shuffle(class_contents)
        n_generated = 0
        needed = args.per_class
        for si, sp in enumerate(styles):
            if n_generated >= needed: break
            per_style = max(1, needed // len(styles))
            for ci in range(per_style):
                if n_generated >= needed: break
                cp = class_contents[(si * per_style + ci) % len(class_contents)]
                try:
                    style = load_image(sp); content = load_image(cp)
                    out = stylize(content, style)
                    out_path = os.path.join(out_dir, f"{safe}_nst_{n_generated + 1:03d}.png")
                    tensor_to_pil(out).save(out_path)
                    log_rows.append(["NST-LBFGS-v2", cid, classes[cid], cp, sp, out_path])
                    n_generated += 1
                    print(f"  [{classes[cid]}] {n_generated}/{needed}  style={Path(sp).name}  content={Path(cp).name}")
                except Exception as e:
                    print(f"  ! NST fail: {e}")

    log_path = os.path.join(out_dir, "generation_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(log_rows)
    print(f"\nNST v2: {len(log_rows) - 1} изображений в {out_dir}")


if __name__ == "__main__":
    main()
