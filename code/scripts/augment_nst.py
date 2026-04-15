"""
Neural Style Transfer (Gatys et al. 2016) на VGG19 для редких классов.
Перенос стиля симптомов болезни с больного листа на здоровый лист.
В нашем датасете "здоровых" нет, поэтому в роли content берётся изображение
другого редкого класса — сами симптомы расширяются на новый визуальный фон.
"""

import os
import csv
import random
from collections import defaultdict, Counter
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
import yaml

DATASET = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset"
OUT_DIR = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06"
NST_DIR = os.path.join(OUT_DIR, "nst")
LOG_CSV = os.path.join(OUT_DIR, "generation_log.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 384
STEPS = 250
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


def load_image(path, size=SIZE):
    img = Image.open(path).convert("RGB")
    return T.Compose([T.Resize((size, size)), T.ToTensor()])(img).unsqueeze(0).to(DEVICE)


def save_image(tensor, path):
    img = tensor.detach().clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
    Image.fromarray((img * 255).astype("uint8")).save(path)


class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.style_layers = {0, 5, 10, 19, 28}  # conv1_1 conv2_1 conv3_1 conv4_1 conv5_1
        self.content_layer = 21  # conv4_2

    def forward(self, x):
        # ImageNet норм.
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        feats = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.style_layers:
                feats[("s", i)] = x
            if i == self.content_layer:
                feats[("c", i)] = x
            if i > max(max(self.style_layers), self.content_layer):
                break
        return feats


def gram(x):
    b, c, h, w = x.shape
    f = x.view(c, h * w)
    return (f @ f.t()) / (c * h * w)


def stylize(content, style, steps=STEPS, content_w=1.0, style_w=1e6):
    extractor = VGGFeatures()
    target = content.clone().requires_grad_(True)
    opt = optim.Adam([target], lr=0.02)
    sf = extractor(style)
    cf = extractor(content)
    style_grams = {k: gram(v) for k, v in sf.items() if k[0] == "s"}
    content_feats = {k: v.detach() for k, v in cf.items() if k[0] == "c"}

    for step in range(steps):
        opt.zero_grad()
        tf = extractor(target)
        c_loss = sum(((tf[k] - content_feats[k]) ** 2).mean() for k in content_feats)
        s_loss = 0
        for k in style_grams:
            s_loss = s_loss + ((gram(tf[k]) - style_grams[k]) ** 2).mean()
        loss = content_w * c_loss + style_w * s_loss
        loss.backward()
        opt.step()
        with torch.no_grad():
            target.clamp_(0, 1)
    return target


def list_class_images(classes):
    """cls_id -> [image_path,...] (берём из train)"""
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
                    cls_set.add(int(p[0]))
        for c in cls_set:
            cls_imgs[c].append(os.path.join(train_img, fn))
    return cls_imgs


def main():
    os.makedirs(NST_DIR, exist_ok=True)
    with open(os.path.join(DATASET, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]

    cls_imgs = list_class_images(classes)

    # Редкие классы по результатам task_03: id 5,6,7,8 (Корневая гниль, Септориоз, Недостаток N, Повреждение заморозками)
    # «Здоровых» нет, поэтому в качестве content берём изображения других классов.
    rare_class_ids = [5, 6, 7, 8]
    content_pool = []
    for c in (1, 2, 3):  # «здоровые в контексте» — крупные классы, имеющие зелёные листья
        content_pool.extend(cls_imgs[c][:30])
    random.Random(SEED).shuffle(content_pool)

    log_rows = [["method", "class_id", "class_name", "content_path", "style_path", "out_path"]]
    examples = []  # для визуализации (cid, content, style, result)

    for cid in rare_class_ids:
        styles = cls_imgs[cid][:5]
        contents = content_pool[:5]
        if not styles or not contents:
            continue
        random.Random(SEED + cid).shuffle(styles)
        random.Random(SEED + cid).shuffle(contents)
        n = 0
        for si, sp in enumerate(styles[:3]):
            for ci, cp in enumerate(contents[:5]):
                if n >= 15:
                    break
                try:
                    style = load_image(sp); content = load_image(cp)
                    out = stylize(content, style)
                    safe = classes[cid].replace(" ", "_").replace("/", "_")
                    out_path = os.path.join(NST_DIR, f"{safe}_nst_{n+1:03d}.png")
                    save_image(out, out_path)
                    log_rows.append(["NST", cid, classes[cid], cp, sp, out_path])
                    if len(examples) < 4 and ci == 0 and si == 0:
                        examples.append((cid, cp, sp, out_path))
                    n += 1
                    print(f"  [{classes[cid]}] {n}/15  style={Path(sp).name} content={Path(cp).name}")
                except Exception as e:
                    print(f"  ! NST fail: {e}")

    # лог
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(log_rows)

    # визуализация
    if examples:
        n = len(examples)
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
        if n == 1: axes = axes[None, :]
        for r, (cid, cp, sp, op) in enumerate(examples):
            for col, (path, title) in enumerate([(cp, "content"), (sp, f"style ({classes[cid]})"), (op, "результат NST")]):
                axes[r, col].imshow(Image.open(path).convert("RGB").resize((SIZE, SIZE)))
                axes[r, col].set_title(title); axes[r, col].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "nst_examples.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"\nNST: всего {len(log_rows)-1} изображений, лог: {LOG_CSV}")


if __name__ == "__main__":
    main()
