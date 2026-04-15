"""
Сравнение NST v2 и Stable Diffusion img2img на ОДНИХ И ТЕХ ЖЕ входных изображениях.
Для каждого из 4 редких классов:
  - 2 content seed-снимка + 2 style/seed snapshot
  - Генерация по каждому методу
  - Сетка: [seed | diffusion strength=0.4 | diffusion strength=0.6 | NST v2]
Результат: красивые PNG для вставки в диплом + сравнительный CSV.
"""

import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from augment_nst_v2 import stylize, load_image, tensor_to_pil, list_class_images

DATASET = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset"
OUT_DIR = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06/compare"
SIZE = 512
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
torch.manual_seed(SEED)

PROMPTS = {
    5: "macro photograph of wheat plant root crown with brown root rot disease symptoms, fungal infection, agricultural close-up, natural lighting, high detail",
    6: "wheat leaf with septoria leaf spot disease, brown spots and yellow halos, agricultural field photography, natural lighting, high detail",
    7: "wheat leaves showing nitrogen deficiency, yellowing of older leaves, pale green tips, agricultural field photo, natural lighting",
    8: "wheat leaves with frost damage, white patches and necrotic edges, early winter morning agricultural field, high detail",
}
NEG = "drawing, painting, blurry, low quality, watermark, text, logo, frame, illustration"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(DATASET, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]
    cls_imgs = list_class_images(classes)

    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    rare_ids = [5, 6, 7, 8]
    content_pool = []
    for c in (1, 2, 3):
        content_pool.extend(cls_imgs[c][:20])
    random.Random(SEED).shuffle(content_pool)

    # Общий лог
    rows = []
    per_class_grids = {}

    for cid in rare_ids:
        if not cls_imgs[cid]: continue
        style_seeds = cls_imgs[cid][:2]
        content_seeds = content_pool[:2]
        class_grid = []  # список строк: [seed, diff0.4, diff0.6, nst]
        for ci, cp in enumerate(content_seeds):
            # для diffusion используем CONTENT как seed (img2img)
            content_pil = Image.open(cp).convert("RGB").resize((SIZE, SIZE))
            # style seed (для nst)
            sp = style_seeds[ci % len(style_seeds)]
            prompt = PROMPTS[cid]

            # Diffusion strength=0.4
            gen = torch.Generator(device=DEVICE).manual_seed(SEED + cid * 100 + ci)
            d04 = pipe(prompt=prompt, negative_prompt=NEG, image=content_pil,
                       strength=0.4, num_inference_steps=30, guidance_scale=7.5, generator=gen).images[0]

            # Diffusion strength=0.6
            gen = torch.Generator(device=DEVICE).manual_seed(SEED + cid * 100 + ci + 1)
            d06 = pipe(prompt=prompt, negative_prompt=NEG, image=content_pil,
                       strength=0.6, num_inference_steps=30, guidance_scale=7.5, generator=gen).images[0]

            # NST v2
            content_t = load_image(cp, SIZE)
            style_t = load_image(sp, SIZE)
            nst_out = stylize(content_t, style_t)
            nst_pil = tensor_to_pil(nst_out)

            # Сохранить отдельные файлы
            safe = classes[cid].replace(" ", "_").replace("/", "_")
            base = f"{safe}_cmp{ci + 1:02d}"
            content_pil.save(os.path.join(OUT_DIR, base + "_0_seed.png"))
            d04.save(os.path.join(OUT_DIR, base + "_1_diffusion_str04.png"))
            d06.save(os.path.join(OUT_DIR, base + "_2_diffusion_str06.png"))
            nst_pil.save(os.path.join(OUT_DIR, base + "_3_nst_v2.png"))

            rows.append([classes[cid], cp, sp, base + "_0_seed.png", base + "_1_diffusion_str04.png",
                         base + "_2_diffusion_str06.png", base + "_3_nst_v2.png"])
            class_grid.append((cp, sp, content_pil, d04, d06, nst_pil))
            print(f"  {classes[cid]} cmp{ci + 1}: seed={Path(cp).name} style={Path(sp).name}")

        per_class_grids[cid] = class_grid

    # CSV-лог
    import csv
    with open(os.path.join(OUT_DIR, "compare_log.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["class", "seed_content", "style_ref", "seed.png", "diffusion_str04.png", "diffusion_str06.png", "nst_v2.png"]] + rows)

    # Грид: на каждый класс отдельный PNG (2 строки × 4 колонки)
    for cid, grid in per_class_grids.items():
        n_rows = len(grid); n_cols = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1: axes = axes[None, :]
        titles = ["Исходное (content)", "Diffusion str=0.4", "Diffusion str=0.6", "NST v2 (LBFGS)"]
        for r, (cp, sp, seed_pil, d04, d06, nst_pil) in enumerate(grid):
            imgs = [seed_pil, d04, d06, nst_pil]
            for c, im in enumerate(imgs):
                axes[r, c].imshow(im)
                axes[r, c].set_title(titles[c] if r == 0 else "", fontsize=11)
                axes[r, c].axis("off")
        fig.suptitle(f"Сравнение методов генерации: {classes[cid]}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"compare_{classes[cid].replace(' ', '_')}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Общий мастер-грид (по одной строке на класс, 4 колонки методов) — для диплома
    all_classes = list(per_class_grids.keys())
    fig, axes = plt.subplots(len(all_classes), 4, figsize=(16, 4 * len(all_classes)))
    if len(all_classes) == 1: axes = axes[None, :]
    titles = ["Исходное (content)", "Diffusion str=0.4", "Diffusion str=0.6", "NST v2 (LBFGS)"]
    for r, cid in enumerate(all_classes):
        cp, sp, seed_pil, d04, d06, nst_pil = per_class_grids[cid][0]  # первый пример класса
        imgs = [seed_pil, d04, d06, nst_pil]
        for c, im in enumerate(imgs):
            axes[r, c].imshow(im)
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=12)
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(classes[cid], fontsize=13, rotation=0, ha="right", va="center", labelpad=80)
    fig.suptitle("Сравнение NST и Stable Diffusion img2img", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "MASTER_COMPARISON.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nСохранено в {OUT_DIR}")


if __name__ == "__main__":
    main()
