"""
Diffusion-аугментация (img2img) для редких классов.
Используется Stable Diffusion v1.5 (runwayml) через diffusers.
Если нет доступа к HF, скрипт мягко падает и логирует причину.
"""

import os
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset"
OUT_DIR = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06"
DIFF_DIR = os.path.join(OUT_DIR, "diffusion")
LOG_CSV = os.path.join(OUT_DIR, "diffusion_log.csv")
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
NEG_PROMPT = "drawing, painting, blurry, low quality, watermark, text, logo, frame, illustration"


def list_class_images(classes):
    train_img = os.path.join(DATASET, "train", "images")
    train_lbl = os.path.join(DATASET, "train", "labels")
    cls_imgs = defaultdict(list)
    for fn in os.listdir(train_img):
        stem, _ = os.path.splitext(fn)
        lbl = os.path.join(train_lbl, stem + ".txt")
        if not os.path.exists(lbl): continue
        with open(lbl) as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 5:
                    cls_imgs[int(p[0])].append(os.path.join(train_img, fn))
                    break
    return cls_imgs


def main():
    os.makedirs(DIFF_DIR, exist_ok=True)
    with open(os.path.join(DATASET, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]

    try:
        from diffusers import StableDiffusionImg2ImgPipeline
    except ImportError as e:
        print(f"diffusers недоступен: {e}")
        with open(LOG_CSV, "w") as f:
            f.write("error,diffusers_not_installed\n")
        return

    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    print(f"Загрузка модели {model_id} (DEVICE={DEVICE})…")
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
        ).to(DEVICE)
        pipe.set_progress_bar_config(disable=True)
    except Exception as e:
        print(f"Не удалось загрузить модель {model_id}: {e}")
        # fallback на sd-2-1
        try:
            model_id = "Lykon/dreamshaper-7"
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
            ).to(DEVICE)
            pipe.set_progress_bar_config(disable=True)
        except Exception as e2:
            print(f"Запасная модель тоже недоступна: {e2}")
            with open(LOG_CSV, "w") as f:
                f.write(f"error,model_unavailable: {e2}\n")
            return

    cls_imgs = list_class_images(classes)
    log_rows = [["method", "class_id", "class_name", "seed_path", "strength", "prompt", "out_path"]]
    examples = {}  # cid -> (seed_path, [outs])

    for cid in (5, 6, 7, 8):
        if cid not in cls_imgs or not cls_imgs[cid]:
            continue
        seeds = random.Random(SEED + cid).sample(cls_imgs[cid], min(3, len(cls_imgs[cid])))
        prompt = PROMPTS[cid]
        safe = classes[cid].replace(" ", "_").replace("/", "_")
        outs = []
        for si, sp in enumerate(seeds):
            try:
                init = Image.open(sp).convert("RGB").resize((SIZE, SIZE))
            except Exception:
                continue
            for vi in range(5):
                strength = random.uniform(0.4, 0.6)
                gen = torch.Generator(device=DEVICE).manual_seed(SEED + cid * 100 + si * 10 + vi)
                try:
                    img = pipe(prompt=prompt, negative_prompt=NEG_PROMPT, image=init,
                               strength=strength, num_inference_steps=30, guidance_scale=7.5,
                               generator=gen).images[0]
                except Exception as e:
                    print(f"  ! diffusion fail: {e}")
                    continue
                idx = len(outs) + 1
                out_path = os.path.join(DIFF_DIR, f"{safe}_diffusion_{idx:03d}.png")
                img.save(out_path)
                outs.append(out_path)
                log_rows.append(["diffusion", cid, classes[cid], sp, f"{strength:.2f}", prompt, out_path])
                if cid not in examples and si == 0 and vi == 0:
                    examples[cid] = [sp]
                if cid in examples and len(examples[cid]) <= 3:
                    examples[cid].append(out_path)
        print(f"  {classes[cid]}: сгенерировано {len(outs)}")

    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(log_rows)

    if examples:
        n = len(examples)
        fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
        if n == 1: axes = axes[None, :]
        for r, (cid, paths) in enumerate(examples.items()):
            for c in range(4):
                if c < len(paths):
                    axes[r, c].imshow(Image.open(paths[c]).convert("RGB").resize((SIZE, SIZE)))
                    axes[r, c].set_title(("seed " + classes[cid]) if c == 0 else f"diffusion {c}")
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "diffusion_examples.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Diffusion: всего {len(log_rows)-1} изображений; лог: {LOG_CSV}")


if __name__ == "__main__":
    main()
