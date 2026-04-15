"""
Финальная генерация Stable Diffusion img2img для добавления в dataset_final.
Количество на каждый класс подобрано по дефициту аннотаций относительно медианы
(с учётом того, что в среднем 1.8 аннотации на изображение в редких классах).

strength=0.4 фиксирован — композиция сохраняется, метки seed-изображения валидны.
На выходе: PNG + manifest.json для build_dataset_final.py.
"""

import os
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from PIL import Image

DATASET = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset"
OUT_DIR = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06/diffusion_final"
MANIFEST_PATH = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06/synth_manifest.json"
SIZE = 512
STRENGTH = 0.4
STEPS_INFER = 30
GUIDANCE = 7.5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Сколько изображений на класс — обосновано в RESULT task_06
# Медиана после balance: 3196. Текущие 4 редких класса: 2556. Дефицит: 640 ann/класс.
# Avg ~1.8 bbox/img в редких классах → теоретический минимум для закрытия ≈ 355 img/класс.
# Берём 50 — демонстрационный объём, покрывает ~28% дефицита, для диплома достаточно
# показать методологию; при тяжёлом обучении увеличить.
PER_CLASS = 50

PROMPTS = {
    5: "macro photograph of wheat plant root crown with brown root rot disease symptoms, fungal infection, agricultural close-up, natural lighting, high detail",
    6: "wheat leaf with septoria leaf spot disease, brown spots and yellow halos, agricultural field photography, natural lighting, high detail",
    7: "wheat leaves showing nitrogen deficiency, yellowing of older leaves, pale green tips, agricultural field photo, natural lighting",
    8: "wheat leaves with frost damage, white patches and necrotic edges, early winter morning agricultural field, high detail",
}
NEG = "drawing, painting, blurry, low quality, watermark, text, logo, frame, illustration, cartoon"

random.seed(SEED)
torch.manual_seed(SEED)


def list_seeds_with_labels(classes):
    """Возвращает cid → [(image_path, label_path), ...] для train."""
    train_img = os.path.join(DATASET, "train", "images")
    train_lbl = os.path.join(DATASET, "train", "labels")
    result = defaultdict(list)
    for fn in os.listdir(train_img):
        stem, _ = os.path.splitext(fn)
        lbl_path = os.path.join(train_lbl, stem + ".txt")
        if not os.path.exists(lbl_path): continue
        with open(lbl_path) as f:
            cls_set = set()
            for line in f:
                p = line.strip().split()
                if len(p) == 5:
                    cls_set.add(int(float(p[0])))
        for c in cls_set:
            result[c].append((os.path.join(train_img, fn), lbl_path))
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(DATASET, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]

    from diffusers import StableDiffusionImg2ImgPipeline
    print("Загрузка SD-1.5…")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    seeds = list_seeds_with_labels(classes)
    rare_ids = [5, 6, 7, 8]
    manifest = []

    for cid in rare_ids:
        if cid not in seeds or not seeds[cid]: continue
        pool = list(seeds[cid])
        random.Random(SEED + cid).shuffle(pool)
        prompt = PROMPTS[cid]
        safe = classes[cid].replace(" ", "_").replace("/", "_")
        for i in range(PER_CLASS):
            img_path, lbl_path = pool[i % len(pool)]
            try:
                init = Image.open(img_path).convert("RGB").resize((SIZE, SIZE))
            except Exception:
                continue
            gen = torch.Generator(device=DEVICE).manual_seed(SEED + cid * 10000 + i)
            out_img = pipe(prompt=prompt, negative_prompt=NEG, image=init,
                           strength=STRENGTH, num_inference_steps=STEPS_INFER,
                           guidance_scale=GUIDANCE, generator=gen).images[0]
            out_path = os.path.join(OUT_DIR, f"{safe}_{i + 1:03d}.png")
            out_img.save(out_path)
            manifest.append({
                "cls_id": cid, "class_name": classes[cid],
                "seed_image_path": img_path, "seed_label_path": lbl_path,
                "out_image_path": out_path, "strength": STRENGTH,
            })
            if (i + 1) % 10 == 0:
                print(f"  {classes[cid]}: {i + 1}/{PER_CLASS}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nDiffusion-final: {len(manifest)} изображений; manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
