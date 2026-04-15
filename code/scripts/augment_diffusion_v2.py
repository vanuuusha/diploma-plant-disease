"""
Генерация Diffusion img2img для закрытия 30% дефицита каждого редкого класса
(после того как oversampling в task_05 закрыл 70%).

Количество на класс рассчитывается от дефицита:
  class_target = 80% медианы
  gap         = class_target - pre_balance_count
  remaining   = 0.3 * gap  (аннотации, которые должен дать Diffusion)
  num_images  = ceil(remaining / avg_annotations_per_image_of_class)

strength=0.40 фиксирован — композиция сохраняется, метки seed валидны.
"""

import os
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml
from PIL import Image
import numpy as np

DATASET_BALANCED = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset_balanced"
DATASET_AUGMENTED = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset_augmented"
DATASET = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset"
OUT_DIR = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06/diffusion_v2"
MANIFEST_PATH = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06/synth_manifest.json"
SIZE = 512
STRENGTH = 0.40
STEPS_INFER = 30
GUIDANCE = 7.5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_FRACTION_OF_MEDIAN = 0.8
DIFFUSION_FRACTION_OF_GAP = 0.3  # Diffusion закрывает 30% дефицита

PROMPTS = {
    5: "macro photograph of wheat plant root crown with brown root rot disease symptoms, fungal infection, agricultural close-up, natural lighting, high detail",
    6: "wheat leaf with septoria leaf spot disease, brown spots and yellow halos, agricultural field photography, natural lighting, high detail",
    7: "wheat leaves showing nitrogen deficiency, yellowing of older leaves, pale green tips, agricultural field photo, natural lighting",
    8: "wheat leaves with frost damage, white patches and necrotic edges, early winter morning agricultural field, high detail",
}
NEG = "drawing, painting, blurry, low quality, watermark, text, logo, frame, illustration, cartoon"

random.seed(SEED)
torch.manual_seed(SEED)


def count_annotations_train(dataset_root):
    """Считает аннотации по классам в train."""
    cc = Counter()
    lbl_dir = os.path.join(dataset_root, "train", "labels")
    for fn in os.listdir(lbl_dir):
        with open(os.path.join(lbl_dir, fn)) as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 5:
                    cc[int(float(p[0]))] += 1
    return cc


def list_seeds_with_labels_train(dataset_root):
    train_img = os.path.join(dataset_root, "train", "images")
    train_lbl = os.path.join(dataset_root, "train", "labels")
    result = defaultdict(list)
    for fn in os.listdir(train_img):
        stem, _ = os.path.splitext(fn)
        lbl_path = os.path.join(train_lbl, stem + ".txt")
        if not os.path.exists(lbl_path): continue
        # отфильтровать: брать ТОЛЬКО оригинальные (не _aug, не _bal) снимки как seeds,
        # чтобы не поощрять цепочку «аугментация → синтетика → ещё синтетика».
        if "_aug" in stem or "_bal" in stem:
            continue
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
    with open(os.path.join(DATASET_BALANCED, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]

    # Планирование количества по классам
    aug_counts = count_annotations_train(DATASET_AUGMENTED)  # состояние до балансировки
    bal_counts = count_annotations_train(DATASET_BALANCED)   # после oversampling (70% gap)
    median_n = float(np.median([v for v in aug_counts.values() if v > 0]))
    final_target = int(median_n * TARGET_FRACTION_OF_MEDIAN)
    print(f"Медиана (до balance): {median_n:.0f}, финальная цель: {final_target}")

    # avg аннотаций на seed-изображение для каждого класса (из исходного dataset)
    per_class_bbox = Counter()
    per_class_img = Counter()
    for cid in range(len(classes)):
        pass
    for fn in os.listdir(os.path.join(DATASET, "train", "labels")):
        with open(os.path.join(DATASET, "train", "labels", fn)) as f:
            cls_set = set()
            bbox_cls = Counter()
            for line in f:
                p = line.strip().split()
                if len(p) == 5:
                    c = int(float(p[0]))
                    cls_set.add(c); bbox_cls[c] += 1
            for c in cls_set:
                per_class_img[c] += 1
                per_class_bbox[c] += bbox_cls[c]
    avg_ann_per_img = {c: (per_class_bbox[c] / per_class_img[c]) if per_class_img[c] else 1.0
                       for c in range(len(classes))}

    rare_ids = [5, 6, 7, 8]
    print(f"{'class':35s} aug  bal  final  gap-total  diff-need  avg/img  num-to-gen")
    gen_plan = {}
    for cid in rare_ids:
        gap_total = final_target - aug_counts[cid]
        diff_need = int(round(DIFFUSION_FRACTION_OF_GAP * gap_total))
        avg = avg_ann_per_img.get(cid, 1.5)
        n_images = max(0, math.ceil(diff_need / avg))
        gen_plan[cid] = (n_images, diff_need)
        print(f"  {classes[cid]:33s} {aug_counts[cid]:4d} {bal_counts[cid]:4d} {final_target:5d} {gap_total:8d}  {diff_need:8d}  {avg:6.2f}  {n_images:6d}")

    from diffusers import StableDiffusionImg2ImgPipeline
    print("\nЗагрузка SD-1.5…")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    seeds = list_seeds_with_labels_train(DATASET_BALANCED)
    manifest = []

    for cid in rare_ids:
        n_target, _ = gen_plan[cid]
        if n_target <= 0: continue
        pool = list(seeds[cid])
        random.Random(SEED + cid).shuffle(pool)
        if not pool:
            print(f"!! нет seed для {classes[cid]}")
            continue
        prompt = PROMPTS[cid]
        safe = classes[cid].replace(" ", "_").replace("/", "_")
        for i in range(n_target):
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
                "out_image_path": out_path, "strength": STRENGTH, "prompt": prompt,
            })
            if (i + 1) % 20 == 0 or i + 1 == n_target:
                print(f"  {classes[cid]}: {i + 1}/{n_target}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nDiffusion v2: {len(manifest)} изображений; manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
