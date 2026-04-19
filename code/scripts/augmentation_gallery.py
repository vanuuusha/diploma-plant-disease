"""
Сохраняет примеры КАЖДОГО типа аугментации по отдельности.
Для одного исходного изображения генерируется N результатов на каждую трансформацию,
плюс комбинированный пайплайн (как в task_04 и в агрессивной версии task_05).

Используется в тексте дипломной работы, чтобы наглядно показать эффект каждой аугментации.
"""

import os
import shutil
import random
import cv2
import numpy as np
import albumentations as A
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml

cv2.setNumThreads(0)

DATASET = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset"
OUT = "/home/vanusha/diplom/diploma-plant-disease/code/results/augmentation_gallery"
SEED = 42
N_SAMPLES = 3            # сколько исходных изображений показать
N_VARIANTS_PER_AUG = 3   # сколько вариантов каждой аугментации
SIZE = 640

random.seed(SEED)
np.random.seed(SEED)


def load_yolo_labels(lbl_path):
    bboxes, labels = [], []
    if not os.path.exists(lbl_path):
        return bboxes, labels
    with open(lbl_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:
                continue
            labels.append(int(float(p[0])))
            bboxes.append(list(map(float, p[1:])))
    return bboxes, labels


# Каждая трансформация — отдельный Compose с p=1.0 для гарантированного применения.
# Это позволяет визуализировать эффект изолированно.
AUGS = {
    "01_HorizontalFlip":          A.HorizontalFlip(p=1.0),
    "02_VerticalFlip":            A.VerticalFlip(p=1.0),
    "03_RandomRotate90":          A.RandomRotate90(p=1.0),
    "04_Affine":                  A.Affine(translate_percent=0.1, scale=(0.85, 1.15), rotate=(-25, 25), p=1.0),
    "05_RandomResizedCrop":       A.RandomResizedCrop(size=(SIZE, SIZE), scale=(0.5, 0.95), p=1.0),
    "06_RandomBrightnessContrast":A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
    "07_HueSaturationValue":      A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=1.0),
    "08_GaussianBlur":            A.GaussianBlur(blur_limit=(5, 9), p=1.0),
    "09_GaussNoise":              A.GaussNoise(std_range=(0.1, 0.2), p=1.0),
    "10_CLAHE":                   A.CLAHE(clip_limit=4.0, p=1.0),
    "11_RandomShadow":            A.RandomShadow(p=1.0, num_shadows_limit=(2, 4)),
    "12_RandomFog":               A.RandomFog(p=1.0),
    "13_ImageCompression":        A.ImageCompression(quality_range=(20, 50), p=1.0),
    "14_ElasticTransform":        A.ElasticTransform(alpha=50, sigma=6, p=1.0),
    "15_GridDistortion":          A.GridDistortion(num_steps=8, distort_limit=0.4, p=1.0),
    "16_RandomToneCurve":         A.RandomToneCurve(scale=0.3, p=1.0),
    "17_Sharpen":                 A.Sharpen(p=1.0),
    "18_ChannelShuffle":          A.ChannelShuffle(p=1.0),
    "19_RGBShift":                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
    "20_MotionBlur":              A.MotionBlur(blur_limit=(7, 13), p=1.0),
}

# Полные пайплайны (как в реальном обучении)
PIPELINES = {
    "PIPELINE_classic": A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.RandomRotate90(p=0.3),
        A.Affine(translate_percent=0.1, scale=(0.85, 1.15), rotate=(-20, 20), p=0.5),
        A.RandomResizedCrop(size=(SIZE, SIZE), scale=(0.7, 1.0), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3), A.GaussNoise(std_range=(0.04, 0.2), p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3), A.RandomShadow(p=0.2), A.RandomFog(p=0.15),
        A.ImageCompression(quality_range=(70, 100), p=0.2),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.1, clip=True)),
    "PIPELINE_aggressive": A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.4), A.RandomRotate90(p=0.4),
        A.Affine(translate_percent=0.15, scale=(0.8, 1.2), rotate=(-30, 30), p=0.7),
        A.RandomResizedCrop(size=(SIZE, SIZE), scale=(0.6, 1.0), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.5),
        A.ElasticTransform(alpha=30, sigma=5, p=0.3), A.GridDistortion(p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2), A.GaussNoise(std_range=(0.04, 0.2), p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3), A.RandomShadow(p=0.2),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.1, clip=True)),
}


def resize(img, max_side=SIZE):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    k = max_side / s
    return cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)


def draw_bboxes(img, bboxes, color=(255, 0, 0)):
    h, w = img.shape[:2]
    out = img.copy()
    for xc, yc, bw, bh in bboxes:
        x1 = int((xc - bw / 2) * w); y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w); y2 = int((yc + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
    return out


def main():
    AUG_TITLES = {
        "01_HorizontalFlip": "Горизонтальное отражение",
        "02_VerticalFlip": "Вертикальное отражение",
        "03_RandomRotate90": "Поворот на 90°",
        "04_Affine": "Аффинное преобразование",
        "05_RandomResizedCrop": "Случайная обрезка",
        "06_RandomBrightnessContrast": "Яркость и контраст",
        "07_HueSaturationValue": "Оттенок и насыщенность",
        "08_GaussianBlur": "Гауссово размытие",
        "09_GaussNoise": "Гауссов шум",
        "10_CLAHE": "Адаптивное выравнивание (CLAHE)",
        "11_RandomShadow": "Случайная тень",
        "12_RandomFog": "Туман",
        "13_ImageCompression": "JPEG-сжатие",
        "14_ElasticTransform": "Эластичная деформация",
        "15_GridDistortion": "Деформация сетки",
        "16_RandomToneCurve": "Тоновая кривая",
        "17_Sharpen": "Повышение резкости",
        "18_ChannelShuffle": "Перестановка каналов",
        "19_RGBShift": "Сдвиг RGB",
        "20_MotionBlur": "Размытие движением",
    }

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)

    with open(os.path.join(DATASET, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]

    # Выбор N репрезентативных изображений (из разных классов)
    train_img = os.path.join(DATASET, "train", "images")
    train_lbl = os.path.join(DATASET, "train", "labels")
    files = sorted(os.listdir(train_img))
    random.Random(SEED).shuffle(files)
    chosen = []
    seen_classes = set()
    for fn in files:
        stem, _ = os.path.splitext(fn)
        bb, lab = load_yolo_labels(os.path.join(train_lbl, stem + ".txt"))
        if not lab:
            continue
        dom = max(set(lab), key=lab.count)
        if dom in seen_classes:
            continue
        seen_classes.add(dom)
        chosen.append((os.path.join(train_img, fn), bb, lab, dom))
        if len(chosen) >= N_SAMPLES:
            break

    print(f"Выбрано {len(chosen)} исходных изображений (классы: {[classes[c] for _, _, _, c in chosen]})")

    # ---- Изолированные аугментации: для каждой — png с (1 ориг + 3 варианта) × N_SAMPLES строк ----
    for aug_name, aug_t in AUGS.items():
        try:
            compose = A.Compose([aug_t], bbox_params=A.BboxParams(
                format="yolo", label_fields=["class_labels"], min_visibility=0.1, clip=True
            ))
        except Exception as e:
            print(f"  ! {aug_name}: {e}")
            continue
        n_rows = N_SAMPLES
        n_cols = N_VARIANTS_PER_AUG + 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes[None, :]
        for r, (img_path, bboxes, labels, dom) in enumerate(chosen):
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = resize(img)
            axes[r, 0].imshow(draw_bboxes(img, bboxes))
            axes[r, 0].set_title(f"оригинал ({classes[dom]})", fontsize=10)
            axes[r, 0].axis("off")
            for v in range(N_VARIANTS_PER_AUG):
                try:
                    out = compose(image=img, bboxes=bboxes, class_labels=labels)
                    aug_img = out["image"]
                    aug_bb = list(out["bboxes"])
                    axes[r, v + 1].imshow(draw_bboxes(aug_img, aug_bb))
                except Exception as e:
                    axes[r, v + 1].text(0.5, 0.5, f"err\n{e}", ha="center", va="center", transform=axes[r, v + 1].transAxes)
                axes[r, v + 1].set_title(f"вариант {v + 1}", fontsize=10)
                axes[r, v + 1].axis("off")
        fig.suptitle(AUG_TITLES.get(aug_name, aug_name), fontsize=14, y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, aug_name + ".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {aug_name}.png")

    # ---- Полные пайплайны ----
    for pname, pipe in PIPELINES.items():
        n_rows = N_SAMPLES
        n_cols = 5
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes[None, :]
        for r, (img_path, bboxes, labels, dom) in enumerate(chosen):
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = resize(img)
            axes[r, 0].imshow(draw_bboxes(img, bboxes))
            axes[r, 0].set_title(f"оригинал ({classes[dom]})", fontsize=10)
            axes[r, 0].axis("off")
            for v in range(n_cols - 1):
                try:
                    out = pipe(image=img, bboxes=bboxes, class_labels=labels)
                    axes[r, v + 1].imshow(draw_bboxes(out["image"], list(out["bboxes"])))
                except Exception as e:
                    axes[r, v + 1].text(0.5, 0.5, f"err\n{e}", ha="center", va="center", transform=axes[r, v + 1].transAxes)
                axes[r, v + 1].set_title(f"вариант {v + 1}", fontsize=10)
                axes[r, v + 1].axis("off")
        pipe_titles = {"PIPELINE_classic": "Классический пайплайн аугментации", "PIPELINE_aggressive": "Агрессивный пайплайн аугментации"}
        fig.suptitle(pipe_titles.get(pname, pname), fontsize=14, y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, pname + ".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {pname}.png")

    # ---- Сетка-обзор: один грид со всеми аугментациями (по 1 варианту), для одного изображения ----
    img_path, bboxes, labels, dom = chosen[0]
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = resize(img)
    items = [("оригинал", img, bboxes)]
    for name, aug_t in AUGS.items():
        try:
            compose = A.Compose([aug_t], bbox_params=A.BboxParams(
                format="yolo", label_fields=["class_labels"], min_visibility=0.1, clip=True
            ))
            out = compose(image=img, bboxes=bboxes, class_labels=labels)
            items.append((name, out["image"], list(out["bboxes"])))
        except Exception:
            items.append((name, img, bboxes))
    n = len(items)
    cols = 7
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)
    for i, (name, im, bb) in enumerate(items):
        axes[i].imshow(draw_bboxes(im, bb))
        axes[i].set_title(AUG_TITLES.get(name, name.replace("_", " ")), fontsize=9)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Все аугментации (исходный класс: {classes[dom]})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ALL_AUGS_OVERVIEW.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved ALL_AUGS_OVERVIEW.png")

    print(f"\nГотово. Файлов в {OUT}: {len(os.listdir(OUT))}")


if __name__ == "__main__":
    main()
