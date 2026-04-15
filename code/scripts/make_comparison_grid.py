"""
Собирает сравнительный грид из уже сгенерированных изображений
(без повторной генерации): Seed / Diffusion / NST v1 / NST v2.
Используется в task_06/RESULT.md.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DATASET = "/home/vanusha/diplom/diploma-plant-disease/code/data/dataset/train/images"
DIFF = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06/diffusion"
NST1 = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06/nst"
NST2 = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06/nst_v2"
OUT = "/home/vanusha/diplom/diploma-plant-disease/code/results/task_06/MASTER_COMPARISON.png"

# seed = content (реальный снимок из датасета)
# Источники для каждого класса получены из *_log.csv:
CASES = [
    ("Корневая гниль",          "14305.jpg", "Корневая_гниль_diffusion_001.png",          "Корневая_гниль_nst_001.png",          "Корневая_гниль_nst_001.png"),
    ("Септориоз",                "10668.jpg", "Септориоз_diffusion_001.png",                "Септориоз_nst_001.png",                "Септориоз_nst_001.png"),
    ("Недостаток N",             "12421.jpg", "Недостаток_N_diffusion_001.png",             "Недостаток_N_nst_001.png",             "Недостаток_N_nst_001.png"),
    ("Повреждение заморозками",  "13557.jpg", "Повреждение_заморозками_diffusion_001.png",  "Повреждение_заморозками_nst_001.png",  "Повреждение_заморозками_nst_001.png"),
]

SIZE = 384


def imload(path):
    try:
        return Image.open(path).convert("RGB").resize((SIZE, SIZE))
    except Exception:
        return Image.new("RGB", (SIZE, SIZE), (200, 200, 200))


def main():
    n_rows = len(CASES); n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    titles = ["Исходное (реальный снимок)", "Stable Diffusion img2img", "NST v1 (Adam, 384px)", "NST v2 (LBFGS, 512px, color-preserve)"]
    for r, (cname, seed, d, n1, n2) in enumerate(CASES):
        imgs = [
            imload(os.path.join(DATASET, seed)),
            imload(os.path.join(DIFF, d)),
            imload(os.path.join(NST1, n1)),
            imload(os.path.join(NST2, n2)),
        ]
        for c, im in enumerate(imgs):
            axes[r, c].imshow(im)
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=11)
            axes[r, c].axis("off")
        axes[r, 0].text(-0.07, 0.5, cname, transform=axes[r, 0].transAxes,
                        rotation=90, ha="right", va="center", fontsize=13, fontweight="bold")
    fig.suptitle("Сравнение методов генеративной аугментации (по одному случаю на класс)", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
