"""
Task 25: метрики дисбаланса на уровне изображений.

Для итогового датасета (до синтетической добавки) считаются три метрики
дисбаланса — IR, CV, H_norm — в двух представлениях:
  * на уровне bbox-ов  — каждая аннотация вносит в счётчик своего класса 1;
  * на уровне изображений — каждый кадр относится к «преобладающему» классу
    (больше всего bbox на кадре; при равенстве — по суммарной площади).

Датасет читается из YOLO-разметки (по всем трём сплитам train/val/test).
Результаты сохраняются в:
  code/results/task_25/image_level_imbalance.yaml
  code/results/task_25/image_vs_bbox_distribution.png
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path("/home/vanusha/diplom/diploma-plant-disease")
DATASET_DIR = ROOT / "code/data/dataset"
SPLITS = ("train", "val", "test")
OUT_DIR = ROOT / "code/results/task_25"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_class_names() -> list[str]:
    with open(DATASET_DIR / "data.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["names"]


def parse_label_file(path: Path) -> list[tuple[int, float]]:
    """Возвращает список (class_id, площадь_bbox) для одного YOLO .txt-файла."""
    out: list[tuple[int, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            w, h = float(parts[3]), float(parts[4])
            out.append((cls, w * h))
    return out


def dominant_class(boxes: list[tuple[int, float]]) -> int | None:
    """Преобладающий класс на кадре: по числу bbox, при равенстве — по площади."""
    if not boxes:
        return None
    counts: Counter[int] = Counter()
    area: dict[int, float] = defaultdict(float)
    for cls, a in boxes:
        counts[cls] += 1
        area[cls] += a
    top_count = max(counts.values())
    candidates = [c for c, n in counts.items() if n == top_count]
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda c: area[c])


def metrics(counts_by_class: dict[int, int], num_classes: int) -> dict[str, float]:
    """IR, CV, H_norm по вектору частот (поддерживает нули → IR=inf)."""
    vals = np.array(
        [counts_by_class.get(c, 0) for c in range(num_classes)], dtype=float
    )
    total = vals.sum()
    max_v, min_v = vals.max(), vals.min()
    ir = float("inf") if min_v == 0 else float(max_v / min_v)
    mu = float(vals.mean())
    sigma = float(vals.std(ddof=0))
    cv = float(sigma / mu) if mu > 0 else 0.0
    probs = (vals / total) if total > 0 else vals
    entropy = float(-sum(p * math.log(p) for p in probs if p > 0))
    h_norm = float(entropy / math.log(num_classes))
    return {
        "IR": ir,
        "CV": cv,
        "H_norm": h_norm,
        "min": float(min_v),
        "max": float(max_v),
        "mean": mu,
        "std": sigma,
        "total": float(total),
    }


def main() -> None:
    class_names = load_class_names()
    K = len(class_names)

    bbox_counts: Counter[int] = Counter()
    image_counts: Counter[int] = Counter()
    n_images_total = 0
    n_images_with_boxes = 0

    for split in SPLITS:
        labels_dir = DATASET_DIR / split / "labels"
        if not labels_dir.is_dir():
            continue
        for label_file in sorted(labels_dir.glob("*.txt")):
            n_images_total += 1
            boxes = parse_label_file(label_file)
            if not boxes:
                continue
            n_images_with_boxes += 1
            for cls, _ in boxes:
                bbox_counts[cls] += 1
            dom = dominant_class(boxes)
            if dom is not None:
                image_counts[dom] += 1

    bbox_metrics = metrics(dict(bbox_counts), K)
    image_metrics = metrics(dict(image_counts), K)

    print(f"Всего изображений: {n_images_total}")
    print(f"С непустой разметкой: {n_images_with_boxes}")
    print(f"Всего bbox: {int(bbox_metrics['total'])}\n")

    header = f"{'id':>3}  {'класс':<30} {'N_bbox':>8} {'N_img':>8}"
    print(header)
    print("-" * len(header))
    for cid, name in enumerate(class_names):
        print(f"{cid:>3}  {name:<30} {bbox_counts.get(cid, 0):>8} {image_counts.get(cid, 0):>8}")

    print("\nМетрики:")
    print(f"{'':14} {'bbox-level':>12} {'image-level':>12}")
    for key in ("IR", "CV", "H_norm", "min", "max", "mean", "std"):
        print(f"{key:<14} {bbox_metrics[key]:>12.4f} {image_metrics[key]:>12.4f}")

    ratio = bbox_metrics["IR"] / image_metrics["IR"] if image_metrics["IR"] > 0 else float("inf")
    print(f"\nIR_bbox / IR_img = {ratio:.2f}")

    top_img_class = max(image_counts, key=lambda c: image_counts[c])
    print(
        "Чаще всего преобладающий класс на кадре: "
        f"{class_names[top_img_class]} ({image_counts[top_img_class]} изображений)"
    )
    never_dominant = [class_names[c] for c in range(K) if image_counts.get(c, 0) == 0]
    if never_dominant:
        print(f"Ни разу не были преобладающими: {never_dominant}")

    result = {
        "dataset_dir": str(DATASET_DIR),
        "splits": list(SPLITS),
        "n_images_total": n_images_total,
        "n_images_with_boxes": n_images_with_boxes,
        "n_bboxes_total": int(bbox_metrics["total"]),
        "num_classes": K,
        "class_names": class_names,
        "bbox_counts_by_class": {class_names[c]: bbox_counts.get(c, 0) for c in range(K)},
        "image_counts_by_class": {class_names[c]: image_counts.get(c, 0) for c in range(K)},
        "bbox_metrics": bbox_metrics,
        "image_metrics": image_metrics,
        "ir_bbox_over_ir_img": ratio,
        "top_dominant_image_class": class_names[top_img_class],
        "never_dominant_classes": never_dominant,
    }
    out_yaml = OUT_DIR / "image_level_imbalance.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, allow_unicode=True, sort_keys=False)
    print(f"\nСохранено: {out_yaml}")

    # Диаграмма: столбики bbox vs images на класс, на двух осях.
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    x = np.arange(K)
    width = 0.38
    bbox_vals = [bbox_counts.get(c, 0) for c in range(K)]
    img_vals = [image_counts.get(c, 0) for c in range(K)]
    ax1.bar(x - width / 2, bbox_vals, width, color="#4C72B0", label="Число bbox")
    ax1.set_ylabel("Число bbox", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, img_vals, width, color="#DD8452", label="Число изображений")
    ax2.set_ylabel("Число изображений", color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=30, ha="right")
    ax1.set_title(
        "Распределение классов на уровне bbox и на уровне изображений\n"
        f"IR_bbox={bbox_metrics['IR']:.2f}×, IR_img={image_metrics['IR']:.2f}×; "
        f"CV_bbox={bbox_metrics['CV']:.3f}, CV_img={image_metrics['CV']:.3f}; "
        f"H_bbox={bbox_metrics['H_norm']:.3f}, H_img={image_metrics['H_norm']:.3f}"
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_png = OUT_DIR / "image_vs_bbox_distribution.png"
    fig.savefig(out_png, dpi=150)
    print(f"Сохранено: {out_png}")


if __name__ == "__main__":
    main()
