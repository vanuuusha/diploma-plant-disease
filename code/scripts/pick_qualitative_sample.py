#!/usr/bin/env python
"""
Pick 10 test images for qualitative comparison across chapter 3 detectors.

Selection criteria (stratified):
    - 1 image per class where the class is dominant bbox → 9 images
    - +1 image with 2+ distinct classes present (multi-class scene)

Deterministic seed=42. Output: code/docs/chapter3_qualitative_sample.txt
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/vanusha/diplom/diploma-plant-disease")
TEST_IMG = ROOT / "code/data/dataset/test/images"
TEST_LBL = ROOT / "code/data/dataset/test/labels"
OUT = ROOT / "code/docs/chapter3_qualitative_sample.txt"

CLASS_NAMES = [
    "Недостаток P2O5",
    "Листовая (бурая) ржавчина",
    "Мучнистая роса",
    "Пиренофороз",
    "Фузариоз",
    "Корневая гниль",
    "Септориоз",
    "Недостаток N",
    "Повреждение заморозками",
]


def read_labels(path: Path):
    """Return list[(cls_id, x, y, w, h)]."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        out.append((cls, x, y, w, h))
    return out


def main():
    random.seed(42)
    files = sorted(p.name for p in TEST_IMG.iterdir() if p.suffix.lower() == ".jpg")
    stats = {}
    for f in files:
        lbl = TEST_LBL / (Path(f).stem + ".txt")
        anns = read_labels(lbl)
        if not anns:
            continue
        cls_list = [a[0] for a in anns]
        dominant = Counter(cls_list).most_common(1)[0][0]
        stats[f] = {
            "n": len(anns),
            "classes": set(cls_list),
            "dominant": dominant,
        }

    by_class = defaultdict(list)
    for f, s in stats.items():
        by_class[s["dominant"]].append(f)

    chosen = []
    for c in range(9):
        if not by_class[c]:
            continue
        candidates = sorted(by_class[c], key=lambda f: -stats[f]["n"])
        top = candidates[: min(10, len(candidates))]
        chosen.append(random.choice(top))

    multi = [f for f, s in stats.items() if len(s["classes"]) >= 2 and f not in chosen]
    if multi:
        multi.sort(key=lambda f: -len(stats[f]["classes"]))
        chosen.append(multi[0])

    chosen = chosen[:10]

    header = [
        "# Chapter 3 — fixed qualitative sample for detectors comparison",
        "# Seed: 42. Selected stratified by dominant class + 1 multi-class scene.",
        "# Each line: filename (relative to code/data/dataset/test/images/).",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(header + chosen) + "\n", encoding="utf-8")

    print(f"Wrote {OUT}")
    for f in chosen:
        s = stats[f]
        cls_str = ", ".join(CLASS_NAMES[c] for c in sorted(s["classes"]))
        print(f"  {f}  n={s['n']}  classes=[{cls_str}]")


if __name__ == "__main__":
    main()
