"""
Compute FID and KID between real field images (dataset_final/train) and three
synthetic sets (Stable Diffusion img2img, NST v1, NST v2).

Uses clean-fid (https://github.com/GaParmar/clean-fid).

Outputs:
  - code/results/task_23/fid_kid_results.csv
  - code/results/task_23/fid_kid_comparison.png
  - code/results/task_23/fid_kid_per_class.csv (optional per-class, if feasible)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path("/home/vanusha/diplom/diploma-plant-disease")
DEFAULT_REAL = ROOT / "code/data/dataset_final/train/images"
DEFAULT_LABELS = ROOT / "code/data/dataset_final/train/labels"
DEFAULT_SYNTH = {
    "diffusion": ROOT / "code/results/task_06/diffusion_v2",
    "nst_v1": ROOT / "code/results/task_06/nst",
    "nst_v2": ROOT / "code/results/task_06/nst_v2",
}
DEFAULT_OUT = ROOT / "code/results/task_23"

# Synthesized rare classes (cls_id -> display/file names)
RARE_CLASSES = {
    5: {"name": "root_rot",      "ru": "Корневая_гниль"},
    6: {"name": "septoria",      "ru": "Септориоз"},
    7: {"name": "nitrogen_def",  "ru": "Недостаток_N"},
    8: {"name": "frost_damage",  "ru": "Повреждение_заморозками"},
}


def build_real_dir(src: Path, dst: Path, include_aug: bool = False) -> int:
    """Materialize a flat directory of real images via symlinks."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if not include_aug and "_aug" in p.stem:
            continue
        link = dst / p.name
        if not link.exists():
            os.symlink(p.resolve(), link)
        n += 1
    return n


def dominant_class_for_label(label_path: Path) -> int | None:
    """Return the class id with the largest bbox area on the image."""
    try:
        lines = label_path.read_text().strip().splitlines()
    except FileNotFoundError:
        return None
    area_by_cls: Dict[int, float] = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        w = float(parts[3])
        h = float(parts[4])
        area_by_cls[cls] = area_by_cls.get(cls, 0.0) + w * h
    if not area_by_cls:
        return None
    return max(area_by_cls, key=area_by_cls.get)


def build_real_per_class_dirs(src_imgs: Path, src_labels: Path, dst_root: Path,
                              classes: Dict[int, Dict[str, str]],
                              include_aug: bool = False) -> Dict[int, int]:
    """Split real images by dominant class into dst_root/{name}/."""
    counts: Dict[int, int] = {c: 0 for c in classes}
    for c in classes:
        (dst_root / classes[c]["name"]).mkdir(parents=True, exist_ok=True)
    for p in sorted(src_imgs.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if not include_aug and "_aug" in p.stem:
            continue
        lbl = src_labels / (p.stem + ".txt")
        c = dominant_class_for_label(lbl)
        if c is None or c not in classes:
            continue
        link = dst_root / classes[c]["name"] / p.name
        if not link.exists():
            os.symlink(p.resolve(), link)
        counts[c] += 1
    return counts


def parse_synth_class(filename: str, classes: Dict[int, Dict[str, str]]) -> int | None:
    """Infer class id from synthetic file name like 'Корневая_гниль_001.png'."""
    stem = Path(filename).stem
    for cid, info in classes.items():
        if stem.startswith(info["ru"]):
            return cid
    return None


def build_synth_per_class_dirs(src: Path, dst_root: Path,
                               classes: Dict[int, Dict[str, str]]) -> Dict[int, int]:
    counts: Dict[int, int] = {c: 0 for c in classes}
    for c in classes:
        (dst_root / classes[c]["name"]).mkdir(parents=True, exist_ok=True)
    for p in sorted(src.iterdir()):
        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        cid = parse_synth_class(p.name, classes)
        if cid is None:
            continue
        link = dst_root / classes[cid]["name"] / p.name
        if not link.exists():
            os.symlink(p.resolve(), link)
        counts[cid] += 1
    return counts


def count_images(path: Path) -> int:
    return sum(
        1 for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def run_fid_kid(real: Path, synth: Path) -> Tuple[float, float]:
    from cleanfid import fid
    fid_score = fid.compute_fid(str(real), str(synth), mode="clean", num_workers=4)
    kid_score = fid.compute_kid(str(real), str(synth), mode="clean", num_workers=4)
    return float(fid_score), float(kid_score)


def make_plot(results: List[Dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [r["method"] for r in results]
    fids = [r["fid"] for r in results]
    kids_scaled = [r["kid"] * 1000.0 for r in results]

    x = np.arange(len(methods))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - width / 2, fids, width, label="FID ↓",
                 color="#2c7fb8", edgecolor="black", linewidth=0.6)
    b2 = ax2.bar(x + width / 2, kids_scaled, width, label="KID × 10³ ↓",
                 color="#d95f0e", edgecolor="black", linewidth=0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("_", " ") for m in methods])
    ax1.set_ylabel("FID")
    ax2.set_ylabel("KID × 10³")
    ax1.set_title("FID / KID relative to real train distribution (lower is closer)")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    for rect, val in zip(b1, fids):
        ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for rect, val in zip(b2, kids_scaled):
        ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    lines = [b1, b2]
    ax1.legend(lines, ["FID ↓", "KID × 10³ ↓"], loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, default=DEFAULT_REAL)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--diffusion", type=Path, default=DEFAULT_SYNTH["diffusion"])
    ap.add_argument("--nst-v1", type=Path, default=DEFAULT_SYNTH["nst_v1"])
    ap.add_argument("--nst-v2", type=Path, default=DEFAULT_SYNTH["nst_v2"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--include-aug", action="store_true",
                    help="Include classical _aug augmented copies in real set")
    ap.add_argument("--per-class", action="store_true",
                    help="Compute per-class FID/KID for rare classes")
    ap.add_argument("--min-class-synth", type=int, default=30,
                    help="Skip per-class metric if synth class has fewer images")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- overall ----
    work = Path(tempfile.mkdtemp(prefix="fidkid_", dir="/tmp"))
    real_flat = work / "real"
    print(f"[prep] Materializing real dir (include_aug={args.include_aug}) ...")
    n_real = build_real_dir(args.real, real_flat, include_aug=args.include_aug)
    print(f"[prep] real_n = {n_real}")

    synth_dirs = {
        "diffusion": args.diffusion,
        "nst_v1": args.nst_v1,
        "nst_v2": args.nst_v2,
    }
    pretty = {
        "diffusion": "Stable Diffusion img2img",
        "nst_v1": "NST v1 (Adam, 384)",
        "nst_v2": "NST v2 (LBFGS, YUV, 512)",
    }
    synth_counts = {k: count_images(v) for k, v in synth_dirs.items()}
    print(f"[prep] synth counts: {synth_counts}")

    results: List[Dict] = []
    for key, path in synth_dirs.items():
        if not path.exists() or synth_counts[key] == 0:
            print(f"[warn] skip {key}: dir missing or empty ({path})")
            continue
        print(f"[fid/kid] {key} ({synth_counts[key]} imgs) vs real ({n_real}) ...")
        fid_score, kid_score = run_fid_kid(real_flat, path)
        print(f"[fid/kid] {key}: FID={fid_score:.3f}  KID={kid_score:.6f}")
        results.append({
            "method": key,
            "pretty": pretty[key],
            "n_real": n_real,
            "n_synth": synth_counts[key],
            "fid": fid_score,
            "kid": kid_score,
        })

    # ---- per-class (optional) ----
    per_class: List[Dict] = []
    if args.per_class and results:
        print("[per-class] building per-class real dirs ...")
        real_cls_root = work / "real_per_class"
        real_cls_counts = build_real_per_class_dirs(
            args.real, args.labels, real_cls_root, RARE_CLASSES,
            include_aug=args.include_aug,
        )
        print(f"[per-class] real counts per class: "
              f"{ {RARE_CLASSES[c]['name']: n for c, n in real_cls_counts.items()} }")

        synth_cls_roots: Dict[str, Path] = {}
        for key, path in synth_dirs.items():
            if not path.exists():
                continue
            droot = work / f"{key}_per_class"
            cnts = build_synth_per_class_dirs(path, droot, RARE_CLASSES)
            synth_cls_roots[key] = droot
            print(f"[per-class] {key} counts: "
                  f"{ {RARE_CLASSES[c]['name']: n for c, n in cnts.items()} }")

        for cid, info in RARE_CLASSES.items():
            cname = info["name"]
            real_sub = real_cls_root / cname
            if not real_sub.exists() or count_images(real_sub) < 50:
                print(f"[per-class] skip class {cname}: too few real imgs")
                continue
            for key, droot in synth_cls_roots.items():
                synth_sub = droot / cname
                if not synth_sub.exists():
                    continue
                n_s = count_images(synth_sub)
                if n_s < args.min_class_synth:
                    # For very small NST subsets: KID still viable but FID unreliable.
                    print(f"[per-class] {key}/{cname}: n_synth={n_s} < min, "
                          f"skipping FID (too few samples)")
                    continue
                try:
                    fid_s, kid_s = run_fid_kid(real_sub, synth_sub)
                except Exception as e:  # clean-fid may fail on very small sets
                    print(f"[per-class] {key}/{cname} failed: {e}")
                    continue
                print(f"[per-class] {key}/{cname}: FID={fid_s:.3f}  KID={kid_s:.6f}  "
                      f"(n_real={count_images(real_sub)}, n_synth={n_s})")
                per_class.append({
                    "class": cname,
                    "class_id": cid,
                    "method": key,
                    "n_real": count_images(real_sub),
                    "n_synth": n_s,
                    "fid": fid_s,
                    "kid": kid_s,
                })

    # ---- save results ----
    csv_path = args.out / "fid_kid_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "pretty", "n_real", "n_synth", "fid", "kid"])
        for r in results:
            w.writerow([r["method"], r["pretty"], r["n_real"], r["n_synth"],
                        f"{r['fid']:.6f}", f"{r['kid']:.8f}"])
    print(f"[save] {csv_path}")

    if per_class:
        pc_path = args.out / "fid_kid_per_class.csv"
        with pc_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "class_id", "method", "n_real", "n_synth", "fid", "kid"])
            for r in per_class:
                w.writerow([r["class"], r["class_id"], r["method"],
                            r["n_real"], r["n_synth"],
                            f"{r['fid']:.6f}", f"{r['kid']:.8f}"])
        print(f"[save] {pc_path}")

    if results:
        plot_path = args.out / "fid_kid_comparison.png"
        make_plot(results, plot_path)
        print(f"[save] {plot_path}")

    # ---- JSON summary ----
    with (args.out / "fid_kid_summary.json").open("w") as f:
        json.dump({"overall": results, "per_class": per_class,
                   "include_aug": args.include_aug, "n_real": n_real}, f,
                  ensure_ascii=False, indent=2)

    # cleanup
    try:
        shutil.rmtree(work)
    except Exception:
        pass

    print("[done]")


if __name__ == "__main__":
    main()
