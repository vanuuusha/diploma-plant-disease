"""
Compute LPIPS-based intra-set diversity for real and synthetic image sets.

For each (method × class) and each method overall, averages LPIPS distance
between N random image pairs. Higher value = more visual diversity within the
set. Uses AlexNet backbone (the LPIPS default 'alex').

Outputs:
  - code/results/task_24/lpips_results.csv         (overall per method)
  - code/results/task_24/lpips_per_class.csv       (per method × class)
  - code/results/task_24/lpips_diversity.png
  - code/results/task_24/lpips_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

ROOT = Path("/home/vanusha/diplom/diploma-plant-disease")
DEFAULT_REAL_IMGS = ROOT / "code/data/dataset_final/train/images"
DEFAULT_REAL_LBL = ROOT / "code/data/dataset_final/train/labels"
DEFAULT_SYNTH = {
    "diffusion": ROOT / "code/results/task_06/diffusion_v2",
    "nst_v1": ROOT / "code/results/task_06/nst",
    "nst_v2": ROOT / "code/results/task_06/nst_v2",
}
DEFAULT_OUT = ROOT / "code/results/task_24"

RARE_CLASSES = {
    5: {"name": "root_rot",      "ru": "Корневая_гниль",         "ru_space": "Корневая гниль"},
    6: {"name": "septoria",      "ru": "Септориоз",              "ru_space": "Септориоз"},
    7: {"name": "nitrogen_def",  "ru": "Недостаток_N",           "ru_space": "Недостаток N"},
    8: {"name": "frost_damage",  "ru": "Повреждение_заморозками", "ru_space": "Повреждение заморозками"},
}

RESOLUTION = 256
RNG_SEED = 42


def dominant_class(label_path: Path) -> int | None:
    try:
        lines = label_path.read_text().strip().splitlines()
    except FileNotFoundError:
        return None
    area: Dict[int, float] = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        w = float(parts[3]); h = float(parts[4])
        area[c] = area.get(c, 0.0) + w * h
    if not area:
        return None
    return max(area, key=area.get)


def list_real_by_class(imgs_dir: Path, lbls_dir: Path,
                       class_ids: Sequence[int],
                       include_aug: bool = False) -> Tuple[Dict[int, List[Path]], List[Path]]:
    per_class: Dict[int, List[Path]] = {c: [] for c in class_ids}
    everything: List[Path] = []
    for p in sorted(imgs_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if not include_aug and "_aug" in p.stem:
            continue
        everything.append(p)
        lbl = lbls_dir / (p.stem + ".txt")
        c = dominant_class(lbl)
        if c is not None and c in per_class:
            per_class[c].append(p)
    return per_class, everything


def list_synth_by_class(synth_dir: Path, class_ids: Sequence[int]) -> Tuple[Dict[int, List[Path]], List[Path]]:
    per_class: Dict[int, List[Path]] = {c: [] for c in class_ids}
    everything: List[Path] = []
    for p in sorted(synth_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        everything.append(p)
        stem = p.stem
        for cid in class_ids:
            prefix = RARE_CLASSES[cid]["ru"]
            if stem.startswith(prefix):
                per_class[cid].append(p)
                break
    return per_class, everything


def load_tensor(path: Path, device):
    import torch
    img = Image.open(path).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def mean_lpips(paths: Sequence[Path], loss, device, n_pairs: int,
               batch: int = 16, rng: random.Random | None = None) -> Tuple[float, float, int]:
    import torch
    if len(paths) < 2:
        return float("nan"), float("nan"), 0
    rng = rng or random.Random(RNG_SEED)
    max_pairs = len(paths) * (len(paths) - 1) // 2
    n_pairs = min(n_pairs, max_pairs)
    pairs = set()
    while len(pairs) < n_pairs:
        a, b = rng.sample(range(len(paths)), 2)
        if a > b:
            a, b = b, a
        pairs.add((a, b))
    pairs = list(pairs)

    # preload unique images to keep IO low
    unique_idx = sorted({i for pair in pairs for i in pair})
    # cap cached tensors — if set is huge, reload per pair
    cache: Dict[int, any] = {}
    if len(unique_idx) <= 1500:
        for i in unique_idx:
            cache[i] = load_tensor(paths[i], device)

    distances: List[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch):
            chunk = pairs[start:start + batch]
            a_batch = []
            b_batch = []
            for ai, bi in chunk:
                a_batch.append(cache[ai] if ai in cache else load_tensor(paths[ai], device))
                b_batch.append(cache[bi] if bi in cache else load_tensor(paths[bi], device))
            x1 = torch.cat(a_batch, dim=0)
            x2 = torch.cat(b_batch, dim=0)
            d = loss(x1, x2).reshape(-1).detach().cpu().numpy()
            distances.extend(float(v) for v in d)
    return float(np.mean(distances)), float(np.std(distances)), len(distances)


def make_plot(per_class_rows: List[Dict], overall_rows: List[Dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["real", "diffusion", "nst_v1", "nst_v2"]
    method_labels = {"real": "Real", "diffusion": "Diffusion",
                     "nst_v1": "NST v1", "nst_v2": "NST v2"}
    classes = ["root_rot", "septoria", "nitrogen_def", "frost_damage"]
    class_labels = {"root_rot": "Root rot", "septoria": "Septoria",
                    "nitrogen_def": "Nitrogen def.", "frost_damage": "Frost damage"}

    values = {m: {c: float("nan") for c in classes} for m in methods}
    for r in per_class_rows:
        if r["method"] in values and r["class"] in values[r["method"]]:
            values[r["method"]][r["class"]] = r["lpips_mean"]

    x = np.arange(len(classes))
    width = 0.20
    colors = {"real": "#4d4d4d", "diffusion": "#2c7fb8",
              "nst_v1": "#d95f0e", "nst_v2": "#fdae6b"}

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for i, m in enumerate(methods):
        heights = [values[m][c] for c in classes]
        ax.bar(x + (i - 1.5) * width, heights, width,
               label=method_labels[m], color=colors[m],
               edgecolor="black", linewidth=0.5)
        for xi, h in zip(x + (i - 1.5) * width, heights):
            if not np.isnan(h):
                ax.text(xi, h, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    # horizontal dashed line — real mean across classes
    real_vals = [values["real"][c] for c in classes if not np.isnan(values["real"][c])]
    if real_vals:
        ax.axhline(float(np.mean(real_vals)), linestyle="--",
                   color=colors["real"], alpha=0.6, linewidth=1.2,
                   label="Real mean")

    ax.set_xticks(x)
    ax.set_xticklabels([class_labels[c] for c in classes])
    ax.set_ylabel("Mean LPIPS ↑ (higher = more diverse)")
    ax.set_title("Intra-class LPIPS diversity (AlexNet, 256px)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, default=DEFAULT_REAL_IMGS)
    ap.add_argument("--labels", type=Path, default=DEFAULT_REAL_LBL)
    ap.add_argument("--diffusion", type=Path, default=DEFAULT_SYNTH["diffusion"])
    ap.add_argument("--nst-v1", type=Path, default=DEFAULT_SYNTH["nst_v1"])
    ap.add_argument("--nst-v2", type=Path, default=DEFAULT_SYNTH["nst_v2"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--include-aug", action="store_true")
    ap.add_argument("--n-pairs-overall", type=int, default=2000)
    ap.add_argument("--n-pairs-per-class", type=int, default=500)
    ap.add_argument("--real-overall-max", type=int, default=1500,
                    help="Cap number of real images used for overall LPIPS "
                         "(random subsample) to keep compute in budget")
    ap.add_argument("--real-per-class-max", type=int, default=800,
                    help="Cap per-class real images to keep pair sampling cheap")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    import lpips as lpips_mod

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[info] device={device}")
    loss = lpips_mod.LPIPS(net="alex", verbose=False).to(device).eval()

    class_ids = list(RARE_CLASSES.keys())
    rng = random.Random(RNG_SEED)

    # Real
    print("[prep] listing real ...")
    real_by_class, real_all = list_real_by_class(args.real, args.labels, class_ids,
                                                 include_aug=args.include_aug)
    print(f"[prep] real: total={len(real_all)}  per-class=" +
          ", ".join(f"{RARE_CLASSES[c]['name']}={len(real_by_class[c])}"
                    for c in class_ids))
    if len(real_all) > args.real_overall_max:
        real_all = rng.sample(real_all, args.real_overall_max)
    for c in class_ids:
        if len(real_by_class[c]) > args.real_per_class_max:
            real_by_class[c] = rng.sample(real_by_class[c], args.real_per_class_max)

    # Synthetic
    synth_by_class: Dict[str, Dict[int, List[Path]]] = {}
    synth_all: Dict[str, List[Path]] = {}
    for name, path in [("diffusion", args.diffusion),
                       ("nst_v1", args.nst_v1),
                       ("nst_v2", args.nst_v2)]:
        if not path.exists():
            print(f"[warn] {name}: {path} missing, skipping")
            continue
        by_cls, all_imgs = list_synth_by_class(path, class_ids)
        synth_by_class[name] = by_cls
        synth_all[name] = all_imgs
        print(f"[prep] {name}: total={len(all_imgs)}  per-class=" +
              ", ".join(f"{RARE_CLASSES[c]['name']}={len(by_cls[c])}"
                        for c in class_ids))

    # Overall LPIPS
    overall_rows: List[Dict] = []
    for name, paths in [("real", real_all)] + [(n, synth_all[n]) for n in synth_all]:
        n_pairs = args.n_pairs_overall if name in ("real", "diffusion") \
                  else args.n_pairs_per_class
        print(f"[lpips] overall {name}: n_imgs={len(paths)} n_pairs_target={n_pairs}")
        mean, std, n_used = mean_lpips(paths, loss, device, n_pairs, rng=rng)
        print(f"[lpips] overall {name}: mean={mean:.4f}  std={std:.4f}  pairs={n_used}")
        overall_rows.append({"method": name, "n_images": len(paths),
                             "n_pairs": n_used, "lpips_mean": mean, "lpips_std": std})

    # Per-class LPIPS
    per_class_rows: List[Dict] = []
    for cid in class_ids:
        cname = RARE_CLASSES[cid]["name"]
        # real
        rp = real_by_class[cid]
        mean, std, n_used = mean_lpips(rp, loss, device, args.n_pairs_per_class, rng=rng)
        per_class_rows.append({"class": cname, "class_id": cid, "method": "real",
                               "n_images": len(rp), "n_pairs": n_used,
                               "lpips_mean": mean, "lpips_std": std})
        print(f"[lpips] real/{cname}: mean={mean:.4f}  n_imgs={len(rp)}")

        for sname in synth_all:
            sp = synth_by_class[sname][cid]
            if len(sp) < 2:
                per_class_rows.append({"class": cname, "class_id": cid, "method": sname,
                                       "n_images": len(sp), "n_pairs": 0,
                                       "lpips_mean": float("nan"),
                                       "lpips_std": float("nan")})
                continue
            mean, std, n_used = mean_lpips(sp, loss, device, args.n_pairs_per_class, rng=rng)
            per_class_rows.append({"class": cname, "class_id": cid, "method": sname,
                                   "n_images": len(sp), "n_pairs": n_used,
                                   "lpips_mean": mean, "lpips_std": std})
            print(f"[lpips] {sname}/{cname}: mean={mean:.4f}  n_imgs={len(sp)}")

    # Save
    ov_path = args.out / "lpips_results.csv"
    with ov_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_images", "n_pairs", "lpips_mean", "lpips_std"])
        for r in overall_rows:
            w.writerow([r["method"], r["n_images"], r["n_pairs"],
                        f"{r['lpips_mean']:.6f}", f"{r['lpips_std']:.6f}"])
    print(f"[save] {ov_path}")

    pc_path = args.out / "lpips_per_class.csv"
    with pc_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "class_id", "method", "n_images", "n_pairs",
                    "lpips_mean", "lpips_std"])
        for r in per_class_rows:
            w.writerow([r["class"], r["class_id"], r["method"],
                        r["n_images"], r["n_pairs"],
                        f"{r['lpips_mean']:.6f}" if not np.isnan(r["lpips_mean"]) else "",
                        f"{r['lpips_std']:.6f}" if not np.isnan(r["lpips_std"]) else ""])
    print(f"[save] {pc_path}")

    plot_path = args.out / "lpips_diversity.png"
    make_plot(per_class_rows, overall_rows, plot_path)
    print(f"[save] {plot_path}")

    with (args.out / "lpips_summary.json").open("w") as f:
        json.dump({"overall": overall_rows, "per_class": per_class_rows,
                   "resolution": RESOLUTION, "net": "alex",
                   "include_aug": args.include_aug}, f,
                  ensure_ascii=False, indent=2)

    print("[done]")


if __name__ == "__main__":
    main()
