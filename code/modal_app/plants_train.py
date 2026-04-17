"""
Modal app для обучения детекторов главы 3 на A100.

Usage:
    # 1. Upload datasets to a shared volume (one-time):
    #    modal volume create plants-dataset
    #    Locally dereference and upload (see code/modal_app/upload_datasets.sh)

    # 2. Run training for one detector × variant:
    modal run code/modal_app/plants_train.py::train_detector \
        --detector rtdetr --variant baseline

    # 3. Fetch results (writes to code/results/task_NN/):
    modal run code/modal_app/plants_train.py::fetch_results \
        --detector rtdetr --variant baseline

Architecture:
    - Volume `plants-dataset`: contains all 4 dataset variants dereferenced
    - Volume `plants-results`: results written by training jobs
    - Image: Python 3.12 + torch + ultralytics + transformers + torchmetrics
    - GPU: A100-40GB (or configurable via env PLANTS_GPU)
    - Timeout: 4 hours per training job
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

GPU_TYPE = os.environ.get("PLANTS_GPU", "A100-40GB")

app = modal.App("plants-train")

# Container image: all deps needed for all 4 detectors, + local code
_PROJ = Path(__file__).parent.parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.9.1",
        "torchvision==0.24.1",
        "ultralytics==8.4.38",
        "transformers==4.57.3",
        "timm",
        "torchmetrics",
        "pycocotools",
        "albumentations==2.0.8",
        "pillow",
        "opencv-python-headless",
        "pandas",
        "matplotlib",
        "scikit-learn",
    )
    .add_local_dir(str(_PROJ / "notebooks"), remote_path="/code/notebooks")
    .add_local_dir(str(_PROJ / "docs"), remote_path="/code/docs")
)

# Persistent volumes
dataset_vol = modal.Volume.from_name("plants-dataset", create_if_missing=True)
results_vol = modal.Volume.from_name("plants-results", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=4 * 3600,
    volumes={"/data": dataset_vol, "/out": results_vol},
)
def train_detector(
    detector: str,
    variant: str,
    epochs: int = 100,
    patience: int = 15,
    batch: int | None = None,
):
    """Train one detector × variant on Modal A100. Returns summary stats."""
    import torch

    os.chdir("/")

    # Patch paths so chapter3_common finds datasets and writes to volume
    sys.path.insert(0, "/code/notebooks")
    import chapter3_common as cc

    cc.ROOT = Path("/")
    cc.QUALITATIVE_SAMPLE = Path("/code/docs/chapter3_qualitative_sample.txt")
    cc.TEST_IMG_DIR = Path("/data/dataset/test/images")
    cc.TEST_LBL_DIR = Path("/data/dataset/test/labels")
    cc.DATASET_VARIANTS = {
        "baseline": Path("/data/dataset/data.yaml"),
        "aug_geom": Path("/data/dataset_augmented/data.yaml"),
        "aug_oversample": Path("/data/dataset_balanced/data.yaml"),
        "aug_diffusion": Path("/data/dataset_final/data.yaml"),
    }

    # Ensure data.yaml on volume points to correct absolute paths
    for v in ["dataset", "dataset_augmented", "dataset_balanced", "dataset_final"]:
        p = Path(f"/data/{v}/data.yaml")
        if p.exists():
            p.write_text(
                f"path: /data/{v}\n"
                "train: train/images\n"
                "val: val/images\n"
                "test: test/images\n"
                "nc: 9\n"
                "names:\n"
                "- Недостаток P2O5\n"
                "- Листовая (бурая) ржавчина\n"
                "- Мучнистая роса\n"
                "- Пиренофороз\n"
                "- Фузариоз\n"
                "- Корневая гниль\n"
                "- Септориоз\n"
                "- Недостаток N\n"
                "- Повреждение заморозками\n"
            )

    print(f"=== Modal: {detector}/{variant} on {torch.cuda.get_device_name(0)} ===")
    print(f"GPU mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    t0 = time.time()

    if detector in ("yolov12", "rtdetr"):
        import chapter3_ultralytics_runner as runner
        # Redirect task_dir to volume
        runner.DETECTOR_CONFIG[detector]["task_dir"] = Path(f"/out/task_{'07' if detector == 'yolov12' else '08'}")
        runner.DETECTOR_CONFIG[detector]["task_dir"].mkdir(parents=True, exist_ok=True)
        b = batch if batch else runner.DETECTOR_CONFIG[detector]["batch"]
        # Override default batch via module-level patch if batch specified
        if batch:
            runner.DETECTOR_CONFIG[detector]["batch"] = batch
        runner.run_single(detector, variant, epochs, patience)
        runner.write_summary(detector)
    elif detector == "faster_rcnn":
        import chapter3_torchvision_runner as runner
        runner.TASK_DIR = Path("/out/task_09")
        runner.TASK_DIR.mkdir(parents=True, exist_ok=True)
        runner.run_single(variant, epochs, patience, batch or 4)
        runner.write_summary()
    elif detector == "detr":
        import chapter3_detr_runner as runner
        runner.TASK_DIR = Path("/out/task_10")
        runner.TASK_DIR.mkdir(parents=True, exist_ok=True)
        runner.run_single(variant, epochs, patience, batch or 16)
        runner.write_summary()
    else:
        raise ValueError(f"Unknown detector: {detector}")

    dt = time.time() - t0
    results_vol.commit()
    print(f"=== Done {detector}/{variant} in {dt/60:.1f} min ===")
    return {"detector": detector, "variant": variant, "duration_min": dt / 60}


@app.function(image=image, volumes={"/data": dataset_vol})
def list_dataset():
    """Verify dataset is uploaded and readable."""
    out = {}
    for v in ["dataset", "dataset_augmented", "dataset_balanced", "dataset_final"]:
        p = Path(f"/data/{v}")
        if p.exists():
            t = len(list((p / "train" / "images").iterdir())) if (p / "train" / "images").exists() else 0
            v_ = len(list((p / "val" / "images").iterdir())) if (p / "val" / "images").exists() else 0
            tt = len(list((p / "test" / "images").iterdir())) if (p / "test" / "images").exists() else 0
            out[v] = {"train": t, "val": v_, "test": tt}
    return out


@app.local_entrypoint()
def main(detector: str = "rtdetr", variant: str = "baseline", epochs: int = 100, patience: int = 15, batch: int = 0):
    """
    CLI entry. Example:
        modal run code/modal_app/plants_train.py --detector rtdetr --variant baseline
    """
    b = batch if batch > 0 else None
    result = train_detector.remote(detector, variant, epochs, patience, b)
    print(result)
