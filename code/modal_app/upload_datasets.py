"""
Upload all 4 dataset variants to modal volume `plants-dataset`.

Handles symlinks in local dataset (which point to ../all_dieseas_class/origins/<class>/)
by following them via modal.Volume.batch_upload(follow_symlinks=True).

Usage:
    modal run code/modal_app/upload_datasets.py
"""
from pathlib import Path

import modal

app = modal.App("plants-upload-datasets")
vol = modal.Volume.from_name("plants-dataset", create_if_missing=True)

PROJECT = Path("/home/vanusha/diplom/diploma-plant-disease")
DATASETS = ["dataset", "dataset_augmented", "dataset_balanced", "dataset_final"]


@app.local_entrypoint()
def main():
    total_files = 0
    total_mb = 0
    print("=== Uploading plant datasets to modal volume ===")
    with vol.batch_upload(force=True) as batch:
        for ds in DATASETS:
            src = PROJECT / "code" / "data" / ds
            if not src.exists():
                print(f"  SKIP {ds}: not found")
                continue
            n_files = 0
            n_bytes = 0
            # Walk dataset dir, upload each real file (resolving symlinks)
            for sub in ["train/images", "train/labels",
                        "val/images", "val/labels",
                        "test/images", "test/labels"]:
                sub_dir = src / sub
                if not sub_dir.exists():
                    continue
                for p in sub_dir.iterdir():
                    # Resolve symlinks here
                    real = p.resolve() if p.is_symlink() else p
                    if real.exists() and real.is_file():
                        remote = f"/{ds}/{sub}/{p.name}"
                        batch.put_file(real, remote)
                        n_files += 1
                        n_bytes += real.stat().st_size
            # data.yaml / README etc.
            for meta in ["data.yaml", "README.md", "summary.yaml"]:
                mf = src / meta
                if mf.exists():
                    batch.put_file(mf, f"/{ds}/{meta}")
                    n_files += 1
                    n_bytes += mf.stat().st_size
            print(f"  {ds}: {n_files} files, {n_bytes / 1e9:.2f} GB")
            total_files += n_files
            total_mb += n_bytes / 1e6
    print(f"=== Total: {total_files} files, {total_mb:.1f} MB ===")


if __name__ == "__main__":
    main()
