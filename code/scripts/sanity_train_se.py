"""
Sanity-check: 1 эпоха YOLOv12n + SE-блоки, подтверждает что patch
совместим с Ultralytics trainer.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from models.chapter4.se_block import SEBlock
from models.chapter4.yolov12_patch import wrap_neck_with, describe_wrap


def main():
    y = YOLO("yolo12n.pt")
    info = wrap_neck_with(y.model, block_factory=SEBlock, context_encoder=None)
    print("Wrap:", describe_wrap(info))
    # 1-epoch sanity
    y.train(
        data="code/data/dataset_final/data.yaml",
        epochs=1,
        imgsz=640,
        batch=8,
        device=0,
        seed=42,
        project="/tmp/sanity_se",
        name="run",
        verbose=True,
        exist_ok=True,
        # все встроенные аугментации отключены (как в гл. 3)
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
        close_mosaic=0, erasing=0.0, crop_fraction=1.0,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
