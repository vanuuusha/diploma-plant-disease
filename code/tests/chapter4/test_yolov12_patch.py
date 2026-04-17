import torch
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from models.chapter4.film_layer import FiLMLayer
from models.chapter4.se_block import SEBlock
from models.chapter4.context_encoder import ContextEncoder
from models.chapter4.yolov12_patch import wrap_neck_with, describe_wrap, ModulatedLayer


def _load_yolov12():
    from ultralytics import YOLO
    # В тестах грузим малую YOLOv12n (~2MB), чтобы не тянуть yolo12m
    y = YOLO("yolo12n.pt")
    return y


def test_wrap_self_attention_se():
    y = _load_yolov12()
    info = wrap_neck_with(y.model, block_factory=SEBlock, context_encoder=None)
    assert len(info["modulated_layers"]) == 3
    assert all(isinstance(m, ModulatedLayer) for m in info["modulated_layers"])
    # dry-run
    x = torch.zeros(1, 3, 640, 640)
    y.model.eval()
    with torch.no_grad():
        out = y.model(x)
    # Detect возвращает tuple или tensor — важна не форма, а отсутствие падения
    assert out is not None


def test_wrap_film_with_context():
    y = _load_yolov12()
    enc = ContextEncoder("mobilenetv3_small_100", pretrained=False)
    info = wrap_neck_with(
        y.model,
        block_factory=lambda ch: FiLMLayer(context_dim=256, feature_channels=ch),
        context_encoder=enc,
    )
    assert len(info["modulated_layers"]) == 3
    # forward с тензором — контекст генерируется автоматически
    x = torch.zeros(1, 3, 640, 640)
    y.model.eval()
    with torch.no_grad():
        _ = y.model(x)
    # проверим, что γ сохранился
    assert info["modulated_layers"][0].last_gamma is not None


def test_wrap_levels_subset_p5only():
    y = _load_yolov12()
    info = wrap_neck_with(y.model, block_factory=SEBlock,
                          context_encoder=None, levels=["P5"])
    assert len(info["modulated_layers"]) == 1


if __name__ == "__main__":
    test_wrap_self_attention_se()
    test_wrap_film_with_context()
    test_wrap_levels_subset_p5only()
    print("yolov12_patch: 3/3 ok")
