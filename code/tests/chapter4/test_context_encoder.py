import os
os.environ.setdefault("HF_HUB_OFFLINE", "0")
import torch
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from models.chapter4.context_encoder import ContextEncoder


def test_mobilenet():
    enc = ContextEncoder("mobilenetv3_small_100", out_dim=256, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = enc(x)
    assert y.shape == (2, 256)


def test_efficientnet():
    enc = ContextEncoder("efficientnet_b0", out_dim=256, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = enc(x)
    assert y.shape == (2, 256)


def test_vit_tiny():
    enc = ContextEncoder("vit_tiny_patch16_224", out_dim=256, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = enc(x)
    assert y.shape == (2, 256)


if __name__ == "__main__":
    test_mobilenet(); test_efficientnet(); test_vit_tiny()
    print("ContextEncoder: 3/3 ok")
