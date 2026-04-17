import torch
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from models.chapter4.late_fusion_head import LateFusionClassifier


def test_shape():
    head = LateFusionClassifier(roi_channels=192, roi_spatial=7,
                                 context_dim=256, num_classes=9)
    roi = torch.randn(4, 192, 7, 7)
    c = torch.randn(4, 256)
    y = head(roi, c)
    assert y.shape == (4, 9)


def test_gradients():
    head = LateFusionClassifier(roi_channels=64, roi_spatial=5,
                                 context_dim=128, num_classes=9)
    roi = torch.randn(2, 64, 5, 5)
    c = torch.randn(2, 128)
    head(roi, c).sum().backward()
    for p in head.parameters():
        assert p.grad is not None


if __name__ == "__main__":
    test_shape(); test_gradients()
    print("LateFusion: 2/2 ok")
