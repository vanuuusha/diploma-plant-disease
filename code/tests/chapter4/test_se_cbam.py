import torch
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from models.chapter4.se_block import SEBlock
from models.chapter4.cbam_block import CBAMBlock


def test_se_shape():
    se = SEBlock(64)
    x = torch.randn(2, 64, 20, 20)
    y = se(x)
    assert y.shape == x.shape


def test_cbam_shape():
    cbam = CBAMBlock(64)
    x = torch.randn(2, 64, 20, 20)
    y = cbam(x)
    assert y.shape == x.shape


def test_se_gradients():
    se = SEBlock(32)
    x = torch.randn(1, 32, 8, 8, requires_grad=True)
    se(x).sum().backward()
    for p in se.parameters():
        assert p.grad is not None


def test_cbam_gradients():
    cbam = CBAMBlock(32)
    x = torch.randn(1, 32, 8, 8, requires_grad=True)
    cbam(x).sum().backward()
    for p in cbam.parameters():
        assert p.grad is not None


if __name__ == "__main__":
    test_se_shape(); test_cbam_shape(); test_se_gradients(); test_cbam_gradients()
    print("SE/CBAM: 4/4 ok")
