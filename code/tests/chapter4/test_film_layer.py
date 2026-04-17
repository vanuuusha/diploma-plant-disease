import torch
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from models.chapter4.film_layer import FiLMLayer


def test_shape():
    film = FiLMLayer(context_dim=256, feature_channels=64)
    f = torch.randn(2, 64, 20, 20)
    c = torch.randn(2, 256)
    y = film(f, c)
    assert y.shape == f.shape


def test_identity_at_init():
    """С нулевым β и γ-инициализацией близко к 0 (→ σ≈0.5) F' ≈ 0.5·F."""
    film = FiLMLayer(context_dim=256, feature_channels=8)
    f = torch.randn(1, 8, 4, 4)
    c = torch.zeros(1, 256)  # если context нулевой → γ=σ(0)=0.5, β=0
    y = film(f, c)
    assert torch.allclose(y, 0.5 * f, atol=1e-6)


def test_gradients_flow():
    film = FiLMLayer(context_dim=256, feature_channels=16)
    f = torch.randn(2, 16, 8, 8, requires_grad=True)
    c = torch.randn(2, 256, requires_grad=True)
    y = film(f, c).sum()
    y.backward()
    for p in film.parameters():
        assert p.grad is not None
    assert c.grad is not None
    assert f.grad is not None


def test_last_gamma_shape():
    film = FiLMLayer(context_dim=64, feature_channels=32)
    c = torch.randn(3, 64)
    g = film.last_gamma(c)
    assert g.shape == (3, 32)
    assert (g >= 0).all() and (g <= 1).all()


if __name__ == "__main__":
    test_shape(); test_identity_at_init(); test_gradients_flow(); test_last_gamma_shape()
    print("FiLMLayer: 4/4 ok")
