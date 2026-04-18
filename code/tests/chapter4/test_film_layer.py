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


def test_near_identity_at_init():
    """γ=1+0.5·tanh(Wc+b), W~N(0, 0.05) → γ ≈ 1 + шум ~0.05.
    При нулевом контексте: W·0 = 0 → γ = 1 + 0.5·tanh(0) = 1, β = 0."""
    film = FiLMLayer(context_dim=256, feature_channels=8)
    f = torch.randn(1, 8, 4, 4)
    c = torch.zeros(1, 256)  # zero context → Wc = 0 → γ=1, β=0
    y = film(f, c)
    assert torch.allclose(y, f, atol=1e-5)
    # С нормальным контекстом γ близко к 1, но не точно
    c2 = torch.randn(1, 256)
    g = film.last_gamma(c2)
    assert (g > 0.5).all() and (g < 1.5).all()


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
    assert (g >= 0).all() and (g <= 2).all()


if __name__ == "__main__":
    test_shape(); test_identity_at_init(); test_gradients_flow(); test_last_gamma_shape()
    print("FiLMLayer: 4/4 ok")
