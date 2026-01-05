import torch

from cc_cad.decoding.cad import fuse_logits


def test_fuse_logits() -> None:
    ctx = torch.tensor([[1.0, 2.0]])
    prior = torch.tensor([[0.5, 0.5]])
    fused = fuse_logits(ctx, prior, alpha=0.5)
    expected = (1.0 + 0.5) * ctx - 0.5 * prior
    assert torch.allclose(fused, expected)
