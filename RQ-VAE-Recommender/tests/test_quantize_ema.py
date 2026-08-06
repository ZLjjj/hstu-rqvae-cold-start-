import torch

from modules.quantize import Quantize
from modules.quantize import QuantizeForwardMode


def test_ema_quantize_updates_codebook_and_backprop():
    quantize = Quantize(
        embed_dim=3,
        n_embed=4,
        do_kmeans_init=False,
        forward_mode=QuantizeForwardMode.EMA,
        dead_code_reset=True,
        dead_code_threshold=0.1,
    )
    quantize.train()

    x = torch.randn(8, 3, requires_grad=True)
    out = quantize(x, temperature=0.2)

    assert out.embeddings.shape == (8, 3)
    assert out.ids.shape == (8,)
    assert quantize.ema_cluster_size.sum().item() > 0

    loss = out.embeddings.sum() + out.loss.mean()
    loss.backward()
    assert x.grad is not None
    assert quantize.last_dead_code_ratio.item() >= 0.0
