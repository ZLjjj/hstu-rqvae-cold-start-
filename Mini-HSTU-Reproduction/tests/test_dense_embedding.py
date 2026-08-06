import torch

from generative_recommenders_pl.models.embeddings.embeddings import (
    DenseFeatureEmbeddingModule,
)


def test_dense_feature_embedding_alignment_and_masking(tmp_path):
    dense_path = tmp_path / "dense_features.pt"
    dense_vectors = torch.randn(3, 384)
    torch.save(
        {
            "item_ids": torch.tensor([1, 3, 5], dtype=torch.long),
            "dense_vectors": dense_vectors,
            "metadata": {"model_name": "all-MiniLM-L6-v2", "dim": 384},
        },
        dense_path,
    )

    module = DenseFeatureEmbeddingModule(
        num_items=6,
        item_embedding_dim=8,
        dense_feature_path=str(dense_path),
        freeze_dense=True,
    )
    ids = torch.tensor([0, 1, 2, 3, 5], dtype=torch.long)
    out = module.get_item_embeddings(ids)

    assert out.shape == (5, 8)
    assert torch.allclose(out[0], torch.zeros(8), atol=1e-6)  # padding id
    assert torch.allclose(out[2], torch.zeros(8), atol=1e-6)  # missing dense id
    assert not torch.allclose(out[1], torch.zeros(8), atol=1e-6)
    assert module._project.weight.requires_grad
