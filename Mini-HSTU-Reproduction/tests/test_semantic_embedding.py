import torch

from generative_recommenders_pl.models.embeddings.embeddings import (
    SemanticCodebookEmbeddingModule,
)


def test_semantic_embedding_padding_and_lookup(tmp_path):
    item_to_codes = torch.tensor(
        [
            [-1, -1],
            [0, 1],
            [2, 3],
            [-1, -1],
        ],
        dtype=torch.long,
    )
    codebook_vectors = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]],
        ],
        dtype=torch.float32,
    )

    bridge_path = tmp_path / "bridge.pt"
    torch.save(
        {
            "item_id_to_codes": item_to_codes,
            "codebook_vectors": codebook_vectors,
        },
        bridge_path,
    )

    module = SemanticCodebookEmbeddingModule(
        num_items=3,
        item_embedding_dim=2,
        item_to_codes_path=str(bridge_path),
        codebook_weights_path=str(bridge_path),
        num_codebooks=2,
        codebook_size=4,
        compose="sum",
        freeze_codebook=True,
    )

    item_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    out = module.get_item_embeddings(item_ids)

    assert out.shape == (4, 2)
    assert torch.allclose(out[0], torch.zeros(2))
    assert torch.allclose(out[3], torch.zeros(2))
    assert torch.allclose(out[1], torch.tensor([21.0, 21.0]))
    assert torch.allclose(out[2], torch.tensor([43.0, 43.0]))


def test_semantic_embedding_metadata_mismatch_raises(tmp_path):
    item_to_codes = torch.tensor(
        [
            [-1, -1],
            [0, 1],
            [2, 3],
        ],
        dtype=torch.long,
    )
    codebook_vectors = torch.randn(2, 4, 2)
    bridge_path = tmp_path / "bridge.pt"
    torch.save(
        {
            "item_id_to_codes": item_to_codes,
            "codebook_vectors": codebook_vectors,
            "metadata": {"n_codebooks": 2, "codebook_size": 4},
        },
        bridge_path,
    )

    try:
        SemanticCodebookEmbeddingModule(
            num_items=2,
            item_embedding_dim=2,
            item_to_codes_path=str(bridge_path),
            codebook_weights_path=str(bridge_path),
            num_codebooks=3,
            codebook_size=4,
            compose="concat_proj",
            freeze_codebook=True,
        )
        raise AssertionError("Expected ValueError for num_codebooks mismatch")
    except ValueError as exc:
        assert "num_codebooks mismatch" in str(exc)

    try:
        SemanticCodebookEmbeddingModule(
            num_items=2,
            item_embedding_dim=2,
            item_to_codes_path=str(bridge_path),
            codebook_weights_path=str(bridge_path),
            num_codebooks=2,
            codebook_size=8,
            compose="concat_proj",
            freeze_codebook=True,
        )
        raise AssertionError("Expected ValueError for codebook_size mismatch")
    except ValueError as exc:
        assert "codebook_size mismatch" in str(exc)
