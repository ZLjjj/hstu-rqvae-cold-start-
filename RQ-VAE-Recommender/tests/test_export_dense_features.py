import torch

from scripts.export_dense_features import export_dense_features


def test_export_dense_features_schema_and_alignment(tmp_path):
    dataset_dir = tmp_path / "dataset" / "ml-1m"
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True)

    embeddings_path = dataset_dir / "embeddings_minilm_l6.pt"
    movies_path = raw_dir / "movies.dat"
    output_path = tmp_path / "out" / "dense_features.pt"

    embeddings = torch.randn(3, 384)
    torch.save({"embeddings": embeddings, "model_name": "all-MiniLM-L6-v2"}, embeddings_path)

    movies_path.write_text(
        "1::Toy Story (1995)::Animation|Children's|Comedy\n"
        "3::Jumanji (1995)::Adventure|Children's|Fantasy\n"
        "5::Grumpier Old Men (1995)::Comedy|Romance\n",
        encoding="latin-1",
    )

    artifact = export_dense_features(
        embeddings_path=str(embeddings_path),
        movies_path=str(movies_path),
        output_path=str(output_path),
    )

    assert output_path.exists()
    assert set(artifact.keys()) == {"item_ids", "dense_vectors", "metadata"}
    assert artifact["item_ids"].dtype == torch.long
    assert artifact["dense_vectors"].shape == (3, 384)
    assert artifact["metadata"]["num_items"] == 3
    assert artifact["metadata"]["dim"] == 384
    assert artifact["metadata"]["model_name"] == "all-MiniLM-L6-v2"


def test_export_dense_features_uses_bridge_ids_on_row_mismatch(tmp_path):
    dataset_dir = tmp_path / "dataset" / "ml-1m"
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True)

    embeddings_path = dataset_dir / "embeddings_minilm_l6.pt"
    movies_path = raw_dir / "movies.dat"
    bridge_path = tmp_path / "bridge_artifacts.pt"
    output_path = tmp_path / "out" / "dense_features.pt"

    embeddings = torch.randn(2, 384)
    torch.save({"embeddings": embeddings, "model_name": "all-MiniLM-L6-v2"}, embeddings_path)
    movies_path.write_text(
        "1::Toy Story (1995)::Animation|Children's|Comedy\n"
        "2::Jumanji (1995)::Adventure|Children's|Fantasy\n"
        "3::Grumpier Old Men (1995)::Comedy|Romance\n",
        encoding="latin-1",
    )
    torch.save({"item_ids": torch.tensor([10, 20], dtype=torch.long)}, bridge_path)

    artifact = export_dense_features(
        embeddings_path=str(embeddings_path),
        movies_path=str(movies_path),
        output_path=str(output_path),
        bridge_path=str(bridge_path),
    )

    assert output_path.exists()
    assert artifact["item_ids"].tolist() == [10, 20]
