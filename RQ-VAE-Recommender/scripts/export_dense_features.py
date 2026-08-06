import argparse
import os
from typing import Dict, Optional, Tuple

import torch


def _to_item_ids_tensor(item_ids_payload) -> Optional[torch.Tensor]:
    if item_ids_payload is None:
        return None
    if torch.is_tensor(item_ids_payload):
        item_ids = item_ids_payload.long()
    else:
        item_ids = torch.tensor(list(item_ids_payload), dtype=torch.long)
    if item_ids.dim() != 1:
        raise ValueError("item_ids must be a 1D tensor")
    return item_ids


def _load_embeddings(embeddings_path: str) -> Tuple[torch.Tensor, str, Optional[torch.Tensor]]:
    payload = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    item_ids = None
    if isinstance(payload, dict):
        dense_vectors = payload.get("embeddings")
        model_name = str(payload.get("model_name", "unknown"))
        item_ids = _to_item_ids_tensor(payload.get("item_ids"))
    else:
        dense_vectors = payload
        model_name = "unknown"
    if not torch.is_tensor(dense_vectors):
        raise ValueError(f"Invalid embeddings payload type: {type(dense_vectors)}")
    if dense_vectors.dim() != 2:
        raise ValueError(f"dense embeddings must be 2D, got shape={tuple(dense_vectors.shape)}")
    return dense_vectors.float(), model_name, item_ids


def _load_movie_ids(movies_path: str) -> torch.Tensor:
    item_ids = []
    with open(movies_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item_ids.append(int(line.split("::", 1)[0]))
    return torch.tensor(item_ids, dtype=torch.long)


def _load_bridge_item_ids(bridge_path: str) -> Optional[torch.Tensor]:
    if not bridge_path or not os.path.exists(bridge_path):
        return None
    payload = torch.load(bridge_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    return _to_item_ids_tensor(payload.get("item_ids"))


def _resolve_item_ids(
    num_vectors: int,
    item_ids_from_embeddings: Optional[torch.Tensor],
    movies_path: str,
    bridge_path: Optional[str],
) -> torch.Tensor:
    if item_ids_from_embeddings is not None and item_ids_from_embeddings.size(0) == num_vectors:
        return item_ids_from_embeddings

    movie_ids = _load_movie_ids(movies_path)
    if movie_ids.size(0) == num_vectors:
        return movie_ids

    bridge_item_ids = _load_bridge_item_ids(bridge_path) if bridge_path else None
    if bridge_item_ids is not None and bridge_item_ids.size(0) == num_vectors:
        return bridge_item_ids

    raise ValueError(
        "Failed to align dense vectors with item ids. "
        f"embeddings_rows={num_vectors}, movies_rows={movie_ids.size(0)}, "
        f"embedding_item_ids_rows={0 if item_ids_from_embeddings is None else item_ids_from_embeddings.size(0)}, "
        f"bridge_item_ids_rows={0 if bridge_item_ids is None else bridge_item_ids.size(0)}"
    )


def export_dense_features(
    embeddings_path: str,
    movies_path: str,
    output_path: str,
    bridge_path: Optional[str] = None,
) -> Dict:
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"embeddings file not found: {embeddings_path}")
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"movies file not found: {movies_path}")

    dense_vectors, model_name, item_ids_from_embeddings = _load_embeddings(embeddings_path)
    item_ids = _resolve_item_ids(
        num_vectors=int(dense_vectors.size(0)),
        item_ids_from_embeddings=item_ids_from_embeddings,
        movies_path=movies_path,
        bridge_path=bridge_path,
    )

    artifact = {
        "item_ids": item_ids,
        "dense_vectors": dense_vectors,
        "metadata": {
            "model_name": model_name,
            "dim": int(dense_vectors.size(1)),
            "num_items": int(item_ids.size(0)),
            "source_embeddings_path": os.path.abspath(embeddings_path),
            "source_movies_path": os.path.abspath(movies_path),
            "source_bridge_path": os.path.abspath(bridge_path) if bridge_path else "",
        },
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(artifact, output_path)
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export dense item features artifact for HSTU dense baseline."
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="dataset/ml-1m",
        help="Dataset root folder containing embeddings_minilm_l6.pt and raw/movies.dat.",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=None,
        help="Optional explicit path to embeddings_minilm_l6.pt",
    )
    parser.add_argument(
        "--movies_path",
        type=str,
        default=None,
        help="Optional explicit path to movies.dat",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="out/rqvae/ml1m/dense_features.pt",
        help="Output path for dense feature artifact.",
    )
    parser.add_argument(
        "--bridge_path",
        type=str,
        default="out/rqvae/ml1m/bridge_artifacts.pt",
        help="Optional bridge artifact used as fallback id mapping when rows mismatch.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    embeddings_path = (
        args.embeddings_path
        if args.embeddings_path is not None
        else os.path.join(args.dataset_folder, "embeddings_minilm_l6.pt")
    )
    movies_path = (
        args.movies_path
        if args.movies_path is not None
        else os.path.join(args.dataset_folder, "raw", "movies.dat")
    )
    artifact = export_dense_features(
        embeddings_path=embeddings_path,
        movies_path=movies_path,
        output_path=args.output_path,
        bridge_path=args.bridge_path,
    )
    print(
        "[export_dense_features] done: "
        f"path={os.path.abspath(args.output_path)}, "
        f"num_items={artifact['metadata']['num_items']}, dim={artifact['metadata']['dim']}, "
        f"model={artifact['metadata']['model_name']}"
    )


if __name__ == "__main__":
    main()
