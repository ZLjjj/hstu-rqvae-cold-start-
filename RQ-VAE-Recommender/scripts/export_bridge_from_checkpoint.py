import argparse
import os
from typing import Dict

import torch

from data.processed import ItemData, RecDataset
from modules.rqvae import RqVae
from train_rqvae import _build_item_seq_batch, _export_bridge_artifact


DATASET_MAP: Dict[str, RecDataset] = {
    "ml-1m": RecDataset.ML_1M,
    "ml_1m": RecDataset.ML_1M,
    "ml1m": RecDataset.ML_1M,
    "ml-32m": RecDataset.ML_32M,
    "ml_32m": RecDataset.ML_32M,
    "ml32m": RecDataset.ML_32M,
    "amazon": RecDataset.AMAZON,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HSTU bridge artifact from a trained RQ-VAE checkpoint."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="out/rqvae/ml1m/checkpoint_9999.pt",
        help="Path to RQ-VAE checkpoint file.",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default="out/rqvae/ml1m/bridge_artifacts.pt",
        help="Output path for bridge artifact.",
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="dataset/ml-1m",
        help="Dataset folder used by ItemData.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-1m",
        choices=sorted(DATASET_MAP.keys()),
        help="Dataset key.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="ml-1m",
        help="Split name used by dataset loader (for ML-1M this can stay 'ml-1m').",
    )
    parser.add_argument(
        "--text_encoder_name",
        type=str,
        default="/root/autodl-tmp/models/all-MiniLM-L6-v2",
        help="Text encoder name/path used if processed cache needs to be built.",
    )
    parser.add_argument(
        "--text_batch_size",
        type=int,
        default=128,
        help="Text encoding batch size when processing raw data.",
    )
    parser.add_argument(
        "--text_cache_path",
        type=str,
        default=None,
        help="Optional cache path for text embeddings.",
    )
    parser.add_argument(
        "--text_encoder_device",
        type=str,
        default="auto",
        help="Text encoder device: auto/cpu/cuda.",
    )
    parser.add_argument(
        "--cold_item_ids_path",
        type=str,
        default=None,
        help="Optional cold item id list (.pt/.json) to fill cold_item_mask in bridge artifact.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for encoding items to semantic ids.",
    )
    parser.add_argument(
        "--force_dataset_process",
        action="store_true",
        help="Force rebuilding processed dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for bridge export (auto-falls back to cpu if cuda unavailable).",
    )
    return parser.parse_args()


def _load_model(checkpoint_path: str, device: torch.device) -> RqVae:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_config" not in state or "model" not in state:
        raise ValueError("checkpoint must contain keys: model_config and model")

    model_config = dict(state["model_config"])
    model_config.pop("self", None)
    model_config.pop("__class__", None)

    rqvae = RqVae(**model_config)
    rqvae.load_state_dict(state["model"], strict=True)
    # Quantize.kmeans_initted is not checkpointed; force-disable kmeans init on export.
    for layer in rqvae.layers:
        layer.do_kmeans_init = False
        layer.kmeans_initted = True
    rqvae.to(device)
    rqvae.eval()
    return rqvae


def _load_ml1m_item_features_fast(dataset_folder: str) -> tuple[torch.Tensor, torch.Tensor]:
    emb_path = os.path.join(dataset_folder, "embeddings_minilm_l6.pt")
    movies_path = os.path.join(dataset_folder, "raw", "movies.dat")
    if not os.path.exists(emb_path) or not os.path.exists(movies_path):
        raise FileNotFoundError(
            "ML-1M fast-path files not found. "
            f"expected: {emb_path} and {movies_path}"
        )

    emb_payload = torch.load(emb_path, map_location="cpu", weights_only=False)
    if isinstance(emb_payload, dict):
        item_x = emb_payload.get("embeddings")
    else:
        item_x = emb_payload
    if not isinstance(item_x, torch.Tensor):
        raise ValueError(f"Invalid embedding payload type: {type(item_x)}")
    item_x = item_x.float()

    item_ids = []
    with open(movies_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item_ids.append(int(line.split("::", 1)[0]))
    external_ids = torch.tensor(item_ids, dtype=torch.long)

    if item_x.size(0) != external_ids.size(0):
        raise ValueError(
            "Mismatch between ML-1M embeddings and movies.dat rows: "
            f"{item_x.size(0)} vs {external_ids.size(0)}"
        )
    return item_x, external_ids


@torch.inference_mode()
def _encode_all_items(
    rqvae: RqVae,
    item_x: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    out = []
    n_items = item_x.size(0)
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        ids = torch.arange(start, end, device=device, dtype=torch.long)
        x = item_x[start:end]
        batch = _build_item_seq_batch(ids=ids, x=x)
        sem_ids = rqvae.get_semantic_ids(batch.x).sem_ids
        out.append(sem_ids.detach().cpu())
    return torch.cat(out, dim=0)


def main() -> None:
    args = _parse_args()
    requested_device = args.device.strip().lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[export_bridge] cuda unavailable, falling back to cpu", flush=True)
    else:
        device = torch.device(requested_device)

    dataset_enum = DATASET_MAP[args.dataset]

    print(
        f"[export_bridge] loading checkpoint={args.checkpoint_path} on device={device}",
        flush=True,
    )
    rqvae = _load_model(checkpoint_path=args.checkpoint_path, device=device)

    item_x = None
    item_external_ids = None
    if dataset_enum == RecDataset.ML_1M and not args.force_dataset_process:
        try:
            item_x, item_external_ids = _load_ml1m_item_features_fast(args.dataset_folder)
            print(
                "[export_bridge] using ML-1M fast path: embeddings_minilm_l6.pt + movies.dat",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[export_bridge] fast path unavailable ({exc}), falling back to ItemData loader",
                flush=True,
            )

    if item_x is None or item_external_ids is None:
        print(
            "[export_bridge] loading item dataset via ItemData: "
            f"root={args.dataset_folder}, split={args.dataset_split}",
            flush=True,
        )
        item_dataset = ItemData(
            root=args.dataset_folder,
            dataset=dataset_enum,
            force_process=args.force_dataset_process,
            train_test_split="all",
            split=args.dataset_split,
            text_encoder_name=args.text_encoder_name,
            text_batch_size=args.text_batch_size,
            text_cache_path=args.text_cache_path,
            text_encoder_device=args.text_encoder_device,
        )
        item_x = item_dataset.item_data
        item_external_ids = item_dataset.item_external_ids

    item_x = item_x.to(device)
    print(
        f"[export_bridge] encoding semantic ids for {item_x.size(0)} items (batch_size={args.batch_size})",
        flush=True,
    )
    corpus_ids = _encode_all_items(
        rqvae=rqvae,
        item_x=item_x,
        batch_size=max(1, int(args.batch_size)),
        device=device,
    )

    _export_bridge_artifact(
        export_path=args.export_path,
        external_item_ids=item_external_ids,
        corpus_ids=corpus_ids,
        rqvae=rqvae,
        vae_n_layers=rqvae.n_layers,
        cold_item_ids_path=args.cold_item_ids_path,
    )
    print(f"[export_bridge] done: {args.export_path}", flush=True)


if __name__ == "__main__":
    main()
