from __future__ import annotations

import abc
import os

import torch

from generative_recommenders_pl.models.utils.initialization import truncated_normal
from generative_recommenders_pl.utils.logger import RankedLogger
from typing import Optional

log = RankedLogger(__name__)


class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self):
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                log.info(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                log.info(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self):
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                log.info(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                log.info(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class SemanticCodebookEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_to_codes_path: str,
        codebook_weights_path: str | None = None,
        num_codebooks: Optional[int] = None,
        codebook_size: Optional[int] = None,
        compose: str = "concat_proj",
        freeze_codebook: bool = True,
    ) -> None:
        super().__init__()
        self._item_embedding_dim = item_embedding_dim
        self._num_items = num_items
        self._compose = compose

        bridge_data = None
        if os.path.exists(item_to_codes_path):
            bridge_data = torch.load(item_to_codes_path, map_location="cpu", weights_only=False)
        else:
            raise FileNotFoundError(
                "item_to_codes_path not found: "
                f"{item_to_codes_path}\n"
                "Please generate bridge artifact first:\n"
                "  cd ../RQ-VAE-Recommender\n"
                "  make train_rqvae config=configs/rqvae_ml1m.gin"
            )

        if isinstance(bridge_data, dict):
            item_to_codes = bridge_data.get("item_id_to_codes")
            bridge_codebooks = bridge_data.get("codebook_vectors")
            bridge_meta = bridge_data.get("metadata", {})
        else:
            item_to_codes = bridge_data
            bridge_codebooks = None
            bridge_meta = {}

        if item_to_codes is None:
            raise ValueError("item_to_codes is missing in semantic bridge artifact")
        if item_to_codes.dim() != 2:
            raise ValueError("item_to_codes must be a 2D tensor [num_items+1, num_codebooks]")

        bridge_num_codebooks = int(bridge_meta.get("n_codebooks", item_to_codes.size(1)))
        resolved_num_codebooks = (
            bridge_num_codebooks if num_codebooks is None else int(num_codebooks)
        )
        if num_codebooks is not None and int(num_codebooks) != bridge_num_codebooks:
            raise ValueError(
                f"num_codebooks mismatch: config={num_codebooks}, bridge={bridge_num_codebooks}"
            )
        if item_to_codes.size(1) < resolved_num_codebooks:
            raise ValueError(
                f"item_to_codes second dim {item_to_codes.size(1)} < num_codebooks {resolved_num_codebooks}"
            )

        if item_to_codes.size(0) <= num_items:
            pad_rows = num_items + 1 - item_to_codes.size(0)
            padding = torch.full((pad_rows, item_to_codes.size(1)), -1, dtype=torch.long)
            item_to_codes = torch.cat([item_to_codes.long(), padding], dim=0)
        item_to_codes = item_to_codes[:, :resolved_num_codebooks].long()

        codebook_weights = None
        if codebook_weights_path is not None:
            payload = torch.load(codebook_weights_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict):
                codebook_weights = payload.get("codebook_vectors")
            else:
                codebook_weights = payload
        elif bridge_codebooks is not None:
            codebook_weights = bridge_codebooks

        if codebook_weights is not None and codebook_weights.dim() != 3:
            raise ValueError("codebook_vectors must be a 3D tensor [L, K, D]")

        bridge_codebook_size = int(
            bridge_meta.get(
                "codebook_size",
                codebook_weights.size(1) if codebook_weights is not None else -1,
            )
        )
        if bridge_codebook_size <= 0:
            raise ValueError("Unable to resolve codebook_size from bridge metadata")
        resolved_codebook_size = (
            bridge_codebook_size if codebook_size is None else int(codebook_size)
        )
        if codebook_size is not None and int(codebook_size) != bridge_codebook_size:
            raise ValueError(
                f"codebook_size mismatch: config={codebook_size}, bridge={bridge_codebook_size}"
            )

        valid_codes = item_to_codes[item_to_codes >= 0]
        if valid_codes.numel() > 0 and int(valid_codes.max().item()) >= resolved_codebook_size:
            raise ValueError(
                "item_to_codes contains code id outside configured codebook_size: "
                f"max_id={int(valid_codes.max().item())}, codebook_size={resolved_codebook_size}"
            )

        self._num_codebooks = resolved_num_codebooks
        self._codebook_size = resolved_codebook_size
        self.register_buffer("_item_to_codes", item_to_codes)

        if codebook_weights is None:
            codebook_dim = item_embedding_dim
        else:
            if codebook_weights.size(0) < resolved_num_codebooks:
                raise ValueError(
                    "codebook_vectors has fewer layers than resolved num_codebooks: "
                    f"{codebook_weights.size(0)} < {resolved_num_codebooks}"
                )
            if codebook_weights.size(1) != resolved_codebook_size:
                raise ValueError(
                    "codebook_vectors size mismatch with resolved codebook_size: "
                    f"{codebook_weights.size(1)} != {resolved_codebook_size}"
                )
            codebook_dim = int(codebook_weights.size(-1))

        self._codebook_embs = torch.nn.ModuleList(
            [
                torch.nn.Embedding(resolved_codebook_size + 1, codebook_dim, padding_idx=0)
                for _ in range(resolved_num_codebooks)
            ]
        )
        for idx, emb in enumerate(self._codebook_embs):
            if codebook_weights is not None and idx < codebook_weights.size(0):
                emb.weight.data.zero_()
                emb.weight.data[1 : resolved_codebook_size + 1] = codebook_weights[
                    idx, :resolved_codebook_size
                ].to(emb.weight.dtype)
            else:
                truncated_normal(emb.weight, mean=0.0, std=0.02)
            if freeze_codebook:
                emb.weight.requires_grad = False

        if compose == "concat_proj":
            self._project = torch.nn.Linear(
                codebook_dim * resolved_num_codebooks, item_embedding_dim
            )
        elif compose == "sum":
            self._project = torch.nn.Identity()
            if codebook_dim != item_embedding_dim:
                raise ValueError("compose=sum requires codebook dim == item_embedding_dim")
        else:
            raise ValueError(f"Unsupported compose mode: {compose}")

        valid_item_mask = (self._item_to_codes >= 0).all(dim=1)
        valid_item_count = int(valid_item_mask.sum().item())
        max_external_item_id = int(self._item_to_codes.size(0) - 1)
        log.info(
            "Loaded semantic bridge: "
            f"path={item_to_codes_path}, num_codebooks={self._num_codebooks}, "
            f"codebook_size={self._codebook_size}, valid_items={valid_item_count}, "
            f"max_item_id={max_external_item_id}"
        )

    def debug_str(self) -> str:
        return f"semantic_codebook_l{self._num_codebooks}_k{self._codebook_size}_{self._compose}"

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = item_ids.long()
        item_ids_safe = item_ids.clamp(min=0, max=self._item_to_codes.size(0) - 1)
        sem_ids = self._item_to_codes[item_ids_safe]
        invalid_items = item_ids <= 0

        code_indices = sem_ids + 1
        code_indices[sem_ids < 0] = 0
        if invalid_items.any():
            code_indices[invalid_items] = 0

        code_embeddings = [
            emb(code_indices[..., idx]) for idx, emb in enumerate(self._codebook_embs)
        ]
        if self._compose == "concat_proj":
            out = self._project(torch.cat(code_embeddings, dim=-1))
        else:
            out = torch.stack(code_embeddings, dim=0).sum(dim=0)

        if invalid_items.any():
            out = out.clone()
            out[invalid_items] = 0.0
        return out

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class DenseFeatureEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        dense_feature_path: str,
        freeze_dense: bool = True,
    ) -> None:
        super().__init__()
        if not os.path.exists(dense_feature_path):
            raise FileNotFoundError(
                "dense_feature_path not found: "
                f"{dense_feature_path}\n"
                "Please export dense features first:\n"
                "  cd ../RQ-VAE-Recommender\n"
                "  make export_dense"
            )
        payload = torch.load(dense_feature_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("dense_feature_path must point to a dict artifact")
        item_ids = payload.get("item_ids")
        dense_vectors = payload.get("dense_vectors")
        if not torch.is_tensor(item_ids) or not torch.is_tensor(dense_vectors):
            raise ValueError("dense feature artifact must include tensor fields: item_ids, dense_vectors")
        if item_ids.dim() != 1:
            raise ValueError("item_ids must be a 1D tensor")
        if dense_vectors.dim() != 2:
            raise ValueError("dense_vectors must be a 2D tensor [N, D]")
        if item_ids.size(0) != dense_vectors.size(0):
            raise ValueError(
                f"item_ids and dense_vectors length mismatch: {item_ids.size(0)} != {dense_vectors.size(0)}"
            )

        self._item_embedding_dim = int(item_embedding_dim)
        self._num_items = int(num_items)
        dense_dim = int(dense_vectors.size(1))
        dense_table = torch.zeros((self._num_items + 1, dense_dim), dtype=torch.float32)
        known_item_mask = torch.zeros((self._num_items + 1,), dtype=torch.bool)

        item_ids = item_ids.long()
        dense_vectors = dense_vectors.float()
        valid = (item_ids >= 1) & (item_ids <= self._num_items)
        valid_item_ids = item_ids[valid]
        valid_vectors = dense_vectors[valid]
        dense_table[valid_item_ids] = valid_vectors
        known_item_mask[valid_item_ids] = True

        if freeze_dense:
            self.register_buffer("_dense_table", dense_table)
        else:
            self._dense_table = torch.nn.Parameter(dense_table)
        self.register_buffer("_known_item_mask", known_item_mask)

        self._project = torch.nn.Linear(dense_dim, self._item_embedding_dim)
        truncated_normal(self._project.weight, mean=0.0, std=0.02)
        if self._project.bias is not None:
            torch.nn.init.zeros_(self._project.bias)

        log.info(
            "Loaded dense feature artifact: "
            f"path={dense_feature_path}, dense_dim={dense_dim}, "
            f"known_items={int(known_item_mask.sum().item())}/{self._num_items}, "
            f"freeze_dense={freeze_dense}"
        )

    def debug_str(self) -> str:
        dense_dim = int(self._dense_table.size(1))
        return f"dense_feat_d{dense_dim}_to{self._item_embedding_dim}"

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = item_ids.long()
        item_ids_safe = item_ids.clamp(min=0, max=self._num_items)
        dense = self._dense_table[item_ids_safe]
        out = self._project(dense)
        invalid_items = (item_ids <= 0) | (~self._known_item_mask[item_ids_safe])
        if invalid_items.any():
            out = out.clone()
            out[invalid_items] = 0.0
        return out

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
