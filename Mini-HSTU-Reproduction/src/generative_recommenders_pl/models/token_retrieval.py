from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from generative_recommenders_pl.models.generative_recommenders import GenerativeRecommenders
from generative_recommenders_pl.models.utils import ops
from generative_recommenders_pl.models.utils.features import (
    SequentialFeatures,
    seq_features_from_row,
)
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)


class HierarchicalTokenRetrieval(GenerativeRecommenders):
    def __init__(
        self,
        semantic_bridge_path: str,
        num_codebooks: Optional[int] = None,
        codebook_size: Optional[int] = None,
        beam_size: int = 8,
        decode_method: str = "hier_beam",
        prefix_constraint: bool = True,
        freeze_codebook: bool = False,
        compute_full_auc: bool = True,
        exact_auc_chunk_size: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if decode_method != "hier_beam":
            raise ValueError("Only decode_method='hier_beam' is supported")

        if not os.path.exists(semantic_bridge_path):
            raise FileNotFoundError(
                "semantic_bridge_path not found: "
                f"{semantic_bridge_path}\n"
                "Please generate bridge artifact first:\n"
                "  cd ../RQ-VAE-Recommender\n"
                "  make train_rqvae config=configs/rqvae_ml1m.gin"
            )
        payload = torch.load(semantic_bridge_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("semantic_bridge_path must point to exported bridge dict")

        item_to_codes = payload.get("item_id_to_codes")
        codebook_vectors = payload.get("codebook_vectors")
        bridge_meta = payload.get("metadata", {})
        if item_to_codes is None or codebook_vectors is None:
            raise ValueError(
                "semantic bridge must include item_id_to_codes and codebook_vectors"
            )

        item_to_codes = item_to_codes.long()
        codebook_vectors = codebook_vectors.float()
        bridge_num_codebooks = int(bridge_meta.get("n_codebooks", item_to_codes.size(1)))
        bridge_codebook_size = int(bridge_meta.get("codebook_size", codebook_vectors.size(1)))
        resolved_num_codebooks = bridge_num_codebooks if num_codebooks is None else int(num_codebooks)
        resolved_codebook_size = bridge_codebook_size if codebook_size is None else int(codebook_size)
        if num_codebooks is not None and int(num_codebooks) != bridge_num_codebooks:
            raise ValueError(
                f"num_codebooks mismatch: config={num_codebooks}, bridge={bridge_num_codebooks}"
            )
        if codebook_size is not None and int(codebook_size) != bridge_codebook_size:
            raise ValueError(
                f"codebook_size mismatch: config={codebook_size}, bridge={bridge_codebook_size}"
            )
        if item_to_codes.size(1) < resolved_num_codebooks:
            raise ValueError("item_to_codes has fewer codebooks than requested")
        if codebook_vectors.size(0) < resolved_num_codebooks:
            raise ValueError("codebook_vectors has fewer codebooks than requested")
        if codebook_vectors.size(1) != resolved_codebook_size:
            raise ValueError(
                f"codebook_vectors second dim {codebook_vectors.size(1)} "
                f"!= resolved codebook_size {resolved_codebook_size}"
            )

        self.num_codebooks = resolved_num_codebooks
        self.codebook_size = resolved_codebook_size
        self.beam_size = beam_size
        self.decode_method = decode_method
        self.prefix_constraint = prefix_constraint
        self.compute_full_auc = bool(compute_full_auc)
        self.exact_auc_chunk_size = max(1, int(exact_auc_chunk_size))
        self._decode_audit = {"rows": 0, "unique_before_fill": 0, "filled": 0}
        self._last_decode_unique_before_fill = 0
        self._last_decode_filled_count = 0

        self.register_buffer("item_to_codes", item_to_codes[:, :resolved_num_codebooks])

        codebook_dim = int(codebook_vectors.size(-1))
        self.code_embs = torch.nn.ModuleList(
            [
                torch.nn.Embedding(resolved_codebook_size + 1, codebook_dim, padding_idx=0)
                for _ in range(resolved_num_codebooks)
            ]
        )
        for i, emb in enumerate(self.code_embs):
            emb.weight.data.zero_()
            emb.weight.data[1 : resolved_codebook_size + 1] = codebook_vectors[
                i, :resolved_codebook_size
            ].to(emb.weight.dtype)
            emb.weight.requires_grad = not freeze_codebook

        self.code_context_proj = torch.nn.Linear(codebook_dim, self.item_embedding_dim)
        self.code_heads = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.item_embedding_dim, resolved_codebook_size)
                for _ in range(resolved_num_codebooks)
            ]
        )

        self._prefix_sets: List[set[Tuple[int, ...]]] = [set() for _ in range(resolved_num_codebooks)]
        self._code_to_items: Dict[Tuple[int, ...], List[int]] = {}
        for item_id in range(1, self.item_to_codes.size(0)):
            codes = tuple(self.item_to_codes[item_id].tolist())
            if min(codes) < 0:
                continue
            for idx in range(resolved_num_codebooks):
                self._prefix_sets[idx].add(codes[: idx + 1])
            self._code_to_items.setdefault(codes, []).append(item_id)

        valid_item_ids = [iid for iid in range(1, self.item_to_codes.size(0)) if int(self.item_to_codes[iid].min().item()) >= 0]
        self.register_buffer(
            "_valid_item_ids",
            torch.tensor(valid_item_ids, dtype=torch.long),
        )
        self.register_buffer(
            "_valid_item_codes",
            self.item_to_codes[self._valid_item_ids],
        )

        log.info(
            "Loaded token bridge: "
            f"path={semantic_bridge_path}, num_codebooks={self.num_codebooks}, "
            f"codebook_size={self.codebook_size}, valid_items={len(valid_item_ids)}, "
            f"max_item_id={self.item_to_codes.size(0) - 1}"
        )

    def _step_code_context(self, level: int, code_ids: torch.Tensor) -> torch.Tensor:
        code_ids = code_ids.long().clamp(min=-1, max=self.codebook_size - 1)
        indices = code_ids + 1
        indices[code_ids < 0] = 0
        emb = self.code_embs[level](indices)
        out = self.code_context_proj(emb)
        out[code_ids < 0] = 0.0
        return out

    @torch.inference_mode()
    def _decode_single(
        self,
        context: torch.Tensor,
        invalid_ids: set[int],
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beams = [((), 0.0, torch.zeros_like(context))]

        for level in range(self.num_codebooks):
            candidates = []
            for prefix, score, prefix_ctx in beams:
                logits = self.code_heads[level]((context + prefix_ctx).unsqueeze(0)).squeeze(0)
                log_probs = torch.log_softmax(logits, dim=-1)
                top_vals, top_ids = torch.topk(
                    log_probs,
                    k=min(self.codebook_size, self.beam_size * 4),
                    largest=True,
                    sorted=True,
                )
                for val, code in zip(top_vals.tolist(), top_ids.tolist()):
                    next_prefix = prefix + (int(code),)
                    if self.prefix_constraint and next_prefix not in self._prefix_sets[level]:
                        continue
                    code_tensor = torch.tensor([code], device=context.device)
                    next_ctx = prefix_ctx + self._step_code_context(level, code_tensor).squeeze(0)
                    candidates.append((next_prefix, score + float(val), next_ctx))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[: self.beam_size]

        scored_items: Dict[int, float] = {}
        for prefix, score, _ in beams:
            items = self._code_to_items.get(tuple(prefix), [])
            for item_id in items:
                if item_id in invalid_ids:
                    continue
                scored_items[item_id] = max(scored_items.get(item_id, -1e9), score)

        unique_before_fill = len(scored_items)
        filled_count = 0
        if unique_before_fill < k:
            exact_scores, exact_item_ids = self._exact_token_item_scores(context.unsqueeze(0))
            if exact_item_ids.numel() > 0:
                flat_scores = exact_scores.squeeze(0)
                rank_indices = torch.argsort(flat_scores, descending=True)
                for idx in rank_indices.tolist():
                    score = float(flat_scores[idx].item())
                    if not torch.isfinite(flat_scores[idx]):
                        continue
                    item_id = int(exact_item_ids[idx].item())
                    if item_id in invalid_ids or item_id in scored_items:
                        continue
                    scored_items[item_id] = score
                    filled_count += 1
                    if len(scored_items) >= k:
                        break

        if not scored_items:
            fallback = []
            seen = set()
            for item_id in self._code_to_items.values():
                for iid in item_id:
                    if iid not in invalid_ids and iid not in seen:
                        seen.add(iid)
                        fallback.append(iid)
            fallback = fallback[:k]
            if not fallback:
                self._last_decode_unique_before_fill = unique_before_fill
                self._last_decode_filled_count = filled_count
                return (
                    torch.zeros((k,), dtype=torch.long, device=context.device),
                    torch.full((k,), -1e9, dtype=torch.float32, device=context.device),
                )
            ids = torch.tensor(fallback, dtype=torch.long, device=context.device)
            scores = torch.linspace(0, -1, steps=ids.numel(), device=context.device)
        else:
            sorted_items = sorted(scored_items.items(), key=lambda x: x[1], reverse=True)[:k]
            ids = torch.tensor([x[0] for x in sorted_items], dtype=torch.long, device=context.device)
            scores = torch.tensor([x[1] for x in sorted_items], dtype=torch.float32, device=context.device)

        if ids.numel() < k:
            pad = k - ids.numel()
            ids = torch.cat([ids, torch.zeros(pad, dtype=torch.long, device=context.device)])
            scores = torch.cat(
                [scores, torch.full((pad,), -1e9, dtype=torch.float32, device=context.device)]
            )
        self._last_decode_unique_before_fill = unique_before_fill
        self._last_decode_filled_count = filled_count
        return ids, scores

    @torch.inference_mode()
    def _retrieve_from_context_embeddings(
        self,
        current_embeddings: torch.Tensor,
        past_ids: torch.Tensor,
        filter_past_ids: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.candidate_index._k if hasattr(self.candidate_index, "_k") else 200
        top_ids, top_scores = [], []
        unique_before_fill_sum = 0
        filled_sum = 0
        for row_idx in range(current_embeddings.size(0)):
            invalid_ids = set()
            if filter_past_ids:
                invalid_ids = set(past_ids[row_idx][past_ids[row_idx] > 0].tolist())
            ids, scores = self._decode_single(current_embeddings[row_idx], invalid_ids, k)
            top_ids.append(ids)
            top_scores.append(scores)
            unique_before_fill_sum += int(self._last_decode_unique_before_fill)
            filled_sum += int(self._last_decode_filled_count)
        self._decode_audit["rows"] += int(current_embeddings.size(0))
        self._decode_audit["unique_before_fill"] += int(unique_before_fill_sum)
        self._decode_audit["filled"] += int(filled_sum)
        return torch.stack(top_ids, dim=0), torch.stack(top_scores, dim=0)

    @torch.inference_mode()
    def retrieve(
        self,
        seq_features: SequentialFeatures,
        filter_past_ids: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_embeddings, _ = self.forward(seq_features)
        current_embeddings = ops.get_current_embeddings(seq_features.past_lengths, seq_embeddings)
        return self._retrieve_from_context_embeddings(
            current_embeddings=current_embeddings,
            past_ids=seq_features.past_ids,
            filter_past_ids=filter_past_ids,
        )

    def _hierarchical_loss(self, context_embeddings: torch.Tensor, target_codes: torch.Tensor) -> torch.Tensor:
        prefix_ctx = torch.zeros_like(context_embeddings)
        losses = []
        for level in range(self.num_codebooks):
            logits = self.code_heads[level](context_embeddings + prefix_ctx)
            targets = target_codes[:, level]
            valid = targets >= 0
            if valid.any():
                losses.append(F.cross_entropy(logits[valid], targets[valid]))
            prefix_ctx = prefix_ctx + self._step_code_context(level, targets)

        if not losses:
            return torch.zeros((), device=context_embeddings.device, requires_grad=True)
        return torch.stack(losses).mean()

    @torch.inference_mode()
    def _exact_token_item_scores(self, context_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute exact item log-prob scores for all valid coded items.
        Returns:
            scores: [B, N_valid_items]
            item_ids: [N_valid_items]
        """
        if self._valid_item_ids.numel() == 0:
            return (
                torch.empty(
                    context_embeddings.size(0), 0, dtype=torch.float32, device=context_embeddings.device
                ),
                self._valid_item_ids,
            )

        item_codes = self._valid_item_codes.to(context_embeddings.device)
        item_ids = self._valid_item_ids.to(context_embeddings.device)
        first_codes = item_codes[:, 0]
        logp0 = torch.log_softmax(self.code_heads[0](context_embeddings), dim=-1)
        scores = logp0[:, first_codes]
        prefix_by_item = torch.zeros(
            item_ids.numel(),
            context_embeddings.size(-1),
            device=context_embeddings.device,
        )

        chunk_size = self.exact_auc_chunk_size
        for level in range(1, self.num_codebooks):
            prev_code_ids = item_codes[:, level - 1]
            ctx_table = self._step_code_context(
                level - 1,
                torch.arange(self.codebook_size, device=context_embeddings.device),
            )
            prefix_by_item = prefix_by_item + ctx_table[prev_code_ids]
            code_targets = item_codes[:, level]

            for start in range(0, item_ids.numel(), chunk_size):
                end = min(start + chunk_size, item_ids.numel())
                prefix_chunk = prefix_by_item[start:end]
                target_chunk = code_targets[start:end]

                level_ctx = context_embeddings.unsqueeze(1) + prefix_chunk.unsqueeze(0)
                logits = self.code_heads[level](
                    level_ctx.reshape(-1, level_ctx.size(-1))
                ).view(context_embeddings.size(0), end - start, self.codebook_size)
                logp = torch.log_softmax(logits, dim=-1)
                gather_target = target_chunk.view(1, -1, 1).expand(
                    context_embeddings.size(0), -1, 1
                )
                scores[:, start:end] = scores[:, start:end] + logp.gather(
                    2, gather_target
                ).squeeze(2)

        return scores, item_ids

    @torch.inference_mode()
    def _compute_full_auc_rank(
        self,
        seq_features: SequentialFeatures,
        target_ids: torch.Tensor,
        context_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_ids = self.candidate_index.ids.squeeze(0).long().to(self.device)
        id_to_col = torch.full(
            (int(candidate_ids.max().item()) + 1,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        id_to_col[candidate_ids] = torch.arange(candidate_ids.size(0), device=self.device)

        full_scores = torch.full(
            (context_embeddings.size(0), candidate_ids.size(0)),
            float("-inf"),
            dtype=torch.float32,
            device=self.device,
        )
        valid_item_scores, valid_item_ids = self._exact_token_item_scores(context_embeddings)
        if valid_item_ids.numel() > 0:
            valid_cols = id_to_col[valid_item_ids]
            valid_mask = valid_cols >= 0
            if valid_mask.any():
                full_scores[:, valid_cols[valid_mask]] = valid_item_scores[:, valid_mask]

        past_ids = seq_features.past_ids.long().clamp(min=0, max=id_to_col.size(0) - 1)
        past_cols = id_to_col[past_ids]
        row_idx = torch.arange(past_cols.size(0), device=self.device).unsqueeze(1).expand_as(
            past_cols
        )
        valid_past = past_cols >= 0
        if valid_past.any():
            full_scores[row_idx[valid_past], past_cols[valid_past]] = float("-inf")

        target_ids = target_ids.view(-1).long().clamp(min=0, max=id_to_col.size(0) - 1)
        target_cols = id_to_col[target_ids]
        target_valid = target_cols >= 0
        safe_target_cols = target_cols.clone()
        safe_target_cols[~target_valid] = 0

        target_scores = full_scores.gather(1, safe_target_cols.unsqueeze(1)).squeeze(1)
        tie_counts = (full_scores == target_scores.unsqueeze(1)).sum(dim=1).float()
        better_counts = (full_scores > target_scores.unsqueeze(1)).sum(dim=1).float()
        auc_ranks = 1.0 + better_counts + 0.5 * (tie_counts - 1.0)
        num_candidates = torch.isfinite(full_scores).sum(dim=1).to(torch.float32)
        auc_ranks = torch.where(target_valid, auc_ranks, num_candidates)
        return auc_ranks, num_candidates

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        seq_features, target_ids, _ = seq_features_from_row(
            batch,
            device=self.device,
            max_output_length=self.gr_output_length + 1,
        )

        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        seq_features = seq_features._replace(past_embeddings=input_embeddings)

        seq_embeddings, _ = self.forward(seq_features)
        current_embeddings = ops.get_current_embeddings(seq_features.past_lengths, seq_embeddings)

        target_ids = target_ids.view(-1).clamp(min=0, max=self.item_to_codes.size(0) - 1)
        target_codes = self.item_to_codes[target_ids]
        loss = self._hierarchical_loss(current_embeddings, target_codes)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.metrics.reset()
        self._decode_audit = {"rows": 0, "unique_before_fill": 0, "filled": 0}

    def validation_step(
        self, batch: tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        seq_features, target_ids, _ = seq_features_from_row(
            batch,
            device=self.device,
            max_output_length=self.gr_output_length + 1,
        )
        target_is_cold = batch.get("target_is_cold")
        if target_is_cold is not None:
            target_is_cold = target_is_cold.to(self.device).view(-1)
        target_train_count = batch.get("target_train_count")
        if target_train_count is not None:
            target_train_count = target_train_count.to(self.device).view(-1)

        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        seq_features = seq_features._replace(past_embeddings=input_embeddings)

        seq_embeddings, _ = self.forward(seq_features)
        current_embeddings = ops.get_current_embeddings(seq_features.past_lengths, seq_embeddings)
        top_k_ids, top_k_scores = self._retrieve_from_context_embeddings(
            current_embeddings=current_embeddings,
            past_ids=seq_features.past_ids,
            filter_past_ids=True,
        )
        auc_ranks, auc_num_candidates = None, None
        if self.compute_full_auc:
            auc_ranks, auc_num_candidates = self._compute_full_auc_rank(
                seq_features=seq_features,
                target_ids=target_ids,
                context_embeddings=current_embeddings,
            )
        self.metrics.update(
            top_k_ids=top_k_ids,
            target_ids=target_ids,
            target_is_cold=target_is_cold,
            target_train_count=target_train_count,
            auc_ranks=auc_ranks,
            auc_num_candidates=auc_num_candidates,
        )

    def on_validation_epoch_end(self) -> None:
        results = self.metrics.compute()
        for k, v in results.items():
            self.log(f"val/{k}", v, on_epoch=True, prog_bar=True, logger=True)
        rows = max(1, int(self._decode_audit["rows"]))
        self.log(
            "val/decoded_unique_before_fill",
            float(self._decode_audit["unique_before_fill"]) / rows,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val/decoded_filled_count",
            float(self._decode_audit["filled"]) / rows,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.metrics.reset()
        if "monitor" in self.configure_optimizer_params:
            return results[self.configure_optimizer_params["monitor"].split("/")[1]]

    def on_test_epoch_start(self) -> None:
        self.metrics.reset()
        self._decode_audit = {"rows": 0, "unique_before_fill": 0, "filled": 0}

    def test_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        results = self.metrics.compute()
        for k, v in results.items():
            self.log(f"test/{k}", v, on_epoch=True, prog_bar=True, logger=True)
        rows = max(1, int(self._decode_audit["rows"]))
        self.log(
            "test/decoded_unique_before_fill",
            float(self._decode_audit["unique_before_fill"]) / rows,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "test/decoded_filled_count",
            float(self._decode_audit["filled"]) / rows,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.metrics.reset()
        if "monitor" in self.configure_optimizer_params:
            return results[self.configure_optimizer_params["monitor"].split("/")[1]]

    def predict_step(
        self, batch: tuple[torch.Tensor], batch_idx: int
    ) -> dict[str, list]:
        seq_features, _, _ = seq_features_from_row(
            batch,
            device=self.device,
            max_output_length=self.gr_output_length + 1,
        )

        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        seq_features = seq_features._replace(past_embeddings=input_embeddings)

        top_k_ids, top_k_scores = self.retrieve(seq_features)
        return {
            "top_k_ids": top_k_ids.cpu().numpy().tolist(),
            "top_k_scores": top_k_scores.cpu().numpy().tolist(),
        }

    def on_predict_epoch_end(self) -> None:
        for i, predictions in enumerate(self.trainer.predict_loop._predictions):
            if predictions and isinstance(predictions[0], dict):
                keys = predictions[0].keys()
                converted_predictions = {
                    key: sum((pred[key] for pred in predictions), []) for key in keys
                }
                self.trainer.predict_loop._predictions[i] = converted_predictions
