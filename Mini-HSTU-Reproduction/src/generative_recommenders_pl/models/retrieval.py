import torch

from generative_recommenders_pl.models.generative_recommenders import (
    GenerativeRecommenders,
)
from generative_recommenders_pl.models.negatives_samples.negative_sampler import (
    InBatchNegativesSampler,
)
from generative_recommenders_pl.models.utils import ops
from generative_recommenders_pl.models.utils.features import (
    SequentialFeatures,
    seq_features_from_row,
)
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)


class Retrieval(GenerativeRecommenders):
    def __init__(
        self,
        compute_full_auc: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.compute_full_auc = bool(compute_full_auc)

    @torch.inference_mode
    def _compute_full_auc_rank(
        self,
        seq_features: SequentialFeatures,
        target_ids: torch.Tensor,
        current_embeddings: torch.Tensor,
        filter_past_ids: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_ids = self.candidate_index.ids.squeeze(0).long().to(self.device)
        candidate_emb = self.candidate_index.embeddings.float().to(self.device)
        if candidate_emb.dim() == 3:
            if candidate_emb.size(0) != 1:
                raise ValueError(
                    f"Unexpected candidate embeddings shape: {tuple(candidate_emb.shape)}"
                )
            candidate_emb = candidate_emb.squeeze(0)
        if candidate_emb.dim() != 2:
            raise ValueError(
                f"Candidate embeddings must be 2D after squeeze, got shape={tuple(candidate_emb.shape)}"
            )

        full_scores = current_embeddings.float() @ candidate_emb.T

        id_to_col = torch.full(
            (int(candidate_ids.max().item()) + 1,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        id_to_col[candidate_ids] = torch.arange(candidate_ids.size(0), device=self.device)

        if filter_past_ids:
            past_ids = seq_features.past_ids.long().clamp(min=0, max=id_to_col.size(0) - 1)
            past_cols = id_to_col[past_ids]
            row_idx = torch.arange(past_cols.size(0), device=self.device).unsqueeze(1).expand_as(
                past_cols
            )
            valid = past_cols >= 0
            if valid.any():
                full_scores[row_idx[valid], past_cols[valid]] = float("-inf")

        target_ids = target_ids.view(-1).long().clamp(min=0, max=id_to_col.size(0) - 1)
        target_cols = id_to_col[target_ids]
        target_valid = target_cols >= 0

        safe_target_cols = target_cols.clone()
        safe_target_cols[~target_valid] = 0
        target_scores = full_scores.gather(1, safe_target_cols.unsqueeze(1)).squeeze(1)
        tie_counts = (full_scores == target_scores.unsqueeze(1)).sum(dim=1).float()
        better_counts = (full_scores > target_scores.unsqueeze(1)).sum(dim=1).float()
        auc_ranks = 1.0 + better_counts + 0.5 * (tie_counts - 1.0)

        # Unknown target id in candidate set -> worst rank.
        num_candidates = torch.isfinite(full_scores).sum(dim=1).to(torch.float32)
        auc_ranks = torch.where(target_valid, auc_ranks, num_candidates)
        return auc_ranks, num_candidates

    @torch.inference_mode
    def retrieve(
        self,
        seq_features: SequentialFeatures,
        filter_past_ids: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the top-k items for the given sequence features.
        """
        seq_embeddings, _ = self.forward(seq_features)  # [B, X]
        current_embeddings = ops.get_current_embeddings(
            seq_features.past_lengths, seq_embeddings
        )

        if self.candidate_index.embeddings is None:
            log.info(
                "Initializing candidate index embeddings with current item embeddings"
            )
            self.candidate_index.update_embeddings(
                self.negatives_sampler.normalize_embeddings(
                    self.embeddings.get_item_embeddings(self.candidate_index.ids)
                )
            )

        top_k_ids, top_k_scores = self.candidate_index.get_top_k_outputs(
            query_embeddings=current_embeddings,
            invalid_ids=(seq_features.past_ids if filter_past_ids else None),
        )
        return top_k_ids, top_k_scores

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning calls this inside the training loop.

        Args:
            batch (tuple[torch.Tensor]): A tuple containing the input and target
                tensors.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss tensor.
        """
        # convert the batch to the sequence features (TODO: move to datamodule)
        seq_features, target_ids, target_ratings = seq_features_from_row(
            batch,
            device=self.device,
            max_output_length=self.gr_output_length + 1,
        )
        # add target_ids at the end of the past_ids
        seq_features.past_ids.scatter_(
            dim=1,
            index=seq_features.past_lengths.view(-1, 1),
            src=target_ids.view(-1, 1),
        )

        # embeddings
        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        # TODO: think a better way than replace, since it creates a new instance
        seq_features = seq_features._replace(past_embeddings=input_embeddings)

        # forward pass
        seq_embeddings, _ = self.forward(seq_features)  # [B, X]

        # prepare loss
        supervision_ids = seq_features.past_ids

        # negative sampling
        if isinstance(self.negatives_sampler, InBatchNegativesSampler):
            # get_item_embeddings currently assume 1-d tensor.
            in_batch_ids = supervision_ids.view(-1)
            self.negatives_sampler.process_batch(
                ids=in_batch_ids,
                presences=(in_batch_ids != 0),
                embeddings=self.embeddings.get_item_embeddings(in_batch_ids),
            )
        else:
            # update embedding in the local negative sampler
            item_emb = getattr(self.embeddings, "_item_emb", None)
            self.negatives_sampler._item_emb = (
                item_emb if item_emb is not None else self.embeddings.get_item_embeddings
            )

        # dense features to jagged features
        # TODO: seems that the target_ids is not used in the loss
        jagged_features = self.dense_to_jagged(
            lengths=seq_features.past_lengths,
            output_embeddings=seq_embeddings[:, :-1, :],  # [B, N-1, D]
            supervision_ids=supervision_ids[:, 1:],  # [B, N-1]
            supervision_embeddings=input_embeddings[:, 1:, :],  # [B, N - 1, D]
            supervision_weights=(supervision_ids[:, 1:] != 0).float(),  # ar_mask
        )

        loss = self.loss.jagged_forward(
            negatives_sampler=self.negatives_sampler,
            similarity=self.similarity,
            **jagged_features,
        )

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        """Lightning calls this at the beginning of the validation epoch."""
        self.metrics.reset()
        self.candidate_index.update_embeddings(
            self.negatives_sampler.normalize_embeddings(
                self.embeddings.get_item_embeddings(self.candidate_index.ids)
            )
        )

    def validation_step(
        self, batch: tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Lightning calls this inside the validation loop.

        Args:
            batch (tuple[torch.Tensor]): A tuple containing the input and target
                tensors.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss tensor.
        """
        # convert the batch to the sequence features (TODO: move to datamodule)
        seq_features, target_ids, target_ratings = seq_features_from_row(
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

        # embeddings
        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        # TODO: think a better way than replace, since it creates a new instance
        seq_features = seq_features._replace(past_embeddings=input_embeddings)

        # forward pass
        top_k_ids, top_k_scores = self.retrieve(seq_features)
        auc_ranks, auc_num_candidates = None, None
        if self.compute_full_auc:
            seq_embeddings, _ = self.forward(seq_features)
            current_embeddings = ops.get_current_embeddings(
                seq_features.past_lengths, seq_embeddings
            )
            auc_ranks, auc_num_candidates = self._compute_full_auc_rank(
                seq_features=seq_features,
                target_ids=target_ids,
                current_embeddings=current_embeddings,
                filter_past_ids=True,
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
        """Lightning calls this at the end of the validation epoch.

        Args:
            outputs (list[torch.Tensor]): A list of the outputs from each validation step.
        """
        results = self.metrics.compute()
        for k, v in results.items():
            self.log(f"val/{k}", v, on_epoch=True, prog_bar=True, logger=True)
        self.metrics.reset()
        if "monitor" in self.configure_optimizer_params:
            return results[self.configure_optimizer_params["monitor"].split("/")[1]]

    def on_test_epoch_start(self) -> None:
        """Lightning calls this at the beginning of the test epoch."""
        self.metrics.reset()
        self.candidate_index.update_embeddings(
            self.negatives_sampler.normalize_embeddings(
                self.embeddings.get_item_embeddings(self.candidate_index.ids)
            )
        )

    def test_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning calls this inside the test loop.

        Args:
            batch (tuple[torch.Tensor]): A tuple containing the input and target
                tensors.
            batch_idx (int): The index of the batch.
        """
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        """Lightning calls this at the end of the test epoch.

        Args:
            outputs (list[torch.Tensor]): A list of the outputs from each test step.
        """
        results = self.metrics.compute()
        for k, v in results.items():
            self.log(f"test/{k}", v, on_epoch=True, prog_bar=True, logger=True)
        self.metrics.reset()
        if "monitor" in self.configure_optimizer_params:
            return results[self.configure_optimizer_params["monitor"].split("/")[1]]

    def on_predict_epoch_start(self) -> None:
        """Lightning calls this at the beginning of the predict epoch."""
        self.candidate_index.update_embeddings(
            self.negatives_sampler.normalize_embeddings(
                self.embeddings.get_item_embeddings(self.candidate_index.ids)
            )
        )

    def predict_step(
        self, batch: tuple[torch.Tensor], batch_idx: int
    ) -> dict[str, list]:
        """Lightning calls this inside the predict loop."""
        seq_features, _, _ = seq_features_from_row(
            batch,
            device=self.device,
            max_output_length=self.gr_output_length + 1,
        )

        # embeddings
        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        # TODO: think a better way than replace, since it creates a new instance
        seq_features = seq_features._replace(past_embeddings=input_embeddings)

        top_k_ids, top_k_scores = self.retrieve(seq_features)
        return {
            "top_k_ids": top_k_ids.cpu().numpy().tolist(),
            "top_k_scores": top_k_scores.cpu().numpy().tolist(),
        }

    def on_predict_epoch_end(self) -> None:
        """Lightning calls this at the end of the predict epoch."""
        # Convert predictions from list of dicts to dict of lists
        for i, predictions in enumerate(self.trainer.predict_loop._predictions):
            if predictions and isinstance(predictions[0], dict):
                keys = predictions[0].keys()
                converted_predictions = {
                    key: sum((pred[key] for pred in predictions), []) for key in keys
                }
                self.trainer.predict_loop._predictions[i] = converted_predictions
