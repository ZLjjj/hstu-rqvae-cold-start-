import torch
import torchmetrics
import torchmetrics.utilities


class RetrievalMetrics(torchmetrics.Metric):
    """
    A metric class for computing various retrieval metrics.

    This class calculates NDCG (Normalized Discounted Cumulative Gain), HR (Hit Rate),
    and MRR (Mean Reciprocal Rank) for a given set of top-k predictions and target IDs.

    Args:
        k (int): The number of top predictions to consider.
        at_k_list (list[int]): List of k values for which to compute NDCG and HR.
        **kwargs: Additional keyword arguments to pass to the parent Metric class.

    Attributes:
        k (int): The number of top predictions to consider.
        at_k_list (list[int]): List of k values for NDCG and HR computation.
        top_k_ids (list): State to store top-k prediction IDs.
        target_ids (list): State to store target IDs.

    Methods:
        update(top_k_ids, target_ids): Update the metric states with new predictions and targets.
        compute(): Compute and return the retrieval metrics.
    """

    def __init__(self, k: int, at_k_list: list[int], **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.at_k_list = at_k_list
        self.add_state("top_k_ids", default=[], dist_reduce_fx="cat")
        self.add_state("target_ids", default=[], dist_reduce_fx="cat")
        self.add_state("target_is_cold", default=[], dist_reduce_fx="cat")
        self.add_state("target_train_count", default=[], dist_reduce_fx="cat")
        self.add_state("auc_ranks", default=[], dist_reduce_fx="cat")
        self.add_state("auc_num_candidates", default=[], dist_reduce_fx="cat")

    def update(
        self,
        top_k_ids: torch.Tensor,
        target_ids: torch.Tensor,
        target_is_cold: torch.Tensor = None,
        target_train_count: torch.Tensor = None,
        auc_ranks: torch.Tensor = None,
        auc_num_candidates: torch.Tensor = None,
        **kwargs,
    ):
        # Defensive device alignment: some retrieval paths may produce CPU ids while
        # targets are on CUDA (or vice versa). Keep metric states homogeneous.
        metric_device = target_ids.device
        if top_k_ids.device != metric_device:
            top_k_ids = top_k_ids.to(metric_device)
        if target_ids.device != metric_device:
            target_ids = target_ids.to(metric_device)

        self.top_k_ids.append(top_k_ids)
        self.target_ids.append(target_ids)
        if target_is_cold is not None:
            if target_is_cold.device != metric_device:
                target_is_cold = target_is_cold.to(metric_device)
            self.target_is_cold.append(target_is_cold)
        if target_train_count is not None:
            if target_train_count.device != metric_device:
                target_train_count = target_train_count.to(metric_device)
            self.target_train_count.append(target_train_count)
        if auc_ranks is not None:
            if auc_ranks.device != metric_device:
                auc_ranks = auc_ranks.to(metric_device)
            self.auc_ranks.append(auc_ranks.to(torch.float32))
            if auc_num_candidates is None:
                auc_num_candidates = torch.full_like(auc_ranks, top_k_ids.size(1))
            elif auc_num_candidates.device != metric_device:
                auc_num_candidates = auc_num_candidates.to(metric_device)
            self.auc_num_candidates.append(auc_num_candidates.to(torch.float32))

    def compute(self):
        # Concatenate the lists of tensors
        top_k_ids = torchmetrics.utilities.dim_zero_cat(self.top_k_ids)
        target_ids = torchmetrics.utilities.dim_zero_cat(self.target_ids)

        assert top_k_ids.size(1) == self.k
        _, rank_indices = torch.max(
            torch.cat(
                [top_k_ids, target_ids],
                dim=1,
            )
            == target_ids,
            dim=1,
        )
        ranks = (rank_indices + 1).to(torch.float32)
        output = {}
        # compute ndcg
        for at_k in self.at_k_list:
            output[f"ndcg@{at_k}"] = torch.where(
                ranks <= at_k,
                1.0 / torch.log2(ranks + 1),
                torch.zeros(1, dtype=torch.float32, device=ranks.device),
            ).mean()
        # compute recall / hit rate
        for at_k in self.at_k_list:
            output[f"hr@{at_k}"] = (ranks <= at_k).to(torch.float32).mean()
        # compute mrr
        output["mrr"] = (1.0 / ranks).mean()
        # Preserve the repository's legacy MRR above for result comparability.
        # A miss is represented by the appended target at rank k+1, so the
        # mathematically standard MRR must explicitly assign misses zero credit.
        output["mrr_corrected"] = torch.where(
            ranks <= self.k,
            1.0 / ranks,
            torch.zeros_like(ranks),
        ).mean()

        if len(self.target_train_count) > 0:
            target_train_count = torchmetrics.utilities.dim_zero_cat(
                self.target_train_count
            ).long().view(-1)
            zero_mask = target_train_count == 0
            few_mask = (target_train_count >= 1) & (target_train_count <= 5)

            def _compute_bucket_metrics(mask: torch.Tensor, prefix: str) -> None:
                if mask.any():
                    bucket_ranks = ranks[mask]
                    for at_k in self.at_k_list:
                        output[f"{prefix}_ndcg@{at_k}"] = torch.where(
                            bucket_ranks <= at_k,
                            1.0 / torch.log2(bucket_ranks + 1),
                            torch.zeros(1, dtype=torch.float32, device=bucket_ranks.device),
                        ).mean()
                    for at_k in self.at_k_list:
                        output[f"{prefix}_hr@{at_k}"] = (
                            bucket_ranks <= at_k
                        ).to(torch.float32).mean()
                else:
                    for at_k in self.at_k_list:
                        output[f"{prefix}_ndcg@{at_k}"] = torch.zeros(
                            1, dtype=torch.float32, device=ranks.device
                        ).squeeze(0)
                    for at_k in self.at_k_list:
                        output[f"{prefix}_hr@{at_k}"] = torch.zeros(
                            1, dtype=torch.float32, device=ranks.device
                        ).squeeze(0)

            _compute_bucket_metrics(zero_mask, "zero")
            _compute_bucket_metrics(few_mask, "few")

        if len(self.auc_ranks) > 0:
            auc_ranks = torchmetrics.utilities.dim_zero_cat(self.auc_ranks).to(torch.float32)
            auc_num_candidates = torchmetrics.utilities.dim_zero_cat(
                self.auc_num_candidates
            ).to(torch.float32)
        else:
            auc_ranks = ranks
            auc_num_candidates = torch.full_like(auc_ranks, float(top_k_ids.size(1)))

        valid_auc = auc_num_candidates > 1
        if valid_auc.any():
            auc_values = ((auc_num_candidates - auc_ranks) / (auc_num_candidates - 1)).clamp(
                min=0.0, max=1.0
            )
            output["auc"] = auc_values[valid_auc].mean()

        if len(self.target_is_cold) > 0:
            target_is_cold = torchmetrics.utilities.dim_zero_cat(self.target_is_cold).bool()
            if target_is_cold.any():
                cold_ranks = ranks[target_is_cold]
                for at_k in self.at_k_list:
                    output[f"cold_ndcg@{at_k}"] = torch.where(
                        cold_ranks <= at_k,
                        1.0 / torch.log2(cold_ranks + 1),
                        torch.zeros(1, dtype=torch.float32, device=cold_ranks.device),
                    ).mean()
                for at_k in self.at_k_list:
                    output[f"cold_hr@{at_k}"] = (cold_ranks <= at_k).to(torch.float32).mean()
                output["cold_mrr"] = (1.0 / cold_ranks).mean()
                output["cold_mrr_corrected"] = torch.where(
                    cold_ranks <= self.k,
                    1.0 / cold_ranks,
                    torch.zeros_like(cold_ranks),
                ).mean()
                if valid_auc.any():
                    cold_valid_auc = target_is_cold & valid_auc
                    if cold_valid_auc.any():
                        cold_auc_values = (
                            (auc_num_candidates - auc_ranks)
                            / (auc_num_candidates - 1)
                        ).clamp(min=0.0, max=1.0)
                        output["cold_auc"] = cold_auc_values[cold_valid_auc].mean()
            warm_mask = ~target_is_cold
            if warm_mask.any():
                warm_ranks = ranks[warm_mask]
                for at_k in self.at_k_list:
                    output[f"warm_ndcg@{at_k}"] = torch.where(
                        warm_ranks <= at_k,
                        1.0 / torch.log2(warm_ranks + 1),
                        torch.zeros(1, dtype=torch.float32, device=warm_ranks.device),
                    ).mean()
                    output[f"warm_hr@{at_k}"] = (
                        warm_ranks <= at_k
                    ).to(torch.float32).mean()
                output["warm_mrr"] = (1.0 / warm_ranks).mean()
                output["warm_mrr_corrected"] = torch.where(
                    warm_ranks <= self.k,
                    1.0 / warm_ranks,
                    torch.zeros_like(warm_ranks),
                ).mean()
                if valid_auc.any():
                    warm_valid_auc = warm_mask & valid_auc
                    if warm_valid_auc.any():
                        warm_auc_values = (
                            (auc_num_candidates - auc_ranks)
                            / (auc_num_candidates - 1)
                        ).clamp(min=0.0, max=1.0)
                        output["warm_auc"] = warm_auc_values[warm_valid_auc].mean()
        return output
