import pytest
import torch

from generative_recommenders_pl.models.metrics.retrieval import RetrievalMetrics


@pytest.fixture
def test_data():
    top_k_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    target_ids = torch.tensor([[2], [6], [3]])
    return top_k_ids, target_ids


def test_initialization():
    k = 3
    at_k_list = [1, 2, 3]
    metric = RetrievalMetrics(k=k, at_k_list=at_k_list)
    assert metric.k == k
    assert metric.at_k_list == at_k_list


def test_update_method(test_data):
    top_k_ids, target_ids = test_data
    metric = RetrievalMetrics(k=3, at_k_list=[1, 2, 3])
    metric.update(top_k_ids, target_ids)
    assert len(metric.top_k_ids) == 1
    assert len(metric.target_ids) == 1
    metric.update(top_k_ids, target_ids)
    assert len(metric.top_k_ids) == 2
    assert len(metric.target_ids) == 2


def test_compute_method(test_data):
    top_k_ids, target_ids = test_data
    metric = RetrievalMetrics(k=3, at_k_list=[1, 2, 3])
    metric.update(top_k_ids, target_ids)
    output = metric.compute()
    assert output["ndcg@1"] == pytest.approx(0.0, abs=5e-5)
    assert output["ndcg@2"] == pytest.approx(0.2103, abs=5e-5)
    assert output["ndcg@3"] == pytest.approx(0.3770, abs=5e-5)
    assert output["hr@1"] == pytest.approx(0.0, abs=5e-5)
    assert output["hr@2"] == pytest.approx(0.3333, abs=5e-5)
    assert output["hr@3"] == pytest.approx(0.6667, abs=5e-5)
    assert output["mrr"] == pytest.approx(0.3611, abs=5e-5)
    assert output["mrr_corrected"] == pytest.approx(0.2778, abs=5e-5)


def test_reset(test_data):
    top_k_ids, target_ids = test_data
    metric = RetrievalMetrics(k=3, at_k_list=[1, 2, 3])
    metric.update(top_k_ids, target_ids)
    metric.reset()
    assert metric.top_k_ids == []
    assert metric.target_ids == []


def test_auc_and_cold_auc():
    metric = RetrievalMetrics(k=3, at_k_list=[1, 2, 3])
    top_k_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    target_ids = torch.tensor([[2], [6]])
    target_is_cold = torch.tensor([0, 1], dtype=torch.bool)
    auc_ranks = torch.tensor([1.0, 4.0])
    auc_num_candidates = torch.tensor([5.0, 5.0])

    metric.update(
        top_k_ids=top_k_ids,
        target_ids=target_ids,
        target_is_cold=target_is_cold,
        auc_ranks=auc_ranks,
        auc_num_candidates=auc_num_candidates,
    )
    output = metric.compute()

    assert output["auc"] == pytest.approx(0.625, abs=1e-6)
    assert output["cold_auc"] == pytest.approx(0.25, abs=1e-6)


def test_zero_and_few_shot_bucket_metrics():
    metric = RetrievalMetrics(k=3, at_k_list=[1, 2, 3])
    top_k_ids = torch.tensor(
        [
            [5, 1, 2],  # rank=1 (zero-shot)
            [7, 2, 9],  # rank=2 (few-shot)
            [4, 8, 6],  # rank=3 (all only)
        ],
        dtype=torch.long,
    )
    target_ids = torch.tensor([[5], [2], [6]], dtype=torch.long)
    target_train_count = torch.tensor([0, 3, 10], dtype=torch.long)
    metric.update(
        top_k_ids=top_k_ids,
        target_ids=target_ids,
        target_train_count=target_train_count,
    )
    output = metric.compute()

    assert output["zero_hr@1"] == pytest.approx(1.0, abs=1e-6)
    assert output["zero_hr@2"] == pytest.approx(1.0, abs=1e-6)
    assert output["zero_ndcg@1"] == pytest.approx(1.0, abs=1e-6)

    assert output["few_hr@1"] == pytest.approx(0.0, abs=1e-6)
    assert output["few_hr@2"] == pytest.approx(1.0, abs=1e-6)
    assert output["few_hr@3"] == pytest.approx(1.0, abs=1e-6)
    assert output["few_ndcg@2"] > 0.0
