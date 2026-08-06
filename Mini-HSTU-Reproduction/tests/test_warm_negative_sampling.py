import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from generative_recommenders_pl.data.reco_dataset import RecoDataModule
from generative_recommenders_pl.models.indexing.candidate_index import CandidateIndex
from generative_recommenders_pl.models.negatives_samples.negative_sampler import (
    LocalNegativesSampler,
)


class _Preprocessor:
    def __init__(self, ratings_path: Path):
        self._ratings_path = str(ratings_path)

    def output_format_csv(self):
        return self._ratings_path


def _make_datamodule(tmp_path: Path, *, exclude_cold: bool, all_ids=None):
    frame = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "sequence_item_ids": ["[1, 2, 3, 4]", "[2, 4, 5, 6]", "[1, 3, 5, 6]"],
            "sequence_ratings": ["[5, 5, 5, 5]"] * 3,
            "sequence_timestamps": ["[1, 2, 3, 4]"] * 3,
        }
    )
    ratings_path = tmp_path / "ratings.csv"
    frame.to_csv(ratings_path, index=False)

    dm = RecoDataModule.__new__(RecoDataModule)
    dm.dataset_name = "ml-1m"
    dm.data_preprocessor = _Preprocessor(ratings_path)
    dm.train_dataset = OmegaConf.create({"ignore_last_n": 1})
    dm.val_dataset = OmegaConf.create({"ignore_last_n": 0})
    dm.test_dataset = OmegaConf.create({"ignore_last_n": 0})
    dm.max_sequence_length = 10
    dm.chronological = True
    dm.positional_sampling_ratio = 1.0
    dm.batch_size = 2
    dm.num_workers = 0
    dm.prefetch = None
    dm.cold_start = OmegaConf.create(
        {
            "enabled": True,
            "cold_item_ratio": 0.5,
            "cold_split_seed": 42,
            "exclude_cold_from_train_negatives": exclude_cold,
            "cold_split_path": str(tmp_path / "cold_start_split.json"),
        }
    )
    dm.all_item_ids = list(all_ids if all_ids is not None else range(1, 7))
    dm.max_item_id = max(dm.all_item_ids, default=0)
    dm.cold_item_ids = []
    dm.target_train_counts = {}
    dm._cold_split_initialized = False
    dm._cold_ratings_frame = None
    return dm


def test_legacy_negative_pool_keeps_cold_items(tmp_path):
    dm = _make_datamodule(tmp_path, exclude_cold=False)
    dm.initialize_cold_start_split()

    assert dm.training_negative_item_ids == dm.all_item_ids
    assert set(dm.cold_item_ids).issubset(dm.training_negative_item_ids)


def test_warm_negative_pool_excludes_cold_and_preserves_order(tmp_path):
    dm = _make_datamodule(tmp_path, exclude_cold=True)
    dm.initialize_cold_start_split()

    expected = [item_id for item_id in dm.all_item_ids if item_id not in dm.cold_item_ids]
    assert dm.training_negative_item_ids == expected
    assert not set(dm.cold_item_ids).intersection(dm.training_negative_item_ids)


def test_local_sampler_never_draws_cold_but_eval_candidates_keep_them(tmp_path):
    dm = _make_datamodule(tmp_path, exclude_cold=True)
    # Accessing the model-facing property must initialize the split before setup().
    pool = dm.training_negative_item_ids
    sampler = LocalNegativesSampler(
        l2_norm=False,
        l2_norm_eps=1e-6,
        all_item_ids=pool,
    )
    sampler._item_emb = torch.nn.Embedding(dm.max_item_id + 1, 4, padding_idx=0)
    sampled_ids, _ = sampler(
        positive_ids=torch.ones(256, dtype=torch.long),
        num_to_sample=32,
    )
    assert not set(sampled_ids.flatten().tolist()).intersection(dm.cold_item_ids)

    candidate_index = CandidateIndex(
        k=100,
        ids=dm.all_item_ids,
        top_k_module=None,
    )
    assert set(dm.cold_item_ids).issubset(candidate_index.ids.flatten().tolist())


def test_split_initialization_and_setup_are_idempotent(tmp_path):
    dm = _make_datamodule(tmp_path, exclude_cold=True)
    first = list(dm.initialize_cold_start_split())
    split_before = Path(dm.cold_split_path).read_text()

    assert dm.initialize_cold_start_split() == first
    dm.setup(stage="validate")
    dm.setup(stage="validate")
    assert dm.cold_item_ids == first
    assert Path(dm.cold_split_path).read_text() == split_before


def test_existing_split_mismatch_is_rejected_without_overwrite(tmp_path):
    dm = _make_datamodule(tmp_path, exclude_cold=True)
    split_path = Path(dm.cold_split_path)
    invalid = {
        "cold_item_ratio": 0.5,
        "cold_split_seed": 42,
        "cold_item_ids": [999],
        "num_cold_items": 1,
        "num_observed_items": 6,
    }
    split_path.write_text(json.dumps(invalid))

    with pytest.raises(ValueError, match="refusing to overwrite"):
        dm.initialize_cold_start_split()
    assert json.loads(split_path.read_text()) == invalid


def test_empty_warm_negative_pool_fails_early(tmp_path):
    dm = _make_datamodule(tmp_path, exclude_cold=True, all_ids=[1, 2])
    dm.cold_start.cold_item_ratio = 0.999

    with pytest.raises(ValueError, match="negative item pool is empty"):
        dm.initialize_cold_start_split()


def test_audit_proves_zero_training_exposure_and_full_eval_candidates(tmp_path):
    dm = _make_datamodule(tmp_path, exclude_cold=True)
    audit = dm.build_negative_sampling_audit()

    assert audit["audit_passed"] is True
    assert audit["cold_items_in_training_histories"] == 0
    assert audit["cold_items_in_training_targets"] == 0
    assert audit["cold_items_in_training_negative_pool"] == 0
    assert audit["cold_items_in_eval_candidates"] == audit["num_cold_items"]
    assert len(audit["training_negative_item_ids_sha256"]) == 64
