import pandas as pd

from generative_recommenders_pl.data.reco_dataset import RecoDataModule, RecoDataset


def test_reco_dataset_includes_target_train_count():
    frame = pd.DataFrame(
        {
            "user_id": [1, 2],
            "sequence_item_ids": ["[1, 2, 3]", "[4, 5, 6]"],
            "sequence_ratings": ["[5, 5, 5]", "[4, 4, 4]"],
            "sequence_timestamps": ["[1, 2, 3]", "[1, 2, 3]"],
        }
    )
    dataset = RecoDataset(
        ratings_file=frame,
        padding_length=5,
        ignore_last_n=0,
        target_train_counts={3: 7},
        split_name="val",
    )

    sample0 = dataset[0]
    sample1 = dataset[1]
    assert sample0["target_ids"] == 3
    assert sample0["target_train_count"] == 7
    assert sample1["target_ids"] == 6
    assert sample1["target_train_count"] == 0


def test_target_train_count_matches_train_visible_rows():
    frame = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "sequence_item_ids": ["[1, 2]", "[3]", "[4, 5, 6]"],
            "sequence_ratings": ["[5, 5]", "[4]", "[4, 4, 4]"],
            "sequence_timestamps": ["[1, 2]", "[1]", "[1, 2, 3]"],
        }
    )

    dm = RecoDataModule.__new__(RecoDataModule)
    dm.train_dataset = type("DummyTrainDataset", (), {"_ignore_last_n": 0})()

    counts = dm._build_target_train_counts(frame, cold_item_ids=[2, 6])

    # row2 has len==1 and is not train-visible; cold ids are removed first.
    # row1 -> [1], dropped because len<2 after cold filtering.
    # row3 -> [4,5], counted.
    assert counts == {4: 1, 5: 1}


def test_strict_cold_items_are_removed_from_training_sequences():
    frame = pd.DataFrame(
        {
            "user_id": [1, 2],
            "sequence_item_ids": ["[1, 2, 3, 4]", "[8, 9]"],
            "sequence_ratings": ["[5, 5, 5, 5]", "[4, 4]"],
            "sequence_timestamps": ["[1, 2, 3, 4]", "[1, 2]"],
        }
    )
    dataset = RecoDataset(
        ratings_file=frame,
        padding_length=5,
        ignore_last_n=1,
        cold_item_ids=[2],
        split_name="train",
    )

    # Hold out the last event, remove cold item 2, and retain [1, 3].
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["target_ids"] == 3
    assert sample["historical_ids"][0].item() == 1
    assert 2 not in sample["historical_ids"].tolist()
