import ast
import hashlib
import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import DictConfig

from generative_recommenders_pl.data.preprocessor import DataProcessor
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)


def load_data(ratings_file: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(ratings_file, pd.DataFrame):
        return ratings_file
    elif isinstance(ratings_file, str) and ratings_file.endswith(".csv"):
        return pd.read_csv(ratings_file, delimiter=",")
    else:
        raise ValueError("ratings_file must be a csv file.")


def save_data(ratings_frame: pd.DataFrame, output_file: str):
    if output_file.endswith(".csv"):
        ratings_frame.to_csv(output_file, index=False)
    else:
        raise ValueError("ratings_file must be a csv file.")


class RecoDataset(torch.utils.data.Dataset):
    """In reverse chronological order."""

    def __init__(
        self,
        ratings_file: str | pd.DataFrame,
        padding_length: int,
        ignore_last_n: int,  # used for creating train/valid/test sets
        shift_id_by: int = 0,
        chronological: bool = False,
        sample_ratio: float = 1.0,
        additional_columns: Optional[List[str]] = [],
        target_train_counts: Optional[Dict[int, int]] = None,
        cold_item_ids: Optional[List[int]] = None,
        split_name: str = "train",
    ) -> None:
        """
        Args:
            ratings_file: str or pd.DataFrame, path to the ratings file or DataFrame.
            padding_length: int, length to pad sequences to.
            ignore_last_n: int, number of last interactions to ignore (used for creating train/valid/test sets).
            shift_id_by: int, value to shift IDs by. Default is 0.
            chronological: bool, whether to sort interactions chronologically. Default is False.
            sample_ratio: float, ratio of data to sample. Default is 1.0 (use all data).
            additional_columns: Optional[List[str]], list of additional columns to include. Default is None.
        """
        super().__init__()

        self.ratings_frame: pd.DataFrame = load_data(ratings_file).copy()
        self._padding_length: int = padding_length
        self._ignore_last_n: int = ignore_last_n
        self._cache = dict()
        self._shift_id_by: int = shift_id_by
        self._chronological: bool = chronological
        self._sample_ratio: float = sample_ratio
        self._additional_columns = additional_columns
        self._target_train_counts = target_train_counts or {}
        self._cold_item_ids = set(int(x) for x in (cold_item_ids or []))
        self._split_name = split_name
        if self._split_name == "train" and self._cold_item_ids:
            self.ratings_frame = self._strict_cold_train_frame(self.ratings_frame)
            # The holdout and cold filtering have already been materialized above.
            self._ignore_last_n = 0
        self.__additional_columns_check()

    def _strict_cold_train_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove held-out cold items from every training sequence.

        Rows with fewer than two remaining interactions cannot form a
        history/target pair and are removed.
        """
        kept_rows = []
        for _, row in frame.iterrows():
            item_ids = list(ast.literal_eval(row.sequence_item_ids))
            ratings = list(ast.literal_eval(row.sequence_ratings))
            timestamps = list(ast.literal_eval(row.sequence_timestamps))
            if self._ignore_last_n > 0:
                item_ids = item_ids[: -self._ignore_last_n]
                ratings = ratings[: -self._ignore_last_n]
                timestamps = timestamps[: -self._ignore_last_n]
            kept = [
                (item_id, rating, timestamp)
                for item_id, rating, timestamp in zip(item_ids, ratings, timestamps)
                if int(item_id) not in self._cold_item_ids
            ]
            if len(kept) < 2:
                continue
            item_ids, ratings, timestamps = map(list, zip(*kept))
            out = row.copy()
            out.sequence_item_ids = repr(item_ids)
            out.sequence_ratings = repr(ratings)
            out.sequence_timestamps = repr(timestamps)
            kept_rows.append(out)
        if not kept_rows:
            raise ValueError("Strict cold-start filtering removed every training row")
        return pd.DataFrame(kept_rows).reset_index(drop=True)

    def __additional_columns_check(self):
        if self._additional_columns:
            columns_status = []
            for column in self._additional_columns:
                # check the column exists and status, like type, max, min, etc.
                column_exists = column in self.ratings_frame.columns
                if not column_exists:
                    raise ValueError(
                        f"Column {column} does not exist in the ratings file."
                    )
                column_type = self.ratings_frame[column].dtype
                max_value = self.ratings_frame[column].max()
                min_value = self.ratings_frame[column].min()
                columns_status.append(
                    {
                        "column": column,
                        "type": column_type,
                        "max": max_value,
                        "min": min_value,
                    }
                )
            log.info(f"Additional columns status: {columns_status}")

    def __len__(self) -> int:
        return len(self.ratings_frame)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if idx in self._cache.keys():
            return self._cache[idx]
        sample = self.load_item(idx)
        self._cache[idx] = sample
        return sample

    def load_item(self, idx) -> Dict[str, torch.Tensor]:
        data = self.ratings_frame.iloc[idx]
        user_id = data.user_id

        def eval_as_list(x, ignore_last_n) -> List[int]:
            y = ast.literal_eval(x)
            y_list = [y] if isinstance(y, int) else list(y)
            if ignore_last_n > 0:
                # for training data creation
                y_list = y_list[:-ignore_last_n]
            return y_list

        def eval_int_list(
            x,
            target_len: int,
            ignore_last_n: int,
            shift_id_by: int,
            sampling_kept_mask: Optional[List[bool]],
        ) -> Tuple[List[int], int]:
            y = eval_as_list(x, ignore_last_n=ignore_last_n)
            if sampling_kept_mask is not None:
                y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
            y_len = len(y)
            y.reverse()
            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]
            return y, y_len

        if self._sample_ratio < 1.0:
            raw_length = len(eval_as_list(data.sequence_item_ids, self._ignore_last_n))
            sampling_kept_mask = (
                torch.rand((raw_length,), dtype=torch.float32) < self._sample_ratio
            ).tolist()
        else:
            sampling_kept_mask = None

        movie_history, movie_history_len = eval_int_list(
            data.sequence_item_ids,
            self._padding_length,
            self._ignore_last_n,
            shift_id_by=self._shift_id_by,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_history_ratings, ratings_len = eval_int_list(
            data.sequence_ratings,
            self._padding_length,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_timestamps, timestamps_len = eval_int_list(
            data.sequence_timestamps,
            self._padding_length,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        assert (
            movie_history_len == timestamps_len
        ), f"history len {movie_history_len} differs from timestamp len {timestamps_len}."
        assert (
            movie_history_len == ratings_len
        ), f"history len {movie_history_len} differs from ratings len {ratings_len}."

        def _truncate_or_pad_seq(
            y: List[int], target_len: int, chronological: bool
        ) -> List[int]:
            y_len = len(y)
            if y_len < target_len:
                y = y + [0] * (target_len - y_len)
            else:
                if not chronological:
                    y = y[:target_len]
                else:
                    y = y[-target_len:]
            assert len(y) == target_len
            return y

        historical_ids = movie_history[1:]
        historical_ratings = movie_history_ratings[1:]
        historical_timestamps = movie_timestamps[1:]
        target_ids = movie_history[0]
        target_ratings = movie_history_ratings[0]
        target_timestamps = movie_timestamps[0]
        if self._chronological:
            historical_ids.reverse()
            historical_ratings.reverse()
            historical_timestamps.reverse()

        max_seq_len = self._padding_length - 1
        history_length = min(len(historical_ids), max_seq_len)
        historical_ids = _truncate_or_pad_seq(
            historical_ids,
            max_seq_len,
            self._chronological,
        )
        historical_ratings = _truncate_or_pad_seq(
            historical_ratings,
            max_seq_len,
            self._chronological,
        )
        historical_timestamps = _truncate_or_pad_seq(
            historical_timestamps,
            max_seq_len,
            self._chronological,
        )
        ret = {
            "user_id": user_id,
            "historical_ids": torch.tensor(historical_ids, dtype=torch.int64),
            "historical_ratings": torch.tensor(historical_ratings, dtype=torch.int64),
            "historical_timestamps": torch.tensor(
                historical_timestamps, dtype=torch.int64
            ),
            "history_lengths": history_length,
            "target_ids": target_ids,
            "target_ratings": target_ratings,
            "target_timestamps": target_timestamps,
            "target_is_cold": int(
                (int(target_ids) - self._shift_id_by) in self._cold_item_ids
            ),
            "target_train_count": int(
                self._target_train_counts.get(
                    int(target_ids) - self._shift_id_by,
                    0,
                )
            ),
        }

        for column in self._additional_columns:
            # currently we do not consider the sequence columns in the additional columns
            ret[column] = data[column]
        return ret


class RecoDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        data_preprocessor: DataProcessor,
        train_dataset: RecoDataset | DictConfig,
        val_dataset: RecoDataset | DictConfig,
        test_dataset: RecoDataset | DictConfig,
        max_sequence_length: int,
        chronological: bool,
        positional_sampling_ratio: float,
        batch_size: int = 32,
        num_workers: int = os.cpu_count() // 4,
        prefetch_factor: int = 4,
        cold_start: Optional[Dict | DictConfig] = None,
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.dataset_name = dataset_name
        self.data_preprocessor: DataProcessor = (
            hydra.utils.instantiate(data_preprocessor)
            if isinstance(data_preprocessor, DictConfig)
            else data_preprocessor
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.max_sequence_length = max_sequence_length
        self.chronological = chronological
        self.positional_sampling_ratio = positional_sampling_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch = prefetch_factor
        self.cold_start = cold_start or {}
        self.cold_item_ids: List[int] = []
        self.target_train_counts: Dict[int, int] = {}
        self._cold_split_initialized = False
        self._cold_ratings_frame: Optional[pd.DataFrame] = None
        self.__init_item_ids()

    def __init_item_ids(self):
        if self.dataset_name == "ml-1m" or self.dataset_name == "ml-20m":
            items = pd.read_csv(
                self.data_preprocessor.processed_item_csv(), delimiter=","
            )
            max_jagged_dimension = 16
            max_item_id = self.data_preprocessor.expected_max_item_id()

            # Initialize dictionaries for lengths and values
            lengths = {
                i: torch.zeros((max_item_id + 1,), dtype=torch.int64) for i in range(3)
            }
            values = {
                i: torch.zeros(
                    (max_item_id + 1, max_jagged_dimension), dtype=torch.int64
                )
                for i in range(3)
            }

            # Define max index ranges for each feature type
            max_ind_ranges = [63, 16383, 511]

            all_item_ids = []
            for df_index, row in items.iterrows():
                movie_id = int(row["movie_id"])
                genres = row["genres"].split("|")
                titles = row["cleaned_title"].split(" ")
                years = [row["year"]]

                # Process each feature type
                for i, feature_set in enumerate([genres, titles, years]):
                    feature_vector = [hash(x) % max_ind_ranges[i] for x in feature_set]
                    lengths[i][movie_id] = min(
                        len(feature_vector), max_jagged_dimension
                    )
                    for j, value in enumerate(feature_vector[:max_jagged_dimension]):
                        values[i][movie_id][j] = value

                all_item_ids.append(movie_id)
            self.all_item_ids = all_item_ids
            self.max_item_id = max_item_id
        else:
            self.all_item_ids = [
                x + 1 for x in range(self.data_preprocessor.expected_num_unique_items())
            ]
            self.max_item_id = self.data_preprocessor.expected_num_unique_items()

    @staticmethod
    def _sequence_ids(value) -> List[int]:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        return [int(x) for x in ([parsed] if isinstance(parsed, int) else parsed)]

    def _resolve_cold_item_ids(self, ratings_frame: pd.DataFrame) -> List[int]:
        enabled = bool(self.cold_start.get("enabled", False))
        if not enabled:
            return []
        ratio = float(self.cold_start.get("cold_item_ratio", 0.1))
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"cold_item_ratio must be in [0, 1), got {ratio}")
        observed_items = sorted(
            {
                item_id
                for value in ratings_frame.sequence_item_ids
                for item_id in self._sequence_ids(value)
            }
        )
        sample_size = int(round(len(observed_items) * ratio))
        rng = random.Random(int(self.cold_start.get("cold_split_seed", 42)))
        return sorted(rng.sample(observed_items, sample_size))

    @property
    def cold_split_path(self) -> str:
        return str(
            self.cold_start.get(
                "cold_split_path",
                f"tmp/{self.dataset_name}/cold_start_split.json",
            )
        )

    @staticmethod
    def _ids_sha256(item_ids: List[int]) -> str:
        payload = ",".join(str(int(item_id)) for item_id in item_ids)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _validate_existing_cold_split(
        self,
        split: Dict,
        expected_cold_item_ids: List[int],
        num_observed_items: int,
    ) -> List[int]:
        required = {
            "cold_item_ratio",
            "cold_split_seed",
            "cold_item_ids",
            "num_cold_items",
            "num_observed_items",
        }
        missing = sorted(required.difference(split))
        if missing:
            raise ValueError(
                f"Existing cold-start split {self.cold_split_path} is missing: {missing}"
            )

        cold_item_ids = [int(item_id) for item_id in split["cold_item_ids"]]
        expected_ratio = float(self.cold_start.get("cold_item_ratio", 0.1))
        expected_seed = int(self.cold_start.get("cold_split_seed", 42))
        errors = []
        if float(split["cold_item_ratio"]) != expected_ratio:
            errors.append("cold_item_ratio")
        if int(split["cold_split_seed"]) != expected_seed:
            errors.append("cold_split_seed")
        if int(split["num_observed_items"]) != num_observed_items:
            errors.append("num_observed_items")
        if int(split["num_cold_items"]) != len(cold_item_ids):
            errors.append("num_cold_items")
        if len(cold_item_ids) != len(set(cold_item_ids)):
            errors.append("duplicate cold_item_ids")
        if cold_item_ids != expected_cold_item_ids:
            errors.append("cold_item_ids")
        if errors:
            raise ValueError(
                "Existing cold-start split does not match the deterministic "
                f"configuration/data ({', '.join(errors)}); refusing to overwrite "
                f"{self.cold_split_path}"
            )
        return cold_item_ids

    def initialize_cold_start_split(
        self,
        ratings_frame: Optional[pd.DataFrame] = None,
    ) -> List[int]:
        """Resolve and freeze the cold split before the model is instantiated.

        Repeated calls are no-ops. An existing split file is treated as an audit
        artifact and must exactly match the deterministic seed/ratio/data result.
        """
        if self._cold_split_initialized:
            return self.cold_item_ids

        if ratings_frame is None:
            ratings_frame = load_data(self.data_preprocessor.output_format_csv())
        self._cold_ratings_frame = ratings_frame
        expected_cold_item_ids = self._resolve_cold_item_ids(ratings_frame)
        observed_items = sorted(
            {
                item_id
                for value in ratings_frame.sequence_item_ids
                for item_id in self._sequence_ids(value)
            }
        )

        if expected_cold_item_ids:
            if os.path.exists(self.cold_split_path):
                try:
                    with open(self.cold_split_path) as f:
                        split = json.load(f)
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Cannot read existing cold-start split {self.cold_split_path}"
                    ) from exc
                self.cold_item_ids = self._validate_existing_cold_split(
                    split,
                    expected_cold_item_ids,
                    len(observed_items),
                )
            else:
                self.cold_item_ids = expected_cold_item_ids
                split_dir = os.path.dirname(self.cold_split_path)
                if split_dir:
                    os.makedirs(split_dir, exist_ok=True)
                with open(self.cold_split_path, "w") as f:
                    json.dump(
                        {
                            "cold_item_ratio": float(
                                self.cold_start.get("cold_item_ratio", 0.1)
                            ),
                            "cold_split_seed": int(
                                self.cold_start.get("cold_split_seed", 42)
                            ),
                            "cold_item_ids": self.cold_item_ids,
                            "num_cold_items": len(self.cold_item_ids),
                            "num_observed_items": len(observed_items),
                        },
                        f,
                        indent=2,
                    )
        else:
            self.cold_item_ids = []

        self.target_train_counts = self._build_target_train_counts(
            ratings_frame,
            self.cold_item_ids,
        )
        self._cold_split_initialized = True
        # Validate the training pool eagerly so invalid experiments fail before
        # any model parameters or trainer state are created.
        _ = self.training_negative_item_ids
        return self.cold_item_ids

    @property
    def training_negative_item_ids(self) -> List[int]:
        """IDs eligible for sampled negatives during training.

        Evaluation candidates deliberately continue to use ``all_item_ids``.
        """
        if not self._cold_split_initialized:
            self.initialize_cold_start_split()

        all_item_ids = [int(item_id) for item_id in self.all_item_ids]
        if not all_item_ids:
            raise ValueError("Training negative item pool is empty")
        if 0 in all_item_ids:
            raise ValueError("Padding ID 0 must not appear in the item pool")
        if len(all_item_ids) != len(set(all_item_ids)):
            raise ValueError("Item IDs must be unique in the training negative pool")

        exclude_cold = bool(self.cold_start.get("enabled", False)) and bool(
            self.cold_start.get("exclude_cold_from_train_negatives", False)
        )
        cold_set = set(self.cold_item_ids)
        pool = (
            [item_id for item_id in all_item_ids if item_id not in cold_set]
            if exclude_cold
            else all_item_ids
        )
        if not pool:
            raise ValueError(
                "Training negative item pool is empty after excluding cold items"
            )
        if len(pool) != len(set(pool)) or 0 in pool:
            raise ValueError("Training negative item pool must be unique and exclude ID 0")
        if exclude_cold and cold_set.intersection(pool):
            raise ValueError("Cold items remain in the warm-only negative pool")
        return pool

    def _training_exposure_audit(self, ratings_frame: pd.DataFrame) -> Dict[str, int]:
        cold_set = set(self.cold_item_ids)
        ignore_last_n = int(getattr(self.train_dataset, "_ignore_last_n", 0))
        if isinstance(self.train_dataset, DictConfig):
            ignore_last_n = int(self.train_dataset.get("ignore_last_n", 0))
        cold_history = 0
        cold_targets = 0
        retained_rows = 0
        for value in ratings_frame.sequence_item_ids:
            item_ids = self._sequence_ids(value)
            if ignore_last_n > 0:
                item_ids = item_ids[:-ignore_last_n]
            item_ids = [item_id for item_id in item_ids if item_id not in cold_set]
            if len(item_ids) < 2:
                continue
            retained_rows += 1
            cold_targets += int(item_ids[-1] in cold_set)
            cold_history += sum(item_id in cold_set for item_id in item_ids[:-1])
        return {
            "retained_training_rows": retained_rows,
            "cold_items_in_training_histories": cold_history,
            "cold_items_in_training_targets": cold_targets,
        }

    def build_negative_sampling_audit(
        self,
        ratings_frame: Optional[pd.DataFrame] = None,
    ) -> Dict:
        self.initialize_cold_start_split(ratings_frame)
        if ratings_frame is None:
            ratings_frame = self._cold_ratings_frame
        if ratings_frame is None:
            ratings_frame = load_data(self.data_preprocessor.output_format_csv())

        all_ids = [int(item_id) for item_id in self.all_item_ids]
        negative_ids = self.training_negative_item_ids
        cold_ids = [int(item_id) for item_id in self.cold_item_ids]
        cold_set = set(cold_ids)
        exposure = self._training_exposure_audit(ratings_frame)
        cold_in_negative_pool = len(cold_set.intersection(negative_ids))
        cold_in_eval_candidates = len(cold_set.intersection(all_ids))
        exclude_cold = bool(self.cold_start.get("enabled", False)) and bool(
            self.cold_start.get("exclude_cold_from_train_negatives", False)
        )
        audit_passed = (
            exposure["cold_items_in_training_histories"] == 0
            and exposure["cold_items_in_training_targets"] == 0
            and cold_in_eval_candidates == len(cold_ids)
            and (not exclude_cold or cold_in_negative_pool == 0)
        )
        if not audit_passed:
            raise ValueError("Cold-start negative-sampling protocol audit failed")

        return {
            "protocol": "strict_cold_start_warm_only_training_negatives",
            "exclude_cold_from_train_negatives": exclude_cold,
            "num_all_items": len(all_ids),
            "num_cold_items": len(cold_ids),
            "num_warm_items": len(all_ids) - len(cold_set.intersection(all_ids)),
            "num_negative_pool_items": len(negative_ids),
            "num_training_negative_items": len(negative_ids),
            "cold_in_train_history": exposure["cold_items_in_training_histories"],
            "cold_in_train_targets": exposure["cold_items_in_training_targets"],
            "cold_in_negative_pool": cold_in_negative_pool,
            "cold_in_eval_candidates": cold_in_eval_candidates,
            "cold_items_in_training_negative_pool": cold_in_negative_pool,
            "cold_items_in_eval_candidates": cold_in_eval_candidates,
            **exposure,
            "cold_split_seed": int(self.cold_start.get("cold_split_seed", 42)),
            "cold_item_ids_hash": self._ids_sha256(cold_ids),
            "negative_pool_ids_hash": self._ids_sha256(negative_ids),
            "all_item_ids_sha256": self._ids_sha256(all_ids),
            "cold_item_ids_sha256": self._ids_sha256(cold_ids),
            "training_negative_item_ids_sha256": self._ids_sha256(negative_ids),
            "cold_split_path": self.cold_split_path,
            "audit_passed": True,
        }

    def write_negative_sampling_audit(self, output_dir: str) -> str:
        audit = self.build_negative_sampling_audit()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "cold_negative_sampling_audit.json")
        with open(output_path, "w") as f:
            json.dump(audit, f, indent=2, sort_keys=True)
        return output_path

    def _build_target_train_counts(
        self,
        ratings_frame: pd.DataFrame,
        cold_item_ids: List[int],
    ) -> Dict[int, int]:
        """Count item occurrences visible to the training objective."""
        cold_set = set(int(x) for x in cold_item_ids)
        ignore_last_n = int(getattr(self.train_dataset, "_ignore_last_n", 0))
        if isinstance(self.train_dataset, DictConfig):
            ignore_last_n = int(self.train_dataset.get("ignore_last_n", 0))
        counts: Counter = Counter()
        for value in ratings_frame.sequence_item_ids:
            item_ids = self._sequence_ids(value)
            if ignore_last_n > 0:
                item_ids = item_ids[:-ignore_last_n]
            item_ids = [item_id for item_id in item_ids if item_id not in cold_set]
            if len(item_ids) < 2:
                continue
            counts.update(item_ids)
        return dict(counts)

    def instantiate_dataset(
        self,
        dataset: RecoDataset | DictConfig,
        *,
        ratings_frame: Optional[pd.DataFrame] = None,
        split_name: str = "train",
    ) -> RecoDataset:
        if isinstance(dataset, DictConfig):
            dataset = dataset.copy()
            kwargs = {}
            if "padding_length" not in dataset:
                kwargs["padding_length"] = self.max_sequence_length + 1
            if "chronological" not in dataset:
                kwargs["chronological"] = self.chronological
            if "position_sampling_ratio" not in dataset:
                kwargs["sample_ratio"] = self.positional_sampling_ratio
            # preload the data for shared dataset
            configured_ratings_file = (
                dataset.pop("ratings_file")
                if "ratings_file" in dataset
                else self.data_preprocessor.output_format_csv()
            )
            ratings_file = (
                ratings_frame
                if ratings_frame is not None
                else load_data(configured_ratings_file)
            )
            return hydra.utils.instantiate(
                dataset,
                ratings_file=ratings_file,
                cold_item_ids=self.cold_item_ids,
                target_train_counts=self.target_train_counts,
                split_name=split_name,
                **kwargs,
            )
        else:
            return dataset

    def setup(self, stage=None):
        ratings_frame = load_data(self.data_preprocessor.output_format_csv())
        self.initialize_cold_start_split(ratings_frame)
        if stage == "fit" or stage is None:
            self.train_dataset = self.instantiate_dataset(
                self.train_dataset,
                ratings_frame=ratings_frame,
                split_name="train",
            )
            self.val_dataset = self.instantiate_dataset(
                self.val_dataset,
                ratings_frame=ratings_frame,
                split_name="val",
            )

        if stage == "test" or stage == "predict" or stage is None:
            self.test_dataset = self.instantiate_dataset(
                self.test_dataset,
                ratings_frame=ratings_frame,
                split_name="test",
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
        )

    def save_predictions(self, output_file: str, predictions: dict):
        """Save the predictions to a file.

        It adds the predictions to the ratings_frame in the test dataset
        since it is used for prediction and saves it to a file. And it
        expects the predictions to be a dictionary of list / numpy arrays,
        which has the same length and order as the test dataset.

        Args:
            output_file: str, path to the output file.
            predictions: dict, predictions to save.
        """
        ratings_frame = self.test_dataset.ratings_frame
        for key, value in predictions.items():
            ratings_frame[key] = value
        save_data(ratings_frame, output_file)
