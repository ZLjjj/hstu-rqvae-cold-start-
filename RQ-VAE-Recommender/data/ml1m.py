import pandas as pd
import torch

from data.preprocessing import PreprocessingMixin
from torch_geometric.data import HeteroData
from torch_geometric.datasets import MovieLens1M


class RawMovieLens1M(MovieLens1M, PreprocessingMixin):
    MOVIE_HEADERS = ["movieId", "title", "genres"]
    USER_HEADERS = ["userId", "gender", "age", "occupation", "zipCode"]
    RATING_HEADERS = ["userId", "movieId", "rating", "timestamp"]

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        force_reload=False,
        split=None,
        text_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        text_batch_size=64,
        text_cache_path=None,
        text_encoder_device="auto",
    ) -> None:
        self.text_encoder_name = text_encoder_name
        self.text_batch_size = text_batch_size
        self.text_cache_path = text_cache_path
        self.text_encoder_device = text_encoder_device
        super(RawMovieLens1M, self).__init__(
            root, transform, pre_transform, force_reload
        )

    def _load_ratings(self):
        return pd.read_csv(
            self.raw_paths[2],
            sep="::",
            header=None,
            names=self.RATING_HEADERS,
            encoding="ISO-8859-1",
            engine="python",
        )

    def process(self, max_seq_len=None) -> None:
        data = HeteroData()
        ratings_df = self._load_ratings()

        # Process movie data:
        full_df = pd.read_csv(
            self.raw_paths[0],
            sep="::",
            header=None,
            index_col="movieId",
            names=self.MOVIE_HEADERS,
            encoding="ISO-8859-1",
            engine="python",
        )
        df = self._remove_low_occurrence(ratings_df, full_df, "movieId")
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        genres = self._process_genres(
            df["genres"].str.get_dummies("|").values, one_hot=True
        )
        genres = torch.from_numpy(genres).to(torch.float)

        titles_text = df["title"].apply(lambda s: s.split("(")[0].strip()).tolist()
        external_item_ids = torch.tensor(df.index.to_list(), dtype=torch.long)
        titles_emb = self._encode_text_feature(
            titles_text,
            item_ids=external_item_ids,
        )

        x = torch.cat([titles_emb, genres], axis=1)

        data["item"].x = x
        data["item"].external_ids = external_item_ids
        split_generator = torch.Generator().manual_seed(42)
        permutation = torch.randperm(x.size(0), generator=split_generator)
        is_train = torch.zeros(x.size(0), dtype=torch.bool)
        is_train[permutation[: int(0.8 * x.size(0))]] = True
        data["item"].is_train = is_train
        # Process user data:
        full_df = pd.read_csv(
            self.raw_paths[1],
            sep="::",
            header=None,
            index_col="userId",
            names=self.USER_HEADERS,
            dtype="str",
            encoding="ISO-8859-1",
            engine="python",
        )
        df = self._remove_low_occurrence(ratings_df, full_df, "userId")
        user_mapping = {idx: i for i, idx in enumerate(df.index)}

        age = df["age"].str.get_dummies().values.argmax(axis=1)[:, None]
        age = torch.from_numpy(age).to(torch.float)

        gender = df["gender"].str.get_dummies().values[:, 0][:, None]
        gender = torch.from_numpy(gender).to(torch.float)

        occupation = df["occupation"].str.get_dummies().values.argmax(axis=1)[:, None]
        occupation = torch.from_numpy(occupation).to(torch.float)

        data["user"].x = torch.cat([age, gender, occupation], dim=-1)

        self.int_user_data = df
        # Process rating data:
        df = self._remove_low_occurrence(ratings_df, ratings_df, ["userId", "movieId"])
        src = [user_mapping[idx] for idx in df["userId"]]
        dst = [movie_mapping[idx] for idx in df["movieId"]]
        edge_index = torch.tensor([src, dst])
        data["user", "rates", "item"].edge_index = edge_index

        rating = torch.from_numpy(df["rating"].values).to(torch.long)
        data["user", "rates", "item"].rating = rating

        time = torch.from_numpy(df["timestamp"].values)
        data["user", "rates", "item"].time = time

        data["item", "rated_by", "user"].edge_index = edge_index.flip([0])
        data["item", "rated_by", "user"].rating = rating
        data["item", "rated_by", "user"].time = time

        df["itemId"] = df["movieId"].apply(lambda x: movie_mapping[x])

        # The integrated cold-start experiment trains only the RQ-VAE item
        # tokenizer; HSTU builds its own chronological interaction dataset.
        # The upstream decoder-history path relies on an older Polars List
        # representation and is intentionally omitted from this item-only
        # preprocessing artifact.

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
