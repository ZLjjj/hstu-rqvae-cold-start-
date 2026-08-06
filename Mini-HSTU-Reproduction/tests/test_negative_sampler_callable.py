import torch

from generative_recommenders_pl.models.negatives_samples.negative_sampler import (
    LocalNegativesSampler,
)


def test_local_negative_sampler_accepts_callable_item_embedding():
    sampler = LocalNegativesSampler(
        l2_norm=False,
        l2_norm_eps=1e-6,
        all_item_ids=[1, 2, 3, 4, 5],
    )
    item_emb = torch.nn.Embedding(6, 8, padding_idx=0)

    sampler._item_emb = lambda ids: item_emb(ids)
    sampled_ids, sampled_embeddings = sampler(
        positive_ids=torch.tensor([1, 2], dtype=torch.long),
        num_to_sample=3,
    )

    assert sampled_ids.shape == (2, 3)
    assert sampled_embeddings.shape == (2, 3, 8)
