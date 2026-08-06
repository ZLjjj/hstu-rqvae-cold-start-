from __future__ import annotations

import types

import pytest
import torch

from generative_recommenders_pl.models.full_token_retrieval import (
    BOS_ID,
    PAD_ID,
    FullTokenBatch,
    HierarchicalFullTokenRetrieval,
    SemanticPathTrie,
    SemanticTokenVocabulary,
    expand_item_history,
    shifted_next_token_targets,
    token_type_ids,
)


class CausalCumsumEncoder(torch.nn.Module):
    def forward(
        self, past_lengths, user_embeddings, valid_mask, past_payloads, **kwargs
    ):
        del past_lengths, past_payloads, kwargs
        return torch.cumsum(user_embeddings * valid_mask, dim=1), []


class DummyCandidateIndex(torch.nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self._k = k
        self.register_buffer("_ids", torch.arange(1, 5).unsqueeze(0))

    @property
    def ids(self):
        return self._ids


class DummyMetrics(torch.nn.Module):
    def reset(self):
        pass

    def update(self, **kwargs):
        self.last = kwargs

    def compute(self):
        return {"ndcg@100": torch.tensor(0.0)}


@pytest.fixture
def vocab():
    return SemanticTokenVocabulary(3, 4)


@pytest.fixture
def bridge(tmp_path):
    codes = torch.tensor([[-1, -1, -1], [0, 1, 2], [1, 2, 3], [0, 1, 2], [-1, -1, -1]])
    vectors = torch.arange(3 * 4 * 4, dtype=torch.float32).view(3, 4, 4) / 10
    path = tmp_path / "bridge.pt"
    torch.save(
        {
            "item_id_to_codes": codes,
            "codebook_vectors": vectors,
            "metadata": {"n_codebooks": 3, "codebook_size": 4},
        },
        path,
    )
    return path, codes, vectors


def make_model(bridge, freeze=False, sequence_encoder=None):
    path, _, _ = bridge
    return HierarchicalFullTokenRetrieval(
        semantic_bridge_path=str(path),
        max_history_items=3,
        freeze_semantic_token_embeddings=freeze,
        beam_size=2,
        adaptive_beam=True,
        max_beam_size=4,
        target_unique_items=3,
        exact_fill=True,
        compute_full_auc=True,
        datamodule=object(),
        embeddings=torch.nn.Embedding(5, 5),
        preprocessor=torch.nn.Identity(),
        sequence_encoder=sequence_encoder or CausalCumsumEncoder(),
        postprocessor=torch.nn.Identity(),
        similarity=torch.nn.Identity(),
        negatives_sampler=torch.nn.Identity(),
        candidate_index=DummyCandidateIndex(),
        loss=torch.nn.Identity(),
        metrics=DummyMetrics(),
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        scheduler=None,
        configure_optimizer_params={"monitor": "val/ndcg@100"},
        gr_output_length=0,
        item_embedding_dim=5,
        compile_model=False,
    )


def sample_history(
    codes, vocab, ids=None, lengths=None, timestamps=None, strategy="filter"
):
    ids = torch.tensor([[1, 2, 0]]) if ids is None else ids
    lengths = torch.tensor([2]) if lengths is None else lengths
    timestamps = torch.tensor([[100, 160, 0]]) if timestamps is None else timestamps
    return expand_item_history(ids, lengths, timestamps, codes, vocab, strategy)


def test_01_token_id_ranges(vocab):
    assert vocab.token_id(0, 0) == 2
    assert vocab.token_id(2, 3) == 13
    assert vocab.vocab_size == 14


def test_02_token_id_round_trip(vocab):
    for level in range(3):
        for code in range(4):
            assert vocab.parse_token_id(vocab.token_id(level, code)) == (level, code)


@pytest.mark.parametrize("special", [PAD_ID, BOS_ID])
def test_03_pad_and_bos_are_not_semantic(vocab, special):
    with pytest.raises(ValueError):
        vocab.parse_token_id(special)


def test_04_same_code_differs_by_level(vocab):
    assert len({vocab.token_id(level, 2) for level in range(3)}) == 3


def test_05_one_item_expands_to_three_tokens(bridge, vocab):
    _, codes, _ = bridge
    out = sample_history(
        codes, vocab, torch.tensor([[1]]), torch.tensor([1]), torch.tensor([[7]])
    )
    assert out.token_ids[0, :4].tolist() == [BOS_ID, 2, 7, 12]


def test_06_multiple_items_preserve_order(bridge, vocab):
    _, codes, _ = bridge
    out = sample_history(codes, vocab)
    assert out.token_ids[0, :7].tolist() == [1, 2, 7, 12, 3, 8, 13]


def test_07_bos_is_first(bridge, vocab):
    _, codes, _ = bridge
    assert sample_history(codes, vocab).token_ids[0, 0] == BOS_ID


def test_08_padding_and_length(bridge, vocab):
    _, codes, _ = bridge
    out = sample_history(codes, vocab)
    assert out.lengths.item() == 7
    assert out.token_ids.shape == (1, 10)
    assert torch.equal(out.token_ids[0, 7:], torch.zeros(3, dtype=torch.long))


def test_09_missing_semantic_id_is_counted_not_padded(bridge, vocab):
    _, codes, _ = bridge
    out = sample_history(
        codes,
        vocab,
        torch.tensor([[1, 4]]),
        torch.tensor([2]),
        torch.tensor([[10, 20]]),
    )
    assert out.missing_item_count == 1
    assert out.lengths.item() == 4


def test_10_missing_semantic_error_strategy(bridge, vocab):
    _, codes, _ = bridge
    with pytest.raises(ValueError, match="no complete Semantic ID"):
        sample_history(
            codes,
            vocab,
            torch.tensor([[4]]),
            torch.tensor([1]),
            torch.tensor([[10]]),
            "error",
        )


def test_11_codebook_vectors_copy_to_correct_rows(bridge):
    model = make_model(bridge)
    _, _, vectors = bridge
    for level in range(3):
        start = model.vocab.token_id(level, 0)
        assert torch.equal(
            model.token_embedding.weight[start : start + 4], vectors[level]
        )


def test_12_pad_embedding_is_zero(bridge):
    model = make_model(bridge)
    assert torch.count_nonzero(model.token_embedding.weight[PAD_ID]) == 0


def test_13_freeze_masks_semantic_gradients_but_not_bos(bridge):
    model = make_model(bridge, freeze=True)
    model.token_embedding(torch.tensor([BOS_ID, 2, 7])).sum().backward()
    grad = model.token_embedding.weight.grad
    assert grad[BOS_ID].abs().sum() > 0
    assert grad[2:].abs().sum() == 0


def test_14_level_type_embedding_ids(vocab):
    ids = torch.tensor([[0, 1, 2, 6, 10]])
    assert token_type_ids(ids, vocab).tolist() == [[0, 1, 2, 3, 4]]


def test_15_same_item_tokens_share_timestamp(bridge, vocab):
    _, codes, _ = bridge
    out = sample_history(codes, vocab)
    assert out.timestamps[0, 1:4].tolist() == [100, 100, 100]


def test_16_different_item_time_gap_is_preserved(bridge, vocab):
    _, codes, _ = bridge
    out = sample_history(codes, vocab)
    assert out.timestamps[0, 4:7].tolist() == [160, 160, 160]
    assert out.timestamps[0, 0].item() == 100


def test_17_causal_encoder_cannot_change_past_output(bridge):
    model = make_model(bridge)
    _, codes, _ = bridge
    history = sample_history(codes, model.vocab)
    first = model.forward_tokens(history)
    changed = history.token_ids.clone()
    changed[0, 5] = model.vocab.token_id(1, 0)
    second = model.forward_tokens(
        FullTokenBatch(changed, history.lengths, history.timestamps, history.type_ids)
    )
    assert torch.allclose(first[:, :5], second[:, :5])


def test_18_next_token_is_strictly_shifted(bridge, vocab):
    _, codes, _ = bridge
    batch = sample_history(codes, vocab)
    inputs, targets, _ = shifted_next_token_targets(batch, vocab, True)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_19_padding_is_not_supervised(bridge, vocab):
    _, codes, _ = bridge
    batch = sample_history(codes, vocab)
    _, _, mask = shifted_next_token_targets(batch, vocab, True)
    assert mask.sum().item() == batch.lengths.item() - 1
    assert not mask[0, batch.lengths.item() - 1 :].any()


def test_20_bos_loss_switch(bridge, vocab):
    _, codes, _ = bridge
    batch = sample_history(codes, vocab)
    assert shifted_next_token_targets(batch, vocab, True)[2][0, 0]
    assert not shifted_next_token_targets(batch, vocab, False)[2][0, 0]


def test_21_each_position_selects_correct_head(bridge):
    model = make_model(bridge)
    _, codes, _ = bridge
    _, diag = model.next_token_loss(sample_history(codes, model.vocab))
    assert [diag[f"count_level_{i}"].item() for i in range(3)] == [1, 2, 2]


def test_22_trie_root_children():
    trie = SemanticPathTrie([(0, 1, 2), (0, 2, 3), (3, 1, 0)])
    assert trie.legal_children(()) == (0, 3)


def test_23_trie_second_level_children():
    trie = SemanticPathTrie([(0, 1, 2), (0, 2, 3)])
    assert trie.legal_children((0,)) == (1, 2)


def test_24_trie_third_level_children():
    trie = SemanticPathTrie([(0, 1, 2), (0, 1, 3)])
    assert trie.legal_children((0, 1)) == (2, 3)


def test_25_beam_only_emits_legal_complete_paths(bridge):
    model = make_model(bridge)
    _, codes, _ = bridge
    paths = model._beam_paths(sample_history(codes, model.vocab), width=4)
    assert paths
    assert all(tuple(path) in model._path_to_items for path, _ in paths)


def test_26_path_maps_back_to_movie_ids(bridge):
    model = make_model(bridge)
    assert model._path_to_items[(1, 2, 3)] == [2]


def test_27_collision_maps_all_items_with_same_path_score(bridge):
    model = make_model(bridge)
    assert model._path_to_items[(0, 1, 2)] == [1, 3]


def test_28_c1_is_reinserted_before_c2_prediction(bridge):
    model = make_model(bridge)
    _, codes, _ = bridge
    history = sample_history(codes, model.vocab)
    seen = []
    model.forward_tokens = types.MethodType(
        lambda self, batch: seen.append(batch.token_ids.clone())
        or torch.zeros(
            batch.token_ids.size(0), batch.token_ids.size(1), self.item_embedding_dim
        ),
        model,
    )
    model._next_logits_single(history, (2,), 1)
    assert seen[0][0, -1].item() == model.vocab.token_id(0, 2)


def test_29_c2_is_reinserted_before_c3_prediction(bridge):
    model = make_model(bridge)
    _, codes, _ = bridge
    history = sample_history(codes, model.vocab)
    seen = []
    model.forward_tokens = types.MethodType(
        lambda self, batch: seen.append(batch.token_ids.clone())
        or torch.zeros(
            batch.token_ids.size(0), batch.token_ids.size(1), self.item_embedding_dim
        ),
        model,
    )
    model._next_logits_single(history, (2, 3), 2)
    assert seen[0][0, -2:].tolist() == [
        model.vocab.token_id(0, 2),
        model.vocab.token_id(1, 3),
    ]


def test_30_exact_scorer_matches_manual_three_forward_sum(bridge):
    model = make_model(bridge)
    _, codes, _ = bridge
    history = sample_history(codes, model.vocab)
    scores, ids = model.exact_item_scores(history, torch.tensor([2]))
    path = tuple(model.item_to_codes[2].tolist())
    manual = sum(
        torch.log_softmax(
            model._next_logits_single(history, path[:level], level), dim=-1
        )[path[level]]
        for level in range(3)
    )
    assert ids.tolist() == [2]
    assert torch.allclose(scores[0], manual, atol=1e-6)


def test_31_old_and_new_config_targets_are_independent():
    from pathlib import Path

    root = Path(__file__).parents[1] / "configs" / "experiment"
    old = (root / "ml-1m-hstu-token-cold.yaml").read_text()
    new = (root / "ml-1m-hstu-full-token-cold.yaml").read_text()
    assert "HierarchicalTokenRetrieval" in old
    assert "HierarchicalFullTokenRetrieval" in new
    assert "HierarchicalFullTokenRetrieval" not in old


def test_32_model_forward_and_backward(bridge):
    model = make_model(bridge)
    _, codes, _ = bridge
    loss, _ = model.next_token_loss(sample_history(codes, model.vocab))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.code_heads[0].weight.grad is not None


def test_33_small_batch_has_no_nan(bridge):
    model = make_model(bridge)
    _, codes, _ = bridge
    loss, diagnostics = model.next_token_loss(sample_history(codes, model.vocab))
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value.float()) for value in diagnostics.values())


def test_34_masks_and_tensor_shapes(bridge):
    model = make_model(bridge)
    _, codes, _ = bridge
    batch = sample_history(codes, model.vocab)
    embedded, valid = model._embed_tokens(batch.token_ids)
    encoded = model.forward_tokens(batch)
    assert embedded.shape == encoded.shape == (1, 10, 5)
    assert valid.shape == (1, 10, 1)


def test_35_dynamic_max_token_length(bridge):
    model = make_model(bridge)
    assert model.max_token_sequence_len == 1 + 3 * (3 + 1)


def test_36_legal_filter_happens_before_topk(bridge):
    model = make_model(bridge)
    _, codes, _ = bridge
    history = sample_history(codes, model.vocab)

    def illegal_is_largest(self, hist, prefixes, level):
        del hist, level
        return torch.tensor([[0.0, 1.0, 100.0, 99.0]]).repeat(len(prefixes), 1)

    model._next_logits_for_prefixes = types.MethodType(illegal_is_largest, model)
    paths = model._beam_paths(history, width=1)
    assert paths[0][0] in model._path_to_items


def test_37_exact_collision_items_receive_equal_scores(bridge):
    model = make_model(bridge)
    _, codes, _ = bridge
    scores, ids = model.exact_item_scores(
        sample_history(codes, model.vocab), torch.tensor([1, 3])
    )
    assert ids.tolist() == [1, 3]
    assert scores[0] == scores[1]


def test_38_corrected_mrr_assigns_zero_to_miss():
    from generative_recommenders_pl.models.metrics.retrieval import RetrievalMetrics

    metric = RetrievalMetrics(k=3, at_k_list=[3])
    metric.update(top_k_ids=torch.tensor([[1, 2, 3]]), target_ids=torch.tensor([[9]]))
    result = metric.compute()
    assert result["mrr_corrected"].item() == 0.0


def test_39_warm_metrics_are_reported():
    from generative_recommenders_pl.models.metrics.retrieval import RetrievalMetrics

    metric = RetrievalMetrics(k=3, at_k_list=[3])
    metric.update(
        top_k_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        target_ids=torch.tensor([[1], [5]]),
        target_is_cold=torch.tensor([True, False]),
        auc_ranks=torch.tensor([1.0, 2.0]),
        auc_num_candidates=torch.tensor([4.0, 4.0]),
    )
    result = metric.compute()
    assert result["warm_hr@3"].item() == 1.0
    assert result["warm_mrr_corrected"].item() == pytest.approx(0.5)
    assert result["warm_auc"].item() == pytest.approx(2 / 3)


def test_40_real_hstu_forward_backward(bridge):
    from generative_recommenders_pl.models.sequential_encoders.hstu import HSTU

    encoder = HSTU(
        max_sequence_len=13,
        max_output_len=0,
        embedding_dim=5,
        item_embedding_dim=5,
        num_blocks=1,
        num_heads=1,
        linear_dim=5,
        attention_dim=5,
        normalization="rel_bias",
        linear_config="uvqk",
        linear_activation="silu",
        linear_dropout_rate=0.0,
        attn_dropout_rate=0.0,
        enable_relative_attention_bias=True,
    )
    model = make_model(bridge, sequence_encoder=encoder)
    _, codes, _ = bridge
    loss, _ = model.next_token_loss(sample_history(codes, model.vocab))
    loss.backward()
    assert torch.isfinite(loss)


def test_41_old_route_b_still_instantiates(bridge):
    from generative_recommenders_pl.models.token_retrieval import (
        HierarchicalTokenRetrieval,
    )

    path, _, _ = bridge
    old = HierarchicalTokenRetrieval(
        semantic_bridge_path=str(path),
        beam_size=2,
        datamodule=object(),
        embeddings=torch.nn.Embedding(5, 5),
        preprocessor=torch.nn.Identity(),
        sequence_encoder=CausalCumsumEncoder(),
        postprocessor=torch.nn.Identity(),
        similarity=torch.nn.Identity(),
        negatives_sampler=torch.nn.Identity(),
        candidate_index=DummyCandidateIndex(),
        loss=torch.nn.Identity(),
        metrics=DummyMetrics(),
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        scheduler=None,
        configure_optimizer_params={"monitor": "val/ndcg@100"},
        gr_output_length=0,
        item_embedding_dim=5,
        compile_model=False,
    )
    assert old.num_codebooks == 3
    assert old.codebook_size == 4


def test_42_result_summarizer_reads_three_ranking_modes(tmp_path):
    from scripts.summarize_full_token_route_b import make_results

    row = {
        "epoch": "7",
        "test/ndcg@100": "0.2",
        "test/mrr_corrected": "0.1",
        "test/cold_hr@100": "0.3",
        "test/cold_ndcg@100": "0.15",
        "test/cold_mrr_corrected": "0.05",
        "test/beam_only/hr@100": "0.18",
        "test/exact_global/hr@100": "0.24",
    }
    markdown = make_results([row], tmp_path / "metrics.csv", tmp_path)
    assert "新 Full-Token Route B | 0.200000" in markdown
    assert "Beam-only | 0.180000" in markdown
    assert "Exact 全局排序 | 0.240000" in markdown
