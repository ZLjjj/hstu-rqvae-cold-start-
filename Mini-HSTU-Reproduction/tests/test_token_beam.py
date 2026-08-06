import pytest
import torch

from generative_recommenders_pl.models.token_retrieval import HierarchicalTokenRetrieval


def test_beam_search_respects_prefix_constraints():
    model = HierarchicalTokenRetrieval.__new__(HierarchicalTokenRetrieval)
    torch.nn.Module.__init__(model)
    model.num_codebooks = 2
    model.codebook_size = 4
    model.beam_size = 2
    model.prefix_constraint = True

    model._prefix_sets = [
        {(0,), (1,)},
        {(0, 0), (1, 2)},
    ]
    model._code_to_items = {
        (0, 0): [11],
        (1, 2): [22],
    }

    model.code_embs = torch.nn.ModuleList(
        [torch.nn.Embedding(5, 3, padding_idx=0), torch.nn.Embedding(5, 3, padding_idx=0)]
    )
    for emb in model.code_embs:
        emb.weight.data.zero_()

    model.code_context_proj = torch.nn.Linear(3, 3, bias=False)
    model.code_context_proj.weight.data.zero_()

    head0 = torch.nn.Linear(3, 4)
    head1 = torch.nn.Linear(3, 4)
    head0.weight.data.zero_()
    head1.weight.data.zero_()
    head0.bias.data = torch.tensor([1.0, 0.5, 0.1, 9.0])
    head1.bias.data = torch.tensor([2.0, 0.0, 1.0, 0.0])
    model.code_heads = torch.nn.ModuleList([head0, head1])

    context = torch.zeros(3)
    ids, scores = model._decode_single(context=context, invalid_ids=set(), k=1)

    assert ids.shape == (1,)
    assert scores.shape == (1,)
    assert ids[0].item() == 11


def test_exact_token_item_scores_stable_mapping():
    model = HierarchicalTokenRetrieval.__new__(HierarchicalTokenRetrieval)
    torch.nn.Module.__init__(model)
    model.num_codebooks = 3
    model.codebook_size = 4
    model.exact_auc_chunk_size = 1

    model.code_embs = torch.nn.ModuleList(
        [
            torch.nn.Embedding(5, 3, padding_idx=0),
            torch.nn.Embedding(5, 3, padding_idx=0),
            torch.nn.Embedding(5, 3, padding_idx=0),
        ]
    )
    for emb in model.code_embs:
        emb.weight.data.zero_()

    model.code_context_proj = torch.nn.Linear(3, 3, bias=False)
    model.code_context_proj.weight.data.zero_()

    head0 = torch.nn.Linear(3, 4)
    head1 = torch.nn.Linear(3, 4)
    head2 = torch.nn.Linear(3, 4)
    for head in (head0, head1, head2):
        head.weight.data.zero_()
    head0.bias.data = torch.tensor([2.0, 0.0, 0.0, -2.0])
    head1.bias.data = torch.tensor([0.0, 1.0, 2.0, 3.0])
    head2.bias.data = torch.tensor([1.0, 0.0, -1.0, -2.0])
    model.code_heads = torch.nn.ModuleList([head0, head1, head2])

    model._valid_item_ids = torch.tensor([1, 2], dtype=torch.long)
    model._valid_item_codes = torch.tensor([[0, 1, 2], [3, 2, 1]], dtype=torch.long)

    context = torch.zeros((2, 3), dtype=torch.float32)
    scores, item_ids = model._exact_token_item_scores(context)

    assert scores.shape == (2, 2)
    assert torch.equal(item_ids, torch.tensor([1, 2]))
    assert torch.allclose(scores[0], scores[1], atol=1e-6)

    logp0 = torch.log_softmax(head0.bias, dim=0)
    logp1 = torch.log_softmax(head1.bias, dim=0)
    logp2 = torch.log_softmax(head2.bias, dim=0)
    expected_item1 = logp0[0] + logp1[1] + logp2[2]
    expected_item2 = logp0[3] + logp1[2] + logp2[1]
    assert scores[0, 0].item() == pytest.approx(expected_item1.item(), abs=1e-6)
    assert scores[0, 1].item() == pytest.approx(expected_item2.item(), abs=1e-6)


def test_decode_single_beam_plus_exact_fill_to_k():
    model = HierarchicalTokenRetrieval.__new__(HierarchicalTokenRetrieval)
    torch.nn.Module.__init__(model)
    model.num_codebooks = 2
    model.codebook_size = 4
    model.beam_size = 1
    model.prefix_constraint = True
    model.exact_auc_chunk_size = 1
    model._decode_audit = {"rows": 0, "unique_before_fill": 0, "filled": 0}
    model._last_decode_unique_before_fill = 0
    model._last_decode_filled_count = 0

    model._prefix_sets = [{(0,), (1,)}, {(0, 0), (1, 1)}]
    model._code_to_items = {(0, 0): [11]}

    model.code_embs = torch.nn.ModuleList(
        [torch.nn.Embedding(5, 3, padding_idx=0), torch.nn.Embedding(5, 3, padding_idx=0)]
    )
    for emb in model.code_embs:
        emb.weight.data.zero_()
    model.code_context_proj = torch.nn.Linear(3, 3, bias=False)
    model.code_context_proj.weight.data.zero_()

    head0 = torch.nn.Linear(3, 4)
    head1 = torch.nn.Linear(3, 4)
    head0.weight.data.zero_()
    head1.weight.data.zero_()
    head0.bias.data = torch.tensor([4.0, 2.0, 0.0, -1.0])
    head1.bias.data = torch.tensor([4.0, 2.0, 0.0, -1.0])
    model.code_heads = torch.nn.ModuleList([head0, head1])

    model._valid_item_ids = torch.tensor([11, 22, 33], dtype=torch.long)
    model._valid_item_codes = torch.tensor(
        [
            [0, 0],
            [1, 1],
            [2, 2],
        ],
        dtype=torch.long,
    )

    context = torch.zeros(3)
    ids, scores = model._decode_single(context=context, invalid_ids=set(), k=3)

    assert ids.shape == (3,)
    assert scores.shape == (3,)
    assert (ids > 0).sum().item() >= 2
    assert model._last_decode_unique_before_fill == 1
    assert model._last_decode_filled_count >= 1
