"""CPU-only correctness smoke for Full-Token Route B; never runs the ML-1M trainer."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch

from generative_recommenders_pl.models.full_token_retrieval import (
    HierarchicalFullTokenRetrieval,
    expand_item_history,
)


class CausalEncoder(torch.nn.Module):
    def forward(
        self, past_lengths, user_embeddings, valid_mask, past_payloads, **kwargs
    ):
        del past_lengths, past_payloads, kwargs
        return torch.cumsum(user_embeddings * valid_mask, dim=1), []


class CandidateIndex(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._k = 4
        self.register_buffer("_ids", torch.arange(1, 7).unsqueeze(0))

    @property
    def ids(self):
        return self._ids


class Metrics(torch.nn.Module):
    def reset(self):
        pass

    def update(self, **kwargs):
        self.last = kwargs

    def compute(self):
        return {"ndcg@100": torch.tensor(0.0)}


def build_model(bridge_path: Path) -> HierarchicalFullTokenRetrieval:
    return HierarchicalFullTokenRetrieval(
        semantic_bridge_path=str(bridge_path),
        max_history_items=3,
        beam_size=2,
        adaptive_beam=True,
        max_beam_size=4,
        target_unique_items=4,
        exact_prefix_batch_size=4,
        datamodule=object(),
        embeddings=torch.nn.Embedding(7, 8),
        preprocessor=torch.nn.Identity(),
        sequence_encoder=CausalEncoder(),
        postprocessor=torch.nn.Identity(),
        similarity=torch.nn.Identity(),
        negatives_sampler=torch.nn.Identity(),
        candidate_index=CandidateIndex(),
        loss=torch.nn.Identity(),
        metrics=Metrics(),
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
        scheduler=None,
        configure_optimizer_params={"monitor": "val/ndcg@100"},
        gr_output_length=0,
        item_embedding_dim=8,
        compile_model=False,
    )


def main() -> None:
    torch.manual_seed(42)
    codes = torch.tensor(
        [
            [-1, -1, -1],
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [2, 0, 0],
            [2, 1, 1],
        ]
    )
    vectors = torch.randn(3, 4, 6)
    with tempfile.TemporaryDirectory(prefix="full-token-smoke-") as directory:
        bridge = Path(directory) / "bridge.pt"
        torch.save(
            {
                "item_id_to_codes": codes,
                "codebook_vectors": vectors,
                "metadata": {"n_codebooks": 3, "codebook_size": 4},
            },
            bridge,
        )
        model = build_model(bridge)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=1e-3
        )
        histories = torch.tensor([[1, 2, 3], [2, 4, 0]])
        lengths = torch.tensor([3, 2])
        timestamps = torch.tensor([[10, 20, 30], [11, 25, 0]])
        token_batch = expand_item_history(
            histories, lengths, timestamps, model.item_to_codes, model.vocab
        )

        started = time.perf_counter()
        diagnostics = None
        for _ in range(2):
            optimizer.zero_grad()
            loss, diagnostics = model.next_token_loss(token_batch)
            loss.backward()
            optimizer.step()
        elapsed = time.perf_counter() - started
        assert diagnostics is not None and torch.isfinite(loss)

        n = int(token_batch.lengths[0].item())
        one = type(token_batch)(
            token_batch.token_ids[:1, :n],
            token_batch.lengths[:1],
            token_batch.timestamps[:1, :n],
            token_batch.type_ids[:1, :n],
        )
        generated = model.generate_path_greedy(one)
        beam_paths = model._beam_paths(one, width=4)
        retrieved_ids, _, audit = model._retrieve_one(one, set(), k=4)
        exact_scores, exact_ids = model.exact_item_scores(one, torch.tensor([1, 2, 3]))
        validation_batch = {
            "history_lengths": lengths,
            "historical_ids": histories,
            "historical_ratings": torch.ones_like(histories, dtype=torch.float32),
            "historical_timestamps": timestamps,
            "target_ids": torch.tensor([4, 6]),
            "target_ratings": torch.ones(2),
            "target_timestamps": torch.tensor([40, 35]),
            "target_is_cold": torch.tensor([False, True]),
            "target_train_count": torch.tensor([5, 0]),
        }
        model.on_validation_epoch_start()
        model.validation_step(validation_batch, 0)
        validation_shape = tuple(model.metrics.last["top_k_ids"].shape)

        print(f"输入物品长度: {lengths.tolist()}")
        print(f"展开后Token长度: {token_batch.token_ids.size(1)}")
        print(f"有效Token数量: {token_batch.lengths.tolist()}")
        print(
            "各层loss:",
            [round(float(diagnostics[f"loss_level_{i}"]), 6) for i in range(3)],
        )
        print(
            "各层accuracy:",
            [round(float(diagnostics[f"accuracy_level_{i}"]), 6) for i in range(3)],
        )
        print(f"三级生成: {generated}")
        print(f"Beam合法路径: {len(beam_paths)}")
        print(f"映射物品数: {int((retrieved_ids > 0).sum())}")
        print(
            f"Exact Scorer形状: scores={tuple(exact_scores.shape)}, ids={tuple(exact_ids.shape)}"
        )
        print(f"Validation Top-K形状: {validation_shape}")
        print(f"Beam诊断: {audit}")
        print(f"两个训练batch耗时: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
