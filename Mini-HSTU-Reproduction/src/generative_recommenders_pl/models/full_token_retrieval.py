from __future__ import annotations

import os
import copy
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from generative_recommenders_pl.models.generative_recommenders import (
    GenerativeRecommenders,
)
from generative_recommenders_pl.models.utils.features import seq_features_from_row
from generative_recommenders_pl.utils.logger import RankedLogger


log = RankedLogger(__name__)
PAD_ID = 0
BOS_ID = 1


class SemanticTokenVocabulary:
    """Layer-aware token IDs for an L-level residual-quantizer vocabulary."""

    def __init__(self, num_codebooks: int, codebook_size: int) -> None:
        if num_codebooks <= 0 or codebook_size <= 0:
            raise ValueError("num_codebooks and codebook_size must be positive")
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.vocab_size = 2 + self.num_codebooks * self.codebook_size

    def token_id(self, level: int, code: int) -> int:
        if not 0 <= level < self.num_codebooks:
            raise ValueError(f"invalid semantic level: {level}")
        if not 0 <= code < self.codebook_size:
            raise ValueError(f"invalid semantic code: {code}")
        return 2 + level * self.codebook_size + code

    def parse_token_id(self, token_id: int) -> Tuple[int, int]:
        if token_id in (PAD_ID, BOS_ID):
            raise ValueError("PAD and BOS are not semantic-code tokens")
        shifted = int(token_id) - 2
        if shifted < 0 or shifted >= self.num_codebooks * self.codebook_size:
            raise ValueError(f"token id outside semantic vocabulary: {token_id}")
        return shifted // self.codebook_size, shifted % self.codebook_size

    def encode_codes(self, codes: Sequence[int]) -> List[int]:
        if len(codes) != self.num_codebooks:
            raise ValueError("one item must contain exactly one code per codebook")
        return [self.token_id(level, int(code)) for level, code in enumerate(codes)]


@dataclass
class FullTokenBatch:
    token_ids: torch.Tensor
    lengths: torch.Tensor
    timestamps: torch.Tensor
    type_ids: torch.Tensor
    missing_item_count: int = 0


def token_type_ids(
    token_ids: torch.Tensor, vocab: SemanticTokenVocabulary
) -> torch.Tensor:
    """0=PAD, 1=BOS, 2..L+1=semantic levels 0..L-1."""
    out = torch.zeros_like(token_ids)
    out[token_ids == BOS_ID] = 1
    semantic = token_ids >= 2
    out[semantic] = 2 + (token_ids[semantic] - 2) // vocab.codebook_size
    if semantic.any() and int(out[semantic].max()) > vocab.num_codebooks + 1:
        raise ValueError("semantic token outside configured vocabulary")
    return out


def expand_item_history(
    item_ids: torch.Tensor,
    lengths: torch.Tensor,
    timestamps: torch.Tensor,
    item_to_codes: torch.Tensor,
    vocab: SemanticTokenVocabulary,
    missing_strategy: str = "filter",
) -> FullTokenBatch:
    """Expand left-aligned item histories to BOS + L semantic tokens per valid item.

    Missing codes are either explicitly filtered and counted, or rejected. They are
    never converted to PAD, which would make missing content indistinguishable from
    sequence padding.
    """
    if missing_strategy not in {"filter", "error"}:
        raise ValueError("missing_strategy must be 'filter' or 'error'")
    if item_ids.shape != timestamps.shape or item_ids.dim() != 2:
        raise ValueError("item_ids and timestamps must be same-shaped [B, N] tensors")
    batch_size, max_items = item_ids.shape
    max_tokens = 1 + vocab.num_codebooks * max_items
    token_ids = torch.zeros(
        (batch_size, max_tokens), dtype=torch.long, device=item_ids.device
    )
    token_timestamps = torch.zeros_like(token_ids)
    token_lengths = torch.ones((batch_size,), dtype=torch.long, device=item_ids.device)
    missing = 0

    for row in range(batch_size):
        valid_n = min(int(lengths[row].item()), max_items)
        pieces: List[int] = [BOS_ID]
        times: List[int] = []
        first_valid_ts: Optional[int] = None
        for col in range(valid_n):
            item_id = int(item_ids[row, col].item())
            valid_item_id = 0 < item_id < item_to_codes.size(0)
            codes = item_to_codes[item_id] if valid_item_id else None
            valid_codes = (
                codes is not None
                and codes.numel() >= vocab.num_codebooks
                and bool(
                    (
                        (codes[: vocab.num_codebooks] >= 0)
                        & (codes[: vocab.num_codebooks] < vocab.codebook_size)
                    ).all()
                )
            )
            if not valid_codes:
                missing += 1
                if missing_strategy == "error":
                    raise ValueError(
                        f"history item {item_id} has no complete Semantic ID"
                    )
                continue
            timestamp = int(timestamps[row, col].item())
            if first_valid_ts is None:
                first_valid_ts = timestamp
            pieces.extend(vocab.encode_codes(codes[: vocab.num_codebooks].tolist()))
            times.extend([timestamp] * vocab.num_codebooks)
        bos_ts = first_valid_ts if first_valid_ts is not None else 0
        full_times = [bos_ts, *times]
        n = len(pieces)
        token_ids[row, :n] = torch.tensor(
            pieces, dtype=torch.long, device=item_ids.device
        )
        token_timestamps[row, :n] = torch.tensor(
            full_times, dtype=torch.long, device=item_ids.device
        )
        token_lengths[row] = n

    return FullTokenBatch(
        token_ids=token_ids,
        lengths=token_lengths,
        timestamps=token_timestamps,
        type_ids=token_type_ids(token_ids, vocab),
        missing_item_count=missing,
    )


def shifted_next_token_targets(
    token_batch: FullTokenBatch,
    vocab: SemanticTokenVocabulary,
    include_bos_to_first_item: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return strict-shift inputs, targets and a non-padding supervision mask."""
    inputs = token_batch.token_ids[:, :-1]
    targets = token_batch.token_ids[:, 1:]
    positions = torch.arange(targets.size(1), device=targets.device).unsqueeze(0)
    mask = positions < (token_batch.lengths - 1).unsqueeze(1)
    mask &= targets != PAD_ID
    if not include_bos_to_first_item and mask.size(1) > 0:
        mask[:, 0] = False
    # Every supervised target must be a semantic code and thus select one head.
    if bool((mask & (targets < 2)).any()):
        raise ValueError("non-semantic token found in next-token supervision")
    return inputs, targets, mask


class SemanticPathTrie:
    def __init__(self, paths: Iterable[Sequence[int]]) -> None:
        self._children: Dict[Tuple[int, ...], set[int]] = {}
        self._terminals: set[Tuple[int, ...]] = set()
        self.path_count = 0
        for raw_path in paths:
            path = tuple(int(x) for x in raw_path)
            if not path:
                continue
            for level, code in enumerate(path):
                self._children.setdefault(path[:level], set()).add(code)
            self._terminals.add(path)
            self.path_count += 1

    def legal_children(self, prefix: Sequence[int]) -> Tuple[int, ...]:
        return tuple(sorted(self._children.get(tuple(prefix), set())))

    def is_complete(self, path: Sequence[int], depth: int) -> bool:
        return len(path) == depth and tuple(path) in self._terminals


class HierarchicalFullTokenRetrieval(GenerativeRecommenders):
    """Full-history Semantic-ID tokenization with true autoregressive decoding."""

    def __init__(
        self,
        semantic_bridge_path: str,
        num_codebooks: Optional[int] = None,
        codebook_size: Optional[int] = None,
        max_history_items: int = 200,
        max_token_sequence_len: Optional[int] = None,
        freeze_semantic_token_embeddings: bool = False,
        include_bos_to_first_item_loss: bool = False,
        missing_semantic_strategy: str = "filter",
        beam_size: int = 32,
        adaptive_beam: bool = True,
        max_beam_size: int = 256,
        target_unique_items: int = 200,
        exact_fill: bool = True,
        exact_fill_on_validation: bool = False,
        compute_full_auc: bool = True,
        compute_full_auc_on_validation: bool = False,
        exact_prefix_batch_size: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not os.path.exists(semantic_bridge_path):
            raise FileNotFoundError(
                f"semantic bridge not found: {semantic_bridge_path}"
            )
        payload = torch.load(
            semantic_bridge_path, map_location="cpu", weights_only=False
        )
        item_to_codes = payload.get("item_id_to_codes")
        codebook_vectors = payload.get("codebook_vectors")
        metadata = payload.get("metadata", {})
        if item_to_codes is None or codebook_vectors is None:
            raise ValueError(
                "bridge must contain item_id_to_codes and codebook_vectors"
            )
        item_to_codes = item_to_codes.long()
        codebook_vectors = codebook_vectors.float()
        levels = int(metadata.get("n_codebooks", item_to_codes.size(1)))
        size = int(metadata.get("codebook_size", codebook_vectors.size(1)))
        if num_codebooks is not None and int(num_codebooks) != levels:
            raise ValueError("num_codebooks does not match bridge")
        if codebook_size is not None and int(codebook_size) != size:
            raise ValueError("codebook_size does not match bridge")
        if codebook_vectors.shape[:2] != (levels, size):
            raise ValueError("codebook_vectors shape does not match bridge metadata")
        if missing_semantic_strategy not in {"filter", "error"}:
            raise ValueError("missing_semantic_strategy must be filter or error")

        self.num_codebooks = levels
        self.codebook_size = size
        self.vocab = SemanticTokenVocabulary(levels, size)
        self.max_history_items = int(max_history_items)
        required_token_len = 1 + levels * (self.max_history_items + 1)
        self.max_token_sequence_len = (
            required_token_len
            if max_token_sequence_len is None
            else int(max_token_sequence_len)
        )
        if self.max_token_sequence_len < required_token_len:
            raise ValueError(
                f"max_token_sequence_len={self.max_token_sequence_len} is smaller than "
                f"required training length {required_token_len}"
            )
        self.include_bos_to_first_item_loss = bool(include_bos_to_first_item_loss)
        self.missing_semantic_strategy = missing_semantic_strategy
        self.beam_size = int(beam_size)
        self.adaptive_beam = bool(adaptive_beam)
        self.max_beam_size = max(self.beam_size, int(max_beam_size))
        self.target_unique_items = int(target_unique_items)
        self.exact_fill = bool(exact_fill)
        self.exact_fill_on_validation = bool(exact_fill_on_validation)
        self.compute_full_auc = bool(compute_full_auc)
        self.compute_full_auc_on_validation = bool(compute_full_auc_on_validation)
        self.exact_prefix_batch_size = max(1, int(exact_prefix_batch_size))
        self.register_buffer("item_to_codes", item_to_codes[:, :levels])

        embedding_dim = int(codebook_vectors.size(-1))
        self.token_embedding = torch.nn.Embedding(
            self.vocab.vocab_size, embedding_dim, padding_idx=PAD_ID
        )
        with torch.no_grad():
            self.token_embedding.weight.zero_()
            torch.nn.init.normal_(
                self.token_embedding.weight[BOS_ID], std=embedding_dim**-0.5
            )
            for level in range(levels):
                start = self.vocab.token_id(level, 0)
                self.token_embedding.weight[start : start + size].copy_(
                    codebook_vectors[level]
                )
        self.freeze_semantic_token_embeddings = bool(freeze_semantic_token_embeddings)
        # PAD is protected by padding_idx. When the optional freeze is enabled, keep
        # BOS trainable while masking gradients for only the semantic rows.
        if self.freeze_semantic_token_embeddings:
            gradient_mask = torch.zeros_like(self.token_embedding.weight)
            gradient_mask[BOS_ID] = 1.0
            self.register_buffer("_token_gradient_mask", gradient_mask)
            self.token_embedding.weight.register_hook(
                lambda grad: grad * self._token_gradient_mask
            )
        self.token_projection = torch.nn.Linear(embedding_dim, self.item_embedding_dim)
        self.position_embedding = torch.nn.Embedding(
            self.max_token_sequence_len, self.item_embedding_dim
        )
        self.level_embedding = torch.nn.Embedding(
            levels + 2, self.item_embedding_dim, padding_idx=0
        )
        self.code_heads = torch.nn.ModuleList(
            [torch.nn.Linear(self.item_embedding_dim, size) for _ in range(levels)]
        )

        # The inherited item embedding is deliberately unused in this route. Freezing it
        # prevents a second trainable semantic table from entering the experiment.
        for parameter in self.embeddings.parameters():
            parameter.requires_grad = False

        valid_ids: List[int] = []
        paths: List[Tuple[int, ...]] = []
        self._path_to_items: Dict[Tuple[int, ...], List[int]] = {}
        candidate_ids = getattr(self.candidate_index, "ids", None)
        allowed_item_ids = (
            set(candidate_ids.detach().cpu().view(-1).tolist())
            if candidate_ids is not None
            else None
        )
        for item_id in range(1, item_to_codes.size(0)):
            if allowed_item_ids is not None and item_id not in allowed_item_ids:
                continue
            path = tuple(int(x) for x in item_to_codes[item_id, :levels].tolist())
            if len(path) != levels or min(path) < 0 or max(path) >= size:
                continue
            valid_ids.append(item_id)
            paths.append(path)
            self._path_to_items.setdefault(path, []).append(item_id)
        self.trie = SemanticPathTrie(paths)
        self.register_buffer(
            "_valid_item_ids", torch.tensor(valid_ids, dtype=torch.long)
        )
        self.register_buffer(
            "_valid_item_codes", self.item_to_codes[self._valid_item_ids]
        )
        collision_paths = sum(len(items) > 1 for items in self._path_to_items.values())
        collision_items = sum(
            len(items) for items in self._path_to_items.values() if len(items) > 1
        )
        self._static_audit = {
            "valid_semantic_items": len(valid_ids),
            "unique_legal_paths": len(self._path_to_items),
            "semantic_collision_paths": collision_paths,
            "semantic_collision_items": collision_items,
        }
        self.beam_metrics = copy.deepcopy(self.metrics)
        self.exact_metrics = copy.deepcopy(self.metrics)
        self._is_test_epoch = False
        self._reset_decode_audit()

    def _reset_decode_audit(self) -> None:
        self._decode_audit = {
            "rows": 0,
            "legal_complete_paths": 0,
            "paths_before_mapping": 0,
            "unique_items_before_fill": 0,
            "collision_items": 0,
            "beam_only_candidates": 0,
            "exact_filled": 0,
            "actual_beam_size": 0,
            "missing_history_items": 0,
            "beam_inference_seconds": 0.0,
            "exact_fill_seconds": 0.0,
            "exact_global_seconds": 0.0,
        }

    def _expand(
        self, ids: torch.Tensor, lengths: torch.Tensor, timestamps: torch.Tensor
    ) -> FullTokenBatch:
        result = expand_item_history(
            ids,
            lengths,
            timestamps,
            self.item_to_codes,
            self.vocab,
            self.missing_semantic_strategy,
        )
        self._decode_audit["missing_history_items"] += result.missing_item_count
        return result

    def _embed_tokens(
        self, token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_ids.size(1) > self.max_token_sequence_len:
            raise ValueError(
                f"token sequence length {token_ids.size(1)} exceeds configured {self.max_token_sequence_len}"
            )
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(
            0
        )
        types = token_type_ids(token_ids, self.vocab)
        projected = self.token_projection(self.token_embedding(token_ids))
        embeddings = (
            projected + self.position_embedding(positions) + self.level_embedding(types)
        )
        valid = (token_ids != PAD_ID).unsqueeze(-1).to(embeddings.dtype)
        return embeddings * valid, valid

    def forward_tokens(self, token_batch: FullTokenBatch) -> torch.Tensor:
        original_width = token_batch.token_ids.size(1)
        if original_width > self.max_token_sequence_len:
            raise ValueError(
                f"token sequence length {original_width} exceeds configured "
                f"{self.max_token_sequence_len}"
            )
        token_ids = token_batch.token_ids
        timestamps = token_batch.timestamps
        if original_width < self.max_token_sequence_len:
            pad_width = self.max_token_sequence_len - original_width
            token_ids = F.pad(token_ids, (0, pad_width), value=PAD_ID)
            timestamps = F.pad(timestamps, (0, pad_width), value=0)
        embeddings, valid = self._embed_tokens(token_ids)
        encoded, _ = self.sequence_encoder(
            past_lengths=token_batch.lengths,
            user_embeddings=embeddings,
            valid_mask=valid,
            past_payloads={"timestamps": timestamps},
        )
        return self.postprocessor(encoded)[:, :original_width]

    def next_token_loss(
        self, token_batch: FullTokenBatch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _, targets, supervision = shifted_next_token_targets(
            token_batch, self.vocab, self.include_bos_to_first_item_loss
        )
        encoded = self.forward_tokens(token_batch)[:, :-1]
        losses: List[torch.Tensor] = []
        diagnostics: Dict[str, torch.Tensor] = {}
        for level, head in enumerate(self.code_heads):
            level_start = self.vocab.token_id(level, 0)
            level_mask = (
                supervision
                & (targets >= level_start)
                & (targets < level_start + self.codebook_size)
            )
            if bool(level_mask.any()):
                logits = head(encoded[level_mask])
                labels = targets[level_mask] - level_start
                level_loss = F.cross_entropy(logits, labels)
                accuracy = (logits.argmax(dim=-1) == labels).float().mean()
                losses.append(level_loss)
            else:
                level_loss = encoded.sum() * 0.0
                accuracy = torch.zeros((), device=encoded.device)
            diagnostics[f"loss_level_{level}"] = level_loss
            diagnostics[f"accuracy_level_{level}"] = accuracy
            diagnostics[f"count_level_{level}"] = level_mask.sum()
        loss = torch.stack(losses).mean() if losses else encoded.sum() * 0.0
        diagnostics["loss"] = loss
        return loss, diagnostics

    def _history_with_prefix(
        self, history: FullTokenBatch, prefix: Sequence[int]
    ) -> FullTokenBatch:
        tokens = [self.vocab.token_id(level, code) for level, code in enumerate(prefix)]
        n = int(history.lengths[0].item())
        total = n + len(tokens)
        ids = torch.zeros((1, total), dtype=torch.long, device=history.token_ids.device)
        ts = torch.zeros_like(ids)
        ids[0, :n] = history.token_ids[0, :n]
        ts[0, :n] = history.timestamps[0, :n]
        if tokens:
            ids[0, n:] = torch.tensor(tokens, dtype=torch.long, device=ids.device)
            ts[0, n:] = history.timestamps[0, max(0, n - 1)]
        lengths = torch.tensor([total], dtype=torch.long, device=ids.device)
        return FullTokenBatch(ids, lengths, ts, token_type_ids(ids, self.vocab))

    def _next_logits_single(
        self, history: FullTokenBatch, prefix: Sequence[int], level: int
    ) -> torch.Tensor:
        augmented = self._history_with_prefix(history, prefix)
        encoded = self.forward_tokens(augmented)
        last = encoded[0, augmented.lengths[0] - 1]
        return self.code_heads[level](last)

    def _next_logits_for_prefixes(
        self, history: FullTokenBatch, prefixes: Sequence[Sequence[int]], level: int
    ) -> torch.Tensor:
        """Batch equal-depth prefixes; mathematically identical to repeated full recomputation."""
        if not prefixes:
            return torch.empty((0, self.codebook_size), device=history.token_ids.device)
        if history.token_ids.size(0) != 1:
            raise ValueError("batched prefix scoring expects one history")
        depth = len(prefixes[0])
        if any(len(prefix) != depth for prefix in prefixes):
            raise ValueError("all prefixes in one batch must have equal depth")
        base_n = int(history.lengths[0].item())
        total = base_n + depth
        batch_size = len(prefixes)
        ids = torch.zeros(
            (batch_size, total), dtype=torch.long, device=history.token_ids.device
        )
        timestamps = torch.zeros_like(ids)
        ids[:, :base_n] = history.token_ids[0, :base_n]
        timestamps[:, :base_n] = history.timestamps[0, :base_n]
        if depth:
            suffix = torch.tensor(
                [
                    [self.vocab.token_id(i, code) for i, code in enumerate(prefix)]
                    for prefix in prefixes
                ],
                dtype=torch.long,
                device=ids.device,
            )
            ids[:, base_n:] = suffix
            timestamps[:, base_n:] = history.timestamps[0, base_n - 1]
        lengths = torch.full((batch_size,), total, dtype=torch.long, device=ids.device)
        encoded = self.forward_tokens(
            FullTokenBatch(ids, lengths, timestamps, token_type_ids(ids, self.vocab))
        )
        return self.code_heads[level](encoded[:, total - 1])

    @torch.inference_mode()
    def generate_path_greedy(self, history: FullTokenBatch) -> Tuple[int, ...]:
        prefix: Tuple[int, ...] = ()
        for level in range(self.num_codebooks):
            logits = self._next_logits_single(history, prefix, level)
            legal = self.trie.legal_children(prefix)
            if not legal:
                raise RuntimeError(f"no legal continuation for prefix {prefix}")
            legal_tensor = torch.tensor(legal, dtype=torch.long, device=logits.device)
            code = int(legal_tensor[logits[legal_tensor].argmax()].item())
            prefix += (code,)
        return prefix

    @torch.inference_mode()
    def _beam_paths(
        self, history: FullTokenBatch, width: int
    ) -> List[Tuple[Tuple[int, ...], float]]:
        beams: List[Tuple[Tuple[int, ...], float]] = [((), 0.0)]
        for level in range(self.num_codebooks):
            candidates: List[Tuple[Tuple[int, ...], float]] = []
            logits_by_prefix: Dict[Tuple[int, ...], torch.Tensor] = {}
            prefixes = [prefix for prefix, _ in beams]
            for start in range(0, len(prefixes), self.exact_prefix_batch_size):
                chunk = prefixes[start : start + self.exact_prefix_batch_size]
                chunk_logits = self._next_logits_for_prefixes(history, chunk, level)
                logits_by_prefix.update(
                    {prefix: chunk_logits[index] for index, prefix in enumerate(chunk)}
                )
            for prefix, score in beams:
                legal = self.trie.legal_children(prefix)
                if not legal:
                    continue
                logits = logits_by_prefix[prefix]
                legal_ids = torch.tensor(legal, dtype=torch.long, device=logits.device)
                legal_logp = torch.log_softmax(logits[legal_ids], dim=0)
                keep = min(width, len(legal))
                vals, indices = torch.topk(legal_logp, keep)
                for value, index in zip(vals.tolist(), indices.tolist()):
                    code = int(legal_ids[index].item())
                    candidates.append((prefix + (code,), score + float(value)))
            candidates.sort(key=lambda pair: pair[1], reverse=True)
            beams = candidates[:width]
        return beams

    @torch.inference_mode()
    def exact_item_scores(
        self, history: FullTokenBatch, item_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """True sum of three full-head log probabilities after real prefix appends."""
        if history.token_ids.size(0) != 1:
            raise ValueError("exact_item_scores currently scores one user at a time")
        ids = (
            self._valid_item_ids
            if item_ids is None
            else item_ids.long().to(self._valid_item_ids.device)
        )
        kept: List[int] = []
        paths: List[Tuple[int, ...]] = []
        for item_id_tensor in ids:
            item_id = int(item_id_tensor.item())
            if not 0 < item_id < self.item_to_codes.size(0):
                continue
            codes = tuple(int(x) for x in self.item_to_codes[item_id].tolist())
            if min(codes) < 0 or max(codes) >= self.codebook_size:
                continue
            kept.append(item_id)
            paths.append(codes)
        if not paths:
            return torch.empty((0,), device=history.token_ids.device), torch.empty(
                (0,), dtype=torch.long, device=history.token_ids.device
            )

        path_scores: Dict[Tuple[int, ...], torch.Tensor] = {
            path: torch.zeros((), device=history.token_ids.device)
            for path in set(paths)
        }
        for level in range(self.num_codebooks):
            unique_prefixes = sorted({path[:level] for path in paths})
            prefix_log_probs: Dict[Tuple[int, ...], torch.Tensor] = {}
            for start in range(0, len(unique_prefixes), self.exact_prefix_batch_size):
                chunk = unique_prefixes[start : start + self.exact_prefix_batch_size]
                logits = self._next_logits_for_prefixes(history, chunk, level)
                log_probs = torch.log_softmax(logits, dim=-1)
                prefix_log_probs.update(
                    {prefix: log_probs[index] for index, prefix in enumerate(chunk)}
                )
            for path in path_scores:
                path_scores[path] = (
                    path_scores[path] + prefix_log_probs[path[:level]][path[level]]
                )
        scores = torch.stack([path_scores[path] for path in paths])
        return scores, torch.tensor(
            kept, dtype=torch.long, device=history.token_ids.device
        )

    @torch.inference_mode()
    def _retrieve_one(
        self,
        history: FullTokenBatch,
        invalid_ids: set[int],
        k: int,
        allow_exact_fill: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        width = self.beam_size
        beam_started = time.perf_counter()
        paths: List[Tuple[Tuple[int, ...], float]] = []
        scored: Dict[int, float] = {}
        collision_items = 0
        while True:
            paths = self._beam_paths(history, width)
            scored = {}
            collision_items = 0
            for path, score in paths:
                mapped = self._path_to_items.get(path, [])
                if len(mapped) > 1:
                    collision_items += len(mapped)
                for item_id in mapped:
                    if item_id not in invalid_ids:
                        scored[item_id] = score
            if (
                not self.adaptive_beam
                or len(scored) >= self.target_unique_items
                or width >= self.max_beam_size
            ):
                break
            width = min(self.max_beam_size, width * 2)
        beam_seconds = time.perf_counter() - beam_started

        beam_only = len(scored)
        beam_ranked = sorted(scored.items(), key=lambda pair: (-pair[1], pair[0]))[:k]
        self._last_beam_only_ids = torch.zeros(
            (k,), dtype=torch.long, device=history.token_ids.device
        )
        if beam_ranked:
            self._last_beam_only_ids[: len(beam_ranked)] = torch.tensor(
                [x[0] for x in beam_ranked], device=history.token_ids.device
            )
        exact_filled = 0
        ranked = list(beam_ranked)
        exact_fill_seconds = 0.0
        use_exact_fill = (
            self.exact_fill if allow_exact_fill is None else allow_exact_fill
        )
        if use_exact_fill and len(scored) < k:
            exact_started = time.perf_counter()
            exact_scores, exact_ids = self.exact_item_scores(history)
            order = torch.argsort(exact_scores, descending=True)
            for index in order.tolist():
                item_id = int(exact_ids[index].item())
                if item_id in invalid_ids or item_id in scored:
                    continue
                scored[item_id] = float(exact_scores[index].item())
                ranked.append((item_id, float(exact_scores[index].item())))
                exact_filled += 1
                if len(scored) >= k:
                    break
            exact_fill_seconds = time.perf_counter() - exact_started
        ranked = ranked[:k]
        out_ids = torch.zeros((k,), dtype=torch.long, device=history.token_ids.device)
        out_scores = torch.full((k,), float("-inf"), device=history.token_ids.device)
        if ranked:
            out_ids[: len(ranked)] = torch.tensor(
                [x[0] for x in ranked], device=out_ids.device
            )
            out_scores[: len(ranked)] = torch.tensor(
                [x[1] for x in ranked], device=out_scores.device
            )
        audit = {
            "legal_complete_paths": len(paths),
            "paths_before_mapping": len(paths),
            "unique_items_before_fill": beam_only,
            "collision_items": collision_items,
            "beam_only_candidates": beam_only,
            "exact_filled": exact_filled,
            "actual_beam_size": width,
            "beam_inference_seconds": beam_seconds,
            "exact_fill_seconds": exact_fill_seconds,
        }
        return out_ids, out_scores, audit

    @torch.inference_mode()
    def retrieve_tokens(
        self,
        histories: FullTokenBatch,
        past_item_ids: torch.Tensor,
        filter_past_ids: bool = True,
        allow_exact_fill: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k = getattr(self.candidate_index, "_k", self.target_unique_items)
        all_ids, all_scores, all_beam_ids = [], [], []
        for row in range(histories.token_ids.size(0)):
            n = int(histories.lengths[row].item())
            one = FullTokenBatch(
                histories.token_ids[row : row + 1, :n],
                histories.lengths[row : row + 1],
                histories.timestamps[row : row + 1, :n],
                histories.type_ids[row : row + 1, :n],
            )
            invalid = (
                set(past_item_ids[row][past_item_ids[row] > 0].tolist())
                if filter_past_ids
                else set()
            )
            ids, scores, audit = self._retrieve_one(
                one, invalid, k, allow_exact_fill=allow_exact_fill
            )
            all_ids.append(ids)
            all_scores.append(scores)
            all_beam_ids.append(self._last_beam_only_ids.clone())
            self._decode_audit["rows"] += 1
            for key, value in audit.items():
                self._decode_audit[key] += value
        self._last_beam_top_ids = torch.stack(all_beam_ids)
        return torch.stack(all_ids), torch.stack(all_scores)

    def _training_token_batch(self, batch: Dict[str, torch.Tensor]) -> FullTokenBatch:
        history_ids = batch["historical_ids"].to(self.device)
        history_ts = batch["historical_timestamps"].to(self.device)
        lengths = batch["history_lengths"].to(self.device)
        targets = batch["target_ids"].to(self.device).view(-1)
        target_ts = batch["target_timestamps"].to(self.device).view(-1)
        batch_size, width = history_ids.shape
        items = torch.zeros(
            (batch_size, width + 1), dtype=torch.long, device=self.device
        )
        times = torch.zeros_like(items)
        items[:, :width] = history_ids
        times[:, :width] = history_ts
        rows = torch.arange(batch_size, device=self.device)
        items[rows, lengths] = targets
        times[rows, lengths] = target_ts
        return self._expand(items, lengths + 1, times)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        token_batch = self._training_token_batch(batch)
        loss, diagnostics = self.next_token_loss(token_batch)
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train/missing_semantic_items",
            float(token_batch.missing_item_count),
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        for key, value in diagnostics.items():
            if key != "loss":
                self.log(
                    f"train/{key}",
                    value.float(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )
        return loss

    def _history_from_row(self, batch: Dict[str, torch.Tensor]) -> FullTokenBatch:
        features, _, _ = seq_features_from_row(batch, self.device, max_output_length=0)
        return self._expand(
            features.past_ids,
            features.past_lengths,
            features.past_payloads["timestamps"],
        )

    @torch.inference_mode()
    def _full_auc_ranks(
        self,
        histories: FullTokenBatch,
        past_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ranks, counts, top_ids = [], [], []
        k = getattr(self.candidate_index, "_k", self.target_unique_items)
        for row in range(histories.token_ids.size(0)):
            n = int(histories.lengths[row].item())
            one = FullTokenBatch(
                histories.token_ids[row : row + 1, :n],
                histories.lengths[row : row + 1],
                histories.timestamps[row : row + 1, :n],
                histories.type_ids[row : row + 1, :n],
            )
            exact_started = time.perf_counter()
            scores, ids = self.exact_item_scores(one)
            self._decode_audit["exact_global_seconds"] += (
                time.perf_counter() - exact_started
            )
            invalid = set(past_ids[row][past_ids[row] > 0].tolist())
            if invalid:
                invalid_mask = torch.tensor(
                    [int(x.item()) in invalid for x in ids], device=scores.device
                )
                scores[invalid_mask] = float("-inf")
            order = torch.argsort(scores, descending=True)
            finite_order = order[torch.isfinite(scores[order])][:k]
            row_top = torch.zeros((k,), dtype=torch.long, device=scores.device)
            row_top[: finite_order.numel()] = ids[finite_order]
            top_ids.append(row_top)
            target = int(target_ids[row].item())
            target_mask = ids == target
            finite_count = torch.isfinite(scores).sum().float()
            if not bool(target_mask.any()) or not bool(
                torch.isfinite(scores[target_mask][0])
            ):
                ranks.append(finite_count)
            else:
                target_score = scores[target_mask][0]
                better = (scores > target_score).sum().float()
                ties = (scores == target_score).sum().float()
                ranks.append(1.0 + better + 0.5 * (ties - 1.0))
            counts.append(finite_count)
        return torch.stack(ranks), torch.stack(counts), torch.stack(top_ids)

    def on_validation_epoch_start(self) -> None:
        self._is_test_epoch = False
        self.metrics.reset()
        self.beam_metrics.reset()
        self.exact_metrics.reset()
        self._exact_metric_updated = False
        self._reset_decode_audit()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        histories = self._history_from_row(batch)
        target_ids = batch["target_ids"].to(self.device).view(-1, 1)
        past_ids = batch["historical_ids"].to(self.device)
        top_ids, _ = self.retrieve_tokens(
            histories,
            past_ids,
            allow_exact_fill=self.exact_fill
            and (self._is_test_epoch or self.exact_fill_on_validation),
        )
        auc_ranks = auc_counts = exact_top_ids = None
        if self.compute_full_auc and (
            self._is_test_epoch or self.compute_full_auc_on_validation
        ):
            auc_ranks, auc_counts, exact_top_ids = self._full_auc_ranks(
                histories, past_ids, target_ids.view(-1)
            )
        cold = batch.get("target_is_cold")
        train_count = batch.get("target_train_count")
        self.metrics.update(
            top_k_ids=top_ids,
            target_ids=target_ids,
            target_is_cold=None if cold is None else cold.to(self.device).view(-1),
            target_train_count=None
            if train_count is None
            else train_count.to(self.device).view(-1),
            auc_ranks=auc_ranks,
            auc_num_candidates=auc_counts,
        )
        common = {
            "target_ids": target_ids,
            "target_is_cold": None if cold is None else cold.to(self.device).view(-1),
            "target_train_count": None
            if train_count is None
            else train_count.to(self.device).view(-1),
        }
        self.beam_metrics.update(
            top_k_ids=self._last_beam_top_ids,
            auc_ranks=None,
            auc_num_candidates=None,
            **common,
        )
        if exact_top_ids is not None:
            self._exact_metric_updated = True
            self.exact_metrics.update(
                top_k_ids=exact_top_ids,
                auc_ranks=auc_ranks,
                auc_num_candidates=auc_counts,
                **common,
            )

    def _epoch_end(self, prefix: str):
        results = self.metrics.compute()
        for key, value in results.items():
            self.log(
                f"{prefix}/{key}", value, on_epoch=True, prog_bar=True, logger=True
            )
        rows = max(1, self._decode_audit["rows"])
        for key, value in self._decode_audit.items():
            if key != "rows":
                self.log(
                    f"{prefix}/{key}", float(value) / rows, on_epoch=True, logger=True
                )
        for key, value in self._static_audit.items():
            self.log(f"{prefix}/{key}", float(value), on_epoch=True, logger=True)
        self.log(
            f"{prefix}/trainable_parameters",
            float(
                sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if parameter.requires_grad
                )
            ),
            on_epoch=True,
            logger=True,
        )
        if torch.cuda.is_available():
            self.log(
                f"{prefix}/peak_memory_mb",
                float(torch.cuda.max_memory_allocated(self.device)) / (1024**2),
                on_epoch=True,
                logger=True,
            )
        metric_groups = [("beam_only", self.beam_metrics)]
        if self._exact_metric_updated:
            metric_groups.append(("exact_global", self.exact_metrics))
        for metric_prefix, metric in metric_groups:
            for key, value in metric.compute().items():
                self.log(
                    f"{prefix}/{metric_prefix}/{key}", value, on_epoch=True, logger=True
                )
        self.metrics.reset()
        self.beam_metrics.reset()
        self.exact_metrics.reset()
        monitor = self.configure_optimizer_params.get("monitor")
        return results.get(monitor.split("/", 1)[1]) if monitor else None

    def on_validation_epoch_end(self):
        return self._epoch_end("val")

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()
        self._is_test_epoch = True

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self._epoch_end("test")

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, List]:
        histories = self._history_from_row(batch)
        ids, scores = self.retrieve_tokens(
            histories, batch["historical_ids"].to(self.device)
        )
        return {"top_k_ids": ids.cpu().tolist(), "top_k_scores": scores.cpu().tolist()}
