import csv
import gin
import json
import os
import numpy as np
import time
import torch
import wandb
import warnings

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.schemas import SeqBatch
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from modules.quantize import QuantizeForwardMode
from modules.rqvae import RqVae
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import parse_config
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm
from typing import Dict, Optional


class CsvMetricLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._needs_header = not os.path.exists(path)

    def log_dict(self, step: int, split: str, metrics: Dict[str, float]) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            if self._needs_header:
                writer.writerow(["step", "split", "metric", "value"])
                self._needs_header = False
            for key, value in metrics.items():
                writer.writerow([step, split, key, float(value)])


def _build_item_seq_batch(ids: torch.Tensor, x: torch.Tensor) -> SeqBatch:
    neg_ids = torch.full_like(ids, -1)
    return SeqBatch(
        user_ids=neg_ids,
        ids=ids,
        ids_fut=neg_ids,
        x=x,
        x_fut=neg_ids,
        seq_mask=torch.ones_like(ids, dtype=torch.bool),
    )


def _sample_item_batch_gpu(item_x: torch.Tensor, batch_size: int) -> SeqBatch:
    ids = torch.randint(
        low=0,
        high=item_x.shape[0],
        size=(batch_size,),
        device=item_x.device,
    )
    x = item_x.index_select(0, ids)
    return _build_item_seq_batch(ids=ids, x=x)


def _resolve_auto_bool(value, *, default: bool) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return default
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"Unsupported bool-like value: {value}")
    return bool(value)


def _resolve_quantize_mode(
    vae_quantize_mode: Optional[str],
    vae_codebook_mode: QuantizeForwardMode,
) -> QuantizeForwardMode:
    if vae_quantize_mode is None:
        return vae_codebook_mode
    name = vae_quantize_mode.lower().strip()
    mapping = {
        "gumbel": QuantizeForwardMode.GUMBEL_SOFTMAX,
        "ste": QuantizeForwardMode.STE,
        "rotation": QuantizeForwardMode.ROTATION_TRICK,
        "ema": QuantizeForwardMode.EMA,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported vae_quantize_mode: {vae_quantize_mode}")
    return mapping[name]


def _load_cold_item_ids(path: Optional[str]) -> list[int]:
    if path is None or not os.path.exists(path):
        return []
    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "cold_item_ids" in data:
            data = data["cold_item_ids"]
        if isinstance(data, torch.Tensor):
            return data.long().tolist()
        return list(data)
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "cold_item_ids" in data:
            data = data["cold_item_ids"]
        return list(data)
    raise ValueError(f"Unsupported cold item file format: {path}")


def _export_bridge_artifact(
    export_path: str,
    external_item_ids: torch.Tensor,
    corpus_ids: torch.Tensor,
    rqvae: RqVae,
    vae_n_layers: int,
    cold_item_ids_path: Optional[str],
) -> None:
    external_item_ids = external_item_ids.long().cpu()
    corpus_ids = corpus_ids.long().cpu()
    max_item_id = int(external_item_ids.max().item())

    item_id_to_codes = torch.full((max_item_id + 1, vae_n_layers), -1, dtype=torch.long)
    item_id_to_codes[external_item_ids] = corpus_ids[:, :vae_n_layers]

    codebook_vectors = []
    for layer in rqvae.layers:
        ids = torch.arange(layer.n_embed, device=layer.device)
        codebook_vectors.append(layer.get_item_embeddings(ids).detach().cpu())

    cold_item_mask = torch.zeros(max_item_id + 1, dtype=torch.bool)
    cold_item_ids = _load_cold_item_ids(cold_item_ids_path)
    if cold_item_ids:
        cold_item_ids = torch.as_tensor(cold_item_ids, dtype=torch.long)
        cold_item_ids = cold_item_ids[(cold_item_ids >= 0) & (cold_item_ids <= max_item_id)]
        cold_item_mask[cold_item_ids] = True

    export_dir = os.path.dirname(export_path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
    torch.save(
        {
            "item_id_to_codes": item_id_to_codes,
            "codebook_vectors": torch.stack(codebook_vectors, dim=0),
            "item_ids": external_item_ids,
            "cold_item_mask": cold_item_mask,
            "metadata": {
                "n_codebooks": vae_n_layers,
                "codebook_size": rqvae.codebook_size,
                "embed_dim": rqvae.embed_dim,
            },
        },
        export_path,
    )


@gin.configurable
def train(
    iterations=50000,
    batch_size=64,
    learning_rate=0.0001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    save_dir_root="out/",
    use_kmeans_init=True,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    do_eval=True,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    eval_every=50000,
    commitment_weight=0.25,
    vae_n_cat_feats=18,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_codebook_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
    vae_quantize_mode=None,
    vae_sim_vq=False,
    vae_n_layers=3,
    vae_ema_decay=0.99,
    vae_ema_eps=1e-5,
    vae_dead_code_threshold=1.0,
    vae_dead_code_reset=False,
    dataset_split="beauty",
    text_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    text_batch_size=64,
    text_cache_path=None,
    text_encoder_device="auto",
    print_every=50,
    profile_every=20,
    train_log_every=20,
    csv_log_every=20,
    gpu_resident_sampling="auto",
    dataloader_num_workers=0,
    dataloader_pin_memory=None,
    dataloader_prefetch_factor=2,
    torch_num_threads=0,
    csv_log_path=None,
    export_bridge_path=None,
    cold_item_ids_path=None,
):
    if wandb_logging:
        params = locals()

    if csv_log_path is None:
        csv_log_path = os.path.join(save_dir_root, "metrics.csv")
    csv_logger = CsvMetricLogger(csv_log_path)
    print_every = max(1, int(print_every))
    profile_every = max(1, int(profile_every))
    train_log_every = max(1, int(train_log_every))
    csv_log_every = max(1, int(csv_log_every))
    dataloader_num_workers = int(dataloader_num_workers)
    torch_num_threads = int(torch_num_threads)
    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)

    codebook_mode = _resolve_quantize_mode(vae_quantize_mode, vae_codebook_mode)

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )
    device = accelerator.device

    data_kwargs = {
        "text_encoder_name": text_encoder_name,
        "text_batch_size": text_batch_size,
        "text_cache_path": text_cache_path,
        "text_encoder_device": text_encoder_device,
    }

    train_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        train_test_split="train" if do_eval else "all",
        split=dataset_split,
        **data_kwargs,
    )
    use_gpu_resident_sampling = _resolve_auto_bool(
        gpu_resident_sampling, default=(device.type == "cuda")
    )
    train_item_x_gpu = None
    train_dataloader = None
    if use_gpu_resident_sampling:
        train_item_x_gpu = train_dataset.item_data.to(device, non_blocking=True)
    else:
        train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=None,
            collate_fn=lambda batch: batch,
            num_workers=dataloader_num_workers,
            pin_memory=(
                (device.type == "cuda")
                if dataloader_pin_memory is None
                else bool(dataloader_pin_memory)
            ),
            persistent_workers=(dataloader_num_workers > 0),
            prefetch_factor=(
                int(dataloader_prefetch_factor)
                if dataloader_num_workers > 0
                else None
            ),
        )
        train_dataloader = cycle(train_dataloader)

    if do_eval:
        eval_dataset = ItemData(
            root=dataset_folder,
            dataset=dataset,
            force_process=False,
            train_test_split="eval",
            split=dataset_split,
            **data_kwargs,
        )
        eval_sampler = BatchSampler(RandomSampler(eval_dataset), batch_size, False)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=None,
            collate_fn=lambda batch: batch,
            num_workers=dataloader_num_workers,
            pin_memory=(
                (device.type == "cuda")
                if dataloader_pin_memory is None
                else bool(dataloader_pin_memory)
            ),
            persistent_workers=(dataloader_num_workers > 0),
            prefetch_factor=(
                int(dataloader_prefetch_factor)
                if dataloader_num_workers > 0
                else None
            ),
        )

    index_dataset = (
        ItemData(
            root=dataset_folder,
            dataset=dataset,
            force_process=False,
            train_test_split="all",
            split=dataset_split,
            **data_kwargs,
        )
        if do_eval
        else train_dataset
    )

    if train_dataloader is not None:
        train_dataloader = accelerator.prepare(train_dataloader)

    if accelerator.is_main_process:
        print(
            "[train_rqvae] "
            f"device={device}, text_encoder_device={text_encoder_device}, "
            f"train_items={len(train_dataset)}, do_eval={do_eval}, batch_size={batch_size}, "
            f"dataloader_num_workers={dataloader_num_workers}, "
            f"gpu_resident_sampling={use_gpu_resident_sampling}",
            flush=True,
        )

    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=use_kmeans_init and pretrained_rqvae_path is None,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=codebook_mode,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight,
        codebook_ema_decay=vae_ema_decay,
        codebook_ema_eps=vae_ema_eps,
        codebook_dead_code_threshold=vae_dead_code_threshold,
        codebook_dead_code_reset=vae_dead_code_reset,
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    wandb_active = False
    if wandb_logging and accelerator.is_main_process:
        try:
            wandb.login()
            wandb.init(project="rq-vae-training", config=params)
            wandb_active = True
        except Exception as exc:
            warnings.warn(
                f"WandB initialization failed ({exc}). Continuing with CSV-only logging."
            )

    start_iter = 0
    if pretrained_rqvae_path is not None:
        model.load_pretrained(pretrained_rqvae_path)
        state = torch.load(pretrained_rqvae_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"] + 1

    model, optimizer = accelerator.prepare(model, optimizer)

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq,
    )
    tokenizer.rq_vae = model

    prev_used_codes: list[set[int]] = [set() for _ in range(vae_n_layers)]

    with tqdm(
        initial=start_iter,
        total=start_iter + iterations,
        disable=not accelerator.is_main_process,
        mininterval=1.0,
    ) as pbar:
        for iter_idx in range(start_iter, start_iter + iterations):
            model.train()
            total_loss = 0
            t = 0.2
            step_start_time = time.perf_counter()
            data_time_s = 0.0
            compute_time_s = 0.0

            if (
                iter_idx == start_iter
                and use_kmeans_init
                and pretrained_rqvae_path is None
            ):
                if use_gpu_resident_sampling:
                    kmeans_ids = torch.arange(
                        min(20000, train_item_x_gpu.shape[0]), device=device
                    )
                    kmeans_x = train_item_x_gpu.index_select(0, kmeans_ids)
                    kmeans_init_data = _build_item_seq_batch(ids=kmeans_ids, x=kmeans_x)
                else:
                    kmeans_init_data = batch_to(
                        train_dataset[torch.arange(min(20000, len(train_dataset)))], device
                    )
                model(kmeans_init_data, t)

            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data_start_time = time.perf_counter()
                if use_gpu_resident_sampling:
                    data = _sample_item_batch_gpu(
                        item_x=train_item_x_gpu, batch_size=batch_size
                    )
                else:
                    data = next_batch(train_dataloader, device)
                data_time_s += time.perf_counter() - data_start_time
                if accelerator.is_main_process and iter_idx == start_iter:
                    model_device = next(model.parameters()).device
                    print(
                        "[train_rqvae] "
                        f"first_batch_x_device={data.x.device}, model_device={model_device}",
                        flush=True,
                    )
                compute_start_time = time.perf_counter()
                with accelerator.autocast():
                    model_output = model(data, gumbel_t=t)
                    loss = model_output.loss / gradient_accumulate_every
                    total_loss += loss
                compute_time_s += time.perf_counter() - compute_start_time

            accelerator.backward(total_loss)
            optimizer.step()
            accelerator.wait_for_everyone()
            step_time_s = time.perf_counter() - step_start_time

            eval_log = {}
            step = iter_idx + 1
            is_last_step = step == (start_iter + iterations)
            needs_eval = do_eval and (step % eval_every == 0 or is_last_step)
            should_print = step % print_every == 0 or is_last_step
            should_profile = step % profile_every == 0
            should_train_log = (
                (step % train_log_every == 0)
                or (step % csv_log_every == 0)
                or should_print
                or needs_eval
                or is_last_step
            )
            train_log = {}
            if should_train_log:
                reconstruction_loss = model_output.reconstruction_loss.detach().cpu().item()
                rqvae_loss = model_output.rqvae_loss.detach().cpu().item()
                total_loss_item = total_loss.detach().cpu().item()
                emb_norms_avg = model_output.embs_norm.mean(axis=0)
                train_log = {
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss_item,
                    "reconstruction_loss": reconstruction_loss,
                    "rqvae_loss": rqvae_loss,
                    "step_time_s": step_time_s,
                    "data_time_s": data_time_s,
                    "compute_time_s": compute_time_s,
                    "reconstruction_loss_per_dim": reconstruction_loss / max(1, vae_input_dim),
                    "rqvae_loss_per_layer": rqvae_loss / max(1, vae_n_layers),
                    "temperature": t,
                    "p_unique_ids": model_output.p_unique_ids.detach().cpu().item(),
                    **{
                        f"emb_avg_norm_{i}": emb_norms_avg[i].detach().cpu().item()
                        for i in range(vae_n_layers)
                    },
                }

            if step % train_log_every == 0 or is_last_step:
                desc_loss = train_log.get("total_loss", total_loss.detach().cpu().item())
                desc_rec = train_log.get(
                    "reconstruction_loss",
                    model_output.reconstruction_loss.detach().cpu().item(),
                )
                desc_rq = train_log.get("rqvae_loss", model_output.rqvae_loss.detach().cpu().item())
                pbar.set_description(
                    f"loss: {desc_loss:.4f}, rl: {desc_rec:.4f}, vl: {desc_rq:.4f}"
                )
            if needs_eval:
                model.eval()
                with tqdm(eval_dataloader, desc=f"Eval {step}", disable=True):
                    eval_losses = [[], [], []]
                    for batch in eval_dataloader:
                        data = batch_to(batch, device)
                        with torch.no_grad():
                            eval_model_output = model(data, gumbel_t=t)
                        eval_losses[0].append(eval_model_output.loss.detach().cpu().item())
                        eval_losses[1].append(
                            eval_model_output.reconstruction_loss.detach().cpu().item()
                        )
                        eval_losses[2].append(eval_model_output.rqvae_loss.detach().cpu().item())

                eval_losses = np.array(eval_losses).mean(axis=-1)
                eval_log.update(
                    {
                        "eval_total_loss": float(eval_losses[0]),
                        "eval_reconstruction_loss": float(eval_losses[1]),
                        "eval_rqvae_loss": float(eval_losses[2]),
                        "eval_reconstruction_loss_per_dim": float(eval_losses[1]) / max(1, vae_input_dim),
                        "eval_rqvae_loss_per_layer": float(eval_losses[2]) / max(1, vae_n_layers),
                    }
                )

            if accelerator.is_main_process:
                if step % save_model_every == 0 or is_last_step:
                    state = {
                        "iter": iter_idx,
                        "model": model.state_dict(),
                        "model_config": model.config,
                        "optimizer": optimizer.state_dict(),
                    }
                    os.makedirs(save_dir_root, exist_ok=True)
                    torch.save(state, os.path.join(save_dir_root, f"checkpoint_{iter_idx}.pt"))

                if needs_eval:
                    tokenizer.reset()
                    model.eval()
                    corpus_ids = tokenizer.precompute_corpus_ids(index_dataset)
                    max_duplicates = corpus_ids[:, -1].max() / corpus_ids.shape[0]

                    _, counts = torch.unique(corpus_ids[:, :-1], dim=0, return_counts=True)
                    p = counts.float() / corpus_ids.shape[0]
                    rqvae_entropy = -(p * torch.log(p + 1e-12)).sum()

                    rqvae = accelerator.unwrap_model(model)
                    for cid in range(vae_n_layers):
                        unique_ids, layer_counts = torch.unique(
                            corpus_ids[:, cid], return_counts=True
                        )
                        probs = layer_counts.float() / layer_counts.sum()
                        entropy = -(probs * torch.log(probs + 1e-12)).sum()
                        perplexity = torch.exp(entropy)
                        used_set = set(unique_ids.detach().cpu().tolist())
                        new_codes = used_set.difference(prev_used_codes[cid])
                        prev_used_codes[cid] = used_set

                        eval_log[f"codebook_usage_{cid}"] = len(used_set) / vae_codebook_size
                        eval_log[f"codebook_entropy_{cid}"] = entropy.detach().cpu().item()
                        eval_log[f"codebook_perplexity_{cid}"] = (
                            perplexity.detach().cpu().item()
                        )
                        eval_log[f"new_code_ratio_{cid}"] = len(new_codes) / vae_codebook_size

                        layer = rqvae.layers[cid]
                        dead_mask = layer.ema_cluster_size < layer.dead_code_threshold
                        eval_log[f"dead_code_ratio_{cid}"] = dead_mask.float().mean().item()

                    eval_log["rqvae_entropy"] = rqvae_entropy.detach().cpu().item()
                    eval_log["max_id_duplicates"] = max_duplicates.detach().cpu().item()

                    if export_bridge_path is not None:
                        _export_bridge_artifact(
                            export_path=export_bridge_path,
                            external_item_ids=index_dataset.item_external_ids,
                            corpus_ids=corpus_ids,
                            rqvae=rqvae,
                            vae_n_layers=vae_n_layers,
                            cold_item_ids_path=cold_item_ids_path,
                        )

                if should_train_log and (step % csv_log_every == 0 or needs_eval or is_last_step):
                    csv_logger.log_dict(step=step, split="train", metrics=train_log)
                if eval_log:
                    csv_logger.log_dict(step=step, split="eval", metrics=eval_log)

                if wandb_active and train_log:
                    wandb.log({**train_log, **eval_log}, step=step)

                if step % print_every == 0 or step == start_iter + iterations:
                    eval_total_loss = eval_log.get("eval_total_loss")
                    eval_msg = (
                        f", eval_total_loss={eval_total_loss:.4f}"
                        if eval_total_loss is not None
                        else ""
                    )
                    gpu_msg = ""
                    if device.type == "cuda":
                        mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                        mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                        gpu_msg = (
                            f", gpu_mem_alloc_mb={mem_alloc:.1f}, "
                            f"gpu_mem_reserved_mb={mem_reserved:.1f}"
                        )
                    print(
                        f"[step {step}] total_loss={train_log['total_loss']:.4f}, "
                        f"rec={train_log['reconstruction_loss']:.4f}, "
                        f"rq={train_log['rqvae_loss']:.4f}, "
                        f"step_time_s={step_time_s:.3f}, data_time_s={data_time_s:.3f}, "
                        f"compute_time_s={compute_time_s:.3f}{eval_msg}{gpu_msg}",
                        flush=True,
                    )
                elif step % profile_every == 0:
                    print(
                        f"[profile step {step}] step_time_s={step_time_s:.3f}, "
                        f"data_time_s={data_time_s:.3f}, compute_time_s={compute_time_s:.3f}",
                        flush=True,
                    )

            pbar.update(1)

    if wandb_active and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
