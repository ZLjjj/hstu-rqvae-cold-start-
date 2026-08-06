import os
import pickle
import subprocess
from typing import Any, Optional

import hydra
import lightning as L
import torch.multiprocessing
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from generative_recommenders_pl.utils.instantiators import (
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
)
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)
torch.multiprocessing.set_sharing_strategy("file_system")


def _current_git_commit(work_dir: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=work_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # log.info(f"config: {OmegaConf.to_yaml(cfg)}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.data, _recursive_=False
    )
    if hasattr(datamodule, "initialize_cold_start_split"):
        datamodule.initialize_cold_start_split()

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(
        cfg.model, datamodule=datamodule, _recursive_=False
    )
    cold_start = getattr(datamodule, "cold_start", {})
    if (
        hasattr(datamodule, "write_negative_sampling_audit")
        and bool(cold_start.get("enabled", False))
    ):
        audit_path = datamodule.write_negative_sampling_audit(cfg.paths.output_dir)
        cold_set = set(datamodule.cold_item_ids)
        negative_pool = datamodule.training_negative_item_ids
        log.info(
            "Cold-start pool audit: "
            f"negative_sampler_pool_size={len(negative_pool)}, "
            f"cold_negative_pool_intersection={len(cold_set.intersection(negative_pool))}, "
            f"candidate_pool_size={len(datamodule.all_item_ids)}, "
            f"cold_candidate_count={len(cold_set.intersection(datamodule.all_item_ids))}, "
            f"audit={audit_path}"
        )

    log.info("Instantiating callbacks...")
    callbacks: list[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )
    git_commit = _current_git_commit(cfg.paths.work_dir)
    if git_commit:
        log.info(f"Git commit: {git_commit}")
        if trainer.is_global_zero:
            commit_path = os.path.join(cfg.paths.output_dir, "git_commit.txt")
            os.makedirs(cfg.paths.output_dir, exist_ok=True)
            with open(commit_path, "w") as f:
                f.write(git_commit + "\n")

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        try:
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        except Exception as exc:
            err_text = str(exc)
            if (
                ckpt_path is not None
                and (
                    isinstance(exc, pickle.UnpicklingError)
                    or "Weights only load failed" in err_text
                )
            ):
                log.warning(
                    "Failed to load best checkpoint for test with torch weights_only mode. "
                    "Retrying test with current in-memory weights."
                )
                trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
            else:
                raise
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3", config_path="../../../configs", config_name="train.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
