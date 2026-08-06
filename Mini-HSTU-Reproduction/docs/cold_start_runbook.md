# HSTU + RQ-VAE Cold-start Runbook (ML-1M, strict cold-start)

## 1) Protocol

- Same split for all groups:
  - `cold_item_ratio=0.1`
  - `cold_split_seed=42`
  - chronological split
- Four experiment groups:
  1. `ml-1m-hstu-cold` (HSTU baseline)
  2. `ml-1m-hstu-dense-cold` (Dense baseline: MiniLM vector + trainable projection)
  3. `ml-1m-hstu-semantic-cold` (Route A)
  4. `ml-1m-hstu-token-cold` (Route B: beam for HR/NDCG/MRR + exact scorer for AUC + beam补全)
- AUC definition: full-candidate AUC.
- Bucket metrics:
  - `zero_hr/ndcg@k`: strict cold (`target_train_count == 0`)
  - `few_hr/ndcg@k`: long-tail (`target_train_count in [1,5]`)
  - `hr/ndcg@k`: all samples.

## 2) Environment and data

```bash
cd /root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction
make env_smoke
make prepare_data data=ml-1m
```

## 3) RQ-VAE bridge artifacts

Train and export during eval:

```bash
cd /root/autodl-tmp/HSTU_RQVAE/RQ-VAE-Recommender
make env_smoke
make train_rqvae config=configs/rqvae_ml1m.gin
```

Export directly from `checkpoint_9999.pt`:

```bash
cd /root/autodl-tmp/HSTU_RQVAE/RQ-VAE-Recommender
make export_bridge \
  checkpoint=out/rqvae/ml1m/checkpoint_9999.pt \
  export_path=out/rqvae/ml1m/bridge_artifacts.pt
```

Export dense features and run audit:

```bash
cd /root/autodl-tmp/HSTU_RQVAE/RQ-VAE-Recommender
make export_dense
make audit_rqvae \
  metrics_csv_path=out/rqvae/ml1m/metrics.csv \
  bridge_path=out/rqvae/ml1m/bridge_artifacts.pt \
  output_json=out/rqvae/ml1m/audit_summary.json
```

Expected bridge:
- `../RQ-VAE-Recommender/out/rqvae/ml1m/bridge_artifacts.pt`
- metadata includes `n_codebooks=3`, `codebook_size=1024`.

## 4) Train and evaluate four groups

```bash
cd /root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction

make train experiment=ml-1m-hstu-cold
make train experiment=ml-1m-hstu-dense-cold
make train experiment=ml-1m-hstu-semantic-cold
make train experiment=ml-1m-hstu-token-cold
```

For quick smoke runs:

```bash
conda run -n NewsRecommend env PYTHONPATH=src \
  python src/generative_recommenders_pl/scripts/train.py \
  experiment=ml-1m-hstu-cold logger=csv \
  trainer.accelerator=cpu trainer.devices=1 \
  trainer.min_epochs=1 trainer.max_epochs=1 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=2 +trainer.limit_test_batches=2
```

## 5) Summarize ablation results

Pass the 4 Hydra run directories:

```bash
cd /root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction

make summarize_ablation \
  baseline_run=logs/train/runs/<baseline_run_dir> \
  dense_run=logs/train/runs/<dense_run_dir> \
  route_a_run=logs/train/runs/<route_a_run_dir> \
  route_b_run=logs/train/runs/<route_b_run_dir> \
  output_markdown=docs/ablation_results_ml1m.md \
  output_csv=docs/ablation_results_ml1m.csv
```

Output table fields:
- `AUC`, `HR@10/50/100`, `NDCG@10/50/100`, `MRR`
- `cold_AUC`, `cold_HR@10/50/100`, `cold_NDCG@10/50/100`, `cold_MRR`
- `zero_HR/NDCG@10/50/100`, `few_HR/NDCG@10/50/100`
- run artifacts: `best_checkpoint`, Hydra config path, git commit.
