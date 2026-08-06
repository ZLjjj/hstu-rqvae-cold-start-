# ML-1M Strict Cold-start Ablation Results

| Experiment | AUC | HR@10 | HR@50 | HR@100 | NDCG@10 | NDCG@50 | NDCG@100 | MRR | cold_AUC | cold_HR@10 | cold_HR@50 | cold_HR@100 | cold_NDCG@10 | cold_NDCG@50 | cold_NDCG@100 | cold_MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hstu_baseline_cold | 0.798121 | 0.238742 | 0.463245 | 0.551159 | 0.131398 | 0.181393 | 0.195707 | 0.113307 | 0.088785 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.004975 |
| rqvae_hstu_route_a | 0.848269 | 0.146026 | 0.329967 | 0.429801 | 0.079866 | 0.119940 | 0.136134 | 0.072635 | 0.614860 | 0.004298 | 0.045845 | 0.095989 | 0.002362 | 0.010693 | 0.018748 | 0.008540 |
| rqvae_hstu_route_b | 0.728247 | 0.049007 | 0.049007 | 0.049007 | 0.032705 | 0.032705 | 0.032705 | 0.032172 | 0.221662 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.004975 |

## Run Artifacts

- `hstu_baseline_cold`
  - run_dir: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-15_11-59-45`
  - csv: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-15_11-59-45/csv/version_0/metrics.csv`
  - config: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-15_11-59-45/.hydra/config.yaml`
  - best_checkpoint: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-15_11-59-45/checkpoints/epoch_070.ckpt`
  - git_commit: `2230a233ac89fcba204dd0d207fb11230a1be3f2`
- `rqvae_hstu_route_a`
  - run_dir: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-15_12-12-07`
  - csv: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-15_12-12-07/csv/version_0/metrics.csv`
  - config: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-15_12-12-07/.hydra/config.yaml`
  - best_checkpoint: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-15_12-12-07/checkpoints/epoch_121.ckpt`
  - git_commit: `2230a233ac89fcba204dd0d207fb11230a1be3f2`
- `rqvae_hstu_route_b`
  - run_dir: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-15_13-09-33`
  - csv: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-15_13-09-33/csv/version_0/metrics.csv`
  - config: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-15_13-09-33/.hydra/config.yaml`
  - best_checkpoint: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-15_13-09-33/checkpoints/epoch_057.ckpt`
  - git_commit: `2230a233ac89fcba204dd0d207fb11230a1be3f2`
