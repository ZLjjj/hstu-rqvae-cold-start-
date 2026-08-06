# ML-1M Strict Cold-start Ablation Results

| Experiment | AUC | HR@10 | HR@50 | HR@100 | NDCG@10 | NDCG@50 | NDCG@100 | MRR | cold_AUC | cold_HR@10 | cold_HR@50 | cold_HR@100 | cold_NDCG@10 | cold_NDCG@50 | cold_NDCG@100 | cold_MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hstu_baseline_cold | 0.668706 | 0.031250 | 0.062500 | 0.125000 | 0.012089 | 0.017961 | 0.028125 | 0.012457 | 0.543984 | 0.000000 | 0.166667 | 0.166667 | 0.000000 | 0.031317 | 0.031317 | 0.008419 |
| rqvae_hstu_route_a | 0.510680 | 0.031250 | 0.093750 | 0.156250 | 0.031250 | 0.045253 | 0.054994 | 0.039368 | 0.423946 | 0.000000 | 0.166667 | 0.166667 | 0.000000 | 0.043775 | 0.043775 | 0.016966 |
| rqvae_hstu_route_b | 0.491786 | 0.000000 | 0.031250 | 0.031250 | 0.000000 | 0.005566 | 0.005566 | 0.005471 | 0.502783 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.004975 |

## Run Artifacts

- `hstu_baseline_cold`
  - run_dir: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-13_17-48-43`
  - csv: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-13_17-48-43/csv/version_0/metrics.csv`
  - config: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-13_17-48-43/.hydra/config.yaml`
  - best_checkpoint: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-13_17-48-43/checkpoints/epoch_000.ckpt`
  - git_commit: `13954e70a8319d11aef23cfd2784db9f297738cf`
- `rqvae_hstu_route_a`
  - run_dir: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-13_17-49-07`
  - csv: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-13_17-49-07/csv/version_0/metrics.csv`
  - config: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-13_17-49-07/.hydra/config.yaml`
  - best_checkpoint: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-13_17-49-07/checkpoints/epoch_000.ckpt`
  - git_commit: `13954e70a8319d11aef23cfd2784db9f297738cf`
- `rqvae_hstu_route_b`
  - run_dir: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-13_17-49-31`
  - csv: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-13_17-49-31/csv/version_0/metrics.csv`
  - config: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-13_17-49-31/.hydra/config.yaml`
  - best_checkpoint: `/root/autodl-tmp/HSTU_RQVAE/Mini-HSTU-Reproduction/logs/train/runs/2026-03-13_17-49-31/checkpoints/epoch_000.ckpt`
  - git_commit: `13954e70a8319d11aef23cfd2784db9f297738cf`
