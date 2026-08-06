# HSTU + RQ-VAE 冷启动集成

本仓库把 `RQ-VAE-Recommender` 和 `Mini-HSTU-Reproduction` 打通，用 RQ-VAE 语义码增强 HSTU 在严格 item 冷启动下的检索能力。

## 1. 项目内容

项目包含 RQ-VAE 语义表示、HSTU 序列推荐、严格冷启动数据划分、
warm-only 训练负样本消融、完整测试指标以及可复现的服务器运行脚本。

## 2. 五组实验定义（统一协议）

1. `ml-1m-hstu-cold`：HSTU baseline（严格冷启动）
2. `ml-1m-hstu-dense-cold`：Dense baseline（MiniLM 稠密向量 + 线性投影）
3. `ml-1m-hstu-semantic-cold`：RQ-VAE + HSTU Route A（语义码 embedding）
4. `ml-1m-hstu-token-cold`：RQ-VAE + HSTU Route B（token AR + beam + exact 补全）
5. `ml-1m-hstu-full-token-cold`：RQ-VAE + HSTU Full-Token Route B（历史与目标均展开为 Semantic Token 序列）

说明：
- Route B 的 HR/NDCG/MRR 来自 beam 推理，若唯一候选数不足 `k`，自动用 exact scorer 补齐。
- Route B 的 AUC/cold_AUC 来自全物品 exact token scorer：
  - `log p(c1|ctx) + log p(c2|ctx,c1) + log p(c3|ctx,c1,c2)`
- Full-Token Route B 在 HSTU 内执行标准 next-token 因果建模；生成 `c1/c2` 后会把预测 Token 追加回序列，再预测下一层 Token。
- 分桶定义：
  - `zero_*`: target 在模型可见训练集中交互次数 `= 0`（Strict-Cold）
  - `few_*`: target 交互次数 `in [1, 5]`（Long-tail）
  - `hr@k/ndcg@k`: 全体样本（All）

## 3. 使用方法

以下命令默认在项目目录下执行。

### 3.1 环境和数据（以Movielens-1M为例）

```bash
cd Mini-HSTU-Reproduction
make env_smoke
make prepare_data data=ml-1m
```

### 3.2 RQ-VAE 训练并导出 bridge

```bash
cd ../RQ-VAE-Recommender
make env_smoke
make train_rqvae config=configs/rqvae_ml1m.gin
make export_dense
```

`make export_dense` 会优先使用 embeddings 自带的 `item_ids`，其次使用 `movies.dat`，若行数不匹配则自动回退到 `bridge_artifacts.pt` 的 `item_ids` 做对齐。

或直接从 checkpoint 导出：

```bash
make export_bridge \
  checkpoint=out/rqvae/ml1m/checkpoint_9999.pt \
  export_path=out/rqvae/ml1m/bridge_artifacts.pt
```

可选：执行 RQ-VAE 质量审计（建议保留审计 JSON）：

```bash
make audit_rqvae \
  metrics_csv_path=out/rqvae/ml1m/metrics.csv \
  bridge_path=out/rqvae/ml1m/bridge_artifacts.pt \
  output_json=out/rqvae/ml1m/audit_summary.json
```

### 3.3 消融实验

先说明两点：

1. `Mini-HSTU-Reproduction/README.md` 里的 `HSTU / HSTU w/ Aux` 指标对应的是**非冷启动**实验。
2. 本项目五组 `*-cold` 实验是**严格 item 冷启动**协议，不能直接与非冷启动指标横向比较。

#### 3.3.1 复现非冷启动 baseline

```bash
cd ../Mini-HSTU-Reproduction
make train experiment=ml-1m-hstu logger=csv
make train experiment=ml-1m-hstu-aux logger=csv
```

#### 3.3.2 严格冷启动五组完整实验

```bash
cd ../Mini-HSTU-Reproduction
make train experiment=ml-1m-hstu-cold logger=csv
make train experiment=ml-1m-hstu-dense-cold logger=csv
make train experiment=ml-1m-hstu-semantic-cold logger=csv
make train experiment=ml-1m-hstu-token-cold logger=csv
make train experiment=ml-1m-hstu-full-token-cold logger=csv
```

建议（可选）：
- 如果有 GPU，增加 `trainer=gpu` 以缩短训练时间。
- 不要额外覆盖 `trainer.max_epochs`、`trainer.limit_train_batches`、`trainer.limit_val_batches`、`trainer.limit_test_batches`。


### 3.4 汇总严格冷启动实验结果

```bash
make summarize_ablation \
  baseline_run=logs/train/runs/<baseline_run_dir> \
  dense_run=logs/train/runs/<dense_run_dir> \
  route_a_run=logs/train/runs/<route_a_run_dir> \
  route_b_run=logs/train/runs/<route_b_run_dir> \
  output_markdown=docs/full_ablation_results_ml1m.md \
  output_csv=docs/full_ablation_results_ml1m.csv
```


## 4. 进一步阅读

- 详细 runbook：`Mini-HSTU-Reproduction/docs/cold_start_runbook.md`
- 自动汇总示例输出：
  - `Mini-HSTU-Reproduction/docs/ablation_results_ml1m.md`
  - `Mini-HSTU-Reproduction/docs/ablation_results_ml1m.csv`
## 5. 结果（MovieLens-1M）

以下为本项目在同一严格 item 冷启动划分、同一 seed（42）下完成的五组最终全量测试结果。由于 ML-1M 数据量较少，这里不展示 few 分桶。

| Experiment | AUC | HR@10 | HR@50 | HR@100 | NDCG@10 | NDCG@50 | NDCG@100 | MRR | cold_AUC | cold_HR@10 | cold_HR@50 | cold_HR@100 | cold_NDCG@10 | cold_NDCG@50 | cold_NDCG@100 | cold_MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hstu_baseline_cold | 0.824046 | 0.261258 | 0.495861 | 0.583113 | 0.146860 | 0.198756 | 0.212941 | 0.126631 | 0.088247 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.004975 |
| hstu_dense_cold | 0.847237 | 0.182285 | 0.375331 | 0.461258 | 0.101129 | 0.143693 | 0.157630 | 0.089605 | 0.475650 | 0.000000 | 0.009174 | 0.020183 | 0.000000 | 0.001840 | 0.003605 | 0.005379 |
| rqvae_hstu_route_a | 0.857546 | 0.155132 | 0.341225 | 0.450497 | 0.082073 | 0.122630 | 0.140380 | 0.072937 | 0.592034 | 0.007339 | 0.055046 | 0.102752 | 0.003708 | 0.013407 | 0.021144 | 0.009640 |
| rqvae_hstu_route_b | 0.748485 | 0.075166 | 0.183444 | 0.263576 | 0.044494 | 0.067558 | 0.080531 | 0.044753 | 0.228927 | 0.001835 | 0.001835 | 0.001835 | 0.000530 | 0.000530 | 0.000530 | 0.005149 |
| rqvae_hstu_full_token_route_b | 0.788509 | 0.115066 | 0.275497 | 0.378808 | 0.061558 | 0.096332 | 0.113028 | 0.057330 | 0.077678 | 0.000000 | 0.001835 | 0.001835 | 0.000000 | 0.000395 | 0.000395 | 0.005042 |

说明：

- 表中 `MRR/cold_MRR` 沿用项目 `RetrievalMetrics` 的原始口径；Top-K 未命中样本使用 `k+1` 作为截断排名，因此 `cold_MRR` 非零不等于 Top-K 命中。严格命中分析应同时查看 HR/NDCG 以及 `mrr_corrected/cold_mrr_corrected`。
- Full-Token 实验使用 FBGEMM GPU 1.0.0、BF16、batch size 128 和 8 个 DataLoader workers；训练50轮，每轮固定5个验证 batch 选择 checkpoint，表中数据来自最佳 checkpoint 的最终全量测试。
- Full-Token 相比旧 Route B，Overall HR@100 提升43.72%，Overall NDCG@100 提升40.35%；但 cold_HR@100 基本不变，cold_NDCG@100 下降25.45%，说明整体增益主要来自 warm 物品。
- HSTU baseline 的整体排序效果最好；Route A 的严格冷启动效果最好。Full-Token 改善了 Route B 的整体效果，但没有解决严格冷启动召回问题。
- Full-Token 的完整配置、日志、GPU采样、checkpoint 和分析见 `reproduction/full_token_fast_completed/`。

### 5.1 排除冷物品负样本后的结果（推荐严格协议）

上表前三组 sampled-softmax 实验的训练负样本来自 `all_item_ids`，因此预留的冷物品仍可能被当作随机负样本。为消除这种负向训练暴露，在保持冷物品划分、模型结构、超参数、训练种子和全物品评估候选集不变的前提下，只将训练负样本池改为 warm items，并重新训练以下三组实验。Route B 和 Full-Token Route B 不使用物品级 `LocalNegativesSampler`，因此不受该修改影响，无需重训。

下表采用修正后的 MRR：目标未进入 Top-200 时贡献为 0。

| Experiment | AUC | HR@10 | HR@50 | HR@100 | HR@200 | NDCG@10 | NDCG@50 | NDCG@100 | NDCG@200 | corrected MRR | cold_AUC | cold_HR@10 | cold_HR@50 | cold_HR@100 | cold_HR@200 | cold_NDCG@10 | cold_NDCG@50 | cold_NDCG@100 | cold_NDCG@200 | cold corrected MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hstu_baseline_cold_warm_neg | 0.898017 | 0.257119 | 0.517219 | 0.609934 | 0.698013 | 0.141072 | 0.198566 | 0.213642 | 0.225978 | 0.120029 | 0.439583 | 0.000000 | 0.003670 | 0.014679 | 0.029358 | 0.000000 | 0.000656 | 0.002371 | 0.004404 | 0.000309 |
| hstu_dense_cold_warm_neg | 0.852512 | 0.182450 | 0.372517 | 0.452318 | 0.541225 | 0.099084 | 0.140951 | 0.153886 | 0.166317 | 0.084597 | 0.594716 | 0.011009 | 0.025688 | 0.045872 | 0.069725 | 0.006355 | 0.009445 | 0.012700 | 0.016015 | 0.006010 |
| rqvae_hstu_route_a_warm_neg | 0.861444 | 0.156126 | 0.345364 | 0.455132 | 0.559768 | 0.083520 | 0.124725 | 0.142516 | 0.157148 | 0.072370 | **0.674347** | **0.036697** | **0.119266** | **0.185321** | **0.258716** | **0.019973** | **0.037461** | **0.047944** | **0.058237** | **0.019836** |

关键冷启动指标的新旧对比如下：

| Model | cold_AUC | cold_HR@10 | cold_HR@100 | cold_NDCG@100 | cold corrected MRR |
|---|---:|---:|---:|---:|---:|
| HSTU baseline：all-item negatives → warm-only negatives | 0.088247 → 0.439583 | 0.000000 → 0.000000 | 0.000000 → 0.014679 | 0.000000 → 0.002371 | 0.000000 → 0.000309 |
| Dense：all-item negatives → warm-only negatives | 0.475650 → 0.594716 | 0.000000 → 0.011009 | 0.020183 → 0.045872 | 0.003605 → 0.012700 | 0.000596 → 0.006010 |
| Route A：all-item negatives → warm-only negatives | 0.592034 → **0.674347** | 0.007339 → **0.036697** | 0.102752 → **0.185321** | 0.021144 → **0.047944** | 0.005551 → **0.019836** |

协议审计结果：371 个冷物品在三组实验的训练历史、训练目标和训练负样本池中出现次数均为 0；包含 3,512 个暖物品的训练负样本池不含任何冷物品；全物品评估候选集仍包含全部 371 个冷物品。三组实验使用相同的 cold split hash 和 `seed=42`。

在 545 条冷目标测试样本中，Route A 的 Top-10 命中由 4 条提高到 20 条，Top-100 命中由 56 条提高到 101 条；Dense 的 Top-100 命中为 25 条，HSTU ID baseline 为 8 条。排除伪负例后，Route A 在 cold AUC、HR、NDCG 和 corrected MRR 上仍全面优于 Dense，说明其冷启动优势不是由单一 AUC 指标造成的。

完整指标、运行配置和审计文件见：

- `reproduction/warm_negative_ablation_completed/warm_negative_ablation_results.md`
- `reproduction/warm_negative_ablation_completed/NEGATIVE_SAMPLING_ABLATION_REPORT.md`
- `reproduction/warm_negative_ablation_completed/`
