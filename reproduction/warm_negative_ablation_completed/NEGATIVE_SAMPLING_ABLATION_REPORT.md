# 严格冷启动负样本采样消融报告

## 协议

唯一变量是训练负样本池改为 warm items；评估候选池仍含全部 cold items。

## Old vs warm-negative-only

| 实验 | 指标 | old | warm-negative-only | absolute_delta | relative_delta |
|---|---|---:|---:|---:|---:|
| hstu_baseline_cold_warm_neg | overall_auc | 0.824046 | 0.898017 | 0.073971 | 0.089766 |
| hstu_baseline_cold_warm_neg | warm_auc |  | 0.943485 |  |  |
| hstu_baseline_cold_warm_neg | cold_auc | 0.088247 | 0.439583 | 0.351336 | 3.981274 |
| hstu_baseline_cold_warm_neg | cold_hr@10 | 0.000000 | 0.000000 | 0.000000 |  |
| hstu_baseline_cold_warm_neg | cold_hr@100 | 0.000000 | 0.014679 | 0.014679 |  |
| hstu_baseline_cold_warm_neg | cold_ndcg@10 | 0.000000 | 0.000000 | 0.000000 |  |
| hstu_baseline_cold_warm_neg | cold_ndcg@100 | 0.000000 | 0.002371 | 0.002371 |  |
| hstu_baseline_cold_warm_neg | cold_mrr_corrected | 0.000000 | 0.000309 | 0.000309 |  |
| hstu_dense_cold_warm_neg | overall_auc | 0.847237 | 0.852512 | 0.005276 | 0.006227 |
| hstu_dense_cold_warm_neg | warm_auc |  | 0.878081 |  |  |
| hstu_dense_cold_warm_neg | cold_auc | 0.475650 | 0.594716 | 0.119065 | 0.250322 |
| hstu_dense_cold_warm_neg | cold_hr@10 | 0.000000 | 0.011009 | 0.011009 |  |
| hstu_dense_cold_warm_neg | cold_hr@100 | 0.020183 | 0.045872 | 0.025688 | 1.272727 |
| hstu_dense_cold_warm_neg | cold_ndcg@10 | 0.000000 | 0.006355 | 0.006355 |  |
| hstu_dense_cold_warm_neg | cold_ndcg@100 | 0.003605 | 0.012700 | 0.009096 | 2.523295 |
| hstu_dense_cold_warm_neg | cold_mrr_corrected | 0.000596 | 0.006010 | 0.005415 | 9.091087 |
| rqvae_hstu_route_a_warm_neg | overall_auc | 0.857546 | 0.861444 | 0.003898 | 0.004546 |
| rqvae_hstu_route_a_warm_neg | warm_auc |  | 0.880001 |  |  |
| rqvae_hstu_route_a_warm_neg | cold_auc | 0.592034 | 0.674347 | 0.082313 | 0.139034 |
| rqvae_hstu_route_a_warm_neg | cold_hr@10 | 0.007339 | 0.036697 | 0.029358 | 4.000000 |
| rqvae_hstu_route_a_warm_neg | cold_hr@100 | 0.102752 | 0.185321 | 0.082569 | 0.803571 |
| rqvae_hstu_route_a_warm_neg | cold_ndcg@10 | 0.003708 | 0.019973 | 0.016265 | 4.386679 |
| rqvae_hstu_route_a_warm_neg | cold_ndcg@100 | 0.021144 | 0.047944 | 0.026801 | 1.267557 |
| rqvae_hstu_route_a_warm_neg | cold_mrr_corrected | 0.005551 | 0.019836 | 0.014285 | 2.573512 |

## 分析问题

1. Baseline cold AUC 是否恢复到接近随机水平（0.5）？
2. Dense 与 Route A 的 cold AUC、HR、NDCG、corrected MRR 是否同步提高？
3. Overall 与 warm 指标是否下降？
4. Route A 相对 Dense 的冷启动优势是否仍存在？
5. 旧协议是否因 cold negative exposure 低估冷启动能力？
6. Cold AUC 改善是否转化为 cold HR@10/100 改善？

结论必须同时依据 AUC、HR、NDCG 和 corrected MRR。

## Route B 排除说明

Route B 与 Full-Token Route B 使用语义码分类交叉熵，不使用 LocalNegativesSampler 的物品随机负采样。共享 code 使排除 cold code 会改变分类空间，属于另一项实验。
