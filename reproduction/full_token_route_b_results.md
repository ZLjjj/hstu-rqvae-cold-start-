# Full-Token Route B 结果表

状态：本地实现与 smoke 阶段。Full-Token 主实验尚未在 GPU 服务器运行；`—` 表示待服务器回填，不代表 0。

## 核心对照

| 实验 | Overall NDCG@100 | Overall corrected MRR | Cold HR@100 | Cold NDCG@100 | Cold corrected MRR |
|---|---:|---:|---:|---:|---:|
| HSTU baseline | 0.212941 | 0.124940 | 0.000000 | 0.000000 | 0.000000 |
| Dense baseline | 0.157630 | 0.087339 | 0.020183 | 0.003605 | 0.000596 |
| Route A | 0.140380 | 0.070762 | 0.102752 | 0.021144 | 0.005551 |
| 旧简化 Route B | 0.080531 | 0.041626 | 0.001835 | 0.000530 | 0.000183 |
| 新 Full-Token Route B | — | — | — | — | — |

前四组数值直接沿用 `reproduction/FINAL_REPORT.md`，不重训。服务器完成后由 `scripts/summarize_full_token_route_b.py` 从 CSV 更新本文件。

## Full-Token 完整指标

| 分组 | AUC | HR@10 | HR@50 | HR@100 | HR@200 | NDCG@10 | NDCG@50 | NDCG@100 | corrected MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | — | — | — | — | — | — | — | — | — |
| Cold | — | — | — | — | — | — | — | — | — |
| Warm | — | — | — | — | — | — | — | — | — |

## Token、搜索与资源诊断

| 指标 | 结果 |
|---|---:|
| level-0 loss / accuracy | — / — |
| level-1 loss / accuracy | — / — |
| level-2 loss / accuracy | — / — |
| Beam 合法完整路径/用户 | — |
| Beam 唯一物品/用户 | — |
| Exact 补齐数/用户 | — |
| Semantic ID 冲突路径 / 受影响物品 | — / — |
| Beam-only 候选召回 | — |
| 实际平均 / 最大 beam | — / — |
| 平均单用户 Beam 推理时间 | — |
| 平均单用户 Exact 时间 | — |
| 训练总时间 / 最佳 epoch | — / — |
| 峰值显存 / 可训练参数量 | — / — |

## 三种排名口径

| 口径 | Overall HR@100 | Cold HR@100 | Overall NDCG@100 | Cold NDCG@100 | AUC |
|---|---:|---:|---:|---:|---:|
| Beam-only | — | — | — | — | 不适用 |
| Beam + Exact 补齐 | — | — | — | — | 不适用 |
| Exact 全局排序 | — | — | — | — | — |

