# Full-Token Route B 加速版最终实验摘要

## 运行状态

- 状态：正常完成，`exit_code=0`
- Git commit：`d2c9173f6bc24daa70a2db23bdaa3d96e4817ed3`
- 运行时间：2026-08-03 03:33:15 至 09:02:40（约 5 小时 29 分）
- 最佳 checkpoint：`epoch_049.ckpt`
- 最佳验证：NDCG@100 = 0.124127，HR@100 = 0.390625
- 环境：PyTorch 2.5.1+cu124，FBGEMM GPU 1.0.0，RTX 4090 D
- 训练协议：batch size 128，BF16 mixed precision，8 workers，50 epochs；每轮固定 5 个验证 batch 选 checkpoint，最终测试为全量测试。

## 最终全量测试

| 指标 | Overall | Cold | Warm |
|---|---:|---:|---:|
| AUC | 0.788509 | 0.077678 | 0.859010 |
| HR@10 | 0.115066 | 0.000000 | 0.126479 |
| HR@50 | 0.275497 | 0.001835 | 0.302639 |
| HR@100 | 0.378808 | 0.001835 | 0.416197 |
| HR@200 | 0.497517 | 0.001835 | 0.546679 |
| NDCG@10 | 0.061558 | 0.000000 | 0.067663 |
| NDCG@50 | 0.096332 | 0.000395 | 0.105848 |
| NDCG@100 | 0.113028 | 0.000395 | 0.124199 |
| NDCG@200 | 0.129598 | 0.000395 | 0.142412 |
| corrected MRR | 0.054830 | 0.000076 | 0.060261 |

## Beam 与 Exact

- 实际 Beam size：256
- Beam-only Cold AUC：0.001623
- Beam-only Cold HR@100：0.001835
- Beam-only Cold NDCG@100：0.000395
- Exact-global AUC：0.788509
- Exact-global NDCG@100：0.105586
- Exact-global Cold AUC：0.077678
- Exact-global Cold HR@100：0.001835
- Exact-global Cold NDCG@100：0.000382
- Beam 映射前平均唯一物品：217.616；Exact 平均补齐：4.263。

## 与旧简化 Route B 比较

| 指标 | 旧 Route B | 新 Full-Token | 相对变化 |
|---|---:|---:|---:|
| Overall HR@100 | 0.263576 | 0.378808 | +43.72% |
| Overall NDCG@100 | 0.080531 | 0.113028 | +40.35% |
| Overall corrected MRR | 0.041626 | 0.054830 | +31.72% |
| Cold AUC | 0.228927 | 0.077678 | -66.07% |
| Cold HR@100 | 0.001835 | 0.001835 | 基本不变 |
| Cold NDCG@100 | 0.000530 | 0.000395 | -25.45% |
| Cold corrected MRR | 0.000183 | 0.000076 | -58.22% |

结论：Full-Token 实现显著改善了整体推荐质量，但没有改善严格冷启动召回；Cold HR@100 与旧 Route B 基本相同，Cold AUC、NDCG 和 corrected MRR 反而下降。整体增益主要来自 warm 物品，Route A 仍是当前冷启动效果最好的路线。

## 资源记录

- GPU 采样数：1977
- 平均 GPU 利用率：57.01%
- 最高 GPU 利用率：98%
- 采样峰值显存：4698 MB
- 模型记录的测试峰值显存：3033.72 MB
- 平均功耗：165.72 W
- 最高温度：51°C

原始 `metrics.csv`、Hydra 配置、主日志、GPU 采样、退出文件和最佳 checkpoint 均已保存在本目录中；`bundle.tgz` 是同步时的完整压缩包。
