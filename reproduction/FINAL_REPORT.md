# Semantic-ID-Rec 严格冷启动复现实验报告

## 1. 实验结论

本次复现完成了 MovieLens-1M 上的四组严格物品冷启动对照实验，并保留了完整命令、环境、检查点、逐轮指标、GPU 采样和异常记录。

核心结论：

1. **普通 HSTU 的整体排序最好，但无法推荐训练期完全未出现的物品。**
   其测试 NDCG@100 为 0.212941，冷物品 HR@100 和修正冷 MRR 均为 0。
2. **RQ-VAE + HSTU Route A 是冷启动效果最好的方案。**
   冷 HR@100 为 0.102752，冷 NDCG@100 为 0.021144，修正冷 MRR 为 0.005551。
3. **Route A 显著优于直接使用 MiniLM 稠密特征。**
   相对 Dense，冷 HR@100 提升约 5.09 倍，冷 NDCG@100 提升约 5.87 倍，修正冷 MRR 提升约 9.32 倍。
4. **Route B 的代价很高，但本次严格设置下效果弱于 Route A。**
   Route B 耗时约 4 小时 15 分钟，冷 HR@100 仅 0.001835；不建议把 beam token 解码作为本数据规模下的默认方案。
5. **仓库原 MRR 实现存在未命中样本仍获得 `1/(k+1)` 的问题。**
   本次保留旧 MRR 用于横向核对，同时新增修正 MRR；本文结论优先采用修正版。

## 2. 实验设置

- 数据集：MovieLens-1M
- 随机种子：42
- 冷物品比例：10%
- 冷物品数量：371
- 严格规则：冷物品从所有训练序列中移除，训练期计数必须为 0
- 最少/最多训练轮数：10 / 500
- 早停：监控 `val/ndcg@100`，patience=20
- 硬件：NVIDIA GeForce RTX 4090 D 24 GB、18 个可用 CPU 核、60 GB 内存
- Python：3.10
- PyTorch：2.5.1+cu124
- Lightning：2.5.0.post0
- 主集成仓库原始提交：`5a0a3a2b44bbf7eef6ce2daa2841ad819099d75b`
- 复现补丁提交：`a1c17e45f9ebb232a2f000f1ad436930a9d00d30`

四组实验：

- HSTU baseline：普通可训练物品 ID embedding
- Dense：冻结 MiniLM 384 维内容向量并线性投影
- Route A：冻结 RQ-VAE 三层语义码本，组合为物品表示
- Route B：语义 token 自回归预测、beam=8、前缀约束和 exact 补全

## 3. 最终测试结果

| 方法 | AUC | HR@100 | NDCG@100 | 修正 MRR | 冷 AUC | 冷 HR@100 | 冷 NDCG@100 | 修正冷 MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HSTU baseline | 0.824046 | **0.583113** | **0.212941** | **0.124940** | 0.088247 | 0.000000 | 0.000000 | 0.000000 |
| Dense MiniLM | 0.847237 | 0.461258 | 0.157630 | 0.087339 | 0.475650 | 0.020183 | 0.003605 | 0.000596 |
| RQ-VAE + HSTU Route A | **0.857546** | 0.450497 | 0.140380 | 0.070762 | **0.592034** | **0.102752** | **0.021144** | **0.005551** |
| RQ-VAE + HSTU Route B | 0.748485 | 0.263576 | 0.080531 | 0.041626 | 0.228927 | 0.001835 | 0.000530 | 0.000183 |

解释：

- 普通 ID embedding 擅长暖物品，但严格冷物品从未获得梯度，所以冷指标为 0。
- Dense 和 Route A 都能从内容侧为冷物品构造表示。
- Route A 的离散语义组合比直接冻结稠密文本向量更适合本实验的候选检索。
- Route B 的层级解码误差会逐层累积，且 beam 候选受语义码冲突和覆盖率影响，导致召回明显偏低。

## 4. 训练轮数与耗时

| 阶段 | 实际轮数/步数 | 最佳检查点 | 墙钟时间 |
|---|---:|---:|---:|
| RQ-VAE | 50,000 steps | step 49,999 | 6 分 50 秒 |
| HSTU baseline | 109 epochs | epoch 87 | 34 分 28 秒 |
| Dense | 193 epochs | epoch 171 | 61 分 |
| Route A | 157 epochs | epoch 135 | 48 分 7 秒 |
| Route B | 80 epochs | epoch 58 | 4 小时 14 分 41 秒 |

四组 HSTU 合计约 6 小时 38 分钟。Route B 单轮约 3.2 分钟，是总耗时的主要来源。

## 5. RQ-VAE 质量审计

质量审计状态：**PASS**。

| 指标 | 结果 |
|---|---:|
| 第 1/2/3 层码本使用率 | 98.73% / 98.63% / 98.93% |
| 第 1/2/3 层死码率 | 0% / 0% / 0% |
| 第 1/2/3 层 perplexity | 767.76 / 809.53 / 819.54 |
| 最大 ID 重复比例 | 0.117% |
| 完整语义码冲突率 | 1.171% |
| 桥接物品数 | 3,416 |
| 全局 ID 空间覆盖率 | 86.44% |
| 冷物品覆盖率 | 91.11%（338/371） |
| 每维重建损失 | 0.02847 |

冷物品并非全部有语义码：371 个冷物品中约 33 个缺少 RQ-VAE 桥接表示。这主要来自 RQ-VAE 数据预处理保留“至少 5 条评分”的物品过滤规则，是冷启动上限的重要限制。

MiniLM 仅保留一个 `model.safetensors`，SHA-256：

`53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`

## 6. 资源消耗

10 秒间隔 GPU 采样共 2,229 条：

- GPU 显存峰值：19,698 MiB
- GPU 显存中位数：978 MiB
- GPU 利用率峰值：60%
- GPU 利用率中位数：17%
- 功耗峰值：132.27 W
- Route B 典型显存：约 1 GB
- Dense 全候选评估峰值接近 20 GB

建议：

- **完整默认复现：24 GB 显存 GPU、至少 32 GB 内存；推荐 60 GB 内存。**
- 16 GB 显存不适合无修改运行 Dense 全候选评估。
- RQ-VAE 本身显存需求很低，约 2 GB 即可；主要资源瓶颈在 HSTU 全候选评估和 Route B 解码。

## 7. 发现的问题与修复

1. 集成仓库的根 `.gitignore` 使用 `**/data/`，导致两个上游项目的源码数据模块没有进入仓库。
   本次从对应上游提交恢复源码，并添加精确白名单。
2. 原始冷启动实现没有保证冷物品从所有训练上下文中移除。
   本次实现确定性冷物品划分、严格训练过滤、训练计数审计并持久化划分。
3. 原 MRR 对未命中样本返回 `1/(k+1)`。
   本次同时输出旧 MRR 和修正 MRR，避免旧表无法对照。
4. RQ-VAE 与 HSTU 的物品 ID 体系可能错位。
   本次保留外部 MovieLens ID，并导出显式 `item_id_to_codes` 桥接张量。
5. 容器 `os.cpu_count()` 返回 512，但实际配额只有 18 核，仓库默认因此创建 128 个 DataLoader worker。
   这不是进程泄漏，但会增加内存与调度开销；后续建议显式设置 `data.num_workers=4~8`。
6. 未安装可选 FBGEMM GPU 算子，运行时使用 PyTorch fallback。
   结果正确，但速度低于 README 中“100 epochs 小于 10 分钟”的理想环境。

## 8. 与仓库报告的关系

仓库完整消融报告记录的最佳检查点为 baseline epoch 70、Route A epoch 121、Route B epoch 57；本次分别为 87、135、58，训练进度量级一致。

本次结果不能要求逐位复现仓库表格，原因包括：

- 本次修复了严格冷启动的数据泄漏问题；
- 本次增加了修正 MRR；
- RQ-VAE 桥接覆盖和物品过滤规则被显式审计；
- 运行环境缺少 FBGEMM；
- 仅运行一个随机种子，没有报告置信区间。

尽管如此，关键排序结论稳定：普通 HSTU 整体最好，Route A 冷启动最好，Dense 次之，Route B 在效果与成本上均不占优。

## 9. 产物与复现记录

服务器根目录：

`/root/autodl-tmp/semantic-id-rec-repro/repo`

关键产物：

- `reproduction/commands.log`：所有正式命令、开始/结束时间与退出码
- `reproduction/environment.txt`：硬件、CUDA、Python 和依赖环境
- `reproduction/requirements-lock.txt`：完整依赖锁定
- `reproduction/gpu_usage.csv`：10 秒粒度 GPU 资源采样
- `reproduction/final_ablation_results.csv`
- `reproduction/final_ablation_results.md`
- `RQ-VAE-Recommender/out/rqvae/ml1m/audit_summary.json`
- `RQ-VAE-Recommender/out/rqvae/ml1m/bridge_artifacts.pt`
- `RQ-VAE-Recommender/out/rqvae/ml1m/dense_features.pt`
- 四组 HSTU run 目录中的配置、逐轮指标和最佳检查点

## 10. 最终判断

如果目标是验证 Semantic ID 是否能帮助严格物品冷启动，本次实验给出的答案是 **能**，但实现路线差异很大：

- Route A 带来最明确的冷启动收益，是后续研究和工程优化的首选。
- Dense 是更简单的内容基线，效果弱于 Route A，但实现和推理成本更低。
- Route B 在 ML-1M 上性价比很低，除非后续重做解码目标、候选覆盖和高效 beam kernel，否则不建议继续投入。
- 下一步最有价值的实验是多随机种子复现、提高 RQ-VAE 冷物品覆盖率，以及 Route A 的可训练码本消融。
