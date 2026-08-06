# Full-Token Route B 实验报告

状态：代码、测试、smoke 与服务器执行材料准备阶段；正式 GPU 结果待回填。本文明确区分已验证设计与尚未运行的实验结论。

## 摘要

旧 Route B 只在输出端对目标物品的三级码做条件分类，历史仍是“一物品一位置”。新 `HierarchicalFullTokenRetrieval` 把全部历史展开成 Semantic Token 序列，在 HSTU 内进行标准 next-token 因果建模，并在生成 `c1/c2` 后把它们真正追加回 HSTU。该改动隔离在新类和新配置中，不改变旧 Route B 的结构、配置或已有结果。

## 1. 新旧 Route B 的真实结构

旧路线：`历史 item ID → 三级码本向量拼接/投影为单个 50D item → HSTU → 固定用户上下文 + 已生成码向量 → 三级头`。因此历史长度仍约为 200，自回归仅存在于目标物品内部，而且下一级没有重新运行 HSTU。

新路线：`历史 item ID → BOS + 每物品三级层级 Token → Token/绝对位置/层级类型嵌入 → HSTU Token 级因果序列 → 分层 next-token loss`。推理依次运行 `history`、`history+c1`、`history+c1+c2`。

## 2. Token 词表与 Embedding

三级码本、每级 1024 个码。`PAD=0`、`BOS=1`，`token(level, code)=2+level×1024+code`，总词表 3074。PAD/BOS 不能反解为普通码。码本向量 `[3,1024,64]` 按层复制进统一表 `[3074,64]`；PAD 行固定为零，BOS 随机初始化并可学习。64D 经线性层投影为 HSTU 的 50D。主实验 `freeze_semantic_token_embeddings=false`，历史输入和生成反馈共用同一表。

输入还叠加展开后绝对 Token 位置嵌入，以及 PAD/BOS/L0/L1/L2 五类 level/type embedding。最大历史为 200 个物品，训练还接一个目标物品，因此最大长度为 `1+3×(200+1)=604`；纯历史最大为 601，生成两个反馈后最大为 603。

## 3. 序列、时间与缺码

例：`[i1,i2] → [BOS,c1(i1),c2(i1),c3(i1),c1(i2),c2(i2),c3(i2)]`。同一物品三个 Token 复制原交互时间戳，BOS 使用首个有效历史物品的时间戳。缺少完整 Semantic ID 的历史物品采用显式 `filter` 策略并累计数量；不会变成 PAD，也不会使用目标内容回填，因而不产生标签泄漏。正式报告需回填训练/验证/测试缺码计数。

## 4. Next-token 目标和因果性

完整序列严格右移：输入为除末 Token 外的序列，目标为除首 Token 外的序列。Padding 不参与损失；默认关闭无个性化上下文的 `BOS→首历史物品 c1`，可通过配置打开。目标 Token 的层决定使用哪个 1024 类分类头，三级交叉熵取平均，并分别记录 loss、accuracy 和样本数。

HSTU 使用上三角未来位置屏蔽。Teacher forcing 的真实 Token 只能作为其后位置的输入，当前输出不能看到自身目标或后续 Token。测试还验证了改变未来 Token 不改变更早输出。

## 5. Token 级生成、Trie 与 Beam

生成 c1 后将带 level-0 偏移的 Token 追加到序列并重新运行 HSTU；c2 同理。当前正确性版本使用完整重算，后续可引入等价 KV cache。

Trie 由 bridge 中所有内容侧有效路径建立，包括冷物品。每一级先取得当前前缀的全部合法子码，再只在合法集合内归一化和 Top-K，避免“全 1024 Top-K 后过滤”丢失合法分支。Trie 暴露冷物品内容路径，但不包含冷物品交互标签；冷物品仍不得出现在 HSTU 训练历史和正目标中。

主配置从 beam 32 开始，自适应翻倍至最多 256，目标为 200 个唯一物品。每用户记录合法完整路径、映射前路径、映射后唯一物品、冲突物品、Beam-only 数量、Exact 补齐数和实际 beam。完整路径映射到全部同码物品；冲突物品共享路径分数，不加 Item-ID residual 或精排器。

## 6. Exact Scorer

候选物品分数严格为：

`log p(c1|history) + log p(c2|history,c1) + log p(c3|history,c1,c2)`。

实现按空前缀、相同 c1、相同 `(c1,c2)` 批量共享计算，但每个条件仍来自真正追加 Token 后的 HSTU 输出。单测将批量结果与逐级手工三次前向求和比较。Exact 用于 overall/cold/warm 全局 AUC、全局排序分析及 Beam 不足时的实验性补齐，不依赖目标路径进入 Beam。

## 7. 数据协议审计

主实验固定 MovieLens-1M、seed 42、既有 371 个冷物品、相同 bridge、相同训练/验证/测试样本、学习率调度、early stopping 和 corrected MRR。正式运行必须记录：冷物品在训练历史/目标/负例的次数，冷物品有效/缺失 Semantic ID 数，RQ-VAE 训练物品与 HSTU 冷物品重叠数，以及验证/测试目标哈希。

本地同一归档 bridge 审计：`item_to_codes=(3953,3)`、`codebook_vectors=(3,1024,64)`；3,416 个物品有完整码，共 3,376 条唯一路径，36 条冲突路径影响 76 个物品。既有审计报告给出冷物品覆盖 338/371（91.11%），即 33 个冷物品缺码。严格训练历史/目标次数、负例次数、RQ-VAE train-split 精确重叠和验证/测试目标哈希仍须服务器启动时用持久化 split 复核。

## 8. 张量形状

| 张量 | 主实验形状/规则 |
|---|---|
| item_to_codes | `[max_item_id+1, 3]` |
| codebook vectors | `[3,1024,64]` |
| unified token embedding | `[3074,64]` |
| projected HSTU input | `[B,T,50]` |
| token IDs / timestamps | `[B,T]`, `T≤604` |
| valid mask | `[B,T,1]` |
| causal mask | `[604,604]` |
| each output head | `[*,1024]` |
| exact scores | `[valid candidate items]` per user |

## 9. 测试和 smoke

本地测试覆盖请求中的 34 项，并额外覆盖动态最大长度、合法集合先过滤、冲突 Exact 同分。`scripts/full_token_smoke.py` 使用合成 bridge、两个用户和两次优化步骤，执行 loss/backward、validation 所需张量流、三级生成、Beam 和小候选 Exact；它不读取 MovieLens，也不会启动完整训练。

本地执行记录（2026-08-03）：Full-Token 专项测试（含参数化用例）全部通过；全仓库 `63 passed, 1 skipped`。配置以 `--cfg job --resolve` 成功解析，统一最大长度为 604。独立 smoke 完成两个训练 batch、一次 validation、三级生成、Beam 和三候选 Exact；loss/输出均无 NaN。一次参考输出为：有效 Token 数 `[10,7]`，三级 loss `[2.855951,2.289483,1.960572]`，三级 accuracy `[0.0,0.2,0.2]`，4 条合法 Beam 路径、4 个映射物品、Exact shape `(3,)`，两个 batch 约 0.013 秒（合成 CPU 环境，仅用于正确性，不代表正式性能）。

## 10. 最终结果

完整指标表见 `reproduction/full_token_route_b_results.md`。前四组直接沿用既有报告；Full-Token 正式结果、三级准确率、Beam/Exact 差异、训练时间、最佳 epoch、峰值显存和参数量均待服务器回填。

<!-- SERVER_RESULTS:START -->

服务器结果摘要：待运行 `scripts/summarize_full_token_route_b.py` 自动回填。

<!-- SERVER_RESULTS:END -->

## 11. 结论问题（待结果后回答）

- 是否提高 cold AUC：待 Full-Token Exact 全局排序结果。
- 是否提高 cold HR：必须分别比较 Beam-only 与 Beam+Exact，不能只看补齐后 Top-200。
- 改善来自概率还是搜索：若 Exact 提升而 Beam-only 不升，主要是搜索覆盖；若 Exact 与 Beam-only 都升，概率模型更可能是主因。
- 实验限制：内容侧 Trie 已知冷路径；同码冲突无法区分；全重算生成和 Exact 仍昂贵；只验证 ML-1M/单 seed。
- 是否值得写进简历：只有在协议审计通过、结果可复现且能诚实说明正负结果后才值得；即便指标未升，若能量化旧实现偏差、搜索瓶颈和资源代价，也可作为高质量工程/研究诊断经历。
- 下一步：等价增量 cache、前缀动态批处理、冲突感知但独立报告的 residual reranker、多 seed 和更大数据集；这些都不应混入本次主实验。
