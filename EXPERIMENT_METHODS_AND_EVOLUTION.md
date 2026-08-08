# 基于 HSTU 与 RQ-VAE 的冷启动推荐：实验方法、结果与演进全记录

> 最后更新：2026-08-08  
> 数据集：MovieLens-1M  
> 当前正式结果：固定随机种子 42  
> 阅读目标：不依赖其他文档，也能理解本项目的问题定义、数据协议、模型公式、五条初始路线、负采样修正、个性化 Route A 及后续消融是怎样一步步完成的。

## 0. 一页读懂整个实验过程

本项目研究的是**严格物品冷启动检索**：预先选出一批冷物品，训练时不允许模型在用户历史、正样本或随机负样本中看到它们；测试时再把它们放回完整物品库，观察模型能否仅依靠内容信息把冷目标排到前面。

实验按下面的顺序推进：

```text
MovieLens-1M
  │
  ├─ 构造严格冷物品集合：371 个冷物品，训练曝光为 0
  │
  ├─ 用标题 MiniLM 向量 + 类型 one-hot 训练 RQ-VAE
  │     └─ 为 3,416 个合格物品导出三层 Semantic ID
  │
  ├─ 第一阶段：五路线对照
  │     ├─ HSTU ID baseline
  │     ├─ MiniLM Dense baseline
  │     ├─ RQ-VAE Route A
  │     ├─ 分层生成 Route B
  │     └─ Full-Token Route B
  │
  ├─ 第二阶段：发现 sampled-softmax 把冷物品当成负样本
  │     └─ 将负样本池改为 warm-only，重训 ID / Dense / Route A
  │
  ├─ 第三阶段：Personalized Route A
  │     ├─ A0：共享内容投影
  │     ├─ A2：暖物品 ID 低秩残差
  │     ├─ A3：内容生成残差
  │     └─ A4：Teacher-Student + 热度门控
  │
  ├─ 第四阶段：机制消融
  │     ├─ 历史侧 H / 候选侧 C 残差开关
  │     ├─ Cold-vs-All / Warm / Cold 候选组
  │     ├─ 冷残差强度 alpha 扫描
  │     └─ 蒸馏权重 beta 与 Frozen Teacher
  │
  └─ 当前下一步：Free-ID Residual
        ├─ 暖物品：内容基础 + 自由 ID 协同残差
        ├─ 冷物品：只保留内容基础
        └─ 同时改为独立 train / validation / test 目标
```

截至当前，最重要的实验结论不是“生成式路线一定更好”，而是：

1. RQ-VAE Route A 比纯 ID 和 Dense baseline 更适合严格冷启动；
2. 把冷物品放进训练负样本池会形成伪负例，必须排除；
3. 暖物品协同残差可以显著提升 Warm/Overall，并通过候选重排间接提升 Cold；
4. 直接根据内容生成冷物品协同残差目前不稳定，往往提高 AUC、却损害 Top-K；
5. 当前证据更支持“暖物品使用内容 + 协同残差，冷物品只使用内容基础”的显式分工。

---

## 1. 任务定义与符号

对用户 \(u\)，按时间排序后的行为序列记为：

$$
S_u=(i_1,i_2,\ldots,i_T).
$$

模型根据历史：

$$
h_u=(i_1,\ldots,i_{T-1})
$$

预测目标物品：

$$
y_u=i_T.
$$

设完整评估物品集合为 \(\mathcal I\)，严格冷物品集合为 \(\mathcal I_c\)，暖物品集合为：

$$
\mathcal I_w=\mathcal I\setminus\mathcal I_c.
$$

严格冷启动约束是：对任意 \(i\in\mathcal I_c\)，HSTU 训练阶段都必须满足：

$$
N_i^{\mathrm{history}}=N_i^{\mathrm{positive}}=N_i^{\mathrm{negative}}=0.
$$

但评估时必须满足：

$$
\mathcal I_c\subseteq\mathcal I_{\mathrm{candidate}},
$$

否则就不是“从完整目录中找冷物品”，而只是一个人为缩小候选集的任务。

---

## 2. 数据是怎样一步步划分的

### 2.1 MovieLens-1M 用户序列

项目读取 MovieLens-1M 的评分记录，按用户分组并按时间戳排序。每个用户形成一条序列，因此原协议共有 6,040 条验证/测试样本，每条样本使用最后一个行为作为目标，其余最近最多 200 个行为作为历史。训练时不是每个用户只计算一次末尾 loss，而是对保留下来的序列构造所有相邻 next-item 监督位置。

项目保留 MovieLens 原始 movie ID，不把物品重新映射成连续编号。完整电影目录中有 3,883 个评估候选；打分时会过滤用户已经交互过的物品，因此每条样本平均约有 3,771 个有效候选。

### 2.2 371 个严格冷物品是怎样得到的

先收集评分序列里真正出现过的物品集合 \(\mathcal I_{\mathrm{observed}}\)。MovieLens-1M 中共有 3,706 个观测物品，然后使用：

$$
|\mathcal I_c|=\mathrm{round}(0.1\times3706)=371
$$

并以固定随机种子 42 无放回抽样，得到 371 个冷物品。物品 ID 排序后写入持久化 JSON；后续运行如果数据、比例、种子或 ID 列表与原文件不一致，程序直接报错，而不是静默覆盖。

### 2.3 545 条冷目标样本是怎样得到的

371 是**冷物品种类数**，545 是**测试样本数**，二者含义不同。对 6,040 个用户分别取最后一次交互作为目标，其中目标落在冷物品集合内的用户有 545 个，所以：

$$
N_{\mathrm{cold}}=545,
\qquad
N_{\mathrm{warm}}=6040-545=5495.
$$

同一个冷物品可能是多个用户的测试目标，所以 545 可以大于 371。

### 2.4 严格冷训练数据如何构造

对训练序列先保留相应时间切分之前的行为，再删除所有冷物品。删除后少于两个行为的用户行不能形成“历史—目标”训练对，会被丢弃。最终审计得到：

| 项目 | 数值 |
|---|---:|
| 完整评估候选 | 3,883 |
| 严格冷物品 | 371 |
| 暖物品 | 3,512 |
| 测试样本 | 6,040 |
| Cold 测试样本 | 545 |
| Warm 测试样本 | 5,495 |
| 训练历史中的冷曝光 | 0 |
| 训练正样本中的冷曝光 | 0 |
| 修正后训练负样本中的冷曝光 | 0 |
| 评估候选中的冷物品 | 371 |

### 2.5 三种划分不能混为一谈

本项目同时存在三套不同含义的划分：

1. **RQ-VAE 物品划分**：对经过“评分次数不少于 5”过滤后的 3,416 个物品，以 seed 42 随机划分 80% 训练、20% 验证；它只用于学习和验证内容量化器。
2. **HSTU 时间目标划分**：按每个用户行为的时间顺序决定训练、验证和测试目标。
3. **严格冷物品划分**：从 3,706 个观测物品中固定抽取 371 个物品，并从 HSTU 训练曝光中全部删除。

RQ-VAE 的 80/20 划分与 HSTU 的冷物品划分相互独立。RQ-VAE 只在其 80% 物品上更新参数，训练完成后为全部 3,416 个合格物品导出 Semantic ID，因此 RQ-VAE 验证物品也能得到零样本编码。

这也意味着：当前“严格冷”严格保证的是 **HSTU 行为训练曝光为 0**，不能进一步声称 371 个冷物品全部被排除在 RQ-VAE 内容训练之外。RQ-VAE 只使用标题和类型做内容重建，不使用目标推荐标签，因此这不构成协同行为泄漏；但如果要研究“连内容编码器也从未见过该物品”的更强协议，就必须让 RQ-VAE 的 item holdout 与 HSTU cold split 显式对齐。

### 2.6 当前结果中的验证/测试限制

历史正式结果使用：

```text
train ignore_last_n = 1
validation ignore_last_n = 0
test ignore_last_n = 0
```

也就是说，训练可见前缀截止到每个用户倒数第二个行为，并在该前缀内逐位置计算 next-item loss；validation 和 test 都使用最后一个行为作为评估目标，两者目标集合相同。checkpoint 又根据 validation 指标选择，所以这些结果适合机制研究和阶段性比较，**不是完全独立的无偏最终测试**。

当前准备中的 Free-ID 实验已经改为：

```text
train ignore_last_n = 2
validation ignore_last_n = 1
test ignore_last_n = 0
```

训练前缀截止到倒数第三个行为，validation 使用倒数第二个目标，test 使用最后一个目标，并增加 validation/test 目标哈希与重叠审计。

### 2.7 “划分哈希与协议审计”具体是什么

程序把关键 ID 或样本记录按固定顺序序列化后计算 SHA-256：

$$
H=\mathrm{SHA256}(x_1\Vert x_2\Vert\cdots\Vert x_n).
$$

哈希不用于训练，而是数据划分的“指纹”。两次实验如果 cold split hash、train/validation/test hash、负样本池 hash 和全候选 hash 一致，才能证明它们比较的是同一批数据。审计还直接统计冷物品在训练历史、正样本、负样本和评估候选中的出现次数，避免仅凭配置文件推测协议正确。

---

## 3. RQ-VAE 如何把内容变成 Semantic ID

### 3.1 内容特征

物品 \(i\) 的标题先由 all-MiniLM-L6-v2 编码为 384 维文本向量，电影类型转为 18 维 one-hot，拼接得到：

$$
x_i=[x_i^{\mathrm{text}};x_i^{\mathrm{genre}}]\in\mathbb R^{402}.
$$

“评分次数不少于 5”来自数据预处理的 5-core 合格物品过滤，用来限定 RQ-VAE 的训练/验证/导出物品集合。需要强调：RQ-VAE 的损失本身只重建内容，并不使用评分次数作为监督，所以这个阈值不是模型公式上的必要条件，而是当前实现的数据质量与目录对齐选择。过滤后有 3,416 个物品；它不会改变 HSTU 的 3,883 个完整评估候选，因此没有 Semantic ID 的物品仍可能作为竞争候选存在，也造成 33 个冷物品缺少 Semantic ID。若目标是覆盖所有冷物品，更合理的后续做法是为完整目录编码，或给缺码物品增加 Dense fallback。

### 3.2 编码器与三级残差量化

编码器将内容特征压缩到 64 维：

$$
h_i=f_\theta(x_i)\in\mathbb R^{64}.
$$

使用三层码本，每层 1,024 个 64 维向量。令初始残差为：

$$
r_i^{(0)}=h_i.
$$

第 \(m\) 层选择距离当前残差最近的码本向量：

$$
c_i^{(m)}=\arg\min_{k\in\{0,\ldots,1023\}}
\left\|r_i^{(m-1)}-q_k^{(m)}\right\|_2^2,
$$

并更新残差：

$$
r_i^{(m)}=r_i^{(m-1)}-q_{c_i^{(m)}}^{(m)}.
$$

三层量化表示为：

$$
\widehat h_i=\sum_{m=1}^{3}q_{c_i^{(m)}}^{(m)},
$$

物品 Semantic ID 为：

$$
\mathrm{SID}(i)=\left(c_i^{(1)},c_i^{(2)},c_i^{(3)}\right).
$$

解码器从 \(\widehat h_i\) 重建原内容特征：

$$
\widehat x_i=d_\phi(\widehat h_i).
$$

代码中的内容重建损失把 384 维连续文本部分和 18 维多标签类型部分分开处理：

$$
\mathcal L_{\mathrm{rec}}^{\mathrm{RQ}}
=\left\|\mathrm{Norm}(\widehat x_i^{\mathrm{text}})-x_i^{\mathrm{text}}\right\|_2^2
+\mathrm{BCELogits}\left(\widehat x_i^{\mathrm{genre}},x_i^{\mathrm{genre}}\right).
$$

每层量化损失为：

$$
\mathcal L_{\mathrm{quant}}^{(m)}
=\left\|\mathrm{sg}\left[r_i^{(m-1)}\right]-q_{c_i^{(m)}}^{(m)}\right\|_2^2
+\lambda_{\mathrm{commit}}
\left\|r_i^{(m-1)}-\mathrm{sg}\left[q_{c_i^{(m)}}^{(m)}\right]\right\|_2^2.
$$

因此总训练目标为：

$$
\mathcal L_{\mathrm{RQ-VAE}}
=\mathcal L_{\mathrm{rec}}^{\mathrm{RQ}}
+\sum_{m=1}^{3}\mathcal L_{\mathrm{quant}}^{(m)},
$$

其中 \(\lambda_{\mathrm{commit}}=0.25\)，码本使用 EMA 更新，衰减率为 0.99，并启用 dead-code reset。

### 3.3 码本 perplexity

对某一层，若第 \(k\) 个码字被选中 \(n_k\) 次，则：

$$
p_k=\frac{n_k}{\sum_j n_j}.
$$

码本 perplexity 为：

$$
\mathrm{PPL}=\exp\left(-\sum_k p_k\log p_k\right).
$$

它可以理解为“等效使用了多少个码字”。本次三层结果分别为 767.76、809.53、819.54，说明没有只挤在少量码字上。

### 3.4 RQ-VAE 配置与质量审计

| 项目 | 配置或结果 |
|---|---|
| 输入维度 | 402 |
| 隐藏层 | 512 → 256 → 128 |
| latent / code 向量维度 | 64 |
| 码本层数 | 3 |
| 每层码本大小 | 1,024 |
| batch size | 2,048 |
| 训练步数 | 50,000 |
| learning rate | \(10^{-4}\) |
| weight decay | 0.01 |
| mixed precision | BF16 |
| 三层码字使用率 | 98.73% / 98.63% / 98.93% |
| 三层 dead code | 0 / 0 / 0 |
| 三层 perplexity | 767.76 / 809.53 / 819.54 |
| 完整三层路径 collision ratio | 1.171% |
| 每个 ID 最大重复比例 | 0.117% |
| 重建误差/维度 | 0.02847 |
| 有效 bridge 物品 | 3,416 |
| 冷物品 Semantic ID 覆盖 | 338 / 371 = 91.11% |

3,416 个有效编码物品形成 3,376 条唯一路径；36 条路径发生冲突，影响 76 个物品。同一路径对应多个物品时，Token 生成模型无法仅靠 Semantic ID 区分这些物品，这也是 Route B 的结构性限制之一。

---

## 4. HSTU 主干、Jagged Sequence 与检索损失

### 4.1 HSTU 输入

序列中第 \(t\) 个物品的输入由物品表示和可学习位置表示组成：

$$
x_t=e_{i_t}+p_t.
$$

时间戳不直接加到物品 embedding 上，而是用于计算注意力中的相对时间偏置。

### 4.2 Jagged Sequence

不同用户历史长度不同。Jagged 表示把所有有效 token 连续存储为：

$$
X_{\mathrm{jagged}}=[X_1;X_2;\ldots;X_B],
$$

再用 offsets：

$$
o=[0,|X_1|,|X_1|+|X_2|,\ldots]
$$

记录每个用户的起止位置。它主要改变存储和算子输入方式，不改变序列语义。项目在 HSTU 计算中按需在 Jagged 和 padded dense 之间转换。

### 4.3 本项目实际使用的 HSTU 注意力

每层先对输入做 LayerNorm，再通过一次线性投影和 SiLU 得到 \(u,v,q,k\)。对位置 \(a,b\)，相对偏置由位置偏置与时间桶偏置组成：

$$
b_{ab}=b^{\mathrm{pos}}_{a-b}
+b^{\mathrm{time}}_{\rho(t_b-t_a)}.
$$

时间差分桶函数为：

$$
\rho(\Delta t)=
\mathrm{clip}\left(
\left\lfloor\frac{\log(\max(|\Delta t|,1))}{0.301}\right\rfloor,
0,128
\right).
$$

本项目的 `rel_bias` 路径不使用 softmax，而是：

$$
a_{ab}=M_{ab}\cdot
\frac{\mathrm{SiLU}\left(q_a^\top k_b+b_{ab}\right)}{N},
$$

其中 \(M\) 是因果 Mask，\(N\) 是当前 padded 最大长度。注意力输出为：

$$
z_a=\sum_b a_{ab}v_b.
$$

HSTU 单元的门控残差可简写为：

$$
x'_a=x_a+W_o\left(u_a\odot\mathrm{LN}(z_a)\right).
$$

正式实验使用 2 个 HSTU block、1 个 head，item embedding、attention 和 linear hidden 维度均为 50。

### 4.4 L2Norm postprocessor 与点积检索

HSTU 输出先做：

$$
\widehat h_u=\frac{h_u}{\max(\|h_u\|_2,10^{-6})}.
$$

候选物品向量也在打分前归一化：

$$
\widehat e_i=\frac{e_i}{\max(\|e_i\|_2,10^{-6})}.
$$

最终分数为：

$$
s(u,i)=\widehat h_u^\top\widehat e_i.
$$

因此它等价于余弦相似度。需要注意：Route A 和 Personalized Route A 不在进入 HSTU 之前提前归一化物品向量；历史侧保留向量模长，用户输出与候选打分侧在原有位置归一化。

### 4.5 sampled-softmax 训练目标

对正物品 \(y\) 和采样负物品集合 \(\mathcal N_u\)，温度 \(T=0.05\)，单个位置的推荐损失为：

$$
\mathcal L_{\mathrm{rec}}
=-\log
\frac{\exp(s(u,y)/T)}
{\exp(s(u,y)/T)+\sum_{j\in\mathcal N_u}\exp(s(u,j)/T)}.
$$

每个有效训练位置采样 128 个负物品。修正后的严格协议只从 3,512 个暖物品中采样。

### 4.6 公共训练环境与选择方式

首轮正式复现实验运行于单张 RTX 4090D，软件环境为 Python 3.10、PyTorch 2.5.1+cu124 和 Lightning 2.5.0.post0。item-level 实验 batch size 为 128，优化器使用 AdamW，初始 learning rate 为 0.001，weight decay 为 0.001；最多训练 500 epochs，以 validation NDCG@100 早停。RQ-VAE 的 50,000 steps 约用时 6 分 50 秒；首轮 HSTU ID、Dense、Route A、Route B 分别在第 109、193、157、80 个 epoch 结束，Route B 总用时约 4 小时 14 分。Full-Token 的服务器快跑配置使用 BF16、batch size 128 的等效梯度累积和 8 个 DataLoader workers，训练 50 epochs，约用时 5 小时 29 分。

这些时间只说明本次硬件与实现的量级，不是脱离环境的通用延迟结论。尤其 Exact Scorer 需要对所有有效物品路径打分，适合离线评估、重排或候选不足时补齐；若在线全量执行，需要缓存前缀概率、分块矩阵计算或两阶段召回。

---

## 5. 第一阶段：五条路线分别怎样做

### 5.1 HSTU ID baseline

每个物品直接查独立 ID embedding 表：

$$
e_i=E_{\mathrm{ID}}[i].
$$

暖物品可以从行为中学习向量；严格冷物品的 ID embedding 没有正向训练信号，因此这是验证“纯协同模型能否处理零曝光物品”的下界。

### 5.2 MiniLM Dense baseline

直接使用标题 MiniLM 向量 \(d_i\in\mathbb R^{384}\)，通过共享线性层投影到 HSTU 的 50 维空间：

$$
e_i=W_d d_i+b_d.
$$

MiniLM 特征冻结，训练共享投影和 HSTU。该路线检验连续文本内容在不经过离散量化时的冷启动能力。

### 5.3 RQ-VAE Route A：语义码向量组合

取三层码本向量并拼接：

$$
z_i=
\left[
q_{c_i^{(1)}}^{(1)};
q_{c_i^{(2)}}^{(2)};
q_{c_i^{(3)}}^{(3)}
\right]
\in\mathbb R^{192}.
$$

所有物品共享一个投影：

$$
e_i=W_0z_i+b,
\qquad W_0\in\mathbb R^{50\times192}.
$$

RQ-VAE 码本冻结；\(W_0\) 与 HSTU 在暖物品行为上训练。冷物品即使没有 ID 训练曝光，只要具有 Semantic ID，就能通过同一个 \(W_0\) 得到推荐空间表示。

这里的 \(W_0\) 更准确地说是一个**共享线性变换/投影**，它可以包含旋转、缩放、剪切和降维，不是严格的纯旋转矩阵。可以把 \(z_i\) 理解为内容基础，把 \(W_0\) 理解为从内容空间到行为推荐空间的全局校准。

### 5.4 Route B：物品级历史 + 分层 Semantic ID 生成

旧 Route B 没有把整段历史展开为 Semantic Token 序列。历史仍以物品为一个位置，并使用语义码组合后的 item embedding 输入 HSTU，得到固定用户上下文 \(h_u\)。目标物品的三层码依次预测：

$$
p(c_1\mid h_u),
$$

$$
p(c_2\mid h_u,c_1),
$$

$$
p(c_3\mid h_u,c_1,c_2).
$$

训练使用 teacher forcing，三级交叉熵取平均：

$$
\mathcal L_{\mathrm{RouteB}}
=\frac{1}{3}\sum_{m=1}^{3}
-\log p\left(c_m^*\mid h_u,c_{<m}^*\right).
$$

推理使用宽度 8 的前缀约束 Beam Search，只允许生成数据中真实存在的 Semantic ID 前缀。完整路径映射回物品；路径冲突时，一个路径可能对应多个物品。

如果 Beam 映射出的唯一、合法、非历史物品少于 200，就对所有具有有效 Semantic ID 的物品计算完整路径分数：

$$
s_{\mathrm{token}}(u,i)
=\log p(c_{i1}\mid h_u)
+\log p(c_{i2}\mid h_u,c_{i1})
+\log p(c_{i3}\mid h_u,c_{i1},c_{i2}),
$$

再排除历史和重复物品，补足 Top-200。这里“Exact”表示准确计算每一条已知物品路径的三级概率和，不表示它的语义目标一定正确。

Route B 的 AUC 使用所有有效 Semantic ID 物品的 Exact 分数；HR/NDCG/MRR 使用 Beam 候选加 Exact 补齐后的 Top-200。

### 5.5 Full-Token Route B：历史和目标全部 Token 化

Full-Token Route B 用层级感知词表：

$$
\mathrm{token}(m,c)=2+m\times1024+c,
$$

其中 0 为 PAD、1 为 BOS，总词表大小：

$$
2+3\times1024=3074.
$$

原物品序列：

$$
(i_1,i_2,\ldots,i_T)
$$

被展开为：

$$
(\mathrm{BOS},c_{11},c_{12},c_{13},c_{21},c_{22},c_{23},\ldots).
$$

训练做严格 next-token causal modeling。预测某个物品的 \(c_3\) 时，可以看到之前所有物品以及本物品的 \(c_1,c_2\)，但不会跨越物品边界把下一物品的 token 当作当前物品内部条件。损失为各层有效 token 的交叉熵均值：

$$
\mathcal L_{\mathrm{FullToken}}
=\frac{1}{3}\sum_{m=1}^{3}
\frac{1}{|\Omega_m|}
\sum_{t\in\Omega_m}
-\log p(c_t^*\mid c_{<t}^*).
$$

推理时先生成 \(c_1\)，把它真的追加回 HSTU 序列；再生成并追加 \(c_2\)，然后预测 \(c_3\)。合法前缀由 Trie 约束，Beam 从 32 自适应扩大到最多 256。若唯一物品不足 200，再使用全路径 Exact Scorer 补齐。

### 5.6 五路线第一轮结果

这轮完成了模型结构对照，但前三个 sampled-softmax 路线当时仍会把预留冷物品采成负例，所以它们保留为历史结果，不作为最终严格主结论。Route B 两组不使用这个物品级负采样器。

#### Cold：545 条冷目标，全目录候选

| Model | AUC | HR@100 | NDCG@100 |
|---|---:|---:|---:|
| HSTU ID | 0.088247 | 0.000000 | 0.000000 |
| MiniLM Dense | 0.475650 | 0.020183 | 0.003605 |
| RQ-VAE Route A | **0.592034** | **0.102752** | **0.021144** |
| Route B | 0.228927 | 0.001835 | 0.000530 |
| Full-Token Route B | 0.077678 | 0.001835 | 0.000395 |

#### Warm：5,495 条暖目标，全目录候选

| Model | AUC | HR@100 | NDCG@100 |
|---|---:|---:|---:|
| HSTU ID | **0.897023** | **0.640947** | **0.234061** |
| MiniLM Dense | 0.884091 | 0.505004 | 0.172906 |
| RQ-VAE Route A | 0.883880 | 0.484987 | 0.152206 |
| Route B | 0.800015 | 0.289536 | 0.088466 |
| Full-Token Route B | 0.859010 | 0.416197 | 0.124199 |

#### Overall：全部 6,040 条样本

| Model | AUC | HR@100 | NDCG@100 |
|---|---:|---:|---:|
| HSTU ID | 0.824046 | **0.583113** | **0.212941** |
| MiniLM Dense | 0.847237 | 0.461258 | 0.157630 |
| RQ-VAE Route A | **0.857546** | 0.450497 | 0.140380 |
| Route B | 0.748485 | 0.263576 | 0.080531 |
| Full-Token Route B | 0.788509 | 0.378808 | 0.113028 |

### 5.7 为什么 Route B 和 Full-Token Route B 的 Cold 很差

实验逐步排除了“只是 Beam 太窄”这一单一解释：Full-Token 把 Beam 扩到 256，平均能得到 217.616 个唯一候选，Exact 只需平均补 4.263 个，但 Cold HR@100 仍只有 0.001835。主要问题包括：

1. **三级误差累积**：任意一层预测错误，完整物品路径就可能错误；
2. **teacher forcing 与推理解码不一致**：训练看到真实前缀，推理看到自身生成前缀；
3. **RQ-VAE 优化的是内容重建，不是下一物品生成**：语义码适合压缩内容，不保证形成最易预测、最利于排序的生成目标；
4. **路径冲突**：76 个物品受完整码冲突影响，同路径内无法仅靠 token 概率区分；
5. **前缀约束会放大早期错误**：错误的 \(c_1\) 会把后续搜索限制在错误子树；
6. **长 Token 序列增加优化难度**：最多 200 个历史物品会展开为约 601 个语义 token，再加 BOS/目标位置；
7. **冷物品没有行为监督**：生成头主要由暖物品目标训练，容易学习暖路径先验。

Full-Token 相比旧 Route B 的 Overall HR@100 从 0.263576 提升到 0.378808，NDCG@100 从 0.080531 提升到 0.113028，说明“真正的 Token 自回归”改善了整体建模；但 Cold HR@100 仍只有约 1/545，说明增益几乎都来自 Warm。

---

## 6. 第二阶段：为什么以及怎样修正负样本采样

### 6.1 发现的问题

第一轮 HSTU ID、Dense 和 Route A 虽然从训练历史和正样本中删除了冷物品，但 `LocalNegativesSampler` 的候选池仍是全部物品。于是冷物品可能被 sampled-softmax 明确训练成“不该推荐”的负例：

也就是同一个物品可能同时满足：

$$
i_c\in\mathcal I_c,
\qquad
i_c\in\mathcal N_u.
$$

这与严格冷启动“训练阶段对冷物品没有行为标签”的定义冲突。

### 6.2 修正方法

只把训练负样本池改为：

$$
\mathcal N_u\subseteq\mathcal I_w,
$$

而评估候选仍为：

$$
\mathcal I_{\mathrm{candidate}}=\mathcal I_w\cup\mathcal I_c.
$$

因此只需重训使用物品级 sampled-softmax 的三组：HSTU ID、Dense、Route A。Route B 与 Full-Token Route B 使用 token 交叉熵，不受这个负采样池修改影响。

### 6.3 修正后的推荐主基线

#### Cold

| Model | AUC | HR@10 | HR@100 | NDCG@10 | NDCG@100 | corrected MRR |
|---|---:|---:|---:|---:|---:|---:|
| HSTU ID | 0.439583 | 0.000000 | 0.014679 | 0.000000 | 0.002371 | 0.000309 |
| MiniLM Dense | 0.594716 | 0.011009 | 0.045872 | 0.006355 | 0.012700 | 0.006010 |
| RQ-VAE Route A / A0 | **0.674347** | **0.036697** | **0.185321** | **0.019973** | **0.047944** | **0.019836** |

#### Warm

| Model | AUC | HR@10 | HR@100 | NDCG@10 | NDCG@100 | corrected MRR |
|---|---:|---:|---:|---:|---:|---:|
| HSTU ID | **0.943485** | **0.282621** | **0.668972** | **0.155064** | **0.234596** | **0.131902** |
| MiniLM Dense | 0.878081 | 0.199454 | 0.492630 | 0.108281 | 0.167889 | 0.092392 |
| RQ-VAE Route A / A0 | 0.880001 | 0.167971 | 0.481893 | 0.089823 | 0.151896 | 0.077581 |

#### Overall

| Model | AUC | HR@10 | HR@100 | NDCG@10 | NDCG@100 | corrected MRR |
|---|---:|---:|---:|---:|---:|---:|
| HSTU ID | **0.898017** | **0.257119** | **0.609934** | **0.141072** | **0.213642** | **0.120029** |
| MiniLM Dense | 0.852512 | 0.182450 | 0.452318 | 0.099084 | 0.153886 | 0.084597 |
| RQ-VAE Route A / A0 | 0.861444 | 0.156126 | 0.455132 | 0.083520 | 0.142516 | 0.072370 |

相对 HSTU ID baseline，Route A 的 Cold AUC 从 0.439583 提升到 0.674347，相对提升 53.40%；Cold NDCG@100 从 0.002371 提升到 0.047944，绝对提升 0.045573。Dense 次之。与此同时，HSTU ID 在 Warm 和 Overall 上仍然最好，清楚地呈现了“协同记忆能力—冷启动泛化能力”的权衡。

---

## 7. 指标到底怎样计算

### 7.1 每条样本的有效候选

对用户 \(u\)，从完整目录中移除历史物品后得到有效候选集 \(\mathcal C_u\)，并确保正目标 \(y_u\) 保留。Cold、Warm、Overall 使用的是同一个全目录候选协议，只是最后对哪些样本求平均不同。

### 7.2 AUC

单正样本检索中，每条样本的 AUC 是正目标击败有效负候选的比例，并对同分计 0.5：

$$
\mathrm{AUC}_u=
\frac{1}{|\mathcal C_u|-1}
\sum_{j\in\mathcal C_u\setminus\{y_u\}}
\left[
\mathbf 1(s_{uy}>s_{uj})
+\frac{1}{2}\mathbf 1(s_{uy}=s_{uj})
\right].
$$

AUC 更像全候选上的平均两两排序正确率，不等同于 MRR。若平均有效候选约为 3,771，则 AUC=0.886 粗略对应平均名次：

$$
1+(1-0.886)\times(3771-1)\approx431.
$$

所以 AUC 很高而 HR@100 不高并不矛盾。

### 7.3 HR、NDCG 与 corrected MRR

设正目标排名为 \(r_u\)：

$$
\mathrm{HR@K}_u=\mathbf 1(r_u\le K),
$$

单正样本时理想 DCG 为 1，因此：

$$
\mathrm{NDCG@K}_u=
\frac{\mathbf 1(r_u\le K)}{\log_2(r_u+1)},
$$

修正后的截断 MRR 为：

$$
\mathrm{MRR@200}_u=
\begin{cases}
1/r_u,&r_u\le200,\\
0,&r_u>200.
\end{cases}
$$

早期实现曾把 Top-200 未命中样本记为 \(1/201\)，导致 MRR 在 HR=0 时仍非零；后续报告统一使用 corrected MRR，未命中贡献为 0。

### 7.4 Cold、Warm、Overall

设样本集合为 \(\mathcal D\)，Cold 子集为 \(\mathcal D_c\)，Warm 子集为 \(\mathcal D_w\)。任意逐样本指标 \(m_u\) 的三种口径为：

$$
m_{\mathrm{cold}}=\frac{1}{545}\sum_{u\in\mathcal D_c}m_u,
$$

$$
m_{\mathrm{warm}}=\frac{1}{5495}\sum_{u\in\mathcal D_w}m_u,
$$

$$
m_{\mathrm{overall}}=\frac{545m_{\mathrm{cold}}+5495m_{\mathrm{warm}}}{6040}.
$$

因此 Overall 主要受 Warm 样本影响；它不是 Cold 与 Warm 的简单平均。

---

## 8. 第三阶段：Personalized Route A

原 Route A 的所有物品共享 \(W_0\)。这一阶段研究：能否保留可泛化的内容基础，同时用低秩残差表达暖物品的个性化协同语义。

### 8.1 A0：Shared W0

$$
e_i=W_0z_i+b.
$$

它就是修正 warm-only negative 后的 Route A 主基线。

### 8.2 A2：Warm-ID Low-Rank Residual

$$
e_i=W_0z_i+b+A\left(g_i^{\mathrm{collab}}\odot(Bz_i)\right),
$$

其中：

$$
A\in\mathbb R^{50\times8},
\qquad
B\in\mathbb R^{8\times192},
\qquad
g_i^{\mathrm{collab}}\in\mathbb R^8.
$$

只为 3,333 个具有正训练曝光的暖物品建立 \(g_i^{\mathrm{collab}}\)；冷物品没有参数索引，严格满足：

$$
g_i^{\mathrm{collab}}=0,
\qquad i\in\mathcal I_c.
$$

损失为：

$$
\mathcal L=\mathcal L_{\mathrm{rec}}+10^{-4}\mathcal L_{\mathrm{reg}}.
$$

它不是为每个物品存一个完整 \(50\times192\) 矩阵，而是用共享 \(A,B\) 和 8 维物品系数构造低秩个性化变化，参数更少、也更容易训练。

### 8.3 A3：Content-Generated Residual

用两层 MLP 从内容生成 8 维系数：

$$
g_i^{\mathrm{content}}=\tanh(G(z_i)),
$$

其中 \(G\) 的结构为 \(192\rightarrow128\rightarrow8\)。最终表示为：

$$
e_i=W_0z_i+b+A\left(g_i^{\mathrm{content}}\odot(Bz_i)\right).
$$

它试图学习“内容 \(\rightarrow\) 协同变化系数”的可泛化函数，因此冷物品也能零样本获得残差。

### 8.4 A4：Teacher-Student + 热度门控

暖物品 ID 系数作为 Teacher，内容生成器作为 Student：

$$
g_i^{\mathrm{content}}=G(z_i).
$$

根据物品训练曝光次数 \(n_i\) 计算：

$$
\lambda_i=\frac{n_i}{n_i+20},
$$

$$
g_i=\lambda_i g_i^{\mathrm{collab}}
+(1-\lambda_i)g_i^{\mathrm{content}}.
$$

严格冷物品满足 \(n_i=0\)，所以完全使用 Student。蒸馏损失为：

$$
\mathcal L_{\mathrm{distill}}
=\left\|G(z_i)-\mathrm{sg}\left(g_i^{\mathrm{collab}}\right)\right\|_2^2.
$$

总损失为：

$$
\mathcal L
=\mathcal L_{\mathrm{rec}}
+\beta\mathcal L_{\mathrm{distill}}
+\gamma\mathcal L_{\mathrm{reg}},
$$

原 A4 使用 \(\beta=0.1\)、\(\gamma=10^{-4}\)。这里的 beta 是**蒸馏损失权重**，不是残差缩放系数。

### 8.5 A0/A2/A3/A4 完整核心结果

#### Cold

| Model | AUC | HR@10 | HR@100 | NDCG@10 | NDCG@100 | corrected MRR |
|---|---:|---:|---:|---:|---:|---:|
| A0 Shared W0 | 0.674347 | 0.036697 | 0.185321 | 0.019973 | 0.047944 | 0.019836 |
| A2 Warm-ID Residual | **0.867796** | **0.056881** | **0.236697** | **0.028621** | **0.063826** | **0.027261** |
| A3 Content-Generated | 0.711488 | 0.022018 | 0.159633 | 0.009838 | 0.034943 | 0.010348 |
| A4 Teacher-Student | 0.752786 | 0.027523 | 0.174312 | 0.015962 | 0.044230 | 0.017623 |

#### Warm

| Model | AUC | HR@10 | HR@100 | NDCG@10 | NDCG@100 | corrected MRR |
|---|---:|---:|---:|---:|---:|---:|
| A0 Shared W0 | 0.880001 | 0.167971 | 0.481893 | 0.089823 | 0.151896 | 0.077581 |
| A2 Warm-ID Residual | 0.927351 | 0.241310 | 0.623840 | 0.132278 | 0.210455 | 0.114238 |
| A3 Content-Generated | 0.915959 | 0.220382 | 0.582712 | 0.117163 | 0.190231 | 0.099783 |
| A4 Teacher-Student | **0.938571** | **0.272611** | **0.665696** | **0.150418** | **0.230533** | **0.128994** |

#### Overall

| Model | AUC | HR@10 | HR@100 | NDCG@10 | NDCG@100 | corrected MRR |
|---|---:|---:|---:|---:|---:|---:|
| A0 Shared W0 | 0.861444 | 0.156126 | 0.455132 | 0.083520 | 0.142516 | 0.072370 |
| A2 Warm-ID Residual | **0.921978** | 0.224669 | 0.588907 | 0.122925 | 0.197224 | 0.106390 |
| A3 Content-Generated | 0.897509 | 0.202483 | 0.544536 | 0.107479 | 0.176219 | 0.091713 |
| A4 Teacher-Student | 0.921807 | **0.250497** | **0.621358** | **0.138286** | **0.213722** | **0.118945** |

A2 的 Cold 指标最好，但冷物品自身没有协同参数；A4 的 Warm 和 Overall Top-K 最好。A3/A4 的原始冷残差提高了 Cold AUC，却没有超过 A0 的 Cold Top-K，这推动了下一阶段的机制消融。

---

## 9. 第四阶段：A2 的 Cold 提升究竟来自哪里

### 9.1 历史侧 H / 候选侧 C 开关

H 表示是否给用户历史中的暖物品使用残差，C 表示是否给待排序的暖候选使用残差。复用同一个 A2 checkpoint，只改变推理路径：

| 设置 | 历史暖残差 | 候选暖残差 |
|---|---:|---:|
| H1-C1 | 开 | 开 |
| H1-C0 | 开 | 关 |
| H0-C1 | 关 | 开 |
| H0-C0 | 关 | 关 |

#### Cold

| Setting | AUC | HR@100 | NDCG@100 | corrected MRR |
|---|---:|---:|---:|---:|
| H1-C1 | 0.867796 | 0.236697 | 0.063826 | 0.027261 |
| H1-C0 | 0.735231 | 0.177982 | 0.042052 | 0.014599 |
| H0-C1 | **0.910856** | **0.355963** | **0.109055** | **0.054612** |
| H0-C0 | 0.730439 | 0.154128 | 0.043110 | 0.020838 |

#### Warm / Overall

| Setting | Warm AUC | Warm HR@100 | Warm NDCG@100 | Overall AUC | Overall HR@100 | Overall NDCG@100 |
|---|---:|---:|---:|---:|---:|---:|
| H1-C1 | **0.927351** | **0.623840** | **0.210455** | **0.921978** | **0.588907** | **0.197224** |
| H1-C0 | 0.815883 | 0.283894 | 0.078245 | 0.808605 | 0.274338 | 0.074979 |
| H0-C1 | 0.870728 | 0.339945 | 0.080390 | 0.874349 | 0.341391 | 0.082977 |
| H0-C0 | 0.782370 | 0.215469 | 0.051212 | 0.777684 | 0.209934 | 0.050481 |

结论是：A2 的 Cold 增益主要来自**候选侧暖物品残差造成的暖候选重排与暖冷分数校准**，不是冷物品获得了协同向量。H0-C1 的 Cold 最高，但 Warm/Overall 明显下降，所以它是机制诊断，不是直接部署方案。

H0-C0 也不等于 A0：它关闭的是 A2 checkpoint 的推理残差，但 \(W_0\)、HSTU 和其他共享参数已经经历过 A2 联合训练。

### 9.2 Cold-vs-All / Warm / Cold 候选组

为了区分“冷目标击败暖候选”和“冷物品内部排序”，又构造三种候选集：

- Cold-vs-All：完整全目录；
- Cold-vs-Warm：所有暖物品 + 当前冷正目标；
- Cold-vs-Cold：只在冷物品之间排序。

关键结果：

| Model | Candidate group | AUC | HR@100 | NDCG@100 | corrected MRR |
|---|---|---:|---:|---:|---:|
| A0 | Cold-vs-All | 0.674347 | 0.185321 | 0.047944 | 0.019836 |
| A2 | Cold-vs-All | **0.867796** | **0.236697** | **0.063826** | **0.027261** |
| A0 | Cold-vs-Warm | 0.673068 | 0.194495 | 0.050722 | 0.021134 |
| A2 | Cold-vs-Warm | **0.882538** | **0.297248** | **0.087670** | **0.044578** |
| A0 | Cold-vs-Cold | 0.686468 | 0.561468 | **0.174702** | **0.088905** |
| A2 | Cold-vs-Cold | **0.728078** | **0.596330** | 0.165969 | 0.071839 |

A2 在 Cold-vs-Warm 上提升最明显；在 Cold-vs-Cold 中虽然 AUC 和 HR@100 上升，NDCG/MRR 反而下降。这进一步说明 A2 主要改善跨暖冷校准，并没有全面解决冷物品内部头部排序。

---

## 10. 第五阶段：冷残差强度 alpha 消融

复用 A3/A4 checkpoint，只缩放严格冷候选的生成残差：

$$
e_i=W_0z_i+b+\alpha A\left(g_i\odot Bz_i\right),
\qquad i\in\mathcal I_c.
$$

暖历史和暖候选保持原 checkpoint 表示，所以 \(\alpha=0\) 不是 A0，而是“保留已经训练好的暖侧协同能力，只关闭冷残差”。

### 10.1 A3 alpha 扫描

| alpha | Cold AUC | Cold HR@100 | Cold NDCG@100 | Cold corrected MRR |
|---:|---:|---:|---:|---:|
| 0 | 0.760817 | **0.196330** | 0.045596 | 0.016108 |
| 0.25 | **0.768406** | 0.192661 | **0.046902** | **0.018135** |
| 0.50 | 0.759847 | 0.185321 | 0.042428 | 0.014025 |
| 0.75 | 0.740109 | 0.170642 | 0.038618 | 0.012447 |
| 1.00 | 0.711488 | 0.159633 | 0.034943 | 0.010348 |

A3 只适合保留少量冷残差，原始强度过大。

### 10.2 A4 alpha 扫描

| alpha | Cold AUC | Cold HR@100 | Cold NDCG@100 | Cold corrected MRR |
|---:|---:|---:|---:|---:|
| 0 | **0.886022** | **0.289908** | **0.076257** | **0.030525** |
| 0.25 | 0.866158 | 0.253211 | 0.065986 | 0.026250 |
| 0.50 | 0.835482 | 0.229358 | 0.058045 | 0.022206 |
| 0.75 | 0.796424 | 0.192661 | 0.049509 | 0.019732 |
| 1.00 | 0.752786 | 0.174312 | 0.044230 | 0.017623 |

A4 的 Cold 指标随 alpha 增大近似单调下降。\(\alpha=0\) 的完整结果为：

| Group | Samples | AUC | HR@10 | HR@100 | NDCG@10 | NDCG@100 | corrected MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold | 545 | 0.886022 | 0.067890 | 0.289908 | 0.032371 | 0.076257 | 0.030525 |
| Warm | 5,495 | 0.927315 | 0.233849 | 0.620018 | 0.128693 | 0.207219 | 0.112216 |
| Overall | 6,040 | 0.923589 | 0.218874 | 0.590232 | 0.120001 | 0.195402 | 0.104845 |

这说明 A4 学到的暖侧空间有价值，但 Student 生成的冷协同残差会把严格冷物品推向错误方向。相对 A0，A4 checkpoint + alpha=0 的 Cold HR@100 从 101/545 个命中提高到 158/545。

但是 alpha 是观察测试结果后扫描得到的，属于**机制诊断**，不能作为无偏最终模型结果。正确做法是先在独立 validation 上选择 alpha，然后只在 test 上评估一次。

---

## 11. 第六阶段：A4 的 beta 与 Frozen Teacher 消融

beta 只控制：

$$
\beta\mathcal L_{\mathrm{distill}},
$$

它影响 Student 模仿 Teacher 的强度。新增 beta=0、beta=0.5，以及冻结 Teacher 的 beta=0.5。

### 11.1 Cold

| Model | AUC | HR@10 | HR@100 | NDCG@10 | NDCG@100 | corrected MRR |
|---|---:|---:|---:|---:|---:|---:|
| A4 beta=0 | 0.553369 | 0.001835 | 0.045872 | 0.000530 | 0.008063 | 0.001267 |
| A4 beta=0.1 | 0.752786 | 0.027523 | 0.174312 | 0.015962 | 0.044230 | 0.017623 |
| A4 beta=0.5 | **0.763874** | **0.033028** | **0.192661** | **0.018022** | **0.048789** | **0.019008** |
| Frozen Teacher, beta=0.5 | 0.692714 | 0.018349 | 0.150459 | 0.009172 | 0.033597 | 0.010535 |

### 11.2 Warm / Overall

| Model | Warm AUC | Warm HR@100 | Warm NDCG@100 | Overall AUC | Overall HR@100 | Overall NDCG@100 |
|---|---:|---:|---:|---:|---:|---:|
| A4 beta=0 | 0.937958 | 0.654777 | 0.226744 | 0.903256 | 0.599834 | 0.207012 |
| A4 beta=0.1 | **0.938571** | **0.665696** | **0.230533** | **0.921807** | **0.621358** | **0.213722** |
| A4 beta=0.5 | 0.937278 | 0.654049 | 0.222329 | 0.921632 | 0.612417 | 0.206670 |
| Frozen Teacher, beta=0.5 | 0.938260 | 0.660055 | 0.228014 | 0.916104 | 0.614073 | 0.210471 |

beta=0 时，Student 和冷残差模长失去约束，Cold 指标明显崩溃；beta 提高到 0.5 后 Cold Top-K 改善，但牺牲少量 Overall。Frozen Teacher 的 Teacher-Student MSE 更低，却没有得到更好排序，说明“Teacher 移动太快”不是主要瓶颈。当前证据只能证明蒸馏在约束 Student，不足以证明已经把有效协同语义迁移到冷物品。

---

## 12. 当前得到的统一解释

综合所有结果，更符合证据的物品表示分工是：

$$
e_i=
\begin{cases}
e_i^{\mathrm{content}}+r_i^{\mathrm{collab}},&n_i>0,\\
e_i^{\mathrm{content}},&n_i=0.
\end{cases}
$$

其中：

$$
e_i^{\mathrm{content}}=W_0z_i+b.
$$

物理意义可以这样理解：

- RQ-VAE 内容向量提供“这个物品本身是什么”的可迁移基础；
- 共享 \(W_0\) 学习内容空间到推荐空间的全局映射；
- 暖物品残差记录“用户行为如何使用这个物品”的协同偏移；
- 冷物品没有行为证据，不应强行生成高强度协同偏移；
- A2 的 Cold 提升主要是整个暖候选空间被校准后，冷内容基础在全局排序中的相对位置改善。

因此不能把 A2 的高 Cold AUC 描述成“冷物品学到了 ID embedding”；冷物品没有 ID 参数。更准确的结论是：**暖侧协同建模与冷侧内容退化相结合，改善了全候选中的暖冷校准。**

---

## 13. 当前下一步：Free-ID Residual（已实现配置，结果待正式训练）

前面的低秩残差仍然要求残差由 \(z_i\) 和低秩系数共同构造。下一步进一步检验更直接的假设：暖物品增加一个自由 ID residual，冷物品只保留内容基础。

### 13.1 模型公式

$$
e_i^{\mathrm{base}}=W_0z_i+b.
$$

对暖物品：

$$
e_i=e_i^{\mathrm{base}}+\eta r_i^{\mathrm{ID}},
$$

对严格冷物品：

$$
e_i=e_i^{\mathrm{base}}.
$$

其中 \(r_i^{\mathrm{ID}}\in\mathbb R^{50}\) 只为正训练曝光暖物品建立，零初始化；冷物品没有残差表索引，无法误访问。当前缩放系数为 \(\eta=0.25\)，正则为：

$$
\mathcal L
=\mathcal L_{\mathrm{rec}}
+10^{-4}\frac{1}{|\mathcal I_w^{\mathrm{batch}}|}
\sum_i\left\|r_i^{\mathrm{ID}}\right\|_2^2.
$$

### 13.2 四组预注册对照

| 实验 | 方法 |
|---|---|
| B0 | 独立 train/validation/test 的 A0 内容基线 |
| B1 | 历史侧和候选侧都使用 warm Free-ID residual |
| B2 | 只有候选侧使用 warm Free-ID residual |
| B3 | 在 B1 上对 residual 使用 0.2 dropout |

这组实验同时修复 validation/test 重叠，并预先定义 checkpoint 选择分数：

$$
S_{\mathrm{val}}
=0.5\,\mathrm{ColdNDCG@100}
+0.25\,\mathrm{ColdHR@100}
+0.25\,\mathrm{OverallNDCG@100}.
$$

配置和测试已经实现，但本文件不虚构尚未产生的服务器正式结果。完成后应追加 B0–B3 的 Cold/Warm/Overall、多个随机种子以及均值和标准差。

---

## 14. 如何从头复现实验

### 14.1 准备 MovieLens-1M 与 HSTU 数据

```bash
cd Mini-HSTU-Reproduction
make env_smoke
make prepare_data data=ml-1m
```

### 14.2 训练 RQ-VAE 并导出 bridge

```bash
cd ../RQ-VAE-Recommender
make env_smoke
make train_rqvae config=configs/rqvae_ml1m.gin
make export_dense
```

bridge 至少包含：

```text
item_id_to_codes: [max_item_id + 1, 3]
codebook_vectors: [3, 1024, 64]
item_ids / metadata
```

建议同时执行质量审计：

```bash
make audit_rqvae \
  metrics_csv_path=out/rqvae/ml1m/metrics.csv \
  bridge_path=out/rqvae/ml1m/bridge_artifacts.pt \
  output_json=out/rqvae/ml1m/audit_summary.json
```

### 14.3 运行五路线历史对照

```bash
cd ../Mini-HSTU-Reproduction
make train experiment=ml-1m-hstu-cold logger=csv
make train experiment=ml-1m-hstu-dense-cold logger=csv
make train experiment=ml-1m-hstu-semantic-cold logger=csv
make train experiment=ml-1m-hstu-token-cold logger=csv
make train experiment=ml-1m-hstu-full-token-cold logger=csv
```

### 14.4 运行推荐的 warm-only negative 主基线

```bash
make train experiment=ml-1m-hstu-cold-warm-neg logger=csv
make train experiment=ml-1m-hstu-dense-cold-warm-neg logger=csv
make train experiment=ml-1m-hstu-semantic-cold-warm-neg logger=csv
```

### 14.5 运行 Personalized Route A

```bash
make train experiment=ml-1m-hstu-semantic-cold-a2-warm-id logger=csv
make train experiment=ml-1m-hstu-semantic-cold-a3-content logger=csv
make train experiment=ml-1m-hstu-semantic-cold-a4-teacher-student logger=csv
```

### 14.6 运行 Free-ID 下一步实验

```bash
make train experiment=ml-1m-hstu-semantic-cold-free-id-b0 logger=csv
make train experiment=ml-1m-hstu-semantic-cold-free-id-b1 logger=csv
make train experiment=ml-1m-hstu-semantic-cold-free-id-b2 logger=csv
make train experiment=ml-1m-hstu-semantic-cold-free-id-b3 logger=csv
```

每轮必须保留：resolved config、best checkpoint、metrics.csv、冷启动 split、负采样审计 JSON、GPU/环境记录和汇总 CSV。只有划分哈希、候选哈希与协议审计一致的实验才能横向比较。

---

## 15. 结果可信度与不能越过的边界

1. 当前历史正式训练只有 seed 42，尚未证明统计显著；建议至少补 3–5 个种子。
2. 历史 validation 与 test 目标重叠，因此不能把当前最高数值宣称为独立测试 SOTA。
3. A4 alpha=0 是测试集扫描后的诊断点，必须在新独立 validation 上预选后再测试。
4. 371 个冷物品中只有 338 个有 Semantic ID；545 条冷正目标全部来自这 338 个物品。其余 33 个物品仍参加候选竞争，但没有正目标样本。
5. 不同论文使用采样候选、Cold-only 候选或全目录候选，绝对 HR/NDCG 不可脱离协议直接比较。
6. AUC 反映全局两两排序，HR/NDCG/MRR 反映头部体验；任何模型结论都必须同时报告二者。
7. 当前严格协议是“零 HSTU 行为曝光”，不是“RQ-VAE 内容编码器也从未见过冷物品内容”；若采用后一种更强定义，需要重新对齐 RQ-VAE holdout。

在完成独立验证、多随机种子和置信区间之前，推荐对外使用下面的表述：

> 在固定 seed 的严格零训练曝光协议下，RQ-VAE Route A 明显改善了冷目标排序；暖物品个性化协同残差进一步提升了 Cold、Warm 与 Overall。机制消融表明，其 Cold 收益主要来自暖候选重排和暖冷分数校准，而内容生成的冷协同残差尚未实现稳定的 Top-K 迁移。

---

## 16. 关联文件

- [严格 Cold/Warm/Overall 总表与论文对照](STRICT_COLD_EXPERIMENTS_AND_BENCHMARKS.md)
- [第一轮完整复现实验报告](reproduction/FINAL_REPORT.md)
- [负样本采样修正报告](reproduction/warm_negative_ablation_completed/NEGATIVE_SAMPLING_ABLATION_REPORT.md)
- [Full-Token Route B 报告](reproduction/FULL_TOKEN_ROUTE_B_REPORT.md)
- [Personalized Route A 四组实验](experiments/personalized_route_a/FOUR_EXPERIMENTS_SUMMARY.md)
- [Personalized Route A 后续机制消融](experiments/personalized_route_a_followup/METHOD_AND_RESULTS_SUMMARY.md)
- [Personalized Route A 后续完整报告](experiments/personalized_route_a_followup/FOLLOWUP_ABLATION_REPORT.md)
