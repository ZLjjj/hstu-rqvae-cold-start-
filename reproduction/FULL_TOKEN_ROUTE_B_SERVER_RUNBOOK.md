# Full-Token Route B 服务器运行手册

本文只用于新实验 `ml-1m-hstu-full-token-cold`。旧 Route A、旧 Route B 和已有四组结果均不得重跑或覆盖。

## 1. 新增文件

- `Mini-HSTU-Reproduction/src/generative_recommenders_pl/models/full_token_retrieval.py`
- `Mini-HSTU-Reproduction/configs/experiment/ml-1m-hstu-full-token-cold.yaml`
- `Mini-HSTU-Reproduction/tests/test_full_token_retrieval.py`
- `Mini-HSTU-Reproduction/scripts/full_token_smoke.py`
- `Mini-HSTU-Reproduction/scripts/summarize_full_token_route_b.py`
- `reproduction/FULL_TOKEN_ROUTE_B_SERVER_RUNBOOK.md`
- `reproduction/FULL_TOKEN_ROUTE_B_REPORT.md`
- `reproduction/full_token_route_b_results.md`
- `reproduction/run_full_token_route_b_server.sh`
- `reproduction/full_token_gpu_sampler.sh`

## 2. 修改文件

- `Mini-HSTU-Reproduction/src/generative_recommenders_pl/models/metrics/retrieval.py`：只新增 warm 指标，旧字段和算法保持不变。
- `Mini-HSTU-Reproduction/makefile`：新增 smoke 和 Full-Token 汇总入口。

旧 `token_retrieval.py` 和旧实验 YAML 未修改。若同步包包含工作区内其他既有改动，应先用 `git diff --stat` 审核，不能用服务器旧文件反向覆盖。

## 3. 准确同步范围

服务器根目录约定为：

```bash
SERVER_ROOT=/root/autodl-tmp/semantic-id-rec-repro/repo
```

同步第 1 节的十个新增文件和第 2 节的两个修改文件；其中五个运行/报告文件同步至 `$SERVER_ROOT/reproduction/`，其余按原相对路径同步。不要同步本地数据、checkpoint、Hydra outputs 或旧报告归档。

## 4. 依赖

没有新增 Python 依赖。继续使用项目锁定的 PyTorch、Lightning、Hydra、TorchMetrics 和 pytest 环境。

## 5. 项目目录检查

```bash
cd /root/autodl-tmp/semantic-id-rec-repro/repo
test -f Mini-HSTU-Reproduction/src/generative_recommenders_pl/models/full_token_retrieval.py
test -f Mini-HSTU-Reproduction/configs/experiment/ml-1m-hstu-full-token-cold.yaml
git status --short
```

记录当前 commit 和工作区差异到实验日志，不清理既有结果。

## 6. MovieLens 数据检查

```bash
cd /root/autodl-tmp/semantic-id-rec-repro/repo/Mini-HSTU-Reproduction
find data tmp -maxdepth 3 -type f 2>/dev/null | sort | head -50
```

应复用前四组实验的 MovieLens-1M 处理产物，不重新下载或重新划分。核对最大物品 ID 3952、有效物品数 3706、最大历史长度 200。

## 7. Bridge artifact 检查

```bash
cd /root/autodl-tmp/semantic-id-rec-repro/repo/Mini-HSTU-Reproduction
test -f ../RQ-VAE-Recommender/out/rqvae/ml1m/bridge_artifacts.pt
python - <<'PY'
import torch
p = torch.load('../RQ-VAE-Recommender/out/rqvae/ml1m/bridge_artifacts.pt', map_location='cpu', weights_only=False)
print('item_id_to_codes', tuple(p['item_id_to_codes'].shape))
print('codebook_vectors', tuple(p['codebook_vectors'].shape))
print('metadata', p.get('metadata', {}))
PY
```

期望码本形状 `[3, 1024, 64]`，不得换 bridge。

## 8. 冷物品划分检查

复用 seed 42 的持久化 cold split。启动日志必须记录：冷物品 371 个；训练历史/训练目标中的冷物品出现次数均为 0；完整 Semantic ID 的冷物品数、缺码数、RQ-VAE 训练物品与 HSTU 冷物品重叠数；验证/测试目标文件哈希与旧实验一致。任何一项不一致都先停止，不得静默重划分。

## 9. GPU 与显存建议

推荐单卡 40–48 GB；现有 RTX 4090 D 24 GB 也应先按本手册 smoke，再视峰值下调物理 batch 和 `exact_prefix_batch_size`。Token 长度从约 200 增至最多 604，注意力矩阵约增至 9 倍。默认物理 batch 32、累积 4 次，保持有效 batch 128。记录 GPU 型号、驱动、CUDA、峰值显存和可训练参数量。

## 10. 快速 smoke test

先做只解析、不训练的配置检查：

```bash
cd /root/autodl-tmp/semantic-id-rec-repro/repo/Mini-HSTU-Reproduction
PYTHONPATH=src /root/autodl-tmp/semantic-id-rec-repro/env/bin/python \
  src/generative_recommenders_pl/scripts/train.py \
  experiment=ml-1m-hstu-full-token-cold seed=42 logger=csv \
  --cfg job --resolve
```

确认新模型类、`max_token_sequence_len=604`、HSTU `max_sequence_len=604`、`max_output_len=0`。

先运行无数据的确定性 CPU/GPU smoke：

```bash
cd /root/autodl-tmp/semantic-id-rec-repro/repo/Mini-HSTU-Reproduction
PYTHONPATH=src /root/autodl-tmp/semantic-id-rec-repro/env/bin/python \
  scripts/full_token_smoke.py
```

再运行真实数据的 2-batch、1-epoch smoke：

```bash
PYTHONPATH=src /root/autodl-tmp/semantic-id-rec-repro/env/bin/python \
  src/generative_recommenders_pl/scripts/train.py \
  experiment=ml-1m-hstu-full-token-cold seed=42 logger=csv \
  trainer.accelerator=gpu trainer.devices=1 \
  trainer.min_epochs=1 trainer.max_epochs=1 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=1 \
  data.batch_size=2 trainer.accumulate_grad_batches=1 \
  model.beam_size=2 model.max_beam_size=4 model.target_unique_items=4 \
  model.compute_full_auc=false model.exact_fill=false \
  tags='[full-token,server-smoke]'
```

检查三级 loss/accuracy、Mask、无 NaN、验证可完成以及峰值显存，再运行完整实验。

## 11. 完整训练命令

```bash
cd /root/autodl-tmp/semantic-id-rec-repro/repo/Mini-HSTU-Reproduction
PYTHONPATH=src /root/autodl-tmp/semantic-id-rec-repro/env/bin/python \
  src/generative_recommenders_pl/scripts/train.py \
  experiment=ml-1m-hstu-full-token-cold seed=42 logger=csv \
  trainer.accelerator=gpu trainer.devices=1
```

配置必须保留 `experiment=ml-1m-hstu-full-token-cold seed=42 logger=csv`。

## 12. 后台运行

```bash
cd /root/autodl-tmp/semantic-id-rec-repro/repo
chmod +x reproduction/run_full_token_route_b_server.sh \
  reproduction/full_token_gpu_sampler.sh
nohup reproduction/run_full_token_route_b_server.sh </dev/null \
  > reproduction/logs/full_token_route_b_launcher.log 2>&1 &
echo $! > reproduction/full_token_route_b_train.pid
nohup reproduction/full_token_gpu_sampler.sh \
  reproduction/full_token_route_b_train.pid \
  reproduction/full_token_route_b_gpu_usage.csv </dev/null \
  > reproduction/logs/full_token_route_b_gpu_sampler.log 2>&1 &
echo $! > reproduction/full_token_route_b_gpu_sampler.pid
```

## 13. 日志查看

```bash
tail -n 120 -f /root/autodl-tmp/semantic-id-rec-repro/repo/reproduction/logs/full_token_route_b_train.log
```

每轮关注总 loss、三级 loss/accuracy、val NDCG@100、overall/cold/warm 指标、Beam 诊断、学习率和异常回溯。

为避免每个 epoch 重复极昂贵的全候选 HSTU Exact，主配置在 validation 使用 Beam-only（`compute_full_auc_on_validation=false, exact_fill_on_validation=false`）；训练结束的最佳-checkpoint test 会恢复 Exact 全局排序和 Exact 补齐，并同时输出 Beam-only、Beam+Exact、Exact-global 三套指标。

## 14. 进程与资源检查

```bash
ps -fp "$(cat /root/autodl-tmp/semantic-id-rec-repro/repo/reproduction/full_token_route_b_train.pid)"
nvidia-smi
free -h
```

SSH 断开不等于训练中断，以 PID、日志更新时间和 GPU 进程为准。

## 15. 中断恢复

找到原运行目录的 `last.ckpt`，不要从头重启：

```bash
PYTHONPATH=src /root/autodl-tmp/semantic-id-rec-repro/env/bin/python \
  src/generative_recommenders_pl/scripts/train.py \
  experiment=ml-1m-hstu-full-token-cold seed=42 logger=csv \
  trainer.accelerator=gpu trainer.devices=1 \
  ckpt_path=/ABSOLUTE/RUN/DIR/checkpoints/last.ckpt \
  tags='[full-token,route-b,cold-resume]'
```

恢复前复制原日志并记录中断时间、原因、原 PID、checkpoint epoch 和新运行目录。

## 16. 最佳 checkpoint 测试

训练入口默认在结束时测试最佳 checkpoint。若需单独复核：

```bash
PYTHONPATH=src /root/autodl-tmp/semantic-id-rec-repro/env/bin/python \
  src/generative_recommenders_pl/scripts/eval.py \
  experiment=ml-1m-hstu-full-token-cold seed=42 logger=csv \
  trainer.accelerator=gpu trainer.devices=1 \
  ckpt_path=/ABSOLUTE/RUN/DIR/checkpoints/BEST.ckpt
```

不得用 `last.ckpt` 冒充最佳 checkpoint。

## 17. 结果汇总

```bash
PYTHONPATH=src /root/autodl-tmp/semantic-id-rec-repro/env/bin/python \
  scripts/summarize_full_token_route_b.py \
  --run_dir /ABSOLUTE/FULL_TOKEN/RUN/DIR \
  --prior_report ../reproduction/FINAL_REPORT.md \
  --output_results ../reproduction/full_token_route_b_results.md \
  --output_report ../reproduction/FULL_TOKEN_ROUTE_B_REPORT.md
```

汇总后人工核对 CSV 最后一组 test 指标、最佳 epoch 和 checkpoint 路径。

## 18. 显存不足时允许调整

按顺序调整：物理 `data.batch_size`（同时反向调整 `trainer.accumulate_grad_batches`，保持有效 batch 128）、`exact_prefix_batch_size`、验证阶段用户分块。Smoke 可临时减小 Beam 和关闭 exact；正式测试必须恢复 Beam 32→自适应最大 256、目标 200，并运行全局 Exact 指标。不得缩短正式历史长度。

## 19. 不得修改的可比性参数

不得修改 seed 42、371 个冷物品及其文件、训练/验证/测试划分、bridge、最大历史物品数 200、三级/1024 码本、学习率与调度/early stopping、评价实现、候选全集、过去物品过滤、BOS loss 默认关闭、缺码策略、Trie 内容可见性。若因资源必须改变有效 batch，需另标为非主实验。

## 20. 结果回传

回传小文件：最佳/最后 metrics CSV、`.hydra/config.yaml`、checkpoint 清单和元信息（不必默认传大 checkpoint）、训练日志、资源采样、协议审计、两个生成的 Markdown 报告。统一放到 `reproduction/full_token_route_b_server_results/` 后打包。

## 21. 常见错误

- bridge 不存在/形状不符：修正相对路径，禁止自动生成新 bridge。
- Token 长度超过 604：检查目标是否被重复追加或历史长度是否超过 200。
- HSTU mask/relative bias 维度错误：确认 sequence encoder 的 `max_sequence_len=604, max_output_len=0`。
- CUDA OOM：按第 18 节缩物理 batch 和 exact 前缀批量，不改协议。
- Beam 候选少：查看合法路径、唯一物品、冲突和实际 beam；不要隐藏 Exact 补齐数。
- cold AUC 缺失：确认正式测试 `compute_full_auc=true`，它不依赖目标是否进入 Beam。
- loss 为 NaN：分别检查三级有效样本数、缺码计数、码范围和时间戳；不要用 PAD 代替缺码。
- checkpoint 无法加载：核对新模型类路径、配置快照和同步 commit，不覆盖旧 Route B 类。
