#!/usr/bin/env python3
"""Summarize completed warm-negative runs from real run artifacts."""

import argparse
import csv
import json
import re
from pathlib import Path

GROUPS = {
    "hstu_baseline_cold_warm_neg": "hstu_baseline_cold",
    "hstu_dense_cold_warm_neg": "hstu_dense_cold",
    "rqvae_hstu_route_a_warm_neg": "rqvae_hstu_route_a",
}
SPLITS = ("overall", "cold", "warm")
RANKS = (10, 50, 100, 200)
METRICS = (
    [f"{s}_auc" for s in SPLITS]
    + [f"{s}_hr@{k}" for s in SPLITS for k in RANKS]
    + [f"{s}_ndcg@{k}" for s in SPLITS for k in RANKS]
    + [f"{s}_mrr_corrected" for s in SPLITS]
)


def read_kv(path):
    result = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def final_test_values(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {}
    for row in rows:
        for key, raw in row.items():
            if key and key.startswith("test/") and raw not in (None, ""):
                try:
                    values[key[5:]] = float(raw)
                except ValueError:
                    pass
    return values


def extract(test, split, metric):
    prefix = "" if split == "overall" else f"{split}_"
    return test.get(prefix + metric)


def load_new(group_dir, experiment):
    if not (group_dir / "SUCCESS").exists():
        raise RuntimeError(f"Run is not marked successful: {group_dir}")
    info = read_kv(group_dir / "exit.txt")
    metrics_path = Path(info["metrics_csv"])
    checkpoint = Path(info["best_checkpoint"])
    train_log = group_dir / "train.log"
    if train_log.exists():
        clean_log = re.sub(r"\x1b\[[0-9;]*m", "", train_log.read_text(errors="replace"))
        matches = re.findall(r"Best ckpt path:\s*(\S+\.ckpt)", clean_log)
        if matches and Path(matches[-1]).is_file():
            checkpoint = Path(matches[-1])
    audit_path = Path(info["audit_json"])
    if not (metrics_path.is_file() and checkpoint.is_file() and audit_path.is_file()):
        raise RuntimeError(f"Missing run artifact for {experiment}")
    test = final_test_values(metrics_path)
    audit = json.loads(audit_path.read_text())
    if not audit.get("audit_passed"):
        raise RuntimeError(f"Audit failed for {experiment}")
    match = re.search(r"epoch[_=](\d+)", checkpoint.name)
    if match:
        best_epoch = int(match.group(1))
    else:
        with metrics_path.open(newline="") as handle:
            metric_rows = list(csv.DictReader(handle))
        validation_rows = [
            row for row in metric_rows
            if row.get("epoch") not in (None, "")
            and row.get("val/ndcg@100") not in (None, "")
        ]
        best_epoch = (
            int(float(max(validation_rows, key=lambda row: float(row["val/ndcg@100"]))["epoch"]))
            if validation_rows else ""
        )
    row = {
        "experiment": experiment,
        "protocol": "warm-negative-only",
        "best_epoch": best_epoch,
        "training_time_seconds": info.get("wall_time_seconds", ""),
        "peak_gpu_memory_mib": info.get("peak_gpu_memory_mib", ""),
        "negative_pool_size": audit.get("num_negative_pool_items", ""),
        "audit_passed": audit.get("audit_passed", False),
        "metrics_csv": str(metrics_path),
        "best_checkpoint": str(checkpoint),
        "audit_json": str(audit_path),
    }
    for split in SPLITS:
        row[f"{split}_auc"] = extract(test, split, "auc")
        row[f"{split}_mrr_corrected"] = extract(test, split, "mrr_corrected")
        for rank in RANKS:
            row[f"{split}_hr@{rank}"] = extract(test, split, f"hr@{rank}")
            row[f"{split}_ndcg@{rank}"] = extract(test, split, f"ndcg@{rank}")
    missing = [metric for metric in METRICS if row.get(metric) is None]
    if missing:
        raise RuntimeError(f"Missing final-test metrics for {experiment}: {missing}")
    return row


def normalize_name(value):
    text = value.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "hstu_baseline": "hstu_baseline_cold",
        "baseline": "hstu_baseline_cold",
        "dense": "hstu_dense_cold",
        "dense_minilm": "hstu_dense_cold",
        "route_a": "rqvae_hstu_route_a",
        "rq_vae_+_hstu_route_a": "rqvae_hstu_route_a",
    }
    return aliases.get(text, text)


def load_old(root):
    paths = [
        root / "reproduction/final_ablation_results.csv",
        root / "reproduction/final_ablation_results.md",
        root / "reproduction/FINAL_REPORT.md",
        root / "reproduction/full_token_fast_completed/SUMMARY.md",
    ]
    for path in paths[1:]:
        if path.exists():
            path.read_text(errors="replace")
    if not paths[0].exists():
        return {}
    with paths[0].open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    old = {}
    for source in source_rows:
        name_key = next((k for k in ("experiment", "method", "name") if source.get(k)), None)
        if name_key is None:
            continue
        row = {}
        old_test = {}
        source_metrics_path = Path(source.get("csv_path", ""))
        if source_metrics_path.is_file():
            old_test = final_test_values(source_metrics_path)
        for metric in METRICS:
            split, bare_metric = metric.split("_", 1)
            from_metrics_csv = extract(old_test, split, bare_metric)
            if from_metrics_csv is not None:
                row[metric] = from_metrics_csv
                continue
            candidates = [metric, f"test/{metric}"]
            if metric.startswith("overall_"):
                bare = metric[len("overall_"):]
                candidates += [bare, bare.upper(), f"test/{bare}"]
            else:
                split_name, bare = metric.split("_", 1)
                candidates += [
                    f"{split_name}_{bare.upper()}",
                    f"{split_name}_{bare.replace('auc', 'AUC').replace('hr', 'HR').replace('ndcg', 'NDCG').replace('mrr', 'MRR')}",
                ]
            for key in candidates:
                raw = source.get(key)
                if raw not in (None, ""):
                    try:
                        row[metric] = float(raw)
                    except ValueError:
                        pass
                    break
        old[normalize_name(source[name_key])] = row
    return old


def fmt(value):
    return "" if value in (None, "") else f"{float(value):.6f}"


def write_outputs(root, rows, old_rows):
    out = root / "reproduction"
    fields = [
        "experiment", "protocol", *METRICS, "best_epoch", "training_time_seconds",
        "peak_gpu_memory_mib", "negative_pool_size", "audit_passed", "metrics_csv",
        "best_checkpoint", "audit_json",
    ]
    with (out / "warm_negative_ablation_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Warm-only 训练负样本消融结果", "",
        "数值由脚本直接读取最终全量测试 metrics.csv。", "",
        "| 实验 | Overall AUC | Cold AUC | Warm AUC | Cold HR@10 | Cold HR@100 | Cold NDCG@100 | Cold corrected MRR | 最佳 epoch | 峰值显存 MiB | 审计 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {fmt(row['overall_auc'])} | {fmt(row['cold_auc'])} | {fmt(row['warm_auc'])} | "
            f"{fmt(row['cold_hr@10'])} | {fmt(row['cold_hr@100'])} | {fmt(row['cold_ndcg@100'])} | "
            f"{fmt(row['cold_mrr_corrected'])} | {row['best_epoch']} | {row['peak_gpu_memory_mib']} | {row['audit_passed']} |"
        )
    for row in rows:
        lines += ["", f"## {row['experiment']}", "",
            "| Split | AUC | HR@10 | HR@50 | HR@100 | HR@200 | NDCG@10 | NDCG@50 | NDCG@100 | NDCG@200 | corrected MRR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for split in SPLITS:
            hr = " | ".join(fmt(row[f"{split}_hr@{rank}"]) for rank in RANKS)
            ndcg = " | ".join(fmt(row[f"{split}_ndcg@{rank}"]) for rank in RANKS)
            lines.append(f"| {split} | {fmt(row[f'{split}_auc'])} | {hr} | {ndcg} | {fmt(row[f'{split}_mrr_corrected'])} |")
    (out / "warm_negative_ablation_results.md").write_text("\n".join(lines) + "\n")

    report = [
        "# 严格冷启动负样本采样消融报告", "", "## 协议", "",
        "唯一变量是训练负样本池改为 warm items；评估候选池仍含全部 cold items。", "",
        "## Old vs warm-negative-only", "",
        "| 实验 | 指标 | old | warm-negative-only | absolute_delta | relative_delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    key_metrics = ("overall_auc", "warm_auc", "cold_auc", "cold_hr@10", "cold_hr@100", "cold_ndcg@10", "cold_ndcg@100", "cold_mrr_corrected")
    for row in rows:
        old = old_rows.get(GROUPS[row["experiment"]], {})
        for metric in key_metrics:
            before, after = old.get(metric), row[metric]
            absolute = None if before is None else after - before
            relative = None if before in (None, 0) else absolute / abs(before)
            report.append(f"| {row['experiment']} | {metric} | {fmt(before)} | {fmt(after)} | {fmt(absolute)} | {fmt(relative)} |")
    report += [
        "", "## 分析问题", "",
        "1. Baseline cold AUC 是否恢复到接近随机水平（0.5）？",
        "2. Dense 与 Route A 的 cold AUC、HR、NDCG、corrected MRR 是否同步提高？",
        "3. Overall 与 warm 指标是否下降？",
        "4. Route A 相对 Dense 的冷启动优势是否仍存在？",
        "5. 旧协议是否因 cold negative exposure 低估冷启动能力？",
        "6. Cold AUC 改善是否转化为 cold HR@10/100 改善？", "",
        "结论必须同时依据 AUC、HR、NDCG 和 corrected MRR。", "",
        "## Route B 排除说明", "",
        "Route B 与 Full-Token Route B 使用语义码分类交叉熵，不使用 LocalNegativesSampler 的物品随机负采样。共享 code 使排除 cold code 会改变分类空间，属于另一项实验。",
    ]
    if not old_rows:
        report += ["", "> 未找到旧结果 CSV，old/delta 暂为空；同步后重跑脚本即可填充。"]
    (out / "NEGATIVE_SAMPLING_ABLATION_REPORT.md").write_text("\n".join(report) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    rows = [load_new(args.result_root / name, name) for name in GROUPS]
    write_outputs(args.root, rows, load_old(args.root))


if __name__ == "__main__":
    main()
