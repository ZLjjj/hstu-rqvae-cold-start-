"""Summarize a Full-Token Route B CSV run into the prepared Markdown artifacts."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional


PRIOR_ROWS = [
    ("HSTU baseline", "0.212941", "0.124940", "0.000000", "0.000000", "0.000000"),
    ("Dense baseline", "0.157630", "0.087339", "0.020183", "0.003605", "0.000596"),
    ("Route A", "0.140380", "0.070762", "0.102752", "0.021144", "0.005551"),
    ("旧简化 Route B", "0.080531", "0.041626", "0.001835", "0.000530", "0.000183"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--prior_report", default="../reproduction/FINAL_REPORT.md")
    parser.add_argument("--resource_json", default=None)
    parser.add_argument(
        "--output_results", default="../reproduction/full_token_route_b_results.md"
    )
    parser.add_argument(
        "--output_report", default="../reproduction/FULL_TOKEN_ROUTE_B_REPORT.md"
    )
    return parser.parse_args()


def metrics_csv(run_dir: Path) -> Path:
    candidates = sorted(glob.glob(str(run_dir / "csv" / "version_*" / "metrics.csv")))
    if not candidates:
        candidates = sorted(
            glob.glob(str(run_dir / "**" / "metrics.csv"), recursive=True)
        )
    if not candidates:
        raise FileNotFoundError(f"metrics.csv not found below {run_dir}")
    return Path(candidates[-1])


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def last(rows: List[Dict[str, str]], *keys: str) -> Optional[float]:
    found = None
    for row in rows:
        for key in keys:
            value = number(row.get(key))
            if value is not None:
                found = value
                break
    return found


def best_epoch(rows: List[Dict[str, str]]) -> Optional[int]:
    candidates = []
    for row in rows:
        score = number(row.get("val/ndcg@100"))
        epoch = number(row.get("epoch"))
        if score is not None and epoch is not None:
            candidates.append((score, int(epoch)))
    return max(candidates)[1] if candidates else None


def fmt(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.6f}"


def metric(rows, group: str, name: str, ranking: str = "") -> Optional[float]:
    prefix = f"test/{ranking}/" if ranking else "test/"
    val_prefix = f"val/{ranking}/" if ranking else "val/"
    key = f"{group}_{name}" if group else name
    return last(rows, prefix + key, val_prefix + key)


def make_results(rows: List[Dict[str, str]], csv_path: Path, run_dir: Path) -> str:
    overall_ndcg = metric(rows, "", "ndcg@100")
    overall_mrr = metric(rows, "", "mrr_corrected")
    cold_hr = metric(rows, "cold", "hr@100")
    cold_ndcg = metric(rows, "cold", "ndcg@100")
    cold_mrr = metric(rows, "cold", "mrr_corrected")
    lines = [
        "# Full-Token Route B 结果表",
        "",
        f"来源运行目录：`{run_dir}`  ",
        f"来源 CSV：`{csv_path}`  ",
        f"最佳验证 epoch：`{best_epoch(rows) if best_epoch(rows) is not None else '未找到'}`",
        "",
        "## 核心对照",
        "",
        "| 实验 | Overall NDCG@100 | Overall corrected MRR | Cold HR@100 | Cold NDCG@100 | Cold corrected MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in PRIOR_ROWS)
    lines.append(
        "| 新 Full-Token Route B | "
        + " | ".join(
            map(fmt, [overall_ndcg, overall_mrr, cold_hr, cold_ndcg, cold_mrr])
        )
        + " |"
    )
    lines.extend(["", "## Full-Token 完整指标", ""])
    columns = [
        "auc",
        "hr@10",
        "hr@50",
        "hr@100",
        "hr@200",
        "ndcg@10",
        "ndcg@50",
        "ndcg@100",
        "mrr_corrected",
    ]
    lines.extend(
        [
            "| 分组 | " + " | ".join(columns) + " |",
            "|---|" + "|".join(["---:"] * len(columns)) + "|",
        ]
    )
    for label, group in (("Overall", ""), ("Cold", "cold"), ("Warm", "warm")):
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(fmt(metric(rows, group, col)) for col in columns)
            + " |"
        )

    lines.extend(["", "## 三种排名口径", ""])
    ranking_columns = ["hr@100", "cold_hr@100", "ndcg@100", "cold_ndcg@100", "auc"]
    lines.extend(
        [
            "| 口径 | " + " | ".join(ranking_columns) + " |",
            "|---|" + "|".join(["---:"] * len(ranking_columns)) + "|",
        ]
    )
    for label, prefix in (
        ("Beam-only", "beam_only"),
        ("Beam + Exact 补齐", ""),
        ("Exact 全局排序", "exact_global"),
    ):
        values = []
        for key in ranking_columns:
            if key.startswith("cold_"):
                values.append(metric(rows, "cold", key[5:], prefix))
            else:
                values.append(metric(rows, "", key, prefix))
        lines.append("| " + label + " | " + " | ".join(map(fmt, values)) + " |")

    lines.extend(["", "## Token、搜索与资源诊断", "", "| 指标 | 结果 |", "|---|---:|"])
    diagnostics = [
        ("level-0 loss", last(rows, "train/loss_level_0_epoch", "train/loss_level_0")),
        (
            "level-0 accuracy",
            last(rows, "train/accuracy_level_0_epoch", "train/accuracy_level_0"),
        ),
        ("level-1 loss", last(rows, "train/loss_level_1_epoch", "train/loss_level_1")),
        (
            "level-1 accuracy",
            last(rows, "train/accuracy_level_1_epoch", "train/accuracy_level_1"),
        ),
        ("level-2 loss", last(rows, "train/loss_level_2_epoch", "train/loss_level_2")),
        (
            "level-2 accuracy",
            last(rows, "train/accuracy_level_2_epoch", "train/accuracy_level_2"),
        ),
        (
            "Beam 合法路径/用户",
            last(rows, "test/legal_complete_paths", "val/legal_complete_paths"),
        ),
        (
            "Beam 唯一物品/用户",
            last(rows, "test/beam_only_candidates", "val/beam_only_candidates"),
        ),
        ("Exact 补齐数/用户", last(rows, "test/exact_filled", "val/exact_filled")),
        ("实际 beam", last(rows, "test/actual_beam_size", "val/actual_beam_size")),
        (
            "缺码历史物品/用户",
            last(rows, "test/missing_history_items", "val/missing_history_items"),
        ),
        (
            "Beam 推理秒/用户",
            last(rows, "test/beam_inference_seconds", "val/beam_inference_seconds"),
        ),
        (
            "Exact 补齐秒/用户",
            last(rows, "test/exact_fill_seconds", "val/exact_fill_seconds"),
        ),
        (
            "Exact 全局秒/用户",
            last(rows, "test/exact_global_seconds", "val/exact_global_seconds"),
        ),
        (
            "冲突路径",
            last(rows, "test/semantic_collision_paths", "val/semantic_collision_paths"),
        ),
        (
            "受冲突影响物品",
            last(rows, "test/semantic_collision_items", "val/semantic_collision_items"),
        ),
        (
            "可训练参数量",
            last(rows, "test/trainable_parameters", "val/trainable_parameters"),
        ),
        ("峰值显存 MB", last(rows, "test/peak_memory_mb", "val/peak_memory_mb")),
    ]
    lines.extend(f"| {label} | {fmt(value)} |" for label, value in diagnostics)
    return "\n".join(lines) + "\n"


def update_report(path: Path, result_summary: str, resource: Dict[str, object]) -> None:
    text = path.read_text() if path.exists() else "# Full-Token Route B 实验报告\n"
    start = "<!-- SERVER_RESULTS:START -->"
    end = "<!-- SERVER_RESULTS:END -->"
    resource_lines = (
        "\n".join(f"- {key}: `{value}`" for key, value in resource.items())
        or "- 资源采样：未提供"
    )
    block = (
        start
        + "\n\n## 服务器自动汇总\n\n"
        + result_summary
        + "\n### 资源与协议补充\n\n"
        + resource_lines
        + "\n\n"
        + end
    )
    if start in text and end in text:
        text = text[: text.index(start)] + block + text[text.index(end) + len(end) :]
    else:
        text += "\n\n" + block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    csv_path = metrics_csv(run_dir)
    rows = read_rows(csv_path)
    prior = Path(args.prior_report)
    if not prior.exists():
        raise FileNotFoundError(f"prior four-experiment report not found: {prior}")
    results = make_results(rows, csv_path, run_dir)
    output_results = Path(args.output_results)
    output_results.parent.mkdir(parents=True, exist_ok=True)
    output_results.write_text(results)
    resource = {}
    if args.resource_json:
        with Path(args.resource_json).open() as handle:
            resource = json.load(handle)
    update_report(Path(args.output_report), results, resource)
    print(f"wrote {output_results.resolve()}")
    print(f"updated {Path(args.output_report).resolve()}")


if __name__ == "__main__":
    main()
