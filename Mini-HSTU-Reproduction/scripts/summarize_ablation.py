import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


METRIC_FIELDS: List[Tuple[str, str]] = [
    ("auc", "AUC"),
    ("hr@10", "HR@10"),
    ("hr@50", "HR@50"),
    ("hr@100", "HR@100"),
    ("ndcg@10", "NDCG@10"),
    ("ndcg@50", "NDCG@50"),
    ("ndcg@100", "NDCG@100"),
    ("mrr", "MRR"),
    ("cold_auc", "cold_AUC"),
    ("cold_hr@10", "cold_HR@10"),
    ("cold_hr@50", "cold_HR@50"),
    ("cold_hr@100", "cold_HR@100"),
    ("cold_ndcg@10", "cold_NDCG@10"),
    ("cold_ndcg@50", "cold_NDCG@50"),
    ("cold_ndcg@100", "cold_NDCG@100"),
    ("cold_mrr", "cold_MRR"),
    ("zero_hr@10", "zero_HR@10"),
    ("zero_hr@50", "zero_HR@50"),
    ("zero_hr@100", "zero_HR@100"),
    ("zero_ndcg@10", "zero_NDCG@10"),
    ("zero_ndcg@50", "zero_NDCG@50"),
    ("zero_ndcg@100", "zero_NDCG@100"),
    ("few_hr@10", "few_HR@10"),
    ("few_hr@50", "few_HR@50"),
    ("few_hr@100", "few_HR@100"),
    ("few_ndcg@10", "few_NDCG@10"),
    ("few_ndcg@50", "few_NDCG@50"),
    ("few_ndcg@100", "few_NDCG@100"),
]


@dataclass
class RunSummary:
    name: str
    run_dir: str
    csv_path: str
    config_path: str
    best_checkpoint: str
    git_commit: str
    metrics: Dict[str, Optional[float]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ML-1M strict cold-start ablation metrics."
    )
    parser.add_argument("--baseline_run", required=True, type=str)
    parser.add_argument("--dense_run", type=str, default=None)
    parser.add_argument("--route_a_run", required=True, type=str)
    parser.add_argument("--route_b_run", required=True, type=str)
    parser.add_argument(
        "--output_markdown",
        type=str,
        default="docs/ablation_results_ml1m.md",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="docs/ablation_results_ml1m.csv",
    )
    return parser.parse_args()


def _resolve_metrics_csv(run_dir: str) -> str:
    pattern = os.path.join(run_dir, "csv", "version_*", "metrics.csv")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"metrics.csv not found under run dir: {run_dir}")
    return candidates[-1]


def _try_float(value: str) -> Optional[float]:
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _last_metric(rows: List[Dict[str, str]], key: str) -> Optional[float]:
    test_key = f"test/{key}"
    val_key = f"val/{key}"
    last_test = None
    last_val = None
    for row in rows:
        if test_key in row:
            v = _try_float(row[test_key])
            if v is not None:
                last_test = v
        if val_key in row:
            v = _try_float(row[val_key])
            if v is not None:
                last_val = v
    return last_test if last_test is not None else last_val


def _resolve_best_checkpoint(run_dir: str) -> str:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return ""
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not ckpts:
        return ""
    non_last = [x for x in ckpts if not x.endswith("last.ckpt")]
    if non_last:
        return non_last[-1]
    return ckpts[-1]


def _resolve_git_commit(run_dir: str) -> str:
    commit_path = os.path.join(run_dir, "git_commit.txt")
    if os.path.exists(commit_path):
        with open(commit_path, "r") as f:
            return f.read().strip()
    return ""


def _summarize_run(name: str, run_dir: str) -> RunSummary:
    run_dir = os.path.abspath(run_dir)
    csv_path = _resolve_metrics_csv(run_dir)
    rows = _read_csv_rows(csv_path)
    metrics = {field: _last_metric(rows, field) for field, _ in METRIC_FIELDS}
    return RunSummary(
        name=name,
        run_dir=run_dir,
        csv_path=csv_path,
        config_path=os.path.join(run_dir, ".hydra", "config.yaml"),
        best_checkpoint=_resolve_best_checkpoint(run_dir),
        git_commit=_resolve_git_commit(run_dir),
        metrics=metrics,
    )


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.6f}"


def _write_csv(path: str, summaries: List[RunSummary]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    headers = [
        "experiment",
        "run_dir",
        "csv_path",
        "config_path",
        "best_checkpoint",
        "git_commit",
        *[name for _, name in METRIC_FIELDS],
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for item in summaries:
            writer.writerow(
                [
                    item.name,
                    item.run_dir,
                    item.csv_path,
                    item.config_path,
                    item.best_checkpoint,
                    item.git_commit,
                    *[_fmt(item.metrics[field]) for field, _ in METRIC_FIELDS],
                ]
            )


def _write_markdown(path: str, summaries: List[RunSummary]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write("# ML-1M Strict Cold-start Ablation Results\n\n")
        f.write("| Experiment | " + " | ".join(name for _, name in METRIC_FIELDS) + " |\n")
        f.write("|---|" + "|".join(["---:"] * len(METRIC_FIELDS)) + "|\n")
        for item in summaries:
            values = " | ".join(_fmt(item.metrics[field]) for field, _ in METRIC_FIELDS)
            f.write(f"| {item.name} | {values} |\n")
        f.write("\n")
        f.write("## Run Artifacts\n\n")
        for item in summaries:
            f.write(f"- `{item.name}`\n")
            f.write(f"  - run_dir: `{item.run_dir}`\n")
            f.write(f"  - csv: `{item.csv_path}`\n")
            f.write(f"  - config: `{item.config_path}`\n")
            f.write(f"  - best_checkpoint: `{item.best_checkpoint}`\n")
            f.write(f"  - git_commit: `{item.git_commit}`\n")


def main() -> None:
    args = _parse_args()
    summaries = [_summarize_run("hstu_baseline_cold", args.baseline_run)]
    if args.dense_run:
        summaries.append(_summarize_run("hstu_dense_cold", args.dense_run))
    summaries.extend(
        [
            _summarize_run("rqvae_hstu_route_a", args.route_a_run),
            _summarize_run("rqvae_hstu_route_b", args.route_b_run),
        ]
    )

    _write_csv(os.path.abspath(args.output_csv), summaries)
    _write_markdown(os.path.abspath(args.output_markdown), summaries)

    print(f"[summarize_ablation] wrote csv: {os.path.abspath(args.output_csv)}")
    print(f"[summarize_ablation] wrote markdown: {os.path.abspath(args.output_markdown)}")


if __name__ == "__main__":
    main()
