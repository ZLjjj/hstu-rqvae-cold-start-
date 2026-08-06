import argparse
import csv
import json
import os
from typing import Dict, Optional

import torch


def _last_eval_metrics(metrics_csv_path: str) -> Dict[str, float]:
    metrics: Dict[str, tuple[int, float]] = {}
    with open(metrics_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != "eval":
                continue
            metric = row.get("metric")
            if not metric:
                continue
            try:
                step = int(float(row.get("step", "0")))
                value = float(row.get("value", "nan"))
            except ValueError:
                continue
            prev = metrics.get(metric)
            if prev is None or step >= prev[0]:
                metrics[metric] = (step, value)
    return {k: v for k, (_, v) in metrics.items()}


def _load_cold_item_ids(path: Optional[str]) -> Optional[set]:
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"cold item file not found: {path}")
    if path.endswith(".json"):
        with open(path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "cold_item_ids" in payload:
            payload = payload["cold_item_ids"]
        return set(int(x) for x in payload)
    if path.endswith(".pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "cold_item_ids" in payload:
            payload = payload["cold_item_ids"]
        if torch.is_tensor(payload):
            payload = payload.tolist()
        return set(int(x) for x in payload)
    raise ValueError(f"unsupported cold item format: {path}")


def audit_rqvae_quality(
    metrics_csv_path: str,
    bridge_path: str,
    cold_items_path: Optional[str] = None,
    min_codebook_usage: float = 0.90,
    max_dead_code_ratio: float = 0.05,
    max_duplicate_ratio: float = 0.02,
    min_overall_coverage: float = 0.80,
    min_cold_coverage: float = 0.80,
) -> Dict:
    if not os.path.exists(metrics_csv_path):
        raise FileNotFoundError(f"metrics.csv not found: {metrics_csv_path}")
    if not os.path.exists(bridge_path):
        raise FileNotFoundError(f"bridge artifact not found: {bridge_path}")

    eval_metrics = _last_eval_metrics(metrics_csv_path)
    bridge = torch.load(bridge_path, map_location="cpu", weights_only=False)
    if not isinstance(bridge, dict):
        raise ValueError("bridge artifact must be a dict")

    item_ids = bridge.get("item_ids")
    item_to_codes = bridge.get("item_id_to_codes")
    if not torch.is_tensor(item_ids):
        if torch.is_tensor(item_to_codes):
            valid_mask = (item_to_codes >= 0).all(dim=1)
            item_ids = torch.nonzero(valid_mask, as_tuple=True)[0].long()
        else:
            raise ValueError("bridge must contain item_ids or item_id_to_codes")
    item_ids = item_ids.long()
    unique_item_ids = set(int(x) for x in item_ids.tolist())

    if torch.is_tensor(item_to_codes):
        max_item_id = int(item_to_codes.size(0) - 1)
        valid_codes = item_to_codes[(item_to_codes >= 0).all(dim=1)]
        if valid_codes.numel() > 0:
            unique_codes = torch.unique(valid_codes, dim=0)
            duplicate_ratio = float(
                (valid_codes.size(0) - unique_codes.size(0)) / max(1, valid_codes.size(0))
            )
        else:
            duplicate_ratio = 1.0
    else:
        max_item_id = max(unique_item_ids) if unique_item_ids else 0
        duplicate_ratio = float("nan")

    overall_coverage = len(unique_item_ids) / max(1, max_item_id)

    cold_item_ids = _load_cold_item_ids(cold_items_path)
    if cold_item_ids is not None and len(cold_item_ids) > 0:
        cold_coverage = len(unique_item_ids & cold_item_ids) / len(cold_item_ids)
    else:
        cold_coverage = None

    codebook_usage_values = [
        v for k, v in eval_metrics.items() if k.startswith("codebook_usage_")
    ]
    dead_code_values = [
        v for k, v in eval_metrics.items() if k.startswith("dead_code_ratio_")
    ]

    checks = {
        "codebook_usage": (
            len(codebook_usage_values) > 0
            and min(codebook_usage_values) >= min_codebook_usage
        ),
        "dead_code_ratio": (
            len(dead_code_values) > 0 and max(dead_code_values) <= max_dead_code_ratio
        ),
        "max_id_duplicates": (
            "max_id_duplicates" in eval_metrics
            and eval_metrics["max_id_duplicates"] <= max_duplicate_ratio
        ),
        "overall_coverage": overall_coverage >= min_overall_coverage,
        "collision_ratio": (
            duplicate_ratio == duplicate_ratio and duplicate_ratio <= max_duplicate_ratio
        ),
    }
    if cold_coverage is not None:
        checks["cold_coverage"] = cold_coverage >= min_cold_coverage

    status = "PASS" if all(checks.values()) else "WARN"
    return {
        "status": status,
        "checks": checks,
        "thresholds": {
            "min_codebook_usage": min_codebook_usage,
            "max_dead_code_ratio": max_dead_code_ratio,
            "max_duplicate_ratio": max_duplicate_ratio,
            "min_overall_coverage": min_overall_coverage,
            "min_cold_coverage": min_cold_coverage,
        },
        "eval_metrics": eval_metrics,
        "bridge_stats": {
            "num_item_ids": len(unique_item_ids),
            "max_item_id": max_item_id,
            "overall_coverage": overall_coverage,
            "cold_coverage": cold_coverage,
            "collision_ratio": duplicate_ratio,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit RQ-VAE training quality and bridge health.")
    parser.add_argument(
        "--metrics_csv_path",
        type=str,
        default="out/rqvae/ml1m/metrics.csv",
    )
    parser.add_argument(
        "--bridge_path",
        type=str,
        default="out/rqvae/ml1m/bridge_artifacts.pt",
    )
    parser.add_argument(
        "--cold_items_path",
        type=str,
        default=None,
    )
    parser.add_argument("--min_codebook_usage", type=float, default=0.90)
    parser.add_argument("--max_dead_code_ratio", type=float, default=0.05)
    parser.add_argument("--max_duplicate_ratio", type=float, default=0.02)
    parser.add_argument("--min_overall_coverage", type=float, default=0.80)
    parser.add_argument("--min_cold_coverage", type=float, default=0.80)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = audit_rqvae_quality(
        metrics_csv_path=args.metrics_csv_path,
        bridge_path=args.bridge_path,
        cold_items_path=args.cold_items_path,
        min_codebook_usage=args.min_codebook_usage,
        max_dead_code_ratio=args.max_dead_code_ratio,
        max_duplicate_ratio=args.max_duplicate_ratio,
        min_overall_coverage=args.min_overall_coverage,
        min_cold_coverage=args.min_cold_coverage,
    )

    print(f"[audit_rqvae_quality] status={summary['status']}")
    print("[audit_rqvae_quality] checks:")
    for k, v in summary["checks"].items():
        print(f"  - {k}: {v}")
    print("[audit_rqvae_quality] bridge_stats:")
    for k, v in summary["bridge_stats"].items():
        print(f"  - {k}: {v}")

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[audit_rqvae_quality] wrote json: {os.path.abspath(args.output_json)}")


if __name__ == "__main__":
    main()
