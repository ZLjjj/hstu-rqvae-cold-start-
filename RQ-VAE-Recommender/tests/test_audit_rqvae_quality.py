import csv
import json

import torch

from scripts.audit_rqvae_quality import audit_rqvae_quality


def _write_metrics_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "metric", "step", "value"])
        writer.writeheader()
        writer.writerows(rows)


def test_audit_rqvae_quality_pass(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    bridge_path = tmp_path / "bridge.pt"
    cold_items_path = tmp_path / "cold_items.json"

    _write_metrics_csv(
        metrics_path,
        [
            {"split": "eval", "metric": "codebook_usage_0", "step": "10", "value": "0.95"},
            {"split": "eval", "metric": "codebook_usage_1", "step": "10", "value": "0.93"},
            {"split": "eval", "metric": "dead_code_ratio_0", "step": "10", "value": "0.01"},
            {"split": "eval", "metric": "dead_code_ratio_1", "step": "10", "value": "0.02"},
            {"split": "eval", "metric": "max_id_duplicates", "step": "10", "value": "0.0"},
        ],
    )

    item_id_to_codes = torch.full((11, 2), -1, dtype=torch.long)
    item_id_to_codes[1] = torch.tensor([0, 1])
    item_id_to_codes[2] = torch.tensor([1, 2])
    item_id_to_codes[3] = torch.tensor([2, 3])
    torch.save(
        {
            "item_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "item_id_to_codes": item_id_to_codes,
        },
        bridge_path,
    )

    cold_items_path.write_text(json.dumps({"cold_item_ids": [1, 3]}), encoding="utf-8")

    summary = audit_rqvae_quality(
        metrics_csv_path=str(metrics_path),
        bridge_path=str(bridge_path),
        cold_items_path=str(cold_items_path),
        min_codebook_usage=0.90,
        max_dead_code_ratio=0.05,
        max_duplicate_ratio=0.02,
        min_overall_coverage=0.20,
        min_cold_coverage=0.80,
    )

    assert summary["status"] == "PASS"
    assert summary["checks"]["codebook_usage"]
    assert summary["checks"]["dead_code_ratio"]
    assert summary["checks"]["overall_coverage"]
    assert summary["checks"]["cold_coverage"]
