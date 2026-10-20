#!/usr/bin/env python3
"""Select temporal-view calls using a threshold or fixed uncertainty budget."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from experiments.analyze_uncertainty_temporal_escalation import choice_margin
from experiments.evaluate_qwen3vl_recent4_full import atomic_write, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recent", type=Path, required=True)
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--threshold", type=Path)
    policy.add_argument(
        "--quantile",
        type=float,
        help="Acquire the lowest-margin fraction without reading cohort labels.",
    )
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-uids", type=Path, required=True)
    parser.add_argument("--output-samples", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    return parser.parse_args()


def load_frozen_threshold(path: Path, model_manifest_sha256: str | None) -> float:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "source_disjoint_internal_threshold_calibration"
        or value.get("eligible_as_frozen_benchmark_controller_config") is not True
        or value.get("benchmark") != "internal_policy_holdout"
        or value.get("model_manifest_sha256") != model_manifest_sha256
    ):
        raise ValueError("threshold is not an eligible source-disjoint calibration")
    threshold = float(value["policy"]["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("invalid calibrated threshold")
    return threshold


def select_rows(
    recent: dict[str, Any], *, threshold: float
) -> tuple[list[str], list[dict[str, Any]]]:
    selected = []
    audit_rows = []
    for row in recent.get("rows") or []:
        uid = str(row.get("uid") or "")
        if not uid:
            raise ValueError("recent result has a row without uid")
        margin = choice_margin(row)
        escalated = margin <= threshold
        if escalated:
            selected.append(uid)
        audit_rows.append(
            {
                "uid": uid,
                "margin": margin,
                "threshold": threshold,
                "escalated": escalated,
            }
        )
    if len(audit_rows) != int(recent.get("num_samples", -1)):
        raise ValueError("recent result sample count diverged")
    if int(recent.get("num_errors", -1)) != 0:
        raise ValueError("recent result contains inference errors")
    if len({row["uid"] for row in audit_rows}) != len(audit_rows):
        raise ValueError("recent result contains duplicate uids")
    return selected, audit_rows


def select_rows_by_quantile(
    recent: dict[str, Any], *, quantile: float
) -> tuple[list[str], list[dict[str, Any]], float]:
    """Select an exact bottom-margin budget with deterministic UID tie-breaking."""

    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    rows = list(recent.get("rows") or [])
    if len(rows) != int(recent.get("num_samples", -1)):
        raise ValueError("recent result sample count diverged")
    if int(recent.get("num_errors", -1)) != 0:
        raise ValueError("recent result contains inference errors")
    values = []
    for row in rows:
        uid = str(row.get("uid") or "")
        if not uid:
            raise ValueError("recent result has a row without uid")
        values.append((choice_margin(row), uid))
    if len({uid for _, uid in values}) != len(values):
        raise ValueError("recent result contains duplicate uids")
    budget = max(1, math.ceil(len(values) * quantile))
    ranked = sorted(values, key=lambda item: (item[0], item[1]))
    selected_set = {uid for _, uid in ranked[:budget]}
    cutoff = float(ranked[budget - 1][0])
    selected = [str(row["uid"]) for row in rows if str(row["uid"]) in selected_set]
    rank_by_uid = {uid: rank for rank, (_, uid) in enumerate(ranked, 1)}
    audit_rows = [
        {
            "uid": str(row["uid"]),
            "margin": choice_margin(row),
            "uncertainty_rank": rank_by_uid[str(row["uid"])],
            "quantile": quantile,
            "escalated": str(row["uid"]) in selected_set,
        }
        for row in rows
    ]
    return selected, audit_rows, cutoff


def main() -> None:
    args = parse_args()
    recent = json.loads(args.recent.read_text(encoding="utf-8"))
    threshold = None
    quantile = None
    cutoff = None
    if args.threshold is not None:
        threshold = load_frozen_threshold(
            args.threshold, recent.get("model_manifest_sha256")
        )
        selected, audit_rows = select_rows(recent, threshold=threshold)
    else:
        quantile = float(args.quantile)
        selected, audit_rows, cutoff = select_rows_by_quantile(
            recent, quantile=quantile
        )
    selected_set = set(selected)
    sample_rows = [
        json.loads(line)
        for line in args.samples.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ordered_uids = [str(row.get("uid") or "") for row in sample_rows]
    recent_uids = [str(row["uid"]) for row in recent["rows"]]
    if ordered_uids != recent_uids:
        raise ValueError("recent result does not follow the frozen sample order")
    selected_samples = [row for row in sample_rows if str(row["uid"]) in selected_set]
    if [str(row["uid"]) for row in selected_samples] != selected:
        raise ValueError("selected UID ordering diverged")
    args.output_uids.parent.mkdir(parents=True, exist_ok=True)
    args.output_uids.write_text("\n".join(selected) + "\n", encoding="utf-8")
    args.output_samples.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in selected_samples
        ),
        encoding="utf-8",
    )
    atomic_write(
        args.output_audit,
        {
            "format": (
                "streameva.internal_threshold_uid_selection.v0"
                if threshold is not None
                else "streameva.unlabeled_budget_uid_selection.v0"
            ),
            "status": "frozen_before_benchmark_temporal_inference",
            "benchmark": recent.get("benchmark"),
            "model_manifest_sha256": recent.get("model_manifest_sha256"),
            "threshold": threshold,
            "quantile": quantile,
            "empirical_margin_cutoff": cutoff,
            "threshold_audit": (
                str(args.threshold.resolve()) if args.threshold is not None else None
            ),
            "threshold_audit_sha256": (
                sha256_file(args.threshold) if args.threshold is not None else None
            ),
            "recent_result": str(args.recent.resolve()),
            "recent_result_sha256": sha256_file(args.recent),
            "samples_sha256": sha256_file(args.samples),
            "question_task_group_or_benchmark_router": False,
            "score_or_gold_fields_read_for_selection": False,
            "benchmark_identity_used_for_policy_parameters": False,
            "cohort_labels_read_for_selection": False,
            "total": len(audit_rows),
            "selected": len(selected),
            "selection_rate": len(selected) / len(audit_rows),
            "rows": audit_rows,
        },
    )
    print(json.dumps({"total": len(audit_rows), "selected": len(selected)}))


if __name__ == "__main__":
    main()
