#!/usr/bin/env python3
"""Audit a task-agnostic recent-to-temporal escalation policy.

The online policy sees only the recent-view first-token generation margin. Gold labels are
used offline to fit one global threshold on a deterministic calibration split
and to score the disjoint exploratory split.  Task, group, benchmark identity,
and answer-space semantics are never policy inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recent", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration-modulus",
        type=int,
        default=2,
        help="SHA256(uid) modulo this value; bucket zero fits the threshold.",
    )
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise ValueError(f"{path}: expected benchmark result with rows")
    if int(value.get("num_errors", 0)) != 0:
        raise ValueError(f"{path}: result contains inference errors")
    return value


def index_rows(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed = {str(row.get("uid")): row for row in result["rows"]}
    if "" in indexed or len(indexed) != len(result["rows"]):
        raise ValueError("result contains missing or duplicate UIDs")
    return indexed


def index_scores(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = (result.get("score") or {}).get("rows")
    if not isinstance(rows, list):
        raise ValueError("result is missing score.rows")
    indexed = {str(row.get("query_id")): row for row in rows}
    if "" in indexed or len(indexed) != len(rows):
        raise ValueError("score rows contain missing or duplicate query IDs")
    return indexed


def index_temporal_scores(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    score_rows = (result.get("score") or {}).get("rows")
    if isinstance(score_rows, list):
        return index_scores(result)
    rows = result.get("rows")
    if not isinstance(rows, list) or any("correct" not in row for row in rows):
        raise ValueError("temporal result lacks official or inherited correctness")
    indexed = {str(row.get("uid")): row for row in rows}
    if "" in indexed or len(indexed) != len(rows):
        raise ValueError("temporal score rows contain missing or duplicate UIDs")
    return indexed


def calibration_bucket(uid: str, modulus: int) -> int:
    if modulus < 2:
        raise ValueError("calibration modulus must be at least two")
    return int(hashlib.sha256(uid.encode("utf-8")).hexdigest(), 16) % modulus


def choice_margin(row: dict[str, Any]) -> float:
    # New confirmatory runs use the same full-vocabulary first-token margin for
    # MCQ and open-ended questions. Older exploratory artifacts contain only
    # the option-letter diagnostic and remain readable for historical audits.
    confidence = row.get("generation_confidence") or row.get("choice_confidence")
    if not isinstance(confidence, dict):
        raise ValueError(f"{row.get('uid')}: missing generation confidence")
    if confidence.get("task_or_group_used") is not False:
        raise ValueError(f"{row.get('uid')}: confidence provenance is not task blind")
    value = float(confidence.get("top2_margin"))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{row.get('uid')}: invalid generation margin")
    return value


def select_prediction(
    recent_row: dict[str, Any], temporal_row: dict[str, Any], threshold: float
) -> tuple[str, bool]:
    escalate = choice_margin(recent_row) <= threshold
    source = temporal_row if escalate else recent_row
    return str(
        source.get("prediction") or source.get("selected_prediction") or ""
    ), escalate


def confidence_gain(
    recent_row: dict[str, Any], temporal_row: dict[str, Any]
) -> float:
    """Return the task-blind confidence change after acquiring the temporal view."""

    return choice_margin(temporal_row) - choice_margin(recent_row)


def select_acquire_commit_prediction(
    recent_row: dict[str, Any],
    temporal_row: dict[str, Any],
    acquire_threshold: float,
    commit_threshold: float,
) -> tuple[str, bool, bool, float]:
    """Apply one shared acquire threshold and one shared safe-commit threshold."""

    acquired = choice_margin(recent_row) <= acquire_threshold
    recent_prediction = str(
        recent_row.get("prediction") or recent_row.get("selected_prediction") or ""
    )
    temporal_prediction = str(
        temporal_row.get("prediction")
        or temporal_row.get("selected_prediction")
        or ""
    )
    gain = confidence_gain(recent_row, temporal_row)
    committed = bool(
        acquired
        and temporal_prediction != recent_prediction
        and gain > commit_threshold
    )
    return (
        temporal_prediction if committed else recent_prediction,
        acquired,
        committed,
        gain,
    )


def evaluate_threshold(
    uids: list[str],
    *,
    recent: dict[str, dict[str, Any]],
    temporal: dict[str, dict[str, Any]],
    recent_scores: dict[str, dict[str, Any]],
    temporal_scores: dict[str, dict[str, Any]],
    threshold: float,
) -> tuple[int, int]:
    correct = 0
    escalated = 0
    for uid in uids:
        _, did_escalate = select_prediction(
            recent[uid], temporal[uid], threshold
        )
        official = temporal_scores[uid] if did_escalate else recent_scores[uid]
        correct += int(bool(official["correct"]))
        escalated += int(did_escalate)
    return correct, escalated


def fit_threshold(
    calibration_uids: list[str],
    *,
    recent: dict[str, dict[str, Any]],
    temporal: dict[str, dict[str, Any]],
    recent_scores: dict[str, dict[str, Any]],
    temporal_scores: dict[str, dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    margins = sorted({choice_margin(recent[uid]) for uid in calibration_uids})
    candidates = [-1.0, *margins, 1.0]
    sweep = []
    for threshold in candidates:
        correct, escalated = evaluate_threshold(
            calibration_uids,
            recent=recent,
            temporal=temporal,
            recent_scores=recent_scores,
            temporal_scores=temporal_scores,
            threshold=threshold,
        )
        sweep.append(
            {
                "threshold": threshold,
                "correct": correct,
                "total": len(calibration_uids),
                "accuracy": correct / len(calibration_uids),
                "escalated": escalated,
            }
        )
    best = max(
        sweep,
        key=lambda row: (
            int(row["correct"]),
            -int(row["escalated"]),
            -float(row["threshold"]),
        ),
    )
    return float(best["threshold"]), sweep


def evaluate_acquire_commit_thresholds(
    uids: list[str],
    *,
    recent: dict[str, dict[str, Any]],
    temporal: dict[str, dict[str, Any]],
    recent_scores: dict[str, dict[str, Any]],
    temporal_scores: dict[str, dict[str, Any]],
    acquire_threshold: float,
    commit_threshold: float,
) -> tuple[int, int, int]:
    correct = 0
    acquired = 0
    committed = 0
    for uid in uids:
        _, did_acquire, did_commit, _ = select_acquire_commit_prediction(
            recent[uid],
            temporal[uid],
            acquire_threshold,
            commit_threshold,
        )
        official = temporal_scores[uid] if did_commit else recent_scores[uid]
        correct += int(bool(official["correct"]))
        acquired += int(did_acquire)
        committed += int(did_commit)
    return correct, acquired, committed


def fit_acquire_commit_thresholds(
    calibration_uids: list[str],
    *,
    recent: dict[str, dict[str, Any]],
    temporal: dict[str, dict[str, Any]],
    recent_scores: dict[str, dict[str, Any]],
    temporal_scores: dict[str, dict[str, Any]],
) -> tuple[float, float, list[dict[str, Any]]]:
    acquire_candidates = [
        -1.000001,
        *sorted({choice_margin(recent[uid]) for uid in calibration_uids}),
        1.000001,
    ]
    commit_candidates = [
        -1.000001,
        *sorted(
            {confidence_gain(recent[uid], temporal[uid]) for uid in calibration_uids}
        ),
        1.000001,
    ]
    sweep = []
    for acquire_threshold in acquire_candidates:
        for commit_threshold in commit_candidates:
            correct, acquired, committed = evaluate_acquire_commit_thresholds(
                calibration_uids,
                recent=recent,
                temporal=temporal,
                recent_scores=recent_scores,
                temporal_scores=temporal_scores,
                acquire_threshold=acquire_threshold,
                commit_threshold=commit_threshold,
            )
            sweep.append(
                {
                    "acquire_threshold": acquire_threshold,
                    "commit_threshold": commit_threshold,
                    "correct": correct,
                    "total": len(calibration_uids),
                    "accuracy": correct / len(calibration_uids),
                    "acquired": acquired,
                    "committed": committed,
                }
            )
    best = max(
        sweep,
        key=lambda row: (
            int(row["correct"]),
            -int(row["acquired"]),
            -int(row["committed"]),
            -float(row["acquire_threshold"]),
            float(row["commit_threshold"]),
        ),
    )
    return (
        float(best["acquire_threshold"]),
        float(best["commit_threshold"]),
        sweep,
    )


def main() -> None:
    args = parse_args()
    recent_result = load_result(args.recent)
    temporal_result = load_result(args.temporal)
    for field in ("benchmark", "model_manifest_sha256", "samples_sha256"):
        if recent_result.get(field) != temporal_result.get(field):
            raise ValueError(f"recent/temporal identity mismatch: {field}")

    recent = index_rows(recent_result)
    temporal = index_rows(temporal_result)
    recent_scores = index_scores(recent_result)
    temporal_scores = index_temporal_scores(temporal_result)
    if (
        set(recent) != set(temporal)
        or set(recent) != set(recent_scores)
        or set(recent) != set(temporal_scores)
    ):
        raise ValueError("recent, temporal, and score UID sets must match exactly")
    if any(row.get("task") is None for row in recent_scores.values()):
        raise ValueError("score rows require task labels for reporting only")

    ordered_uids = [str(row["uid"]) for row in recent_result["rows"]]
    calibration_uids = [
        uid
        for uid in ordered_uids
        if calibration_bucket(uid, args.calibration_modulus) == 0
    ]
    evaluation_uids = [uid for uid in ordered_uids if uid not in calibration_uids]
    if not calibration_uids or not evaluation_uids:
        raise ValueError("deterministic split produced an empty partition")

    acquire_threshold, commit_threshold, sweep = fit_acquire_commit_thresholds(
        calibration_uids,
        recent=recent,
        temporal=temporal,
        recent_scores=recent_scores,
        temporal_scores=temporal_scores,
    )
    rows = []
    calibration_set = set(calibration_uids)
    by_task: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for uid in ordered_uids:
        prediction, escalated, committed, gain = select_acquire_commit_prediction(
            recent[uid], temporal[uid], acquire_threshold, commit_threshold
        )
        official = temporal_scores[uid] if committed else recent_scores[uid]
        correct = bool(official["correct"])
        partition = "calibration" if uid in calibration_set else "evaluation"
        task = str(recent_scores[uid]["task"])
        if partition == "evaluation":
            by_task[task][0] += int(correct)
            by_task[task][1] += 1
        rows.append(
            {
                "uid": uid,
                "partition": partition,
                "margin": choice_margin(recent[uid]),
                "temporal_margin": choice_margin(temporal[uid]),
                "confidence_gain": gain,
                "acquire_threshold": acquire_threshold,
                "commit_threshold": commit_threshold,
                "escalated": escalated,
                "committed": committed,
                "joint_required_after_escalation": bool(
                    escalated and temporal[uid].get("joint_required", False)
                ),
                "prediction": prediction,
                "correct": correct,
                "task": task,
                "task_used_by_policy": False,
            }
        )

    def partition_summary(partition: str) -> dict[str, Any]:
        chosen = [row for row in rows if row["partition"] == partition]
        correct = sum(int(row["correct"]) for row in chosen)
        escalated = sum(int(row["escalated"]) for row in chosen)
        committed = sum(int(row["committed"]) for row in chosen)
        joint_calls = sum(
            int(row["joint_required_after_escalation"]) for row in chosen
        )
        return {
            "correct": correct,
            "total": len(chosen),
            "accuracy": correct / len(chosen),
            "escalated": escalated,
            "escalation_rate": escalated / len(chosen),
            "committed": committed,
            "commit_rate": committed / len(chosen),
            "joint_calls": joint_calls,
            "mean_model_calls": (len(chosen) + escalated + joint_calls)
            / len(chosen),
        }

    internal_calibration = (
        recent_result.get("benchmark") == "internal_policy_holdout"
        and recent_result.get("status")
        == "source_disjoint_internal_controller_calibration"
        and temporal_result.get("status")
        == "source_disjoint_internal_controller_calibration"
    )
    output = {
        "format": "streameva.uncertainty_temporal_escalation_audit.v0",
        "status": (
            "source_disjoint_internal_threshold_calibration"
            if internal_calibration
            else "benchmark_selected_exploratory_not_confirmatory"
        ),
        "eligible_as_frozen_benchmark_controller_config": internal_calibration,
        "benchmark": recent_result.get("benchmark"),
        "model_manifest_sha256": recent_result.get("model_manifest_sha256"),
        "policy": {
            "first_view": (
                "recorded_initial_causal_view" if internal_calibration else "recent"
            ),
            "second_view": (
                "all_recorded_causal_views"
                if internal_calibration
                else temporal_result.get("inference_policy")
            ),
            "trigger": "recent_first_step_generation_top2_margin_leq_global_threshold",
            "threshold": acquire_threshold,
            "acquire_threshold": acquire_threshold,
            "commit_threshold": commit_threshold,
            "commit_signal": "temporal_minus_recent_full_vocabulary_top2_margin",
            "commit_rule": "changed_answer_and_confidence_gain_gt_global_threshold",
            "task_group_or_question_router": False,
            "gold_available_at_runtime": False,
        },
        "split": {
            "method": "sha256(uid) modulo calibration_modulus",
            "calibration_modulus": args.calibration_modulus,
            "calibration_bucket": 0,
        },
        "calibration": partition_summary("calibration"),
        "evaluation": partition_summary("evaluation"),
        "evaluation_by_task_reporting_only": {
            task: {"correct": values[0], "total": values[1], "accuracy": values[0] / values[1]}
            for task, values in sorted(by_task.items())
        },
        "calibration_sweep": sweep,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "acquire_threshold": acquire_threshold,
                "commit_threshold": commit_threshold,
                **output["evaluation"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
