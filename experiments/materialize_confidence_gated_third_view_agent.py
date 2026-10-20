#!/usr/bin/env python3
"""Materialize a task-blind confidence-gated third-view temporal agent.

The third view is acquired on the same answer-disagreement cohort as the direct
third-view agent.  Its proposal is committed only when its first-token margin
improves over the margin of the view that supplied the current answer by more
than one source-disjoint, globally frozen threshold.  Runtime policy inputs do
not include task, benchmark identity, or gold labels.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from experiments.analyze_uncertainty_temporal_escalation import choice_margin
from experiments.materialize_third_view_disagreement_agent import (
    IDENTITY_FIELDS,
    _frame_indices,
    _index,
)


def materialize(
    recent: dict[str, Any],
    event: dict[str, Any],
    base_agent: dict[str, Any],
    third: dict[str, Any],
    threshold: dict[str, Any],
) -> dict[str, Any]:
    for field in IDENTITY_FIELDS:
        values = {value.get(field) for value in (recent, event, base_agent, third)}
        if len(values) != 1:
            raise ValueError(f"view identity mismatch: {field}")
    if (
        threshold.get("status") != "source_disjoint_internal_threshold_calibration"
        or threshold.get("eligible_as_frozen_benchmark_controller_config") is not True
        or threshold.get("model_manifest_sha256")
        != recent.get("model_manifest_sha256")
    ):
        raise ValueError("threshold is not an identity-aligned internal calibration")
    commit_threshold = float(threshold["policy"]["commit_threshold"])

    recent_rows, event_rows, base_rows, third_rows = map(
        _index, (recent, event, base_agent, third)
    )
    if set(base_rows) != set(recent_rows) or not set(event_rows) <= set(recent_rows):
        raise ValueError("base/recent/event UID coverage diverged")
    disagreements = {
        uid
        for uid, row in event_rows.items()
        if str(row.get("prediction") or "")
        != str(recent_rows[uid].get("prediction") or "")
    }
    if not disagreements <= set(third_rows):
        raise ValueError("third-view coverage must include the event disagreement cohort")

    output_rows: list[dict[str, Any]] = []
    committed_count = 0
    for recent_row in recent.get("rows") or []:
        uid = str(recent_row["uid"])
        base = copy.deepcopy(base_rows[uid])
        if uid not in disagreements:
            base["third_view_policy"] = {
                "triggered": False,
                "reason": "no_acquired_event_disagreement",
                "question_task_group_or_benchmark_router": False,
            }
            output_rows.append(base)
            continue

        event_row = event_rows[uid]
        third_row = third_rows[uid]
        if event_row.get("error") or third_row.get("error"):
            raise ValueError(f"{uid}: acquired view contains an error")
        if not bool((base.get("agent_policy") or {}).get("escalated")):
            raise ValueError(f"{uid}: third view was triggered outside acquired cohort")
        timestamps = (third_row.get("sampling") or {}).get("frame_timestamps") or []
        if timestamps and max(map(float, timestamps)) > float(third_row["query_time"]) + 1e-3:
            raise ValueError(f"{uid}: third view crosses the causal frontier")

        recent_prediction = str(recent_row.get("prediction") or "")
        event_prediction = str(event_row.get("prediction") or "")
        current_prediction = str(base.get("prediction") or "")
        third_prediction = str(third_row.get("prediction") or "")
        if not all((recent_prediction, event_prediction, current_prediction, third_prediction)):
            raise ValueError(f"{uid}: an agent proposal is empty")
        if recent_prediction == event_prediction:
            raise ValueError(f"{uid}: third view lacks a disagreement trigger")
        if current_prediction == recent_prediction:
            current_source = recent_row
            current_view = "recent4"
        elif current_prediction == event_prediction:
            current_source = event_row
            current_view = "causalbudget20"
        else:
            raise ValueError(f"{uid}: current answer is not supplied by recent or event view")

        confidence_gain = choice_margin(third_row) - choice_margin(current_source)
        committed = bool(
            third_prediction != current_prediction
            and confidence_gain > commit_threshold
        )
        if committed:
            committed_count += 1
            base["prediction"] = third_prediction
            base["raw_response"] = third_row.get("raw_response")
        unique_indices = _frame_indices(event_row) | _frame_indices(third_row)
        base.update(
            {
                "current_prediction": current_prediction,
                "third_view_prediction": third_prediction,
                "third_view_policy": {
                    "triggered": True,
                    "trigger": "acquired_event_prediction_differs_from_recent_prediction",
                    "commit": "global_source_disjoint_confidence_gain",
                    "current_view": current_view,
                    "current_margin": choice_margin(current_source),
                    "third_margin": choice_margin(third_row),
                    "confidence_gain": confidence_gain,
                    "commit_threshold": commit_threshold,
                    "committed": committed,
                    "question_task_group_or_benchmark_router": False,
                    "gold_available_to_policy": False,
                },
                "media_calls": int(base.get("media_calls") or 2) + 1,
                "num_turns": int(base.get("num_turns") or 2) + 1,
                "model_processed_frames": int(base.get("model_processed_frames") or 24)
                + int(third_row.get("frames_observed") or 32),
                "unique_frames": len(unique_indices) if unique_indices else 32,
                "latency_seconds": float(base.get("latency_seconds") or 0.0)
                + float(third_row.get("latency_seconds") or 0.0),
                "visual_tokens": int(recent_row.get("visual_tokens") or 0)
                + int(event_row.get("visual_tokens") or 0)
                + int(third_row.get("visual_tokens") or 0),
            }
        )
        output_rows.append(base)

    if any(not str(row.get("prediction") or "") for row in output_rows):
        raise ValueError("materialized agent contains an empty prediction")
    triggered = len(disagreements)
    return {
        **{
            key: base_agent.get(key)
            for key in (
                "annotation",
                "annotation_sha256",
                "benchmark",
                "evaluation_protocol",
                "evaluation_protocol_sha256",
                "model",
                "model_manifest_sha256",
                "processor",
            )
        },
        "format": "streameva.confidence_gated_third_view_agent.v0",
        "status": "benchmark_selected_router_free_development_agent",
        "classification": "benchmark_selected_development_not_confirmatory",
        "comparable_under_locked_recent4_protocol": False,
        "policy": {
            "first_view": "recent4",
            "second_view": "frozen_uncertainty_triggered_causalbudget20",
            "third_view": "causalbudget32r8_on_acquired_answer_disagreement",
            "third_view_commit": "global_source_disjoint_confidence_gain",
            "commit_threshold": commit_threshold,
            "question_task_group_or_benchmark_router": False,
            "benchmark_identity_available_to_policy": False,
            "gold_available_to_policy": False,
        },
        "num_samples": len(output_rows),
        "num_errors": 0,
        "third_view_triggered": triggered,
        "third_view_input_rows": len(third_rows),
        "third_view_unused_precomputed_rows": len(set(third_rows) - disagreements),
        "third_view_committed": committed_count,
        "third_view_trigger_rate": triggered / len(output_rows),
        "third_view_commit_rate": committed_count / triggered if triggered else 0.0,
        "mean_model_calls": fmean(int(row.get("media_calls") or 1) for row in output_rows),
        "mean_unique_frames": fmean(int(row.get("unique_frames") or 4) for row in output_rows),
        "mean_model_processed_frames": fmean(
            int(row.get("model_processed_frames") or 4) for row in output_rows
        ),
        "rows": output_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recent", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--base-agent", type=Path, required=True)
    parser.add_argument("--third-view", type=Path, required=True)
    parser.add_argument("--threshold-audit", type=Path, required=True)
    parser.add_argument("--third-view-model-identity-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    values = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            args.recent,
            args.event,
            args.base_agent,
            args.third_view,
            args.threshold_audit,
        )
    ]
    identity_override = None
    if args.third_view_model_identity_sha256:
        if len(args.third_view_model_identity_sha256) != 64 or any(
            value not in "0123456789abcdef"
            for value in args.third_view_model_identity_sha256
        ):
            raise ValueError("third-view model identity is malformed")
        third = values[3]
        if third.get("model_manifest_sha256") not in {
            None,
            args.third_view_model_identity_sha256,
        }:
            raise ValueError("third-view model identity override diverges")
        if third.get("model_manifest_sha256") is None:
            third["model_manifest_sha256"] = args.third_view_model_identity_sha256
            identity_override = {
                "field": "model_manifest_sha256",
                "source": "previously_sealed_checkpoint_identity",
                "value": args.third_view_model_identity_sha256,
            }
    result = materialize(*values)
    result["third_view_identity_override"] = identity_override
    result["artifacts"] = {
        "recent": str(args.recent.resolve()),
        "event": str(args.event.resolve()),
        "base_agent": str(args.base_agent.resolve()),
        "third_view": str(args.third_view.resolve()),
        "threshold_audit": str(args.threshold_audit.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_samples": result["num_samples"],
                "triggered": result["third_view_triggered"],
                "committed": result["third_view_committed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
