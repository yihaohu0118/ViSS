#!/usr/bin/env python3
"""Materialize a router-free third-view agent on event-view disagreements.

The frozen first stage decides whether to acquire the event view.  A third
32-frame view is acquired only when that event proposal disagrees with the
recent proposal, and its answer is committed directly.  Runtime decisions use
predictions and acquisition state only; task names, benchmark identity, and
gold labels are never inputs to the policy.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import fmean
from typing import Any


IDENTITY_FIELDS = (
    "benchmark",
    "model_manifest_sha256",
    "annotation_sha256",
    "evaluation_protocol_sha256",
)


def _index(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = value.get("rows") or []
    indexed = {str(row.get("uid") or ""): row for row in rows}
    if "" in indexed or len(indexed) != len(rows):
        raise ValueError("result rows contain empty or duplicate UIDs")
    return indexed


def _frame_indices(row: dict[str, Any]) -> set[int]:
    return {int(value) for value in (row.get("sampling") or {}).get("frame_indices") or []}


def materialize(
    recent: dict[str, Any],
    event: dict[str, Any],
    base_agent: dict[str, Any],
    third: dict[str, Any],
) -> dict[str, Any]:
    for field in IDENTITY_FIELDS:
        values = {value.get(field) for value in (recent, event, base_agent, third)}
        if len(values) != 1:
            raise ValueError(f"view identity mismatch: {field}")
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
    if set(third_rows) != disagreements:
        raise ValueError("third-view coverage must equal the event disagreement cohort")

    output_rows = []
    for recent_row in recent.get("rows") or []:
        uid = str(recent_row["uid"])
        base = copy.deepcopy(base_rows[uid])
        if uid not in third_rows:
            base["third_view_policy"] = {
                "triggered": False,
                "reason": "no_acquired_event_disagreement",
                "question_task_group_or_benchmark_router": False,
            }
            output_rows.append(base)
            continue

        proposal = event_rows[uid]
        adjudicator = third_rows[uid]
        if proposal.get("error") or adjudicator.get("error"):
            raise ValueError(f"{uid}: acquired view contains an error")
        if not bool((base.get("agent_policy") or {}).get("escalated")):
            raise ValueError(f"{uid}: third view was triggered outside the acquired cohort")
        timestamps = (adjudicator.get("sampling") or {}).get("frame_timestamps") or []
        if timestamps and max(map(float, timestamps)) > float(adjudicator["query_time"]) + 1e-3:
            raise ValueError(f"{uid}: third view crosses the causal frontier")
        recent_prediction = str(recent_row.get("prediction") or "")
        event_prediction = str(proposal.get("prediction") or "")
        third_prediction = str(adjudicator.get("prediction") or "")
        if not recent_prediction or not event_prediction or not third_prediction:
            raise ValueError(f"{uid}: an agent proposal is empty")
        if recent_prediction == event_prediction:
            raise ValueError(f"{uid}: third view lacks a disagreement trigger")

        unique_indices = _frame_indices(proposal) | _frame_indices(adjudicator)
        base.update(
            {
                "prediction": third_prediction,
                "raw_response": adjudicator.get("raw_response"),
                "third_view_prediction": third_prediction,
                "third_view_policy": {
                    "triggered": True,
                    "trigger": "acquired_event_prediction_differs_from_recent_prediction",
                    "commit": "direct_third_view_prediction",
                    "recent_prediction": recent_prediction,
                    "event_prediction": event_prediction,
                    "question_task_group_or_benchmark_router": False,
                    "gold_available_to_policy": False,
                },
                "media_calls": int(base.get("media_calls") or 2) + 1,
                "num_turns": int(base.get("num_turns") or 2) + 1,
                "model_processed_frames": int(base.get("model_processed_frames") or 24)
                + int(adjudicator.get("frames_observed") or 32),
                "unique_frames": len(unique_indices) if unique_indices else 32,
                "latency_seconds": float(base.get("latency_seconds") or 0.0)
                + float(adjudicator.get("latency_seconds") or 0.0),
                "visual_tokens": int(recent_row.get("visual_tokens") or 0)
                + int(proposal.get("visual_tokens") or 0)
                + int(adjudicator.get("visual_tokens") or 0),
            }
        )
        output_rows.append(base)
    if any(not str(row.get("prediction") or "") for row in output_rows):
        raise ValueError("materialized agent contains an empty prediction")

    triggered = len(third_rows)
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
        "format": "streameva.third_view_disagreement_agent.v0",
        "status": "benchmark_selected_router_free_development_agent",
        "classification": "benchmark_selected_development_not_confirmatory",
        "comparable_under_locked_recent4_protocol": False,
        "policy": {
            "first_view": "recent4",
            "second_view": "frozen_uncertainty_triggered_causalbudget20",
            "third_view": "causalbudget32r8_on_acquired_answer_disagreement",
            "third_view_commit": "direct_prediction",
            "question_task_group_or_benchmark_router": False,
            "benchmark_identity_available_to_policy": False,
            "gold_available_to_policy": False,
        },
        "num_samples": len(output_rows),
        "num_errors": 0,
        "third_view_triggered": triggered,
        "third_view_trigger_rate": triggered / len(output_rows),
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    values = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.recent, args.event, args.base_agent, args.third_view)
    ]
    result = materialize(*values)
    result["artifacts"] = {
        "recent": str(args.recent.resolve()),
        "event": str(args.event.resolve()),
        "base_agent": str(args.base_agent.resolve()),
        "third_view": str(args.third_view.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_samples": result["num_samples"],
                "third_view_triggered": result["third_view_triggered"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
