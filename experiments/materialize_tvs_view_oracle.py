#!/usr/bin/env python3
"""Materialize a strict post-hoc oracle over three complete causal views."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import fmean
from typing import Any

try:
    from experiments.tvs_correctness import is_correct, load_gold
except ModuleNotFoundError:  # Direct script execution without repository PYTHONPATH.
    from tvs_correctness import is_correct, load_gold


IDENTITY_FIELDS = (
    "benchmark",
    "model_manifest_sha256",
    "annotation_sha256",
    "evaluation_protocol_sha256",
)


def _index(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = value.get("rows") or []
    result = {str(row["uid"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate view UID")
    return result


def materialize(
    recent: dict[str, Any],
    event: dict[str, Any],
    history: dict[str, Any],
    gold: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    views = [_index(value) for value in (recent, event, history)]
    uid_set = set(views[0])
    if any(set(view) != uid_set for view in views[1:]) or set(gold) != uid_set:
        raise ValueError("strict oracle requires exact three-view and gold coverage")
    for field in IDENTITY_FIELDS:
        expected = recent.get(field)
        if expected in (None, ""):
            raise ValueError(f"recent view lacks identity: {field}")
        for value in (event, history):
            observed = value.get(field)
            if observed not in (None, "", expected):
                raise ValueError(f"view identity mismatch: {field}")

    rows = []
    chosen_counts = [0, 0, 0]
    for source in recent.get("rows") or []:
        uid = str(source["uid"])
        candidates = [view[uid] for view in views]
        correct = [is_correct(row.get("prediction"), gold[uid]) for row in candidates]
        chosen_index = next((i for i, value in enumerate(correct) if value), 0)
        chosen_counts[chosen_index] += 1
        output = copy.deepcopy(candidates[chosen_index])
        output.update(
            {
                "uid": uid,
                "prediction": str(candidates[chosen_index].get("prediction") or ""),
                "media_calls": 3,
                "model_processed_frames": sum(
                    int(row.get("frames_observed") or fallback)
                    for row, fallback in zip(candidates, (4, 20, 32), strict=True)
                ),
                "tvs_oracle": {
                    "analysis_only": True,
                    "gold_available_to_selector": True,
                    "correct_proposals": correct,
                    "chosen_view": ("recent", "event", "history")[chosen_index],
                },
            }
        )
        rows.append(output)
    return {
        **{field: recent.get(field) for field in IDENTITY_FIELDS},
        "format": "iclr_video.strict_three_view_oracle.v1",
        "status": "complete",
        "analysis_only": True,
        "num_samples": len(rows),
        "num_errors": 0,
        "mean_model_calls": 3.0,
        "mean_processed_frames": fmean(row["model_processed_frames"] for row in rows),
        "chosen_view_counts": dict(zip(("recent", "event", "history"), chosen_counts, strict=True)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("streamingbench", "ovo"), required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--recent", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    values = [json.loads(path.read_text()) for path in (args.recent, args.event, args.history)]
    result = materialize(*values, load_gold(args.benchmark, args.annotation))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "rows": result["num_samples"],
        "mean_model_calls": result["mean_model_calls"],
        "mean_processed_frames": result["mean_processed_frames"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
