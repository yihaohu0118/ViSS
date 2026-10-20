#!/usr/bin/env python3
"""Merge disjoint agent-evaluation shards into canonical official UID order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


IDENTITY_FIELDS = (
    "format",
    "classification",
    "benchmark_selected",
    "source_policy_gate_admitted",
    "schema_version",
    "benchmark",
    "model",
    "model_config_sha256",
    "model_manifest_sha256",
    "processor",
    "processor_sha256",
    "processor_contract",
    "controller_processor",
    "controller_processor_sha256",
    "answer_processor",
    "answer_processor_contract",
    "processor_geometry",
    "adapter",
    "evaluator",
    "evaluation_protocol",
    "evaluation_protocol_sha256",
    "protocol_validation",
    "policy_hash",
    "policy_head",
    "policy_head_sha256",
    "policy_report_sha256",
    "policy_calibration_sha256",
    "global_action_threshold",
    "policy_inputs",
    "budget",
    "policy_training_report",
    "one_shared_commit_policy",
    "benchmark_task_or_group_router",
    "question_keyword_router",
    "answer_space_router",
    "gold_answer_available_to_commit_policy",
    "commit_policy",
    "unparseable_or_error_policy",
    "maximum_unique_frames",
    "direct_prediction_cache",
    "direct_prediction_cache_sha256",
    "source_gate",
    "source_gate_sha256",
    "global_action_thresholds",
    "acquisition_identity",
    "action_abi_schema_sha256",
    "inference_policy_hash",
    "inference_policy",
    "policy_config",
    "comparable_under_locked_recent4_protocol",
    "generation_config",
    "execution_optimization",
    "system_prompt_sha256",
    "prompt_contract_sha256",
    "launch_marker",
    "launch_marker_sha256",
    "annotation",
    "annotation_sha256",
    "samples",
    "samples_sha256",
    "runner_sha256",
    "software",
    "policy",
    "identity_hash",
    "protocol_sha256",
    "evaluator_sha256",
    "generation",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_expected_uids(path: Path) -> list[str]:
    uids: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        uid = str(json.loads(line).get("uid") or "")
        if not uid:
            raise ValueError(f"{path}:{line_number} has no uid")
        uids.append(uid)
    if len(uids) != len(set(uids)):
        raise ValueError(f"{path} contains duplicate uids")
    return uids


def grouped_accuracy(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field) or "unknown")].append(row)
    result: dict[str, Any] = {}
    for name in sorted(grouped):
        group_rows = grouped[name]
        correct = sum(int(row.get("correct") or 0) for row in group_rows)
        total = len(group_rows)
        result[name] = {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
        }
    return result


def aggregate_score(
    score_rows: list[dict[str, Any]], benchmark: str
) -> dict[str, Any]:
    correct = sum(int(row.get("correct") or 0) for row in score_rows)
    total = len(score_rows)
    by_task = grouped_accuracy(score_rows, "task")
    by_group = grouped_accuracy(score_rows, "group")
    score: dict[str, Any] = {
        "accuracy": correct / total if total else 0.0,
        "by_group": by_group,
        "by_task": by_task,
        "num_queries": total,
        "rows": score_rows,
    }
    if benchmark == "ovo":
        tasks_by_group: dict[str, set[str]] = defaultdict(set)
        for row in score_rows:
            tasks_by_group[str(row.get("group") or "unknown")].add(
                str(row.get("task") or "unknown")
            )
        official_groups: dict[str, Any] = {}
        for group in sorted(tasks_by_group):
            task_names = sorted(tasks_by_group[group])
            official_groups[group] = {
                "accuracy": fmean(by_task[name]["accuracy"] for name in task_names),
                "num_tasks": len(task_names),
            }
        score["ovo_official"] = {
            "accuracy": fmean(
                value["accuracy"] for value in official_groups.values()
            ),
            "by_group": official_groups,
        }
    return score


def aggregate_efficiency(rows: list[dict[str, Any]]) -> dict[str, float]:
    def mean(field: str, *fallbacks: str) -> float:
        def value(row: dict[str, Any]) -> float:
            for name in (field, *fallbacks):
                if row.get(name) is not None:
                    return float(row[name])
            return 0.0

        return fmean(value(row) for row in rows) if rows else 0.0

    result = {
        "avg_frames_observed": mean(
            "frames_observed", "unique_frames_acquired", "unique_frames"
        ),
        "avg_model_processed_frames": mean(
            "model_processed_frames", "processed_frames"
        ),
        "avg_media_calls": mean("media_calls"),
        "avg_total_tokens": (
            mean("total_tokens")
            if any(row.get("total_tokens") is not None for row in rows)
            else mean("input_tokens") + mean("output_tokens")
        ),
        "avg_turns": mean("num_turns"),
    }
    if any(row.get("commit_latency_seconds") is not None for row in rows):
        result["avg_commit_latency_seconds"] = mean("commit_latency_seconds")
    if any(row.get("commit_visual_tokens") is not None for row in rows):
        result["avg_commit_visual_tokens"] = mean("commit_visual_tokens")
    if any(row.get("latency_seconds") is not None for row in rows):
        result["avg_latency_seconds"] = mean("latency_seconds")
    return result


def aggregate_agent_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    terminal_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        terminal_counts[str(row.get("policy_terminal_event") or "unknown")] += 1
    total = len(rows)
    failures = sum(
        terminal_counts[event]
        for event in ("malformed_action", "illegal_observe", "missing_answer")
    )
    return {
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "num_policy_failures": failures,
        "strict_terminal_answer_rate": (
            terminal_counts["answered"] / total if total else 0.0
        ),
        "malformed_action_rate": (
            terminal_counts["malformed_action"] / total if total else 0.0
        ),
        "illegal_observe_rate": (
            terminal_counts["illegal_observe"] / total if total else 0.0
        ),
        "missing_answer_rate": (
            terminal_counts["missing_answer"] / total if total else 0.0
        ),
    }


def aggregate_execution_statistics(
    shard_statistics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sum disjoint-shard cache counters and per-row stage timings."""

    counter_fields = (
        "decode_window_cache_hits",
        "decode_window_cache_misses",
        "decode_window_singleflight_hits",
        "temporal_budget_cache_hits",
        "temporal_budget_cache_misses",
        "video_reader_cache_hits",
        "video_reader_cache_misses",
        "stage_timing_samples",
    )
    stage_totals: dict[str, float] = defaultdict(float)
    result = {
        field: sum(int(stats.get(field) or 0) for stats in shard_statistics)
        for field in counter_fields
    }
    for stats in shard_statistics:
        for name, seconds in (stats.get("stage_seconds_total") or {}).items():
            stage_totals[str(name)] += float(seconds or 0.0)
    result["stage_seconds_total"] = {
        name: round(seconds, 6) for name, seconds in sorted(stage_totals.items())
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=-1)
    parser.add_argument(
        "--expected-uids",
        type=Path,
        help="Official JSONL export whose UID order becomes the canonical order.",
    )
    args = parser.parse_args()

    rows_by_uid: dict[str, dict[str, Any]] = {}
    score_by_uid: dict[str, dict[str, Any]] = {}
    seen_in: dict[str, str] = {}
    metadata: list[dict[str, Any]] = []
    execution_statistics: list[dict[str, Any]] = []
    row_repairs: list[dict[str, Any]] = []
    identity: dict[str, Any] | None = None
    benchmark: str | None = None

    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        # A sharded run intentionally records a different ``samples`` JSONL
        # path (and therefore a different samples hash) in every shard.  When
        # a frozen full-export is supplied via --expected-uids, that export is
        # the canonical merged sample manifest; all other identity fields
        # must still match exactly.
        comparable_identity_fields = tuple(
            field
            for field in IDENTITY_FIELDS
            if args.expected_uids is None
            or field not in {"samples", "samples_sha256"}
        )
        current_identity = {
            field: data.get(field) for field in comparable_identity_fields
        }
        if identity is None:
            identity = current_identity
            benchmark = str(data.get("benchmark") or "") or None
        elif current_identity != identity:
            changed = [
                field
                for field in comparable_identity_fields
                if current_identity.get(field) != identity.get(field)
            ]
            # Annotation files are frequently remounted from a volatile
            # location (for example /dev/shm) to persistent storage when an
            # evaluation is resumed.  The absolute path is not part of the
            # benchmark identity when both shards carry the same non-empty
            # content hash.  Keep the hash itself in the frozen identity so
            # different annotation contents remain a hard failure.
            if (
                "annotation" in changed
                and current_identity.get("annotation_sha256")
                and current_identity.get("annotation_sha256")
                == identity.get("annotation_sha256")
            ):
                changed.remove("annotation")
            if not changed:
                current_identity = identity
            else:
                raise ValueError(
                    f"{path} does not match the frozen shard identity: {changed}"
                )
        shard_rows = data.get("rows")
        if not isinstance(shard_rows, list):
            raise ValueError(f"{path} has no rows list")
        for row in shard_rows:
            uid = str(row.get("uid") or "")
            if not uid:
                raise ValueError(f"{path} contains a row without uid")
            if uid in rows_by_uid:
                raise ValueError(f"duplicate uid {uid!r} in {seen_in[uid]} and {path}")
            rows_by_uid[uid] = row
            seen_in[uid] = str(path)
            row_benchmark = str(row.get("benchmark") or "")
            if row_benchmark:
                if benchmark is None:
                    benchmark = row_benchmark
                elif row_benchmark != benchmark:
                    raise ValueError(
                        f"mixed benchmarks in shards: {benchmark!r} and {row_benchmark!r}"
                    )
        shard_score_rows = (data.get("score") or {}).get("rows") or []
        if not isinstance(shard_score_rows, list):
            raise ValueError(f"{path} score.rows is not a list")
        for score_row in shard_score_rows:
            uid = str(score_row.get("query_id") or "")
            if not uid:
                raise ValueError(f"{path} contains a score row without query_id")
            if uid in score_by_uid:
                raise ValueError(f"duplicate score query_id {uid!r}")
            score_by_uid[uid] = score_row
        metadata.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "num_samples": len(shard_rows),
                "sample_slice": data.get("sample_slice"),
                "samples": data.get("samples"),
                "samples_sha256": data.get("samples_sha256"),
                "annotation": data.get("annotation"),
                "annotation_sha256": data.get("annotation_sha256"),
            }
        )
        if isinstance(data.get("execution_statistics"), dict):
            execution_statistics.append(data["execution_statistics"])
        row_repairs.extend(data.get("row_repairs") or [])

    if args.expected_count >= 0 and len(rows_by_uid) != args.expected_count:
        raise ValueError(
            f"merged {len(rows_by_uid)} rows, expected {args.expected_count}"
        )

    expected_uids = (
        load_expected_uids(args.expected_uids)
        if args.expected_uids is not None
        else list(rows_by_uid)
    )
    expected_set = set(expected_uids)
    actual_set = set(rows_by_uid)
    if actual_set != expected_set:
        missing = [uid for uid in expected_uids if uid not in actual_set]
        extra = sorted(actual_set - expected_set)
        raise ValueError(
            f"official UID mismatch: missing={missing[:20]} extra={extra[:20]}"
        )
    if set(score_by_uid) != expected_set:
        missing = [uid for uid in expected_uids if uid not in score_by_uid]
        extra = sorted(set(score_by_uid) - expected_set)
        raise ValueError(
            f"score UID mismatch: missing={missing[:20]} extra={extra[:20]}"
        )

    rows = [rows_by_uid[uid] for uid in expected_uids]
    score_rows = [score_by_uid[uid] for uid in expected_uids]
    empty_predictions = [
        str(row.get("uid"))
        for row in rows
        if not str(row.get("prediction") or "").strip()
    ]
    errors = [
        row for row in rows if row.get("error") or row.get("commit_error")
    ]
    merged_identity = dict(identity or {})
    if args.expected_uids is not None:
        merged_identity["samples"] = str(args.expected_uids)
        merged_identity["samples_sha256"] = sha256(args.expected_uids)
    result = {
        "benchmark": benchmark,
        **merged_identity,
        "canonical_official_uid_order": args.expected_uids is not None,
        "efficiency": aggregate_efficiency(rows),
        "empty_prediction_uids": empty_predictions,
        "num_empty_predictions": len(empty_predictions),
        "num_errors": len(errors),
        "num_future_frame_violations": sum(
            bool(row.get("future_frame_violation")) for row in rows
        ),
        "num_samples": len(rows),
        "strict_terminal_answer_rate": (
            sum(bool(row.get("strict_terminal_answer")) for row in rows) / len(rows)
            if rows else 0.0
        ),
        "score": aggregate_score(score_rows, benchmark or ""),
        "row_repairs": row_repairs,
        "source_shards": metadata,
        "expected_uid_source": (
            {
                "path": str(args.expected_uids),
                "sha256": sha256(args.expected_uids),
            }
            if args.expected_uids is not None
            else None
        ),
        "rows": rows,
    }
    if execution_statistics:
        result["execution_statistics"] = aggregate_execution_statistics(
            execution_statistics
        )
    if any("policy_terminal_event" in row for row in rows):
        result["agent_behavior"] = aggregate_agent_behavior(rows)
        result["strict_terminal_answer_rate"] = result["agent_behavior"][
            "strict_terminal_answer_rate"
        ]
    if any("commit_action" in row for row in rows):
        result.update(
            {
                "num_commit_errors": sum(
                    bool(row.get("commit_error")) for row in rows
                ),
                "num_unparsed": sum(
                    row.get("commit_parsed") is False for row in rows
                ),
                "decision_counts": {
                    action: sum(row.get("commit_action") == action for row in rows)
                    for action in (
                        "not_applicable",
                        "unchanged",
                        "retain",
                        "commit",
                    )
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_samples": len(rows),
                "num_errors": len(errors),
                "num_empty_predictions": len(empty_predictions),
                "canonical_official_uid_order": args.expected_uids is not None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
