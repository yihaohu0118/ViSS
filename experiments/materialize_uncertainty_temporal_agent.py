#!/usr/bin/env python3
"""Materialize one full prediction set from recent and triggered temporal calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from experiments.evaluate_qwen3vl_recent4_full import atomic_write
from experiments.analyze_uncertainty_temporal_escalation import choice_margin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recent", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--threshold-audit", type=Path)
    parser.add_argument(
        "--temporal-model-identity-sha256",
        help=(
            "Previously sealed model identity to attach when the temporal result "
            "was launched without embedding it; avoids rehashing the model."
        ),
    )
    parser.add_argument(
        "--commit-threshold",
        type=float,
        help=(
            "Optional fixed proposal-minus-current margin threshold. This is "
            "required for an unlabeled fixed-budget selection and overrides "
            "the commit threshold stored in an internal calibration."
        ),
    )
    parser.add_argument(
        "--fallback-recent-on-temporal-error",
        action="store_true",
        help=(
            "Fail closed to the aligned recent4 prediction when a preselected "
            "temporal call has a recorded runtime error. The fallback is fixed "
            "before scoring and never reads benchmark labels."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_view_name(policy: str, mode: str, *, escalated: bool) -> str:
    if not escalated:
        return "recent4"
    if policy in {"causalbudget20", "causalbudget32r8"}:
        suffix = "32_recent8" if policy == "causalbudget32r8" else "20"
        mode_name = {
            "native-video": "native_video",
            "image-list": "image_list",
            "hybrid-history-video-recent-images": (
                "hybrid_history_video_recent_images"
            ),
            "lowres-history-recent-images": "lowres_history_recent_images",
        }[mode]
        return f"causal_temporal{suffix}_{mode_name}"
    if policy in {"causalhybrid32", "causalhybrid32r8"}:
        recent_frames = 8 if policy == "causalhybrid32r8" else 4
        history_frames = 32 - recent_frames
        if mode == "image-list":
            return (
                f"causal_uniform_history{history_frames}_plus_recent"
                f"{recent_frames}_image_list"
            )
        return f"causal_hybrid_history{history_frames}_plus_recent{recent_frames}"
    if policy == "prefix32":
        return "causal_prefix32"
    raise ValueError(f"unsupported second-view policy: {policy}")


def second_view_policy_name(policy: str, mode: str) -> str:
    if policy in {"causalbudget20", "causalbudget32r8"}:
        suffix = "32_recent8" if policy == "causalbudget32r8" else "20"
        prefix = {
            "native-video": "native_video",
            "image-list": "image_list",
            "hybrid-history-video-recent-images": (
                "hybrid_history_video_recent_images"
            ),
            "lowres-history-recent-images": "lowres_history_recent_images",
        }[mode]
        return f"{prefix}_causal_temporal{suffix}"
    if policy in {"causalhybrid32", "causalhybrid32r8"}:
        if mode == "hybrid-history-video-recent-images":
            return (
                "hybrid_history_video_recent_images_causal_hybrid32r8"
                if policy == "causalhybrid32r8"
                else "hybrid_history_video_recent_images_causal_hybrid32"
            )
        return (
            "image_list_uniform_history24_plus_recent8"
            if policy == "causalhybrid32r8"
            else "image_list_uniform_history28_plus_recent4"
        )
    if policy == "prefix32":
        return "image_list_causal_prefix32"
    raise ValueError(f"unsupported second-view policy: {policy}")


def materialize(
    recent: dict[str, Any],
    temporal: dict[str, Any],
    selection: dict[str, Any],
    threshold: dict[str, Any] | None,
    *,
    commit_threshold_override: float | None = None,
    fallback_recent_on_temporal_error: bool = False,
) -> dict[str, Any]:
    for field in (
        "benchmark",
        "model_manifest_sha256",
        "annotation_sha256",
        "evaluation_protocol_sha256",
    ):
        if recent.get(field) != temporal.get(field):
            raise ValueError(f"recent/temporal identity mismatch: {field}")
    selection_format = selection.get("format")
    fixed_quantile = None
    frozen_threshold = None
    if threshold is not None:
        if (
            threshold.get("status")
            != "source_disjoint_internal_threshold_calibration"
            or threshold.get("eligible_as_frozen_benchmark_controller_config")
            is not True
            or threshold.get("model_manifest_sha256")
            != recent.get("model_manifest_sha256")
        ):
            raise ValueError("threshold is not an eligible internal calibration")
        frozen_threshold = float(threshold["policy"]["threshold"])
        commit_threshold = threshold["policy"].get("commit_threshold")
        if selection_format not in (None, "streameva.internal_threshold_uid_selection.v0"):
            raise ValueError("threshold calibration and selection mode diverge")
    else:
        if selection_format != "streameva.unlabeled_budget_uid_selection.v0":
            raise ValueError("selection is not an unlabeled fixed-budget audit")
        fixed_quantile = float(selection.get("quantile"))
        if not 0.0 < fixed_quantile <= 1.0:
            raise ValueError("fixed uncertainty quantile is invalid")
        if selection.get("benchmark_identity_used_for_policy_parameters") is not False:
            raise ValueError("fixed-budget acquisition used benchmark identity")
        if selection.get("cohort_labels_read_for_selection") is not False:
            raise ValueError("fixed-budget acquisition used cohort labels")
        if commit_threshold_override is None:
            raise ValueError("fixed-budget selection requires a commit threshold")
        commit_threshold = commit_threshold_override
    if commit_threshold_override is not None:
        commit_threshold = commit_threshold_override
    if commit_threshold is not None:
        commit_threshold = float(commit_threshold)
    if (
        selection.get("status")
        != "frozen_before_benchmark_temporal_inference"
        or selection.get("benchmark") != recent.get("benchmark")
        or selection.get("model_manifest_sha256")
        != recent.get("model_manifest_sha256")
        or selection.get("score_or_gold_fields_read_for_selection") is not False
    ):
        raise ValueError("selection audit is not frozen or identity-aligned")
    if threshold is not None and float(selection.get("threshold")) != frozen_threshold:
        raise ValueError("selection threshold diverges from internal calibration")

    recent_rows = {str(row["uid"]): row for row in recent.get("rows") or []}
    temporal_rows = {str(row["uid"]): row for row in temporal.get("rows") or []}
    decisions = {str(row["uid"]): row for row in selection.get("rows") or []}
    if (
        len(recent_rows) != len(recent.get("rows") or [])
        or len(temporal_rows) != len(temporal.get("rows") or [])
        or len(decisions) != len(selection.get("rows") or [])
    ):
        raise ValueError("duplicate UID in input artifacts")
    if set(decisions) != set(recent_rows):
        raise ValueError("selection audit does not cover the recent result")
    selected = {uid for uid, row in decisions.items() if bool(row["escalated"])}
    temporal_input_count = len(temporal_rows)
    if not selected.issubset(temporal_rows):
        missing = sorted(selected - set(temporal_rows))
        raise ValueError(
            f"temporal inference is missing {len(missing)} selected UIDs"
        )
    temporal_rows = {uid: temporal_rows[uid] for uid in selected}
    temporal_error_uids = {
        uid for uid, row in temporal_rows.items() if bool(row.get("error"))
    }
    if temporal_error_uids and not fallback_recent_on_temporal_error:
        raise ValueError(
            f"{len(temporal_error_uids)} selected temporal rows contain errors"
        )
    valid_temporal_rows = {
        uid: row for uid, row in temporal_rows.items() if uid not in temporal_error_uids
    }
    if not valid_temporal_rows:
        raise ValueError("no valid temporal row remains to establish the second-view policy")
    second_view_policies = {
        str((row.get("sampling") or {}).get("frame_policy") or "")
        for row in valid_temporal_rows.values()
    }
    if second_view_policies not in (
        {"causalbudget20"},
        {"causalbudget32r8"},
        {"prefix32"},
        {"causalhybrid32"},
        {"causalhybrid32r8"},
    ):
        raise ValueError(
            "second-view rows must share causalbudget20, causalbudget32r8, prefix32, or "
            "causalhybrid32*"
        )
    second_view_policy = next(iter(second_view_policies))
    second_view_modes = {
        str(row.get("visual_input_mode") or "")
        for row in valid_temporal_rows.values()
    }
    if len(second_view_modes) != 1:
        raise ValueError("second-view rows do not share one visual input mode")
    second_view_mode = next(iter(second_view_modes))
    if second_view_mode not in {
        "native-video",
        "hybrid-history-video-recent-images",
        "lowres-history-recent-images",
        "image-list",
    }:
        raise ValueError("second-view visual input mode is missing or unsupported")

    rows = []
    for recent_row in recent.get("rows") or []:
        uid = str(recent_row["uid"])
        decision = decisions[uid]
        escalated = bool(decision["escalated"])
        temporal_error_fallback = escalated and uid in temporal_error_uids
        proposal = (
            recent_row
            if temporal_error_fallback or not escalated
            else temporal_rows[uid]
        )
        sampling = proposal.get("sampling") or {}
        if sampling.get("frame_timestamps") and max(sampling["frame_timestamps"]) > float(
            proposal["query_time"]
        ) + 1e-3:
            raise ValueError(f"{uid}: selected inference crosses the causal frontier")
        recent_frames = int(recent_row.get("frames_observed") or 4)
        chosen_frames = int(proposal.get("frames_observed") or recent_frames)
        probe_frames = 0
        extra_probe_frames = 0
        if escalated and not temporal_error_fallback:
            if sampling.get("frame_policy") != second_view_policy:
                raise ValueError(f"{uid}: second-view policy diverged")
            if second_view_policy in {"causalbudget20", "causalbudget32r8"}:
                candidate_indices = sampling.get("candidate_frame_indices")
                if not isinstance(candidate_indices, list) or not candidate_indices:
                    raise ValueError(f"{uid}: temporal row lacks probe-frame audit")
                probe_frames = len(candidate_indices)
                extra_probe_frames = int(sampling.get("extra_frames_inspected", -1))
                if extra_probe_frames != max(0, probe_frames - chosen_frames):
                    raise ValueError(f"{uid}: temporal probe accounting diverged")
            elif second_view_policy == "prefix32" and chosen_frames != 32:
                raise ValueError(f"{uid}: prefix32 row does not expose 32 frames")
            elif second_view_policy in {
                "causalhybrid32",
                "causalhybrid32r8",
            } and not (
                1 <= chosen_frames <= 32
            ):
                raise ValueError(f"{uid}: causalhybrid32* frame budget is invalid")
        recent_prediction = str(recent_row.get("prediction") or "")
        proposal_prediction = str(proposal.get("prediction") or "")
        confidence_gain = None
        committed = escalated and not temporal_error_fallback
        if escalated and not temporal_error_fallback and commit_threshold is not None:
            confidence_gain = choice_margin(proposal) - choice_margin(recent_row)
            committed = bool(
                proposal_prediction != recent_prediction
                and confidence_gain > commit_threshold
            )
        final_prediction = proposal_prediction if committed else recent_prediction
        rows.append(
            {
                **proposal,
                "uid": uid,
                "prediction": final_prediction,
                "current_prediction": recent_prediction,
                "proposal_prediction": proposal_prediction if escalated else None,
                "agent_policy": {
                    "margin": float(decision["margin"]),
                    "threshold": frozen_threshold,
                    "uncertainty_quantile": fixed_quantile,
                    "uncertainty_rank": decision.get("uncertainty_rank"),
                    "escalated": escalated,
                    "temporal_error_fallback": temporal_error_fallback,
                    "temporal_error": (
                        str(temporal_rows[uid].get("error"))
                        if temporal_error_fallback
                        else None
                    ),
                    "commit_threshold": commit_threshold,
                    "confidence_gain": confidence_gain,
                    "committed": committed,
                    "selected_view": (
                        "recent4_temporal_error_fallback"
                        if temporal_error_fallback
                        else selected_view_name(
                            second_view_policy, second_view_mode, escalated=escalated
                        )
                    ),
                    "question_task_group_or_benchmark_router": False,
                },
                "unique_frames": chosen_frames,
                "model_processed_frames": (
                    recent_frames + chosen_frames
                    if escalated and not temporal_error_fallback
                    else recent_frames
                ),
                "low_resolution_probe_frames": probe_frames,
                "extra_probe_frames_not_visible_to_vlm": extra_probe_frames,
                "total_decoded_frame_instances": (
                    recent_frames + probe_frames + chosen_frames
                    if escalated and not temporal_error_fallback
                    else recent_frames
                ),
                "media_calls": 2 if escalated else 1,
                "num_turns": 2 if escalated else 1,
                "latency_seconds": float(recent_row.get("latency_seconds") or 0.0)
                + (
                        float(temporal_rows[uid].get("latency_seconds") or 0.0)
                    if escalated
                    else 0.0
                ),
            }
        )
    if any(not row["prediction"] for row in rows):
        raise ValueError("materialized agent contains an empty prediction")
    escalated_count = len(selected)
    return {
        "format": "streameva.uncertainty_temporal_agent_eval.v0",
        "status": (
            "internally_calibrated_router_free_agent"
            if threshold is not None
            else "fixed_budget_router_free_agent"
        ),
        "benchmark": recent.get("benchmark"),
        "model": recent.get("model"),
        "model_manifest_sha256": recent.get("model_manifest_sha256"),
        "processor": recent.get("processor"),
        "annotation": recent.get("annotation"),
        "annotation_sha256": recent.get("annotation_sha256"),
        "evaluation_protocol": recent.get("evaluation_protocol"),
        "evaluation_protocol_sha256": recent.get("evaluation_protocol_sha256"),
        "comparable_under_locked_recent4_protocol": False,
        "policy": {
            "first_view": "recent4",
            "uncertainty": "first_step_full_vocabulary_top2_margin",
            "threshold": frozen_threshold,
            "uncertainty_quantile": fixed_quantile,
            "commit_threshold": commit_threshold,
            "second_view": second_view_policy_name(
                second_view_policy, second_view_mode
            ),
            "commit_rule": (
                "changed_answer_and_confidence_gain_gt_global_threshold"
                if commit_threshold is not None
                else "legacy_always_accept_temporal_proposal"
            ),
            "question_task_group_or_benchmark_router": False,
            "benchmark_identity_available_to_policy": False,
            "gold_available_to_policy": False,
            "temporal_error_policy": (
                "fail_closed_to_recent4"
                if fallback_recent_on_temporal_error
                else "reject_materialization"
            ),
        },
        "num_samples": len(rows),
        "num_errors": 0,
        "escalated": escalated_count,
        "temporal_input_num_rows": temporal_input_count,
        "temporal_unused_num_rows": temporal_input_count - escalated_count,
        "temporal_error_fallback_count": len(temporal_error_uids),
        "escalation_rate": escalated_count / len(rows),
        "mean_model_calls": 1.0 + escalated_count / len(rows),
        "mean_unique_frames": fmean(row["unique_frames"] for row in rows),
        "mean_model_processed_frames": fmean(
            row["model_processed_frames"] for row in rows
        ),
        "mean_low_resolution_probe_frames": fmean(
            row["low_resolution_probe_frames"] for row in rows
        ),
        "mean_extra_probe_frames_not_visible_to_vlm": fmean(
            row["extra_probe_frames_not_visible_to_vlm"] for row in rows
        ),
        "mean_total_decoded_frame_instances": fmean(
            row["total_decoded_frame_instances"] for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    recent = json.loads(args.recent.read_text(encoding="utf-8"))
    temporal = json.loads(args.temporal.read_text(encoding="utf-8"))
    identity_override_applied = False
    if args.temporal_model_identity_sha256 is not None:
        identity = args.temporal_model_identity_sha256.strip().lower()
        if len(identity) != 64 or any(
            character not in "0123456789abcdef" for character in identity
        ):
            raise ValueError("temporal model identity must be a 64-character SHA-256")
        embedded_identity = temporal.get("model_manifest_sha256")
        if embedded_identity not in (None, "", identity):
            raise ValueError("temporal model identity override conflicts with artifact")
        if embedded_identity in (None, ""):
            temporal["model_manifest_sha256"] = identity
            identity_override_applied = True
    selection = json.loads(args.selection_audit.read_text(encoding="utf-8"))
    threshold = (
        json.loads(args.threshold_audit.read_text(encoding="utf-8"))
        if args.threshold_audit is not None
        else None
    )
    output = materialize(
        recent,
        temporal,
        selection,
        threshold,
        commit_threshold_override=args.commit_threshold,
        fallback_recent_on_temporal_error=args.fallback_recent_on_temporal_error,
    )
    output["artifacts"] = {
        "recent": {"path": str(args.recent.resolve())},
        "temporal": {"path": str(args.temporal.resolve())},
        "selection_audit": {"path": str(args.selection_audit.resolve())},
    }
    if args.threshold_audit is not None:
        output["artifacts"]["threshold_audit"] = {
            "path": str(args.threshold_audit.resolve()),
        }
    output["artifact_checksums_skipped"] = True
    output["artifact_checksum_skip_reason"] = (
        "native artifacts succeeded and identities are embedded in their manifests"
    )
    output["temporal_model_identity_override_applied"] = identity_override_applied
    atomic_write(args.output, output)
    print(
        json.dumps(
            {
                "benchmark": output["benchmark"],
                "num_samples": output["num_samples"],
                "escalated": output["escalated"],
                "mean_model_calls": output["mean_model_calls"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
