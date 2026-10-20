#!/usr/bin/env python3
"""Freeze a source-disjoint holdout and a quality-ranked capability selection.

This selector never reads benchmark rows, labels, predictions, task IDs, or
question templates.  It joins an already sealed non-benchmark teacher release
back to its generation evidence, removes nominal capability rows whose answer
was available too early, and reserves a policy holdout before selecting SFT
rows.  A final training selection additionally requires a complete student
screen over the structural candidate set.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from viss_core.data_v0.canonical import (
    canonical_json_line,
    file_sha256,
    read_jsonl,
    stable_id,
)
from viss_core.data_v0.semantic_release import validate_semantic_release


SELECTION_FORMAT = "iclr_video.quality_capability_selection.v1"
HOLDOUT_FORMAT = "iclr_video.nonbenchmark_policy_holdout.v1"
STUDENT_SCREEN_FORMAT = "iclr_video.nonbenchmark_student_screen.v1"
PROFILE_KIND = "nonbenchmark_capability_profile"
PARENT_KIND = "teacher_generated_nonbenchmark_visual_mcq"
PROFILES = (
    "backward_epm",
    "backward_asi",
    "rec_counting",
    "sequence_crr",
    "sequence_ssr",
)
DEFAULT_HOLDOUT = {profile: 60 for profile in PROFILES}
DEFAULT_TRAIN = {"backward_epm": 120, "backward_asi": 180}


class QualityCapabilitySelectionError(ValueError):
    pass


def _target(row: dict[str, Any]) -> str:
    values = [
        str(item.get("text") or "").strip()
        for message in row.get("messages") or []
        if message.get("role") == "assistant" and message.get("loss") is True
        for item in message.get("content") or []
        if item.get("type") == "text"
    ]
    if len(values) != 1 or values[0] not in set("ABCD"):
        raise QualityCapabilitySelectionError("row lacks one A-D target")
    return values[0]


def _single_parent(row: dict[str, Any], kind: str) -> str:
    values = [
        str(parent.get("id") or "")
        for parent in row.get("parents") or []
        if parent.get("kind") == kind
    ]
    if len(values) != 1 or not values[0]:
        raise QualityCapabilitySelectionError(f"row lacks one {kind} parent")
    return values[0]


def _rank(seed: str, profile: str, answer: str, uid: str) -> str:
    return hashlib.sha256(
        f"quality-capability-v1:{seed}:{profile}:{answer}:{uid}".encode()
    ).hexdigest()


def _balanced_select(
    rows: Iterable[dict[str, Any]],
    *,
    quota: int,
    seed: str,
    profile: str,
) -> list[dict[str, Any]]:
    if quota <= 0 or quota % 4:
        raise QualityCapabilitySelectionError("balanced quota must be positive / 4")
    by_answer: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_answer[str(row["answer"])].append(row)
    per_answer = quota // 4
    selected = []
    for answer in "ABCD":
        values = sorted(
            by_answer.get(answer, []),
            key=lambda row: (_rank(seed, profile, answer, str(row["uid"])), str(row["uid"])),
        )
        if len(values) < per_answer:
            raise QualityCapabilitySelectionError(
                f"{profile}/{answer} has {len(values)} rows; requires {per_answer}"
            )
        selected.extend(values[:per_answer])
    return sorted(selected, key=lambda row: str(row["uid"]))


def _image_paths(request: dict[str, Any]) -> list[str]:
    return [
        str(image)
        for group in request.get("groups") or []
        for image in group.get("images") or []
    ]


def _structural_candidate(
    request: dict[str, Any], generation: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    profile = str(request.get("target_capability") or "")
    groups = [
        tuple(str(image) for image in group.get("images") or [])
        for group in request.get("groups") or []
    ]
    flattened = [image for group in groups for image in group]
    evidence_groups = generation.get("evidence_groups") or []
    first_sufficient = generation.get("first_sufficient_group")
    minimum_sufficient = {
        "backward_epm": 1,
        "backward_asi": 2,
        "rec_counting": 1,
        "sequence_crr": 1,
        "sequence_ssr": 1,
    }.get(profile)
    diagnostics = {
        "first_sufficient_group": first_sufficient,
        "evidence_groups": len(evidence_groups),
        "observation_groups": len(groups),
        "distinct_observation_groups": len(set(groups)),
        "image_positions": len(flattened),
        "unique_images": len(set(flattened)),
        "duplicate_image_positions": len(flattened) - len(set(flattened)),
    }
    eligible = (
        minimum_sufficient is not None
        and generation.get("decision") == "accept"
        and generation.get("confidence") == "high"
        and isinstance(first_sufficient, int)
        and first_sufficient >= minimum_sufficient
        and len(evidence_groups) >= 2
        and len(set(groups)) >= 2
        and len(set(flattened)) >= 6
    )
    return bool(eligible), diagnostics


def _student_keep(row: dict[str, Any], audit: dict[str, Any]) -> bool:
    if audit.get("error") is not None:
        return False
    target = str(row["answer"])
    if str(audit.get("target") or "") != target:
        raise QualityCapabilitySelectionError("student audit target diverges")
    prediction = str(audit.get("prediction") or "")
    margin = float(audit.get("visible_option_margin"))
    if not 0.0 <= margin <= 1.0:
        raise QualityCapabilitySelectionError("student audit margin is outside [0,1]")
    return prediction != target or margin <= 0.25


def _parse_quota(value: str) -> tuple[str, int]:
    try:
        profile, raw = value.rsplit("=", 1)
        count = int(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("quota must be PROFILE=COUNT") from exc
    if profile not in PROFILES or count <= 0 or count % 4:
        raise argparse.ArgumentTypeError("quota must use a known profile and be / 4")
    return profile, count


def _excluded_parent_ids(releases: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for raw in releases:
        root = raw.resolve(strict=True)
        validate_semantic_release(root, verify_media_hashes=False)
        for row in read_jsonl(
            root / "trainer/sft.jsonl",
            verify_content_hashes=True,
            require_canonical=True,
        ):
            excluded.update(
                str(parent["id"])
                for parent in row.get("parents") or []
                if parent.get("kind") == PARENT_KIND and parent.get("id")
            )
    return excluded


def build(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    release = args.release.resolve(strict=True)
    pack = args.pack.resolve(strict=True)
    generation_path = args.generation.resolve(strict=True)
    release_manifest = validate_semantic_release(release, verify_media_hashes=False)
    pack_manifest_path = pack / "manifest.json"
    pack_manifest = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    if (
        (release_manifest.get("inputs") or {}).get("pack_manifest_sha256")
        != file_sha256(pack_manifest_path)
        or (release_manifest.get("inputs") or {}).get("generation_sha256")
        != file_sha256(generation_path)
    ):
        raise QualityCapabilitySelectionError("release is not bound to pack/generation")
    requests = read_jsonl(pack / "requests.jsonl", require_canonical=True)
    generation_rows = read_jsonl(generation_path, require_canonical=True)
    trainer_rows = read_jsonl(
        release / "trainer/sft.jsonl",
        verify_content_hashes=True,
        require_canonical=True,
    )
    request_by_uid = {str(row["uid"]): row for row in requests}
    generation_by_uid = {str(row["uid"]): row for row in generation_rows}
    trainer_by_uid = {str(row["uid"]): row for row in trainer_rows}
    if len(request_by_uid) != len(requests) or len(generation_by_uid) != len(generation_rows):
        raise QualityCapabilitySelectionError("request/generation UIDs collide")

    excluded = _excluded_parent_ids(list(args.exclude_release or []))
    joined = []
    for request_uid, request in request_by_uid.items():
        trainer_uid = stable_id("benchmarkdistillsft", request_uid, digest_length=24)
        trainer = trainer_by_uid.get(trainer_uid)
        generation = generation_by_uid.get(request_uid)
        if trainer is None:
            continue
        if generation is None:
            raise QualityCapabilitySelectionError("filtered row lacks generation record")
        parent_id = _single_parent(trainer, PARENT_KIND)
        profile = _single_parent(trainer, PROFILE_KIND)
        if profile != str(request.get("target_capability") or ""):
            raise QualityCapabilitySelectionError("request/trainer profile diverges")
        if parent_id in excluded:
            continue
        joined.append(
            {
                "uid": trainer_uid,
                "request_uid": request_uid,
                "parent_id": parent_id,
                "profile": profile,
                "answer": _target(trainer),
                "source_video_basename": str(request["source_video_basename"]),
                "request": request,
                "generation": generation,
            }
        )

    holdout_quotas = dict(args.holdout_quota or DEFAULT_HOLDOUT)
    train_quotas = dict(args.train_quota or DEFAULT_TRAIN)
    holdout_rows = []
    for profile, quota in sorted(holdout_quotas.items()):
        holdout_rows.extend(
            _balanced_select(
                (row for row in joined if row["profile"] == profile),
                quota=quota,
                seed=args.seed + ":holdout",
                profile=profile,
            )
        )
    holdout_uids = {str(row["uid"]) for row in holdout_rows}
    holdout_videos = {str(row["source_video_basename"]) for row in holdout_rows}
    if len(holdout_uids) != len(holdout_rows) or len(holdout_videos) != len(holdout_rows):
        raise QualityCapabilitySelectionError("policy holdout is not source-video unique")

    structural = []
    for row in joined:
        if row["uid"] in holdout_uids:
            continue
        eligible, diagnostics = _structural_candidate(
            row["request"], row["generation"]
        )
        if eligible:
            structural.append({**row, "diagnostics": diagnostics})

    audit_by_uid: dict[str, dict[str, Any]] | None = None
    audit_record = None
    if args.student_audit is not None:
        audit_path = args.student_audit.resolve(strict=True)
        audit_value = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit_value.get("format") != STUDENT_SCREEN_FORMAT
            or audit_value.get("benchmark_rows_or_labels_used") is not False
            or audit_value.get("source_release_manifest_sha256")
            != file_sha256(release / "manifest.json")
        ):
            raise QualityCapabilitySelectionError("student screen contract diverges")
        audit_rows = list(audit_value.get("rows") or [])
        audit_by_uid = {str(row.get("uid") or ""): row for row in audit_rows}
        if set(audit_by_uid) != {str(row["uid"]) for row in structural}:
            raise QualityCapabilitySelectionError("student screen coverage diverges")
        audit_record = {"path": str(audit_path), "sha256": file_sha256(audit_path)}

    selected_rows = []
    if audit_by_uid is not None:
        hard = [row for row in structural if _student_keep(row, audit_by_uid[row["uid"]])]
        for profile, quota in sorted(train_quotas.items()):
            selected_rows.extend(
                _balanced_select(
                    (row for row in hard if row["profile"] == profile),
                    quota=quota,
                    seed=args.seed + ":train",
                    profile=profile,
                )
            )

    def public_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: row[key]
            for key in (
                "uid",
                "request_uid",
                "parent_id",
                "profile",
                "answer",
                "source_video_basename",
            )
        } | ({"diagnostics": row["diagnostics"]} if "diagnostics" in row else {})

    holdout = {
        "format": HOLDOUT_FORMAT,
        "status": "sealed_before_training_selection",
        "source_release": str(release),
        "source_release_manifest_sha256": file_sha256(release / "manifest.json"),
        "selection_seed": args.seed + ":holdout",
        "quotas": holdout_quotas,
        "rows": [public_row(row) for row in sorted(holdout_rows, key=lambda row: row["uid"])],
        "parent_ids": sorted(row["parent_id"] for row in holdout_rows),
        "contracts": {
            "source_video_disjoint_from_sft_selection": True,
            "may_enter_sft_or_rl": False,
            "benchmark_rows_or_labels_used": False,
        },
    }
    selection = {
        "format": SELECTION_FORMAT,
        "status": (
            "sealed_nonbenchmark_quality_selection"
            if audit_by_uid is not None
            else "awaiting_complete_student_screen"
        ),
        "training_allowed": audit_by_uid is not None,
        "source_release": str(release),
        "source_release_manifest_sha256": file_sha256(release / "manifest.json"),
        "pack_manifest_sha256": file_sha256(pack_manifest_path),
        "generation_sha256": file_sha256(generation_path),
        "excluded_parent_ids": len(excluded),
        "holdout_parent_ids_sha256": hashlib.sha256(
            canonical_json_line(sorted(holdout["parent_ids"]))
        ).hexdigest(),
        "structural_candidates": [
            public_row(row) for row in sorted(structural, key=lambda row: row["uid"])
        ],
        "structural_counts_by_profile": dict(
            sorted(collections.Counter(row["profile"] for row in structural).items())
        ),
        "student_screen": audit_record,
        "train_quotas": train_quotas,
        "selected_rows": [
            public_row(row) for row in sorted(selected_rows, key=lambda row: row["uid"])
        ],
        "parent_ids": sorted(row["parent_id"] for row in selected_rows),
        "transformation": "deduplicate_identical_images_keep_first_chronological_occurrence",
        "contracts": {
            "benchmark_rows_labels_predictions_or_tasks_used": False,
            "counterfactual_rows": 0,
            "holdout_overlap": 0,
            "already_trained_parent_overlap": 0,
            "teacher_confidence_high": True,
            "two_order_option_consensus_inherited": True,
        },
    }
    return holdout, selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--exclude-release", type=Path, action="append", default=[])
    parser.add_argument("--student-audit", type=Path)
    parser.add_argument("--holdout-quota", type=_parse_quota, action="append")
    parser.add_argument("--train-quota", type=_parse_quota, action="append")
    parser.add_argument("--seed", default="quality-capability-v1")
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    args = parser.parse_args()
    if args.holdout_quota:
        args.holdout_quota = dict(args.holdout_quota)
    if args.train_quota:
        args.train_quota = dict(args.train_quota)
    for path in (args.holdout_output, args.selection_output):
        if path.exists():
            raise FileExistsError(path)
    holdout, selection = build(args)
    args.holdout_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.holdout_output.write_bytes(canonical_json_line(holdout))
    args.selection_output.write_bytes(canonical_json_line(selection))
    print(
        json.dumps(
            {
                "holdout_rows": len(holdout["rows"]),
                "structural_counts_by_profile": selection["structural_counts_by_profile"],
                "selected_rows": len(selection["selected_rows"]),
                "training_allowed": selection["training_allowed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
