#!/usr/bin/env python3
"""Seal the high-confidence agreement subset of a blind-audited SFT release.

The visual auditor never receives the source answer.  This builder also never
uses the auditor prediction as a new label: it only admits an existing source
row when a complete, hash-bound audit independently recovered that row's
answer.  Partial audits are deliberately ineligible for materialization.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from viss_core.data_v0.canonical import (
    canonical_json_line,
    file_sha256,
    read_jsonl,
    write_jsonl,
)
from viss_core.data_v0.semantic_release import (
    FORMAT,
    validate_semantic_release,
)


AUDIT_ROW_FORMAT = "streameva.semantic_release.blind_visual_audit_row.v0"
AUDIT_SUMMARY_FORMAT = "streameva.semantic_release.blind_visual_audit.v0"
FILTER_FORMAT = "streameva.semantic_release.blind_visual_filter.v0"


class BlindAuditFilterError(ValueError):
    """Raised when an audit is incomplete or not independent of its labels."""


def _target(row: dict[str, Any]) -> str:
    values = [
        str(item.get("text") or "")
        for message in row.get("messages") or []
        if message.get("role") == "assistant" and message.get("loss") is True
        for item in message.get("content") or []
        if item.get("type") == "text"
    ]
    if len(values) != 1:
        raise BlindAuditFilterError("semantic row has no unique target")
    return values[0]


def _images(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(item["image"])
        for row in rows
        for message in row["messages"]
        for item in message["content"]
        if item.get("type") == "image"
    }


def _answer_balanced_subset(
    rows: list[dict[str, Any]], *, count: int, seed: str
) -> list[dict[str, Any]]:
    if count <= 0 or count > len(rows):
        raise BlindAuditFilterError("subset row count exceeds audited agreements")
    by_answer: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_answer[_target(row)].append(row)
    if set(by_answer) != set("ABCD"):
        raise BlindAuditFilterError("answer-balanced subset requires A/B/C/D rows")
    base, remainder = divmod(count, 4)
    selected: list[dict[str, Any]] = []
    for index, answer in enumerate("ABCD"):
        quota = base + int(index < remainder)
        ranked = sorted(
            by_answer[answer],
            key=lambda row: hashlib.sha256(
                f"blind-filter-subset:{seed}:{row['uid']}".encode()
            ).hexdigest(),
        )
        if len(ranked) < quota:
            raise BlindAuditFilterError(
                f"answer {answer} has {len(ranked)} rows but needs {quota}"
            )
        selected.extend(ranked[:quota])
    return sorted(selected, key=lambda row: str(row["uid"]))


def _materialize(source: Path, destination: Path, *, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink() or not source.is_file():
        raise BlindAuditFilterError(f"unsafe source media: {source}")
    if mode in {"hardlink", "auto"}:
        try:
            os.link(source, destination)
            return
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, destination)


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve(strict=True)
    audit_path = args.audit.resolve(strict=True)
    summary_path = args.audit_summary.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.copy_mode not in {"copy", "hardlink", "auto"}:
        raise BlindAuditFilterError("unsupported copy mode")
    allowed_confidence = set(args.confidence)
    if not allowed_confidence or not allowed_confidence <= {"high", "medium"}:
        raise BlindAuditFilterError("confidence must be high and/or medium")
    if not 0.0 <= args.min_source_accuracy <= 1.0:
        raise BlindAuditFilterError("minimum source accuracy must lie in [0, 1]")
    if args.min_rows <= 0:
        raise BlindAuditFilterError("minimum retained rows must be positive")

    source_manifest = validate_semantic_release(source, verify_media_hashes=False)
    source_trainer = source / "trainer/sft.jsonl"
    source_rows = read_jsonl(
        source_trainer, verify_content_hashes=True, require_canonical=True
    )
    rows_by_uid = {str(row["uid"]): row for row in source_rows}
    if len(rows_by_uid) != len(source_rows):
        raise BlindAuditFilterError("source UIDs are not unique")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required_summary = {
        "format": AUDIT_SUMMARY_FORMAT,
        "status": "admit_semantic_release",
        "admitted": True,
        "release_id": source_manifest["release_id"],
        "release_manifest_sha256": file_sha256(source / "manifest.json"),
        "trainer_sha256": file_sha256(source_trainer),
        "sample_rows": len(source_rows),
        "valid_rows": len(source_rows),
        "errors": 0,
        "expected_answers_sent_to_auditor": False,
        "auditor_outputs_admitted_to_training": False,
        "benchmark_rows_used": 0,
        "result_rows_sha256": file_sha256(audit_path),
    }
    if any(summary.get(key) != value for key, value in required_summary.items()):
        raise BlindAuditFilterError("audit summary is partial, rejected, or unbound")
    accuracy = float(summary.get("accuracy", -1.0))
    if accuracy < args.min_source_accuracy:
        raise BlindAuditFilterError(
            f"blind source agreement {accuracy:.4f} is below "
            f"{args.min_source_accuracy:.4f}"
        )

    audit_rows = read_jsonl(audit_path, require_canonical=True)
    audit_by_uid: dict[str, dict[str, Any]] = {}
    for item in audit_rows:
        uid = str(item.get("uid") or "")
        if (
            item.get("format") != AUDIT_ROW_FORMAT
            or not uid
            or uid in audit_by_uid
            or uid not in rows_by_uid
        ):
            raise BlindAuditFilterError("audit row identity diverged")
        expected = _target(rows_by_uid[uid])
        prediction = item.get("prediction")
        if (
            item.get("expected") != expected
            or item.get("error") is not None
            or item.get("confidence") not in {"high", "medium", "low"}
            or item.get("correct") is not (prediction == expected)
        ):
            raise BlindAuditFilterError("audit decision is malformed")
        audit_by_uid[uid] = item
    if set(audit_by_uid) != set(rows_by_uid):
        raise BlindAuditFilterError("blind audit does not cover every source row")

    selected = [
        row
        for uid, row in rows_by_uid.items()
        if audit_by_uid[uid]["correct"] is True
        and audit_by_uid[uid]["confidence"] in allowed_confidence
    ]
    selected.sort(key=lambda row: str(row["uid"]))
    eligible_rows = len(selected)
    max_rows = getattr(args, "max_rows", None)
    selection_seed = str(getattr(args, "selection_seed", "blind-filter-subset-v0"))
    if max_rows is not None:
        selected = _answer_balanced_subset(
            selected, count=int(max_rows), seed=selection_seed
        )
    if len(selected) < args.min_rows:
        raise BlindAuditFilterError(
            f"only {len(selected)} high-confidence rows survived; "
            f"requires {args.min_rows}"
        )

    source_inventory_path = source / "manifests/media_inventory.jsonl"
    source_inventory = {
        str(item["path"]): item
        for item in read_jsonl(source_inventory_path, require_canonical=True)
    }
    referenced = _images(selected)
    if not referenced <= set(source_inventory):
        raise BlindAuditFilterError("selected media is absent from source inventory")

    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial.", dir=output.parent))
    try:
        write_jsonl(selected, scratch / "trainer/sft.jsonl")
        selected_inventory = [source_inventory[path] for path in sorted(referenced)]
        write_jsonl(selected_inventory, scratch / "manifests/media_inventory.jsonl")
        for item in selected_inventory:
            relative = Path(str(item["path"]))
            source_file = source / relative
            destination = scratch / relative
            _materialize(source_file, destination, mode=args.copy_mode)
            if (
                destination.stat().st_size != int(item["bytes"])
                or file_sha256(destination) != item["sha256"]
            ):
                raise BlindAuditFilterError("materialized media hash diverged")

        answers = collections.Counter(_target(row) for row in selected)
        frames = collections.Counter(
            sum(
                item.get("type") == "image"
                for message in row["messages"]
                for item in message["content"]
            )
            for row in selected
        )
        confidence = collections.Counter(
            str(audit_by_uid[str(row["uid"])]["confidence"]) for row in selected
        )
        manifest = {
            "artifacts": {
                "manifests/media_inventory.jsonl": file_sha256(
                    scratch / "manifests/media_inventory.jsonl"
                ),
                "trainer/sft.jsonl": file_sha256(scratch / "trainer/sft.jsonl"),
            },
            "contracts": source_manifest["contracts"],
            "counts": {
                "media_bytes": sum(int(item["bytes"]) for item in selected_inventory),
                "media_files": len(selected_inventory),
                "trainer_rows": len(selected),
            },
            "distributions": {
                "answer": dict(sorted(answers.items())),
                "blind_auditor_confidence": dict(sorted(confidence.items())),
                "frames_per_row": {
                    str(key): value for key, value in sorted(frames.items())
                },
            },
            "format": FORMAT,
            "inputs": {
                **source_manifest["inputs"],
                "blind_visual_filter": {
                    "format": FILTER_FORMAT,
                    "source_release_id": source_manifest["release_id"],
                    "source_manifest_sha256": file_sha256(source / "manifest.json"),
                    "source_trainer_sha256": file_sha256(source_trainer),
                    "audit_sha256": file_sha256(audit_path),
                    "audit_summary_sha256": file_sha256(summary_path),
                    "auditor_model": str(summary.get("auditor_model") or ""),
                    "auditor_received_expected_answers": False,
                    "auditor_predictions_used_as_labels": False,
                    "complete_source_coverage_required": True,
                    "eligible_rows_after_audit_filter": eligible_rows,
                    "filter_builder_sha256": file_sha256(Path(__file__).resolve(strict=True)),
                    "max_rows": max_rows,
                    "source_agreement_accuracy": accuracy,
                    "subset_selection_seed": selection_seed if max_rows is not None else None,
                    "subset_selection_strategy": (
                        "answer_balanced_stable_hash" if max_rows is not None else "all"
                    ),
                    "allowed_confidence": sorted(allowed_confidence),
                    "source_rows": len(source_rows),
                    "selected_rows": len(selected),
                },
            },
            "license": source_manifest["license"],
            "model_lineage": source_manifest["model_lineage"],
            "public_release_allowed": False,
            "release_id": args.release_id,
            "status": "sealed_internal_training_release",
            "training_allowed": True,
            "training_scope": "pure_vlm_semantic_sft_only",
        }
        (scratch / "manifest.json").write_bytes(canonical_json_line(manifest))
        validate_semantic_release(scratch, verify_media_hashes=True)
        os.replace(scratch, output)
        return manifest
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument(
        "--confidence", action="append", choices=("high", "medium"), default=[]
    )
    parser.add_argument("--min-source-accuracy", type=float, default=0.90)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--selection-seed", default="blind-filter-subset-v0")
    parser.add_argument(
        "--copy-mode", choices=("copy", "hardlink", "auto"), default="hardlink"
    )
    args = parser.parse_args()
    if not args.confidence:
        args.confidence = ["high"]
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
