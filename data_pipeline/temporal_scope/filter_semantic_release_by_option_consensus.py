#!/usr/bin/env python3
"""Seal source-labelled rows that survive an option-invariant visual audit.

The two-presentation auditor never receives the source answer.  This builder
also never treats an auditor prediction as a replacement label: a source row
is eligible only when both option orders independently recover the unchanged
source answer with an allowed confidence level.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from data_pipeline.temporal_scope.filter_semantic_release_by_blind_audit import (
    BlindAuditFilterError,
    _answer_balanced_subset,
    _images,
    _materialize,
    _target,
)
from viss_core.data_v0.canonical import (
    canonical_json_line,
    file_sha256,
    read_jsonl,
    write_jsonl,
)
from viss_core.data_v0.semantic_release import FORMAT, validate_semantic_release


AUDIT_ROW_FORMAT = "streameva.semantic_release.option_consensus_audit_row.v0"
AUDIT_SUMMARY_FORMAT = "streameva.semantic_release.option_consensus_audit.v0"
FILTER_FORMAT = "streameva.semantic_release.option_consensus_filter.v0"


def _validate_presentations(
    *,
    source_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    allowed_confidence: set[str],
) -> tuple[list[dict[str, Any]], int]:
    rows_by_uid = {str(row["uid"]): row for row in source_rows}
    if len(rows_by_uid) != len(source_rows):
        raise BlindAuditFilterError("source UIDs are not unique")
    presentations: dict[str, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
    for item in audit_rows:
        uid = str(item.get("uid") or "")
        presentation = item.get("presentation")
        if (
            item.get("format") != AUDIT_ROW_FORMAT
            or uid not in rows_by_uid
            or presentation not in {0, 1}
            or presentation in presentations[uid]
        ):
            raise BlindAuditFilterError("option-consensus audit identity diverged")
        expected = _target(rows_by_uid[uid])
        prediction = item.get("prediction_source")
        mapping = item.get("displayed_to_source")
        if (
            item.get("expected_source") != expected
            or item.get("error") is not None
            or item.get("confidence") not in {"high", "medium", "low"}
            or item.get("correct_source") is not (prediction == expected)
            or not isinstance(mapping, list)
            or sorted(mapping) != list("ABCD")
            or (presentation == 0 and mapping != list("ABCD"))
            or (presentation == 1 and mapping == list("ABCD"))
        ):
            raise BlindAuditFilterError("option-consensus audit decision is malformed")
        presentations[uid][int(presentation)] = item
    if set(presentations) != set(rows_by_uid) or any(
        set(values) != {0, 1} for values in presentations.values()
    ):
        raise BlindAuditFilterError("option-consensus audit lacks complete two-order coverage")

    stable_rows = 0
    selected: list[dict[str, Any]] = []
    for uid, source_row in rows_by_uid.items():
        pair = [presentations[uid][0], presentations[uid][1]]
        predictions = {str(item["prediction_source"]) for item in pair}
        if (
            len(predictions) == 1
            and all(str(item["confidence"]) in allowed_confidence for item in pair)
        ):
            stable_rows += 1
            if next(iter(predictions)) == _target(source_row):
                selected.append(source_row)
    selected.sort(key=lambda row: str(row["uid"]))
    return selected, stable_rows


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
    if not 0.0 <= args.min_consensus_source_accuracy <= 1.0:
        raise BlindAuditFilterError("minimum consensus accuracy must lie in [0, 1]")
    if args.min_rows <= 0:
        raise BlindAuditFilterError("minimum retained rows must be positive")

    source_manifest = validate_semantic_release(source, verify_media_hashes=False)
    source_trainer = source / "trainer/sft.jsonl"
    source_rows = read_jsonl(
        source_trainer, verify_content_hashes=True, require_canonical=True
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required_summary = {
        "format": AUDIT_SUMMARY_FORMAT,
        "status": "complete_option_consensus_audit",
        "release_id": source_manifest["release_id"],
        "release_manifest_sha256": file_sha256(source / "manifest.json"),
        "trainer_sha256": file_sha256(source_trainer),
        "sample_rows": len(source_rows),
        "presentations": 2 * len(source_rows),
        "errors": 0,
        "expected_answers_sent_to_auditor": False,
        "auditor_outputs_admitted_to_training": False,
        "benchmark_rows_used": 0,
        "result_rows_sha256": file_sha256(audit_path),
    }
    if any(summary.get(key) != value for key, value in required_summary.items()):
        raise BlindAuditFilterError("consensus audit is partial, unsafe, or unbound")
    request_contract = summary.get("request_contract") or {}
    if (
        request_contract.get("presentations_per_row") != 2
        or request_contract.get("second_option_order_non_identity") is not True
        or request_contract.get("temperature") != 0
    ):
        raise BlindAuditFilterError("consensus request contract diverged")

    audit_rows = read_jsonl(audit_path, require_canonical=True)
    selected, stable_rows = _validate_presentations(
        source_rows=source_rows,
        audit_rows=audit_rows,
        allowed_confidence=allowed_confidence,
    )
    summary_stable = int(summary.get("high_confidence_option_consensus_rows", -1))
    if allowed_confidence == {"high"} and stable_rows != summary_stable:
        raise BlindAuditFilterError("auditor summary stable-row count diverged")
    source_agreements = len(selected)
    consensus_accuracy = source_agreements / stable_rows if stable_rows else 0.0
    if consensus_accuracy < args.min_consensus_source_accuracy:
        raise BlindAuditFilterError(
            f"stable consensus source agreement {consensus_accuracy:.4f} is below "
            f"{args.min_consensus_source_accuracy:.4f}"
        )
    eligible_rows = len(selected)
    if args.max_rows is not None:
        selected = _answer_balanced_subset(
            selected, count=int(args.max_rows), seed=args.selection_seed
        )
    if len(selected) < args.min_rows:
        raise BlindAuditFilterError(
            f"only {len(selected)} option-invariant source agreements survived; "
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
            destination = scratch / relative
            _materialize(source / relative, destination, mode=args.copy_mode)
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
                "frames_per_row": {
                    str(key): value for key, value in sorted(frames.items())
                },
            },
            "format": FORMAT,
            "inputs": {
                **source_manifest["inputs"],
                "option_consensus_filter": {
                    "format": FILTER_FORMAT,
                    "source_release_id": source_manifest["release_id"],
                    "source_manifest_sha256": file_sha256(source / "manifest.json"),
                    "source_trainer_sha256": file_sha256(source_trainer),
                    "audit_sha256": file_sha256(audit_path),
                    "audit_summary_sha256": file_sha256(summary_path),
                    "auditor_model": str(summary.get("auditor_model") or ""),
                    "auditor_received_expected_answers": False,
                    "auditor_predictions_used_as_labels": False,
                    "presentations_per_row": 2,
                    "option_invariance_required": True,
                    "source_agreement_required": True,
                    "stable_consensus_rows": stable_rows,
                    "stable_consensus_source_agreements": source_agreements,
                    "stable_consensus_source_accuracy": consensus_accuracy,
                    "eligible_rows_after_filter": eligible_rows,
                    "allowed_confidence": sorted(allowed_confidence),
                    "max_rows": args.max_rows,
                    "subset_selection_seed": (
                        args.selection_seed if args.max_rows is not None else None
                    ),
                    "filter_builder_sha256": file_sha256(Path(__file__).resolve(strict=True)),
                    "shared_filter_helpers_sha256": file_sha256(
                        Path(__file__)
                        .with_name("filter_semantic_release_by_blind_audit.py")
                        .resolve(strict=True)
                    ),
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
    parser.add_argument("--min-consensus-source-accuracy", type=float, default=0.80)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--selection-seed", default="option-consensus-filter-v0")
    parser.add_argument(
        "--copy-mode", choices=("copy", "hardlink", "auto"), default="hardlink"
    )
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
