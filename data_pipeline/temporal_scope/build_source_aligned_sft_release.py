#!/usr/bin/env python3
"""Compose one source-aligned, benchmark-firewalled pure-VLM SFT release.

The inputs are already sealed semantic releases built from non-benchmark
training videos.  This compiler keeps the original semantic labels, removes
overlap by cited source identity and exact pixel/question/answer identity,
canonicalizes media paths by SHA256, and emits the only JSONL consumed by one
joint SFT run.  Benchmark examples and auditor predictions are never labels.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

from viss_core.data_v0.canonical import (
    attach_content_sha256,
    canonical_json_bytes,
    canonical_json_line,
    file_sha256,
    read_jsonl,
    stable_id,
    write_jsonl,
)
from viss_core.data_v0.schema import validate_row
from viss_core.data_v0.semantic_release import (
    EXPECTED_PUBLIC_UPSTREAM_MANIFEST_SHA256,
    FORMAT,
    validate_semantic_release,
)


class SourceAlignedSFTError(ValueError):
    """Raised when source releases cannot be composed without leakage."""


def _target(row: dict[str, Any]) -> str:
    values = [
        str(item.get("text") or "")
        for message in row.get("messages") or []
        if message.get("role") == "assistant" and message.get("loss") is True
        for item in message.get("content") or []
        if item.get("type") == "text"
    ]
    if len(values) != 1:
        raise SourceAlignedSFTError("semantic row has no unique target")
    return values[0]


def _question(row: dict[str, Any]) -> str:
    values = [
        str(item.get("text") or "")
        for message in row.get("messages") or []
        if message.get("role") == "user"
        for item in message.get("content") or []
        if item.get("type") == "text"
    ]
    if len(values) != 1:
        raise SourceAlignedSFTError("semantic row has no unique public prompt")
    return " ".join(values[0].split())


def _images(row: dict[str, Any]) -> list[str]:
    return [
        str(item["image"])
        for message in row.get("messages") or []
        for item in message.get("content") or []
        if item.get("type") == "image"
    ]


def _cited_parents(row: dict[str, Any]) -> set[str]:
    return {
        str(parent["id"])
        for parent in row.get("parents") or []
        if parent.get("kind") == "cited_nonbenchmark_refresh_projection"
        and parent.get("id")
    }


def _semantic_identity(
    row: dict[str, Any], inventory: dict[str, dict[str, Any]]
) -> str:
    try:
        image_sha256s = [str(inventory[path]["sha256"]) for path in _images(row)]
    except KeyError as exc:
        raise SourceAlignedSFTError("row references undeclared source media") from exc
    payload = {
        "answer": _target(row),
        "image_sha256s": image_sha256s,
        "question": _question(row),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _canonical_media_path(relative: str, digest: str) -> str:
    suffix = Path(relative).suffix.casefold()
    if not suffix or len(suffix) > 10:
        raise SourceAlignedSFTError("source media suffix is invalid")
    return (Path("media/sha256") / digest[:2] / f"{digest}{suffix}").as_posix()


def _materialize(source: Path, destination: Path, *, mode: str) -> None:
    if source.is_symlink() or not source.is_file():
        raise SourceAlignedSFTError(f"unsafe source media: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode in {"hardlink", "auto"}:
        try:
            os.link(source, destination)
            return
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, destination)


def _lineage_key(manifest: dict[str, Any]) -> tuple[str, str, str]:
    lineage = manifest.get("model_lineage") or {}
    return (
        str(lineage.get("required_sha256s_sha256") or ""),
        str(lineage.get("upstream_repo") or ""),
        str(lineage.get("upstream_revision") or ""),
    )


def _teacher_used_for_labels(manifest: dict[str, Any]) -> bool:
    value = (manifest.get("contracts") or {}).get("teacher_used_for_labels")
    if not isinstance(value, bool):
        raise SourceAlignedSFTError(
            "source teacher-label provenance is absent or malformed"
        )
    return value


def _load_legacy_v62di_source(root: Path, *, mode: str = "v62di_only"):
    raise SourceAlignedSFTError("Private legacy adapters are not distributed; supply a standard source release")


def _load_legacy_repcount_source(root: Path):
    raise SourceAlignedSFTError("Private legacy adapters are not distributed; supply a standard source release")


def build(args: argparse.Namespace) -> dict[str, Any]:
    sealed_source_count = (
        len(args.source)
        + int(getattr(args, "legacy_v62di_source", None) is not None)
        + int(getattr(args, "legacy_repcount_source", None) is not None)
    )
    # A hash-bound parent selection is also a legitimate one-source filtering
    # operation.  Unfiltered releases remain compositions and therefore still
    # require at least two sealed inputs.
    if sealed_source_count < 2 and getattr(args, "include_parent_selection", None) is None:
        raise SourceAlignedSFTError(
            "at least two sealed sources are required unless a parent selection is supplied"
        )
    if args.copy_mode not in {"copy", "hardlink", "auto"}:
        raise SourceAlignedSFTError("unsupported copy mode")
    sample_loss_weight_cap = float(args.sample_loss_weight_cap)
    if not math.isfinite(sample_loss_weight_cap) or sample_loss_weight_cap < 0:
        raise SourceAlignedSFTError(
            "sample loss weight cap must be finite and non-negative"
        )
    excluded_parent_kinds = {
        str(kind).strip()
        for kind in getattr(args, "exclude_parent_kind", [])
        if str(kind).strip()
    }
    included_source_stage_release_ids = {
        str(release_id).strip()
        for release_id in getattr(args, "include_source_stage_release_id", [])
        if str(release_id).strip()
    }
    parent_selection_path = getattr(args, "include_parent_selection", None)
    selected_parent_ids: set[str] = set()
    parent_selection_record: dict[str, Any] | None = None
    if parent_selection_path is not None:
        parent_selection_path = parent_selection_path.resolve(strict=True)
        selection = json.loads(parent_selection_path.read_text(encoding="utf-8"))
        if selection.get("format") not in {
            "iclr_video.capability_parent_selection.v0",
            "iclr_video.capability_parent_selection.v1",
        }:
            raise SourceAlignedSFTError("parent selection format diverged")
        if selection.get("format").endswith(".v1") and (
            selection.get("status") != "sealed_deduplicated_union"
            or (selection.get("contracts") or {}).get(
                "benchmark_content_used_for_selection"
            )
            is not False
            or (selection.get("contracts") or {}).get(
                "source_release_membership_verified"
            )
            is not True
        ):
            raise SourceAlignedSFTError("v1 parent selection contract diverged")
        parent_ids = selection.get("parent_ids") or []
        if (
            not isinstance(parent_ids, list)
            or not parent_ids
            or any(not isinstance(value, str) or not value for value in parent_ids)
            or len(set(parent_ids)) != len(parent_ids)
        ):
            raise SourceAlignedSFTError("parent selection IDs are malformed")
        selected_parent_ids = set(parent_ids)
        parent_selection_record = {
            "path": str(parent_selection_path),
            "sha256": file_sha256(parent_selection_path),
            "parent_ids": len(selected_parent_ids),
            "profiles": list(
                selection.get("profiles")
                or sorted((selection.get("counts_by_profile") or {}).keys())
            ),
        }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    ordered_curriculum = bool(getattr(args, "ordered_curriculum", False))
    retain_semantic_exposures = bool(
        getattr(args, "retain_semantic_exposures", False)
    )
    legacy_v62di_source = getattr(args, "legacy_v62di_source", None)
    legacy_v62di_mode = getattr(args, "legacy_v62di_mode", "v62di_only")
    legacy_repcount_source = getattr(args, "legacy_repcount_source", None)
    if legacy_v62di_source is not None and not ordered_curriculum:
        raise SourceAlignedSFTError(
            "legacy V62DI input is allowed only for ordered curriculum"
        )
    if legacy_repcount_source is not None and not ordered_curriculum:
        raise SourceAlignedSFTError(
            "legacy RepCount input is allowed only for ordered curriculum"
        )
    expected_lineage: tuple[str, str, str] | None = None
    seen_identity: set[str] = set()
    seen_cited_parent: dict[str, str] = {}
    accepted: list[tuple[dict[str, Any], Path, dict[str, dict[str, Any]], str]] = []
    source_records = []
    rejected: collections.Counter[str] = collections.Counter()
    observed_selected_parent_ids: set[str] = set()

    if legacy_v62di_source is not None:
        legacy_root = legacy_v62di_source.resolve(strict=True)
        legacy_rows, legacy_inventory, legacy_record = _load_legacy_v62di_source(
            legacy_root, mode=legacy_v62di_mode
        )
        for row in legacy_rows:
            identity = _semantic_identity(row, legacy_inventory)
            accepted.append((row, legacy_root, legacy_inventory, identity))
            seen_identity.add(identity)
        source_records.append(legacy_record)

    source_specs: list[tuple[str, Path]] = []
    for source_index, raw_root in enumerate(args.source):
        source_specs.append(("semantic", raw_root))
        if source_index == 0 and legacy_repcount_source is not None:
            source_specs.append(("legacy_repcount", legacy_repcount_source))

    for source_kind, raw_root in source_specs:
        root = raw_root.resolve(strict=True)
        if source_kind == "legacy_repcount":
            rows, inventory, source_record = _load_legacy_repcount_source(root)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        else:
            manifest = validate_semantic_release(root, verify_media_hashes=False)
        lineage_key = _lineage_key(manifest)
        if expected_lineage is None:
            expected_lineage = lineage_key
        elif lineage_key != expected_lineage:
            raise SourceAlignedSFTError("source model lineages diverged")
        if lineage_key[0] != EXPECTED_PUBLIC_UPSTREAM_MANIFEST_SHA256:
            raise SourceAlignedSFTError("source does not descend from pinned Instruct")
        if (manifest.get("contracts") or {}).get("benchmark_rows_or_labels_admitted", 0) != 0:
            raise SourceAlignedSFTError("source admits benchmark rows or labels")
        teacher_used_for_labels = _teacher_used_for_labels(manifest)

        if source_kind == "semantic":
            inventory_rows = read_jsonl(
                root / "manifests/media_inventory.jsonl", require_canonical=True
            )
            inventory = {str(item["path"]): item for item in inventory_rows}
            if len(inventory) != len(inventory_rows):
                raise SourceAlignedSFTError("source media inventory paths collide")
            rows = read_jsonl(
                root / "trainer/sft.jsonl",
                verify_content_hashes=True,
                require_canonical=True,
            )
            if included_source_stage_release_ids:
                stages = (manifest.get("inputs") or {}).get("curriculum_stages") or []
                selected_stages = [
                    stage
                    for stage in stages
                    if str(stage.get("release_id"))
                    in included_source_stage_release_ids
                ]
                selected_ids = {
                    str(stage["release_id"]) for stage in selected_stages
                }
                missing_ids = included_source_stage_release_ids - selected_ids
                if missing_ids:
                    raise SourceAlignedSFTError(
                        "requested source stages are absent: "
                        + ",".join(sorted(missing_ids))
                    )
                selected_rows = []
                for stage in sorted(
                    selected_stages, key=lambda item: int(item["start_row"])
                ):
                    start = int(stage["start_row"])
                    end = int(stage["end_row_exclusive"])
                    if start < 0 or end <= start or end > len(rows):
                        raise SourceAlignedSFTError(
                            "source curriculum stage range is invalid"
                        )
                    if end - start != int(stage["rows"]):
                        raise SourceAlignedSFTError(
                            "source curriculum stage row count diverged"
                        )
                    selected_rows.extend(rows[start:end])
                rows = selected_rows
        accepted_before = len(accepted)
        for row in rows:
            row_parent_ids = {
                str(parent.get("id") or "") for parent in row.get("parents") or []
            }
            selected_matches = row_parent_ids & selected_parent_ids
            if selected_parent_ids and not selected_matches:
                rejected["outside_parent_selection"] += 1
                continue
            observed_selected_parent_ids.update(selected_matches)
            row_parent_kinds = {
                str(parent.get("kind") or "") for parent in row.get("parents") or []
            }
            excluded = sorted(row_parent_kinds & excluded_parent_kinds)
            if excluded:
                for kind in excluded:
                    rejected[f"excluded_parent_kind:{kind}"] += 1
                continue
            identity = _semantic_identity(row, inventory)
            cited = _cited_parents(row)
            conflicts = {
                seen_cited_parent[parent]
                for parent in cited
                if parent in seen_cited_parent
            }
            if conflicts and conflicts != {identity}:
                raise SourceAlignedSFTError(
                    "one cited source parent maps to conflicting semantic rows"
                )
            retain_repeated_rows = ordered_curriculum or retain_semantic_exposures
            if identity in seen_identity and not retain_repeated_rows:
                rejected["duplicate_semantic_identity"] += 1
                for parent in cited:
                    seen_cited_parent.setdefault(parent, identity)
                continue
            if conflicts and not retain_repeated_rows:
                rejected["duplicate_cited_source_parent"] += 1
                continue
            seen_identity.add(identity)
            for parent in cited:
                seen_cited_parent[parent] = identity
            accepted.append((row, root, inventory, identity))
        record = (
            source_record
            if source_kind == "legacy_repcount"
            else {
                "accepted_rows": len(accepted) - accepted_before,
                "manifest_sha256": file_sha256(root / "manifest.json"),
                "release_id": manifest["release_id"],
                "root": str(root),
                "source_rows": len(rows),
                "teacher_used_for_labels": teacher_used_for_labels,
                "trainer_sha256": file_sha256(root / "trainer/sft.jsonl"),
            }
        )
        record["accepted_rows"] = len(accepted) - accepted_before
        source_records.append(record)

    if selected_parent_ids:
        missing_selected_parent_ids = selected_parent_ids - observed_selected_parent_ids
        if missing_selected_parent_ids:
            raise SourceAlignedSFTError(
                f"{len(missing_selected_parent_ids)} selected parent IDs were absent"
            )

    within_stage_shuffle_seed = int(
        getattr(args, "shuffle_within_stage_seed", -1)
    )
    if within_stage_shuffle_seed >= 0:
        shuffled: list[
            tuple[dict[str, Any], Path, dict[str, dict[str, Any]], str]
        ] = []
        cursor = 0
        for stage_index, record in enumerate(source_records):
            count = int(record["accepted_rows"])
            stage = accepted[cursor : cursor + count]
            random.Random(within_stage_shuffle_seed + stage_index).shuffle(stage)
            shuffled.extend(stage)
            cursor += count
        if cursor != len(accepted):
            raise SourceAlignedSFTError("curriculum stage row accounting diverged")
        accepted = shuffled

    assert expected_lineage is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial.", dir=output.parent))
    try:
        output_inventory: dict[str, dict[str, Any]] = {}
        media_sources: dict[str, Path] = {}
        output_rows = []
        by_source: collections.Counter[str] = collections.Counter()
        weights_before: collections.Counter[float] = collections.Counter()
        weights_after: collections.Counter[float] = collections.Counter()
        reweighted_rows = 0
        source_stage = {
            item["root"]: index for index, item in enumerate(source_records)
        }
        source_row_index: collections.Counter[str] = collections.Counter()
        repeated_semantic_exposures = len(accepted) - len(seen_identity)
        stage_cursor = 0
        curriculum_stages = []
        for stage_index, record in enumerate(source_records):
            stage_rows = int(record["accepted_rows"])
            curriculum_stages.append(
                {
                    "end_row_exclusive": stage_cursor + stage_rows,
                    "index": stage_index,
                    "release_id": record["release_id"],
                    "rows": stage_rows,
                    "start_row": stage_cursor,
                }
            )
            stage_cursor += stage_rows
        if stage_cursor != len(accepted):
            raise SourceAlignedSFTError("curriculum manifest row accounting diverged")
        for row, root, inventory, identity in accepted:
            messages = copy.deepcopy(row["messages"])
            for message in messages:
                for item in message.get("content") or []:
                    if item.get("type") != "image":
                        continue
                    source_relative = str(item["image"])
                    declaration = inventory[source_relative]
                    digest = str(declaration["sha256"])
                    canonical_relative = _canonical_media_path(source_relative, digest)
                    previous = output_inventory.get(canonical_relative)
                    normalized = {
                        "bytes": int(declaration["bytes"]),
                        "path": canonical_relative,
                        "sha256": digest,
                    }
                    if previous is not None and previous != normalized:
                        raise SourceAlignedSFTError("canonical media declarations conflict")
                    output_inventory.setdefault(canonical_relative, normalized)
                    media_sources.setdefault(canonical_relative, root / source_relative)
                    item["image"] = canonical_relative
            source_weight = float(row.get("sample_loss_weight", 1.0))
            output_weight = (
                min(source_weight, sample_loss_weight_cap)
                if sample_loss_weight_cap > 0
                else source_weight
            )
            weights_before[source_weight] += 1
            weights_after[output_weight] += 1
            reweighted_rows += int(output_weight != source_weight)
            stage_index = source_stage[str(root)]
            row_index = source_row_index[str(root)]
            source_row_index[str(root)] += 1
            uid_parts: tuple[Any, ...] = (identity,)
            if ordered_curriculum or retain_semantic_exposures:
                uid_parts = (stage_index, row_index, identity)
            clean = attach_content_sha256(
                {
                    "schema": "streameva.sft_row.v1",
                    "uid": stable_id(
                        "sourcealignedsft", *uid_parts, digest_length=24
                    ),
                    "component": "preservation",
                    "group_id": stable_id(
                        "sourcealignedsftgroup", *uid_parts, digest_length=24
                    ),
                    "split": "train",
                    "messages": messages,
                    "sample_loss_weight": output_weight,
                    "parents": copy.deepcopy(row.get("parents") or []),
                }
            )
            validate_row(clean)
            output_rows.append(clean)
            source_release = next(
                item["release_id"]
                for item in source_records
                if item["root"] == str(root)
            )
            by_source[source_release] += 1

        if not ordered_curriculum:
            output_rows.sort(key=lambda row: str(row["uid"]))
        if len({str(row["uid"]) for row in output_rows}) != len(output_rows):
            raise SourceAlignedSFTError("output UIDs collide")
        write_jsonl(output_rows, scratch / "trainer/sft.jsonl")
        inventory_rows = [output_inventory[path] for path in sorted(output_inventory)]
        write_jsonl(inventory_rows, scratch / "manifests/media_inventory.jsonl")
        for item in inventory_rows:
            relative = str(item["path"])
            destination = scratch / relative
            _materialize(media_sources[relative], destination, mode=args.copy_mode)
            if (
                destination.stat().st_size != int(item["bytes"])
                or file_sha256(destination) != item["sha256"]
            ):
                raise SourceAlignedSFTError("materialized media hash diverged")

        answers = collections.Counter(_target(row) for row in output_rows)
        frames = collections.Counter(str(len(_images(row))) for row in output_rows)
        manifest = {
            "artifacts": {
                "manifests/media_inventory.jsonl": file_sha256(
                    scratch / "manifests/media_inventory.jsonl"
                ),
                "trainer/sft.jsonl": file_sha256(scratch / "trainer/sft.jsonl"),
            },
            "contracts": {
                "agent_action_targets": 0,
                "assistant_targets_plain_semantic_only": True,
                "benchmark_examples_admitted": 0,
                "benchmark_firewall_inherited": True,
                "counterfactual_rows_admitted": 0,
                "environment_state_actor_targets": 0,
                "legacy_stage_metadata_admitted": 0,
                "loss_on_assistant_only": True,
                "policy_holdout_admitted": 0,
                "private_verifier_actor_targets": 0,
                "all_sample_loss_weights_one": set(weights_after) == {1.0},
                "sample_loss_weight_cap": (
                    sample_loss_weight_cap if sample_loss_weight_cap > 0 else None
                ),
                "sample_loss_weight_rows_changed": reweighted_rows,
                "self_contained_media": True,
                "teacher_used_for_labels": any(
                    bool(item["teacher_used_for_labels"])
                    for item in source_records
                ),
                "training_runs_required": 1,
                "ordered_curriculum": ordered_curriculum,
                "retain_semantic_exposures": retain_semantic_exposures,
                "repeated_semantic_exposures": repeated_semantic_exposures,
                "shuffle_within_stage_seed": (
                    within_stage_shuffle_seed
                    if within_stage_shuffle_seed >= 0
                    else None
                ),
            },
            "counts": {
                "media_bytes": sum(int(item["bytes"]) for item in inventory_rows),
                "media_files": len(inventory_rows),
                "trainer_rows": len(output_rows),
            },
            "distributions": {
                "answer": dict(sorted(answers.items())),
                "frames_per_row": dict(sorted(frames.items())),
                "sample_loss_weight": {
                    str(weight): count
                    for weight, count in sorted(weights_after.items())
                },
                "source_release": dict(sorted(by_source.items())),
            },
            "format": FORMAT,
            "inputs": {
                "benchmark_rows_or_labels_admitted": 0,
                "builder_sha256": file_sha256(Path(__file__).resolve(strict=True)),
                "deduplication": {
                    "cited_source_parent": True,
                    "exact_pixel_question_answer_identity": True,
                    "retain_semantic_exposures": retain_semantic_exposures,
                    "rejected": dict(sorted(rejected.items())),
                },
                "excluded_parent_kinds": sorted(excluded_parent_kinds),
                "included_source_stage_release_ids": sorted(
                    included_source_stage_release_ids
                ),
                "parent_selection": parent_selection_record,
                "final_release_must_restart_from": "Qwen/Qwen3-VL-8B-Instruct",
                "rows_relabelled": 0,
                "rows_reweighted": reweighted_rows,
                "sample_loss_weight_policy": {
                    "cap": (
                        sample_loss_weight_cap
                        if sample_loss_weight_cap > 0
                        else None
                    ),
                    "source_distribution": {
                        str(weight): count
                        for weight, count in sorted(weights_before.items())
                    },
                    "output_distribution": {
                        str(weight): count
                        for weight, count in sorted(weights_after.items())
                    },
                },
                "sources": source_records,
                "curriculum_stages": curriculum_stages,
                "trainer_row_order": (
                    "source_stage_then_source_row"
                    if ordered_curriculum
                    else "canonical_uid"
                ),
            },
            "license": {
                "public_media_redistribution_claimed": False,
                "release_plan": "code_recipe_selection_and_source_checksums",
                "scope": "internal_noncommercial_research_only",
            },
            "model_lineage": {
                "required_initialization": "Qwen/Qwen3-VL-8B-Instruct",
                "required_sha256s_sha256": expected_lineage[0],
                "upstream_repo": expected_lineage[1],
                "upstream_revision": expected_lineage[2],
            },
            "public_release_allowed": False,
            "release_id": args.release_id,
            "status": "sealed_internal_training_release",
            "training_allowed": True,
            "training_scope": "pure_vlm_semantic_sft_only",
        }
        (scratch / "manifest.json").write_bytes(canonical_json_line(manifest))
        validate_semantic_release(scratch, verify_media_hashes=True)
        os.replace(scratch, output)
        return manifest | {"output": str(output)}
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument(
        "--copy-mode", choices=("copy", "hardlink", "auto"), default="hardlink"
    )
    parser.add_argument(
        "--sample-loss-weight-cap",
        type=float,
        default=0.0,
        help="Cap inherited sample weights; zero preserves source weights.",
    )
    parser.add_argument(
        "--exclude-parent-kind",
        action="append",
        default=[],
        help="Exclude rows carrying this exact parent kind; may be repeated.",
    )
    parser.add_argument(
        "--include-source-stage-release-id",
        action="append",
        default=[],
        help=(
            "For a composite semantic source, admit only rows inside the "
            "named hash-bound curriculum stages; may be repeated."
        ),
    )
    parser.add_argument(
        "--include-parent-selection",
        type=Path,
        help=(
            "Hash-bind a capability selection and admit only semantic rows "
            "whose cited parent ID is listed in it."
        ),
    )
    parser.add_argument(
        "--ordered-curriculum",
        action="store_true",
        help="Preserve source-stage order and retain repeated stage exposures.",
    )
    parser.add_argument(
        "--retain-semantic-exposures",
        action="store_true",
        help=(
            "Retain every source row as an intentional training exposure while "
            "keeping the normal globally canonicalized row order."
        ),
    )
    parser.add_argument(
        "--legacy-v62di-source",
        type=Path,
        help="Pinned legacy release; admits only its 2,320 V62DI rows.",
    )
    parser.add_argument(
        "--legacy-v62di-mode",
        choices=("v62di_only", "full"),
        default="v62di_only",
        help="Use only V62DI rows or the complete committed 3,594-row pool.",
    )
    parser.add_argument(
        "--legacy-repcount-source",
        type=Path,
        help="Pinned 2,172-row legacy RepCount source, inserted after source 1.",
    )
    parser.add_argument(
        "--shuffle-within-stage-seed",
        type=int,
        default=-1,
        help="Deterministically shuffle each stage while preserving stage order.",
    )
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
