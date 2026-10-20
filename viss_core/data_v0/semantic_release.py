"""Validation and loading for the frozen temporal-scope semantic SFT release."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from viss_core.data_v0.canonical import file_sha256, read_jsonl
from viss_core.data_v0.schema import validate_row

FORMAT = "streameva.temporal_scope_semantic_sft.v0"
TRAINER_RELATIVE_PATH = "trainer/sft.jsonl"
MANIFEST_RELATIVE_PATH = "manifest.json"
EXPECTED_PUBLIC_UPSTREAM_MANIFEST_SHA256 = os.environ.get("VISS_UPSTREAM_MANIFEST_SHA256", "")
ANSWER_RE = re.compile(r"^(?:[A-D]|Yes|No|10|[0-9])$")
DENSE_TEMPORAL_PARENT_KIND = "teacher_generated_pixel_verified_dense_temporal"
PURE_VLM_SCOPE = "pure_vlm_semantic_sft_only"
def _valid_semantic_target(row: dict[str, Any], text: str) -> bool:
    if ANSWER_RE.fullmatch(text) is not None:
        return True
    parent_kinds = {
        str(parent.get("kind") or "") for parent in row.get("parents") or []
    }
    words = text.split()
    return (
        DENSE_TEMPORAL_PARENT_KIND in parent_kinds
        and 20 <= len(words) <= 220
        and "<" not in text
        and ">" not in text
        and "```" not in text
    )


class SemanticReleaseError(ValueError):
    """Raised when the semantic SFT artifact is not a sealed release."""


def infer_semantic_release_root(path: Path) -> Path | None:
    path = path.resolve()
    for parent in path.parents:
        manifest = parent / MANIFEST_RELATIVE_PATH
        if not manifest.is_file():
            continue
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("format") == FORMAT:
            return parent
    return None


def validate_semantic_release(
    root: Path, *, verify_media_hashes: bool
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / MANIFEST_RELATIVE_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "artifacts",
        "contracts",
        "counts",
        "distributions",
        "format",
        "inputs",
        "license",
        "model_lineage",
        "public_release_allowed",
        "release_id",
        "status",
        "training_allowed",
        "training_scope",
    }
    if set(manifest) != required:
        raise SemanticReleaseError("semantic release manifest fields are invalid")
    training_scope = manifest.get("training_scope")
    if (
        manifest.get("format") != FORMAT
        or manifest.get("status") != "sealed_internal_training_release"
        or manifest.get("training_allowed") is not True
        or manifest.get("public_release_allowed") is not False
        or training_scope != PURE_VLM_SCOPE
    ):
        raise SemanticReleaseError("semantic release state is invalid")
    contracts = manifest.get("contracts", {})
    if (
        contracts.get("agent_action_targets") != 0
        or contracts.get("environment_state_actor_targets") != 0
        or contracts.get("private_verifier_actor_targets") != 0
        or contracts.get("assistant_targets_plain_semantic_only") is not True
        or contracts.get("self_contained_media") is not True
    ):
        raise SemanticReleaseError("semantic release contracts diverged")
    if not re.fullmatch(r"[0-9a-f]{64}", EXPECTED_PUBLIC_UPSTREAM_MANIFEST_SHA256):
        raise SemanticReleaseError("Set VISS_UPSTREAM_MANIFEST_SHA256 to your trusted upstream identity")
    lineage = manifest.get("model_lineage", {})
    if (
        lineage.get("required_sha256s_sha256")
        != EXPECTED_PUBLIC_UPSTREAM_MANIFEST_SHA256
    ):
        raise SemanticReleaseError("semantic release public-upstream lineage diverged")

    trainer = root / TRAINER_RELATIVE_PATH
    inventory_path = root / "manifests/media_inventory.jsonl"
    artifacts = manifest.get("artifacts", {})
    if (
        set(artifacts) != {TRAINER_RELATIVE_PATH, "manifests/media_inventory.jsonl"}
        or not trainer.is_file()
        or trainer.is_symlink()
        or file_sha256(trainer) != artifacts.get(TRAINER_RELATIVE_PATH)
        or not inventory_path.is_file()
        or inventory_path.is_symlink()
        or file_sha256(inventory_path)
        != artifacts.get("manifests/media_inventory.jsonl")
    ):
        raise SemanticReleaseError("semantic release artifacts diverged")
    rows = read_jsonl(trainer, verify_content_hashes=True, require_canonical=True)
    if len(rows) != manifest.get("counts", {}).get("trainer_rows"):
        raise SemanticReleaseError("semantic release row count diverged")
    uids = set()
    content_hashes = set()
    referenced_media = set()
    for row in rows:
        validate_row(row)
        if row["uid"] in uids or row["content_sha256"] in content_hashes:
            raise SemanticReleaseError("semantic release contains duplicate rows")
        uids.add(row["uid"])
        content_hashes.add(row["content_sha256"])
        if [message["role"] for message in row["messages"]] != ["user", "assistant"]:
            raise SemanticReleaseError("non-VLM dialogue role admitted")
        assistant = row["messages"][1]
        texts = [item["text"] for item in assistant["content"] if item["type"] == "text"]
        if len(texts) != 1 or not _valid_semantic_target(row, texts[0]):
            raise SemanticReleaseError("non-semantic assistant target admitted")
        images = [
            item["image"]
            for message in row["messages"]
            for item in message["content"]
            if item["type"] == "image"
        ]
        invalid_frame_count = len(images) < 4 or len(images) > 32 or len(images) % 4
        if invalid_frame_count:
            raise SemanticReleaseError("unsupported semantic frame count")
        referenced_media.update(images)

    inventory = read_jsonl(inventory_path, require_canonical=True)
    if len(inventory) != manifest.get("counts", {}).get("media_files"):
        raise SemanticReleaseError("semantic media inventory count diverged")
    declared = set()
    total_bytes = 0
    for item in inventory:
        if set(item) != {"bytes", "path", "sha256"}:
            raise SemanticReleaseError("semantic media inventory row is malformed")
        relative = str(item["path"])
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SemanticReleaseError("semantic media path escapes release") from exc
        if (
            relative in declared
            or not relative.startswith("media/sha256/")
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item["bytes"]
            or (verify_media_hashes and file_sha256(path) != item["sha256"])
        ):
            raise SemanticReleaseError("semantic media inventory diverged")
        declared.add(relative)
        total_bytes += int(item["bytes"])
    if referenced_media != declared:
        raise SemanticReleaseError("semantic media reference set diverged")
    counts = manifest["counts"]
    if total_bytes != counts.get("media_bytes"):
        raise SemanticReleaseError("semantic media byte count diverged")
    return manifest


def load_semantic_sft_rows(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    root = infer_semantic_release_root(path)
    if root is None or path.relative_to(root).as_posix() != TRAINER_RELATIVE_PATH:
        raise SemanticReleaseError(
            f"semantic trainer must be exactly {TRAINER_RELATIVE_PATH}"
        )
    manifest = validate_semantic_release(root, verify_media_hashes=False)
    inputs = manifest.get("inputs") or {}
    selection = inputs.get("selection") or {}
    if (
        selection.get("blind_audit_required_before_training") is True
        and "blind_visual_filter" not in inputs
    ):
        raise SemanticReleaseError(
            "refresh semantic candidate requires a complete blind-audit filter"
        )
    if (
        selection.get("option_consensus_required_before_training") is True
        and "option_consensus_filter" not in inputs
    ):
        raise SemanticReleaseError(
            "distilled semantic candidate requires a complete option-consensus filter"
        )
    return read_jsonl(path, verify_content_hashes=True, require_canonical=True)
