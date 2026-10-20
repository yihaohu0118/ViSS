from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCKED_PROTOCOL = PROJECT_ROOT / "configs/evaluation/recent4_full_v1.json"


class ProtocolValidationError(ValueError):
    """Raised when the locked evaluation protocol or its exports drift."""


def load_locked_protocol(path: Path = LOCKED_PROTOCOL) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "locked":
        raise ProtocolValidationError(f"{path}: protocol is not locked")
    if protocol.get("protocol_id") != "recent4_full_v1":
        raise ProtocolValidationError(f"{path}: unexpected protocol_id")

    inference = protocol.get("inference") or {}
    required_inference = {
        "causal_prefix_only": True,
        "sampling_fps": 1.0,
        "context_policy": "recent_n",
        "recent_frames": 4,
        "allow_future_frames": False,
        "allow_global_frames": False,
        "allow_extra_frame_inspection": False,
        "allow_persistent_visual_or_text_memory": False,
    }
    for key, expected in required_inference.items():
        if inference.get(key) != expected:
            raise ProtocolValidationError(
                f"{path}: inference.{key}={inference.get(key)!r}, expected {expected!r}"
            )
    processor = inference.get("processor_contract") or {}
    if not re.fullmatch(r"[0-9a-f]{64}", str(processor.get("preprocessor_config_sha256", ""))):
        raise ProtocolValidationError("Configure the protocol with your processor identity first")
    required_processor = {
        "image_processor_type": "Qwen2VLImageProcessorFast",
        "patch_size": 16,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "shortest_edge": 65536,
        "longest_edge": 16777216,
        "reference_visual_tokens_per_frame": 576,
    }
    for key, expected in required_processor.items():
        if processor.get(key) != expected:
            raise ProtocolValidationError(
                f"{path}: inference.processor_contract.{key}="
                f"{processor.get(key)!r}, expected {expected!r}"
            )
    return protocol


def validate_locked_protocol(path: Path = LOCKED_PROTOCOL) -> dict[str, Any]:
    protocol = load_locked_protocol(path)
    root = path.resolve().parents[2]
    reports: dict[str, Any] = {}
    for benchmark, benchmark_config in protocol["benchmarks"].items():
        export = root / benchmark_config["export"]
        rows = _load_jsonl(export)
        expected_count = int(benchmark_config["expected_queries"])
        if len(rows) != expected_count:
            raise ProtocolValidationError(
                f"{export}: {len(rows)} rows, expected {expected_count}"
            )
        digest = _sha256(export)
        if digest != benchmark_config["export_sha256"]:
            raise ProtocolValidationError(
                f"{export}: sha256 {digest}, expected {benchmark_config['export_sha256']}"
            )

        uids = [str(row.get("uid") or "") for row in rows]
        if any(not uid for uid in uids) or len(set(uids)) != len(uids):
            raise ProtocolValidationError(f"{export}: missing or duplicate uid")
        labels = Counter(str(row.get("benchmark") or "") for row in rows)
        if labels != Counter({benchmark: expected_count}):
            raise ProtocolValidationError(f"{export}: benchmark labels {dict(labels)}")

        report: dict[str, Any] = {
            "export": str(export.relative_to(root)),
            "sha256": digest,
            "queries": len(rows),
        }
        expected_groups = benchmark_config.get("expected_groups")
        if expected_groups is not None:
            actual_groups = Counter(
                str((row.get("metadata") or {}).get("group") or "") for row in rows
            )
            if dict(actual_groups) != expected_groups:
                raise ProtocolValidationError(
                    f"{export}: groups {dict(actual_groups)}, expected {expected_groups}"
                )
            report["groups"] = dict(sorted(actual_groups.items()))

        expected_tasks = benchmark_config.get("expected_tasks")
        if expected_tasks is not None:
            actual_tasks = Counter(
                str((row.get("metadata") or {}).get("task") or "") for row in rows
            )
            if dict(sorted(actual_tasks.items())) != expected_tasks:
                raise ProtocolValidationError(
                    f"{export}: tasks {dict(actual_tasks)}, expected {expected_tasks}"
                )
            report["tasks"] = dict(sorted(actual_tasks.items()))
        reports[benchmark] = report

    return {
        "status": "locked_protocol_valid",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(path.resolve().relative_to(root)),
        "benchmarks": reports,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProtocolValidationError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
