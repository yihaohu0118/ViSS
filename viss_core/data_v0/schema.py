"""Minimal canonical schema for the released pure-VLM semantic rows."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from viss_core.data_v0.canonical import validate_content_sha256


SCHEMA_TRAIN_ROW = "streameva.sft_row.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{2,127}$")


class SchemaError(ValueError):
    """Raised when a semantic training row violates its release ABI."""


def validate_row(row: Mapping[str, Any], *, verify_hash: bool = True) -> None:
    """Validate one plain-answer video SFT row and reject legacy schemas."""

    if not isinstance(row, Mapping) or row.get("schema") != SCHEMA_TRAIN_ROW:
        raise SchemaError("only the semantic SFT-row schema is supported")
    _exact_keys(
        row,
        required={
            "schema",
            "uid",
            "component",
            "group_id",
            "split",
            "messages",
            "sample_loss_weight",
            "parents",
            "content_sha256",
        },
        optional={"decision_loss_profile"},
        path="sft_row",
    )
    _identifier(row["uid"], "sft_row.uid")
    _identifier(row["group_id"], "sft_row.group_id")
    component = row["component"]
    if component != "preservation":
        raise SchemaError("semantic SFT row uses a non-semantic component")
    if row["split"] not in {"train", "policy_holdout"}:
        raise SchemaError("semantic SFT split must be train or policy_holdout")
    if row.get("decision_loss_profile", "full") != "full":
        raise SchemaError("semantic SFT decision-loss profile diverged")
    weight = row["sample_loss_weight"]
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or weight <= 0
    ):
        raise SchemaError("sample_loss_weight must be finite and positive")

    messages = _sequence(row["messages"], "sft_row.messages")
    if len(messages) != 2 or [message.get("role") for message in messages] != [
        "user",
        "assistant",
    ]:
        raise SchemaError("semantic SFT must contain user then assistant")
    for index, message in enumerate(messages):
        path = f"sft_row.messages[{index}]"
        _exact_keys(message, required={"role", "content", "loss"}, path=path)
        expected_loss = message["role"] == "assistant"
        if message["loss"] is not expected_loss:
            raise SchemaError("only the assistant semantic answer may carry loss")
        content = _sequence(message["content"], f"{path}.content")
        if not content:
            raise SchemaError(f"{path}.content must not be empty")
        for item_index, item in enumerate(content):
            item_path = f"{path}.content[{item_index}]"
            if not isinstance(item, Mapping):
                raise SchemaError(f"{item_path} must be an object")
            if item.get("type") == "text":
                _exact_keys(item, required={"type", "text"}, path=item_path)
                _nonempty_string(item["text"], f"{item_path}.text")
            elif item.get("type") == "image":
                if message["role"] != "user":
                    raise SchemaError("only the user may contain images")
                _exact_keys(
                    item,
                    required={"type", "image", "frame_id"},
                    path=item_path,
                )
                _relative_uri(item["image"], f"{item_path}.image")
                _identifier(item["frame_id"], f"{item_path}.frame_id")
            else:
                raise SchemaError(f"{item_path}.type must be text or image")
    assistant_content = messages[1]["content"]
    if len(assistant_content) != 1 or assistant_content[0].get("type") != "text":
        raise SchemaError("assistant must contain exactly one semantic text answer")
    _parent_list(row["parents"], "sft_row.parents")
    _forbid_absolute_paths(row)
    if verify_hash:
        try:
            validate_content_sha256(row)
        except ValueError as error:
            raise SchemaError(str(error)) from error


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    path: str,
    optional: set[str] | None = None,
) -> None:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{path} must be an object")
    optional = optional or set()
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing or unknown:
        raise SchemaError(
            f"{path} fields diverged: missing={sorted(missing)} unknown={sorted(unknown)}"
        )


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SchemaError(f"{path} must be an array")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{path} must be non-empty text")
    return value


def _identifier(value: Any, path: str) -> str:
    value = _nonempty_string(value, path)
    if _ID_RE.fullmatch(value) is None:
        raise SchemaError(f"{path} is not a stable identifier")
    return value


def _relative_uri(value: Any, path: str) -> str:
    value = _nonempty_string(value, path)
    if value.startswith(("/", "file:")) or ".." in value.split("/"):
        raise SchemaError(f"{path} must be release-relative")
    return value


def _parent_list(value: Any, path: str) -> None:
    for index, parent in enumerate(_sequence(value, path)):
        parent_path = f"{path}[{index}]"
        _exact_keys(
            parent,
            required={"kind", "id", "sha256"},
            path=parent_path,
        )
        _nonempty_string(parent["kind"], f"{parent_path}.kind")
        _identifier(parent["id"], f"{parent_path}.id")
        if not isinstance(parent["sha256"], str) or _SHA256_RE.fullmatch(
            parent["sha256"]
        ) is None:
            raise SchemaError(f"{parent_path}.sha256 must be lowercase SHA256")


def _forbid_absolute_paths(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _forbid_absolute_paths(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _forbid_absolute_paths(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and value.startswith(("/", "file://")):
        raise SchemaError(f"{path}: absolute filesystem paths are forbidden")
