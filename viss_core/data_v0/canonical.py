from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


CONTENT_SHA256_FIELD = "content_sha256"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STABLE_ID_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


class CanonicalizationError(ValueError):
    """Raised when a value cannot be represented by the canonical JSON format."""


class ContentHashError(ValueError):
    """Raised when a row's declared content hash is absent or invalid."""


def _normalize_json(value: Any, *, location: str = "$") -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(f"{location}: non-finite floats are not valid")
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        original_keys: dict[str, str] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(
                    f"{location}: JSON object keys must be strings, got {type(key).__name__}"
                )
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                first = original_keys[normalized_key]
                raise CanonicalizationError(
                    f"{location}: keys {first!r} and {key!r} collide after NFC normalization"
                )
            original_keys[normalized_key] = key
            normalized[normalized_key] = _normalize_json(
                child,
                location=f"{location}.{normalized_key}",
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(child, location=f"{location}[{index}]")
            for index, child in enumerate(value)
        ]
    raise CanonicalizationError(
        f"{location}: unsupported JSON value type {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact, sorted-key, NFC-normalized UTF-8 JSON bytes."""

    normalized = _normalize_json(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_line(value: Any) -> bytes:
    """Return one canonical JSONL record, including its trailing LF."""

    return canonical_json_bytes(value) + b"\n"


def record_content_sha256(
    record: Mapping[str, Any],
    *,
    field: str = CONTENT_SHA256_FIELD,
) -> str:
    """Hash a canonical JSONL row after removing its top-level hash field."""

    if not isinstance(record, Mapping):
        raise CanonicalizationError("content hashing requires a JSON object")
    payload = dict(record)
    payload.pop(field, None)
    return hashlib.sha256(canonical_json_line(payload)).hexdigest()


def attach_content_sha256(
    record: Mapping[str, Any],
    *,
    field: str = CONTENT_SHA256_FIELD,
    reject_mismatch: bool = True,
) -> dict[str, Any]:
    """Return a copy carrying its computed content hash.

    Existing matching hashes are preserved. By default, an existing stale hash
    is rejected instead of being silently repaired.
    """

    output = dict(record)
    expected = record_content_sha256(output, field=field)
    declared = output.get(field)
    if declared is not None and reject_mismatch:
        _validate_declared_sha256(declared, field=field)
        if not hmac.compare_digest(declared, expected):
            raise ContentHashError(
                f"{field} mismatch: declared {declared}, expected {expected}"
            )
    output[field] = expected
    return output


def validate_content_sha256(
    record: Mapping[str, Any],
    *,
    field: str = CONTENT_SHA256_FIELD,
) -> bool:
    """Strictly validate a row's lowercase SHA256 declaration."""

    if not isinstance(record, Mapping):
        raise ContentHashError("content hash validation requires a JSON object")
    if field not in record:
        raise ContentHashError(f"missing required {field}")
    declared = record[field]
    _validate_declared_sha256(declared, field=field)
    expected = record_content_sha256(record, field=field)
    if not hmac.compare_digest(declared, expected):
        raise ContentHashError(
            f"{field} mismatch: declared {declared}, expected {expected}"
        )
    return True


def _validate_declared_sha256(value: Any, *, field: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContentHashError(f"{field} must be exactly 64 lowercase hex characters")


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 of a file's exact bytes."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(
    prefix: str,
    *parts: Any,
    digest_length: int = 64,
) -> str:
    """Build a stable prefixed ID from structured, canonicalized components."""

    normalized_prefix = unicodedata.normalize("NFC", prefix).rstrip("_")
    if _STABLE_ID_PREFIX_RE.fullmatch(normalized_prefix) is None:
        raise ValueError(
            "prefix must start with an ASCII letter and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    if not 8 <= digest_length <= 64:
        raise ValueError("digest_length must be between 8 and 64")
    digest = hashlib.sha256(canonical_json_bytes(list(parts))).hexdigest()
    return f"{normalized_prefix}_{digest[:digest_length]}"


def write_jsonl(
    rows: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    overwrite: bool = False,
    attach_hashes: bool = False,
    sort_key: str | Callable[[Mapping[str, Any]], Any] | None = None,
) -> Path:
    """Write canonical JSONL, refusing to overwrite by default.

    Rows are preserved in input order unless ``sort_key`` is supplied. Any
    pre-existing ``content_sha256`` is validated before bytes are written.
    """

    materialized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise CanonicalizationError(
                f"row {index}: JSONL rows must be objects, got {type(row).__name__}"
            )
        output = attach_content_sha256(row) if attach_hashes else dict(row)
        if CONTENT_SHA256_FIELD in output:
            try:
                validate_content_sha256(output)
            except ContentHashError as exc:
                raise ContentHashError(f"row {index}: {exc}") from exc
        materialized.append(output)

    if sort_key is not None:
        if isinstance(sort_key, str):
            key_name = sort_key

            def canonical_sort_value(row: Mapping[str, Any]) -> bytes:
                if key_name not in row:
                    raise KeyError(f"sort key {key_name!r} is missing from a row")
                return canonical_json_bytes(row[key_name])

        elif callable(sort_key):

            def canonical_sort_value(row: Mapping[str, Any]) -> bytes:
                return canonical_json_bytes(sort_key(row))

        else:
            raise TypeError("sort_key must be a field name, callable, or None")
        materialized.sort(key=canonical_sort_value)

    payload = b"".join(canonical_json_line(row) for row in materialized)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if overwrite else "xb"
    with output_path.open(mode) as handle:
        handle.write(payload)
    return output_path


def read_jsonl(
    path: str | Path,
    *,
    verify_content_hashes: bool = False,
    require_canonical: bool = False,
) -> list[dict[str, Any]]:
    """Read object-only JSONL with optional hash and exact-canonical checks."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"line {line_number}: blank JSONL records are forbidden")
            try:
                text = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid UTF-8") from exc
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: JSONL row must be an object")
            if require_canonical and raw_line != canonical_json_line(value):
                raise ValueError(f"line {line_number}: row is not canonical JSONL")
            if verify_content_hashes:
                try:
                    validate_content_sha256(value)
                except ContentHashError as exc:
                    raise ContentHashError(f"line {line_number}: {exc}") from exc
            rows.append(value)
    return rows


def validate_jsonl_content_sha256(
    path: str | Path,
    *,
    require_canonical: bool = True,
) -> int:
    """Validate every row and return the number of verified records."""

    return len(
        read_jsonl(
            path,
            verify_content_hashes=True,
            require_canonical=require_canonical,
        )
    )


# Short aliases for callers that already operate in a canonical-data context.
canonical_bytes = canonical_json_bytes
canonical_line = canonical_json_line
content_sha256 = record_content_sha256
