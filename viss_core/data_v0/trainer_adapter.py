"""Strict adapter from a sealed semantic release to Qwen messages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from viss_core.data_v0.semantic_release import (
    infer_semantic_release_root,
    load_semantic_sft_rows,
)


def load_training_rows(
    path: Path, *, allow_nontrainable: bool = False
) -> list[dict[str, Any]]:
    """Load only a validated, training-authorized semantic release."""

    if allow_nontrainable:
        raise RuntimeError("the release trainer has no non-trainable bypass")
    path = path.resolve()
    if path.suffix != ".jsonl":
        raise ValueError("semantic training accepts canonical JSONL only")
    return load_semantic_sft_rows(path)


def infer_release_root(path: Path) -> Path | None:
    return infer_semantic_release_root(path)


def normalize_messages(
    messages: Any,
    *,
    media_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Resolve release-local images and preserve assistant-only loss ownership."""

    if not isinstance(messages, list):
        raise TypeError("semantic messages must be a list")
    roles = [message.get("role") for message in messages]
    if roles != ["user", "assistant"]:
        raise ValueError("semantic messages require user then assistant")
    root = media_root.resolve() if media_root is not None else None
    if root is None:
        raise RuntimeError("semantic media requires an explicit release root")
    output: list[dict[str, Any]] = []
    for message in messages:
        raw_content = message.get("content")
        if not isinstance(raw_content, list) or not raw_content:
            raise TypeError("semantic message content must be a non-empty list")
        content: list[dict[str, Any]] = []
        for item in raw_content:
            if not isinstance(item, dict):
                raise TypeError("semantic content item must be an object")
            if item.get("type") == "text":
                content.append({"type": "text", "text": str(item.get("text") or "")})
                continue
            if item.get("type") != "image" or not isinstance(item.get("image"), str):
                raise ValueError("semantic content item has an unsupported type")
            if message["role"] != "user":
                raise ValueError("only user states may contain semantic images")
            relative = Path(item["image"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("semantic image path must be release-relative")
            image_path = (root / relative).resolve(strict=True)
            try:
                image_path.relative_to(root)
            except ValueError as error:
                raise RuntimeError("semantic image path escapes the release") from error
            if not image_path.is_file() or image_path.is_symlink():
                raise RuntimeError("semantic image is missing or unsafe")
            content.append({"type": "image", "image": Image.open(image_path).convert("RGB")})
        output.append(
            {
                "role": str(message["role"]),
                "content": content,
                "loss": bool(message.get("loss")),
            }
        )
    return output
