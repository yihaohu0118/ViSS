#!/usr/bin/env python3
"""Build a source-disjoint, pixel-grounded VST prefix-32 semantic pack.

The upstream VST metadata provides cumulative count states.  A recent-only
window cannot in general support those labels, so this compiler exposes a
uniform causal prefix ending strictly before the query time.  It emits only
plain semantic answers (open integer and balanced MCQ); no agent action or
private state is ever an actor target.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from data_pipeline.temporal_scope.semantic_mcq import (
    DIRECT_PROMPT_PREFIX,
    DIRECT_PROMPT_SUFFIX,
    rebalance_answer_slot,
)
from viss_core.data_v0.canonical import (
    attach_content_sha256,
    canonical_json_line,
    file_sha256,
    stable_id,
    write_jsonl,
)
from viss_core.data_v0.schema import validate_row


FORMAT = "streameva.vst_prefix32_semantic.v1"
PREFLIGHT_FORMAT = "streameva.vst_count_metadata_preflight.v1"
OPTION_LETTERS = "ABCD"
TARGET_COUNTS = tuple(range(11))
PREFIX_FRAME_MAX_PIXELS = 128 * 28 * 28
PREFIX_FRAME_MAX_SIDE = 504


class VSTPrefixSemanticError(ValueError):
    """Raised when a prefix-32 semantic candidate violates its contract."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--train-per-count", type=int, default=92)
    parser.add_argument("--holdout-per-count", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=173)
    args = parser.parse_args()
    print(json.dumps(materialize(args), indent=2, sort_keys=True))


def select_available_states(
    rows: Sequence[Mapping[str, Any]],
    *,
    media_root: Path,
    train_per_count: int,
    holdout_per_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select balanced, source-disjoint train and holdout states."""

    if train_per_count <= 0 or holdout_per_count <= 0:
        raise VSTPrefixSemanticError("train/holdout quotas must be positive")
    by_count: dict[int, list[dict[str, Any]]] = {
        value: [] for value in TARGET_COUNTS
    }
    seen_videos: set[str] = set()
    seen_ids: set[str] = set()
    for raw in rows:
        row = dict(raw)
        answer = int(row.get("answer", -1))
        relative = Path(str(row.get("source_video_relative") or ""))
        candidate_id = str(row.get("candidate_id") or "")
        if (
            answer not in by_count
            or not candidate_id
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.casefold() != ".mp4"
        ):
            raise VSTPrefixSemanticError("selection row is malformed")
        if candidate_id in seen_ids or relative.as_posix() in seen_videos:
            raise VSTPrefixSemanticError("selection IDs/videos must be unique")
        seen_ids.add(candidate_id)
        seen_videos.add(relative.as_posix())
        source = (media_root / relative).resolve()
        if not source.is_file() or source.is_symlink():
            continue
        row["source_video"] = str(source)
        by_count[answer].append(row)

    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    required = train_per_count + holdout_per_count
    for answer in TARGET_COUNTS:
        values = by_count[answer]
        values.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}|{answer}|{row['candidate_id']}".encode()
            ).digest()
        )
        if len(values) < required:
            raise VSTPrefixSemanticError(
                f"count {answer} has {len(values)} available videos; need {required}"
            )
        for row in values[:holdout_per_count]:
            value = dict(row)
            value["split"] = "policy_holdout"
            holdout.append(value)
        for row in values[holdout_per_count:required]:
            value = dict(row)
            value["split"] = "train"
            train.append(value)
    return (
        sorted(train, key=lambda row: str(row["candidate_id"])),
        sorted(holdout, key=lambda row: str(row["candidate_id"])),
    )


def uniform_causal_indices(
    *, source_total: int, source_fps: float, query_time: float, frames: int
) -> tuple[list[int], list[float]]:
    """Return exactly ``frames`` uniform indices whose timestamps are causal."""

    if (
        source_total <= 0
        or not math.isfinite(source_fps)
        or source_fps <= 0
        or not math.isfinite(query_time)
        or query_time <= 0
        or frames <= 1
    ):
        raise VSTPrefixSemanticError("invalid video sampling metadata")
    # ceil(t * fps) - 1 is strictly earlier than t, including exact boundaries.
    last = min(source_total - 1, max(0, math.ceil(query_time * source_fps) - 1))
    if last == 0:
        indices = [0] * frames
    else:
        indices = [
            int(round(position * last / (frames - 1))) for position in range(frames)
        ]
    timestamps = [index / source_fps for index in indices]
    if any(timestamp >= query_time + 1e-9 for timestamp in timestamps):
        raise VSTPrefixSemanticError("prefix sampler exposed a future frame")
    return indices, timestamps


def project_semantic_rows(
    row: Mapping[str, Any],
    *,
    relative_paths: Sequence[str],
    frame_timestamps: Sequence[float],
    source_sha256: str,
    selection_sha256: str,
) -> list[dict[str, Any]]:
    """Project one grounded count state into open and balanced-MCQ rows."""

    if len(relative_paths) != 32 or len(frame_timestamps) != 32:
        raise VSTPrefixSemanticError("prefix projection requires exactly 32 frames")
    answer = int(row["answer"])
    if answer not in TARGET_COUNTS:
        raise VSTPrefixSemanticError("count answer is outside 0--10")
    candidate_id = str(row["candidate_id"])
    split = str(row["split"])
    if split not in {"train", "policy_holdout"}:
        raise VSTPrefixSemanticError("invalid semantic split")
    images = [
        {
            "type": "image",
            "image": path,
            "frame_id": stable_id(
                "vstprefixframe", candidate_id, index, timestamp, digest_length=20
            ),
        }
        for index, (path, timestamp) in enumerate(
            zip(relative_paths, frame_timestamps, strict=True)
        )
    ]
    parent = {
        "kind": "vst_count_metadata_state",
        "id": stable_id("vstprefixsource", candidate_id, digest_length=20),
        "sha256": selection_sha256,
    }
    group_id = stable_id("vstprefixgroup", candidate_id, digest_length=20)
    question = " ".join(str(row["question"]).split())
    answer_text = str(answer)
    open_row = attach_content_sha256(
        {
            "schema": "streameva.sft_row.v1",
            "uid": stable_id("sftvstprefixopen", candidate_id, digest_length=20),
            "component": "preservation",
            "group_id": group_id,
            "split": split,
            "decision_loss_profile": "full",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *images,
                        {
                            "type": "text",
                            "text": (
                                f"{question}\nAnswer with one integer from 0 to 10 "
                                "and no other text."
                            ),
                        },
                    ],
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_text}],
                    "loss": True,
                },
            ],
            "sample_loss_weight": 1.0,
            "parents": [parent],
        }
    )

    distractors = [value for value in TARGET_COUNTS if value != answer]
    distractors.sort(
        key=lambda value: (
            abs(value - answer),
            hashlib.sha256(f"{candidate_id}|{value}".encode()).digest(),
        )
    )
    options, mcq_answer = rebalance_answer_slot(
        [answer_text, *[str(value) for value in distractors[:3]]],
        answer="A",
        answer_slot=int(hashlib.sha256(candidate_id.encode()).hexdigest(), 16) % 4,
        seed=f"{candidate_id}|prefix32-mcq-v1",
    )
    option_text = "\n".join(
        f"{letter}. {value}"
        for letter, value in zip(OPTION_LETTERS, options, strict=True)
    )
    mcq_prompt = (
        f"{DIRECT_PROMPT_PREFIX}\n\nQuestion: {question}\n\n"
        f"Options:\n{option_text}\n\n{DIRECT_PROMPT_SUFFIX}"
    )
    mcq_row = attach_content_sha256(
        {
            "schema": "streameva.sft_row.v1",
            "uid": stable_id("sftvstprefixmcq", candidate_id, digest_length=20),
            "component": "preservation",
            "group_id": group_id,
            "split": split,
            "decision_loss_profile": "full",
            "messages": [
                {
                    "role": "user",
                    "content": [*images, {"type": "text", "text": mcq_prompt}],
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": mcq_answer}],
                    "loss": True,
                },
            ],
            "sample_loss_weight": 1.0,
            "parents": [parent],
        }
    )
    for output in (open_row, mcq_row):
        validate_row(output)
    # Bind source pixels in the manifest-facing auxiliary state without leaking
    # an absolute path into the strict trainer row.
    del source_sha256
    return [open_row, mcq_row]


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    selection = args.selection.resolve()
    preflight_path = args.preflight.resolve()
    media_root = args.media_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.frames != 32:
        raise VSTPrefixSemanticError("this release is frozen to exactly 32 frames")
    if not 1 <= args.workers <= 32:
        raise VSTPrefixSemanticError("workers must lie in [1, 32]")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    declaration = preflight.get("selection_artifact", {})
    if (
        preflight.get("format") != PREFLIGHT_FORMAT
        or preflight.get("status") != "pass_metadata_firewall_and_selection"
        or preflight.get("contracts", {}).get("basename_hits_admitted") != 0
        or preflight.get("contracts", {}).get("question_hits_admitted") != 0
        or declaration.get("sha256") != file_sha256(selection)
    ):
        raise VSTPrefixSemanticError("metadata preflight is not valid or hash-bound")
    rows = [
        json.loads(line)
        for line in selection.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train, holdout = select_available_states(
        rows,
        media_root=media_root,
        train_per_count=args.train_per_count,
        holdout_per_count=args.holdout_per_count,
        seed=args.seed,
    )
    selected = [*train, *holdout]
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial.", dir=output.parent))
    try:
        media_out = scratch / "media"

        def decode(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
            return str(row["candidate_id"]), _decode_prefix(row, media_out, args.frames)

        decoded: dict[str, dict[str, Any]] = {}
        errors: list[dict[str, str]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(decode, row): row for row in selected}
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                try:
                    candidate_id, state = future.result()
                    decoded[candidate_id] = state
                except Exception as exc:  # pragma: no cover - remote media failures
                    errors.append(
                        {
                            "candidate_id": str(row["candidate_id"]),
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
        write_jsonl(errors, scratch / "decode_errors.jsonl")
        if errors or len(decoded) != len(selected):
            raise VSTPrefixSemanticError(f"prefix decode failed for {len(errors)} rows")

        train_rows: list[dict[str, Any]] = []
        holdout_rows: list[dict[str, Any]] = []
        source_inventory = []
        selection_sha = file_sha256(selection)
        for row in selected:
            state = decoded[str(row["candidate_id"])]
            source_sha = file_sha256(str(row["source_video"]))
            source_inventory.append(
                {
                    "candidate_id": row["candidate_id"],
                    "source_video_relative": row["source_video_relative"],
                    "source_sha256": source_sha,
                    "split": row["split"],
                    "query_time": row["query_time"],
                    "answer": row["answer"],
                    "frame_indices": state["frame_indices"],
                    "frame_timestamps": state["frame_timestamps"],
                }
            )
            projected = project_semantic_rows(
                row,
                relative_paths=state["relative_paths"],
                frame_timestamps=state["frame_timestamps"],
                source_sha256=source_sha,
                selection_sha256=selection_sha,
            )
            (train_rows if row["split"] == "train" else holdout_rows).extend(projected)
        write_jsonl(train_rows, scratch / "trainer/sft.jsonl", sort_key="uid")
        write_jsonl(
            holdout_rows, scratch / "holdout/policy_holdout.jsonl", sort_key="uid"
        )
        write_jsonl(
            source_inventory,
            scratch / "source_inventory.jsonl",
            sort_key="candidate_id",
        )
        media_files = sorted(path for path in media_out.rglob("*.jpg") if path.is_file())
        manifest = {
            "format": FORMAT,
            "status": "materialized_internal_research_candidate",
            "training_allowed": True,
            "public_release_allowed": False,
            "protocol": {
                "causal_prefix_only": True,
                "frames": 32,
                "per_frame_max_pixels": PREFIX_FRAME_MAX_PIXELS,
                "sampling": "exact_uniform_indices_over_[0,query_time)",
                "visual_input_mode": "independent_image_frames",
            },
            "counts": {
                "train_source_states": len(train),
                "holdout_source_states": len(holdout),
                "train_rows": len(train_rows),
                "holdout_rows": len(holdout_rows),
                "media_files": len(media_files),
                "train_by_answer": dict(
                    sorted(collections.Counter(str(row["answer"]) for row in train).items())
                ),
                "holdout_by_answer": dict(
                    sorted(
                        collections.Counter(str(row["answer"]) for row in holdout).items()
                    )
                ),
            },
            "inputs": {
                "selection_sha256": selection_sha,
                "preflight_sha256": file_sha256(preflight_path),
                "builder_sha256": file_sha256(Path(__file__)),
            },
            "artifacts": {
                "trainer/sft.jsonl": file_sha256(scratch / "trainer/sft.jsonl"),
                "holdout/policy_holdout.jsonl": file_sha256(
                    scratch / "holdout/policy_holdout.jsonl"
                ),
                "source_inventory.jsonl": file_sha256(
                    scratch / "source_inventory.jsonl"
                ),
                "decode_errors.jsonl": file_sha256(scratch / "decode_errors.jsonl"),
            },
            "contracts": {
                "all_labels_metadata_derived": True,
                "all_pixels_strictly_before_query": True,
                "benchmark_firewall_inherited": True,
                "holdout_source_disjoint": True,
                "no_agent_actions": True,
                "no_private_state_actor_targets": True,
                "source_videos_globally_unique": True,
                "teacher_used_for_labels": False,
            },
            "license": {
                "public_media_redistribution_claimed": False,
                "release_mode": "recipe_plus_source_checksums",
            },
        }
        (scratch / "manifest.json").write_bytes(canonical_json_line(manifest))
        os.replace(scratch, output)
        return {**manifest, "output": str(output)}
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise


def _decode_prefix(
    row: Mapping[str, Any], media_root: Path, frames: int
) -> dict[str, Any]:
    from decord import VideoReader

    source = Path(str(row["source_video"]))
    reader = VideoReader(str(source))
    try:
        fps = float(reader.get_avg_fps())
        total = len(reader)
        indices, timestamps = uniform_causal_indices(
            source_total=total,
            source_fps=fps,
            query_time=float(row["query_time"]),
            frames=frames,
        )
        arrays = reader.get_batch(indices).asnumpy()
    finally:
        del reader
    relative_paths = []
    for array in arrays:
        image = Image.fromarray(array)
        height, width = image.height, image.width
        resized_height, resized_width = _rounded_size(height, width)
        if (resized_height, resized_width) != (height, width):
            image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95, subsampling=0, optimize=False)
        payload = buffer.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        relative = Path("media") / digest[:2] / f"{digest}.jpg"
        destination = media_root.parent / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("xb") as handle:
                handle.write(payload)
        except FileExistsError:
            pass
        if file_sha256(destination) != digest:
            raise VSTPrefixSemanticError("decoded frame hash mismatch")
        relative_paths.append(relative.as_posix())
    return {
        "relative_paths": relative_paths,
        "frame_indices": indices,
        "frame_timestamps": timestamps,
    }


def _rounded_size(height: int, width: int) -> tuple[int, int]:
    factor = 28
    max_pixels = PREFIX_FRAME_MAX_PIXELS
    max_side = PREFIX_FRAME_MAX_SIDE
    scale = min(1.0, max_side / max(height, width))
    if height * width * scale * scale > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / (height * width)))
    new_height = max(factor, int(round(height * scale / factor)) * factor)
    new_width = max(factor, int(round(width * scale / factor)) * factor)
    while new_height * new_width > max_pixels:
        if new_height >= new_width and new_height > factor:
            new_height -= factor
        elif new_width > factor:
            new_width -= factor
        else:
            break
    return new_height, new_width


if __name__ == "__main__":
    main()
