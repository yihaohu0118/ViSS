#!/usr/bin/env python3
"""Run the locked recent4_full_v1 protocol with an auditable Qwen3-VL model.

The runner is intentionally direct: one causal four-frame observation, one
model call, and no retrieval, global frames, inspection, or persistent memory.
It writes every sampled source-frame index and enough frozen identity metadata
to reject mixed-protocol shard merges.
"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from viss_core.benchmarks import (
    load_queries,
    resolve_video_path,
    score_predictions,
)
from viss_core.evaluation_protocol import (
    LOCKED_PROTOCOL,
    load_locked_protocol,
    validate_locked_protocol,
)
from viss_core.predecoded_cache import PredecodedRecent4Cache
from viss_core.temporal_budget import (
    TemporalBudgetConfig,
    build_frame_signals,
    select_temporal_budget,
)


PROMPT_CONTRACT = (
    "StreamingBench uses the SimpleStream release MCQ user template. OVO "
    "backward/realtime uses the SimpleStream release one-letter user template; "
    "OVO forward uses the frozen official task wrapper. No system message."
)
STREAMING_PROMPT_TEMPLATE = (
    "You are an advanced video question-answering AI assistant. "
    "You have been provided with some frames from the video and a multiple-choice question. "
    "Your task is to analyze the video and provide the best answer.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Only give the best option's letter (A, B, C, or D) directly."
)

FRAME_POLICIES = (
    "recent4",
    "wide8",
    "wide12",
    "wide16",
    "prefix8",
    "prefix16",
    "prefix32",
    "prefix64",
    "causalbudget20",
    "causalbudget32r8",
    "causalhybrid32",
    "causalhybrid32r8",
)
FRAME_POLICY_FPS = {
    "recent4": 1.0,
    "wide8": 0.5,
    "wide12": 1.0 / 3.0,
    "wide16": 0.25,
}
FRAME_POLICY_NUM_FRAMES = {
    "recent4": 4,
    "wide8": 4,
    "wide12": 4,
    "wide16": 4,
    "prefix8": 8,
    "prefix16": 16,
    "prefix32": 32,
    "prefix64": 64,
    "causalbudget20": 20,
    "causalbudget32r8": 32,
    "causalhybrid32": 32,
    "causalhybrid32r8": 32,
}
VISUAL_INPUT_MODES = (
    "image-list",
    "native-video",
    "hybrid-history-video-recent-images",
    "storyboard-recent-images",
    "storyboard-keyframes-recent-images",
    "lowres-history-recent-images",
)

TEMPORAL_REVIEW_TEMPLATE = (
    "Temporal review: the initial high-resolution recent view proposed {recent}, "
    "and the independently selected history view proposed {history}. Both proposals "
    "may be wrong. Re-examine the currently visible history and recent frames, then "
    "answer the original question using the required output format."
)
ANONYMOUS_TEMPORAL_REVIEW_TEMPLATE = (
    "Temporal review: two independent visual passes proposed Candidate 1: {first} "
    "and Candidate 2: {second}. Either or both candidates may be wrong. "
    "Re-examine all currently visible history and recent frames without assuming "
    "either candidate is privileged, then answer the original question using the "
    "required output format."
)
ANONYMOUS_MULTI_CANDIDATE_TEMPLATE = (
    "Independent causal visual passes proposed the following unordered candidate "
    "answers: {candidates}. The candidates are hints, not votes; any or all may be "
    "wrong. Verify them against the currently visible pixels and temporal order, "
    "then answer the original question using the required output format."
)
TEMPORAL_DELIBERATION_TEMPLATE = (
    "Analyze the visible causal sequence step by step before deciding. Check the "
    "events, state changes, repetitions, and temporal order against every choice. "
    "End with one final line in the exact form `Answer: X`, where X is the requested "
    "option letter, Yes/No value, or number."
)
CANDIDATE_EVIDENCE_DELIBERATION_TEMPLATE = (
    "Treat every proposed candidate as an untrusted hypothesis. For each candidate, "
    "locate concrete visible evidence and eliminate candidates that conflict with the "
    "pixels or event order. Do not use proposal frequency or candidate position as "
    "evidence. Be terse: write at most one line of at most ten words per candidate, "
    "use at most four evidence lines total, and do not narrate. You must end with one "
    "final line in the exact form `Answer: X`, where X is the requested option letter, "
    "Yes/No value, or number."
)
CANDIDATE_BINARY_VERIFICATION_TEMPLATE = (
    "Candidate answer under isolated verification: {candidate}\n"
    "Using only the visible causal video evidence, decide whether this candidate "
    "is the best supported answer to the original problem after comparing it "
    "against the listed alternatives. Answer exactly Yes or No."
)
CANDIDATE_REJECTION_VERIFICATION_TEMPLATE = (
    "Candidate answer under isolated rejection check: {candidate}\n"
    "Using only the visible causal video evidence, decide whether this candidate "
    "should be rejected as the best answer to the original problem after comparing "
    "it against the listed alternatives. Answer exactly Yes or No."
)
CANDIDATE_PAIRWISE_VERIFICATION_TEMPLATE = (
    "Pairwise candidate verification:\n"
    "Choice A: {first}\n"
    "Choice B: {second}\n"
    "Using only the visible causal video evidence, decide which of these two "
    "choices is better supported as the answer to the original problem after "
    "checking the listed alternatives. Answer exactly A or B."
)
CANDIDATE_REVISIT_TEMPLATE = (
    "Candidate answer under isolated verification: {candidate}\n"
    "The {history_count} historical frames are numbered 1 through "
    "{history_count} in chronological order and are followed by the latest "
    "frames. Select the historical source-resolution frames most useful for "
    "verifying or refuting this candidate against the alternatives. Do not "
    "decide the answer yet. Return exactly {revisit_count} distinct frame "
    "numbers in the form `Frames: i,j,...` and nothing else."
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("streamingbench", "ovo"))
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--processor", type=Path, default=None)
    parser.add_argument("--model-manifest", type=Path, default=None)
    parser.add_argument(
        "--model-identity-sha256",
        help="Previously sealed model-manifest identity; no file checksum is recomputed.",
    )
    parser.add_argument("--protocol", type=Path, default=LOCKED_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument(
        "--uids",
        nargs="+",
        help="Optional exact UID subset for smoke tests; canonical export order is retained.",
    )
    parser.add_argument(
        "--uids-file",
        type=Path,
        help=(
            "Optional newline-delimited exact UID subset. This avoids very long "
            "command lines for balanced formal shards; canonical export order is retained."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size. Batch size does not alter prompts or sampled frames.",
    )
    parser.add_argument(
        "--flush-every-batches",
        type=int,
        default=1,
        help="Atomically checkpoint the resumable JSON after this many inference batches.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument(
        "--record-choice-confidence",
        action="store_true",
        help=(
            "Record first-generation-step A-Z choice probabilities for an "
            "auditable, task-agnostic uncertainty trigger. This does not alter "
            "the generated answer or select a benchmark/task-specific policy."
        ),
    )
    parser.add_argument(
        "--frame-policy",
        choices=FRAME_POLICIES,
        default="recent4",
        help=(
            "Training-free four-frame temporal allocation. recent4 is the locked "
            "baseline; all alternatives remain causal and expose exactly four "
            "independent images but are experimental, non-comparable policies."
        ),
    )
    parser.add_argument(
        "--visual-input-mode",
        choices=VISUAL_INPUT_MODES,
        default="image-list",
        help=(
            "image-list reproduces the locked SimpleStream-compatible protocol. "
            "native-video feeds the identical decoded frame set through Qwen3-VL's "
            "video processor with original frame indices and source FPS, preserving "
            "native timestamp tokens and video_grid_thw. "
            "hybrid-history-video-recent-images sends selected historical frames "
            "through that timestamp-aware video path while retaining the final four "
            "frames as independent high-resolution images. Both non-default modes are "
            "auditable experimental A/Bs and are never marked comparable to "
            "recent4_full_v1."
        ),
    )
    parser.add_argument(
        "--history-max-edge",
        type=int,
        default=336,
        help=(
            "Maximum pixel edge for historical frames in the uniform low-resolution "
            "history interface. Recent frames retain their decoded resolution."
        ),
    )
    parser.add_argument(
        "--semantic-revisit-count",
        type=int,
        default=0,
        help=(
            "If positive, run one uniform query-conditioned planning turn over "
            "the low-resolution causal view, then revisit this many historical "
            "frames at source resolution before answering. This uses no task, "
            "benchmark, prediction, or gold routing fields."
        ),
    )
    parser.add_argument(
        "--semantic-revisit-planner-max-tokens",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--semantic-revisit-planner-mode",
        choices=("image-list", "storyboard16"),
        default="image-list",
        help=(
            "Uniform visual interface for the semantic-revisit planning turn. "
            "storyboard16 packs chronological thumbnails into globally numbered "
            "16-frame contact sheets before the model chooses source-resolution "
            "revisits; it changes no causal frame or task routing decision."
        ),
    )
    parser.add_argument(
        "--review-recent-result",
        type=Path,
        help="Optional prior recent-view result for router-free temporal review.",
    )
    parser.add_argument(
        "--review-history-result",
        type=Path,
        help="Optional prior history-view result for router-free temporal review.",
    )
    parser.add_argument(
        "--review-proposal-label-mode",
        choices=("source-aware", "anonymous-hash-counterbalanced"),
        default="source-aware",
        help=(
            "source-aware reproduces the historical reviewer prompt. The anonymous "
            "mode removes recent/history role labels and deterministically swaps the "
            "two candidate positions by UID hash, preventing a fixed proposal-source "
            "or prompt-position prior."
        ),
    )
    parser.add_argument(
        "--agent-candidate-result",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional complete model-generated prediction artifact. Two or three "
            "such artifacts may provide anonymous candidate hints to the uniform "
            "semantic-revisit answer call. Only UID and prediction are consumed."
        ),
    )
    parser.add_argument(
        "--deliberation-mode",
        choices=(
            "none",
            "temporal-verify",
            "candidate-verify",
            "candidate-evidence-verify",
        ),
        default="none",
        help=(
            "Optional task-blind temporal deliberation prompt. It changes no frames "
            "or routing decision and requires an explicit final Answer line. "
            "candidate-verify applies the same reasoning contract after anonymous "
            "candidate hints from the uniformly acquired views. candidate-evidence-"
            "verify additionally requires visible evidence and elimination for every "
            "untrusted candidate under the same fixed pixel contract."
        ),
    )
    parser.add_argument(
        "--candidate-binary-verify",
        action="store_true",
        help=(
            "After the uniform direct answer pass, independently score every "
            "source-anonymous candidate with the same visible pixels and one "
            "Yes/No support prompt, then commit the highest-supported candidate. "
            "This consumes no task, benchmark, score, or gold field."
        ),
    )
    parser.add_argument(
        "--candidate-binary-verify-batch-size",
        type=int,
        default=2,
        help="Maximum isolated candidate checks per verifier forward pass.",
    )
    parser.add_argument(
        "--candidate-antipodal-verify",
        action="store_true",
        help=(
            "Score every anonymous candidate with matched support and rejection "
            "prompts over identical pixels, then rank by the mean of support-Yes "
            "and rejection-No probabilities. This removes prompt-polarity bias "
            "without task, benchmark, score, or gold inputs."
        ),
    )
    parser.add_argument(
        "--candidate-pairwise-verify",
        action="store_true",
        help=(
            "After isolated candidate scoring, compare every anonymous candidate "
            "pair twice with reversed A/B order over identical pixels. Commit by "
            "the mean order-symmetric pairwise support, without task, benchmark, "
            "score, or gold inputs."
        ),
    )
    parser.add_argument(
        "--candidate-conditioned-revisit-count",
        type=int,
        default=0,
        help=(
            "If positive, give every anonymous candidate the same bounded "
            "hypothesis-conditioned historical revisit before binary scoring."
        ),
    )
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--decode-cache-entries",
        type=int,
        default=8,
        help=(
            "Cache exact recent-window decodes for repeated adjacent benchmark "
            "queries. This only avoids duplicate video I/O and does not change "
            "the locked sampling policy; use 0 to disable."
        ),
    )
    parser.add_argument(
        "--video-reader-cache-entries",
        type=int,
        default=2,
        help=(
            "Keep this many decord VideoReader objects open per evaluator. "
            "StreamingBench has several adjacent query times per source video; "
            "reusing the indexed reader avoids reparsing the same large MP4. "
            "Frame indices and resizing remain delegated to qwen-vl-utils."
        ),
    )
    parser.add_argument(
        "--video-reader-num-threads",
        type=int,
        default=2,
        help=(
            "Bound decoder threads per decord VideoReader. Decord's automatic "
            "setting can create hundreds of threads on large CPU hosts; an "
            "explicit positive value changes only decoder concurrency, not "
            "frame indices or pixels."
        ),
    )
    parser.add_argument(
        "--causalbudget-index-plan",
        type=Path,
        help=(
            "Optional query-independent precomputed causalbudget20 frame-index "
            "plan. The model-visible source frames are still decoded by the "
            "same exact FFmpeg path; only the redundant 16x16 probe pass is "
            "reused after an index-equivalence audit."
        ),
    )
    parser.add_argument(
        "--allow-question-conditioned-index-plan",
        action="store_true",
        help=(
            "Allow a frozen frame-index plan produced by the same uniform "
            "question-conditioned visual planner for every sample. Task/group, "
            "gold, prior answer predictions, and future frames remain forbidden."
        ),
    )
    parser.add_argument(
        "--decode-prefetch-batches",
        type=int,
        default=0,
        help=(
            "For a frozen causalbudget index plan, decode this many future "
            "inference batches concurrently while the GPU handles the current "
            "batch. This is an execution-only optimization: the same exact "
            "FFmpeg decoder receives the same frozen source indices."
        ),
    )
    parser.add_argument(
        "--predecoded-cache",
        type=Path,
        help=(
            "CPU-materialized, DATA_READY exact recent4 mmap cache. When set, "
            "the GPU lease performs no reusable video decoding."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_causalbudget_index_plan(
    path: Path,
    *,
    benchmark: str,
    wanted: list[str],
    model_frames: int = 20,
    recent_frames: int = 4,
    allow_question_conditioned: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load a complete question-blind execution cache for selected UIDs."""

    resolved = path.resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    contracts = value.get("contracts") or {}
    if (
        value.get("format")
        not in {
            "iclr_video.causalbudget20_index_plan.v0",
            "iclr_video.causalbudget_index_plan.v1",
        }
        or value.get("status")
        not in {
            "complete_query_independent_execution_cache",
            "complete_question_conditioned_execution_cache",
        }
        or value.get("benchmark") != benchmark
        or contracts.get("question_text_read_for_plan")
        is not bool(allow_question_conditioned)
        or contracts.get("task_or_group_read_for_plan") is not False
        or contracts.get("model_prediction_read_for_plan") is not False
        or contracts.get("gold_answer_read") is not False
        or contracts.get("future_frames_read") is not False
    ):
        raise ValueError("causalbudget20 index plan contract diverged")
    rows = {str(row.get("uid") or ""): row for row in value.get("rows") or []}
    if len(rows) != len(value.get("rows") or []) or any(uid not in rows for uid in wanted):
        raise ValueError("causalbudget20 index plan lacks exact requested coverage")
    if value.get("format") == "iclr_video.causalbudget20_index_plan.v0":
        if (model_frames, recent_frames) != (20, 4):
            raise ValueError("legacy causalbudget20 plan cannot serve another budget")
    elif (
        int(value.get("model_frames", -1)) != model_frames
        or int(value.get("recent_frames", -1)) != recent_frames
    ):
        raise ValueError("causalbudget index plan frame contract diverged")
    return (
        {uid: rows[uid] for uid in wanted},
        {
            "path": str(resolved),
            "sha256": (
                sha256_file(resolved)
                if value.get("format") == "iclr_video.causalbudget20_index_plan.v0"
                else None
            ),
            "checksum_skipped": value.get("format")
            == "iclr_video.causalbudget_index_plan.v1",
            "question_conditioned": bool(allow_question_conditioned),
        },
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield value


def selected_uids(
    path: Path,
    start: int,
    end: int,
    requested: list[str] | None = None,
) -> list[str]:
    uids = [str(row.get("uid") or "") for row in iter_jsonl(path)]
    if any(not uid for uid in uids) or len(set(uids)) != len(uids):
        raise ValueError(f"{path}: missing or duplicate uid")
    if requested is not None:
        requested_set = {str(uid) for uid in requested}
        if len(requested_set) != len(requested):
            raise ValueError("--uids contains duplicates")
        unknown = sorted(requested_set - set(uids))
        if unknown:
            raise ValueError(f"--uids absent from frozen export: {unknown}")
        return [uid for uid in uids if uid in requested_set]
    stop = len(uids) if end < 0 else min(end, len(uids))
    return uids[max(0, start) : stop]


def rounded_size(
    height: int, width: int, *, factor: int, max_pixels: int, max_side: int
) -> tuple[int, int]:
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


def recent_window_bounds(
    visible_until: float, recent_frames: int, sampling_fps: float
) -> tuple[float, float]:
    end = max(0.0, float(visible_until))
    start = max(0.0, end - float(recent_frames) / float(sampling_fps))
    return start, end


def frame_policy_sampling_fps(policy: str) -> float:
    """Map each exact-uniform policy to the official decoder's nominal FPS."""

    try:
        return FRAME_POLICY_FPS[policy]
    except KeyError as error:
        raise ValueError(f"unsupported frame policy: {policy}") from error


def resolve_frame_policy(requested_policy: str) -> str:
    """Validate one fixed, question-independent model-only frame policy."""

    if requested_policy not in FRAME_POLICY_NUM_FRAMES:
        raise ValueError(f"unsupported frame policy: {requested_policy}")
    return requested_policy


def build_visual_messages(
    *,
    frames: list[Any],
    prompt: str,
    visual_input_mode: str,
    history_frames: list[Any] | None = None,
    recent_frames: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Build one modality-explicit Qwen chat turn without hidden routing.

    ``native-video`` deliberately uses one video placeholder for the complete
    ordered frame set.  The hybrid mode uses one video placeholder for history
    followed by independent recent-image placeholders. Pixel tensors and
    immutable source timing metadata are supplied later to the processor; no
    file path is placed in the prompt and no question-dependent policy decision
    occurs here.
    """

    if visual_input_mode in {
        "image-list",
        "storyboard-recent-images",
        "storyboard-keyframes-recent-images",
        "lowres-history-recent-images",
    }:
        visual_content = [
            {"type": "image", "image": frame} for frame in frames
        ]
    elif visual_input_mode == "native-video":
        if not frames:
            raise ValueError("native-video requires at least one decoded frame")
        visual_content = [{"type": "video"}]
    elif visual_input_mode == "hybrid-history-video-recent-images":
        history_frames = list(history_frames or [])
        recent_frames = list(recent_frames or [])
        if not recent_frames:
            raise ValueError("hybrid visual input requires recent image frames")
        visual_content = []
        if history_frames:
            visual_content.append({"type": "video"})
        visual_content.extend(
            {"type": "image", "image": frame} for frame in recent_frames
        )
    else:
        raise ValueError(f"unsupported visual input mode: {visual_input_mode}")
    return [
        {
            "role": "user",
            "content": [*visual_content, {"type": "text", "text": prompt}],
        }
    ]


def compose_temporal_storyboard(
    frames: list[Any],
    timestamps: list[float],
    *,
    columns: int = 4,
    tile_width: int = 224,
    tile_height: int = 126,
    label_height: int = 20,
    label_start: int = 1,
) -> Any:
    """Render a query-independent chronological contact sheet.

    Every selected causal frame appears once and in source order.  The only
    overlay is its already-public relative timestamp, so the operation exposes
    no question, task, benchmark, prediction, or gold-answer information.
    """

    if not frames or len(frames) != len(timestamps):
        raise ValueError("storyboard frames and timestamps must be aligned")
    if columns <= 0 or min(tile_width, tile_height, label_height, label_start) <= 0:
        raise ValueError("storyboard geometry must be positive")
    from PIL import Image, ImageDraw

    rows = math.ceil(len(frames) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (source, timestamp) in enumerate(zip(frames, timestamps, strict=True)):
        image = source.convert("RGB")
        image.thumbnail((tile_width, tile_height))
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        offset_x = x + (tile_width - image.width) // 2
        offset_y = y + (tile_height - image.height) // 2
        sheet.paste(image, (offset_x, offset_y))
        draw.rectangle(
            (x, y + tile_height, x + tile_width - 1, y + tile_height + label_height - 1),
            fill="black",
        )
        draw.text(
            (x + 4, y + tile_height + 3),
            f"{label_start + index:02d}  t={float(timestamp):.1f}s",
            fill="white",
        )
    return sheet


def downscale_temporal_history(
    frames: list[Any],
    frame_indices: list[int],
    recent_frame_indices: set[int],
    *,
    max_edge: int,
) -> list[Any]:
    """Downscale every historical image by one fixed, query-blind rule."""

    if not frames or len(frames) != len(frame_indices) or max_edge <= 0:
        raise ValueError("low-resolution history inputs are invalid")
    output = []
    for frame, frame_index in zip(frames, frame_indices, strict=True):
        if int(frame_index) in recent_frame_indices:
            output.append(frame)
            continue
        image = frame.convert("RGB").copy()
        image.thumbnail((max_edge, max_edge))
        output.append(image)
    return output


def render_semantic_revisit_prompt(
    answer_prompt: str,
    *,
    history_count: int,
    revisit_count: int,
    planner_mode: str = "image-list",
) -> str:
    """Request one bounded evidence revisit with the same policy for every query."""

    count = min(history_count, revisit_count)
    interface = (
        "The chronological contact sheets label every frame with its global frame "
        "number and relative timestamp. "
        if planner_mode == "storyboard16"
        else ""
    )
    if planner_mode not in {"image-list", "storyboard16"}:
        raise ValueError(f"unsupported semantic-revisit planner mode: {planner_mode}")
    return (
        f"Original problem:\n{answer_prompt}\n\n"
        f"{interface}"
        f"The {history_count} historical video frames are numbered 1 through "
        f"{history_count} in chronological order and are followed by the latest "
        "frames. Select the historical frames whose source-resolution pixels are "
        "most useful for resolving the original problem above. Do not answer the "
        f"problem yet. Return exactly {count} distinct frame numbers in the form "
        "`Frames: i,j,...` and nothing else."
    )


def parse_semantic_revisit_positions(
    raw: str,
    *,
    history_count: int,
    revisit_count: int,
    fallback_positions: list[int],
) -> list[int]:
    """Parse 1-indexed planner output and fill failures by one fixed fallback."""

    target = min(history_count, revisit_count)
    if target <= 0:
        return []
    match = re.search(r"\bframes?\s*:\s*([0-9][0-9,\s]*)", raw, re.IGNORECASE)
    parsed = []
    if match:
        parsed = [int(value) for value in re.findall(r"\d+", match.group(1))]
    selected: list[int] = []
    for position in [*parsed, *fallback_positions]:
        if 1 <= position <= history_count and position not in selected:
            selected.append(position)
        if len(selected) == target:
            break
    if len(selected) != target:
        raise ValueError("semantic revisit fallback does not cover history")
    return sorted(selected)


def pad_native_video_temporal_axis(
    frames: list[Any],
    frame_indices: list[int],
    *,
    temporal_patch_size: int,
) -> tuple[list[Any], list[int], int]:
    """Pad a causal singleton with repeats of its last already-visible frame.

    Qwen's native-video processor requires at least one complete temporal
    patch. Repetition changes tensor shape only: it does not add a new source
    frame or move the causal frontier.
    """

    if temporal_patch_size <= 0:
        raise ValueError("temporal patch size must be positive")
    if not frames or len(frames) != len(frame_indices):
        raise ValueError("native-video frames and indices must be aligned")
    repeats = max(0, temporal_patch_size - len(frames))
    if repeats == 0:
        return frames, frame_indices, 0
    return (
        [*frames, *([frames[-1]] * repeats)],
        [*frame_indices, *([frame_indices[-1]] * repeats)],
        repeats,
    )


def partition_hybrid_frames(
    frames: list[Any],
    sampling: dict[str, Any],
    *,
    temporal_patch_size: int,
) -> tuple[list[Any], list[int], list[Any], list[int]]:
    """Split selected frames into timestamp-video history and recent images.

    Qwen's video processor cannot encode fewer than ``temporal_patch_size``
    frames. A singleton history is therefore preserved, in chronological
    order, as an independent image instead of being duplicated or discarded.
    The final return value records those fallback indices for the audit log.
    """

    if temporal_patch_size <= 0:
        raise ValueError("temporal patch size must be positive")
    recent_ids = {
        int(value) for value in sampling.get("recent_frame_indices") or []
    }
    history_frames: list[Any] = []
    history_indices: list[int] = []
    recent_frames: list[Any] = []
    for frame, frame_index in zip(
        frames, sampling.get("frame_indices") or [], strict=True
    ):
        if int(frame_index) in recent_ids:
            recent_frames.append(frame)
        else:
            history_frames.append(frame)
            history_indices.append(int(frame_index))
    fallback_indices: list[int] = []
    if history_frames and len(history_frames) < temporal_patch_size:
        fallback_indices = list(history_indices)
        recent_frames = [*history_frames, *recent_frames]
        history_frames = []
        history_indices = []
    return history_frames, history_indices, recent_frames, fallback_indices


def render_temporal_review_prompt(
    base_prompt: str,
    *,
    recent_prediction: str,
    history_prediction: str,
    uid: str | None = None,
    label_mode: str = "source-aware",
) -> str:
    proposals = [recent_prediction.strip(), history_prediction.strip()]
    if any(
        not value
        or len(value) > 16
        or any(character in value for character in "\r\n\t")
        for value in proposals
    ):
        raise ValueError("temporal review proposal is empty or unsafe")
    if label_mode == "source-aware":
        review = TEMPORAL_REVIEW_TEMPLATE.format(
            recent=proposals[0], history=proposals[1]
        )
    elif label_mode == "anonymous-hash-counterbalanced":
        if not uid:
            raise ValueError("anonymous temporal review requires a non-empty UID")
        swap = int(hashlib.sha256(uid.encode()).hexdigest(), 16) % 2 == 1
        ordered = list(reversed(proposals)) if swap else proposals
        review = ANONYMOUS_TEMPORAL_REVIEW_TEMPLATE.format(
            first=ordered[0], second=ordered[1]
        )
    else:
        raise ValueError(f"unsupported temporal review label mode: {label_mode}")
    return f"{base_prompt}\n\n{review}"


def render_anonymous_candidate_prompt(
    base_prompt: str,
    *,
    predictions: list[str],
    uid: str,
) -> tuple[str, list[str]]:
    """Append task-blind, source-anonymous candidate hints in a stable order."""

    if not uid:
        raise ValueError("anonymous candidate review requires a non-empty UID")
    candidates = sorted(set(value.strip() for value in predictions))
    if any(
        not value
        or len(value) > 16
        or any(character in value for character in "\r\n\t")
        for value in candidates
    ):
        raise ValueError("anonymous candidate review contains an unsafe proposal")
    if len(candidates) < 2:
        return base_prompt, candidates
    ordered = sorted(
        candidates,
        key=lambda value: hashlib.sha256(f"{uid}\0{value}".encode()).hexdigest(),
    )
    rendered = ", ".join(
        f"Candidate {index}: {value}"
        for index, value in enumerate(ordered, start=1)
    )
    review = ANONYMOUS_MULTI_CANDIDATE_TEMPLATE.format(candidates=rendered)
    return f"{base_prompt}\n\n{review}", ordered


def strip_direct_answer_instructions(base_prompt: str) -> str:
    """Remove benchmark output instructions before appending a new exact contract."""

    prompt = base_prompt
    prompt = re.sub(
        r"\n?Only give the best option's letter(?: \(A, B, C, or D\))? directly\.",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(
        r"\n?Answer only with [“\"]?Yes[”\"]? or [“\"]?No[”\"]?\.",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(
        r"\n?Provide your answer as a single number[^\n]*",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(
        r"\n?Do not include any additional text or explanation in your response\.",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    return prompt.rstrip()


def render_temporal_deliberation_prompt(base_prompt: str) -> str:
    """Replace direct-only decoding instructions with one shared reasoning contract."""

    prompt = strip_direct_answer_instructions(base_prompt)
    return f"{prompt.rstrip()}\n\n{TEMPORAL_DELIBERATION_TEMPLATE}"


def render_candidate_evidence_deliberation_prompt(base_prompt: str) -> str:
    """Require task-blind, pixel-cited arbitration among anonymous candidates."""

    prompt = strip_direct_answer_instructions(base_prompt)
    return (
        f"{prompt.rstrip()}\n\n{CANDIDATE_EVIDENCE_DELIBERATION_TEMPLATE}"
    )


def render_candidate_display(candidate: str, options: list[str]) -> str:
    """Map a normalized letter back to visible option text without hidden fields."""

    value = candidate.strip()
    if (
        len(value) == 1
        and value.isalpha()
        and 0 <= ord(value.upper()) - ord("A") < len(options)
    ):
        value = options[ord(value.upper()) - ord("A")]
    if (
        not value
        or len(value) > 256
        or any(character in value for character in "\r\n\t")
    ):
        raise ValueError("binary candidate verification contains an unsafe proposal")
    return value


def render_candidate_binary_verification_prompt(
    base_prompt: str,
    *,
    candidate: str,
    options: list[str],
) -> str:
    """Render one source-anonymous candidate check under a uniform pixel contract."""

    value = render_candidate_display(candidate, options)
    prompt = strip_direct_answer_instructions(base_prompt)
    return (
        f"{prompt}\n\n"
        f"{CANDIDATE_BINARY_VERIFICATION_TEMPLATE.format(candidate=value)}"
    )


def render_candidate_rejection_verification_prompt(
    base_prompt: str,
    *,
    candidate: str,
    options: list[str],
) -> str:
    """Render the polarity-reversed half of an antipodal candidate check."""

    value = render_candidate_display(candidate, options)
    prompt = strip_direct_answer_instructions(base_prompt)
    return (
        f"{prompt}\n\n"
        f"{CANDIDATE_REJECTION_VERIFICATION_TEMPLATE.format(candidate=value)}"
    )


def render_candidate_pairwise_verification_prompt(
    base_prompt: str,
    *,
    first_candidate: str,
    second_candidate: str,
    options: list[str],
) -> str:
    """Render one order-specific half of a source-anonymous candidate duel."""

    first = render_candidate_display(first_candidate, options)
    second = render_candidate_display(second_candidate, options)
    if first_candidate.strip() == second_candidate.strip():
        raise ValueError("pairwise candidate verification requires distinct choices")
    prompt = strip_direct_answer_instructions(base_prompt)
    return (
        f"{prompt}\n\n"
        f"{CANDIDATE_PAIRWISE_VERIFICATION_TEMPLATE.format(first=first, second=second)}"
    )


def render_candidate_revisit_prompt(
    base_prompt: str,
    *,
    candidate: str,
    options: list[str],
    history_count: int,
    revisit_count: int,
) -> str:
    """Request a symmetric bounded pixel revisit for one candidate hypothesis."""

    target = min(history_count, revisit_count)
    if target <= 0:
        raise ValueError("candidate-conditioned revisit requires historical frames")
    value = render_candidate_display(candidate, options)
    prompt = strip_direct_answer_instructions(base_prompt)
    return (
        f"{prompt}\n\n"
        f"{CANDIDATE_REVISIT_TEMPLATE.format(candidate=value, history_count=history_count, revisit_count=target)}"
    )


def choose_binary_verified_candidate(
    *, uid: str, candidate_probabilities: dict[str, float]
) -> str:
    """Commit the strongest candidate; resolve exact ties without source ordering."""

    if not uid or not candidate_probabilities:
        raise ValueError("binary candidate verification requires UID and candidates")
    for candidate, probability in candidate_probabilities.items():
        if not candidate or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("binary candidate probability is invalid")
    return max(
        candidate_probabilities,
        key=lambda candidate: (
            candidate_probabilities[candidate],
            hashlib.sha256(f"{uid}\0{candidate}".encode()).hexdigest(),
        ),
    )


def load_temporal_review_proposals(
    *,
    recent_path: Path,
    history_path: Path,
    wanted: list[str],
    benchmark: str,
    annotation_sha256: str,
    model_manifest_sha256: str | None,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Load two model-generated proposal views without reading score/gold data."""

    proposal_maps: dict[str, dict[str, str]] = {}
    inputs: dict[str, Any] = {}
    for name, path in (("recent", recent_path), ("history", history_path)):
        resolved = path.resolve(strict=True)
        value = json.loads(resolved.read_text(encoding="utf-8"))
        if (
            value.get("benchmark") != benchmark
            or value.get("annotation_sha256") != annotation_sha256
            or value.get("model_manifest_sha256") != model_manifest_sha256
            or int(value.get("num_errors", -1)) != 0
        ):
            raise ValueError(f"temporal review {name} identity diverged")
        rows = {
            str(row.get("uid") or ""): str(row.get("prediction") or "").strip()
            for row in value.get("rows") or []
        }
        if any(uid not in rows or not rows[uid] for uid in wanted):
            raise ValueError(f"temporal review {name} lacks complete predictions")
        proposal_maps[name] = {uid: rows[uid] for uid in wanted}
        inputs[name] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "inference_policy": value.get("inference_policy"),
            "source_contains_score_metadata": "score" in value,
            "decision_fields_used": ["rows[].uid", "rows[].prediction"],
            "score_or_gold_fields_used_for_decision": False,
        }
    return proposal_maps, inputs


def load_anonymous_candidate_proposals(
    *,
    paths: list[Path],
    wanted: list[str],
    benchmark: str,
    annotation_sha256: str,
    model_manifest_sha256: str | None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Load predictions only, while rejecting score/gold use in the policy."""

    proposal_maps: list[dict[str, str]] = []
    inputs: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve(strict=True)
        value = json.loads(resolved.read_text(encoding="utf-8"))
        embedded_model_identity = value.get("model_manifest_sha256")
        if (
            value.get("benchmark") != benchmark
            or value.get("annotation_sha256") != annotation_sha256
            or embedded_model_identity not in (None, "", model_manifest_sha256)
            or int(value.get("num_errors", -1)) != 0
        ):
            raise ValueError("anonymous candidate result identity diverged")
        rows = {
            str(row.get("uid") or ""): str(row.get("prediction") or "").strip()
            for row in value.get("rows") or []
        }
        covered = {uid: rows[uid] for uid in wanted if rows.get(uid)}
        if not covered:
            raise ValueError("anonymous candidate result has no requested predictions")
        proposal_maps.append(covered)
        inputs.append(
            {
                "path": str(resolved),
                "inference_policy": value.get("inference_policy"),
                "requested_rows": len(wanted),
                "covered_rows": len(covered),
                "coverage": len(covered) / len(wanted) if wanted else 0.0,
                "sparse_candidate_source": len(covered) != len(wanted),
                "decision_fields_used": ["rows[].uid", "rows[].prediction"],
                "score_or_gold_fields_used_for_decision": False,
                "proposal_source_label_visible_to_actor": False,
            }
        )
    return proposal_maps, inputs


def native_video_metadata_payload(
    sampling: dict[str, Any], *, frame_indices: list[int] | None = None
) -> dict[str, Any]:
    """Return the exact metadata needed for Qwen3-VL timestamp construction."""

    all_indices = [int(value) for value in sampling.get("frame_indices") or []]
    all_timestamps = [
        float(value) for value in sampling.get("frame_timestamps") or []
    ]
    timestamps_by_index = dict(zip(all_indices, all_timestamps, strict=True))
    indices = (
        all_indices
        if frame_indices is None
        else [int(value) for value in frame_indices]
    )
    timestamps = [timestamps_by_index[index] for index in indices]
    if not indices or len(indices) != len(timestamps):
        raise ValueError("native-video frame indices/timestamps are missing or mismatched")
    source_fps = float(sampling["source_fps"])
    if source_fps <= 0:
        raise ValueError("native-video source FPS must be positive")
    reconstructed = [index / source_fps for index in indices]
    if any(abs(left - right) > 1e-3 for left, right in zip(reconstructed, timestamps)):
        raise ValueError("native-video timestamps diverge from source frame indices")
    causal_frontier = float(sampling["effective_visible_until"])
    if max(timestamps) > causal_frontier + 1e-3:
        raise ValueError("native-video metadata crosses the causal frontier")
    return {
        "total_num_frames": int(sampling["source_total_frames"]),
        "fps": source_fps,
        "width": int(sampling["source_width"]),
        "height": int(sampling["source_height"]),
        "duration": float(sampling["source_duration"]),
        "video_backend": "predecoded_exact_causal_indices",
        "frames_indices": indices,
    }


def clamp_visible_until_to_source(
    visible_until: float, source_fps: float, source_total_frames: int
) -> tuple[float, bool]:
    last_frame_time = max(
        0.0,
        (int(source_total_frames) - 1) / max(float(source_fps), 1e-6),
    )
    requested = max(0.0, float(visible_until))
    effective = min(requested, last_frame_time)
    return effective, effective < requested


class DecordReaderPool:
    """LRU pool implementing qwen-vl-utils' decord backend exactly.

    The public backend creates a new ``VideoReader`` for every query. That is
    particularly costly for StreamingBench, where five different query times
    commonly refer to the same long MP4. This callable preserves the upstream
    frame-range, linspace, metadata, and tensor conversion code while only
    reusing the already indexed reader.
    """

    def __init__(self, capacity: int, num_threads: int = 2) -> None:
        if capacity <= 0:
            raise ValueError("reader pool capacity must be positive")
        if num_threads <= 0:
            raise ValueError("reader thread count must be positive")
        self.capacity = int(capacity)
        self.num_threads = int(num_threads)
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()
        # qwen-vl-utils may invoke this backend from decode-prefetch workers.
        # A Decord VideoReader is not safe for concurrent get_batch calls on
        # the same compressed stream (libav's async_lock aborts the process),
        # while independent videos can be decoded safely in parallel.
        self._path_locks: dict[str, threading.RLock] = {}
        self._readers: OrderedDict[str, Any] = OrderedDict()

    def reader(self, path: str) -> tuple[Any, bool]:
        from decord import VideoReader

        with self._lock:
            if path in self._readers:
                reader = self._readers.pop(path)
                self._readers[path] = reader
                self.hits += 1
                return reader, True
            reader = VideoReader(path, num_threads=self.num_threads)
            self._readers[path] = reader
            self.misses += 1
            while len(self._readers) > self.capacity:
                _, evicted = self._readers.popitem(last=False)
                del evicted
            return reader, False

    def path_lock(self, path: str) -> threading.RLock:
        with self._lock:
            return self._path_locks.setdefault(path, threading.RLock())

    def metadata(self, path: str) -> tuple[float, int, int, int]:
        with self.path_lock(path):
            reader, _ = self.reader(path)
            source_fps = float(reader.get_avg_fps())
            source_total = len(reader)
            height, width, _ = reader[0].shape
            return source_fps, source_total, int(height), int(width)

    def __call__(self, ele: dict[str, Any]) -> tuple[Any, dict[str, Any], float]:
        import torch
        from qwen_vl_utils import vision_process

        path = str(ele["video"])
        with self.path_lock(path):
            reader, _ = self.reader(path)
            total_frames = len(reader)
            video_fps = float(reader.get_avg_fps())
            start_frame, end_frame, selected_total = (
                vision_process.calculate_video_frame_range(
                    ele, total_frames, video_fps
                )
            )
            nframes = vision_process.smart_nframes(
                ele, total_frames=selected_total, video_fps=video_fps
            )
            indices = (
                torch.linspace(start_frame, end_frame, nframes)
                .round()
                .long()
                .tolist()
            )
            video = torch.tensor(reader.get_batch(indices).asnumpy()).permute(
                0, 3, 1, 2
            )
        metadata = {
            "fps": video_fps,
            "frames_indices": indices,
            "total_num_frames": selected_total,
            "video_backend": "decord",
        }
        sample_fps = nframes / max(selected_total, 1e-6) * video_fps
        return video, metadata, sample_fps


_DECORD_READER_POOL: DecordReaderPool | None = None


def install_decord_reader_pool(capacity: int, num_threads: int = 2) -> bool:
    """Install the exact reader-reuse backend when decord is selected."""
    global _DECORD_READER_POOL

    if capacity < 0:
        raise ValueError("video reader cache capacity must be non-negative")
    if capacity == 0:
        _DECORD_READER_POOL = None
        return False
    from qwen_vl_utils import vision_process

    if vision_process.get_video_reader_backend() != "decord":
        _DECORD_READER_POOL = None
        return False
    pool = DecordReaderPool(capacity, num_threads=num_threads)
    vision_process.VIDEO_READER_BACKENDS["decord"] = pool
    _DECORD_READER_POOL = pool
    return True


@lru_cache(maxsize=8)
def probe_source_video(path: str) -> tuple[float, int, int, int]:
    """Return immutable source metadata without reopening grouped videos."""
    try:
        if _DECORD_READER_POOL is not None:
            return _DECORD_READER_POOL.metadata(path)
        from decord import VideoReader

        reader = VideoReader(path)
        try:
            source_fps = float(reader.get_avg_fps())
            source_total = len(reader)
            height, width, _ = reader[0].shape
            return source_fps, source_total, int(height), int(width)
        finally:
            del reader
    except Exception:
        # Some public OVO MP4s have a malformed tail that makes Decord exhaust
        # its EOF retry budget even though FFmpeg can decode all model-visible
        # frames. Metadata probing must therefore fail over before the exact
        # FFmpeg pixel path is attempted.
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,nb_frames,nb_read_frames,width,height,duration",
                "-of",
                "json",
                path,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(f"FFprobe source metadata failed: {detail}")
        streams = json.loads(completed.stdout.decode("utf-8")).get("streams") or []
        if len(streams) != 1:
            raise RuntimeError("FFprobe did not return one video stream")
        stream = streams[0]
        numerator, denominator = str(stream["avg_frame_rate"]).split("/", 1)
        source_fps = float(numerator) / float(denominator)
        count = stream.get("nb_frames") or stream.get("nb_read_frames")
        if count in (None, "N/A"):
            count = round(float(stream["duration"]) * source_fps)
        source_total = int(count)
        if source_fps <= 0 or source_total <= 0:
            raise RuntimeError("FFprobe returned invalid source timing")
        return source_fps, source_total, int(stream["height"]), int(stream["width"])


def decode_recent_window(
    path: Path,
    *,
    visible_until: float,
    recent_frames: int,
    sampling_fps: float,
) -> tuple[list[Any], dict[str, Any]]:
    import torch
    import torch.nn.functional as functional
    from decord import VideoReader
    from PIL import Image
    from qwen_vl_utils.vision_process import fetch_video

    source_fps, source_total, height, width = probe_source_video(str(path))
    reader = None
    requested_visible_until = max(0.0, float(visible_until))
    effective_visible_until, visible_until_clamped = clamp_visible_until_to_source(
        requested_visible_until, source_fps, source_total
    )
    window_start, window_end = recent_window_bounds(
        effective_visible_until, recent_frames, sampling_fps
    )
    decode_backend = "qwen_vl_utils.fetch_video"
    early_prefix_fallback = False
    try:
        frames, metadata = fetch_video(
            {
                "video": str(path),
                "fps": float(sampling_fps),
                "video_start": window_start,
                "video_end": window_end + 1e-4,
            },
            # Preserve the public SimpleStream release default even though the
            # Qwen3-VL processor itself uses 16-pixel patches.
            image_patch_size=14,
            return_video_metadata=True,
        )
        indices = [int(value) for value in metadata.get("frames_indices") or []]
        if int(frames.shape[0]) != len(indices):
            raise ValueError("qwen-vl-utils frame-index metadata mismatch")
    except ValueError as error:
        # qwen-vl-utils requires at least two raw frames. A few OVO queries
        # occur within the first raw-frame interval. Repeat frame zero so the
        # query remains scoreable without introducing any future pixel.
        if window_end * source_fps >= 1.0:
            raise
        early_prefix_fallback = True
        decode_backend = f"decord_frame0_repeat_after:{type(error).__name__}"
        reader = VideoReader(str(path))
        first = torch.from_numpy(reader[0].asnumpy()).permute(2, 0, 1)
        resized_height, resized_width = rounded_size(
            height,
            width,
            factor=28,
            max_pixels=560 * 1008,
            max_side=1008,
        )
        first = (
            functional.interpolate(
                first.unsqueeze(0).float(),
                size=(resized_height, resized_width),
                mode="bicubic",
                align_corners=False,
            )
            .clamp_(0, 255)
            .to(torch.uint8)[0]
        )
        frames = first.unsqueeze(0).repeat(recent_frames, 1, 1, 1)
        indices = [0] * recent_frames
    finally:
        if reader is not None:
            del reader

    if int(frames.shape[0]) > recent_frames:
        frames = frames[-recent_frames:]
        indices = indices[-recent_frames:]
    while int(frames.shape[0]) < recent_frames:
        frames = torch.cat((frames[:1], frames), dim=0)
        indices.insert(0, indices[0])
    timestamps = [index / max(source_fps, 1e-6) for index in indices]
    # Average FPS values can place the selected boundary frame a few
    # microseconds past the requested timestamp even though the decoder chose
    # the same integer source frame at the causal boundary. One millisecond is
    # still far below a source-frame interval for these benchmarks.
    if timestamps and max(timestamps) > window_end + 1e-3:
        raise RuntimeError(
            f"future frame sampled: max={max(timestamps)} query={window_end}"
        )
    pil_frames = [
        Image.fromarray(
            frame.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        )
        for frame in frames
    ]
    return pil_frames, {
        "source_fps": source_fps,
        "source_total_frames": source_total,
        "source_duration": source_total / max(source_fps, 1e-6),
        "source_last_frame_timestamp": (
            (source_total - 1) / max(source_fps, 1e-6)
        ),
        "requested_visible_until": requested_visible_until,
        "effective_visible_until": effective_visible_until,
        "visible_until_clamped_to_source": visible_until_clamped,
        "window_start": window_start,
        "window_end_exclusive": window_end,
        "frame_indices": indices,
        "frame_timestamps": timestamps,
        "num_sampled_frames": len(indices),
        "source_height": height,
        "source_width": width,
        "decoded_height": int(frames.shape[-2]),
        "decoded_width": int(frames.shape[-1]),
        "decode_backend": decode_backend,
        "early_prefix_fallback": early_prefix_fallback,
        "frame_policy": "recent4",
        "extra_frames_inspected": 0,
    }


def _qwen_video_resize_limits(
    vision_process: Any, nframes: int
) -> tuple[int, int, float]:
    """Return resize limits across released qwen-vl-utils APIs.

    Older releases expose pixel limits directly while newer releases expose
    token counts plus the spatial merge size.  Both branches reproduce the
    corresponding ``fetch_video`` calculation instead of pinning the evaluator
    to one package revision.
    """

    if nframes <= 0:
        raise ValueError("video resize requires at least one frame")
    if hasattr(vision_process, "IMAGE_FACTOR"):
        image_factor = int(vision_process.IMAGE_FACTOR)
        min_pixels = int(vision_process.VIDEO_MIN_PIXELS)
        total_pixels = float(vision_process.VIDEO_TOTAL_PIXELS)
        max_pixels_cap = float(vision_process.VIDEO_MAX_PIXELS)
    else:
        image_factor = int(14 * vision_process.SPATIAL_MERGE_SIZE)
        min_pixels = int(
            vision_process.VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
        )
        total_pixels = float(
            vision_process.MODEL_SEQ_LEN * image_factor * image_factor * 0.9
        )
        max_pixels_cap = float(
            vision_process.VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
        )
    max_pixels = max(
        min(
            max_pixels_cap,
            total_pixels / nframes * vision_process.FRAME_FACTOR,
        ),
        int(min_pixels * 1.05),
    )
    return image_factor, min_pixels, max_pixels


def decode_prefix_window_ffmpeg(
    path: Path,
    *,
    visible_until: float,
    recent_frames: int,
    sampling_fps: float,
    num_threads: int = 8,
    apply_qwen_video_resize: bool = True,
) -> tuple[list[Any], dict[str, Any]]:
    """Decode exact prefix indices in one sequential FFmpeg pass.

    Decord's ``get_batch`` performs a separate random-access decode for many
    H.264 indices. A 32-frame causal prefix can consequently take minutes even
    when all indices occur in the first few seconds. FFmpeg uses the same
    libav decoder and RGB24 conversion, but visits the source once and emits
    only the exact qwen-vl-utils indices. The resize path below is copied from
    qwen-vl-utils so pixels and visual-token geometry remain unchanged.
    """
    if num_threads <= 0:
        raise ValueError("FFmpeg decoder thread count must be positive")

    import numpy as np
    import torch
    from PIL import Image
    from qwen_vl_utils import vision_process
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transforms_functional

    source_fps, source_total, height, width = probe_source_video(str(path))
    requested_visible_until = max(0.0, float(visible_until))
    effective_visible_until, visible_until_clamped = clamp_visible_until_to_source(
        requested_visible_until, source_fps, source_total
    )
    window_start, window_end = recent_window_bounds(
        effective_visible_until, recent_frames, sampling_fps
    )
    # Preserve the locked qwen-vl-utils/decord frame-zero fallback exactly.
    # This path does not perform the expensive multi-index get_batch call.
    early_prefix_fallback = window_end * source_fps < 1.0
    if early_prefix_fallback and apply_qwen_video_resize:
        return decode_recent_window(
            path,
            visible_until=visible_until,
            recent_frames=recent_frames,
            sampling_fps=sampling_fps,
        )
    element = {
        "video": str(path),
        "fps": float(sampling_fps),
        "video_start": window_start,
        "video_end": window_end + 1e-4,
    }
    if early_prefix_fallback:
        indices = [0] * recent_frames
    else:
        start_frame, end_frame, selected_total = (
            vision_process.calculate_video_frame_range(
                element, source_total, source_fps
            )
        )
        nframes = vision_process.smart_nframes(
            element, total_frames=selected_total, video_fps=source_fps
        )
        indices = (
            torch.linspace(start_frame, end_frame, nframes)
            .round()
            .long()
            .tolist()
        )

    ordered_unique = list(dict.fromkeys(int(value) for value in indices))
    select = "+".join(f"eq(n\\,{value})" for value in ordered_unique)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        str(num_threads),
        "-i",
        str(path),
        "-vf",
        f"select={select}",
        "-vsync",
        "0",
        "-frames:v",
        str(len(ordered_unique)),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"FFmpeg prefix decode failed: {detail}")
    frame_bytes = int(height * width * 3)
    expected_bytes = len(ordered_unique) * frame_bytes
    if len(completed.stdout) != expected_bytes:
        raise RuntimeError(
            "FFmpeg prefix frame count mismatch: "
            f"expected_bytes={expected_bytes} actual_bytes={len(completed.stdout)}"
        )
    unique_array = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
        len(ordered_unique), height, width, 3
    )
    by_index = {
        value: unique_array[position]
        for position, value in enumerate(ordered_unique)
    }
    resized_height, resized_width = height, width
    if apply_qwen_video_resize:
        image_factor, min_pixels, max_pixels = _qwen_video_resize_limits(
            vision_process, len(indices)
        )
        resized_height, resized_width = vision_process.smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        # torchvision's batched uint8 bicubic path becomes pathologically slow
        # for 32 full-HD frames. Applying the identical operation frame by
        # frame is pixel-exact, avoids the multi-gigabyte intermediate, and is
        # over an order of magnitude faster on the benchmark hosts.
        pil_frames = []
        for value in indices:
            frame = torch.from_numpy(by_index[int(value)].copy()).permute(2, 0, 1)
            frame = transforms_functional.resize(
                frame,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            pil_frames.append(
                Image.fromarray(
                    frame.clamp(0, 255)
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
            )
        if len(pil_frames) > recent_frames:
            pil_frames = pil_frames[-recent_frames:]
            indices = indices[-recent_frames:]
        while len(pil_frames) < recent_frames:
            pil_frames.insert(0, pil_frames[0].copy())
            indices.insert(0, indices[0])
    else:
        if len(indices) > recent_frames:
            indices = indices[-recent_frames:]
        while len(indices) < recent_frames:
            indices.insert(0, indices[0])
        pil_frames = [
            Image.fromarray(by_index[int(value)].copy()) for value in indices
        ]
    timestamps = [index / max(source_fps, 1e-6) for index in indices]
    if timestamps and max(timestamps) > window_end + 1e-3:
        raise RuntimeError(
            f"future frame sampled: max={max(timestamps)} query={window_end}"
        )
    return pil_frames, {
        "source_fps": source_fps,
        "source_total_frames": source_total,
        "source_duration": source_total / max(source_fps, 1e-6),
        "source_last_frame_timestamp": (
            (source_total - 1) / max(source_fps, 1e-6)
        ),
        "requested_visible_until": requested_visible_until,
        "effective_visible_until": effective_visible_until,
        "visible_until_clamped_to_source": visible_until_clamped,
        "window_start": window_start,
        "window_end_exclusive": window_end,
        "frame_indices": indices,
        "frame_timestamps": timestamps,
        "num_sampled_frames": len(indices),
        "source_height": height,
        "source_width": width,
        "decoded_height": resized_height,
        "decoded_width": resized_width,
        "decode_backend": (
            "ffmpeg_sequential_qwen_video_resize"
            if apply_qwen_video_resize
            else "ffmpeg_sequential_source_rgb"
        ),
        "early_prefix_fallback": early_prefix_fallback,
        "frame_policy": "recent4",
        "extra_frames_inspected": 0,
    }


def decode_prefix_windows_ffmpeg_batched(
    path: Path,
    *,
    visible_untils: list[float],
    prefix_frames: int = 32,
    num_threads: int = 8,
    apply_qwen_video_resize: bool = True,
) -> list[tuple[list[Any], dict[str, Any]]]:
    """Decode several exact causal-prefix requests in one FFmpeg pass.

    Benchmark exports often place many query times from the same long video in
    one inference batch.  Calling :func:`decode_prefix_window_ffmpeg` for each
    query replays the same compressed stream from frame zero many times.  This
    function computes the exact upstream indices for every query, decodes their
    sorted union once, and reconstructs each request independently.  No query
    receives a frame outside its own exact index set.
    """

    if not visible_untils:
        return []
    if prefix_frames <= 0 or num_threads <= 0:
        raise ValueError("batched prefix frame and thread counts must be positive")

    import numpy as np
    import torch
    from PIL import Image
    from qwen_vl_utils import vision_process
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transforms_functional

    source_fps, source_total, height, width = probe_source_video(str(path))
    specs: list[dict[str, Any]] = []
    early_results: dict[int, tuple[list[Any], dict[str, Any]]] = {}
    for position, raw_visible_until in enumerate(visible_untils):
        requested_visible_until = max(0.0, float(raw_visible_until))
        effective_visible_until, visible_until_clamped = (
            clamp_visible_until_to_source(
                requested_visible_until, source_fps, source_total
            )
        )
        policy_fps = prefix_frames / max(requested_visible_until, 1e-6)
        window_start, window_end = recent_window_bounds(
            effective_visible_until, prefix_frames, policy_fps
        )
        early_prefix_fallback = window_end * source_fps < 1.0
        if early_prefix_fallback and apply_qwen_video_resize:
            frames, sampling = decode_recent_window(
                path,
                visible_until=raw_visible_until,
                recent_frames=prefix_frames,
                sampling_fps=policy_fps,
            )
            sampling = dict(sampling)
            sampling.update(
                {
                    "frame_policy": f"prefix{prefix_frames}",
                    "policy_sampling_fps": policy_fps,
                    "policy_frame_budget": prefix_frames,
                }
            )
            early_results[position] = (frames, sampling)
            specs.append({"early": True})
            continue
        element = {
            "video": str(path),
            "fps": float(policy_fps),
            "video_start": window_start,
            "video_end": window_end + 1e-4,
        }
        if early_prefix_fallback:
            indices = [0] * prefix_frames
        else:
            start_frame, end_frame, selected_total = (
                vision_process.calculate_video_frame_range(
                    element, source_total, source_fps
                )
            )
            nframes = vision_process.smart_nframes(
                element, total_frames=selected_total, video_fps=source_fps
            )
            indices = (
                torch.linspace(start_frame, end_frame, nframes)
                .round()
                .long()
                .tolist()
            )
        specs.append(
            {
                "early": False,
                "requested_visible_until": requested_visible_until,
                "effective_visible_until": effective_visible_until,
                "visible_until_clamped": visible_until_clamped,
                "window_start": window_start,
                "window_end": window_end,
                "early_prefix_fallback": early_prefix_fallback,
                "policy_fps": policy_fps,
                "indices": [int(value) for value in indices],
            }
        )

    ordered_unique = sorted(
        {
            int(value)
            for spec in specs
            if not spec["early"]
            for value in spec["indices"]
        }
    )
    by_index: dict[int, Any] = {}
    decode_backend = (
        "ffmpeg_batched_sequential_qwen_video_resize"
        if apply_qwen_video_resize
        else "ffmpeg_batched_sequential_source_rgb"
    )
    sparse_random_access_fallback = False
    if ordered_unique:
        # Sequential FFmpeg is fastest for dense, short prefixes.  For long
        # sparse prefixes it needlessly decodes thousands of intervening
        # frames, while Decord can seek to the exact same libav frame indices.
        # The ratio is public media geometry only and cannot route by task or
        # question.  A pooled reader is accessed by at most one per-video
        # future, so its native decoder is never used concurrently.
        sparsity_ratio = (ordered_unique[-1] + 1) / len(ordered_unique)
        use_sparse_random_access = (
            sparsity_ratio >= 64.0 and _DECORD_READER_POOL is not None
        )
        if use_sparse_random_access:
            try:
                reader, _ = _DECORD_READER_POOL.reader(str(path))
                unique_array = reader.get_batch(ordered_unique).asnumpy()
                decode_backend = (
                    "decord_batched_sparse_qwen_video_resize"
                    if apply_qwen_video_resize
                    else "decord_batched_sparse_source_rgb"
                )
            except Exception:
                # Some public H.264 files have a damaged final GOP.  Decord's
                # bounded EOF recovery can reject them even though FFmpeg can
                # emit every requested exact index.  Fall back per video; the
                # requested index set, resize path, and causal frontier remain
                # unchanged.
                use_sparse_random_access = False
                sparse_random_access_fallback = True
                decode_backend = (
                    "ffmpeg_batched_sequential_after_decord_error_qwen_video_resize"
                    if apply_qwen_video_resize
                    else "ffmpeg_batched_sequential_after_decord_error_source_rgb"
                )
        if not use_sparse_random_access:
            select = "+".join(f"eq(n\\,{value})" for value in ordered_unique)
            command = [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                str(num_threads),
                "-i",
                str(path),
                "-vf",
                f"select={select}",
                "-vsync",
                "0",
                "-frames:v",
                str(len(ordered_unique)),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
                raise RuntimeError(f"FFmpeg batched prefix decode failed: {detail}")
            frame_bytes = int(height * width * 3)
            expected_bytes = len(ordered_unique) * frame_bytes
            if len(completed.stdout) != expected_bytes:
                raise RuntimeError(
                    "FFmpeg batched prefix frame count mismatch: "
                    f"expected_bytes={expected_bytes} "
                    f"actual_bytes={len(completed.stdout)}"
                )
            unique_array = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
                len(ordered_unique), height, width, 3
            )
        by_index = {
            value: unique_array[position]
            for position, value in enumerate(ordered_unique)
        }

    resized_height, resized_width = height, width
    if apply_qwen_video_resize:
        image_factor, min_pixels, max_pixels = _qwen_video_resize_limits(
            vision_process, prefix_frames
        )
        resized_height, resized_width = vision_process.smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    results: list[tuple[list[Any], dict[str, Any]]] = []
    for position, spec in enumerate(specs):
        if spec["early"]:
            results.append(early_results[position])
            continue
        indices = list(spec["indices"])
        if apply_qwen_video_resize:
            pil_frames = []
            for value in indices:
                frame = torch.from_numpy(by_index[int(value)].copy()).permute(2, 0, 1)
                frame = transforms_functional.resize(
                    frame,
                    [resized_height, resized_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
                pil_frames.append(
                    Image.fromarray(
                        frame.clamp(0, 255)
                        .to(torch.uint8)
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                )
        else:
            pil_frames = [
                Image.fromarray(by_index[int(value)].copy()) for value in indices
            ]
        if len(pil_frames) > prefix_frames:
            pil_frames = pil_frames[-prefix_frames:]
            indices = indices[-prefix_frames:]
        while len(pil_frames) < prefix_frames:
            pil_frames.insert(0, pil_frames[0].copy())
            indices.insert(0, indices[0])
        timestamps = [index / max(source_fps, 1e-6) for index in indices]
        if timestamps and max(timestamps) > spec["window_end"] + 1e-3:
            raise RuntimeError(
                "future frame sampled by batched prefix decoder: "
                f"max={max(timestamps)} query={spec['window_end']}"
            )
        results.append(
            (
                pil_frames,
                {
                    "source_fps": source_fps,
                    "source_total_frames": source_total,
                    "source_duration": source_total / max(source_fps, 1e-6),
                    "source_last_frame_timestamp": (
                        (source_total - 1) / max(source_fps, 1e-6)
                    ),
                    "requested_visible_until": spec["requested_visible_until"],
                    "effective_visible_until": spec["effective_visible_until"],
                    "visible_until_clamped_to_source": spec[
                        "visible_until_clamped"
                    ],
                    "window_start": spec["window_start"],
                    "window_end_exclusive": spec["window_end"],
                    "frame_indices": indices,
                    "frame_timestamps": timestamps,
                    "num_sampled_frames": len(indices),
                    "source_height": height,
                    "source_width": width,
                    "decoded_height": resized_height,
                    "decoded_width": resized_width,
                    "decode_backend": decode_backend,
                    "decode_fallback_from_sparse_random_access": (
                        sparse_random_access_fallback
                    ),
                    "early_prefix_fallback": spec["early_prefix_fallback"],
                    "frame_policy": f"prefix{prefix_frames}",
                    "policy_sampling_fps": spec["policy_fps"],
                    "policy_frame_budget": prefix_frames,
                    "extra_frames_inspected": 0,
                },
            )
        )
    return results


def decode_source_window_decord(
    path: Path,
    *,
    visible_until: float,
    recent_frames: int,
    sampling_fps: float,
    num_threads: int = 8,
    apply_qwen_video_resize: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    """Return exact qwen sampling indices as unresized source RGB frames."""
    if num_threads <= 0:
        raise ValueError("decord decoder thread count must be positive")

    import torch
    from decord import VideoReader
    from PIL import Image
    from qwen_vl_utils import vision_process
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transforms_functional

    source_fps, source_total, height, width = probe_source_video(str(path))
    requested_visible_until = max(0.0, float(visible_until))
    effective_visible_until, visible_until_clamped = clamp_visible_until_to_source(
        requested_visible_until, source_fps, source_total
    )
    window_start, window_end = recent_window_bounds(
        effective_visible_until, recent_frames, sampling_fps
    )
    early_prefix_fallback = window_end * source_fps < 1.0
    if early_prefix_fallback:
        indices = [0] * recent_frames
    else:
        element = {
            "video": str(path),
            "fps": float(sampling_fps),
            "video_start": window_start,
            "video_end": window_end + 1e-4,
        }
        start_frame, end_frame, selected_total = (
            vision_process.calculate_video_frame_range(
                element, source_total, source_fps
            )
        )
        nframes = vision_process.smart_nframes(
            element, total_frames=selected_total, video_fps=source_fps
        )
        indices = (
            torch.linspace(start_frame, end_frame, nframes)
            .round()
            .long()
            .tolist()
        )
    if len(indices) > recent_frames:
        indices = indices[-recent_frames:]
    while len(indices) < recent_frames:
        indices.insert(0, indices[0])

    if _DECORD_READER_POOL is not None:
        reader, _ = _DECORD_READER_POOL.reader(str(path))
        owns_reader = False
    else:
        reader = VideoReader(str(path), num_threads=num_threads)
        owns_reader = True
    try:
        ordered_unique = list(dict.fromkeys(int(value) for value in indices))
        try:
            unique_array = reader.get_batch(ordered_unique).asnumpy()
            by_index = {
                value: unique_array[position]
                for position, value in enumerate(ordered_unique)
            }
            decode_backend = "decord_source_rgb_exact_indices"
        except Exception as error:
            # A few public H.264 files have a damaged final GOP that makes
            # Decord exhaust its EOF retry budget. FFmpeg still decodes the
            # exact requested source indices, so fail over per video without
            # changing the causal frontier or frame-selection policy.
            import numpy as np

            fallback_frames = decode_exact_indices_ffmpeg(path, ordered_unique)
            by_index = {
                value: np.asarray(fallback_frames[position])
                for position, value in enumerate(ordered_unique)
            }
            decode_backend = f"ffmpeg_exact_indices_after_{type(error).__name__}"
    finally:
        if owns_reader:
            del reader
    resized_height, resized_width = height, width
    if apply_qwen_video_resize:
        image_factor, min_pixels, max_pixels = _qwen_video_resize_limits(
            vision_process, len(indices)
        )
        resized_height, resized_width = vision_process.smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        pil_frames = []
        for value in indices:
            frame = torch.from_numpy(by_index[int(value)].copy()).permute(2, 0, 1)
            frame = transforms_functional.resize(
                frame,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            pil_frames.append(
                Image.fromarray(
                    frame.clamp(0, 255)
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
            )
    else:
        pil_frames = [
            Image.fromarray(by_index[int(value)].copy()) for value in indices
        ]
    timestamps = [index / max(source_fps, 1e-6) for index in indices]
    if timestamps and max(timestamps) > window_end + 1e-3:
        raise RuntimeError(
            f"future frame sampled: max={max(timestamps)} query={window_end}"
        )
    return pil_frames, {
        "source_fps": source_fps,
        "source_total_frames": source_total,
        "source_duration": source_total / max(source_fps, 1e-6),
        "source_last_frame_timestamp": (
            (source_total - 1) / max(source_fps, 1e-6)
        ),
        "requested_visible_until": requested_visible_until,
        "effective_visible_until": effective_visible_until,
        "visible_until_clamped_to_source": visible_until_clamped,
        "window_start": window_start,
        "window_end_exclusive": window_end,
        "frame_indices": indices,
        "frame_timestamps": timestamps,
        "num_sampled_frames": len(indices),
        "source_height": height,
        "source_width": width,
        "decoded_height": resized_height,
        "decoded_width": resized_width,
        "decode_backend": (
            "decord_exact_indices_qwen_video_resize"
            if apply_qwen_video_resize
            and decode_backend == "decord_source_rgb_exact_indices"
            else (
                f"{decode_backend}_qwen_video_resize"
                if apply_qwen_video_resize
                else decode_backend
            )
        ),
        "early_prefix_fallback": early_prefix_fallback,
        "frame_policy": "recent4",
        "extra_frames_inspected": 0,
    }


def exact_causal_window_indices(
    path: Path,
    *,
    visible_until: float,
    requested_frames: int,
    sampling_fps: float,
) -> tuple[list[int], dict[str, Any]]:
    """Resolve exact upstream Qwen frame indices without decoding their pixels."""

    if requested_frames <= 0 or sampling_fps <= 0:
        raise ValueError("frame count and sampling FPS must be positive")
    import torch
    from qwen_vl_utils import vision_process

    source_fps, source_total, height, width = probe_source_video(str(path))
    requested_visible_until = max(0.0, float(visible_until))
    effective_visible_until, visible_until_clamped = clamp_visible_until_to_source(
        requested_visible_until, source_fps, source_total
    )
    window_start, window_end = recent_window_bounds(
        effective_visible_until, requested_frames, sampling_fps
    )
    if window_end * source_fps < 1.0:
        indices = [0]
    else:
        element = {
            "video": str(path),
            "fps": float(sampling_fps),
            "video_start": window_start,
            "video_end": window_end + 1e-4,
        }
        start_frame, end_frame, selected_total = (
            vision_process.calculate_video_frame_range(
                element, source_total, source_fps
            )
        )
        nframes = vision_process.smart_nframes(
            element, total_frames=selected_total, video_fps=source_fps
        )
        indices = (
            torch.linspace(start_frame, end_frame, nframes)
            .round()
            .long()
            .tolist()
        )
    if len(indices) > requested_frames:
        indices = indices[-requested_frames:]
    indices = list(dict.fromkeys(int(value) for value in indices))
    timestamps = [index / source_fps for index in indices]
    if timestamps and max(timestamps) > window_end + 1e-3:
        raise RuntimeError("exact window index crosses causal frontier")
    return indices, {
        "source_fps": source_fps,
        "source_total_frames": source_total,
        "source_duration": source_total / source_fps,
        "source_last_frame_timestamp": (source_total - 1) / source_fps,
        "requested_visible_until": requested_visible_until,
        "effective_visible_until": effective_visible_until,
        "visible_until_clamped_to_source": visible_until_clamped,
        "window_start": window_start,
        "window_end_exclusive": window_end,
        "source_height": height,
        "source_width": width,
    }


def decode_exact_indices_ffmpeg(
    path: Path, indices: list[int], *, probe_side: int | None = None
) -> list[Any]:
    """Decode explicit sorted source indices once, optionally as tiny probes."""

    if not indices or indices != sorted(set(indices)):
        raise ValueError("exact FFmpeg indices must be non-empty, sorted, and unique")
    if probe_side is not None and probe_side <= 0:
        raise ValueError("probe side must be positive")
    import numpy as np
    from PIL import Image

    _, _, source_height, source_width = probe_source_video(str(path))
    selected = "+".join(f"eq(n\\,{index})" for index in indices)
    filter_graph = f"select={selected}"
    height, width = source_height, source_width
    if probe_side is not None:
        filter_graph += f",scale={probe_side}:{probe_side}"
        height = width = int(probe_side)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "8",
        "-i",
        str(path),
        "-vf",
        filter_graph,
        "-vsync",
        "0",
        "-frames:v",
        str(len(indices)),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    completed = subprocess.run(
        command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"FFmpeg exact-index decode failed: {detail}")
    frame_bytes = int(height * width * 3)
    if len(completed.stdout) != len(indices) * frame_bytes:
        raise RuntimeError("FFmpeg exact-index frame count mismatch")
    array = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
        len(indices), height, width, 3
    )
    if probe_side is not None:
        return [
            (frame.astype("float32").reshape(-1) / 255.0).tolist()
            for frame in array
        ]
    return [Image.fromarray(frame.copy()) for frame in array]


def decode_causal_temporal_budget(
    path: Path,
    *,
    visible_until: float,
    candidate_frames: int = 64,
    model_frames: int = 20,
    recent_frames: int = 4,
    probe_side: int = 16,
) -> tuple[list[Any], dict[str, Any]]:
    """Build a router-free recent-plus-curvature causal view."""

    if candidate_frames < model_frames or model_frames < recent_frames:
        raise ValueError("invalid candidate/model/recent temporal budgets")
    effective = max(float(visible_until), 1e-6)
    prefix_indices, source = exact_causal_window_indices(
        path,
        visible_until=visible_until,
        requested_frames=candidate_frames,
        sampling_fps=candidate_frames / effective,
    )
    recent_indices, _ = exact_causal_window_indices(
        path,
        visible_until=visible_until,
        requested_frames=recent_frames,
        sampling_fps=1.0,
    )
    recent_indices = recent_indices[-recent_frames:]
    earliest_recent = min(recent_indices)
    historical = [index for index in prefix_indices if index < earliest_recent]
    candidate_indices = sorted(set([*historical, *recent_indices]))
    features = decode_exact_indices_ffmpeg(
        path, candidate_indices, probe_side=probe_side
    )
    timestamps = [index / source["source_fps"] for index in candidate_indices]
    signals = build_frame_signals(
        frame_indices=candidate_indices,
        timestamps=timestamps,
        features=features,
    )
    controller = select_temporal_budget(
        signals,
        causal_frontier=source["effective_visible_until"],
        config=TemporalBudgetConfig(
            max_model_frames=model_frames,
            recent_frames=min(recent_frames, len(recent_indices)),
            # The confirmatory v0 controller has no query/frame embedding
            # backend. Keep this weight explicitly zero rather than letting
            # tied all-zero relevance ranks add a misleading constant term.
            query_weight=0.0,
        ),
    )
    selected_indices = [int(value) for value in controller["selected_frame_indices"]]
    selected_frames = decode_exact_indices_ffmpeg(path, selected_indices)
    selected_timestamps = [index / source["source_fps"] for index in selected_indices]
    if max(selected_timestamps) > source["effective_visible_until"] + 1e-3:
        raise RuntimeError("temporal budget selected a future frame")
    return selected_frames, {
        **source,
        "frame_indices": selected_indices,
        "frame_timestamps": selected_timestamps,
        "num_sampled_frames": len(selected_indices),
        "decoded_height": source["source_height"],
        "decoded_width": source["source_width"],
        "decode_backend": "ffmpeg_lowres_probe_then_exact_source_indices",
        "early_prefix_fallback": len(candidate_indices) < recent_frames,
        "frame_policy": (
            "causalbudget20"
            if model_frames == 20 and recent_frames == 4
            else f"causalbudget{model_frames}r{recent_frames}"
        ),
        "policy_sampling_fps": None,
        "policy_frame_budget": model_frames,
        "probe_frame_budget": candidate_frames,
        "probe_resolution": [probe_side, probe_side],
        "candidate_frame_indices": candidate_indices,
        "candidate_frame_timestamps": timestamps,
        "recent_frame_indices": [
            int(row["frame_index"])
            for row in controller["selected"]
            if row["allocation"] == "recent_high_fidelity"
        ],
        "extra_frames_inspected": controller["extra_frames_inspected"],
        "controller": controller,
    }


def uniform_prefix_indices(last_index_inclusive: int, budget: int) -> list[int]:
    """Select a deterministic uniform set from ``[0, last_index_inclusive]``.

    The helper never pads or repeats source indices.  Early prefixes therefore
    expose fewer than the maximum budget rather than pretending duplicated
    pixels are additional observations.
    """

    if budget <= 0:
        raise ValueError("uniform prefix budget must be positive")
    if last_index_inclusive < 0:
        return []
    count = min(int(budget), int(last_index_inclusive) + 1)
    if count == 1:
        return [int(last_index_inclusive)]
    values = [
        round(position * last_index_inclusive / (count - 1))
        for position in range(count)
    ]
    indices = list(dict.fromkeys(int(value) for value in values))
    if indices != sorted(indices) or len(indices) != count:
        raise RuntimeError("uniform prefix index construction is not unique")
    return indices


def decode_causal_hybrid_prefix(
    path: Path,
    *,
    visible_until: float,
    model_frames: int = 32,
    recent_frames: int = 4,
) -> tuple[list[Any], dict[str, Any]]:
    """Decode uniform causal history plus an exact high-fidelity recent window.

    History is later sent through Qwen3-VL's timestamp-aware video path while
    the exact recent window remains a list of independently resized images.
    This isolates the central research hypothesis: long-range context should
    augment, not replace or downsample, current visual grounding.
    """

    if model_frames <= recent_frames or recent_frames <= 0:
        raise ValueError("hybrid history/recent budgets are invalid")
    recent_indices, source = exact_causal_window_indices(
        path,
        visible_until=visible_until,
        requested_frames=recent_frames,
        sampling_fps=1.0,
    )
    if not recent_indices:
        raise RuntimeError("hybrid prefix has no causal recent frame")
    history_budget = model_frames - recent_frames
    history_indices = uniform_prefix_indices(
        min(recent_indices) - 1, history_budget
    )
    selected_indices = sorted(set([*history_indices, *recent_indices]))
    if len(selected_indices) > model_frames:
        raise RuntimeError("hybrid prefix exceeds its unique-frame budget")
    selected_frames = decode_exact_indices_ffmpeg(path, selected_indices)
    selected_timestamps = [
        index / source["source_fps"] for index in selected_indices
    ]
    if max(selected_timestamps) > source["effective_visible_until"] + 1e-3:
        raise RuntimeError("hybrid prefix selected a future frame")
    frame_policy = (
        "causalhybrid32"
        if model_frames == 32 and recent_frames == 4
        else f"causalhybrid{model_frames}r{recent_frames}"
    )
    return selected_frames, {
        **source,
        "window_start": 0.0,
        "frame_indices": selected_indices,
        "frame_timestamps": selected_timestamps,
        "num_sampled_frames": len(selected_indices),
        "decoded_height": source["source_height"],
        "decoded_width": source["source_width"],
        "decode_backend": (
            "ffmpeg_exact_uniform_history_plus_exact_recent"
            f"{recent_frames}"
        ),
        "early_prefix_fallback": len(selected_indices) < model_frames,
        "frame_policy": frame_policy,
        "policy_sampling_fps": None,
        "policy_frame_budget": model_frames,
        "history_frame_budget": history_budget,
        "history_frame_indices": history_indices,
        "recent_frame_indices": recent_indices,
        "extra_frames_inspected": 0,
        "controller": {
            "policy_id": "uniform_history_preserved_recent_v0",
            "benchmark_task_or_group_router": False,
            "question_keyword_router": False,
            "query_conditioned_frame_selection": False,
            "model_visible_frames": len(selected_indices),
            "maximum_model_visible_frames": model_frames,
            "history_frames": len(history_indices),
            "recent_frames": len(recent_indices),
            "future_frame_violations": 0,
        },
    }


class RecentWindowCache:
    """Small LRU for exact repeated decode requests.

    Benchmark exports commonly place several questions for the same video and
    query timestamp next to each other. Reopening and decoding that identical
    four-frame window for every question is pure I/O duplication. Cache keys
    include every sampling input, and defensive copies prevent processors from
    mutating cached image or metadata objects.
    """

    def __init__(
        self,
        capacity: int,
        decoder: Callable[..., tuple[list[Any], dict[str, Any]]] = decode_recent_window,
    ) -> None:
        if capacity < 0:
            raise ValueError("decode cache capacity must be non-negative")
        self.capacity = int(capacity)
        self.decoder = decoder
        self.hits = 0
        self.misses = 0
        self.singleflight_hits = 0
        self._lock = threading.RLock()
        self._entries: OrderedDict[
            tuple[str, str, int, str], tuple[tuple[Any, ...], dict[str, Any]]
        ] = OrderedDict()
        self._inflight: dict[
            tuple[str, str, int, str], concurrent.futures.Future[Any]
        ] = {}

    def get(
        self,
        path: Path,
        *,
        visible_until: float,
        recent_frames: int,
        sampling_fps: float,
        frame_policy: str = "recent4",
    ) -> tuple[list[Any], dict[str, Any], bool]:
        key = (
            str(path),
            float(visible_until).hex(),
            int(recent_frames),
            float(sampling_fps).hex() + ":" + frame_policy,
        )
        with self._lock:
            if self.capacity and key in self._entries:
                cached_frames, cached_sampling = self._entries.pop(key)
                self._entries[key] = (cached_frames, cached_sampling)
                self.hits += 1
                return (
                    [copy.copy(frame) for frame in cached_frames],
                    copy.deepcopy(cached_sampling),
                    True,
                )
            inflight = self._inflight.get(key)
            if inflight is None:
                inflight = concurrent.futures.Future()
                self._inflight[key] = inflight
                decode_owner = True
            else:
                decode_owner = False

        if not decode_owner:
            cached_frames, cached_sampling = inflight.result()
            with self._lock:
                self.hits += 1
                self.singleflight_hits += 1
            return (
                [copy.copy(frame) for frame in cached_frames],
                copy.deepcopy(cached_sampling),
                True,
            )

        policy_frames = FRAME_POLICY_NUM_FRAMES.get(frame_policy)
        if policy_frames is None:
            raise ValueError(f"unresolved frame policy: {frame_policy}")
        if frame_policy.startswith("prefix"):
            policy_fps = policy_frames / max(float(visible_until), 1e-6)
        else:
            policy_fps = (
                float(sampling_fps)
                if frame_policy == "recent4"
                else frame_policy_sampling_fps(frame_policy)
            )
        # Prefix policies request sparse samples across the full causal
        # history.  qwen-vl-utils' generic Decord path performs expensive
        # random seeks for every query, even though the exact upstream frame
        # indices are deterministic.  The dedicated FFmpeg decoder below
        # reproduces those indices and Qwen resize semantics while visiting
        # the compressed prefix once.  Recent/wide policies retain the public
        # decoder unchanged.
        decoder = (
            decode_prefix_window_ffmpeg
            if frame_policy.startswith("prefix")
            and self.decoder is decode_recent_window
            else self.decoder
        )
        try:
            frames, sampling = decoder(
                path,
                visible_until=visible_until,
                recent_frames=policy_frames,
                sampling_fps=policy_fps,
            )
        except BaseException as error:
            with self._lock:
                self._inflight.pop(key, None)
                if not inflight.done():
                    inflight.set_exception(error)
            raise
        sampling = dict(sampling)
        sampling["frame_policy"] = frame_policy
        sampling["policy_sampling_fps"] = policy_fps
        sampling["policy_frame_budget"] = policy_frames
        with self._lock:
            self.misses += 1
            canonical_frames = tuple(copy.copy(frame) for frame in frames)
            canonical_sampling = copy.deepcopy(sampling)
            if self.capacity:
                self._entries[key] = (canonical_frames, canonical_sampling)
                while len(self._entries) > self.capacity:
                    self._entries.popitem(last=False)
            self._inflight.pop(key, None)
            if not inflight.done():
                inflight.set_result((canonical_frames, canonical_sampling))
        return frames, sampling, False


class CausalTemporalBudgetCache:
    """LRU for the expensive, deterministic event-budget decoder.

    OVO commonly asks several questions at the same public video endpoint.
    The event controller is query-independent, so decoding and probing that
    exact prefix again cannot change its selected frames.  The cache key
    includes the resolved path and the exact floating-point endpoint; callers
    receive defensive frame and metadata copies.
    """

    def __init__(
        self, capacity: int, *, model_frames: int = 20, recent_frames: int = 4
    ) -> None:
        if capacity < 0:
            raise ValueError("temporal budget cache capacity must be non-negative")
        self.capacity = int(capacity)
        self.model_frames = int(model_frames)
        self.recent_frames = int(recent_frames)
        self.hits = 0
        self.misses = 0
        self._entries: OrderedDict[
            tuple[str, str], tuple[tuple[Any, ...], dict[str, Any]]
        ] = OrderedDict()

    def get(
        self, path: Path, *, visible_until: float
    ) -> tuple[list[Any], dict[str, Any], bool]:
        key = (str(path), float(visible_until).hex())
        if self.capacity and key in self._entries:
            cached_frames, cached_sampling = self._entries.pop(key)
            self._entries[key] = (cached_frames, cached_sampling)
            self.hits += 1
            return (
                [frame.copy() for frame in cached_frames],
                copy.deepcopy(cached_sampling),
                True,
            )
        frames, sampling = decode_causal_temporal_budget(
            path,
            visible_until=visible_until,
            model_frames=self.model_frames,
            recent_frames=self.recent_frames,
        )
        self.misses += 1
        if self.capacity:
            self._entries[key] = (
                tuple(frame.copy() for frame in frames),
                copy.deepcopy(sampling),
            )
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
        return frames, sampling, False


def render_question(
    benchmark: str, question: str, options: list[str]
) -> str:
    if not options:
        return question
    rendered = []
    for index, option in enumerate(options):
        label = chr(ord("A") + index)
        text = re.sub(
            r"^[\(\[]?[A-Da-d][\)\].:\s-]+", "", str(option)
        ).strip()
        rendered.append(f"{label}. {text}")
    options_text = "\n".join(rendered)
    if benchmark == "streamingbench":
        return STREAMING_PROMPT_TEMPLATE.format(
            question=question, options=options_text
        )
    return (
        question
        + "\nOptions: "
        + "; ".join(rendered)
        + ";\nOnly give the best option's letter directly."
    )


def normalize_answer(raw: str, question: str, options: list[str]) -> str:
    text = " ".join(str(raw).strip().split())
    if options:
        # Prefer an exact semantic option match before looking for a leading
        # option letter.  Without this ordering, a valid answer such as
        # "A map" is mistaken for option A because it starts with the letter
        # A; the displayed option may be B, C, or D.
        normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
        for index, option in enumerate(options):
            option_text = re.sub(
                r"^[\(\[]?[A-Da-d][\)\].:\s-]+", "", str(option)
            )
            option_normalized = re.sub(
                r"[^a-z0-9]+", " ", option_text.casefold()
            ).strip()
            if normalized == option_normalized:
                return chr(ord("A") + index)
        for pattern in (
            r"^[\(\[]?([A-D])(?:[\)\].:\s-]|$)",
            r"\b(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?([A-D])\b",
        ):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return text
    if re.search(
        r"answer only with [“\"]?yes[”\"]? or [“\"]?no", question, re.IGNORECASE
    ):
        explicit = re.findall(
            r"\b(?:final\s+)?answer\s*(?:is|:)\s*(yes|no)\b",
            text,
            flags=re.IGNORECASE,
        )
        if explicit:
            return explicit[-1].title()
        match = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
        return match.group(1).title() if match else text
    if re.search(
        r"single number|single integer|indicating the total count",
        question,
        re.IGNORECASE,
    ):
        explicit = re.findall(
            r"\b(?:final\s+)?answer\s*(?:is|:)\s*(-?\d+)\b",
            text,
            flags=re.IGNORECASE,
        )
        if explicit:
            return explicit[-1]
        match = re.search(r"\b\d+\b", text)
        return match.group(0) if match else text
    return text


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_processor_contract(
    processor_path: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed when a candidate silently changes the visual-token budget."""

    contract = (protocol.get("inference") or {}).get("processor_contract") or {}
    config_path = processor_path / "preprocessor_config.json"
    if not config_path.is_file():
        raise ValueError(f"processor preprocessor config is missing: {config_path}")
    digest = sha256_file(config_path)
    if digest != contract.get("preprocessor_config_sha256"):
        raise ValueError(
            "processor preprocessor_config SHA-256 diverged from the locked protocol: "
            f"{digest}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "image_processor_type": contract.get("image_processor_type"),
        "patch_size": contract.get("patch_size"),
        "merge_size": contract.get("merge_size"),
        "temporal_patch_size": contract.get("temporal_patch_size"),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"processor {key}={config.get(key)!r}, expected {value!r}")
    size = config.get("size") or {}
    for key in ("shortest_edge", "longest_edge"):
        if size.get(key) != contract.get(key):
            raise ValueError(
                f"processor size.{key}={size.get(key)!r}, expected {contract.get(key)!r}"
            )
    return {
        "preprocessor_config_sha256": digest,
        "image_processor_type": config["image_processor_type"],
        "patch_size": int(config["patch_size"]),
        "merge_size": int(config["merge_size"]),
        "temporal_patch_size": int(config["temporal_patch_size"]),
        "shortest_edge": int(size["shortest_edge"]),
        "longest_edge": int(size["longest_edge"]),
        "reference_visual_tokens_per_frame": int(
            contract["reference_visual_tokens_per_frame"]
        ),
    }


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if args.uids is not None and args.uids_file is not None:
        raise ValueError("--uids and --uids-file are mutually exclusive")
    if args.batch_size <= 0 or args.flush_every_batches <= 0:
        raise ValueError("batch size and flush interval must be positive")
    if args.candidate_binary_verify_batch_size <= 0:
        raise ValueError("candidate verifier batch size must be positive")
    if args.candidate_conditioned_revisit_count < 0:
        raise ValueError("candidate-conditioned revisit count must be non-negative")
    if args.temperature <= 0 or not 0.0 < args.top_p <= 1.0 or args.top_k <= 0:
        raise ValueError("sampling parameters are invalid")
    if args.history_max_edge <= 0:
        raise ValueError("history max edge must be positive")
    if args.semantic_revisit_count < 0 or args.semantic_revisit_planner_max_tokens <= 0:
        raise ValueError("semantic revisit arguments are invalid")
    if args.model_identity_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", args.model_identity_sha256
    ):
        raise ValueError("model identity must be one lowercase SHA-256 value")
    if args.model_manifest is not None and args.model_identity_sha256 is not None:
        raise ValueError("model manifest path and sealed identity are mutually exclusive")
    if args.decode_cache_entries < 0 or args.video_reader_cache_entries < 0:
        raise ValueError("cache capacities must be non-negative")
    if args.decode_prefetch_batches < 0:
        raise ValueError("decode prefetch batches must be non-negative")
    if args.video_reader_num_threads <= 0:
        raise ValueError("video reader thread count must be positive")
    if args.predecoded_cache is not None and args.visual_input_mode not in {
        "image-list",
        "lowres-history-recent-images",
    }:
        raise ValueError(
            "predecoded cache requires image-list or lowres-history-recent-images"
        )
    review_requested = (
        args.review_recent_result is not None
        or args.review_history_result is not None
    )
    candidate_review_requested = bool(args.agent_candidate_result)
    deliberation_requested = args.deliberation_mode != "none"
    candidate_deliberation = bool(
        candidate_review_requested
        and args.deliberation_mode
        in {"candidate-verify", "candidate-evidence-verify"}
    )
    candidate_binary_verify = bool(args.candidate_binary_verify)
    semantic_revisit_requested = args.semantic_revisit_count > 0
    fixed_candidate_evidence_view = args.visual_input_mode == "image-list" and (
        args.frame_policy in {"causalbudget20", "causalbudget32r8"}
        or args.predecoded_cache is not None
    )
    if (
        args.deliberation_mode
        in {"candidate-verify", "candidate-evidence-verify"}
        and not candidate_review_requested
    ):
        raise ValueError(
            f"{args.deliberation_mode} requires anonymous candidate results"
        )
    if candidate_binary_verify and (
        not candidate_review_requested
        or not (semantic_revisit_requested or fixed_candidate_evidence_view)
        or deliberation_requested
        or args.do_sample
    ):
        raise ValueError(
            "binary candidate verification requires deterministic anonymous "
            "candidates plus the uniform semantic-revisit pixel view"
        )
    if args.candidate_conditioned_revisit_count and not candidate_binary_verify:
        raise ValueError(
            "candidate-conditioned revisit requires binary candidate verification"
        )
    if args.candidate_antipodal_verify and not candidate_binary_verify:
        raise ValueError(
            "antipodal candidate verification requires binary candidate verification"
        )
    if args.candidate_pairwise_verify and not candidate_binary_verify:
        raise ValueError(
            "pairwise candidate verification requires binary candidate verification"
        )
    if args.candidate_pairwise_verify and args.candidate_antipodal_verify:
        raise ValueError(
            "pairwise and antipodal candidate verification are mutually exclusive"
        )
    if args.candidate_pairwise_verify and args.candidate_conditioned_revisit_count:
        raise ValueError(
            "pairwise verification requires one identical fixed pixel view"
        )
    if review_requested and deliberation_requested:
        raise ValueError("temporal review and standalone deliberation are mutually exclusive")
    if semantic_revisit_requested and (
        args.frame_policy != "causalbudget32r8"
        or args.visual_input_mode != "lowres-history-recent-images"
        or review_requested
        or (deliberation_requested and not candidate_deliberation)
        or args.do_sample
    ):
        raise ValueError(
            "semantic revisit requires deterministic causalbudget32r8 "
            "lowres-history-recent-images without source-labelled review or "
            "standalone deliberation"
        )
    if candidate_review_requested and (
        not (semantic_revisit_requested or fixed_candidate_evidence_view)
        or review_requested
        or args.deliberation_mode
        not in {"none", "candidate-verify", "candidate-evidence-verify"}
        or len(args.agent_candidate_result) not in (2, 3)
    ):
        raise ValueError(
            "anonymous candidate review requires either semantic revisit or a "
            "fixed causalbudget20/causalbudget32r8 image-list evidence view and exactly two or "
            "three complete candidate result artifacts"
        )
    valid_review_view = (
        args.frame_policy
        in {
            "causalbudget20",
            "causalbudget32r8",
            "causalhybrid32",
            "causalhybrid32r8",
        }
        and args.visual_input_mode == "hybrid-history-video-recent-images"
    ) or (
        args.frame_policy == "prefix32"
        and args.visual_input_mode == "image-list"
    )
    if review_requested and not (
        args.review_recent_result is not None
        and args.review_history_result is not None
        and valid_review_view
    ):
        raise ValueError(
            "temporal review requires both proposal results and either "
            "causalbudget*/causalhybrid32* hybrid input or a fixed prefix32 "
            "image-list input"
        )
    decord_reader_pool_installed = install_decord_reader_pool(
        args.video_reader_cache_entries,
        num_threads=args.video_reader_num_threads,
    )

    protocol_report = validate_locked_protocol(args.protocol)
    protocol = load_locked_protocol(args.protocol)
    benchmark_config = protocol["benchmarks"][args.benchmark]
    if sha256_file(args.samples) != benchmark_config["export_sha256"]:
        raise ValueError(f"{args.samples}: not the locked {args.benchmark} export")

    sample_rows = list(iter_jsonl(args.samples))
    sample_by_uid = {str(row["uid"]): row for row in sample_rows}
    requested_uids = args.uids
    if args.uids_file is not None:
        requested_uids = [
            line.strip()
            for line in args.uids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    wanted = selected_uids(
        args.samples, args.start_index, args.end_index, requested_uids
    )
    queries = load_queries(args.benchmark, args.annotation)
    query_by_uid = {query.query_id: query for query in queries}
    unknown = [uid for uid in wanted if uid not in query_by_uid]
    if unknown:
        raise ValueError(f"sample UIDs absent from annotation: {unknown[:20]}")

    predecoded_cache = (
        PredecodedRecent4Cache(
            args.predecoded_cache,
            benchmark=args.benchmark,
            samples_sha256=benchmark_config["export_sha256"],
            annotation_sha256=sha256_file(args.annotation),
            wanted=wanted,
            frame_policy=args.frame_policy,
            visual_input_mode=args.visual_input_mode,
        )
        if args.predecoded_cache is not None
        else None
    )

    index_plan: dict[str, dict[str, Any]] | None = None
    index_plan_artifact: dict[str, Any] | None = None
    if args.causalbudget_index_plan is not None:
        if args.frame_policy not in {"causalbudget20", "causalbudget32r8"}:
            raise ValueError("causalbudget index plan requires causalbudget*")
        planned_model_frames = FRAME_POLICY_NUM_FRAMES[args.frame_policy]
        planned_recent_frames = 8 if args.frame_policy == "causalbudget32r8" else 4
        index_plan, index_plan_artifact = load_causalbudget_index_plan(
            args.causalbudget_index_plan,
            benchmark=args.benchmark,
            wanted=wanted,
            model_frames=planned_model_frames,
            recent_frames=planned_recent_frames,
            allow_question_conditioned=args.allow_question_conditioned_index_plan,
        )
    if (
        args.decode_prefetch_batches
        and index_plan is None
        and args.frame_policy in {"causalbudget20", "causalbudget32r8"}
    ):
        raise ValueError(
            "causalbudget decode prefetch requires a frozen index plan"
        )

    model_manifest_sha256 = (
        args.model_identity_sha256
        or (sha256_file(args.model_manifest) if args.model_manifest else None)
    )
    review_proposals: dict[str, dict[str, str]] | None = None
    review_inputs: dict[str, Any] | None = None
    if review_requested:
        assert args.review_recent_result is not None
        assert args.review_history_result is not None
        review_proposals, review_inputs = load_temporal_review_proposals(
            recent_path=args.review_recent_result,
            history_path=args.review_history_result,
            wanted=wanted,
            benchmark=args.benchmark,
            annotation_sha256=sha256_file(args.annotation),
            model_manifest_sha256=model_manifest_sha256,
        )
    candidate_proposals: list[dict[str, str]] | None = None
    candidate_inputs: list[dict[str, Any]] | None = None
    if candidate_review_requested:
        candidate_proposals, candidate_inputs = load_anonymous_candidate_proposals(
            paths=args.agent_candidate_result,
            wanted=wanted,
            benchmark=args.benchmark,
            annotation_sha256=sha256_file(args.annotation),
            model_manifest_sha256=model_manifest_sha256,
        )

    processor_path = args.processor or args.model
    processor_contract = validate_processor_contract(processor_path, protocol)
    experimental_policy = (
        args.frame_policy != "recent4"
        or args.visual_input_mode != "image-list"
        or deliberation_requested
        or semantic_revisit_requested
        or candidate_review_requested
    )
    policy_config = (
        protocol["inference"]
        if not experimental_policy
        else {
            "causal_prefix_only": True,
            "frame_policy": args.frame_policy,
            "visual_input_mode": args.visual_input_mode,
            "controller_input": None,
            "routing_rule": None,
            "model_visible_frames": FRAME_POLICY_NUM_FRAMES[args.frame_policy],
            "maximum_model_visible_frames": FRAME_POLICY_NUM_FRAMES[args.frame_policy],
            "extra_frames_inspected": 0,
            "persistent_memory": False,
            "training_free": True,
        }
    )
    if args.frame_policy in {"causalbudget20", "causalbudget32r8"}:
        if args.visual_input_mode not in {
            "image-list",
            "native-video",
            "hybrid-history-video-recent-images",
            "storyboard-recent-images",
            "storyboard-keyframes-recent-images",
            "lowres-history-recent-images",
        }:
            raise ValueError(
                "causalbudget* requires image-list, native-video, or hybrid "
                "history-video input so its historical pixel budget remains bounded"
            )
        policy_config.update(
            {
                "controller_input": "causal_prefix_rgb_probe_16x16",
                "routing_rule": (
                    "uniform_question_conditioned_semantic_revisit_union_event_score"
                    if args.allow_question_conditioned_index_plan
                    else "one_continuous_novelty_curvature_recency_score"
                ),
                "question_conditioned_index_plan": bool(
                    args.allow_question_conditioned_index_plan
                ),
                "candidate_probe_frames": 64,
                "recent_frames_reserved": (
                    8 if args.frame_policy == "causalbudget32r8" else 4
                ),
                "visual_interface": args.visual_input_mode,
                "storyboard_contract": (
                    "chronological_4_column_timestamped_sheet_plus_exact_recent8"
                    if args.visual_input_mode
                    in {
                        "storyboard-recent-images",
                        "storyboard-keyframes-recent-images",
                    }
                    else None
                ),
                "lowres_history_contract": (
                    {
                        "history_max_edge": args.history_max_edge,
                        "recent_frames_keep_source_resolution": True,
                        "question_task_group_or_benchmark_router": False,
                    }
                    if args.visual_input_mode == "lowres-history-recent-images"
                    else None
                ),
                "model_visible_frames": FRAME_POLICY_NUM_FRAMES[args.frame_policy],
                "maximum_model_visible_frames": FRAME_POLICY_NUM_FRAMES[
                    args.frame_policy
                ],
                "extra_frames_inspected": (
                    32 if args.frame_policy == "causalbudget32r8" else 44
                ),
                "benchmark_task_or_group_router": False,
                "question_keyword_router": False,
                "semantic_revisit": (
                    {
                        "planner": "uniform_query_conditioned_historical_frame_revisit",
                        "planner_visual_interface": args.semantic_revisit_planner_mode,
                        "planner_storyboard_chunk_frames": (
                            16
                            if args.semantic_revisit_planner_mode == "storyboard16"
                            else None
                        ),
                        "historical_frames_revisited": args.semantic_revisit_count,
                        "source_resolution_revisit": True,
                        "question_type_router": False,
                        "benchmark_task_router": False,
                        "gold_available": False,
                        "prior_prediction_candidates_available": (
                            candidate_review_requested
                        ),
                    }
                    if semantic_revisit_requested
                    else None
                ),
            }
        )
    if args.frame_policy in {"causalhybrid32", "causalhybrid32r8"}:
        if args.visual_input_mode not in {
            "image-list",
            "hybrid-history-video-recent-images",
        }:
            raise ValueError(
                "causalhybrid32* requires image-list or hybrid "
                "history-video/recent-images input"
            )
        policy_config.update(
            {
                "controller_input": "causal_source_frame_indices_only",
                "routing_rule": (
                    "fixed_uniform_history_plus_exact_recent8"
                    if args.frame_policy == "causalhybrid32r8"
                    else "fixed_uniform_history_plus_exact_recent4"
                ),
                "history_frame_budget": (
                    24 if args.frame_policy == "causalhybrid32r8" else 28
                ),
                "recent_frames_reserved": (
                    8 if args.frame_policy == "causalhybrid32r8" else 4
                ),
                "visual_interface": args.visual_input_mode,
                "model_visible_frames": 32,
                "maximum_model_visible_frames": 32,
                "extra_frames_inspected": 0,
                "benchmark_task_or_group_router": False,
                "question_keyword_router": False,
                "query_conditioned_frame_selection": False,
            }
        )
    if review_requested:
        policy_config.update(
            {
                "temporal_review": "recent_proposal+history_proposal+hybrid_pixels",
                "review_template_sha256": hashlib.sha256(
                    (
                        ANONYMOUS_TEMPORAL_REVIEW_TEMPLATE
                        if args.review_proposal_label_mode
                        == "anonymous-hash-counterbalanced"
                        else TEMPORAL_REVIEW_TEMPLATE
                    ).encode()
                ).hexdigest(),
                "review_proposal_label_mode": args.review_proposal_label_mode,
                "proposal_selection_uses_gold": False,
                "proposal_selection_uses_task_or_group": False,
            }
        )
    if candidate_review_requested:
        policy_config.update(
            {
                "candidate_verification": (
                    "anonymous_uid_hash_counterbalanced_distinct_predictions"
                ),
                "candidate_result_count": len(args.agent_candidate_result),
                "candidate_template_sha256": hashlib.sha256(
                    ANONYMOUS_MULTI_CANDIDATE_TEMPLATE.encode()
                ).hexdigest(),
                "candidate_fields_used": ["rows[].uid", "rows[].prediction"],
                "candidate_source_labels_visible_to_actor": False,
                "candidate_score_or_gold_fields_used": False,
                "question_task_group_or_benchmark_router": False,
                "binary_candidate_verifier": (
                    {
                        "decision": (
                            "maximum_order_symmetric_pairwise_support"
                            if args.candidate_pairwise_verify
                            else (
                                "maximum_antipodal_support_probability"
                                if args.candidate_antipodal_verify
                                else "maximum_isolated_yes_probability"
                            )
                        ),
                        "candidate_sources_visible": False,
                        "same_pixels_per_candidate": not bool(
                            args.candidate_conditioned_revisit_count
                        ),
                        "verification_batch_size": args.candidate_binary_verify_batch_size,
                        "antipodal_support_and_rejection": (
                            args.candidate_antipodal_verify
                        ),
                        "order_symmetric_pairwise_verification": (
                            args.candidate_pairwise_verify
                        ),
                        "candidate_conditioned_revisit_count": (
                            args.candidate_conditioned_revisit_count
                        ),
                        "direct_answer_added_as_candidate": True,
                        "question_task_group_or_benchmark_router": False,
                        "score_or_gold_fields_used": False,
                    }
                    if candidate_binary_verify
                    else None
                ),
            }
        )
    if deliberation_requested:
        deliberation_template = (
            CANDIDATE_EVIDENCE_DELIBERATION_TEMPLATE
            if args.deliberation_mode == "candidate-evidence-verify"
            else TEMPORAL_DELIBERATION_TEMPLATE
        )
        policy_config.update(
            {
                "deliberation": args.deliberation_mode,
                "deliberation_template_sha256": hashlib.sha256(
                    deliberation_template.encode()
                ).hexdigest(),
                "question_task_group_or_benchmark_router": False,
            }
        )
    if candidate_binary_verify:
        prompt_contract = (
            f"{PROMPT_CONTRACT}\n{ANONYMOUS_MULTI_CANDIDATE_TEMPLATE}\n"
            f"{CANDIDATE_BINARY_VERIFICATION_TEMPLATE}"
            f"{' ' + CANDIDATE_REJECTION_VERIFICATION_TEMPLATE if args.candidate_antipodal_verify else ''}"
            f"{' ' + CANDIDATE_PAIRWISE_VERIFICATION_TEMPLATE if args.candidate_pairwise_verify else ''}"
            f"{' ' + CANDIDATE_REVISIT_TEMPLATE if args.candidate_conditioned_revisit_count else ''}"
        )
    elif candidate_deliberation:
        deliberation_template = (
            CANDIDATE_EVIDENCE_DELIBERATION_TEMPLATE
            if args.deliberation_mode == "candidate-evidence-verify"
            else TEMPORAL_DELIBERATION_TEMPLATE
        )
        prompt_contract = (
            f"{PROMPT_CONTRACT}\n{ANONYMOUS_MULTI_CANDIDATE_TEMPLATE}\n"
            f"{deliberation_template}"
        )
    elif review_requested:
        prompt_contract = (
            f"{PROMPT_CONTRACT}\n"
            f"{ANONYMOUS_TEMPORAL_REVIEW_TEMPLATE if args.review_proposal_label_mode == 'anonymous-hash-counterbalanced' else TEMPORAL_REVIEW_TEMPLATE}"
        )
    elif deliberation_requested:
        prompt_contract = f"{PROMPT_CONTRACT}\n{TEMPORAL_DELIBERATION_TEMPLATE}"
    elif candidate_review_requested:
        prompt_contract = f"{PROMPT_CONTRACT}\n{ANONYMOUS_MULTI_CANDIDATE_TEMPLATE}"
    else:
        prompt_contract = PROMPT_CONTRACT
    identity = {
        "format": (
            "iclr_video.native_video_timestamp_ab_eval.v1"
            if args.visual_input_mode == "native-video"
            else (
                "iclr_video.training_free_active4_eval.v1"
                if experimental_policy
                else "iclr_video.recent4_full_eval.v1"
            )
        ),
        "schema_version": "recent4-full-1",
        "benchmark": args.benchmark,
        "model": str(args.model),
        "model_manifest": str(args.model_manifest) if args.model_manifest else None,
        "model_manifest_sha256": model_manifest_sha256,
        "processor": str(processor_path),
        "processor_contract": processor_contract,
        "processor_config_sha256": processor_contract[
            "preprocessor_config_sha256"
        ],
        "adapter": None,
        "evaluator": str(Path(__file__).resolve()),
        "evaluation_protocol": protocol["protocol_id"],
        "evaluation_protocol_sha256": sha256_file(args.protocol),
        "protocol_validation": protocol_report["status"],
        "inference_policy": (
            f"training_free_{args.frame_policy}_{args.visual_input_mode}"
            f"{'_temporal_review' if review_requested else ''}_v1"
            f"{'_temporal_deliberation' if deliberation_requested else ''}"
            f"{'_anonymous_candidate_verify' if candidate_review_requested else ''}"
            f"{'_binary_candidate_commit' if candidate_binary_verify else ''}"
            f"{'_antipodal' if args.candidate_antipodal_verify else ''}"
            f"{'_pairwise' if args.candidate_pairwise_verify else ''}"
            f"{'_candidate_revisit' + str(args.candidate_conditioned_revisit_count) if args.candidate_conditioned_revisit_count else ''}"
            f"{'_semantic_revisit_' + args.semantic_revisit_planner_mode if semantic_revisit_requested else ''}"
            if experimental_policy
            else "recent4_full_v1"
        ),
        "policy_config": policy_config,
        "inference_policy_hash": canonical_hash(policy_config),
        "comparable_under_locked_recent4_protocol": not experimental_policy,
        "generation_config": {
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else 0.0,
            "top_p": args.top_p if args.do_sample else None,
            "top_k": args.top_k if args.do_sample else None,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "record_choice_confidence": args.record_choice_confidence,
            "candidate_binary_verify": candidate_binary_verify,
            "candidate_antipodal_verify": args.candidate_antipodal_verify,
            "candidate_pairwise_verify": args.candidate_pairwise_verify,
            "choice_confidence_semantics": (
                "first generation step full-vocabulary margin for control, plus visible-option margin when available"
                if args.record_choice_confidence
                else None
            ),
            "attn_implementation": args.attn_implementation,
            "dtype": "bfloat16",
        },
        "execution_optimization": {
            "inference_batch_size": args.batch_size,
            "flush_every_batches": args.flush_every_batches,
            "decode_cache_entries": args.decode_cache_entries,
            "video_reader_cache_entries": args.video_reader_cache_entries,
            "video_reader_num_threads": args.video_reader_num_threads,
            "decord_reader_pool_installed": decord_reader_pool_installed,
            "cache_key": "video_path+query_time+recent_frames+sampling_fps+frame_policy",
            "source_metadata_cache_entries": 8,
            "semantics": (
                "exact defensive-copy frame cache plus exact qwen-vl-utils "
                "decord reader reuse; sampling and resizing unchanged"
            ),
            "causalbudget_index_plan": index_plan_artifact,
            "decode_prefetch_batches": args.decode_prefetch_batches,
            "decode_prefetch_semantics": (
                (
                    "same exact FFmpeg decoder and frozen source indices, overlapped with GPU inference"
                    if index_plan is not None
                    else (
                        "same exact recent-window decoder and frozen query timestamp, overlapped with GPU inference"
                        if args.frame_policy == "recent4"
                        else "same fixed causal-hybrid decoder, overlapped with GPU inference"
                    )
                )
                if args.decode_prefetch_batches
                else None
            ),
        },
        "system_prompt": "",
        "system_prompt_sha256": hashlib.sha256(b"").hexdigest(),
        "prompt_contract": prompt_contract,
        "prompt_contract_sha256": hashlib.sha256(prompt_contract.encode()).hexdigest(),
        "temporal_review_inputs": review_inputs,
        "anonymous_candidate_inputs": candidate_inputs,
        "annotation": str(args.annotation),
        "annotation_sha256": sha256_file(args.annotation),
        "samples": str(args.samples),
        "samples_sha256": sha256_file(args.samples),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "software": {
            "transformers": importlib.metadata.version("transformers"),
            "qwen_vl_utils": importlib.metadata.version("qwen-vl-utils"),
            "decord": importlib.metadata.version("decord"),
        },
    }

    rows: list[dict[str, Any]] = []
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        for field in (
            "model",
            "model_manifest_sha256",
            "evaluation_protocol_sha256",
            "inference_policy_hash",
            "annotation_sha256",
            "samples_sha256",
            "runner_sha256",
            "execution_optimization",
            "processor_config_sha256",
        ):
            if previous.get(field) != identity.get(field):
                raise ValueError(f"resume identity mismatch: {field}")
        wanted_set = set(wanted)
        rows = [
            row
            for row in previous.get("rows", [])
            if str(row.get("uid")) in wanted_set
        ]
        if args.retry_errors:
            rows = [row for row in rows if not row.get("error")]
    completed = {str(row.get("uid")) for row in rows}

    processor = AutoProcessor.from_pretrained(
        processor_path, trust_remote_code=True, local_files_only=True
    )
    # Decoder-only batched generation must read logits at the final real token,
    # not at right-padding positions.
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
    ).eval()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    patch_size = int(processor.image_processor.patch_size)
    merge_size = int(processor.image_processor.merge_size)
    temporal_patch_size = int(processor.video_processor.temporal_patch_size)
    size_factor = patch_size * merge_size
    identity["processor_geometry"] = {
        "patch_size": patch_size,
        "merge_size": merge_size,
        "temporal_patch_size": temporal_patch_size,
        "size_factor": size_factor,
        "visual_input_mode": args.visual_input_mode,
        "qwen_vl_utils_fetch_video_image_patch_size": 14,
        "shortest_edge": processor_contract["shortest_edge"],
        "longest_edge": processor_contract["longest_edge"],
        "reference_visual_tokens_per_frame": processor_contract[
            "reference_visual_tokens_per_frame"
        ],
    }
    decode_cache = RecentWindowCache(args.decode_cache_entries)
    temporal_budget_cache = CausalTemporalBudgetCache(
        args.decode_cache_entries,
        model_frames=(32 if args.frame_policy == "causalbudget32r8" else 20),
        recent_frames=(8 if args.frame_policy == "causalbudget32r8" else 4),
    )

    pending = [(position, uid) for position, uid in enumerate(wanted) if uid not in completed]

    def payload() -> dict[str, Any]:
        predictions = {
            str(item["uid"]): item.get("prediction", "") for item in rows
        }
        stage_names = (
            "decode_seconds",
            "prompt_prepare_seconds",
            "processor_seconds",
            "h2d_seconds",
            "generate_seconds",
            "decode_output_seconds",
        )
        stage_totals = {
            name: round(
                sum(float((item.get("timing") or {}).get(name, 0.0)) for item in rows),
                6,
            )
            for name in stage_names
        }
        return {
            **identity,
            "execution_statistics": {
                "decode_window_cache_hits": decode_cache.hits,
                "decode_window_cache_misses": decode_cache.misses,
                "decode_window_singleflight_hits": decode_cache.singleflight_hits,
                "temporal_budget_cache_hits": temporal_budget_cache.hits,
                "temporal_budget_cache_misses": temporal_budget_cache.misses,
                "video_reader_cache_hits": (
                    _DECORD_READER_POOL.hits if _DECORD_READER_POOL else 0
                ),
                "video_reader_cache_misses": (
                    _DECORD_READER_POOL.misses if _DECORD_READER_POOL else 0
                ),
                "stage_seconds_total": stage_totals,
                "stage_timing_samples": sum(bool(item.get("timing")) for item in rows),
            },
            "sample_slice": {
                "start": args.start_index,
                "end": args.end_index,
                "uids": args.uids,
                "uids_file": str(args.uids_file) if args.uids_file else None,
                "uids_file_sha256": (
                    sha256_file(args.uids_file) if args.uids_file else None
                ),
            },
            "num_samples": len(rows),
            "num_errors": sum(bool(item.get("error")) for item in rows),
            "rows": rows,
            "score": score_predictions(
                [query_by_uid[str(item["uid"])] for item in rows],
                predictions,
            ),
        }

    def run_semantic_revisit_planner(prepared: list[dict[str, Any]]) -> None:
        """Select a bounded exact-pixel revisit using one uniform planner call."""

        planned: list[dict[str, Any]] = []
        planner_texts: list[str] = []
        planner_image_lists: list[list[Any]] = []
        for item in prepared:
            sampling = item["sampling"]
            frame_indices = [
                int(value) for value in sampling.get("frame_indices") or []
            ]
            recent_ids = {
                int(value) for value in sampling.get("recent_frame_indices") or []
            }
            history_offsets = [
                offset
                for offset, frame_index in enumerate(frame_indices)
                if frame_index not in recent_ids
            ]
            recent_offsets = [
                offset
                for offset, frame_index in enumerate(frame_indices)
                if frame_index in recent_ids
            ]
            if not history_offsets:
                item["semantic_revisit"] = {
                    "planner_called": False,
                    "reason": "no_historical_frame_before_recent_frontier",
                    "question_task_group_or_benchmark_router": False,
                }
                continue
            planner_frames = [
                item["frames"][offset]
                for offset in [*history_offsets, *recent_offsets]
            ]
            planner_timestamps = [
                float(sampling["frame_timestamps"][offset])
                for offset in [*history_offsets, *recent_offsets]
            ]
            planner_model_frames = planner_frames
            if args.semantic_revisit_planner_mode == "storyboard16":
                planner_model_frames = [
                    compose_temporal_storyboard(
                        planner_frames[start : start + 16],
                        planner_timestamps[start : start + 16],
                        columns=4,
                        tile_width=224,
                        tile_height=126,
                        label_height=24,
                        label_start=start + 1,
                    )
                    for start in range(0, len(planner_frames), 16)
                ]
            planner_prompt = render_semantic_revisit_prompt(
                item["answer_prompt"],
                history_count=len(history_offsets),
                revisit_count=args.semantic_revisit_count,
                planner_mode=args.semantic_revisit_planner_mode,
            )
            planner_messages = build_visual_messages(
                frames=planner_model_frames,
                prompt=planner_prompt,
                visual_input_mode="image-list",
            )
            planner_texts.append(
                processor.apply_chat_template(
                    planner_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            planner_image_lists.append(planner_model_frames)
            planned.append(
                {
                    "item": item,
                    "frame_indices": frame_indices,
                    "history_offsets": history_offsets,
                    "recent_offsets": recent_offsets,
                    "planner_frames": planner_model_frames,
                    "planner_source_frame_count": len(planner_frames),
                }
            )
        if not planned:
            return

        planner_started = time.time()
        flat_images = [frame for frames in planner_image_lists for frame in frames]
        planner_inputs = processor(
            text=planner_texts,
            images=flat_images,
            padding=True,
            return_tensors="pt",
        )
        planner_grids = planner_inputs.image_grid_thw.detach().cpu().tolist()
        planner_input_tokens = planner_inputs.attention_mask.sum(dim=-1).tolist()
        padded_input_tokens = int(planner_inputs.input_ids.shape[-1])
        model_inputs = {key: value.to(model.device) for key, value in planner_inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=args.semantic_revisit_planner_max_tokens,
            )
        completion_ids = generated[:, padded_input_tokens:]
        decoded = processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        latency_share = (time.time() - planner_started) / len(planned)
        grid_offset = 0
        for plan, raw, input_tokens, output_ids, image_count in zip(
            planned,
            decoded,
            planner_input_tokens,
            completion_ids,
            map(len, planner_image_lists),
            strict=True,
        ):
            item = plan["item"]
            sampling = item["sampling"]
            history_offsets = plan["history_offsets"]
            frame_indices = plan["frame_indices"]
            score_by_frame = {
                int(value["frame_index"]): float(value["score"])
                for value in (sampling.get("controller") or {}).get("selected") or []
            }
            fallback_positions = sorted(
                range(1, len(history_offsets) + 1),
                key=lambda position: (
                    score_by_frame.get(
                        frame_indices[history_offsets[position - 1]], 0.0
                    ),
                    -position,
                ),
                reverse=True,
            )
            selected_positions = parse_semantic_revisit_positions(
                raw,
                history_count=len(history_offsets),
                revisit_count=args.semantic_revisit_count,
                fallback_positions=fallback_positions,
            )
            selected_offsets = [
                history_offsets[position - 1] for position in selected_positions
            ]
            answer_offsets = sorted({*selected_offsets, *plan["recent_offsets"]})
            source_frames = item["revisit_source_frames"]
            item["frames"] = [source_frames[offset] for offset in answer_offsets]
            answer_messages = build_visual_messages(
                frames=item["frames"],
                prompt=item["answer_prompt"],
                visual_input_mode="image-list",
            )
            item["text"] = processor.apply_chat_template(
                answer_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            item_grids = planner_grids[grid_offset : grid_offset + image_count]
            grid_offset += image_count
            planner_visual_tokens = sum(
                int(grid[0] * grid[1] * grid[2] // (merge_size**2))
                for grid in item_grids
            )
            selected_frame_indices = [frame_indices[offset] for offset in answer_offsets]
            sampling.update(
                {
                    "semantic_revisit_selected_history_positions": selected_positions,
                    "semantic_revisit_model_input_frame_indices": selected_frame_indices,
                    "semantic_revisit_count": len(selected_offsets),
                    "semantic_revisit_question_conditioned": True,
                    "semantic_revisit_question_task_group_or_benchmark_router": False,
                }
            )
            item["planner_visual_tokens"] = planner_visual_tokens
            item["planner_input_tokens"] = int(input_tokens)
            item["planner_output_tokens"] = int(output_ids.shape[-1])
            item["planner_latency_seconds"] = latency_share
            item["planner_image_grid_thw"] = item_grids
            item["planner_frames"] = plan["planner_frames"]
            item["planner_source_frame_count"] = plan["planner_source_frame_count"]
            item["semantic_revisit"] = {
                "planner_called": True,
                "planner_mode": args.semantic_revisit_planner_mode,
                "raw_response": raw.strip(),
                "selected_history_positions": selected_positions,
                "selected_frame_indices": selected_frame_indices,
                "fallback_rule": "query_independent_controller_score",
                "question_task_group_or_benchmark_router": False,
                "gold_available": False,
                "prior_prediction_candidates_available": (
                    candidate_review_requested
                ),
            }

    def infer(prepared: list[dict[str, Any]]) -> None:
        if semantic_revisit_requested and any(
            "semantic_revisit" not in item for item in prepared
        ):
            run_semantic_revisit_planner(prepared)
        texts = [str(item["text"]) for item in prepared]
        if args.visual_input_mode in {
            "image-list",
            "storyboard-recent-images",
            "storyboard-keyframes-recent-images",
            "lowres-history-recent-images",
        }:
            image_lists = [item["frames"] for item in prepared]
            video_lists = [[] for _ in prepared]
        elif args.visual_input_mode == "native-video":
            image_lists = [[] for _ in prepared]
            video_lists = [item["frames"] for item in prepared]
        else:
            image_lists = [item["recent_frames"] for item in prepared]
            video_lists = [item["history_frames"] for item in prepared]

        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
        }
        flat_images = [frame for frames in image_lists for frame in frames]
        if flat_images:
            processor_kwargs["images"] = flat_images
        metadata_payloads: list[dict[str, Any] | None] = [None] * len(prepared)
        nonempty_videos = []
        video_metadata = []
        for item_index, (item, frames) in enumerate(
            zip(prepared, video_lists, strict=True)
        ):
            if not frames:
                continue
            from transformers.video_utils import VideoMetadata

            if args.visual_input_mode == "native-video":
                video_indices = item["native_video_frame_indices"]
            else:
                video_indices = [
                    int(value) for value in item["history_frame_indices"]
                ]
            payload = native_video_metadata_payload(
                item["sampling"], frame_indices=video_indices
            )
            metadata_payloads[item_index] = payload
            item["native_video_metadata"] = payload
            nonempty_videos.append(frames)
            video_metadata.append(VideoMetadata(**payload))
        if nonempty_videos:
            processor_kwargs.update(
                {
                    "videos": nonempty_videos,
                    "video_metadata": video_metadata,
                    "do_sample_frames": False,
                }
            )
        processor_started = time.perf_counter()
        inputs = processor(**processor_kwargs)
        processor_seconds = time.perf_counter() - processor_started
        image_grids = (
            inputs.image_grid_thw.detach().cpu().tolist()
            if flat_images
            else []
        )
        video_grids = (
            inputs.video_grid_thw.detach().cpu().tolist()
            if nonempty_videos
            else []
        )
        image_counts = [len(frames) for frames in image_lists]
        video_counts = [int(bool(frames)) for frames in video_lists]
        if len(image_grids) != sum(image_counts):
            raise RuntimeError("image grid count mismatch")
        if len(video_grids) != sum(video_counts):
            raise RuntimeError("video grid count mismatch")
        padded_input_tokens = int(inputs.input_ids.shape[-1])
        meaningful_input_tokens = inputs.attention_mask.sum(dim=-1).tolist()
        h2d_started = time.perf_counter()
        model_inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
            if key != "video_metadata"
        }
        h2d_seconds = time.perf_counter() - h2d_started
        inference_started = time.perf_counter()
        generation_kwargs: dict[str, Any] = {"do_sample": args.do_sample}
        if args.do_sample:
            generation_kwargs.update(
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                }
            )
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                **generation_kwargs,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=args.record_choice_confidence,
                output_scores=args.record_choice_confidence,
            )
        generate_seconds = time.perf_counter() - inference_started
        decode_output_started = time.perf_counter()
        if args.record_choice_confidence:
            generated_sequences = generated.sequences
            first_step_scores = generated.scores[0].float().cpu()
        else:
            generated_sequences = generated
            first_step_scores = None
        completion_ids = generated_sequences[:, padded_input_tokens:]
        decoded = processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decode_output_seconds = time.perf_counter() - decode_output_started
        timing_share = {
            "processor_seconds": processor_seconds / len(prepared),
            "h2d_seconds": h2d_seconds / len(prepared),
            "generate_seconds": generate_seconds / len(prepared),
            "decode_output_seconds": decode_output_seconds / len(prepared),
        }
        inference_share = sum(timing_share.values())
        image_offset = 0
        video_offset = 0
        for batch_index, (
            item,
            raw,
            input_tokens,
            ids,
            image_count,
            video_count,
        ) in enumerate(zip(
            prepared,
            decoded,
            meaningful_input_tokens,
            completion_ids,
            image_counts,
            video_counts,
            strict=True,
        )):
            frame_count = int(
                item["sampling"].get("storyboard_source_frame_count")
                or (
                    len(item["sampling"].get("frame_indices") or [])
                    if semantic_revisit_requested
                    else 0
                )
                or len(item["frames"])
            )
            item_image_grids = image_grids[
                image_offset : image_offset + image_count
            ]
            item_video_grids = video_grids[
                video_offset : video_offset + video_count
            ]
            image_offset += image_count
            video_offset += video_count
            visual_tokens = sum(
                int(grid[0] * grid[1] * grid[2] // (merge_size**2))
                for grid in [*item_image_grids, *item_video_grids]
            )
            raw = raw.strip()
            row = item["row"]
            item["timing"].update(timing_share)
            sample = item["sample"]
            choice_confidence = None
            generation_confidence = None
            if first_step_scores is not None:
                full_probabilities = torch.softmax(
                    first_step_scores[batch_index], dim=-1
                )
                full_values, full_ids = torch.topk(full_probabilities, k=2)
                generation_confidence = {
                    "top_token_ids": [int(value) for value in full_ids.tolist()],
                    "top_tokens": [
                        processor.tokenizer.decode([int(value)])
                        for value in full_ids.tolist()
                    ],
                    "top_probabilities": [
                        float(value) for value in full_values.tolist()
                    ],
                    "top_probability": float(full_values[0]),
                    "top2_margin": float(full_values[0] - full_values[1]),
                    "source": "first_generation_step_full_vocabulary_logits",
                    "task_or_group_used": False,
                }
                option_count = min(26, len(sample.get("options") or []))
                if option_count >= 2:
                    letters = [
                        chr(ord("A") + index) for index in range(option_count)
                    ]
                    token_ids: list[int] = []
                    for letter in letters:
                        encoded = processor.tokenizer.encode(
                            letter, add_special_tokens=False
                        )
                        if len(encoded) != 1:
                            raise RuntimeError(
                                f"choice {letter!r} is not a single tokenizer token: {encoded}"
                            )
                        token_ids.append(int(encoded[0]))
                    logits = first_step_scores[batch_index, token_ids]
                    probabilities = torch.softmax(logits, dim=-1)
                    ordered = torch.argsort(probabilities, descending=True)
                    top1 = int(ordered[0])
                    top2 = int(ordered[1])
                    choice_confidence = {
                        "letters": letters,
                        "probabilities": {
                            letter: float(probabilities[index])
                            for index, letter in enumerate(letters)
                        },
                        "top_choice": letters[top1],
                        "top_probability": float(probabilities[top1]),
                        "top2_margin": float(
                            probabilities[top1] - probabilities[top2]
                        ),
                        "source": "first_generation_step_choice_logits",
                        "task_or_group_used": False,
                    }
            planner_input_tokens = int(item.get("planner_input_tokens") or 0)
            planner_output_tokens = int(item.get("planner_output_tokens") or 0)
            planner_visual_tokens = int(item.get("planner_visual_tokens") or 0)
            planner_latency = float(item.get("planner_latency_seconds") or 0.0)
            row.update(
                {
                    "prediction": normalize_answer(
                        raw,
                        str(sample["question"]),
                        [str(option) for option in sample.get("options") or []],
                    ),
                    "raw_response": raw,
                    "choice_confidence": choice_confidence,
                    "generation_confidence": generation_confidence,
                    "error": None,
                    "sampling": item["sampling"],
                    "batch_fallback": bool(row.get("batch_fallback", False)),
                    "decode_cache_hit": item["decode_cache_hit"],
                    "frames_observed": frame_count,
                    "media_calls": 2 if item.get("planner_frames") else 1,
                    "num_turns": 2 if item.get("planner_frames") else 1,
                    "model_processed_frames": int(
                        item.get("planner_source_frame_count")
                        or len(item.get("planner_frames") or [])
                    )
                    + len(item["frames"]),
                    "input_tokens": int(input_tokens) + planner_input_tokens,
                    "output_tokens": int(ids.shape[-1]) + planner_output_tokens,
                    "total_tokens": int(input_tokens)
                    + int(ids.shape[-1])
                    + planner_input_tokens
                    + planner_output_tokens,
                    "image_grid_thw": item_image_grids,
                    "planner_image_grid_thw": item.get("planner_image_grid_thw") or [],
                    "video_grid_thw": item_video_grids,
                    "visual_input_mode": args.visual_input_mode,
                    "native_video_metadata": item.get("native_video_metadata"),
                    "semantic_revisit": item.get("semantic_revisit"),
                    "visual_tokens": visual_tokens + planner_visual_tokens,
                    "latency_seconds": item["prepare_seconds"]
                    + planner_latency
                    + inference_share,
                    "timing": dict(item["timing"]),
                }
            )

        if candidate_binary_verify:
            # Release the direct-generation tensors before isolated candidate
            # checks. Candidate-conditioned revisits, when enabled, are symmetric:
            # every candidate gets the same planner and exact-pixel budget.
            del model_inputs, inputs, generated, generated_sequences, completion_ids
            verification_units: list[dict[str, Any]] = []
            for item in prepared:
                row = item["row"]
                candidate_values = {
                    str(value).strip()
                    for value in [
                        *(row.get("anonymous_candidate_proposals") or []),
                        row.get("prediction") or "",
                    ]
                    if str(value).strip()
                }
                candidates = sorted(
                    candidate_values,
                    key=lambda candidate: hashlib.sha256(
                        f"{row['uid']}\0{candidate}".encode()
                    ).hexdigest(),
                )
                row["direct_prediction_before_binary_verify"] = row["prediction"]
                row["binary_candidate_verification"] = {
                    "candidates": [],
                    "selected": row["prediction"],
                    "candidate_sources_visible": False,
                    "same_pixels_per_candidate": not bool(
                        args.candidate_conditioned_revisit_count
                    ),
                    "antipodal_verification": args.candidate_antipodal_verify,
                    "pairwise_verification": args.candidate_pairwise_verify,
                    "candidate_conditioned_revisit_count": (
                        args.candidate_conditioned_revisit_count
                    ),
                    "question_task_group_or_benchmark_router": False,
                    "score_or_gold_fields_used": False,
                }
                if len(candidates) < 2:
                    row["binary_candidate_verification"]["reason"] = (
                        "all_independent_proposals_agree"
                    )
                    continue
                for candidate in candidates:
                    verification_units.append(
                        {
                            "item": item,
                            "candidate": candidate,
                            "frames": list(item["frames"]),
                            "candidate_revisit": None,
                        }
                    )

            if args.candidate_conditioned_revisit_count:
                revisit_started = time.time()
                for start in range(
                    0,
                    len(verification_units),
                    args.candidate_binary_verify_batch_size,
                ):
                    chunk = verification_units[
                        start : start + args.candidate_binary_verify_batch_size
                    ]
                    planner_texts: list[str] = []
                    planner_frame_lists: list[list[Any]] = []
                    planner_metadata: list[dict[str, Any]] = []
                    for unit in chunk:
                        item = unit["item"]
                        sampling = item["sampling"]
                        frame_indices = [
                            int(value) for value in sampling.get("frame_indices") or []
                        ]
                        recent_ids = {
                            int(value)
                            for value in sampling.get("recent_frame_indices") or []
                        }
                        history_offsets = [
                            offset
                            for offset, frame_index in enumerate(frame_indices)
                            if frame_index not in recent_ids
                        ]
                        recent_offsets = [
                            offset
                            for offset, frame_index in enumerate(frame_indices)
                            if frame_index in recent_ids
                        ]
                        if not history_offsets:
                            unit["candidate_revisit"] = {
                                "planner_called": False,
                                "reason": "no_historical_frame_before_recent_frontier",
                            }
                            planner_metadata.append({"skip": True, "unit": unit})
                            continue
                        planner_prompt = render_candidate_revisit_prompt(
                            item["base_answer_prompt"],
                            candidate=unit["candidate"],
                            options=[
                                str(option)
                                for option in item["sample"].get("options") or []
                            ],
                            history_count=len(history_offsets),
                            revisit_count=args.candidate_conditioned_revisit_count,
                        )
                        planner_frames = list(item["planner_frames"])
                        planner_messages = build_visual_messages(
                            frames=planner_frames,
                            prompt=planner_prompt,
                            visual_input_mode="image-list",
                        )
                        planner_texts.append(
                            processor.apply_chat_template(
                                planner_messages,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        )
                        planner_frame_lists.append(planner_frames)
                        planner_metadata.append(
                            {
                                "skip": False,
                                "unit": unit,
                                "frame_indices": frame_indices,
                                "history_offsets": history_offsets,
                                "recent_offsets": recent_offsets,
                            }
                        )
                    active_metadata = [
                        value for value in planner_metadata if not value["skip"]
                    ]
                    if not active_metadata:
                        continue
                    planner_inputs = processor(
                        text=planner_texts,
                        images=[
                            frame
                            for frames in planner_frame_lists
                            for frame in frames
                        ],
                        padding=True,
                        return_tensors="pt",
                    )
                    planner_grids = (
                        planner_inputs.image_grid_thw.detach().cpu().tolist()
                    )
                    planner_input_tokens = (
                        planner_inputs.attention_mask.sum(dim=-1).tolist()
                    )
                    planner_model_inputs = {
                        key: value.to(model.device)
                        for key, value in planner_inputs.items()
                    }
                    with torch.inference_mode():
                        planner_output = model.generate(
                            **planner_model_inputs,
                            do_sample=False,
                            max_new_tokens=args.semantic_revisit_planner_max_tokens,
                        )
                    planner_padded_tokens = int(planner_inputs.input_ids.shape[-1])
                    planner_completion_ids = planner_output[:, planner_padded_tokens:]
                    planner_decoded = processor.batch_decode(
                        planner_completion_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    grid_offset = 0
                    for metadata, raw, input_tokens, output_ids, frames in zip(
                        active_metadata,
                        planner_decoded,
                        planner_input_tokens,
                        planner_completion_ids,
                        planner_frame_lists,
                        strict=True,
                    ):
                        unit = metadata["unit"]
                        item = unit["item"]
                        sampling = item["sampling"]
                        frame_indices = metadata["frame_indices"]
                        history_offsets = metadata["history_offsets"]
                        score_by_frame = {
                            int(value["frame_index"]): float(value["score"])
                            for value in (sampling.get("controller") or {}).get(
                                "selected"
                            )
                            or []
                        }
                        fallback_positions = sorted(
                            range(1, len(history_offsets) + 1),
                            key=lambda position: (
                                score_by_frame.get(
                                    frame_indices[history_offsets[position - 1]], 0.0
                                ),
                                -position,
                            ),
                            reverse=True,
                        )
                        selected_positions = parse_semantic_revisit_positions(
                            raw,
                            history_count=len(history_offsets),
                            revisit_count=args.candidate_conditioned_revisit_count,
                            fallback_positions=fallback_positions,
                        )
                        selected_offsets = [
                            history_offsets[position - 1]
                            for position in selected_positions
                        ]
                        answer_offsets = sorted(
                            {*selected_offsets, *metadata["recent_offsets"]}
                        )
                        unit["frames"] = [
                            item["revisit_source_frames"][offset]
                            for offset in answer_offsets
                        ]
                        selected_frame_indices = [
                            frame_indices[offset] for offset in answer_offsets
                        ]
                        item_grids = planner_grids[
                            grid_offset : grid_offset + len(frames)
                        ]
                        grid_offset += len(frames)
                        planner_visual_tokens = sum(
                            int(grid[0] * grid[1] * grid[2] // (merge_size**2))
                            for grid in item_grids
                        )
                        unit["candidate_revisit"] = {
                            "planner_called": True,
                            "raw_response": raw.strip(),
                            "selected_history_positions": selected_positions,
                            "selected_frame_indices": selected_frame_indices,
                            "question_task_group_or_benchmark_router": False,
                            "gold_available": False,
                        }
                        row = item["row"]
                        row["media_calls"] += 1
                        row["num_turns"] += 1
                        row["model_processed_frames"] += len(frames)
                        row["input_tokens"] += int(input_tokens)
                        row["output_tokens"] += int(output_ids.shape[-1])
                        row["total_tokens"] += int(input_tokens) + int(
                            output_ids.shape[-1]
                        )
                        row["visual_tokens"] += planner_visual_tokens
                    del planner_model_inputs, planner_inputs, planner_output
                revisit_latency = time.time() - revisit_started
                for item in prepared:
                    item["row"]["latency_seconds"] += revisit_latency / max(
                        1, len(prepared)
                    )

            for unit in verification_units:
                item = unit["item"]
                verification_prompt = render_candidate_binary_verification_prompt(
                    item["base_answer_prompt"],
                    candidate=unit["candidate"],
                    options=[
                        str(option) for option in item["sample"].get("options") or []
                    ],
                )
                messages = build_visual_messages(
                    frames=unit["frames"],
                    prompt=verification_prompt,
                    visual_input_mode="image-list",
                )
                unit["text"] = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            yes_ids = processor.tokenizer.encode("Yes", add_special_tokens=False)
            no_ids = processor.tokenizer.encode("No", add_special_tokens=False)
            if len(yes_ids) != 1 or len(no_ids) != 1:
                raise RuntimeError(
                    "binary candidate verifier requires single-token Yes and No labels"
                )
            label_ids = [int(yes_ids[0]), int(no_ids[0])]
            verification_started = time.time()
            verification_batch_size = args.candidate_binary_verify_batch_size
            for start in range(0, len(verification_units), verification_batch_size):
                chunk = verification_units[start : start + verification_batch_size]
                image_counts = [len(unit["frames"]) for unit in chunk]
                verifier_inputs = processor(
                    text=[unit["text"] for unit in chunk],
                    images=[
                        frame
                        for unit in chunk
                        for frame in unit["frames"]
                    ],
                    padding=True,
                    return_tensors="pt",
                )
                verifier_grids = verifier_inputs.image_grid_thw.detach().cpu().tolist()
                verifier_input_tokens = (
                    verifier_inputs.attention_mask.sum(dim=-1).tolist()
                )
                verifier_model_inputs = {
                    key: value.to(model.device)
                    for key, value in verifier_inputs.items()
                }
                with torch.inference_mode():
                    verifier_output = model.generate(
                        **verifier_model_inputs,
                        do_sample=False,
                        max_new_tokens=1,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                label_logits = verifier_output.scores[0][:, label_ids].float().cpu()
                label_probabilities = torch.softmax(label_logits, dim=-1)
                grid_offset = 0
                for unit, probabilities, input_tokens, image_count in zip(
                    chunk,
                    label_probabilities,
                    verifier_input_tokens,
                    image_counts,
                    strict=True,
                ):
                    item = unit["item"]
                    row = item["row"]
                    item_grids = verifier_grids[
                        grid_offset : grid_offset + image_count
                    ]
                    grid_offset += image_count
                    visual_tokens = sum(
                        int(grid[0] * grid[1] * grid[2] // (merge_size**2))
                        for grid in item_grids
                    )
                    row["binary_candidate_verification"]["candidates"].append(
                        {
                            "candidate": unit["candidate"],
                            "yes_probability": float(probabilities[0]),
                            "no_probability": float(probabilities[1]),
                            "candidate_revisit": unit["candidate_revisit"],
                        }
                    )
                    row["media_calls"] += 1
                    row["num_turns"] += 1
                    row["model_processed_frames"] += image_count
                    row["input_tokens"] += int(input_tokens)
                    row["output_tokens"] += 1
                    row["total_tokens"] += int(input_tokens) + 1
                    row["visual_tokens"] += visual_tokens
                del verifier_model_inputs, verifier_inputs, verifier_output

            if args.candidate_antipodal_verify:
                for unit in verification_units:
                    item = unit["item"]
                    rejection_prompt = render_candidate_rejection_verification_prompt(
                        item["base_answer_prompt"],
                        candidate=unit["candidate"],
                        options=[
                            str(option)
                            for option in item["sample"].get("options") or []
                        ],
                    )
                    messages = build_visual_messages(
                        frames=unit["frames"],
                        prompt=rejection_prompt,
                        visual_input_mode="image-list",
                    )
                    unit["rejection_text"] = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                for start in range(
                    0, len(verification_units), verification_batch_size
                ):
                    chunk = verification_units[
                        start : start + verification_batch_size
                    ]
                    image_counts = [len(unit["frames"]) for unit in chunk]
                    rejection_inputs = processor(
                        text=[unit["rejection_text"] for unit in chunk],
                        images=[
                            frame
                            for unit in chunk
                            for frame in unit["frames"]
                        ],
                        padding=True,
                        return_tensors="pt",
                    )
                    rejection_grids = (
                        rejection_inputs.image_grid_thw.detach().cpu().tolist()
                    )
                    rejection_input_tokens = (
                        rejection_inputs.attention_mask.sum(dim=-1).tolist()
                    )
                    rejection_model_inputs = {
                        key: value.to(model.device)
                        for key, value in rejection_inputs.items()
                    }
                    with torch.inference_mode():
                        rejection_output = model.generate(
                            **rejection_model_inputs,
                            do_sample=False,
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                    rejection_logits = rejection_output.scores[0][
                        :, label_ids
                    ].float().cpu()
                    rejection_probabilities = torch.softmax(
                        rejection_logits, dim=-1
                    )
                    grid_offset = 0
                    for unit, probabilities, input_tokens, image_count in zip(
                        chunk,
                        rejection_probabilities,
                        rejection_input_tokens,
                        image_counts,
                        strict=True,
                    ):
                        item = unit["item"]
                        row = item["row"]
                        record = next(
                            value
                            for value in row["binary_candidate_verification"][
                                "candidates"
                            ]
                            if value["candidate"] == unit["candidate"]
                        )
                        rejection_yes = float(probabilities[0])
                        rejection_no = float(probabilities[1])
                        record.update(
                            {
                                "rejection_yes_probability": rejection_yes,
                                "rejection_no_probability": rejection_no,
                                "antipodal_support_probability": 0.5
                                * (float(record["yes_probability"]) + rejection_no),
                                "polarity_prompts_same_pixels": True,
                            }
                        )
                        item_grids = rejection_grids[
                            grid_offset : grid_offset + image_count
                        ]
                        grid_offset += image_count
                        visual_tokens = sum(
                            int(grid[0] * grid[1] * grid[2] // (merge_size**2))
                            for grid in item_grids
                        )
                        row["media_calls"] += 1
                        row["num_turns"] += 1
                        row["model_processed_frames"] += image_count
                        row["input_tokens"] += int(input_tokens)
                        row["output_tokens"] += 1
                        row["total_tokens"] += int(input_tokens) + 1
                        row["visual_tokens"] += visual_tokens
                    del rejection_model_inputs, rejection_inputs, rejection_output

            verification_latency = time.time() - verification_started
            for item in prepared:
                row = item["row"]
                records = row["binary_candidate_verification"]["candidates"]
                if not records:
                    continue
                probabilities = {
                    str(record["candidate"]): float(
                        record[
                            "antipodal_support_probability"
                            if args.candidate_antipodal_verify
                            else "yes_probability"
                        ]
                    )
                    for record in records
                }
                selected = choose_binary_verified_candidate(
                    uid=str(row["uid"]), candidate_probabilities=probabilities
                )
                row["prediction"] = selected
                row["binary_candidate_verification"]["selected"] = selected
                row["binary_candidate_verification"]["decision"] = (
                    "maximum_antipodal_support_probability"
                    if args.candidate_antipodal_verify
                    else "maximum_isolated_yes_probability"
                )
                row["latency_seconds"] += verification_latency / max(
                    1, len(prepared)
                )

            if args.candidate_pairwise_verify:
                pairwise_started = time.time()
                directional_units: list[dict[str, Any]] = []
                for item in prepared:
                    row = item["row"]
                    records = row["binary_candidate_verification"]["candidates"]
                    candidates = sorted(
                        (str(record["candidate"]) for record in records),
                        key=lambda candidate: hashlib.sha256(
                            f"{row['uid']}\0{candidate}".encode()
                        ).hexdigest(),
                    )
                    if len(candidates) < 2:
                        continue
                    pairwise = {
                        "pairs": [],
                        "same_pixels_for_both_orders": True,
                        "candidate_sources_visible": False,
                        "question_task_group_or_benchmark_router": False,
                        "score_or_gold_fields_used": False,
                    }
                    row["binary_candidate_verification"]["pairwise"] = pairwise
                    options = [
                        str(option) for option in item["sample"].get("options") or []
                    ]
                    for first_index in range(len(candidates)):
                        for second_index in range(first_index + 1, len(candidates)):
                            first = candidates[first_index]
                            second = candidates[second_index]
                            pair_record = {
                                "first_candidate": first,
                                "second_candidate": second,
                            }
                            pairwise["pairs"].append(pair_record)
                            for reversed_order in (False, True):
                                shown_first, shown_second = (
                                    (second, first)
                                    if reversed_order
                                    else (first, second)
                                )
                                prompt = render_candidate_pairwise_verification_prompt(
                                    item["base_answer_prompt"],
                                    first_candidate=shown_first,
                                    second_candidate=shown_second,
                                    options=options,
                                )
                                messages = build_visual_messages(
                                    frames=item["frames"],
                                    prompt=prompt,
                                    visual_input_mode="image-list",
                                )
                                directional_units.append(
                                    {
                                        "item": item,
                                        "pair_record": pair_record,
                                        "reversed_order": reversed_order,
                                        "frames": list(item["frames"]),
                                        "text": processor.apply_chat_template(
                                            messages,
                                            tokenize=False,
                                            add_generation_prompt=True,
                                        ),
                                    }
                                )

                a_ids = processor.tokenizer.encode("A", add_special_tokens=False)
                b_ids = processor.tokenizer.encode("B", add_special_tokens=False)
                if len(a_ids) != 1 or len(b_ids) != 1:
                    raise RuntimeError(
                        "pairwise candidate verifier requires single-token A and B labels"
                    )
                pair_label_ids = [int(a_ids[0]), int(b_ids[0])]
                for start in range(
                    0, len(directional_units), verification_batch_size
                ):
                    chunk = directional_units[
                        start : start + verification_batch_size
                    ]
                    image_counts = [len(unit["frames"]) for unit in chunk]
                    pair_inputs = processor(
                        text=[unit["text"] for unit in chunk],
                        images=[
                            frame
                            for unit in chunk
                            for frame in unit["frames"]
                        ],
                        padding=True,
                        return_tensors="pt",
                    )
                    pair_grids = pair_inputs.image_grid_thw.detach().cpu().tolist()
                    pair_input_tokens = pair_inputs.attention_mask.sum(dim=-1).tolist()
                    pair_model_inputs = {
                        key: value.to(model.device) for key, value in pair_inputs.items()
                    }
                    with torch.inference_mode():
                        pair_output = model.generate(
                            **pair_model_inputs,
                            do_sample=False,
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                    pair_logits = pair_output.scores[0][
                        :, pair_label_ids
                    ].float().cpu()
                    pair_probabilities = torch.softmax(pair_logits, dim=-1)
                    grid_offset = 0
                    for unit, probabilities, input_tokens, image_count in zip(
                        chunk,
                        pair_probabilities,
                        pair_input_tokens,
                        image_counts,
                        strict=True,
                    ):
                        row = unit["item"]["row"]
                        record = unit["pair_record"]
                        prefix = "reverse" if unit["reversed_order"] else "forward"
                        record[f"{prefix}_a_probability"] = float(probabilities[0])
                        record[f"{prefix}_b_probability"] = float(probabilities[1])
                        item_grids = pair_grids[
                            grid_offset : grid_offset + image_count
                        ]
                        grid_offset += image_count
                        visual_tokens = sum(
                            int(grid[0] * grid[1] * grid[2] // (merge_size**2))
                            for grid in item_grids
                        )
                        row["media_calls"] += 1
                        row["num_turns"] += 1
                        row["model_processed_frames"] += image_count
                        row["input_tokens"] += int(input_tokens)
                        row["output_tokens"] += 1
                        row["total_tokens"] += int(input_tokens) + 1
                        row["visual_tokens"] += visual_tokens
                    del pair_model_inputs, pair_inputs, pair_output

                for item in prepared:
                    row = item["row"]
                    pairwise = row["binary_candidate_verification"].get("pairwise")
                    if not pairwise:
                        continue
                    support_totals: dict[str, float] = {}
                    support_counts: dict[str, int] = {}
                    for record in pairwise["pairs"]:
                        first = str(record["first_candidate"])
                        second = str(record["second_candidate"])
                        first_support = 0.5 * (
                            float(record["forward_a_probability"])
                            + float(record["reverse_b_probability"])
                        )
                        second_support = 0.5 * (
                            float(record["forward_b_probability"])
                            + float(record["reverse_a_probability"])
                        )
                        record["first_order_symmetric_support"] = first_support
                        record["second_order_symmetric_support"] = second_support
                        for candidate, support in (
                            (first, first_support),
                            (second, second_support),
                        ):
                            support_totals[candidate] = (
                                support_totals.get(candidate, 0.0) + support
                            )
                            support_counts[candidate] = support_counts.get(candidate, 0) + 1
                    averaged = {
                        candidate: support_totals[candidate] / support_counts[candidate]
                        for candidate in support_totals
                    }
                    selected = choose_binary_verified_candidate(
                        uid=str(row["uid"]), candidate_probabilities=averaged
                    )
                    pairwise["candidate_support"] = averaged
                    pairwise["selected"] = selected
                    pairwise["decision"] = (
                        "maximum_order_symmetric_pairwise_support"
                    )
                    row["prediction"] = selected
                    row["binary_candidate_verification"]["selected"] = selected
                    row["binary_candidate_verification"]["decision"] = (
                        "maximum_order_symmetric_pairwise_support"
                    )
                    row["latency_seconds"] += (
                        time.time() - pairwise_started
                    ) / max(1, len(prepared))

    batches_since_flush = 0
    batch_offsets = list(range(0, len(pending), args.batch_size))
    prefetch_workers = max(1, args.batch_size * args.decode_prefetch_batches)
    prefetch_pool = (
        concurrent.futures.ThreadPoolExecutor(max_workers=prefetch_workers)
        if args.decode_prefetch_batches
        else None
    )
    prefetched: dict[
        str,
        concurrent.futures.Future[Any]
        | tuple[concurrent.futures.Future[Any], int],
    ] = {}

    def schedule_prefetch(batch_number: int) -> None:
        if prefetch_pool is None or predecoded_cache is not None:
            return
        # Prefix policies repeatedly ask several causal endpoints from the
        # same long video.  Decode the union of their exact frame indices once
        # instead of replaying FFmpeg from frame zero for every query.  Each
        # UID still receives only its own query-independent prefix indices.
        resolved_prefetch_policy = resolve_frame_policy(args.frame_policy)
        if index_plan is None and resolved_prefetch_policy.startswith("prefix"):
            grouped: OrderedDict[str, tuple[Path, list[tuple[str, float]]]] = (
                OrderedDict()
            )
            for future_batch in range(
                batch_number,
                min(
                    len(batch_offsets),
                    batch_number + args.decode_prefetch_batches + 1,
                ),
            ):
                future_offset = batch_offsets[future_batch]
                for _, future_uid in pending[
                    future_offset : future_offset + args.batch_size
                ]:
                    if future_uid in prefetched:
                        continue
                    future_query = query_by_uid[future_uid]
                    future_video = resolve_video_path(
                        future_query, args.video_root
                    )
                    video_key = str(future_video)
                    if video_key not in grouped:
                        grouped[video_key] = (future_video, [])
                    grouped[video_key][1].append(
                        (
                            future_uid,
                            float(sample_by_uid[future_uid]["query_time"]),
                        )
                    )
            prefix_frames = FRAME_POLICY_NUM_FRAMES[resolved_prefetch_policy]
            for future_video, requests in grouped.values():
                future = prefetch_pool.submit(
                    decode_prefix_windows_ffmpeg_batched,
                    future_video,
                    visible_untils=[value[1] for value in requests],
                    prefix_frames=prefix_frames,
                )
                for result_index, (future_uid, _) in enumerate(requests):
                    prefetched[future_uid] = (future, result_index)
            return
        for future_batch in range(
            batch_number,
            min(len(batch_offsets), batch_number + args.decode_prefetch_batches + 1),
        ):
            future_offset = batch_offsets[future_batch]
            for _, future_uid in pending[
                future_offset : future_offset + args.batch_size
            ]:
                if future_uid in prefetched:
                    continue
                future_query = query_by_uid[future_uid]
                future_video = resolve_video_path(future_query, args.video_root)
                if index_plan is not None:
                    planned = index_plan[future_uid]
                    future_indices = [
                        int(value)
                        for value in (planned.get("sampling") or {}).get("frame_indices") or []
                    ]
                    if not 1 <= len(future_indices) <= FRAME_POLICY_NUM_FRAMES[
                        args.frame_policy
                    ]:
                        raise ValueError(
                            "causalbudget index plan is outside its frame budget"
                        )
                    prefetched[future_uid] = prefetch_pool.submit(
                        decode_exact_indices_ffmpeg,
                        future_video,
                        future_indices,
                    )
                elif args.frame_policy in {"causalhybrid32", "causalhybrid32r8"}:
                    future_sample = sample_by_uid[future_uid]
                    prefetched[future_uid] = prefetch_pool.submit(
                        decode_causal_hybrid_prefix,
                        future_video,
                        visible_until=float(future_sample["query_time"]),
                        recent_frames=(
                            8 if args.frame_policy == "causalhybrid32r8" else 4
                        ),
                    )
                else:
                    future_sample = sample_by_uid[future_uid]
                    prefetched[future_uid] = prefetch_pool.submit(
                        decode_cache.get,
                        future_video,
                        visible_until=float(future_sample["query_time"]),
                        recent_frames=int(protocol["inference"]["recent_frames"]),
                        sampling_fps=float(protocol["inference"]["sampling_fps"]),
                        frame_policy=resolve_frame_policy(args.frame_policy),
                    )

    try:
      for batch_number, offset in enumerate(batch_offsets):
        schedule_prefetch(batch_number)
        batch = pending[offset : offset + args.batch_size]
        prepared: list[dict[str, Any]] = []
        batch_rows: list[dict[str, Any]] = []
        for position, uid in batch:
            query = query_by_uid[uid]
            sample = sample_by_uid[uid]
            video_path = resolve_video_path(query, args.video_root)
            started = time.time()
            row: dict[str, Any] = {
                "uid": uid,
                "benchmark": args.benchmark,
                "task": query.task,
                "group": query.group,
                "query_time": float(sample["query_time"]),
                "video": str(video_path),
                "sample_position": position,
            }
            batch_rows.append(row)
            try:
                decode_started = time.perf_counter()
                options = [
                    str(option) for option in sample.get("options") or []
                ]
                resolved_frame_policy = resolve_frame_policy(args.frame_policy)
                if predecoded_cache is not None:
                    frames, sampling = predecoded_cache.get(
                        uid,
                        video_path=video_path,
                        query_time=float(sample["query_time"]),
                    )
                    decode_cache_hit = True
                elif resolved_frame_policy in {
                    "causalbudget20",
                    "causalbudget32r8",
                }:
                    if index_plan is not None:
                        planned = index_plan[uid]
                        planned_video = Path(str(planned["video"]))
                        same_video = (
                            planned_video.resolve() == video_path.resolve()
                            or planned_video.name == video_path.name
                        )
                        if (
                            not same_video
                            or float(planned["query_time"]).hex()
                            != float(sample["query_time"]).hex()
                        ):
                            raise ValueError("causalbudget index-plan row diverged")
                        sampling = copy.deepcopy(planned["sampling"])
                        sampling["index_plan_video_relocated"] = (
                            planned_video.resolve() != video_path.resolve()
                        )
                        frame_indices = [
                            int(value) for value in sampling.get("frame_indices") or []
                        ]
                        if not 1 <= len(frame_indices) <= FRAME_POLICY_NUM_FRAMES[
                            resolved_frame_policy
                        ]:
                            raise ValueError(
                                "causalbudget index plan is outside its frame budget"
                            )
                        if uid in prefetched:
                            frames = prefetched.pop(uid).result()
                        else:
                            frames = decode_exact_indices_ffmpeg(video_path, frame_indices)
                        sampling["decode_backend"] = (
                            "precomputed_probe_plan_then_exact_source_indices"
                        )
                        sampling["index_plan_reused"] = True
                        decode_cache_hit = False
                    else:
                        frames, sampling, decode_cache_hit = temporal_budget_cache.get(
                            video_path,
                            visible_until=float(sample["query_time"]),
                        )
                elif resolved_frame_policy in {
                    "causalhybrid32",
                    "causalhybrid32r8",
                }:
                    if uid in prefetched:
                        frames, sampling = prefetched.pop(uid).result()
                    else:
                        frames, sampling = decode_causal_hybrid_prefix(
                            video_path,
                            visible_until=float(sample["query_time"]),
                            recent_frames=(
                                8
                                if resolved_frame_policy == "causalhybrid32r8"
                                else 4
                            ),
                        )
                    decode_cache_hit = False
                else:
                    if uid in prefetched:
                        prefetched_value = prefetched.pop(uid)
                        if isinstance(prefetched_value, tuple):
                            prefetched_future, result_index = prefetched_value
                            frames, sampling = prefetched_future.result()[result_index]
                            decode_cache_hit = False
                        else:
                            frames, sampling, decode_cache_hit = (
                                prefetched_value.result()
                            )
                    else:
                        frames, sampling, decode_cache_hit = decode_cache.get(
                            video_path,
                            visible_until=float(sample["query_time"]),
                            recent_frames=int(protocol["inference"]["recent_frames"]),
                            sampling_fps=float(protocol["inference"]["sampling_fps"]),
                            frame_policy=resolved_frame_policy,
                        )
                sampling["requested_frame_policy"] = args.frame_policy
                sampling["resolved_frame_policy"] = resolved_frame_policy
                decode_seconds = time.perf_counter() - decode_started
                prompt_prepare_started = time.perf_counter()
                history_frames: list[Any] = []
                recent_frames_for_hybrid: list[Any] = []
                history_frame_indices: list[int] = []
                native_video_frame_indices = [
                    int(value) for value in sampling.get("frame_indices") or []
                ]
                revisit_source_frames = (
                    list(frames) if semantic_revisit_requested else []
                )
                if args.visual_input_mode == "native-video":
                    (
                        frames,
                        native_video_frame_indices,
                        temporal_padding_repeats,
                    ) = pad_native_video_temporal_axis(
                        frames,
                        native_video_frame_indices,
                        temporal_patch_size=temporal_patch_size,
                    )
                    sampling["native_video_temporal_padding_repeats"] = (
                        temporal_padding_repeats
                    )
                    sampling["native_video_model_input_frame_indices"] = (
                        native_video_frame_indices
                    )
                if args.visual_input_mode == "hybrid-history-video-recent-images":
                    (
                        history_frames,
                        history_frame_indices,
                        recent_frames_for_hybrid,
                        fallback_indices,
                    ) = partition_hybrid_frames(
                        frames,
                        sampling,
                        temporal_patch_size=temporal_patch_size,
                    )
                    sampling["hybrid_history_image_fallback_indices"] = (
                        fallback_indices
                    )
                if args.visual_input_mode in {
                    "storyboard-recent-images",
                    "storyboard-keyframes-recent-images",
                }:
                    source_frame_count = len(frames)
                    source_timestamps = [
                        float(value)
                        for value in sampling.get("frame_timestamps") or []
                    ]
                    recent_ids = {
                        int(value)
                        for value in sampling.get("recent_frame_indices") or []
                    }
                    exact_recent = [
                        frame
                        for frame, frame_index in zip(
                            frames,
                            sampling.get("frame_indices") or [],
                            strict=True,
                        )
                        if int(frame_index) in recent_ids
                    ]
                    exact_keyframes: list[Any] = []
                    exact_keyframe_indices: list[int] = []
                    if (
                        args.visual_input_mode
                        == "storyboard-keyframes-recent-images"
                    ):
                        selected_scores = {
                            int(value["frame_index"]): float(value["score"])
                            for value in (sampling.get("controller") or {}).get(
                                "selected"
                            )
                            or []
                        }
                        history = [
                            (position, int(frame_index), frame)
                            for position, (frame_index, frame) in enumerate(
                                zip(
                                    sampling.get("frame_indices") or [],
                                    frames,
                                    strict=True,
                                )
                            )
                            if int(frame_index) not in recent_ids
                        ]
                        chosen = sorted(
                            history,
                            key=lambda value: (
                                selected_scores.get(value[1], 0.0),
                                -value[0],
                            ),
                            reverse=True,
                        )[:8]
                        chosen.sort(key=lambda value: value[0])
                        exact_keyframe_indices = [value[1] for value in chosen]
                        exact_keyframes = [value[2] for value in chosen]
                    storyboard = compose_temporal_storyboard(
                        frames, source_timestamps
                    )
                    frames = [storyboard, *exact_keyframes, *exact_recent]
                    sampling.update(
                        {
                            "storyboard_source_frame_count": source_frame_count,
                            "storyboard_model_image_count": len(frames),
                            "storyboard_columns": 4,
                            "storyboard_tile_size": [224, 126],
                            "storyboard_timestamp_overlay": True,
                            "storyboard_exact_keyframe_indices": exact_keyframe_indices,
                            "storyboard_exact_keyframe_rule": (
                                "top8_query_independent_temporal_controller_score"
                                if exact_keyframe_indices
                                else "none"
                            ),
                            "storyboard_question_task_group_or_benchmark_router": False,
                        }
                    )
                if args.visual_input_mode == "lowres-history-recent-images":
                    frame_indices_for_resize = [
                        int(value)
                        for value in sampling.get("frame_indices") or []
                    ]
                    recent_ids_for_resize = {
                        int(value)
                        for value in sampling.get("recent_frame_indices") or []
                    }
                    frames = downscale_temporal_history(
                        frames,
                        frame_indices_for_resize,
                        recent_ids_for_resize,
                        max_edge=args.history_max_edge,
                    )
                    sampling.update(
                        {
                            "history_max_edge": args.history_max_edge,
                            "history_resolution_policy": "fixed_max_edge_thumbnail",
                            "recent_resolution_policy": "unchanged_source_decode",
                            "resolution_question_task_group_or_benchmark_router": False,
                        }
                    )
                prompt = render_question(
                    args.benchmark,
                    str(sample["question"]),
                    options,
                )
                if args.visual_input_mode in {
                    "storyboard-recent-images",
                    "storyboard-keyframes-recent-images",
                }:
                    detail = (
                        "The following images are selected historical keyframes "
                        "and then the latest frames at full fidelity."
                        if args.visual_input_mode
                        == "storyboard-keyframes-recent-images"
                        else "The following images are the latest frames at full fidelity."
                    )
                    prompt = (
                        "The first image is a chronological storyboard; each panel "
                        f"is labeled with its relative video time. {detail}\n\n" + prompt
                    )
                base_answer_prompt = prompt
                if review_proposals is not None:
                    prompt = render_temporal_review_prompt(
                        prompt,
                        recent_prediction=review_proposals["recent"][uid],
                        history_prediction=review_proposals["history"][uid],
                        uid=uid,
                        label_mode=args.review_proposal_label_mode,
                    )
                    row["temporal_review_proposals"] = {
                        "recent": review_proposals["recent"][uid],
                        "history": review_proposals["history"][uid],
                    }
                elif candidate_proposals is not None:
                    prompt, ordered_candidates = render_anonymous_candidate_prompt(
                        prompt,
                        predictions=[
                            proposal[uid]
                            for proposal in candidate_proposals
                            if uid in proposal
                        ],
                        uid=uid,
                    )
                    row["anonymous_candidate_proposals"] = ordered_candidates
                    row["anonymous_candidate_source_labels_visible"] = False
                    row["anonymous_candidate_score_or_gold_fields_used"] = False
                    if candidate_deliberation:
                        prompt = (
                            render_candidate_evidence_deliberation_prompt(prompt)
                            if args.deliberation_mode
                            == "candidate-evidence-verify"
                            else render_temporal_deliberation_prompt(prompt)
                        )
                elif deliberation_requested:
                    prompt = render_temporal_deliberation_prompt(prompt)
                messages = build_visual_messages(
                    frames=frames,
                    prompt=prompt,
                    visual_input_mode=args.visual_input_mode,
                    history_frames=history_frames,
                    recent_frames=recent_frames_for_hybrid,
                )
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prepared.append(
                    {
                        "decode_cache_hit": decode_cache_hit,
                        "frames": frames,
                        "history_frames": history_frames,
                        "history_frame_indices": history_frame_indices,
                        "native_video_frame_indices": native_video_frame_indices,
                        "recent_frames": recent_frames_for_hybrid,
                        "prepare_seconds": time.time() - started,
                        "timing": {
                            "decode_seconds": decode_seconds,
                            "prompt_prepare_seconds": (
                                time.perf_counter() - prompt_prepare_started
                            ),
                        },
                        "row": row,
                        "sample": sample,
                        "sampling": sampling,
                        "text": text,
                        "answer_prompt": prompt,
                        "base_answer_prompt": base_answer_prompt,
                        "revisit_source_frames": revisit_source_frames,
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "prediction": "",
                        "raw_response": "",
                        "error": f"{type(exc).__name__}: {exc}",
                        "frames_observed": 0,
                        "media_calls": 0,
                        "num_turns": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "visual_tokens": 0,
                        "latency_seconds": time.time() - started,
                    }
                )

        if prepared:
            try:
                infer(prepared)
            except Exception as batch_exc:
                # A heterogeneous batch may exceed memory even though each
                # exact sample is valid. Retry one by one before recording an
                # error, preserving correctness and resumability.
                if len(prepared) == 1:
                    item = prepared[0]
                    item["row"].update(
                        {
                            "prediction": "",
                            "raw_response": "",
                            "error": f"{type(batch_exc).__name__}: {batch_exc}",
                            "frames_observed": 0,
                            "media_calls": 0,
                            "num_turns": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "visual_tokens": 0,
                            "latency_seconds": item["prepare_seconds"],
                        }
                    )
                else:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for item in prepared:
                        item["row"]["batch_fallback"] = True
                        try:
                            infer([item])
                        except Exception as exc:
                            item["row"].update(
                                {
                                    "prediction": "",
                                    "raw_response": "",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "frames_observed": 0,
                                    "media_calls": 0,
                                    "num_turns": 0,
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                    "total_tokens": 0,
                                    "visual_tokens": 0,
                                    "latency_seconds": item["prepare_seconds"],
                                }
                            )

        rows.extend(batch_rows)
        batches_since_flush += 1
        if batches_since_flush >= args.flush_every_batches:
            atomic_write(args.output, payload())
            batches_since_flush = 0
        for row in batch_rows:
            print(
                json.dumps(
                    {
                        "uid": row["uid"],
                        "position": row["sample_position"],
                        "prediction": row["prediction"],
                        "error": row["error"],
                        "frames": row["frames_observed"],
                        "visual_tokens": row["visual_tokens"],
                        "latency_seconds": round(row["latency_seconds"], 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=True, cancel_futures=True)
        if predecoded_cache is not None:
            predecoded_cache.close()

    # Always rewrite once at completion, including no-op resume runs, so the
    # output identity and score are fully materialized.
    atomic_write(args.output, payload())


if __name__ == "__main__":
    main()
