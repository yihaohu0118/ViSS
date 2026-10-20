"""Router-free causal temporal budgeting for 2D streaming video.

The controller deliberately has no task, benchmark, answer-space, or protocol
input.  It receives a single chronological stream of continuous frame signals,
reserves a recent high-fidelity window, and spends the remaining fixed model
budget on high-value historical transitions.  Query compatibility may enter as
one continuous score, but it cannot select a different policy or frame budget.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence


POLICY_ID = "unified_causal_temporal_budget_v0"
TRANSITION_PAIR_POLICY_ID = "unified_causal_transition_pair_budget_v0"


class TemporalBudgetError(ValueError):
    """Raised when the controller input violates its causal public contract."""


@dataclass(frozen=True)
class TemporalBudgetConfig:
    max_model_frames: int = 20
    recent_frames: int = 4
    novelty_weight: float = 0.20
    curvature_weight: float = 0.45
    query_weight: float = 0.25
    recency_weight: float = 0.10

    def validate(self) -> None:
        if self.max_model_frames <= 0:
            raise TemporalBudgetError("max_model_frames must be positive")
        if self.recent_frames <= 0:
            raise TemporalBudgetError("recent_frames must be positive")
        if self.recent_frames > self.max_model_frames:
            raise TemporalBudgetError("recent window exceeds model frame budget")
        weights = (
            self.novelty_weight,
            self.curvature_weight,
            self.query_weight,
            self.recency_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise TemporalBudgetError("controller weights must be finite and non-negative")
        if sum(weights) <= 0:
            raise TemporalBudgetError("at least one controller weight must be positive")

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            {"policy_id": POLICY_ID, **asdict(self)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrameSignal:
    frame_index: int
    timestamp: float
    novelty: float
    curvature: float
    query_relevance: float = 0.0

    def validate(self) -> None:
        if self.frame_index < 0:
            raise TemporalBudgetError("frame index must be non-negative")
        values = (
            self.timestamp,
            self.novelty,
            self.curvature,
            self.query_relevance,
        )
        if any(not math.isfinite(value) for value in values):
            raise TemporalBudgetError("frame signals must be finite")


def _rank_normalize(values: Sequence[float]) -> list[float]:
    """Average-tie rank normalization in [0, 1], invariant to score scale."""

    if not values:
        return []
    if len(values) == 1:
        return [0.0]
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    offset = 0
    while offset < len(order):
        end = offset + 1
        while end < len(order) and values[order[end]] == values[order[offset]]:
            end += 1
        average = (offset + end - 1) / 2.0
        normalized = average / (len(order) - 1)
        for position in range(offset, end):
            ranks[order[position]] = normalized
        offset = end
    return ranks


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise TemporalBudgetError("feature vectors must have the same non-zero width")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    return 1.0 - cosine


def trajectory_change_scores(
    features: Sequence[Sequence[float]],
) -> tuple[list[float], list[float]]:
    """Compute first-order novelty and second-order trajectory curvature.

    Curvature is the cosine distance between consecutive displacement vectors,
    matching the scale-free geometric signal used by curvature-aware streaming
    memory without committing to a particular visual feature backend.
    """

    if not features:
        return [], []
    width = len(features[0])
    if width <= 0 or any(len(feature) != width for feature in features):
        raise TemporalBudgetError("features must form a non-empty rectangular matrix")
    vectors = [[float(value) for value in feature] for feature in features]
    if any(not math.isfinite(value) for row in vectors for value in row):
        raise TemporalBudgetError("features must be finite")

    novelty = [0.0]
    novelty.extend(
        _cosine_distance(vectors[index - 1], vectors[index])
        for index in range(1, len(vectors))
    )
    curvature = [0.0] * len(vectors)
    for index in range(1, len(vectors) - 1):
        previous_delta = [
            current - previous
            for previous, current in zip(
                vectors[index - 1], vectors[index], strict=True
            )
        ]
        next_delta = [
            future - current
            for current, future in zip(
                vectors[index], vectors[index + 1], strict=True
            )
        ]
        curvature[index] = _cosine_distance(previous_delta, next_delta)
    return novelty, curvature


def build_frame_signals(
    *,
    frame_indices: Sequence[int],
    timestamps: Sequence[float],
    features: Sequence[Sequence[float]],
    query_relevance: Sequence[float] | None = None,
) -> list[FrameSignal]:
    if not (len(frame_indices) == len(timestamps) == len(features)):
        raise TemporalBudgetError("indices, timestamps, and features must align")
    if query_relevance is None:
        query_relevance = [0.0] * len(frame_indices)
    if len(query_relevance) != len(frame_indices):
        raise TemporalBudgetError("query relevance must align with frames")
    novelty, curvature = trajectory_change_scores(features)
    return [
        FrameSignal(
            frame_index=int(index),
            timestamp=float(timestamp),
            novelty=float(novelty_value),
            curvature=float(curvature_value),
            query_relevance=float(relevance),
        )
        for index, timestamp, novelty_value, curvature_value, relevance in zip(
            frame_indices,
            timestamps,
            novelty,
            curvature,
            query_relevance,
            strict=True,
        )
    ]


def select_temporal_budget(
    signals: Sequence[FrameSignal],
    *,
    causal_frontier: float,
    config: TemporalBudgetConfig = TemporalBudgetConfig(),
) -> dict[str, Any]:
    """Select one deterministic, chronologically returned causal frame set."""

    config.validate()
    if not math.isfinite(causal_frontier) or causal_frontier < 0:
        raise TemporalBudgetError("causal frontier must be finite and non-negative")
    ordered = sorted(signals, key=lambda item: (item.timestamp, item.frame_index))
    if not ordered:
        raise TemporalBudgetError("temporal budget requires at least one frame")
    for signal in ordered:
        signal.validate()
        if signal.timestamp > causal_frontier + 1e-3:
            raise TemporalBudgetError("candidate frame crosses the causal frontier")
    indices = [signal.frame_index for signal in ordered]
    if len(indices) != len(set(indices)):
        raise TemporalBudgetError("candidate frame indices must be unique")
    if any(
        right.timestamp + 1e-9 < left.timestamp
        for left, right in zip(ordered, ordered[1:])
    ):
        raise TemporalBudgetError("candidate timestamps must be chronological")

    novelty = _rank_normalize([signal.novelty for signal in ordered])
    curvature = _rank_normalize([signal.curvature for signal in ordered])
    relevance = _rank_normalize([signal.query_relevance for signal in ordered])
    recency = _rank_normalize([signal.timestamp for signal in ordered])
    scored: list[dict[str, Any]] = []
    for signal, n_value, c_value, q_value, r_value in zip(
        ordered, novelty, curvature, relevance, recency, strict=True
    ):
        score = (
            config.novelty_weight * n_value
            + config.curvature_weight * c_value
            + config.query_weight * q_value
            + config.recency_weight * r_value
        )
        scored.append(
            {
                "frame_index": signal.frame_index,
                "timestamp": signal.timestamp,
                "score": score,
                "components": {
                    "novelty_rank": n_value,
                    "curvature_rank": c_value,
                    "query_relevance_rank": q_value,
                    "recency_rank": r_value,
                },
            }
        )

    recent_count = min(config.recent_frames, len(scored))
    recent_ids = {row["frame_index"] for row in scored[-recent_count:]}
    remaining = max(0, config.max_model_frames - recent_count)
    historical = [row for row in scored if row["frame_index"] not in recent_ids]
    chosen_history = sorted(
        historical,
        key=lambda row: (-row["score"], -row["timestamp"], row["frame_index"]),
    )[:remaining]
    chosen_ids = recent_ids | {row["frame_index"] for row in chosen_history}
    selected = [row for row in scored if row["frame_index"] in chosen_ids]
    for row in selected:
        row["allocation"] = (
            "recent_high_fidelity"
            if row["frame_index"] in recent_ids
            else "selected_history"
        )

    return {
        "policy_id": POLICY_ID,
        "policy_config": asdict(config),
        "policy_config_sha256": config.sha256,
        "benchmark_task_or_group_router": False,
        "question_keyword_router": False,
        "single_continuous_scoring_rule": True,
        "causal_frontier": float(causal_frontier),
        "candidate_frames_inspected": len(scored),
        "model_visible_frames": len(selected),
        "extra_frames_inspected": max(0, len(scored) - len(selected)),
        "selected_frame_indices": [row["frame_index"] for row in selected],
        "selected_timestamps": [row["timestamp"] for row in selected],
        "selected": selected,
        "future_frame_violations": 0,
    }


def select_transition_pair_budget(
    signals: Sequence[FrameSignal],
    *,
    causal_frontier: float,
    config: TemporalBudgetConfig = TemporalBudgetConfig(),
) -> dict[str, Any]:
    """Select task-blind historical before/after pairs plus the recent window.

    ``select_temporal_budget`` spends every historical slot independently.  A
    high-curvature frame can therefore be shown without its immediately
    preceding state, which is a poor interface for action change, occurrence,
    and temporal-order decisions.  This variant keeps the same scoring rule,
    frame budget, and recent reservation, but greedily spends historical slots
    on ``(previous, anchor)`` pairs.  Unused slots are filled by the original
    global ranking.  The question, task, benchmark, and model prediction never
    enter selection.
    """

    reference = select_temporal_budget(
        signals, causal_frontier=causal_frontier, config=config
    )
    ordered = sorted(signals, key=lambda item: (item.timestamp, item.frame_index))
    recent_count = min(config.recent_frames, len(ordered))
    recent_ids = {signal.frame_index for signal in ordered[-recent_count:]}
    history = [signal for signal in ordered if signal.frame_index not in recent_ids]
    history_budget = max(0, config.max_model_frames - recent_count)

    novelty = _rank_normalize([signal.novelty for signal in ordered])
    curvature = _rank_normalize([signal.curvature for signal in ordered])
    relevance = _rank_normalize([signal.query_relevance for signal in ordered])
    recency = _rank_normalize([signal.timestamp for signal in ordered])
    score_by_id = {
        signal.frame_index: (
            config.novelty_weight * n_value
            + config.curvature_weight * c_value
            + config.query_weight * q_value
            + config.recency_weight * r_value
        )
        for signal, n_value, c_value, q_value, r_value in zip(
            ordered, novelty, curvature, relevance, recency, strict=True
        )
    }
    history_position = {
        signal.frame_index: position for position, signal in enumerate(history)
    }
    ranked_anchors = sorted(
        history,
        key=lambda signal: (
            -score_by_id[signal.frame_index],
            -signal.timestamp,
            signal.frame_index,
        ),
    )
    chosen: set[int] = set()
    anchors: set[int] = set()
    contexts: set[int] = set()
    for anchor in ranked_anchors:
        if len(chosen) >= history_budget:
            break
        position = history_position[anchor.frame_index]
        context = history[position - 1] if position > 0 else None
        pair = (
            [context.frame_index, anchor.frame_index]
            if context is not None
            else [anchor.frame_index]
        )
        additions = [frame_id for frame_id in pair if frame_id not in chosen]
        if not additions or len(chosen) + len(additions) > history_budget:
            continue
        chosen.update(additions)
        anchors.add(anchor.frame_index)
        if context is not None:
            contexts.add(context.frame_index)

    for signal in ranked_anchors:
        if len(chosen) >= history_budget:
            break
        chosen.add(signal.frame_index)

    chosen_ids = recent_ids | chosen
    selected = []
    for signal in ordered:
        if signal.frame_index not in chosen_ids:
            continue
        if signal.frame_index in recent_ids:
            allocation = "recent_high_fidelity"
        elif signal.frame_index in anchors:
            allocation = "transition_anchor"
        elif signal.frame_index in contexts:
            allocation = "transition_context"
        else:
            allocation = "selected_history_fill"
        selected.append(
            {
                "frame_index": signal.frame_index,
                "timestamp": signal.timestamp,
                "score": score_by_id[signal.frame_index],
                "allocation": allocation,
            }
        )

    policy_payload = json.dumps(
        {"policy_id": TRANSITION_PAIR_POLICY_ID, **asdict(config)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "policy_id": TRANSITION_PAIR_POLICY_ID,
        "policy_config": asdict(config),
        "policy_config_sha256": hashlib.sha256(policy_payload.encode("utf-8")).hexdigest(),
        "reference_policy_config_sha256": reference["policy_config_sha256"],
        "benchmark_task_or_group_router": False,
        "question_keyword_router": False,
        "single_continuous_scoring_rule": True,
        "causal_frontier": float(causal_frontier),
        "candidate_frames_inspected": len(ordered),
        "model_visible_frames": len(selected),
        "extra_frames_inspected": max(0, len(ordered) - len(selected)),
        "selected_frame_indices": [row["frame_index"] for row in selected],
        "selected_timestamps": [row["timestamp"] for row in selected],
        "selected": selected,
        "transition_pairs": sum(
            row["allocation"] == "transition_context" for row in selected
        ),
        "future_frame_violations": 0,
    }


__all__ = [
    "FrameSignal",
    "POLICY_ID",
    "TRANSITION_PAIR_POLICY_ID",
    "TemporalBudgetConfig",
    "TemporalBudgetError",
    "build_frame_signals",
    "select_temporal_budget",
    "select_transition_pair_budget",
    "trajectory_change_scores",
]
