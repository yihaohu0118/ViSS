from experiments.analyze_uncertainty_temporal_escalation import (
    calibration_bucket,
    choice_margin,
    fit_acquire_commit_thresholds,
    fit_threshold,
    select_acquire_commit_prediction,
    select_prediction,
)


def row(uid: str, prediction: str, margin: float) -> dict:
    return {
        "uid": uid,
        "prediction": prediction,
        "choice_confidence": {
            "top2_margin": margin,
            "task_or_group_used": False,
        },
    }


def test_policy_escalates_only_from_global_margin() -> None:
    recent = row("a", "A", 0.1)
    temporal = row("a", "B", 0.9)
    assert select_prediction(recent, temporal, 0.2) == ("B", True)
    assert select_prediction(recent, temporal, 0.05) == ("A", False)


def test_fit_threshold_prefers_accuracy_then_fewer_escalations() -> None:
    recent = {
        "a": row("a", "A", 0.1),
        "b": row("b", "B", 0.8),
    }
    temporal = {
        "a": row("a", "B", 0.9),
        "b": row("b", "A", 0.9),
    }
    recent_scores = {
        "a": {"correct": False},
        "b": {"correct": True},
    }
    temporal_scores = {
        "a": {"correct": True},
        "b": {"correct": False},
    }
    threshold, sweep = fit_threshold(
        ["a", "b"],
        recent=recent,
        temporal=temporal,
        recent_scores=recent_scores,
        temporal_scores=temporal_scores,
    )
    assert threshold == 0.1
    assert max(item["correct"] for item in sweep) == 2


def test_calibration_split_is_deterministic() -> None:
    assert calibration_bucket("stable-uid", 7) == calibration_bucket(
        "stable-uid", 7
    )


def test_choice_margin_rejects_non_blind_provenance() -> None:
    bad = row("a", "A", 0.1)
    bad["choice_confidence"]["task_or_group_used"] = True
    try:
        choice_margin(bad)
    except ValueError as error:
        assert "not task blind" in str(error)
    else:
        raise AssertionError("expected task-blind provenance failure")


def test_full_vocabulary_generation_margin_takes_precedence() -> None:
    value = row("a", "A", 0.9)
    value["generation_confidence"] = {
        "top2_margin": 0.2,
        "task_or_group_used": False,
    }
    assert choice_margin(value) == 0.2


def test_safe_commit_keeps_current_answer_when_proposal_is_less_confident() -> None:
    recent = row("a", "A", 0.2)
    temporal = row("a", "B", 0.1)
    assert select_acquire_commit_prediction(recent, temporal, 0.3, 0.0) == (
        "A",
        True,
        False,
        -0.1,
    )


def test_joint_fit_can_acquire_but_reject_harmful_changed_answer() -> None:
    recent = {
        "a": row("a", "A", 0.1),
        "b": row("b", "B", 0.2),
    }
    temporal = {
        "a": row("a", "C", 0.8),
        "b": row("b", "D", 0.1),
    }
    recent_scores = {"a": {"correct": False}, "b": {"correct": True}}
    temporal_scores = {"a": {"correct": True}, "b": {"correct": False}}
    acquire, commit, sweep = fit_acquire_commit_thresholds(
        ["a", "b"],
        recent=recent,
        temporal=temporal,
        recent_scores=recent_scores,
        temporal_scores=temporal_scores,
    )
    assert acquire >= 0.1
    assert -0.1 <= commit < 0.7
    assert max(item["correct"] for item in sweep) == 2
