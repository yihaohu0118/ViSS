from experiments.materialize_confidence_gated_third_view_agent import materialize


def _confidence(margin):
    return {"top2_margin": margin, "task_or_group_used": False}


def _row(uid, prediction, margin, *, escalated=True):
    return {
        "uid": uid,
        "prediction": prediction,
        "generation_confidence": _confidence(margin),
        "sampling": {"frame_indices": [1], "frame_timestamps": [0.0]},
        "query_time": 1.0,
        "frames_observed": 1,
        "agent_policy": {"escalated": escalated},
    }


def _value(rows):
    return {
        "benchmark": "streamingbench",
        "model_manifest_sha256": "model",
        "annotation_sha256": "annotation",
        "evaluation_protocol_sha256": "protocol",
        "rows": rows,
    }


def _threshold(commit=-0.1):
    return {
        "status": "source_disjoint_internal_threshold_calibration",
        "eligible_as_frozen_benchmark_controller_config": True,
        "model_manifest_sha256": "model",
        "policy": {"commit_threshold": commit},
    }


def test_third_view_uses_current_answer_source_margin() -> None:
    recent = _value([_row("x", "A", 0.8), _row("y", "A", 0.2)])
    event = _value([_row("x", "B", 0.5), _row("y", "B", 0.8)])
    base = _value([_row("x", "A", 0.5), _row("y", "B", 0.8)])
    third = _value([_row("x", "C", 0.4), _row("y", "C", 0.6)])
    result = materialize(recent, event, base, third, _threshold(-0.3))
    rows = {row["uid"]: row for row in result["rows"]}
    assert rows["x"]["prediction"] == "A"
    assert rows["x"]["third_view_policy"]["current_view"] == "recent4"
    assert rows["x"]["third_view_policy"]["committed"] is False
    assert rows["y"]["prediction"] == "C"
    assert rows["y"]["third_view_policy"]["current_view"] == "causalbudget20"
    assert rows["y"]["third_view_policy"]["committed"] is True
    assert result["third_view_committed"] == 1


def test_unchanged_third_answer_never_counts_as_commit() -> None:
    recent = _value([_row("x", "A", 0.2)])
    event = _value([_row("x", "B", 0.3)])
    base = _value([_row("x", "B", 0.3)])
    third = _value([_row("x", "B", 0.9)])
    result = materialize(recent, event, base, third, _threshold())
    assert result["rows"][0]["prediction"] == "B"
    assert result["rows"][0]["third_view_policy"]["committed"] is False


def test_full_precomputed_third_view_only_triggers_on_event_disagreement() -> None:
    recent = _value([_row("x", "A", 0.2), _row("z", "A", 0.4)])
    event = _value([_row("x", "B", 0.3), _row("z", "A", 0.5)])
    base = _value([_row("x", "B", 0.3), _row("z", "A", 0.5)])
    third = _value([_row("x", "C", 0.9), _row("z", "C", 0.9)])
    result = materialize(recent, event, base, third, _threshold())
    rows = {row["uid"]: row for row in result["rows"]}
    assert rows["x"]["prediction"] == "C"
    assert rows["z"]["prediction"] == "A"
    assert rows["z"]["third_view_policy"]["triggered"] is False
    assert result["third_view_triggered"] == 1
    assert result["third_view_input_rows"] == 2
    assert result["third_view_unused_precomputed_rows"] == 1
