from experiments.materialize_third_view_disagreement_agent import materialize


IDENTITY = {
    "benchmark": "demo",
    "model_manifest_sha256": "m",
    "annotation_sha256": "a",
    "evaluation_protocol_sha256": "p",
    "annotation": "annotation.json",
    "evaluation_protocol": {},
    "model": "model",
    "processor": "processor",
}


def _result(rows: list[dict]) -> dict:
    return {**IDENTITY, "rows": rows}


def test_third_view_commits_only_on_acquired_disagreement() -> None:
    recent = _result(
        [
            {"uid": "u0", "prediction": "A", "visual_tokens": 10},
            {"uid": "u1", "prediction": "D", "visual_tokens": 10},
        ]
    )
    event = _result(
        [
            {
                "uid": "u0",
                "prediction": "B",
                "error": None,
                "visual_tokens": 20,
                "sampling": {"frame_indices": list(range(20))},
            }
        ]
    )
    base = _result(
        [
            {
                "uid": "u0",
                "prediction": "A",
                "agent_policy": {"escalated": True},
                "media_calls": 2,
                "num_turns": 2,
                "model_processed_frames": 24,
                "sampling": {"frame_indices": list(range(20))},
            },
            {"uid": "u1", "prediction": "D", "media_calls": 1},
        ]
    )
    third = _result(
        [
            {
                "uid": "u0",
                "prediction": "C",
                "raw_response": "C",
                "error": None,
                "query_time": 31.0,
                "frames_observed": 32,
                "visual_tokens": 30,
                "sampling": {
                    "frame_indices": list(range(32)),
                    "frame_timestamps": list(map(float, range(32))),
                },
            }
        ]
    )
    result = materialize(recent, event, base, third)
    rows = {row["uid"]: row for row in result["rows"]}
    assert rows["u0"]["prediction"] == "C"
    assert rows["u0"]["media_calls"] == 3
    assert rows["u0"]["model_processed_frames"] == 56
    assert rows["u0"]["unique_frames"] == 32
    assert rows["u0"]["visual_tokens"] == 60
    assert rows["u1"]["prediction"] == "D"
    assert rows["u1"]["third_view_policy"]["triggered"] is False
    assert result["policy"]["question_task_group_or_benchmark_router"] is False
