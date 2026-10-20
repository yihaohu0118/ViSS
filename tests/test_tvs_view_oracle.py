import json
from pathlib import Path

from experiments.materialize_tvs_view_oracle import materialize
from experiments.tvs_correctness import is_correct, load_gold


def _artifact(predictions: dict[str, str], frames: int) -> dict:
    return {
        "benchmark": "streamingbench",
        "model_manifest_sha256": "m",
        "annotation_sha256": "a",
        "evaluation_protocol_sha256": "p",
        "rows": [
            {"uid": uid, "prediction": prediction, "frames_observed": frames}
            for uid, prediction in predictions.items()
        ],
    }


def test_strict_oracle_selects_first_correct_view_and_charges_all_calls() -> None:
    recent = _artifact({"q1": "A", "q2": "B"}, 4)
    event = _artifact({"q1": "C", "q2": "D"}, 20)
    history = _artifact({"q1": "B", "q2": "A"}, 32)
    gold = {
        "q1": {"kind": "streaming_mcq", "answer": "C", "task": "X"},
        "q2": {"kind": "streaming_mcq", "answer": "A", "task": "X"},
    }
    result = materialize(recent, event, history, gold)
    assert [row["prediction"] for row in result["rows"]] == ["C", "A"]
    assert result["mean_model_calls"] == 3.0
    assert result["mean_processed_frames"] == 56.0
    assert result["chosen_view_counts"] == {"recent": 0, "event": 1, "history": 1}


def test_official_correctness_contracts(tmp_path: Path) -> None:
    annotation = tmp_path / "ovo.json"
    annotation.write_text(json.dumps([
        {"id": 1, "task": "EPM", "gt": 2},
        {"id": 2, "task": "REC", "test_info": [{"count": 3}]},
        {"id": 3, "task": "SSR", "test_info": [{"type": 0}, {"type": 1}]},
    ]))
    gold = load_gold("ovo", annotation)
    assert is_correct("C", gold["ovo_1"])
    assert is_correct("count = 3", gold["ovo_2_0"])
    assert is_correct("N", gold["ovo_3_0"])
    assert is_correct("Yes", gold["ovo_3_1"])
