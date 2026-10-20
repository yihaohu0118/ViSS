from copy import deepcopy

import pytest

from data_pipeline.temporal_scope.filter_semantic_release_by_blind_audit import (
    BlindAuditFilterError,
)
from data_pipeline.temporal_scope.filter_semantic_release_by_option_consensus import (
    _validate_presentations,
)


def _source(uid: str, answer: str) -> dict:
    return {
        "uid": uid,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {
                "role": "assistant",
                "loss": True,
                "content": [{"type": "text", "text": answer}],
            },
        ],
    }


def _audit(uid: str, presentation: int, answer: str, *, confidence: str = "high") -> dict:
    mapping = list("ABCD") if presentation == 0 else list("BCDA")
    return {
        "format": "streameva.semantic_release.option_consensus_audit_row.v0",
        "uid": uid,
        "presentation": presentation,
        "displayed_to_source": mapping,
        "expected_source": "B",
        "prediction_source": answer,
        "confidence": confidence,
        "error": None,
        "correct_source": answer == "B",
    }


def test_option_consensus_selects_only_stable_source_agreements() -> None:
    sources = [_source("kept", "B"), _source("wrong", "B"), _source("unstable", "B")]
    audits = [
        _audit("kept", 0, "B"),
        _audit("kept", 1, "B"),
        _audit("wrong", 0, "C"),
        _audit("wrong", 1, "C"),
        _audit("unstable", 0, "B"),
        _audit("unstable", 1, "C"),
    ]
    selected, stable = _validate_presentations(
        source_rows=sources, audit_rows=audits, allowed_confidence={"high"}
    )
    assert [row["uid"] for row in selected] == ["kept"]
    assert stable == 2


def test_option_consensus_rejects_missing_or_identity_second_order() -> None:
    source = _source("u", "B")
    first = _audit("u", 0, "B")
    with pytest.raises(BlindAuditFilterError, match="complete two-order"):
        _validate_presentations(
            source_rows=[source], audit_rows=[first], allowed_confidence={"high"}
        )
    second = _audit("u", 1, "B")
    second["displayed_to_source"] = list("ABCD")
    with pytest.raises(BlindAuditFilterError, match="malformed"):
        _validate_presentations(
            source_rows=[source],
            audit_rows=[first, second],
            allowed_confidence={"high"},
        )


def test_option_consensus_rejects_tampered_correctness_flag() -> None:
    source = _source("u", "B")
    rows = [_audit("u", 0, "B"), _audit("u", 1, "B")]
    tampered = deepcopy(rows)
    tampered[1]["correct_source"] = False
    with pytest.raises(BlindAuditFilterError, match="malformed"):
        _validate_presentations(
            source_rows=[source], audit_rows=tampered, allowed_confidence={"high"}
        )
