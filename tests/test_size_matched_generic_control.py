from data_pipeline.temporal_scope.build_size_matched_generic_control import (
    select_parent_ids,
)


def _row(parent_id: str, kind: str) -> dict:
    return {"parents": [{"id": parent_id, "kind": kind}]}


def test_selection_is_deterministic_and_stratified() -> None:
    rows = [
        *[_row(f"source-{i}", "source_semantic_row") for i in range(8)],
        *[
            _row(f"cited-{i}", "cited_nonbenchmark_refresh_projection")
            for i in range(5)
        ],
    ]
    quotas = {
        "source_semantic_row": 4,
        "cited_nonbenchmark_refresh_projection": 2,
    }
    first, counts = select_parent_ids(rows, seed="control-v1", quotas=quotas)
    second, _ = select_parent_ids(list(reversed(rows)), seed="control-v1", quotas=quotas)
    assert first == second
    assert len(first) == 6
    assert counts == quotas
