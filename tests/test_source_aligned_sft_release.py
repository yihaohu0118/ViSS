from pathlib import Path

from data_pipeline.temporal_scope.build_source_aligned_sft_release import (
    SourceAlignedSFTError,
    _canonical_media_path,
    _semantic_identity,
    _teacher_used_for_labels,
)
import pytest


def _row(image: str = "media/one.jpg", question: str = "Which one?") -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "loss": False,
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "loss": True,
                "content": [{"type": "text", "text": "A"}],
            },
        ]
    }


def test_semantic_identity_binds_pixels_question_and_answer() -> None:
    inventory = {"media/one.jpg": {"sha256": "a" * 64}}
    baseline = _semantic_identity(_row(), inventory)
    assert baseline == _semantic_identity(_row(question="  Which   one? "), inventory)
    assert baseline != _semantic_identity(_row(question="Which event?"), inventory)
    changed = {"media/one.jpg": {"sha256": "b" * 64}}
    assert baseline != _semantic_identity(_row(), changed)


def test_canonical_media_path_is_content_addressed() -> None:
    digest = "ab" + "1" * 62
    assert _canonical_media_path("legacy/frame.JPG", digest) == (
        f"media/sha256/ab/{digest}.jpg"
    )


def test_teacher_label_provenance_is_strict_and_propagatable() -> None:
    assert _teacher_used_for_labels(
        {"contracts": {"teacher_used_for_labels": True}}
    )
    assert not _teacher_used_for_labels(
        {"contracts": {"teacher_used_for_labels": False}}
    )
    with pytest.raises(SourceAlignedSFTError):
        _teacher_used_for_labels({"contracts": {}})


def test_legacy_v62di_loader_defaults_narrow_but_supports_committed_full_pool() -> None:
    from data_pipeline.temporal_scope.build_source_aligned_sft_release import _load_legacy_v62di_source
    with pytest.raises(SourceAlignedSFTError, match="Private legacy adapters"):
        _load_legacy_v62di_source(Path("synthetic-unused-source"))


def test_legacy_repcount_normalization_is_hash_pinned_and_field_narrow() -> None:
    from data_pipeline.temporal_scope.build_source_aligned_sft_release import _load_legacy_repcount_source
    with pytest.raises(SourceAlignedSFTError, match="Private legacy adapters"):
        _load_legacy_repcount_source(Path("synthetic-unused-source"))


def test_pinned_legacy_inputs_count_as_sealed_sources() -> None:
    text = Path(
        "data_pipeline/temporal_scope/build_source_aligned_sft_release.py"
    ).read_text()
    assert "sealed_source_count" in text
    assert 'getattr(args, "legacy_v62di_source", None)' in text
    assert 'getattr(args, "legacy_repcount_source", None)' in text


def test_hash_bound_parent_selection_allows_one_source_filtering_only() -> None:
    text = Path(
        "data_pipeline/temporal_scope/build_source_aligned_sft_release.py"
    ).read_text()
    assert 'getattr(args, "include_parent_selection", None) is None' in text
    assert "unless a parent selection is supplied" in text


def test_composite_source_stage_filter_is_range_and_manifest_bound() -> None:
    text = Path(
        "data_pipeline/temporal_scope/build_source_aligned_sft_release.py"
    ).read_text()
    assert "--include-source-stage-release-id" in text
    assert "requested source stages are absent" in text
    assert "source curriculum stage row count diverged" in text
    assert '"included_source_stage_release_ids"' in text


def test_capability_parent_selection_is_hash_bound_and_fail_closed() -> None:
    text = Path(
        "data_pipeline/temporal_scope/build_source_aligned_sft_release.py"
    ).read_text()
    assert "--include-parent-selection" in text
    assert "iclr_video.capability_parent_selection.v0" in text
    assert "iclr_video.capability_parent_selection.v1" in text
    assert "source_release_membership_verified" in text
    assert "selected parent IDs were absent" in text
    assert '"parent_selection": parent_selection_record' in text


def test_ordered_curriculum_is_one_run_and_preserves_source_order() -> None:
    text = Path(
        "data_pipeline/temporal_scope/build_source_aligned_sft_release.py"
    ).read_text()
    assert '"training_runs_required": 1' in text
    assert '"source_stage_then_source_row"' in text
    assert "if not ordered_curriculum" in text


def test_additive_composition_can_retain_exposures_without_curriculum_order() -> None:
    text = Path(
        "data_pipeline/temporal_scope/build_source_aligned_sft_release.py"
    ).read_text()
    assert "--retain-semantic-exposures" in text
    assert "ordered_curriculum or retain_semantic_exposures" in text
    assert '"retain_semantic_exposures": retain_semantic_exposures' in text
