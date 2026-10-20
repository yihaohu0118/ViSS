from experiments.materialize_uncertainty_temporal_agent import materialize


def _result(rows: list[dict]) -> dict:
    return {
        "benchmark": "x",
        "model_manifest_sha256": "m",
        "annotation_sha256": "a",
        "evaluation_protocol_sha256": "p",
        "rows": rows,
    }


def test_materializer_uses_temporal_only_for_preselected_uids() -> None:
    recent = _result(
        [
            {"uid": "a", "prediction": "A", "query_time": 10, "frames_observed": 4},
            {"uid": "b", "prediction": "B", "query_time": 10, "frames_observed": 4},
        ]
    )
    temporal = _result(
        [
            {
                "uid": "a",
                "prediction": "C",
                "query_time": 10,
                "frames_observed": 20,
                "sampling": {
                    "frame_policy": "causalbudget20",
                    "frame_timestamps": [0, 10],
                    "candidate_frame_indices": list(range(64)),
                    "extra_frames_inspected": 44,
                },
                "visual_input_mode": "native-video",
            }
        ]
    )
    selection = {
        "status": "frozen_before_benchmark_temporal_inference",
        "threshold": 0.3,
        "benchmark": "x",
        "model_manifest_sha256": "m",
        "score_or_gold_fields_read_for_selection": False,
        "rows": [
            {"uid": "a", "margin": 0.2, "escalated": True},
            {"uid": "b", "margin": 0.8, "escalated": False},
        ],
    }
    threshold = {
        "status": "source_disjoint_internal_threshold_calibration",
        "eligible_as_frozen_benchmark_controller_config": True,
        "model_manifest_sha256": "m",
        "policy": {"threshold": 0.3},
    }
    output = materialize(recent, temporal, selection, threshold)
    assert [row["prediction"] for row in output["rows"]] == ["C", "B"]
    assert output["escalated"] == 1
    assert output["mean_model_calls"] == 1.5
    assert output["mean_low_resolution_probe_frames"] == 32
    assert output["mean_extra_probe_frames_not_visible_to_vlm"] == 22
    assert output["mean_model_processed_frames"] == 14
    assert output["mean_total_decoded_frame_instances"] == 46
    assert output["rows"][0]["total_decoded_frame_instances"] == 88
    assert output["rows"][1]["total_decoded_frame_instances"] == 4


def test_materializer_rejects_lower_confidence_changed_proposal() -> None:
    recent = _result(
        [
            {
                "uid": "a",
                "prediction": "A",
                "query_time": 10,
                "frames_observed": 4,
                "generation_confidence": {
                    "top2_margin": 0.3,
                    "task_or_group_used": False,
                },
            }
        ]
    )
    temporal = _result(
        [
            {
                "uid": "a",
                "prediction": "C",
                "query_time": 10,
                "frames_observed": 20,
                "generation_confidence": {
                    "top2_margin": 0.2,
                    "task_or_group_used": False,
                },
                "sampling": {
                    "frame_policy": "causalbudget20",
                    "frame_timestamps": [0, 10],
                    "candidate_frame_indices": list(range(20)),
                    "extra_frames_inspected": 0,
                },
                "visual_input_mode": "native-video",
            }
        ]
    )
    selection = {
        "status": "frozen_before_benchmark_temporal_inference",
        "threshold": 0.4,
        "benchmark": "x",
        "model_manifest_sha256": "m",
        "score_or_gold_fields_read_for_selection": False,
        "rows": [{"uid": "a", "margin": 0.3, "escalated": True}],
    }
    threshold = {
        "status": "source_disjoint_internal_threshold_calibration",
        "eligible_as_frozen_benchmark_controller_config": True,
        "model_manifest_sha256": "m",
        "policy": {"threshold": 0.4, "commit_threshold": 0.0},
    }
    output = materialize(recent, temporal, selection, threshold)
    assert output["rows"][0]["prediction"] == "A"
    assert output["rows"][0]["proposal_prediction"] == "C"
    assert output["rows"][0]["agent_policy"]["committed"] is False


def test_materializer_names_image_list_event20_interface() -> None:
    recent = _result(
        [{
            "uid": "a", "prediction": "A", "query_time": 10,
            "frames_observed": 4,
            "generation_confidence": {"top2_margin": 0.1, "task_or_group_used": False},
        }]
    )
    temporal = _result(
        [{
            "uid": "a", "prediction": "C", "query_time": 10,
            "frames_observed": 20,
            "generation_confidence": {"top2_margin": 0.4, "task_or_group_used": False},
            "sampling": {
                "frame_policy": "causalbudget20", "frame_timestamps": [0, 10],
                "candidate_frame_indices": list(range(20)), "extra_frames_inspected": 0,
            },
            "visual_input_mode": "image-list",
        }]
    )
    selection = {
        "status": "frozen_before_benchmark_temporal_inference",
        "threshold": 0.5, "benchmark": "x", "model_manifest_sha256": "m",
        "score_or_gold_fields_read_for_selection": False,
        "rows": [{"uid": "a", "margin": 0.1, "escalated": True}],
    }
    threshold = {
        "status": "source_disjoint_internal_threshold_calibration",
        "eligible_as_frozen_benchmark_controller_config": True,
        "model_manifest_sha256": "m",
        "policy": {"threshold": 0.5, "commit_threshold": 0.0},
    }
    output = materialize(recent, temporal, selection, threshold)
    assert output["policy"]["second_view"] == "image_list_causal_temporal20"
    assert output["rows"][0]["agent_policy"]["selected_view"] == (
        "causal_temporal20_image_list"
    )


def test_materializer_names_image_list_event32_recent8_interface() -> None:
    recent = _result(
        [{
            "uid": "a", "prediction": "A", "query_time": 10,
            "frames_observed": 4,
            "generation_confidence": {"top2_margin": 0.1, "task_or_group_used": False},
        }]
    )
    temporal = _result(
        [{
            "uid": "a", "prediction": "C", "query_time": 10,
            "frames_observed": 32,
            "generation_confidence": {"top2_margin": 0.4, "task_or_group_used": False},
            "sampling": {
                "frame_policy": "causalbudget32r8", "frame_timestamps": [0, 10],
                "candidate_frame_indices": list(range(64)),
                "extra_frames_inspected": 32,
            },
            "visual_input_mode": "image-list",
        }]
    )
    selection = {
        "status": "frozen_before_benchmark_temporal_inference",
        "threshold": 0.5, "benchmark": "x", "model_manifest_sha256": "m",
        "score_or_gold_fields_read_for_selection": False,
        "rows": [{"uid": "a", "margin": 0.1, "escalated": True}],
    }
    threshold = {
        "status": "source_disjoint_internal_threshold_calibration",
        "eligible_as_frozen_benchmark_controller_config": True,
        "model_manifest_sha256": "m",
        "policy": {"threshold": 0.5, "commit_threshold": 0.0},
    }
    output = materialize(recent, temporal, selection, threshold)
    assert output["policy"]["second_view"] == (
        "image_list_causal_temporal32_recent8"
    )
    assert output["rows"][0]["agent_policy"]["selected_view"] == (
        "causal_temporal32_recent8_image_list"
    )


def test_materializer_accepts_full_lowres_temporal_superset() -> None:
    recent = _result(
        [
            {"uid": "a", "prediction": "A", "query_time": 10, "frames_observed": 4},
            {"uid": "b", "prediction": "B", "query_time": 10, "frames_observed": 4},
        ]
    )
    temporal_rows = []
    for uid, prediction in (("a", "C"), ("b", "D")):
        temporal_rows.append(
            {
                "uid": uid,
                "prediction": prediction,
                "query_time": 10,
                "frames_observed": 32,
                "sampling": {
                    "frame_policy": "causalbudget32r8",
                    "frame_timestamps": [0, 10],
                    "candidate_frame_indices": list(range(64)),
                    "extra_frames_inspected": 32,
                },
                "visual_input_mode": "lowres-history-recent-images",
            }
        )
    temporal = _result(temporal_rows)
    selection = {
        "status": "frozen_before_benchmark_temporal_inference",
        "threshold": 0.3,
        "benchmark": "x",
        "model_manifest_sha256": "m",
        "score_or_gold_fields_read_for_selection": False,
        "rows": [
            {"uid": "a", "margin": 0.2, "escalated": True},
            {"uid": "b", "margin": 0.8, "escalated": False},
        ],
    }
    threshold = {
        "status": "source_disjoint_internal_threshold_calibration",
        "eligible_as_frozen_benchmark_controller_config": True,
        "model_manifest_sha256": "m",
        "policy": {"threshold": 0.3},
    }
    output = materialize(recent, temporal, selection, threshold)
    assert [row["prediction"] for row in output["rows"]] == ["C", "B"]
    assert output["temporal_input_num_rows"] == 2
    assert output["temporal_unused_num_rows"] == 1
    assert output["policy"]["second_view"] == (
        "lowres_history_recent_images_causal_temporal32_recent8"
    )
    assert output["rows"][0]["agent_policy"]["selected_view"] == (
        "causal_temporal32_recent8_lowres_history_recent_images"
    )


def test_materializer_names_uniform32_image_list_interface() -> None:
    recent = _result(
        [{
            "uid": "a", "prediction": "A", "query_time": 10,
            "frames_observed": 4,
            "generation_confidence": {"top2_margin": 0.1, "task_or_group_used": False},
        }]
    )
    temporal = _result(
        [{
            "uid": "a", "prediction": "C", "query_time": 10,
            "frames_observed": 32,
            "generation_confidence": {"top2_margin": 0.4, "task_or_group_used": False},
            "sampling": {
                "frame_policy": "causalhybrid32", "frame_timestamps": [0, 10],
            },
            "visual_input_mode": "image-list",
        }]
    )
    selection = {
        "status": "frozen_before_benchmark_temporal_inference",
        "threshold": 0.5, "benchmark": "x", "model_manifest_sha256": "m",
        "score_or_gold_fields_read_for_selection": False,
        "rows": [{"uid": "a", "margin": 0.1, "escalated": True}],
    }
    threshold = {
        "status": "source_disjoint_internal_threshold_calibration",
        "eligible_as_frozen_benchmark_controller_config": True,
        "model_manifest_sha256": "m",
        "policy": {"threshold": 0.5, "commit_threshold": 0.0},
    }
    output = materialize(recent, temporal, selection, threshold)
    assert output["policy"]["second_view"] == (
        "image_list_uniform_history28_plus_recent4"
    )
    assert output["rows"][0]["agent_policy"]["selected_view"] == (
        "causal_uniform_history28_plus_recent4_image_list"
    )


def test_materializer_names_hybrid32_recent8_interface() -> None:
    recent = _result(
        [{
            "uid": "a", "prediction": "A", "query_time": 10,
            "frames_observed": 4,
            "generation_confidence": {"top2_margin": 0.1, "task_or_group_used": False},
        }]
    )
    temporal = _result(
        [{
            "uid": "a", "prediction": "C", "query_time": 10,
            "frames_observed": 32,
            "generation_confidence": {"top2_margin": 0.4, "task_or_group_used": False},
            "sampling": {
                "frame_policy": "causalhybrid32r8", "frame_timestamps": [0, 10],
            },
            "visual_input_mode": "hybrid-history-video-recent-images",
        }]
    )
    selection = {
        "status": "frozen_before_benchmark_temporal_inference",
        "threshold": 0.5, "benchmark": "x", "model_manifest_sha256": "m",
        "score_or_gold_fields_read_for_selection": False,
        "rows": [{"uid": "a", "margin": 0.1, "escalated": True}],
    }
    threshold = {
        "status": "source_disjoint_internal_threshold_calibration",
        "eligible_as_frozen_benchmark_controller_config": True,
        "model_manifest_sha256": "m",
        "policy": {"threshold": 0.5, "commit_threshold": 0.0},
    }
    output = materialize(recent, temporal, selection, threshold)
    assert output["policy"]["second_view"] == (
        "hybrid_history_video_recent_images_causal_hybrid32r8"
    )
    assert output["rows"][0]["agent_policy"]["selected_view"] == (
        "causal_hybrid_history24_plus_recent8"
    )


def test_materializer_accepts_task_blind_fixed_budget_selection() -> None:
    recent = _result(
        [
            {
                "uid": "a",
                "prediction": "A",
                "query_time": 10,
                "frames_observed": 4,
                "generation_confidence": {
                    "top2_margin": 0.1,
                    "task_or_group_used": False,
                },
            },
            {
                "uid": "b",
                "prediction": "B",
                "query_time": 10,
                "frames_observed": 4,
            },
        ]
    )
    temporal = _result(
        [
            {
                "uid": "a",
                "prediction": "C",
                "query_time": 10,
                "frames_observed": 20,
                "generation_confidence": {
                    "top2_margin": 0.3,
                    "task_or_group_used": False,
                },
                "sampling": {
                    "frame_policy": "causalbudget20",
                    "frame_timestamps": [0, 10],
                    "candidate_frame_indices": list(range(64)),
                    "extra_frames_inspected": 44,
                },
                "visual_input_mode": "native-video",
            }
        ]
    )
    selection = {
        "format": "streameva.unlabeled_budget_uid_selection.v0",
        "status": "frozen_before_benchmark_temporal_inference",
        "benchmark": "x",
        "model_manifest_sha256": "m",
        "quantile": 0.5,
        "score_or_gold_fields_read_for_selection": False,
        "benchmark_identity_used_for_policy_parameters": False,
        "cohort_labels_read_for_selection": False,
        "rows": [
            {
                "uid": "a",
                "margin": 0.1,
                "uncertainty_rank": 1,
                "escalated": True,
            },
            {
                "uid": "b",
                "margin": 0.7,
                "uncertainty_rank": 2,
                "escalated": False,
            },
        ],
    }
    output = materialize(
        recent,
        temporal,
        selection,
        None,
        commit_threshold_override=0.0,
    )
    assert [row["prediction"] for row in output["rows"]] == ["C", "B"]
    assert output["status"] == "fixed_budget_router_free_agent"
    assert output["policy"]["uncertainty_quantile"] == 0.5
    assert output["rows"][0]["agent_policy"]["uncertainty_rank"] == 1


def test_materializer_accounts_for_prefix32_without_hidden_probe() -> None:
    recent = _result(
        [
            {
                "uid": "a",
                "prediction": "A",
                "query_time": 10,
                "frames_observed": 4,
                "generation_confidence": {
                    "top2_margin": 0.1,
                    "task_or_group_used": False,
                },
            }
        ]
    )
    temporal = _result(
        [
            {
                "uid": "a",
                "prediction": "B",
                "query_time": 10,
                "frames_observed": 32,
                "generation_confidence": {
                    "top2_margin": 0.5,
                    "task_or_group_used": False,
                },
                "sampling": {
                    "frame_policy": "prefix32",
                    "frame_timestamps": [0, 10],
                },
                "visual_input_mode": "image-list",
            }
        ]
    )
    selection = {
        "format": "streameva.unlabeled_budget_uid_selection.v0",
        "status": "frozen_before_benchmark_temporal_inference",
        "benchmark": "x",
        "model_manifest_sha256": "m",
        "quantile": 1.0,
        "score_or_gold_fields_read_for_selection": False,
        "benchmark_identity_used_for_policy_parameters": False,
        "cohort_labels_read_for_selection": False,
        "rows": [
            {"uid": "a", "margin": 0.1, "uncertainty_rank": 1, "escalated": True}
        ],
    }
    output = materialize(
        recent,
        temporal,
        selection,
        None,
        commit_threshold_override=-1.0,
    )
    row = output["rows"][0]
    assert row["prediction"] == "B"
    assert row["agent_policy"]["selected_view"] == "causal_prefix32"
    assert row["low_resolution_probe_frames"] == 0
    assert row["model_processed_frames"] == 36
    assert row["total_decoded_frame_instances"] == 36


def test_materializer_can_fail_closed_to_recent_on_temporal_error() -> None:
    recent = _result(
        [
            {
                "uid": "a",
                "prediction": "A",
                "query_time": 10,
                "frames_observed": 4,
            },
            {
                "uid": "b",
                "prediction": "B",
                "query_time": 10,
                "frames_observed": 4,
            },
        ]
    )
    temporal = _result(
        [
            {
                "uid": "a",
                "prediction": "",
                "error": "OutOfMemoryError",
                "query_time": 10,
                "frames_observed": 0,
            },
            {
                "uid": "b",
                "prediction": "C",
                "query_time": 10,
                "frames_observed": 20,
                "sampling": {
                    "frame_policy": "causalbudget20",
                    "frame_timestamps": [0, 10],
                    "candidate_frame_indices": list(range(20)),
                    "extra_frames_inspected": 0,
                },
                "visual_input_mode": "image-list",
            },
        ]
    )
    selection = {
        "status": "frozen_before_benchmark_temporal_inference",
        "threshold": 0.3,
        "benchmark": "x",
        "model_manifest_sha256": "m",
        "score_or_gold_fields_read_for_selection": False,
        "rows": [
            {"uid": "a", "margin": 0.1, "escalated": True},
            {"uid": "b", "margin": 0.2, "escalated": True},
        ],
    }
    threshold = {
        "status": "source_disjoint_internal_threshold_calibration",
        "eligible_as_frozen_benchmark_controller_config": True,
        "model_manifest_sha256": "m",
        "policy": {"threshold": 0.3},
    }
    output = materialize(
        recent,
        temporal,
        selection,
        threshold,
        fallback_recent_on_temporal_error=True,
    )
    assert [row["prediction"] for row in output["rows"]] == ["A", "C"]
    assert output["temporal_error_fallback_count"] == 1
    assert output["policy"]["temporal_error_policy"] == "fail_closed_to_recent4"
    assert output["rows"][0]["agent_policy"]["temporal_error_fallback"] is True
    assert output["rows"][0]["agent_policy"]["committed"] is False
    assert output["rows"][0]["agent_policy"]["selected_view"] == (
        "recent4_temporal_error_fallback"
    )
