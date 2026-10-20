import concurrent.futures
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.evaluate_qwen3vl_recent4_full import (
    CausalTemporalBudgetCache,
    DecordReaderPool,
    RecentWindowCache,
    _qwen_video_resize_limits,
    frame_policy_sampling_fps,
    resolve_frame_policy,
    clamp_visible_until_to_source,
    normalize_answer,
    probe_source_video,
    recent_window_bounds,
    render_question,
    render_semantic_revisit_prompt,
    selected_uids,
    build_visual_messages,
    compose_temporal_storyboard,
    downscale_temporal_history,
    parse_semantic_revisit_positions,
    native_video_metadata_payload,
    partition_hybrid_frames,
    render_anonymous_candidate_prompt,
    render_candidate_binary_verification_prompt,
    render_candidate_rejection_verification_prompt,
    render_candidate_pairwise_verification_prompt,
    render_candidate_revisit_prompt,
    choose_binary_verified_candidate,
    render_temporal_review_prompt,
    render_temporal_deliberation_prompt,
    render_candidate_evidence_deliberation_prompt,
    decode_causal_temporal_budget,
    decode_causal_hybrid_prefix,
    load_causalbudget_index_plan,
    load_anonymous_candidate_proposals,
    uniform_prefix_indices,
)


class Recent4RunnerTest(unittest.TestCase):
    def test_anonymous_candidate_sources_may_be_sparse(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            path.write_text(
                json.dumps(
                    {
                        "benchmark": "streamingbench",
                        "annotation_sha256": "annotation",
                        "model_manifest_sha256": "model",
                        "num_errors": 0,
                        "rows": [{"uid": "q0", "prediction": "A"}],
                    }
                )
            )
            proposals, inputs = load_anonymous_candidate_proposals(
                paths=[path],
                wanted=["q0", "q1"],
                benchmark="streamingbench",
                annotation_sha256="annotation",
                model_manifest_sha256="model",
            )
        self.assertEqual(proposals, [{"q0": "A"}])
        self.assertEqual(inputs[0]["covered_rows"], 1)
        self.assertEqual(inputs[0]["requested_rows"], 2)
        self.assertTrue(inputs[0]["sparse_candidate_source"])

    def test_semantic_revisit_parser_is_bounded_and_fills_deterministically(self):
        self.assertEqual(
            parse_semantic_revisit_positions(
                "Frames: 7, 2, 7, 99",
                history_count=8,
                revisit_count=4,
                fallback_positions=[8, 6, 4, 2],
            ),
            [2, 6, 7, 8],
        )

    def test_semantic_revisit_instruction_overrides_answer_format(self):
        prompt = render_semantic_revisit_prompt(
            "Question\nOnly give the best option's letter directly.",
            history_count=24,
            revisit_count=8,
        )
        self.assertTrue(prompt.endswith("`Frames: i,j,...` and nothing else."))

    def test_semantic_revisit_storyboard_instruction_has_global_labels(self):
        prompt = render_semantic_revisit_prompt(
            "Question",
            history_count=24,
            revisit_count=8,
            planner_mode="storyboard16",
        )
        self.assertIn("global frame number", prompt)
        self.assertIn("numbered 1 through 24", prompt)

    def test_storyboard_preserves_chronology_and_fixed_geometry(self) -> None:
        from PIL import Image

        frames = [Image.new("RGB", (32, 16), color) for color in ("red", "green", "blue")]
        sheet = compose_temporal_storyboard(
            frames, [0.0, 1.0, 2.0], columns=2, tile_width=40, tile_height=20
        )
        self.assertEqual(sheet.size, (80, 80))
        self.assertEqual(sheet.getpixel((20, 10)), (255, 0, 0))
        self.assertEqual(sheet.getpixel((60, 10)), (0, 128, 0))
        self.assertEqual(sheet.getpixel((20, 50)), (0, 0, 255))

        offset_sheet = compose_temporal_storyboard(
            frames[:2], [3.0, 4.0], columns=2, label_start=17
        )
        self.assertEqual(offset_sheet.size[0], 448)

    def test_lowres_history_keeps_recent_frames_exact(self) -> None:
        from PIL import Image

        frames = [Image.new("RGB", (640, 360), color) for color in ("red", "blue")]
        output = downscale_temporal_history(frames, [3, 4], {4}, max_edge=224)
        self.assertEqual(output[0].size, (224, 126))
        self.assertIs(output[1], frames[1])
        self.assertEqual(frames[0].size, (640, 360))

    def test_deliberation_replaces_conflicting_direct_only_instruction(self) -> None:
        prompt = render_temporal_deliberation_prompt(
            "Question\nOnly give the best option's letter (A, B, C, or D) directly."
        )
        self.assertNotIn("Only give", prompt)
        self.assertIn("Analyze the visible causal sequence", prompt)
        self.assertIn("Answer: X", prompt)

    def test_anonymous_candidates_can_be_verified_with_one_deliberation_contract(
        self,
    ) -> None:
        prompt, ordered = render_anonymous_candidate_prompt(
            "Question\nOnly give the best option's letter directly.",
            predictions=["A", "C", "A"],
            uid="q0",
        )
        prompt = render_temporal_deliberation_prompt(prompt)
        self.assertEqual(sorted(ordered), ["A", "C"])
        self.assertIn("Candidate 1", prompt)
        self.assertIn("Answer: X", prompt)

    def test_candidate_evidence_deliberation_requires_pixel_cited_elimination(
        self,
    ) -> None:
        prompt, ordered = render_anonymous_candidate_prompt(
            "Question\nOnly give the best option's letter directly.",
            predictions=["B", "D", "B"],
            uid="q1",
        )
        prompt = render_candidate_evidence_deliberation_prompt(prompt)
        self.assertEqual(sorted(ordered), ["B", "D"])
        self.assertNotIn("Only give", prompt)
        self.assertIn("untrusted hypothesis", prompt)
        self.assertIn("at most ten words per candidate", prompt)
        self.assertIn("do not narrate", prompt)
        self.assertIn("Answer: X", prompt)

    def test_binary_candidate_verifier_is_isolated_and_removes_direct_contract(self):
        prompt = render_candidate_binary_verification_prompt(
            "Question: Where is it?\nOptions:\nA. left\nB. right\n"
            "Only give the best option's letter directly.",
            candidate="B",
            options=["A. left", "B. right"],
        )
        self.assertNotIn("Only give", prompt)
        self.assertIn("Candidate answer under isolated verification: B. right", prompt)
        self.assertTrue(prompt.endswith("Answer exactly Yes or No."))

    def test_binary_candidate_commit_is_source_order_invariant(self):
        first = choose_binary_verified_candidate(
            uid="q0", candidate_probabilities={"A": 0.3, "C": 0.8}
        )
        second = choose_binary_verified_candidate(
            uid="q0", candidate_probabilities={"C": 0.8, "A": 0.3}
        )
        self.assertEqual(first, "C")
        self.assertEqual(first, second)

    def test_antipodal_candidate_prompt_reverses_only_the_decision_polarity(self):
        prompt = render_candidate_rejection_verification_prompt(
            "Question: Where is it?\nOptions:\nA. left\nB. right\n"
            "Only give the best option's letter directly.",
            candidate="B",
            options=["A. left", "B. right"],
        )
        self.assertNotIn("Only give", prompt)
        self.assertIn("isolated rejection check: B. right", prompt)
        self.assertIn("should be rejected as the best answer", prompt)
        self.assertTrue(prompt.endswith("Answer exactly Yes or No."))

    def test_pairwise_candidate_prompt_is_source_anonymous_and_order_explicit(self):
        prompt = render_candidate_pairwise_verification_prompt(
            "Question: Where is it?\nOptions:\nA. left\nB. right\n"
            "Only give the best option's letter directly.",
            first_candidate="B",
            second_candidate="A",
            options=["A. left", "B. right"],
        )
        self.assertNotIn("Only give", prompt)
        self.assertIn("Choice A: B. right", prompt)
        self.assertIn("Choice B: A. left", prompt)
        self.assertTrue(prompt.endswith("Answer exactly A or B."))

    def test_candidate_conditioned_revisit_is_bounded_and_answer_free(self):
        prompt = render_candidate_revisit_prompt(
            "Question: What happened?\nOptions:\nA. left\nB. right\n"
            "Only give the best option's letter directly.",
            candidate="A",
            options=["A. left", "B. right"],
            history_count=24,
            revisit_count=4,
        )
        self.assertNotIn("Only give", prompt)
        self.assertIn("Candidate answer under isolated verification: A. left", prompt)
        self.assertIn("Return exactly 4 distinct frame numbers", prompt)
        self.assertTrue(prompt.endswith("and nothing else."))

    def test_deliberative_explicit_answer_has_priority_for_open_outputs(self) -> None:
        self.assertEqual(
            normalize_answer(
                "There are 2 early events, then one more. Answer: 3",
                "Answer with a single number indicating the total count.",
                [],
            ),
            "3",
        )
        self.assertEqual(
            normalize_answer(
                "The first condition suggests no, but the final state does. Answer: Yes",
                'Please answer only with "Yes" or "No".',
                [],
            ),
            "Yes",
        )

    def test_causalbudget_index_plan_requires_question_blind_complete_coverage(
        self,
    ) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(
                json.dumps(
                    {
                        "format": "iclr_video.causalbudget20_index_plan.v0",
                        "status": "complete_query_independent_execution_cache",
                        "benchmark": "ovo",
                        "contracts": {
                            "question_text_read_for_plan": False,
                            "task_or_group_read_for_plan": False,
                            "model_prediction_read_for_plan": False,
                            "gold_answer_read": False,
                            "future_frames_read": False,
                        },
                        "rows": [{"uid": "u1", "sampling": {}}],
                    }
                )
            )
            rows, artifact = load_causalbudget_index_plan(
                path, benchmark="ovo", wanted=["u1"]
            )
            self.assertEqual(set(rows), {"u1"})
            self.assertEqual(artifact["path"], str(path.resolve()))

            value = json.loads(path.read_text())
            value.update(
                {
                    "format": "iclr_video.causalbudget_index_plan.v1",
                    "model_frames": 32,
                    "recent_frames": 8,
                }
            )
            path.write_text(json.dumps(value))
            rows, _ = load_causalbudget_index_plan(
                path,
                benchmark="ovo",
                wanted=["u1"],
                model_frames=32,
                recent_frames=8,
            )
            self.assertEqual(set(rows), {"u1"})

    def test_temporal_budget_cache_reuses_exact_query_independent_prefix(self) -> None:
        decoder_path = (
            "experiments.evaluate_qwen3vl_recent4_full."
            "decode_causal_temporal_budget"
        )
        with patch(
            decoder_path,
            return_value=([{"pixels": [1]}], {"frame_indices": [3]}),
        ) as decoder:
            cache = CausalTemporalBudgetCache(2)
            frames, sampling, hit = cache.get(Path("v.mp4"), visible_until=4.0)
            self.assertFalse(hit)
            frames[0]["pixels"] = [2]
            sampling["frame_indices"].append(4)
            frames2, sampling2, hit2 = cache.get(
                Path("v.mp4"), visible_until=4.0
            )
            self.assertTrue(hit2)
            self.assertEqual(frames2, [{"pixels": [1]}])
            self.assertEqual(sampling2, {"frame_indices": [3]})
            cache.get(Path("v.mp4"), visible_until=5.0)
            self.assertEqual(decoder.call_count, 2)

    def test_native_video_turn_has_one_video_placeholder_and_no_route(self) -> None:
        frames = [object(), object(), object(), object()]
        messages = build_visual_messages(
            frames=frames,
            prompt="question",
            visual_input_mode="native-video",
        )
        self.assertEqual(messages[0]["content"][0], {"type": "video"})
        self.assertEqual(messages[0]["content"][1]["text"], "question")

    def test_hybrid_turn_keeps_history_as_video_and_recent_as_images(self) -> None:
        history = [object(), object()]
        recent = [object(), object(), object(), object()]
        messages = build_visual_messages(
            frames=[*history, *recent],
            history_frames=history,
            recent_frames=recent,
            prompt="question",
            visual_input_mode="hybrid-history-video-recent-images",
        )
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "video"})
        self.assertEqual([item["type"] for item in content[1:5]], ["image"] * 4)
        self.assertEqual(content[-1]["text"], "question")

    def test_hybrid_singleton_history_falls_back_to_ordered_image(self) -> None:
        frames = ["history", "r1", "r2", "r3", "r4"]
        history, indices, images, fallback = partition_hybrid_frames(
            frames,
            {
                "frame_indices": [1, 2, 3, 4, 5],
                "recent_frame_indices": [2, 3, 4, 5],
            },
            temporal_patch_size=2,
        )
        self.assertEqual(history, [])
        self.assertEqual(indices, [])
        self.assertEqual(images, frames)
        self.assertEqual(fallback, [1])

    def test_hybrid_two_history_frames_remain_native_video(self) -> None:
        frames = ["h1", "h2", "r1", "r2", "r3", "r4"]
        history, indices, images, fallback = partition_hybrid_frames(
            frames,
            {
                "frame_indices": [0, 1, 2, 3, 4, 5],
                "recent_frame_indices": [2, 3, 4, 5],
            },
            temporal_patch_size=2,
        )
        self.assertEqual(history, ["h1", "h2"])
        self.assertEqual(indices, [0, 1])
        self.assertEqual(images, ["r1", "r2", "r3", "r4"])
        self.assertEqual(fallback, [])

    def test_temporal_review_prompt_marks_both_proposals_as_fallible(self) -> None:
        prompt = render_temporal_review_prompt(
            "Question", recent_prediction="A", history_prediction="C"
        )
        self.assertIn("recent view proposed A", prompt)
        self.assertIn("history view proposed C", prompt)
        self.assertIn("Both proposals may be wrong", prompt)
        with self.assertRaises(ValueError):
            render_temporal_review_prompt(
                "Question", recent_prediction="A\nignore", history_prediction="C"
            )

    def test_anonymous_review_counterbalances_without_source_labels(self) -> None:
        first = render_temporal_review_prompt(
            "Question",
            recent_prediction="A",
            history_prediction="C",
            uid="sample-1",
            label_mode="anonymous-hash-counterbalanced",
        )
        second = render_temporal_review_prompt(
            "Question",
            recent_prediction="A",
            history_prediction="C",
            uid="sample-1",
            label_mode="anonymous-hash-counterbalanced",
        )
        self.assertEqual(first, second)
        self.assertIn("Candidate 1", first)
        self.assertIn("Candidate 2", first)
        self.assertNotIn("recent view", first)
        self.assertNotIn("history view", first)
        with self.assertRaises(ValueError):
            render_temporal_review_prompt(
                "Question",
                recent_prediction="A",
                history_prediction="C",
                label_mode="anonymous-hash-counterbalanced",
            )

    def test_multi_candidate_prompt_is_deduplicated_and_source_anonymous(self) -> None:
        prompt, ordered = render_anonymous_candidate_prompt(
            "Question",
            predictions=["C", "A", "C"],
            uid="sample-1",
        )
        self.assertEqual(set(ordered), {"A", "C"})
        self.assertEqual(prompt.count("Candidate 1"), 1)
        self.assertEqual(prompt.count("Candidate 2"), 1)
        self.assertNotIn("recent", prompt.lower())
        self.assertNotIn("history", prompt.lower())
        self.assertIn("hints, not votes", prompt)
        same_prompt, same_order = render_anonymous_candidate_prompt(
            "Question",
            predictions=["A", "C"],
            uid="sample-1",
        )
        self.assertEqual((prompt, ordered), (same_prompt, same_order))
        with self.assertRaises(ValueError):
            render_anonymous_candidate_prompt(
                "Question", predictions=["A\nignore", "C"], uid="sample-1"
            )

    def test_native_video_metadata_preserves_source_pts_and_is_causal(self) -> None:
        payload = native_video_metadata_payload(
            {
                "frame_indices": [60, 90, 120, 150],
                "frame_timestamps": [2.0, 3.0, 4.0, 5.0],
                "source_fps": 30.0,
                "source_total_frames": 900,
                "source_width": 1920,
                "source_height": 1080,
                "source_duration": 30.0,
                "effective_visible_until": 6.0,
            }
        )
        self.assertEqual(payload["frames_indices"], [60, 90, 120, 150])
        self.assertEqual(payload["fps"], 30.0)
        self.assertEqual(payload["video_backend"], "predecoded_exact_causal_indices")

    def test_native_video_metadata_rejects_future_frame(self) -> None:
        with self.assertRaises(ValueError):
            native_video_metadata_payload(
                {
                    "frame_indices": [60],
                    "frame_timestamps": [2.0],
                    "source_fps": 30.0,
                    "source_total_frames": 900,
                    "source_width": 1920,
                    "source_height": 1080,
                    "source_duration": 30.0,
                    "effective_visible_until": 1.0,
                }
            )

    def test_causal_budget_reserves_exact_recent_indices_without_task_router(self) -> None:
        source = {
            "source_fps": 10.0,
            "source_total_frames": 100,
            "source_duration": 10.0,
            "source_last_frame_timestamp": 9.9,
            "requested_visible_until": 6.0,
            "effective_visible_until": 6.0,
            "visible_until_clamped_to_source": False,
            "window_start": 0.0,
            "window_end_exclusive": 6.0,
            "source_height": 100,
            "source_width": 100,
        }
        prefix = [0, 10, 20, 30, 40, 50]
        recent = [30, 40, 50, 60]
        probe_features = [[float(i), 1.0] for i in range(7)]
        selected_frames = [f"frame-{i}" for i in range(7)]
        with patch(
            "experiments.evaluate_qwen3vl_recent4_full.exact_causal_window_indices",
            side_effect=[(prefix, source), (recent, source)],
        ), patch(
            "experiments.evaluate_qwen3vl_recent4_full.decode_exact_indices_ffmpeg",
            side_effect=[probe_features, selected_frames],
        ):
            frames, sampling = decode_causal_temporal_budget(
                Path("video.mp4"), visible_until=6.0
            )
        self.assertEqual(frames, selected_frames)
        self.assertTrue({30, 40, 50, 60}.issubset(sampling["frame_indices"]))
        self.assertFalse(
            sampling["controller"]["benchmark_task_or_group_router"]
        )
        self.assertEqual(
            sampling["controller"]["policy_config"]["query_weight"], 0.0
        )
        self.assertEqual(sampling["controller"]["future_frame_violations"], 0)

    def test_uniform_history_has_unique_indices_and_no_padding(self) -> None:
        self.assertEqual(uniform_prefix_indices(-1, 28), [])
        self.assertEqual(uniform_prefix_indices(2, 28), [0, 1, 2])
        values = uniform_prefix_indices(99, 28)
        self.assertEqual(len(values), 28)
        self.assertEqual(values, sorted(set(values)))
        self.assertEqual((values[0], values[-1]), (0, 99))

    def test_hybrid32_preserves_recent4_as_separate_exact_indices(self) -> None:
        source = {
            "source_fps": 10.0,
            "source_total_frames": 100,
            "source_duration": 10.0,
            "source_last_frame_timestamp": 9.9,
            "requested_visible_until": 6.0,
            "effective_visible_until": 6.0,
            "visible_until_clamped_to_source": False,
            "window_start": 2.0,
            "window_end_exclusive": 6.0,
            "source_height": 100,
            "source_width": 100,
        }
        recent = [30, 40, 50, 60]
        with patch(
            "experiments.evaluate_qwen3vl_recent4_full.exact_causal_window_indices",
            return_value=(recent, source),
        ), patch(
            "experiments.evaluate_qwen3vl_recent4_full.decode_exact_indices_ffmpeg",
            side_effect=lambda _path, indices: [f"frame-{value}" for value in indices],
        ):
            frames, sampling = decode_causal_hybrid_prefix(
                Path("video.mp4"), visible_until=6.0
            )
        self.assertEqual(sampling["recent_frame_indices"], recent)
        self.assertEqual(len(sampling["history_frame_indices"]), 28)
        self.assertEqual(len(sampling["frame_indices"]), 32)
        self.assertEqual(sampling["frame_indices"], sorted(set(sampling["frame_indices"])))
        self.assertEqual(len(frames), 32)
        self.assertFalse(sampling["controller"]["benchmark_task_or_group_router"])
        self.assertFalse(sampling["controller"]["question_keyword_router"])

    def test_qwen_video_resize_limits_supports_pixel_api(self) -> None:
        module = SimpleNamespace(
            IMAGE_FACTOR=28,
            VIDEO_MIN_PIXELS=100_352,
            VIDEO_MAX_PIXELS=602_112,
            VIDEO_TOTAL_PIXELS=90_316_800,
            FRAME_FACTOR=2,
        )
        self.assertEqual(
            _qwen_video_resize_limits(module, 32),
            (28, 100_352, 602_112),
        )

    def test_qwen_video_resize_limits_supports_token_api(self) -> None:
        module = SimpleNamespace(
            SPATIAL_MERGE_SIZE=2,
            VIDEO_MIN_TOKEN_NUM=128,
            VIDEO_MAX_TOKEN_NUM=768,
            MODEL_SEQ_LEN=128_000,
            FRAME_FACTOR=2,
        )
        factor, minimum, maximum = _qwen_video_resize_limits(module, 32)
        self.assertEqual((factor, minimum), (28, 100_352))
        self.assertEqual(maximum, 602_112)

    def test_training_free_active4_policies_only_change_official_sampling_fps(self) -> None:
        self.assertEqual(frame_policy_sampling_fps("recent4"), 1.0)
        self.assertEqual(frame_policy_sampling_fps("wide8"), 0.5)
        self.assertAlmostEqual(frame_policy_sampling_fps("wide12"), 1.0 / 3.0)
        self.assertEqual(frame_policy_sampling_fps("wide16"), 0.25)

    def test_modelonly_frame_policy_is_question_independent(self) -> None:
        self.assertEqual(resolve_frame_policy("recent4"), "recent4")
        self.assertEqual(resolve_frame_policy("wide12"), "wide12")
        self.assertEqual(resolve_frame_policy("prefix8"), "prefix8")
        self.assertEqual(resolve_frame_policy("prefix32"), "prefix32")
        self.assertEqual(resolve_frame_policy("causalbudget20"), "causalbudget20")
        self.assertEqual(
            resolve_frame_policy("causalbudget32r8"), "causalbudget32r8"
        )
        self.assertEqual(resolve_frame_policy("causalhybrid32"), "causalhybrid32")
        self.assertEqual(
            resolve_frame_policy("causalhybrid32r8"), "causalhybrid32r8"
        )
        with self.assertRaises(ValueError):
            resolve_frame_policy("adaptive_query_budget32")

    def test_source_probe_has_small_bounded_cache(self) -> None:
        self.assertEqual(probe_source_video.cache_parameters()["maxsize"], 8)

    def test_decord_reader_pool_bounds_native_decoder_threads(self) -> None:
        calls = []

        class Reader:
            pass

        def video_reader(path, *, num_threads):
            calls.append((path, num_threads))
            return Reader()

        pool = DecordReaderPool(2, num_threads=3)
        with patch.dict(
            "sys.modules", {"decord": SimpleNamespace(VideoReader=video_reader)}
        ):
            first, first_hit = pool.reader("a.mp4")
            second, second_hit = pool.reader("a.mp4")

        self.assertIs(first, second)
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(calls, [("a.mp4", 3)])

    def test_decord_reader_pool_serializes_same_video_prefetch(self) -> None:
        state_lock = __import__("threading").Lock()
        active = 0
        peak = 0
        instances = 0

        class Frame:
            shape = (16, 32, 3)

        class Reader:
            def get_avg_fps(self):
                nonlocal active, peak
                with state_lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.01)
                with state_lock:
                    active -= 1
                return 30.0

            def __len__(self):
                return 10

            def __getitem__(self, index):
                return Frame()

        def video_reader(path, *, num_threads):
            nonlocal instances
            instances += 1
            return Reader()

        pool = DecordReaderPool(2, num_threads=3)
        with patch.dict(
            "sys.modules", {"decord": SimpleNamespace(VideoReader=video_reader)}
        ):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                values = list(executor.map(pool.metadata, ["a.mp4"] * 8))

        self.assertEqual(values, [(30.0, 10, 16, 32)] * 8)
        self.assertEqual(instances, 1)
        self.assertEqual(peak, 1)

    def test_exact_decode_cache_reuses_only_matching_sampling_request(self) -> None:
        from pathlib import Path

        class Frame:
            def __init__(self, value: int) -> None:
                self.value = value

            def copy(self):
                return Frame(self.value)

        calls = []

        def decoder(path, **kwargs):
            calls.append((path, kwargs))
            return [Frame(len(calls))], {"frame_indices": [len(calls)]}

        cache = RecentWindowCache(2, decoder=decoder)
        first_frames, first_sampling, first_hit = cache.get(
            Path("video.mp4"),
            visible_until=10.0,
            recent_frames=4,
            sampling_fps=1.0,
        )
        first_frames[0].value = 99
        first_sampling["frame_indices"][0] = 99
        second_frames, second_sampling, second_hit = cache.get(
            Path("video.mp4"),
            visible_until=10.0,
            recent_frames=4,
            sampling_fps=1.0,
        )
        _, _, third_hit = cache.get(
            Path("video.mp4"),
            visible_until=11.0,
            recent_frames=4,
            sampling_fps=1.0,
        )

        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertFalse(third_hit)
        self.assertEqual(len(calls), 2)
        self.assertEqual(second_frames[0].value, 1)
        self.assertEqual(second_sampling["frame_indices"], [1])
        self.assertEqual(second_sampling["frame_policy"], "recent4")
        self.assertEqual(second_sampling["policy_sampling_fps"], 1.0)
        self.assertEqual((cache.hits, cache.misses), (1, 2))

    def test_prefix_policy_uses_dedicated_sequential_decoder(self) -> None:
        cache = RecentWindowCache(0)
        decoded = ([object()], {"frame_indices": [0]})
        with patch(
            "experiments.evaluate_qwen3vl_recent4_full.decode_prefix_window_ffmpeg",
            return_value=decoded,
        ) as decoder:
            frames, sampling, hit = cache.get(
                Path("video.mp4"),
                visible_until=10.0,
                recent_frames=4,
                sampling_fps=1.0,
                frame_policy="prefix32",
            )

        self.assertIs(frames, decoded[0])
        self.assertFalse(hit)
        self.assertEqual(sampling["frame_policy"], "prefix32")
        decoder.assert_called_once_with(
            Path("video.mp4"),
            visible_until=10.0,
            recent_frames=32,
            sampling_fps=3.2,
        )

    def test_exact_decode_cache_is_safe_for_parallel_prefix_misses(self) -> None:
        class Frame:
            def __init__(self, value: str) -> None:
                self.value = value

            def copy(self):
                return Frame(self.value)

        def decoder(path, **kwargs):
            time.sleep(0.01)
            return [Frame(str(path))], {"frame_indices": [str(path)]}

        cache = RecentWindowCache(8, decoder=decoder)
        paths = [Path(f"video-{index}.mp4") for index in range(8)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            values = list(
                executor.map(
                    lambda path: cache.get(
                        path,
                        visible_until=10.0,
                        recent_frames=32,
                        sampling_fps=3.2,
                        frame_policy="prefix32",
                    ),
                    paths,
                )
            )
        self.assertEqual([value[0][0].value for value in values], list(map(str, paths)))
        self.assertEqual(cache.misses, 8)

    def test_exact_decode_cache_singleflights_matching_parallel_requests(self) -> None:
        class Frame:
            def copy(self):
                return Frame()

        calls = []

        def decoder(path, **kwargs):
            calls.append((path, kwargs))
            time.sleep(0.05)
            return [Frame()], {"frame_indices": [1]}

        cache = RecentWindowCache(8, decoder=decoder)

        def request(_):
            return cache.get(
                Path("same-video.mp4"),
                visible_until=10.0,
                recent_frames=4,
                sampling_fps=1.0,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            values = list(executor.map(request, range(16)))

        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 15)
        self.assertGreaterEqual(cache.singleflight_hits, 1)
        self.assertEqual(sum(bool(value[2]) for value in values), 15)

    def test_visible_prefix_is_clamped_to_last_real_frame(self) -> None:
        effective, clamped = clamp_visible_until_to_source(10.0, 2.0, 9)
        self.assertEqual(effective, 4.0)
        self.assertTrue(clamped)
        effective, clamped = clamp_visible_until_to_source(3.0, 2.0, 9)
        self.assertEqual(effective, 3.0)
        self.assertFalse(clamped)

    def test_explicit_uid_subset_keeps_export_order(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text(
                '{"uid":"a"}\n{"uid":"b"}\n{"uid":"c"}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                selected_uids(path, 0, -1, ["c", "a"]),
                ["a", "c"],
            )

    def test_batched_runner_keeps_auditable_resume_controls(self) -> None:
        source = Path("experiments/evaluate_qwen3vl_recent4_full.py").read_text(
            encoding="utf-8"
        )
        launcher = Path("scripts/run_recent4_single_benchmark.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('processor.tokenizer.padding_side = "left"', source)
        self.assertIn("validate_processor_contract(processor_path, protocol)", source)
        self.assertIn('"inference_batch_size": args.batch_size', source)
        self.assertIn('"execution_optimization",', source)
        self.assertIn('--uids-file "$uid_file"', launcher)
        self.assertIn('uids[index::count]', launcher)
        self.assertIn('--batch-size "$attempt_batch"', launcher)
        self.assertIn('attempt_batch=$EVAL_BATCH_SIZE', launcher)
        self.assertIn('|| smoke_rc=$?', launcher)
        self.assertIn('|| runner_rc=$?', launcher)
        self.assertIn(
            'after writing a complete validated result',
            launcher,
        )

    def test_recent_window_is_half_open_and_causal(self) -> None:
        self.assertEqual(recent_window_bounds(10.0, 4, 1.0), (6.0, 10.0))
        self.assertEqual(recent_window_bounds(1.0, 4, 1.0), (0.0, 1.0))

    def test_answer_normalization(self) -> None:
        options = ["A. first", "B. second", "C. third", "D. fourth"]
        self.assertEqual(normalize_answer("(C) third", "question", options), "C")
        self.assertEqual(normalize_answer("second", "question", options), "B")
        self.assertEqual(normalize_answer("A map", "question", ["A. compass", "B. A map", "C. sky", "D. radar"]), "B")
        self.assertEqual(
            normalize_answer(
                "Yes",
                "Answer only with “Yes” or “No”.",
                [],
            ),
            "Yes",
        )

    def test_streaming_prompt_matches_release_contract(self) -> None:
        prompt = render_question(
            "streamingbench",
            "What is visible?",
            ["A. first", "B. second", "C. third", "D. fourth"],
        )
        self.assertIn("advanced video question-answering AI assistant", prompt)
        self.assertTrue(prompt.endswith(
            "Only give the best option's letter (A, B, C, or D) directly."
        ))

    def test_ovo_mcq_prompt_matches_release_contract(self) -> None:
        prompt = render_question(
            "ovo",
            "What happened?",
            ["one", "two", "three", "four"],
        )
        self.assertEqual(
            prompt,
            "What happened?\nOptions: A. one; B. two; C. three; D. four;"
            "\nOnly give the best option's letter directly.",
        )

    def test_ovo_requires_full_runtime_asset_attestation(self) -> None:
        launcher = Path("scripts/run_recent4_single_benchmark.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("OVO_RUNTIME_REPORT", launcher)
        self.assertIn(".expanded_queries == 3035", launcher)
        self.assertIn(".covered_queries == 3035", launcher)
        self.assertIn(".missing_videos == 0", launcher)
        self.assertIn(".annotation_sha256 == $annotation_sha256", launcher)
        self.assertIn(".frozen_export_sha256 == $frozen_export_sha256", launcher)


if __name__ == "__main__":
    unittest.main()
