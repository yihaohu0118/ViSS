#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from viss_core.benchmarks import load_prediction_file


OVO_GROUP_TASKS = {
    "backward": ("EPM", "ASI", "HLD"),
    "realtime": ("OCR", "ACR", "ATR", "STU", "FPD", "OJR"),
    "forward": ("REC", "SSR", "CRR"),
}

INVALID_EMPTY_MODEL_RESPONSE = "__INVALID_EMPTY_MODEL_RESPONSE__"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt agent predictions to the original benchmark formats and invoke their scorers."
    )
    parser.add_argument("--benchmark", required=True, choices=("streamingbench", "ovo"))
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--official-input-json", type=Path, default=None)
    args = parser.parse_args()

    if args.benchmark == "streamingbench":
        report, official_input = score_streamingbench(
            args.official_root, args.annotation, args.predictions
        )
    else:
        report, official_input = score_ovo(args.official_root, args.annotation, args.predictions)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.official_input_json:
        args.official_input_json.parent.mkdir(parents=True, exist_ok=True)
        args.official_input_json.write_text(
            json.dumps(official_input, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


def score_streamingbench(
    official_root: Path, annotation: Path, predictions_path: Path
) -> tuple[dict[str, Any], Any]:
    predictions = load_prediction_file(predictions_path)
    data = json.loads(annotation.read_text(encoding="utf-8"))
    model_key = "AgenticSFT"
    expected_ids: list[str] = []
    empty_model_response_ids: list[str] = []
    for video_index, entry in enumerate(data):
        for question_index, question in enumerate(entry.get("questions") or []):
            query_id = f"streaming_{video_index}_{question_index}"
            expected_ids.append(query_id)
            raw = predictions.get(query_id)
            letter = _mcq_letter(raw, question.get("options") or [])
            response = letter if letter is not None else str(raw or "")
            if not response:
                # The original scorer skips false-y responses instead of counting
                # them as incorrect. Preserve the model failure without imputing an
                # answer by passing an explicit invalid value through the scorer.
                empty_model_response_ids.append(query_id)
                response = INVALID_EMPTY_MODEL_RESPONSE
            question[model_key] = [response]
    _require_exact_coverage(predictions, expected_ids, "StreamingBench")

    count_script = official_root / "src" / "data" / "count.py"
    if not count_script.is_file():
        raise FileNotFoundError(count_script)
    with tempfile.TemporaryDirectory(prefix="streamingbench_official_score_") as tmp:
        tmp_path = Path(tmp)
        official_input = tmp_path / "predictions.json"
        official_input.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(count_script),
                "--model",
                model_key,
                "--task",
                "real",
                "--src",
                str(official_input),
            ],
            cwd=tmp_path,
            check=True,
            text=True,
            capture_output=True,
        )
        stats = json.loads((tmp_path / f"{model_key}_stats.json").read_text(encoding="utf-8"))

    total = stats.get("total") or {}
    by_task = {
        task: {
            "accuracy": float(values["accuracy"]),
            "correct": float(values["correct"]),
            "total": int(values["total"]),
        }
        for task, values in stats.items()
        if task != "total"
    }
    if int(total.get("total", 0)) != len(expected_ids):
        raise RuntimeError(
            f"official StreamingBench scorer counted {total.get('total')} of {len(expected_ids)} queries"
        )
    report = {
        "benchmark": "streamingbench",
        "official_scorer": str(count_script),
        "official_scorer_sha256": _sha256(count_script),
        "official_repository_commit": _git_commit(official_root),
        "official_stdout": completed.stdout.strip(),
        "num_queries": len(expected_ids),
        "num_predictions": len(expected_ids),
        "coverage": 1.0,
        "num_empty_model_responses": len(empty_model_response_ids),
        "empty_model_response_ids": empty_model_response_ids,
        "empty_response_scoring": "counted_as_incorrect_without_answer_imputation",
        "accuracy": float(total["accuracy"]),
        "correct": float(total["correct"]),
        "total": int(total["total"]),
        "by_task": by_task,
    }
    return report, data


def score_ovo(
    official_root: Path, annotation: Path, predictions_path: Path
) -> tuple[dict[str, Any], Any]:
    predictions = load_prediction_file(predictions_path)
    annotations = json.loads(annotation.read_text(encoding="utf-8"))
    results: dict[str, list[dict[str, Any]]] = {key: [] for key in OVO_GROUP_TASKS}
    expected_ids: list[str] = []
    empty_model_response_ids: list[str] = []
    for source in annotations:
        item = copy.deepcopy(source)
        task = str(item.get("task") or "").upper()
        group = next((name for name, tasks in OVO_GROUP_TASKS.items() if task in tasks), None)
        if group is None:
            raise ValueError(f"unknown OVO task: {task}")
        if group == "forward":
            for index, test_info in enumerate(item.get("test_info") or []):
                query_id = f"ovo_{item['id']}_{index}"
                expected_ids.append(query_id)
                response = str(predictions.get(query_id) or "")
                if not response:
                    empty_model_response_ids.append(query_id)
                test_info["response"] = response
        else:
            query_id = f"ovo_{item['id']}"
            expected_ids.append(query_id)
            letter = _mcq_letter(predictions.get(query_id), item.get("options") or [])
            response = letter if letter is not None else str(predictions.get(query_id) or "")
            if not response:
                empty_model_response_ids.append(query_id)
            item["response"] = response
            item["ground_truth"] = chr(ord("A") + int(item["gt"]))
        results[group].append(item)
    _require_exact_coverage(predictions, expected_ids, "OVO-Bench")

    score_module_path = official_root / "utils" / "OVOBenchScore.py"
    if not score_module_path.is_file():
        raise FileNotFoundError(score_module_path)
    spec = importlib.util.spec_from_file_location("official_ovo_score", score_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(score_module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = module.OVOBenchOfflineScore(SimpleNamespace(model="AgenticSFT"), copy.deepcopy(results))

    _, backward_scores = scorer.calculate_score_backward_realtime(copy.deepcopy(results["backward"]))
    _, realtime_scores = scorer.calculate_score_backward_realtime(copy.deepcopy(results["realtime"]))
    _, forward_scores = scorer.calculate_score_forward(copy.deepcopy(results["forward"]))
    score_maps = {
        "backward": backward_scores,
        "realtime": realtime_scores,
        "forward": forward_scores,
    }
    by_task: dict[str, dict[str, Any]] = {}
    official_by_group: dict[str, dict[str, Any]] = {}
    for group, expected_tasks in OVO_GROUP_TASKS.items():
        scores = score_maps[group]
        if set(scores) != set(expected_tasks):
            raise RuntimeError(f"official OVO scorer tasks for {group}: {sorted(scores)}")
        task_accuracies = []
        for task in expected_tasks:
            values = scores[task]
            accuracy = sum(values) / len(values)
            by_task[task] = {
                "accuracy": accuracy,
                "correct": float(sum(values)),
                "total": len(values),
            }
            task_accuracies.append(accuracy)
        official_by_group[group] = {
            "accuracy": mean(task_accuracies),
            "num_tasks": len(task_accuracies),
        }
    official_overall = mean(value["accuracy"] for value in official_by_group.values())
    derived_metrics = _ovo_derived_metrics(by_task, official_by_group)

    stdout_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer):
        module.OVOBenchOfflineScore(
            SimpleNamespace(model="AgenticSFT"), copy.deepcopy(results)
        ).score()
    official_stdout = stdout_buffer.getvalue().strip()
    _verify_ovo_stdout(official_stdout, by_task, official_by_group, official_overall)

    report = {
        "benchmark": "ovo",
        "official_scorer": str(score_module_path),
        "official_scorer_sha256": _sha256(score_module_path),
        "official_repository_commit": _git_commit(official_root),
        "official_stdout": official_stdout,
        "num_queries": len(expected_ids),
        "num_predictions": len(expected_ids),
        "coverage": 1.0,
        "num_empty_model_responses": len(empty_model_response_ids),
        "empty_model_response_ids": empty_model_response_ids,
        "empty_response_scoring": "passed_unchanged_to_original_scorer_without_answer_imputation",
        "by_task": by_task,
        "ovo_official": {
            "name": "Total Avg",
            "aggregation": "mean of Real-Time, Backward, and Forward task-macro averages",
            "accuracy": official_overall,
            "by_group": official_by_group,
        },
        **derived_metrics,
    }
    return report, results


def _ovo_derived_metrics(
    by_task: dict[str, dict[str, Any]],
    by_group: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    micro_correct = sum(float(value["correct"]) for value in by_task.values())
    micro_total = sum(int(value["total"]) for value in by_task.values())
    if micro_total <= 0:
        raise RuntimeError("OVO-Bench has no scored predictions")
    rt_bt_accuracy = mean(
        float(by_group[group]["accuracy"]) for group in ("realtime", "backward")
    )
    return {
        "ovo_micro_overall": {
            "name": "Overall",
            "aggregation": "micro average over all expanded OVO queries",
            "accuracy": micro_correct / micro_total,
            "correct": micro_correct,
            "total": micro_total,
        },
        "ovo_rt_bt_avg": {
            "name": "RT/BT Avg",
            "aggregation": "mean of Real-Time and Backward task-macro averages",
            "accuracy": rt_bt_accuracy,
        },
    }


def _require_exact_coverage(
    predictions: dict[str, Any], expected_ids: list[str], benchmark: str
) -> None:
    expected = set(expected_ids)
    missing = sorted(expected - set(predictions))
    if missing:
        raise RuntimeError(f"{benchmark} missing {len(missing)} predictions: {missing[:10]}")
    if len(expected_ids) != len(expected):
        raise RuntimeError(f"{benchmark} contains duplicate query IDs")


def _mcq_letter(prediction: Any, _options: list[Any]) -> str | None:
    text = str(prediction or "").strip()
    match = re.match(r"^[\(\[]?([A-D])(?:[\)\].:\s-]|$)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _verify_ovo_stdout(
    stdout: str,
    by_task: dict[str, dict[str, Any]],
    by_group: dict[str, dict[str, Any]],
    overall: float,
) -> None:
    printed_tasks = {
        task: float(value)
        for task, value in re.findall(r"Task:\s*(\w+),\s*Acc:\s*([0-9.]+)", stdout)
    }
    for task, values in by_task.items():
        if printed_tasks.get(task) != round(values["accuracy"] * 100, 2):
            raise RuntimeError(f"official OVO stdout mismatch for {task}")
    labels = {"backward": "Backward", "realtime": "Realtime", "forward": "Forward"}
    for group, label in labels.items():
        match = re.search(rf"{label} Avg\.:\s*([0-9.]+)", stdout)
        if match is None or float(match.group(1)) != round(by_group[group]["accuracy"] * 100, 2):
            raise RuntimeError(f"official OVO stdout mismatch for {group} Avg")
    match = re.search(r"Total Avg\.:\s*([0-9.]+)", stdout)
    if match is None or float(match.group(1)) != round(overall * 100, 2):
        raise RuntimeError("official OVO stdout mismatch for Total Avg")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


if __name__ == "__main__":
    main()
