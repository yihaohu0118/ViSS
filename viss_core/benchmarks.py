from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence


OVO_BACKWARD_TASKS = {"EPM", "ASI", "HLD"}
OVO_REALTIME_TASKS = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
OVO_FORWARD_TASKS = {"REC", "SSR", "CRR"}

# The public StreamingBench Real-Time annotation has one malformed timestamp.
# Its surrounding interval is [0:12:00 - 0:13:00], the audio filename is
# sample_132_00_12450.wav, and the source video is only 969.828 seconds long.
# Interpreting ``00:12450`` as 12:50 is therefore the only causal, in-range
# reading.  Keep this correction keyed by official UID so it cannot silently
# affect a different sample with a superficially similar timestamp.
STREAMING_TIMESTAMP_OVERRIDES = {"streaming_61_4": 12 * 60 + 50.0}

# Keep the forward-task prompts byte-for-byte aligned in meaning and output
# contract with OVO-Bench's public ``constant.py``.  These tasks are not plain
# video QA: SSR/CRR are Yes/No state-sufficiency decisions, and REC requires a
# bare integer.  Dropping the wrapper makes CRR look like open-ended QA and is
# therefore a benchmark protocol error, not merely a prompt variation.
OVO_REC_PROMPT = """You're watching a video in which people may perform a certain type of action repetively.
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one.
Now, answer the following question: {}
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response."""

OVO_SSR_PROMPT = """You're watching a tutorial video which contain a sequential of steps.
The following is one step from the whole procedures:
{}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response."""

OVO_CRR_PROMPT = """You're responsible of answering questions based on the video content.
The following question are relevant to the latest frames, i.e. the end of the video.
{}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response."""


@dataclass
class BenchmarkQuery:
    query_id: str
    benchmark: str
    video_path: str
    query_time: float
    question: str
    answer: str
    options: List[str] = field(default_factory=list)
    task: str = ""
    group: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_streameva_sample(self, clip_seconds: float = 20.0) -> Dict[str, Any]:
        start = max(0.0, self.query_time - clip_seconds)
        return {
            "uid": self.query_id,
            "benchmark": self.benchmark,
            "video": self.video_path,
            "question_type": "MCQ" if self.options else "FF",
            "query_time": self.query_time,
            "question": self.question,
            "options": list(self.options),
            "answer": self.answer,
            "support_keywords": _support_keywords(self.answer),
            "clips": [
                {
                    "start": start,
                    "end": self.query_time,
                    "summary": "",
                    "oracle_evidence": [],
                }
            ],
            "metadata": {
                "task": self.task,
                "group": self.group,
                **self.metadata,
            },
        }


def load_queries(benchmark: str, path: Path) -> List[BenchmarkQuery]:
    name = benchmark.lower()
    if name in {"ovo", "ovo-bench", "ovobench"}:
        return load_ovo(path)
    if name in {"streamingbench", "streaming"}:
        return load_streamingbench(path)
    raise ValueError(f"unsupported benchmark: {benchmark}")


def load_ovo(path: Path) -> List[BenchmarkQuery]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError("OVO-Bench annotations must be a JSON list")
    queries = []
    for item in data:
        task = str(item.get("task", ""))
        group = _ovo_group(task)
        if task in OVO_FORWARD_TASKS:
            queries.extend(_load_ovo_forward_item(item, task, group))
            continue
        gt = item.get("gt")
        options = [str(option) for option in item.get("options", [])]
        answer = str(item.get("answer", ""))
        if options and isinstance(gt, int) and 0 <= gt < len(options):
            answer = options[gt]
        queries.append(
            BenchmarkQuery(
                query_id=f"ovo_{item.get('id')}",
                benchmark="ovo",
                video_path=str(item.get("video", "")),
                query_time=float(item.get("realtime", 0.0)),
                question=str(item.get("question", "")),
                answer=answer,
                options=options,
                task=task,
                group=group,
                metadata={
                    "id": item.get("id"),
                    "gt": gt,
                    "answer_text": item.get("answer", ""),
                },
            )
        )
    return queries


def _load_ovo_forward_item(
    item: Dict[str, Any],
    task: str,
    group: str,
) -> List[BenchmarkQuery]:
    queries = []
    for idx, test_info in enumerate(item.get("test_info", [])):
        query_id = f"ovo_{item.get('id')}_{idx}"
        realtime = float(test_info.get("realtime", item.get("realtime", 0.0)))
        if task == "REC":
            activity = str(item.get("activity", "the target action"))
            question = OVO_REC_PROMPT.format(f"How many times did they {activity}?")
            answer = str(test_info.get("count", ""))
        elif task == "SSR":
            step = str(test_info.get("step", "the queried step"))
            question = OVO_SSR_PROMPT.format(step)
            answer = "Yes" if int(test_info.get("type", 0)) == 1 else "No"
        else:
            question = OVO_CRR_PROMPT.format(str(item.get("question", "")))
            answer = "Yes" if int(test_info.get("type", 0)) == 1 else "No"
        queries.append(
            BenchmarkQuery(
                query_id=query_id,
                benchmark="ovo",
                video_path=str(item.get("video", "")),
                query_time=realtime,
                question=question,
                answer=answer,
                task=task,
                group=group,
                metadata={
                    "id": item.get("id"),
                    "test_index": idx,
                    "test_info": dict(test_info),
                    "activity": item.get("activity"),
                    "tutorial": item.get("tutorial"),
                    "ask_time": item.get("ask_time"),
                    "clue_time": item.get("clue_time"),
                    "answer_text": item.get("answer", ""),
                },
            )
        )
    return queries


def load_streamingbench(path: Path) -> List[BenchmarkQuery]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError("StreamingBench annotations must be a JSON list")
    queries = []
    for video_index, subset in enumerate(_iter_streaming_subsets(data)):
        video_path = str(subset.get("video_path", ""))
        video_category = subset.get("video_categories")
        for question_index, question in enumerate(subset.get("questions", [])):
            query_id = f"streaming_{video_index}_{question_index}"
            raw_timestamp = str(question.get("time_stamp", "0"))
            task = str(question.get("task_type", ""))
            answer = str(
                question.get(
                    "answer",
                    question.get("ground_truth_output", ""),
                )
            )
            queries.append(
                BenchmarkQuery(
                    query_id=query_id,
                    benchmark="streamingbench",
                    video_path=video_path,
                    query_time=streaming_query_time(query_id, raw_timestamp),
                    question=str(question.get("question", "")),
                    answer=answer,
                    options=[str(option) for option in question.get("options", [])],
                    task=task,
                    group=_streaming_group(task),
                    metadata={
                        "video_index": video_index,
                        "question_index": question_index,
                        "time": question.get("time"),
                        "audio_path": question.get("audio_path"),
                        "required_ability": question.get("required_ability"),
                        "video_categories": video_category,
                        "ground_truth_time_stamp": question.get("ground_truth_time_stamp"),
                        "ground_truth_output": question.get("ground_truth_output"),
                        "raw_time_stamp": raw_timestamp,
                        "timestamp_override_applied": (
                            query_id in STREAMING_TIMESTAMP_OVERRIDES
                        ),
                    },
                )
            )
    return queries


def score_predictions(
    queries: Sequence[BenchmarkQuery],
    predictions: Dict[str, Any],
) -> Dict[str, Any]:
    rows = []
    for query in queries:
        pred = predictions.get(query.query_id)
        correct = _score_one(query, pred)
        rows.append(
            {
                "query_id": query.query_id,
                "benchmark": query.benchmark,
                "task": query.task,
                "group": query.group,
                "prediction": pred,
                "answer": query.answer,
                "correct": correct,
            }
        )
    by_group = _aggregate(rows, "group")
    by_task = _aggregate(rows, "task")
    report = {
        "num_queries": len(rows),
        "accuracy": mean(row["correct"] for row in rows) if rows else 0.0,
        "by_group": by_group,
        "by_task": by_task,
        "rows": rows,
    }
    if rows and all(row["benchmark"] == "ovo" for row in rows):
        report["ovo_official"] = _ovo_official_aggregate(by_task)
    return report


def _ovo_official_aggregate(by_task: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Return OVO-Bench's task-macro group averages and group-macro overall."""
    group_tasks = {
        "backward": sorted(OVO_BACKWARD_TASKS),
        "realtime": sorted(OVO_REALTIME_TASKS),
        "forward": sorted(OVO_FORWARD_TASKS),
    }
    by_group = {}
    group_scores = []
    for group, tasks in group_tasks.items():
        scores = [
            float(by_task[task]["accuracy"])
            for task in tasks
            if task in by_task and int(by_task[task].get("total", 0)) > 0
        ]
        if not scores:
            continue
        score = mean(scores)
        by_group[group] = {"accuracy": score, "num_tasks": len(scores)}
        group_scores.append(score)
    return {"accuracy": mean(group_scores) if group_scores else 0.0, "by_group": by_group}


def load_prediction_file(path: Path) -> Dict[str, Any]:
    """Load predictions from common StreamEVA/Qwen smoke JSON or JSONL files."""
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        data = json.loads(stripped)
    except JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return _predictions_from_data(data)


def load_query_ids_from_jsonl(path: Path) -> List[str]:
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = row.get("uid", row.get("query_id", row.get("id")))
            if query_id is not None:
                ids.append(str(query_id))
    return ids


def build_score_report(
    queries: Sequence[BenchmarkQuery],
    predictions: Dict[str, Any],
    *,
    query_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    query_by_id = {query.query_id: query for query in queries}
    if query_ids is None:
        selected_ids = sorted(query_by_id)
    else:
        selected_ids = [str(query_id) for query_id in query_ids if str(query_id) in query_by_id]
    selected_queries = [query_by_id[query_id] for query_id in selected_ids]
    answered_ids = [query_id for query_id in selected_ids if query_id in predictions]
    answered_queries = [query_by_id[query_id] for query_id in answered_ids]
    full_score = score_predictions(selected_queries, predictions)
    answered_score = score_predictions(answered_queries, predictions)
    unknown_prediction_ids = sorted(str(query_id) for query_id in predictions if str(query_id) not in query_by_id)
    missing_ids = [query_id for query_id in selected_ids if query_id not in predictions]
    return {
        "num_queries": len(selected_queries),
        "num_predictions": len(predictions),
        "num_answered": len(answered_ids),
        "coverage": len(answered_ids) / len(selected_queries) if selected_queries else 0.0,
        "accuracy": full_score["accuracy"],
        "answered_accuracy": answered_score["accuracy"],
        "by_group": full_score["by_group"],
        "by_task": full_score["by_task"],
        "ovo_official": full_score.get("ovo_official"),
        "answered_by_group": answered_score["by_group"],
        "answered_by_task": answered_score["by_task"],
        "answered_ovo_official": answered_score.get("ovo_official"),
        "num_missing_predictions": len(missing_ids),
        "missing_predictions_sample": missing_ids[:20],
        "num_unknown_predictions": len(unknown_prediction_ids),
        "unknown_predictions_sample": unknown_prediction_ids[:20],
        "rows": full_score["rows"],
    }


def oracle_predictions(queries: Iterable[BenchmarkQuery]) -> Dict[str, str]:
    predictions = {}
    for query in queries:
        answer_letter = _answer_letter(query) if query.options else None
        predictions[query.query_id] = answer_letter.upper() if answer_letter else query.answer
    return predictions


def export_streameva_jsonl(
    queries: Iterable[BenchmarkQuery],
    output_path: Path,
    *,
    clip_seconds: float = 20.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for query in queries:
            f.write(json.dumps(query.to_streameva_sample(clip_seconds), ensure_ascii=False) + "\n")


def resolve_video_path(query: BenchmarkQuery, root: Path) -> Path:
    raw = Path(query.video_path)
    if raw.is_absolute():
        return raw
    cleaned = str(raw)
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    candidates = _chunked_video_candidates(query, root) + [
        root / cleaned,
        root / "data" / "src_videos" / cleaned,
        root / "data" / "chunked_videos" / cleaned,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _chunked_video_candidates(query: BenchmarkQuery, root: Path) -> List[Path]:
    if query.benchmark != "ovo":
        return []
    item_id = query.metadata.get("id")
    if item_id is None:
        return []
    chunked_dir = root / "data" / "chunked_videos"
    if query.group == "forward":
        test_index = query.metadata.get("test_index")
        if test_index is not None:
            return [chunked_dir / f"{item_id}_{test_index}.mp4"]
    return [chunked_dir / f"{item_id}.mp4"]


def asset_coverage(queries: Sequence[BenchmarkQuery], video_root: Path) -> Dict[str, Any]:
    rows = []
    for query in queries:
        path = resolve_video_path(query, video_root)
        exists = path.exists()
        rows.append(
            {
                "query_id": query.query_id,
                "benchmark": query.benchmark,
                "group": query.group,
                "task": query.task,
                "video": query.video_path,
                "resolved_video": str(path),
                "exists": exists,
            }
        )
    covered = sum(1 for row in rows if row["exists"])
    by_group = _coverage_aggregate(rows, "group")
    by_task = _coverage_aggregate(rows, "task")
    missing_videos = sorted({row["resolved_video"] for row in rows if not row["exists"]})
    present_videos = sorted({row["resolved_video"] for row in rows if row["exists"]})
    return {
        "num_queries": len(rows),
        "covered_queries": covered,
        "coverage": covered / len(rows) if rows else 0.0,
        "num_present_videos": len(present_videos),
        "num_missing_videos": len(missing_videos),
        "by_group": by_group,
        "by_task": by_task,
        "missing_videos_sample": missing_videos[:20],
        "present_videos_sample": present_videos[:20],
        "rows": rows,
    }


def parse_timestamp(value: str) -> float:
    value = value.strip()
    if not value:
        return 0.0
    if re.fullmatch(r"\d+(\.\d+)?", value):
        return float(value)
    parts = value.split(":")
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total


def streaming_query_time(query_id: str, raw_timestamp: str) -> float:
    return STREAMING_TIMESTAMP_OVERRIDES.get(
        str(query_id), parse_timestamp(str(raw_timestamp))
    )


def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _predictions_from_data(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        if isinstance(data.get("predictions"), dict):
            return {str(key): value for key, value in data["predictions"].items()}
        if isinstance(data.get("rows"), list):
            return _predictions_from_rows(data["rows"])
        if _looks_like_prediction_mapping(data):
            return {str(key): value for key, value in data.items()}
        return {}
    if isinstance(data, list):
        return _predictions_from_rows(data)
    return {}


def _predictions_from_rows(rows: Sequence[Any]) -> Dict[str, Any]:
    predictions = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        query_id = row.get("uid", row.get("query_id", row.get("id")))
        if query_id is None:
            continue
        prediction = _row_prediction(row)
        if prediction is not None:
            predictions[str(query_id)] = prediction
    return predictions


def _row_prediction(row: Dict[str, Any]) -> Any:
    for key in ["prediction", "pred", "model_answer", "output", "response", "raw_response"]:
        if key in row:
            return row[key]
    return None


def _looks_like_prediction_mapping(data: Dict[str, Any]) -> bool:
    if not data:
        return True
    report_keys = {"model", "score", "rows", "num_samples", "samples", "metadata"}
    if report_keys & set(data):
        return False
    return all(not isinstance(value, (dict, list)) for value in data.values())


def _ovo_group(task: str) -> str:
    if task in OVO_BACKWARD_TASKS:
        return "backward"
    if task in OVO_REALTIME_TASKS:
        return "realtime"
    if task in OVO_FORWARD_TASKS:
        return "forward"
    return "unknown"


def _streaming_group(task: str) -> str:
    if task == "Sequential Question Answering":
        return "contextual_sqa"
    if task == "Proactive Output":
        return "contextual_proactive"
    if task in {"Misleading Context Understanding", "Anomaly Context Understanding"}:
        return "contextual"
    if task in {
        "Emotion Recognition",
        "Scene Understanding",
        "Source Discrimination",
        "Multimodal Alignment",
    }:
        return "omni_source"
    return "real_time_visual"


def _iter_streaming_subsets(data: List[Any]) -> Iterable[Dict[str, Any]]:
    for item in data:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, list):
            for subitem in item:
                if isinstance(subitem, dict):
                    yield subitem


def _score_one(query: BenchmarkQuery, prediction: Any) -> float:
    if prediction is None:
        return 0.0
    raw_prediction = str(prediction).strip()
    pred = _normalize_prediction(raw_prediction)
    answer = _normalize_prediction(query.answer)
    if not pred:
        return 0.0
    if query.options:
        answer_letter = _answer_letter(query)
        if answer_letter:
            explicit_letter = re.match(
                r"^[\(\[]?([a-z])(?:[\)\].:\s-]|$)",
                raw_prediction,
                flags=re.IGNORECASE,
            )
            if explicit_letter:
                return 1.0 if explicit_letter.group(1).lower() == answer_letter else 0.0
            option_index = ord(answer_letter) - ord("a")
            if 0 <= option_index < len(query.options):
                option_text = _normalize_prediction(
                    re.sub(
                        r"^[\(\[]?[a-z][\)\].:\s-]+",
                        "",
                        query.options[option_index],
                        flags=re.IGNORECASE,
                    )
                )
                return 1.0 if pred == option_text or option_text in pred else 0.0
    if query.task == "REC":
        pred_number = _rec_number(raw_prediction)
        answer_number = "".join(re.findall(r"\d+", answer))
        return 1.0 if pred_number and pred_number == answer_number else 0.0
    return 1.0 if pred == answer or answer in pred else 0.0


def _answer_letter(query: BenchmarkQuery) -> str:
    gt = query.metadata.get("gt")
    if isinstance(gt, int) and 0 <= gt < len(query.options):
        return chr(ord("a") + gt)
    answer = _normalize_prediction(query.answer)
    for idx, option in enumerate(query.options):
        if _normalize_prediction(option) == answer:
            return chr(ord("a") + idx)
        if _normalize_prediction(option).startswith(answer):
            return chr(ord("a") + idx)
    if len(answer) == 1 and answer.isalpha():
        return answer
    return ""


def _rec_number(text: str) -> str:
    """Canonicalize concise count answers while leaving the official scorer unchanged."""
    digits = "".join(re.findall(r"\d+", str(text)))
    if digits:
        return digits
    normalized = _normalize_prediction(str(text))
    words = {
        "zero": "0",
        "none": "0",
        "once": "1",
        "one": "1",
        "one time": "1",
        "twice": "2",
        "two": "2",
        "two times": "2",
        "thrice": "3",
        "three": "3",
        "three times": "3",
        "four": "4",
        "four times": "4",
        "five": "5",
        "five times": "5",
        "six": "6",
        "six times": "6",
        "seven": "7",
        "seven times": "7",
        "eight": "8",
        "eight times": "8",
        "nine": "9",
        "nine times": "9",
        "ten": "10",
        "ten times": "10",
    }
    return words.get(normalized, "")


def _normalize_prediction(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^[\(\[]?([a-z])[\)\].:\s-]+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _aggregate(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = {}
    for row in rows:
        name = str(row.get(key) or "unknown")
        buckets.setdefault(name, []).append(float(row["correct"]))
    return {
        name: {
            "total": len(values),
            "correct": sum(values),
            "accuracy": sum(values) / len(values) if values else 0.0,
        }
        for name, values in sorted(buckets.items())
    }


def _coverage_aggregate(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[bool]] = {}
    for row in rows:
        name = str(row.get(key) or "unknown")
        buckets.setdefault(name, []).append(bool(row["exists"]))
    return {
        name: {
            "total": len(values),
            "covered": sum(1 for value in values if value),
            "coverage": sum(1 for value in values if value) / len(values) if values else 0.0,
        }
        for name, values in sorted(buckets.items())
    }


def _support_keywords(answer: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z0-9]+", answer.lower())
        if len(token) > 1
    ][:5]
