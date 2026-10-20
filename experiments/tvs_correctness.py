"""Official-protocol per-query correctness helpers for TVS analyses."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


OVO_FORWARD_TASKS = {"REC", "SSR", "CRR"}


def _mcq_letter(value: Any) -> str | None:
    text = str(value or "").strip()
    match = re.match(r"^[\(\[]?([A-D])(?:[\)\].:\s-]|$)", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def load_gold(benchmark: str, annotation: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(annotation.read_text())
    gold: dict[str, dict[str, Any]] = {}
    if benchmark == "streamingbench":
        for video_index, entry in enumerate(data):
            for question_index, question in enumerate(entry.get("questions") or []):
                uid = f"streaming_{video_index}_{question_index}"
                gold[uid] = {
                    "kind": "streaming_mcq",
                    "answer": str(question["answer"]),
                    "task": str(question.get("task_type") or ""),
                }
        return gold
    if benchmark != "ovo":
        raise ValueError(f"unsupported benchmark: {benchmark}")
    for item in data:
        task = str(item.get("task") or "").upper()
        if task not in OVO_FORWARD_TASKS:
            uid = f"ovo_{item['id']}"
            gold[uid] = {
                "kind": "ovo_mcq",
                "answer": chr(ord("A") + int(item["gt"])),
                "task": task,
            }
            continue
        for index, test_info in enumerate(item.get("test_info") or []):
            uid = f"ovo_{item['id']}_{index}"
            if task == "REC":
                gold[uid] = {
                    "kind": "ovo_rec",
                    "answer": str(test_info["count"]),
                    "task": task,
                }
            else:
                gold[uid] = {
                    "kind": "ovo_binary",
                    "answer": int(test_info["type"]),
                    "task": task,
                }
    return gold


def is_correct(prediction: Any, gold: dict[str, Any]) -> bool:
    text = str(prediction or "")
    kind = gold["kind"]
    if kind == "streaming_mcq":
        letter = _mcq_letter(text)
        response = letter if letter is not None else text
        return response == str(gold["answer"])
    if kind == "ovo_mcq":
        letter = _mcq_letter(text)
        response = letter if letter is not None else text
        return str(gold["answer"]) in response
    if kind == "ovo_rec":
        return "".join(re.findall(r"\d+", text)) == str(gold["answer"])
    if kind == "ovo_binary":
        positive = int(gold["answer"]) == 1
        if (text == "Y" and positive) or (text == "N" and not positive):
            return True
        return ("Yes" if positive else "No") in text
    raise ValueError(f"unknown gold kind: {kind}")
