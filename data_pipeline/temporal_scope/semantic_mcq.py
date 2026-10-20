"""Shared pure-VLM prompt and deterministic MCQ utilities."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence


DIRECT_PROMPT_PREFIX = (
    "You are an advanced video question-answering AI assistant. You have been "
    "provided with causal frames from the video and a multiple-choice question. "
    "Analyze only the visible frames and provide the best answer."
)
DIRECT_PROMPT_SUFFIX = "Only give the best option's letter (A, B, C, or D) directly."
OPTION_LETTERS = "ABCD"
DIRECT_PROMPT_RE = re.compile(
    r"(?:^|\n)Question:\s*(?P<question>.*?)\n+\s*Options:\s*\n"
    r"(?P<options>.*?)\n+\s*Only give the best option's letter",
    re.DOTALL,
)
OPTION_LINE_RE = re.compile(r"^\s*([A-D])[.):]\s*(.+?)\s*$")


class SemanticMCQError(ValueError):
    """Raised when a deterministic semantic MCQ cannot be constructed."""


def parse_direct_prompt(value: str) -> tuple[str, list[str]]:
    """Recover one canonical public question and its four visible options."""

    match = DIRECT_PROMPT_RE.search(str(value))
    if match is None:
        raise SemanticMCQError("canonical direct prompt is malformed")
    question = " ".join(match.group("question").split())
    parsed: dict[str, str] = {}
    for line in match.group("options").splitlines():
        option = OPTION_LINE_RE.fullmatch(line)
        if option is None:
            if line.strip():
                raise SemanticMCQError("canonical option line is malformed")
            continue
        letter, text = option.groups()
        if letter in parsed:
            raise SemanticMCQError("canonical option letter is duplicated")
        parsed[letter] = " ".join(text.split())
    if not question or list(parsed) != list(OPTION_LETTERS):
        raise SemanticMCQError("canonical prompt lacks one question and A--D options")
    options = [parsed[letter] for letter in OPTION_LETTERS]
    if any(not option for option in options) or len(set(options)) != len(options):
        raise SemanticMCQError("canonical prompt options are empty or duplicated")
    return question, options


def rebalance_answer_slot(
    options: Sequence[str],
    *,
    answer: str,
    answer_slot: int,
    seed: str,
) -> tuple[list[str], str]:
    """Move the target to a fixed slot and hash-order the three distractors."""

    if (
        len(options) != len(OPTION_LETTERS)
        or answer not in OPTION_LETTERS
        or answer_slot not in range(len(OPTION_LETTERS))
        or len(set(options)) != len(OPTION_LETTERS)
    ):
        raise SemanticMCQError("invalid four-option answer-space contract")
    target_index = OPTION_LETTERS.index(answer)
    target = str(options[target_index])
    distractors = [
        (index, str(option))
        for index, option in enumerate(options)
        if index != target_index
    ]
    distractors.sort(
        key=lambda item: hashlib.sha256(
            f"{seed}|{item[0]}|{item[1]}".encode("utf-8")
        ).digest()
    )
    balanced: list[str | None] = [None] * len(OPTION_LETTERS)
    balanced[answer_slot] = target
    remaining = [index for index, value in enumerate(balanced) if value is None]
    for slot, (_, option) in zip(remaining, distractors, strict=True):
        balanced[slot] = option
    if any(value is None for value in balanced):
        raise SemanticMCQError("balanced options are incomplete")
    return [str(value) for value in balanced], OPTION_LETTERS[answer_slot]
