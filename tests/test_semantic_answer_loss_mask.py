from __future__ import annotations

import torch

from experiments.train_qwen3vl_sft import _assistant_only_labels_and_weights


class _CharacterTokenizer:
    im_end_id = 1

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        del add_special_tokens
        value: dict[str, object] = {
            "input_ids": [ord(character) + 10 for character in text]
        }
        if return_offsets_mapping:
            value["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return value

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.im_end_id if token == "<|im_end|>" else None


class _BoundaryMergingTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False and add_generation_prompt is False
        return "".join(
            item["text"]
            for message in messages
            for item in message["content"]
            if item["type"] == "text"
        )

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        assert add_special_tokens is False
        if text == "ABC":
            value: dict[str, object] = {"input_ids": [100, 101]}
            if return_offsets_mapping:
                value["offset_mapping"] = [(0, 2), (2, 3)]
            return value
        return {"input_ids": [200 + ord(character) for character in text]}

    def convert_tokens_to_ids(self, token: str) -> None:
        del token
        return None


def test_only_assistant_semantic_answer_and_terminator_receive_loss() -> None:
    tokenizer = _CharacterTokenizer()
    question = "A"
    answer = "B"
    input_ids = torch.tensor(
        tokenizer(question)["input_ids"]
        + tokenizer(answer)["input_ids"]
        + [tokenizer.im_end_id],
        dtype=torch.long,
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": question}],
            "loss": False,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
            "loss": True,
        },
    ]
    labels, weights = _assistant_only_labels_and_weights(
        tokenizer, input_ids, messages
    )
    assert labels.tolist() == [-100, ord("B") + 10, tokenizer.im_end_id]
    assert weights.tolist() == [0.0, 1.0, 1.0]


def test_rendered_offsets_handle_tokens_crossing_message_boundaries() -> None:
    tokenizer = _BoundaryMergingTokenizer()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "A"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "B"}]},
        {"role": "user", "content": [{"type": "text", "text": "C"}]},
    ]
    labels, weights = _assistant_only_labels_and_weights(
        tokenizer, torch.tensor([100, 101]), messages
    )
    assert labels.tolist() == [100, -100]
    assert weights.tolist() == [1.0, 0.0]
