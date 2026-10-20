#!/usr/bin/env python3
"""Audited full-parameter Qwen3-VL trainer for the pure-VLM backbone."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset, SequentialSampler

try:
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except ModuleNotFoundError:
    AutoModelForImageTextToText = None
    AutoProcessor = None
    TrainingArguments = None
    set_seed = None

    class Trainer:  # type: ignore[no-redef]
        """Import-only fallback for dependency-light contract tests."""

        pass

from viss_core.data_v0.trainer_adapter import (
    infer_release_root,
    load_training_rows,
    normalize_messages,
)
class MediaSemanticDataset(Dataset):
    """Load only a sealed pure-semantic JSONL release."""

    def __init__(
        self,
        path: Path,
        processor: Any,
        *,
        max_samples: int = 0,
        media_root: Path | None = None,
        development_unsealed_overlay: bool = False,
    ) -> None:
        self.development_unsealed_overlay = development_unsealed_overlay
        if development_unsealed_overlay:
            self.rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            self.rows = _load_training_rows(path)
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.processor = processor
        inferred_root = _infer_release_root(path)
        if development_unsealed_overlay:
            if media_root is None:
                raise RuntimeError("development overlay requires --media-root")
            self.media_root = media_root.resolve(strict=True)
        else:
            if inferred_root is None:
                raise RuntimeError("semantic release root could not be inferred")
            if media_root is not None and media_root.resolve() != inferred_root:
                raise RuntimeError("--media-root may not override sealed release media")
            self.media_root = inferred_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        raw_weight = row.get("sample_loss_weight")
        sample_loss_weight = float(1.0 if raw_weight is None else raw_weight)
        if not math.isfinite(sample_loss_weight) or sample_loss_weight <= 0:
            raise RuntimeError(
                "train-ready contract violated: "
                f"uid={row.get('uid') or f'row_{index}'} "
                f"sample_loss_weight={sample_loss_weight}"
            )
        return {
            "uid": str(row.get("uid") or f"row_{index}"),
            "messages": (
                _normalize_development_messages(row["messages"])
                if self.development_unsealed_overlay
                else _normalize_messages(row["messages"], media_root=self.media_root)
            ),
            "sample_loss_weight": sample_loss_weight,
        }


@dataclass
class MediaSemanticCollator:
    processor: Any
    max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded = []
        labels_list = []
        weights_list = []
        for feature in features:
            messages = feature["messages"]
            item = _encode_processor_item(self.processor, messages)
            if item["input_ids"].numel() > self.max_length:
                raise RuntimeError(
                    f"train-ready contract violated: uid={feature['uid']} "
                    f"tokens={item['input_ids'].numel()} max_length={self.max_length}"
                )
            labels, loss_weights = _assistant_only_labels_and_weights(
                getattr(self.processor, "tokenizer", self.processor),
                item["input_ids"],
                messages,
            )
            if not torch.any(labels != -100):
                raise RuntimeError(
                    f"train-ready contract violated: uid={feature['uid']} "
                    "has no assistant labels"
                )
            encoded.append(item)
            labels_list.append(labels)
            weights_list.append(loss_weights)

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if tokenizer.pad_token_id is None:
            raise RuntimeError("tokenizer has no pad token")
        maximum = max(item["input_ids"].numel() for item in encoded)
        batch: Dict[str, list[torch.Tensor]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "loss_weights": [],
            "sample_loss_weight": [],
        }
        pixel_values = []
        image_grid_thw = []
        for item, labels, loss_weights, feature in zip(
            encoded, labels_list, weights_list, features, strict=True
        ):
            pad = maximum - item["input_ids"].numel()
            batch["input_ids"].append(
                torch.nn.functional.pad(
                    item["input_ids"], (0, pad), value=tokenizer.pad_token_id
                )
            )
            batch["attention_mask"].append(
                torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0)
            )
            batch["labels"].append(
                torch.nn.functional.pad(labels, (0, pad), value=-100)
            )
            batch["loss_weights"].append(
                torch.nn.functional.pad(loss_weights, (0, pad), value=0.0)
            )
            batch["sample_loss_weight"].append(
                torch.tensor(
                    float(feature["sample_loss_weight"]), dtype=torch.float32
                )
            )
            if "pixel_values" in item:
                pixel_values.append(item["pixel_values"])
            if "image_grid_thw" in item:
                image_grid_thw.append(item["image_grid_thw"])
        output = {key: torch.stack(value) for key, value in batch.items()}
        if pixel_values:
            output["pixel_values"] = torch.cat(pixel_values, dim=0)
        if image_grid_thw:
            output["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
        return output


def main() -> None:
    if any(
        value is None
        for value in (
            AutoModelForImageTextToText,
            AutoProcessor,
            TrainingArguments,
            set_seed,
        )
    ):
        raise RuntimeError(
            "training requires transformers; the dependency-light import path "
            "supports only data and loss-mask contract tests"
        )
    parser = argparse.ArgumentParser(
        description="Full-parameter Qwen3-VL pure-semantic SFT."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--val-media-root", type=Path)
    parser.add_argument(
        "--development-unsealed-overlay",
        action="store_true",
        help="Development only: read an unsealed JSONL overlay with absolute media paths.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-pixels", type=int, default=0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--vision-projector-lr-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply the optimizer learning rate for trainable visual "
            "merger/deepstack projector parameters."
        ),
    )
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("cosine", "constant_with_warmup"),
        default="cosine",
    )
    parser.add_argument(
        "--train-order", choices=("shuffled", "sequential"), default="shuffled"
    )
    parser.add_argument(
        "--stage-lr-plan",
        type=Path,
        help="Audited row-boundary learning-rate plan for one ordered run.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--deepspeed", type=Path, required=True)
    parser.add_argument("--freeze-vision-tower", action="store_true")
    parser.add_argument("--train-vision-projector", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--skip-final-save", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.max_length <= 0 or args.max_pixels < 0:
        raise ValueError("length must be positive and max-pixels non-negative")
    if (
        not math.isfinite(args.vision_projector_lr_scale)
        or args.vision_projector_lr_scale <= 0
    ):
        raise ValueError("vision-projector-lr-scale must be finite and positive")
    if args.train_vision_projector and not args.freeze_vision_tower:
        raise ValueError("--train-vision-projector requires --freeze-vision-tower")
    if args.vision_projector_lr_scale != 1.0 and not args.train_vision_projector:
        raise ValueError(
            "a non-default vision-projector LR scale requires a trainable projector"
        )
    if not args.deepspeed.is_file():
        raise FileNotFoundError(args.deepspeed)

    set_seed(args.seed)
    has_eval = args.val is not None and args.val.exists()
    training_args = _training_args(args, has_eval)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    _configure_processor_pixel_budget(processor, args.max_pixels)
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if args.freeze_vision_tower:
        model.visual.requires_grad_(False)
        if args.train_vision_projector:
            for name in ("merger", "deepstack_merger_list"):
                module = getattr(model.visual, name, None)
                if module is None:
                    raise RuntimeError(
                        f"Qwen3-VL visual module lacks projector component {name!r}"
                    )
                module.requires_grad_(True)
    _audit_trainable_parameters(model, args)

    if _is_main_process():
        trainable = sum(
            _logical_parameter_numel(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total = sum(
            _logical_parameter_numel(parameter) for parameter in model.parameters()
        )
        if total <= 0:
            raise RuntimeError("could not determine logical parameter count")
        print(
            {
                "training_mode": "full_parameter_no_adapter",
                "vision_tower_frozen": args.freeze_vision_tower,
                "trainable_parameters": trainable,
                "total_parameters": total,
                "trainable_fraction": trainable / total,
            },
            flush=True,
        )

    train_dataset = MediaSemanticDataset(
        args.train,
        processor,
        max_samples=args.max_train_samples,
        media_root=args.media_root,
        development_unsealed_overlay=args.development_unsealed_overlay,
    )
    global_batch = (
        args.per_device_batch_size
        * args.gradient_accumulation_steps
        * int(os.environ.get("WORLD_SIZE", "1"))
    )
    stage_lr_plan = (
        _load_stage_lr_plan(
            args.stage_lr_plan,
            release_manifest=inferred_root_manifest(train_dataset.media_root),
            dataset_rows=len(train_dataset),
            global_batch=global_batch,
            optimizer_learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
        )
        if args.stage_lr_plan is not None
        else None
    )
    eval_dataset = (
        MediaSemanticDataset(
            args.val,
            processor,
            max_samples=args.max_val_samples,
            media_root=args.val_media_root,
        )
        if has_eval
        else None
    )
    trainer = SemanticTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MediaSemanticCollator(processor, args.max_length),
        train_order=args.train_order,
        stage_lr_plan=stage_lr_plan,
        vision_projector_lr_scale=args.vision_projector_lr_scale,
        vision_projector_parameter_ids={
            id(parameter)
            for name in ("merger", "deepstack_merger_list")
            for parameter in getattr(model.visual, name).parameters()
        }
        if args.train_vision_projector
        else set(),
    )
    trainer.train()
    if args.skip_final_save:
        if _is_main_process():
            print(
                {
                    "output": str(args.output),
                    "training_mode": "full_parameter_no_adapter",
                    "final_model": None,
                    "systems_smoke_skipped_final_save": True,
                },
                flush=True,
            )
        return
    _release_save_reserve()
    final_dir = args.output / "final_model"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    if _is_main_process():
        print(
            {
                "output": str(args.output),
                "training_mode": "full_parameter_no_adapter",
                "final_model": str(final_dir),
            },
            flush=True,
        )


def _release_save_reserve() -> None:
    raw = os.environ.get("STREAMEVA_SAVE_RESERVE_PATH")
    if raw and _is_main_process():
        path = Path(raw).resolve()
        if (
            path.name != "save_reserve.bin"
            or path.parent.parent != Path("/tmp")
            or not path.parent.name.startswith("streameva_")
        ):
            raise RuntimeError("save reserve path is outside the short runtime root")
        path.unlink(missing_ok=True)
        os.sync()
        print(
            {
                "save_reserve_released": str(path),
                "free_bytes": shutil.disk_usage(path.parent).free,
            },
            flush=True,
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _audit_trainable_parameters(model: Any, args: argparse.Namespace) -> None:
    if not all(
        parameter.requires_grad for parameter in model.language_model.parameters()
    ):
        raise RuntimeError("full-parameter contract froze a language parameter")
    if not all(parameter.requires_grad for parameter in model.lm_head.parameters()):
        raise RuntimeError("full-parameter contract froze an LM-head parameter")
    if not args.freeze_vision_tower:
        return
    projector_ids: set[int] = set()
    if args.train_vision_projector:
        for name in ("merger", "deepstack_merger_list"):
            projector_ids.update(
                id(parameter)
                for parameter in getattr(model.visual, name).parameters()
            )
    for parameter in model.visual.parameters():
        if parameter.requires_grad != (id(parameter) in projector_ids):
            raise RuntimeError("frozen-vision/trainable-projector contract violated")


def _load_training_rows(
    path: Path, *, training_schema: str = "semantic"
) -> List[Dict[str, Any]]:
    if training_schema != "semantic":
        raise ValueError(f"unsupported training schema: {training_schema}")
    return load_training_rows(path)


def _normalize_development_messages(messages: Any) -> List[Dict[str, Any]]:
    """Normalize an explicit development overlay without weakening release loading."""
    if not isinstance(messages, list):
        raise TypeError("development messages must be a list")
    roles = [message.get("role") for message in messages]
    if roles != ["user", "assistant"]:
        raise ValueError("development overlay accepts pure-VLM rows only")
    output: List[Dict[str, Any]] = []
    for message in messages:
        content: List[Dict[str, Any]] = []
        for item in message.get("content") or []:
            if item.get("type") == "text":
                content.append({"type": "text", "text": str(item.get("text") or "")})
                continue
            if item.get("type") != "image" or message["role"] != "user":
                raise ValueError("development overlay contains unsupported content")
            image_path = Path(str(item.get("image") or ""))
            if not image_path.is_absolute() or not image_path.is_file():
                raise RuntimeError("development overlay image must be an existing absolute path")
            content.append(
                {"type": "image", "image": Image.open(image_path).convert("RGB")}
            )
        if not content:
            raise ValueError("development overlay message content is empty")
        output.append(
            {
                "role": str(message["role"]),
                "content": content,
                "loss": bool(message.get("loss")),
            }
        )
    return output


def _infer_release_root(
    path: Path, *, training_schema: str = "semantic"
) -> Path | None:
    if training_schema != "semantic":
        raise ValueError(f"unsupported training schema: {training_schema}")
    return infer_release_root(path)


def _normalize_messages(
    messages: Any,
    *,
    media_root: Path | None = None,
    training_schema: str = "semantic",
) -> List[Dict[str, Any]]:
    if training_schema != "semantic":
        raise ValueError(f"unsupported training schema: {training_schema}")
    return normalize_messages(messages, media_root=media_root)


def _load_images(messages: List[Dict[str, Any]]) -> List[Image.Image]:
    images = []
    for message in messages:
        for item in message.get("content", []):
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            image = item.get("image")
            if isinstance(image, Image.Image):
                images.append(image)
            elif image:
                images.append(Image.open(image).convert("RGB"))
    return images


def _configure_processor_pixel_budget(
    processor: Any, max_pixels: int
) -> Dict[str, int]:
    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    if image_processor is None or not isinstance(size, dict):
        raise RuntimeError("processor has no configurable image size contract")
    shortest = int(size.get("shortest_edge", 0))
    longest = int(size.get("longest_edge", 0))
    if shortest <= 0 or longest <= 0 or shortest > longest:
        raise RuntimeError("processor image size contract is invalid")
    requested = longest if max_pixels == 0 else int(max_pixels)
    if requested < shortest:
        raise ValueError(f"max_pixels={requested} is below shortest_edge={shortest}")
    image_processor.size = {
        "shortest_edge": shortest,
        "longest_edge": requested,
    }
    return dict(image_processor.size)


def _encode_processor_item(
    processor: Any, messages: List[Dict[str, Any]]
) -> Dict[str, torch.Tensor]:
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    images = _load_images(messages)
    if rendered.count("<|image_pad|>") != len(images):
        raise RuntimeError("chat-template image slots and message images differ")
    kwargs: Dict[str, Any] = {
        "text": rendered,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if images:
        kwargs["images"] = images
    item = processor(**kwargs)
    return {
        key: (
            value.squeeze(0)
            if hasattr(value, "dim")
            and value.dim() > 1
            and key in {"input_ids", "attention_mask"}
            else value
        )
        for key, value in item.items()
    }


def _locate_rendered_text_token_positions(
    tokenizer: Any,
    input_ids: torch.Tensor,
    messages: List[Dict[str, Any]],
) -> Dict[tuple[int, int], List[int]] | None:
    """Align message text spans through chat rendering and visual expansion."""

    if not hasattr(tokenizer, "apply_chat_template"):
        return None
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (KeyError, NotImplementedError, TypeError, ValueError):
        return None
    template_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if template_ids is None or offsets is None:
        return None
    if hasattr(template_ids, "tolist"):
        template_ids = template_ids.tolist()
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if template_ids and isinstance(template_ids[0], list):
        template_ids = template_ids[0]
    if offsets and isinstance(offsets[0][0], list):
        offsets = offsets[0]

    actual_ids = input_ids.tolist()
    visual_pad_ids = {
        token_id
        for token in ("<|image_pad|>", "<|video_pad|>")
        for token_id in [tokenizer.convert_tokens_to_ids(token)]
        if isinstance(token_id, int) and token_id >= 0
    }
    template_to_actual: Dict[int, int] = {}
    actual_index = 0
    for template_index, token_id in enumerate(template_ids):
        if actual_index >= len(actual_ids) or actual_ids[actual_index] != token_id:
            raise RuntimeError("rendered template and processor input IDs diverged")
        template_to_actual[template_index] = actual_index
        actual_index += 1
        if token_id in visual_pad_ids:
            while actual_index < len(actual_ids) and actual_ids[actual_index] == token_id:
                actual_index += 1
    if actual_index != len(actual_ids):
        raise RuntimeError("processor input IDs contain non-template tokens")

    output: Dict[tuple[int, int], List[int]] = {}
    character_cursor = 0
    for message_index, message in enumerate(messages):
        text_items = [
            item
            for item in message.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        for text_index, item in enumerate(text_items):
            text = str(item.get("text", ""))
            start = rendered.find(text, character_cursor)
            if start < 0:
                raise RuntimeError("message text is absent from the rendered template")
            end = start + len(text)
            positions = [
                template_to_actual[index]
                for index, pair in enumerate(offsets)
                if int(pair[0]) < end and int(pair[1]) > start
            ]
            if text and not positions:
                raise RuntimeError("rendered message text has no token positions")
            output[(message_index, text_index)] = positions
            character_cursor = end
    return output


def _assistant_only_labels(
    tokenizer: Any,
    input_ids: torch.Tensor,
    messages: List[Dict[str, Any]],
) -> torch.Tensor:
    labels, _ = _assistant_only_labels_and_weights(tokenizer, input_ids, messages)
    return labels


def _assistant_only_labels_and_weights(
    tokenizer: Any,
    input_ids: torch.Tensor,
    messages: List[Dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise assistant semantic text and its turn terminator only."""

    labels = torch.full_like(input_ids, -100)
    weights = torch.zeros_like(input_ids, dtype=torch.float32)
    ids = input_ids.tolist()
    cursor = 0
    rendered_positions = _locate_rendered_text_token_positions(
        tokenizer, input_ids, messages
    )
    for message_index, message in enumerate(messages):
        is_assistant = message.get("role") == "assistant"
        supervise = is_assistant and message.get("loss", True) is not False
        text_items = [
            item
            for item in message.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        for text_index, item in enumerate(text_items):
            text = str(item.get("text", ""))
            if rendered_positions is not None:
                positions = rendered_positions[(message_index, text_index)]
                start = positions[0] if positions else cursor
                semantic_end = positions[-1] + 1 if positions else cursor
            else:
                target = tokenizer(text, add_special_tokens=False)["input_ids"]
                start = _find_subsequence(ids, target, cursor)
                if start < 0:
                    raise RuntimeError("could not locate message text in role order")
                semantic_end = min(start + len(target), labels.numel())
            end = semantic_end
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if (
                is_assistant
                and text_index == len(text_items) - 1
                and im_end_id is not None
                and end < labels.numel()
                and ids[end] == im_end_id
            ):
                end += 1
            cursor = end
            if supervise:
                labels[start:end] = input_ids[start:end]
                weights[start:end] = 1.0
    return labels, weights


def _find_subsequence(haystack: List[int], needle: List[int], start: int) -> int:
    if not needle:
        return -1
    for index in range(max(start, 0), len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            return index
    return -1


def _split_optimizer_groups_for_projector(
    param_groups: list[dict[str, Any]],
    *,
    projector_parameter_ids: set[int],
    learning_rate_scale: float,
) -> tuple[list[dict[str, Any]], int]:
    """Preserve group options while isolating/scaling projector parameters."""

    replacement_groups: list[dict[str, Any]] = []
    projector_parameters = 0
    for original in param_groups:
        projector = [
            parameter
            for parameter in original["params"]
            if id(parameter) in projector_parameter_ids
        ]
        backbone = [
            parameter
            for parameter in original["params"]
            if id(parameter) not in projector_parameter_ids
        ]
        if backbone:
            group = dict(original)
            group["params"] = backbone
            replacement_groups.append(group)
        if projector:
            group = dict(original)
            group["params"] = projector
            group["lr"] = float(group["lr"]) * learning_rate_scale
            if "initial_lr" in group:
                group["initial_lr"] = (
                    float(group["initial_lr"]) * learning_rate_scale
                )
            replacement_groups.append(group)
            projector_parameters += len(projector)
    return replacement_groups, projector_parameters


class SemanticTokenTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        train_order: str = "shuffled",
        stage_lr_plan: dict[str, Any] | None = None,
        vision_projector_lr_scale: float = 1.0,
        vision_projector_parameter_ids: set[int] | None = None,
        **kwargs: Any,
    ) -> None:
        if train_order not in {"shuffled", "sequential"}:
            raise ValueError("unsupported train order")
        if stage_lr_plan is not None and train_order != "sequential":
            raise ValueError("stage learning-rate plan requires sequential stage order")
        self.train_order = train_order
        self.stage_lr_plan = stage_lr_plan
        self.vision_projector_lr_scale = float(vision_projector_lr_scale)
        self.vision_projector_parameter_ids = set(
            vision_projector_parameter_ids or set()
        )
        self._projector_lr_groups_applied = False
        if (
            self.vision_projector_lr_scale != 1.0
            and not self.vision_projector_parameter_ids
        ):
            raise ValueError("projector LR scaling requires projector parameters")
        super().__init__(*args, **kwargs)

    def create_optimizer(self) -> Any:
        """Split Trainer's established decay groups and scale projector LR only."""

        optimizer = super().create_optimizer()
        if self.vision_projector_lr_scale == 1.0:
            return optimizer
        if self._projector_lr_groups_applied:
            return optimizer
        if optimizer is None:
            raise RuntimeError("optimizer is absent while applying projector LR")

        replacement_groups, projector_parameters = (
            _split_optimizer_groups_for_projector(
                list(optimizer.param_groups),
                projector_parameter_ids=self.vision_projector_parameter_ids,
                learning_rate_scale=self.vision_projector_lr_scale,
            )
        )
        if projector_parameters != len(self.vision_projector_parameter_ids):
            raise RuntimeError(
                "optimizer projector parameter coverage diverged: "
                f"{projector_parameters} != "
                f"{len(self.vision_projector_parameter_ids)}"
            )
        optimizer.param_groups[:] = replacement_groups
        self._projector_lr_groups_applied = True
        if _is_main_process():
            print(
                {
                    "optimizer_lr_policy": "split_visual_projector",
                    "base_learning_rate": self.args.learning_rate,
                    "vision_projector_lr_scale": self.vision_projector_lr_scale,
                    "vision_projector_learning_rate": (
                        self.args.learning_rate
                        * self.vision_projector_lr_scale
                    ),
                    "projector_parameter_tensors": projector_parameters,
                    "optimizer_parameter_groups": len(replacement_groups),
                },
                flush=True,
            )
        return optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: Any = None
    ) -> Any:
        if self.stage_lr_plan is None:
            try:
                return super().create_scheduler(num_training_steps, optimizer)
            except TypeError:
                return super().create_scheduler(num_training_steps)
        if num_training_steps != int(self.stage_lr_plan["total_steps"]):
            raise RuntimeError(
                "stage LR plan and Trainer optimization steps diverged: "
                f"{self.stage_lr_plan['total_steps']} != {num_training_steps}"
            )
        if self.lr_scheduler is None:
            active_optimizer = self.optimizer if optimizer is None else optimizer
            if active_optimizer is None:
                raise RuntimeError("optimizer is absent while creating stage scheduler")
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                active_optimizer,
                lr_lambda=lambda step: _stage_lr_multiplier(
                    self.stage_lr_plan, int(step)
                ),
            )
        return self.lr_scheduler

    def _get_train_sampler(self, train_dataset: Any = None) -> Any:
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self.train_order == "sequential":
            return SequentialSampler(dataset)
        try:
            return super()._get_train_sampler(train_dataset)
        except TypeError:
            return super()._get_train_sampler()

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        del num_items_in_batch
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights")
        sample_loss_weight = inputs.pop("sample_loss_weight").float()
        outputs = model(**inputs)
        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous().float()
        loss = _sparse_weighted_token_loss(
            shift_logits, shift_labels, shift_weights, sample_loss_weight
        )
        return (loss, outputs) if return_outputs else loss


def _sparse_weighted_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    sample_loss_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute the same weighted CE without densifying every prompt token.

    Qwen emits logits for the visual tokens, prompt and assistant answer, while
    the semantic SFT contract supervises only assistant answer positions.  The
    previous implementation cast the complete ``batch x sequence x vocab``
    tensor to fp32 and then discarded all ignored positions.  Selecting active
    positions first is mathematically equivalent and avoids the transient
    13--20 GiB allocation seen on long examples.
    """

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise RuntimeError("logits and labels do not align")
    if loss_weights.shape != labels.shape:
        raise RuntimeError("token loss weights do not align with labels")
    if sample_loss_weight.shape != (logits.shape[0],):
        raise RuntimeError("sample loss weights do not align with the batch")

    active = labels.ne(-100) & loss_weights.gt(0)
    active_batch, _ = torch.nonzero(active, as_tuple=True)
    if active_batch.numel() == 0:
        raise RuntimeError("batch has no positively weighted assistant labels")
    active_logits = logits[active].float()
    active_labels = labels[active]
    active_weights = loss_weights[active].float()
    token_loss = torch.nn.functional.cross_entropy(
        active_logits, active_labels, reduction="none"
    )
    numerator = torch.zeros(
        logits.shape[0], device=logits.device, dtype=torch.float32
    ).scatter_add_(0, active_batch, token_loss * active_weights)
    denominator = torch.zeros_like(numerator).scatter_add_(
        0, active_batch, active_weights
    )
    per_sample_loss = numerator / denominator.clamp_min(1e-8)
    return (per_sample_loss * sample_loss_weight.float()).mean()


def inferred_root_manifest(release_root: Path) -> dict[str, Any]:
    path = release_root / "manifest.json"
    if not path.is_file():
        raise RuntimeError("sealed release manifest is absent")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_stage_lr_plan(
    path: Path,
    *,
    release_manifest: dict[str, Any],
    dataset_rows: int,
    global_batch: int,
    optimizer_learning_rate: float,
    warmup_ratio: float,
) -> dict[str, Any]:
    """Validate and compile row boundaries into optimizer-step boundaries."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != "iclr_video.stage_lr_plan.v1":
        raise RuntimeError("unsupported stage LR plan format")
    if int(raw.get("dataset_rows", -1)) != dataset_rows or global_batch <= 0:
        raise RuntimeError("stage LR plan dataset/global-batch contract diverged")
    if int(raw.get("global_batch", -1)) != global_batch:
        raise RuntimeError("stage LR plan global batch diverged")
    manifest_stages = (release_manifest.get("inputs") or {}).get(
        "curriculum_stages"
    )
    stages = raw.get("stages")
    if not isinstance(stages, list) or not stages or not isinstance(manifest_stages, list):
        raise RuntimeError("stage LR plan or release stages are absent")
    if len(stages) != len(manifest_stages):
        raise RuntimeError("stage LR plan count diverged from release")
    compiled = []
    previous_row = 0
    previous_step = 0
    maximum_lr = 0.0
    for index, (stage, manifest_stage) in enumerate(
        zip(stages, manifest_stages, strict=True)
    ):
        end_row = int(stage.get("end_row_exclusive", -1))
        learning_rate = float(stage.get("learning_rate", 0.0))
        if (
            int(stage.get("index", -1)) != index
            or int(manifest_stage.get("index", -1)) != index
            or str(stage.get("release_id")) != str(manifest_stage.get("release_id"))
            or end_row != int(manifest_stage.get("end_row_exclusive", -1))
            or end_row <= previous_row
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise RuntimeError("stage LR entry diverged from release manifest")
        end_step = math.ceil(end_row / global_batch)
        if end_step <= previous_step:
            raise RuntimeError("stage has no optimizer steps")
        stage_steps = end_step - previous_step
        warmup_steps = math.ceil(stage_steps * warmup_ratio)
        compiled.append(
            {
                "end_step_exclusive": end_step,
                "index": index,
                "learning_rate": learning_rate,
                "release_id": str(stage["release_id"]),
                "start_step": previous_step,
                "warmup_steps": warmup_steps,
            }
        )
        maximum_lr = max(maximum_lr, learning_rate)
        previous_row = end_row
        previous_step = end_step
    if previous_row != dataset_rows:
        raise RuntimeError("stage LR plan does not cover the complete dataset")
    if not math.isclose(
        maximum_lr, optimizer_learning_rate, rel_tol=1e-9, abs_tol=0.0
    ):
        raise RuntimeError("optimizer LR must equal the maximum stage LR")
    return {
        "base_learning_rate": optimizer_learning_rate,
        "format": raw["format"],
        "stages": compiled,
        "total_steps": previous_step,
    }


def _stage_lr_multiplier(plan: dict[str, Any], step: int) -> float:
    """Return a per-stage warmup/cosine multiplier for one optimizer step."""

    base_lr = float(plan["base_learning_rate"])
    for stage in plan["stages"]:
        start = int(stage["start_step"])
        end = int(stage["end_step_exclusive"])
        if step >= end:
            continue
        offset = max(step - start, 0)
        duration = end - start
        warmup = int(stage["warmup_steps"])
        scale = float(stage["learning_rate"]) / base_lr
        if warmup > 0 and offset < warmup:
            return scale * (offset / warmup)
        decay_steps = max(duration - warmup, 1)
        progress = min(max((offset - warmup) / decay_steps, 0.0), 1.0)
        return scale * 0.5 * (1.0 + math.cos(math.pi * progress))
    return 0.0


def _training_args(args: argparse.Namespace, has_eval: bool) -> Any:
    kwargs: Dict[str, Any] = {
        "output_dir": str(args.output),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "bf16": True,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "max_steps": args.max_steps,
        "gradient_checkpointing": True,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optim": "adamw_torch",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "tf32": True,
        "save_only_model": True,
        "report_to": "none",
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "seed": args.seed,
        "deepspeed": str(args.deepspeed),
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if args.save_steps >= 999999 and "save_strategy" in signature.parameters:
        kwargs["save_strategy"] = "no"
    if args.dataloader_num_workers > 0:
        if "dataloader_persistent_workers" in signature.parameters:
            kwargs["dataloader_persistent_workers"] = True
        if "dataloader_prefetch_factor" in signature.parameters:
            kwargs["dataloader_prefetch_factor"] = 2
    if "data_seed" in signature.parameters:
        kwargs["data_seed"] = args.seed
    evaluation_key = (
        "eval_strategy"
        if "eval_strategy" in signature.parameters
        else "evaluation_strategy"
    )
    kwargs[evaluation_key] = "steps" if has_eval else "no"
    if has_eval:
        kwargs["eval_steps"] = max(args.save_steps, args.logging_steps)
    return TrainingArguments(**kwargs)


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _logical_parameter_numel(parameter: torch.nn.Parameter) -> int:
    return int(getattr(parameter, "ds_numel", parameter.numel()))


if __name__ == "__main__":
    main()
