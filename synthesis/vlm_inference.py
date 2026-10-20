"""Shared model-only VLM inference helpers.

The release evaluator imports this module directly. Adapter loading remains
an optional compatibility feature for historical experiments and is never a
dependency of the plain-checkpoint path.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Mapping

import torch

_PROCESSOR_LOCK = threading.Lock()


OPTION_LABELS = tuple("ABCDEFG")


def _managed_cuda_device() -> int:
    """Select this worker's logical GPU without changing process visibility."""

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return torch.cuda.current_device()
    try:
        device = int(local_rank)
    except ValueError as exc:
        raise RuntimeError(f"invalid LOCAL_RANK={local_rank!r}") from exc
    count = torch.cuda.device_count()
    if not 0 <= device < count:
        raise RuntimeError(f"LOCAL_RANK={device} outside {count} visible CUDA devices")
    torch.cuda.set_device(device)
    return device


def normalize_gt(raw) -> str:
    """Flatten nested parquet answers to one normalized option letter."""

    answer = raw
    while isinstance(answer, (list, tuple)) and len(answer) == 1:
        answer = answer[0]
    return str(answer).upper().strip()


def load_model(
    model_path: str,
    adapter_path: str | None = None,
    processor_path: str | None = None,
):
    """Materialize a Qwen3-VL checkpoint on the declared CUDA device.

    ``processor_path`` is deliberately independent from ``model_path``.  A
    training checkpoint may persist its training-time pixel budget in
    ``preprocessor_config.json``; evaluation must be able to pin the official
    base processor so two checkpoints see identical pixels.
    """

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "model evaluation requires the 'transformers' package"
        ) from exc

    processor = AutoProcessor.from_pretrained(
        processor_path or model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("multimodal policy evaluation requires CUDA")

    # A single-worker evaluation uses its current device; a multi-worker direct
    # launcher uses LOCAL_RANK to select one logical device from the caller's
    # fixed visible set.
    device = _managed_cuda_device()
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "loading an adapter requires the optional 'peft' package"
            ) from exc
        model = PeftModel.from_pretrained(
            model, adapter_path, autocast_adapter_dtype=False
        )

    model.config.use_cache = True
    model.eval()
    return model, processor


def build_generate_inputs(processor, images: list, prompt: str) -> dict[str, Any]:
    """Run the CPU processor only. GPU generate is a separate call.

    Prefetching this dict behind the previous ``model.generate`` is what keeps
    a 128-frame H100 from sitting at 0% utilization between items. Content,
    padding, and tokenization are unchanged from :func:`predict`.
    """

    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    processor_kwargs = {"text": rendered, "return_tensors": "pt"}
    if images:
        processor_kwargs["images"] = images
    with _PROCESSOR_LOCK:
        return dict(processor(**processor_kwargs))


def _to_model_device(inputs: Mapping[str, Any], device) -> dict[str, Any]:
    """Host-to-device copy that can overlap the next CPU processor call."""

    moved: dict[str, Any] = {}
    for key, value in dict(inputs).items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        tensor = value
        if getattr(device, "type", None) == "cuda" and tensor.device.type == "cpu":
            try:
                tensor = tensor.pin_memory()
            except RuntimeError:
                pass
            moved[key] = tensor.to(device, non_blocking=True)
        else:
            moved[key] = tensor.to(device)
    return moved


@torch.inference_mode()
def generate_from_inputs(
    model,
    processor,
    inputs: Mapping[str, Any],
    max_new_tokens: int = 128,
) -> str:
    """Deterministic generate from already-processed tensors."""

    device_inputs = _to_model_device(inputs, model.device)
    output = model.generate(
        **device_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    prompt_length = device_inputs["input_ids"].shape[1]
    with _PROCESSOR_LOCK:
        text = processor.batch_decode(
            output[:, prompt_length:],
            skip_special_tokens=True,
        )[0]
    return text.strip()


@torch.inference_mode()
def predict(
    model,
    processor,
    images: list,
    prompt: str,
    max_new_tokens: int = 128,
) -> str:
    """Run deterministic model-only generation on an ordered image list."""

    return generate_from_inputs(
        model,
        processor,
        build_generate_inputs(processor, images, prompt),
        max_new_tokens=max_new_tokens,
    )


@torch.inference_mode()
def predict_conversation(
    model,
    processor,
    messages: list[Mapping[str, Any]],
    max_new_tokens: int = 128,
) -> str:
    """Continue one multimodal conversation with ordered image blocks.

    This is the sequential counterpart of :func:`predict`.  Images may occur
    in multiple user turns; the function collects them in transcript order and
    keeps every prior assistant action in context.  It performs no tool logic
    and never mutates the supplied conversation.
    """

    images = []
    copied: list[dict[str, Any]] = []
    for raw_message in messages:
        content = []
        for raw_part in raw_message.get("content", []):
            # Arrow/Parquet materializes heterogeneous content structs as a
            # union and may add null-valued image/video keys to text parts.
            # Qwen's template treats key presence as a vision placeholder,
            # so strip only null union fields before rendering.
            part = {
                str(key): value
                for key, value in dict(raw_part).items()
                if value is not None
            }
            if part.get("type") == "image":
                image = part.get("image")
                if image is None:
                    raise ValueError("conversation image part has no image")
                images.append(image)
            content.append(part)
        copied.append({"role": str(raw_message.get("role") or ""), "content": content})
    rendered = processor.apply_chat_template(
        copied, tokenize=False, add_generation_prompt=True
    )
    kwargs = {"text": rendered, "return_tensors": "pt"}
    if images:
        kwargs["images"] = images
    inputs = processor(**kwargs)
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    text = processor.batch_decode(
        output[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )[0]
    return text.strip()


@torch.inference_mode()
def predict_native_video(
    model,
    processor,
    frames: list,
    prompt: str,
    *,
    option_images: list | None = None,
    fps: float = 2.0,
    max_new_tokens: int = 128,
) -> str:
    """Run one ordered frame sequence through Qwen's native video path.

    The frame set and question are unchanged; only the processor
    serialization differs from :func:`predict`. Option images remain
    independent image blocks, matching the upstream OVO-S Qwen wrapper.
    """

    if not frames:
        raise ValueError("native-video inference requires at least one frame")
    if fps <= 0:
        raise ValueError("native-video fps must be positive")
    # qwen-vl-utils distinguishes encoded-video inputs from already decoded
    # frame lists.  For a frame list it reads ``sample_fps`` and ``raw_fps``;
    # the superficially similar ``fps`` key is ignored.  Set both explicitly
    # so Qwen3-VL's prompt timestamps (frame_index / raw_fps) follow the
    # declared time base instead of silently falling back to 2 FPS.
    content = [{
        "type": "video",
        "video": frames,
        "sample_fps": float(fps),
        "raw_fps": float(fps),
    }]
    content.extend(
        {"type": "image", "image": image} for image in (option_images or [])
    )
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    from qwen_vl_utils import process_vision_info

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_metadata = None
    if videos is not None:
        videos, video_metadata = zip(*videos)
        videos, video_metadata = list(videos), list(video_metadata)
    processor_kwargs = {"text": rendered, "return_tensors": "pt"}
    if images:
        processor_kwargs["images"] = images
    if videos:
        processor_kwargs["videos"] = videos
        processor_kwargs["video_metadata"] = video_metadata
        processor_kwargs.update(video_kwargs)
    inputs = processor(**processor_kwargs)
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    text = processor.batch_decode(
        output[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )[0]
    return text.strip()


@torch.inference_mode()
def predict_batch(
    model,
    processor,
    image_batches: list[list],
    prompts: list[str],
    max_new_tokens: int = 128,
) -> list[str]:
    """Batch deterministic inference without changing any per-item content."""

    if not image_batches or len(image_batches) != len(prompts):
        raise ValueError("image_batches and prompts must have equal positive length")
    conversations = []
    for images, prompt in zip(image_batches, prompts, strict=True):
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        conversations.append([{"role": "user", "content": content}])
    rendered = [
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in conversations
    ]
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        # Decoder-only generation must left-pad heterogeneous prompt lengths.
        # Right padding makes generation begin after padding rather than after
        # each real prompt and changes answers even with greedy decoding.
        tokenizer.padding_side = "left"
    inputs = processor(
        text=rendered,
        images=image_batches,
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    texts = processor.batch_decode(
        output[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    return [text.strip() for text in texts]


def _single_token_option_ids(processor, labels: list[str]) -> list[int]:
    """Resolve answer labels to their exact one-token decoder IDs."""

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("processor has no tokenizer for option scoring")
    token_ids: list[int] = []
    for raw_label in labels:
        label = str(raw_label).strip().upper()
        if label not in OPTION_LABELS:
            raise ValueError(f"unsupported option label: {raw_label!r}")
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"option label {label!r} is not a single token: {encoded}")
        token_ids.append(int(encoded[0]))
    return token_ids


@torch.inference_mode()
def predict_option_logprobs_batch(
    model,
    processor,
    image_batches: list[list],
    prompts: list[str],
    option_labels: list[list[str]],
) -> list[dict[str, float]]:
    """Score visible MCQ labels at the first decoder position.

    Scores are normalized only over the labels that are valid for each item.
    This answer-blind path is useful for aggregating semantic option evidence
    across fixed label permutations without generating explanations.
    """

    if not image_batches or not (
        len(image_batches) == len(prompts) == len(option_labels)
    ):
        raise ValueError("images, prompts, and option_labels must have equal positive length")
    conversations = []
    for images, prompt in zip(image_batches, prompts, strict=True):
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        conversations.append([{"role": "user", "content": content}])
    rendered = [
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in conversations
    ]
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    inputs = processor(
        text=rendered,
        images=image_batches,
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    # Qwen3-VL accepts ``logits_to_keep`` and can avoid materializing a
    # sequence-length by vocabulary tensor when only the next token is scored.
    next_token_logits = model(
        **inputs, use_cache=False, logits_to_keep=1
    ).logits[:, -1, :].float()
    output: list[dict[str, float]] = []
    for row_index, raw_labels in enumerate(option_labels):
        labels = [str(label).strip().upper() for label in raw_labels]
        if not labels or len(labels) != len(set(labels)):
            raise ValueError(f"option labels must be non-empty and unique: {labels}")
        token_ids = _single_token_option_ids(processor, labels)
        values = torch.log_softmax(next_token_logits[row_index, token_ids], dim=-1)
        output.append(
            {label: float(value) for label, value in zip(labels, values.tolist(), strict=True)}
        )
    return output
