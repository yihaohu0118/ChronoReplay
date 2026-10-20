"""Task-blind prompt utilities for causal video multiple-choice learning.

This module deliberately lives outside the benchmark package.  Training and
evaluation can therefore share one deployment prompt without importing task
categories, benchmark annotations, or score-time metadata.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_LABEL_PREFIX = re.compile(r"^\s*([A-Z])\s*[.)]\s*(.*)$", re.DOTALL)
_DATA_IMAGE = re.compile(
    r"^data:image/[A-Za-z0-9.+\-]+;base64,",
    re.IGNORECASE,
)
IMAGE_OPTION_PLACEHOLDER = "[See option image labeled {label}]"
IMAGE_OPTIONS_INSTRUCTION = (
    "Note: The images that follow the video correspond to options "
    "{labels} in order. Choose the option whose image best matches the question."
)


def normalize_mcq_options(value: Any) -> dict[str, str]:
    """Normalize mapping/list options without consulting task metadata."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        options = {str(key).strip().upper(): str(item) for key, item in value.items()}
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        options: dict[str, str] = {}
        for index, item in enumerate(value):
            fallback = chr(ord("A") + index)
            text = str(item)
            match = _LABEL_PREFIX.match(text)
            label, answer = (
                (match.group(1), match.group(2)) if match else (fallback, text)
            )
            if label in options:
                raise ValueError(f"duplicate MCQ option label: {label}")
            options[label] = answer
    else:
        raise ValueError("MCQ options must be a mapping or a non-string sequence")
    if not 2 <= len(options) <= 8:
        raise ValueError(f"MCQ requires 2-8 options, got {len(options)}")
    expected = [chr(ord("A") + index) for index in range(len(options))]
    if sorted(options) != expected:
        raise ValueError(f"MCQ option labels must be contiguous from A: {sorted(options)}")
    return {label: options[label] for label in expected}


def build_direct_mcq_prompt(question: str, options: Any) -> str:
    """Build the single task-blind prompt used by direct VLM SFT and eval."""

    question = str(question).strip()
    if not question:
        raise ValueError("question must be non-empty")
    normalized = normalize_mcq_options(options)
    options_text = "\n".join(
        f"{label}. {answer}" for label, answer in normalized.items()
    )
    return f"""You are evaluating a video understanding task. Based on the video frames provided, answer the following multiple choice question.

Question: {question}

Options:
{options_text}

Instructions:
- Respond with ONLY the letter of your answer (e.g., "A" or "B").
- Do not include any explanation or additional text.

Your answer:"""


def is_image_option(value: Any) -> bool:
    """Identify an inline image option without depending on benchmark code."""

    return isinstance(value, str) and _DATA_IMAGE.match(value) is not None


def visible_mcq_options(options: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """Replace inline image payloads by actor-visible labeled placeholders."""

    normalized = normalize_mcq_options(options)
    labels = [label for label, value in normalized.items() if is_image_option(value)]
    visible = {
        label: (
            IMAGE_OPTION_PLACEHOLDER.format(label=label)
            if label in labels
            else value
        )
        for label, value in normalized.items()
    }
    return visible, labels


def build_multimodal_mcq_prompt(options_prompt: str, image_labels: Sequence[str]) -> str:
    """Append the deployment instruction for trailing labeled option images."""

    labels = [str(label) for label in image_labels]
    if not labels:
        return options_prompt
    return options_prompt + "\n\n" + IMAGE_OPTIONS_INSTRUCTION.format(
        labels=", ".join(labels)
    )
