"""Actor-visible identities for paired spatial evidence views."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Mapping

from PIL import Image

from .media import resolve_media_path


OPTION_LINE = re.compile(r"^\s*([A-H])\.\s+(.+?)\s*$")


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _messages(value: Any) -> list[Mapping[str, Any]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _message_texts(message: Mapping[str, Any]) -> list[str]:
    content = message.get("content")
    if hasattr(content, "tolist"):
        content = content.tolist()
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    return [
        str(item.get("text") or "")
        for item in content
        if isinstance(item, Mapping) and item.get("type") == "text"
    ]


def _message_parts(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    content = message.get("content")
    if hasattr(content, "tolist"):
        content = content.tolist()
    if not isinstance(content, list):
        return []
    return [item for item in content if isinstance(item, Mapping)]


def _canonical_image_identity(path_value: Any) -> str:
    """Hash decoded pixels, not compiler paths or container bytes."""

    path_text = _text(path_value)
    if not path_text:
        return ""
    path = resolve_media_path(path_text)
    if not path.is_file():
        raise RuntimeError(f"visible option image does not exist: {path}")
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            header = f"image-rgb-v1\0{rgb.width}\0{rgb.height}\0".encode("ascii")
            return hashlib.sha256(header + rgb.tobytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot decode visible option image: {path}") from exc


def _selected_option_image_identity(
    user_messages: list[Mapping[str, Any]],
    *,
    answer: str,
    option_labels: list[str],
) -> str:
    """Resolve an image option from actor-visible multimodal content order.

    ``image_role``/``option_label`` are used only as consistency checks.  The
    semantic identity itself is decoded from the same ordered pixels consumed
    by the model, so independently relabelled views still match.
    """

    parts = [part for message in user_messages for part in _message_parts(message)]
    explicit = [
        part
        for part in parts
        if _text(part.get("type")) == "image"
        and _text(part.get("image_role")) == "option"
    ]
    if explicit:
        if len(explicit) != len(option_labels):
            raise RuntimeError(
                "visible image-option count disagrees with MCQ options: "
                f"images={len(explicit)} labels={len(option_labels)}"
            )
        declared = [_text(part.get("option_label")) for part in explicit]
        if any(declared) and declared != option_labels:
            raise RuntimeError(
                "visible image-option labels disagree with content order: "
                f"declared={declared} expected={option_labels}"
            )
        selected = explicit[option_labels.index(answer)]
        return _canonical_image_identity(selected.get("image"))

    # Older actor-visible records may omit compiler annotations.  Their prompt
    # contract says the final N images are options A.. in order.
    images = [
        part
        for part in parts
        if _text(part.get("type")) == "image" and _text(part.get("image"))
    ]
    if len(images) >= len(option_labels):
        selected = images[-len(option_labels) + option_labels.index(answer)]
        return _canonical_image_identity(selected.get("image"))
    return ""


def visible_selected_option_identity(messages: Any) -> str:
    """Hash the selected semantic option using actor-visible content only.

    Dense and context views may relabel the same answer independently. The
    release removes compiler-private trajectory fields, so this reconstructs
    the invariant identity from the ordinary MCQ prompt and its one-letter
    assistant target. It never reads task labels or benchmark annotations.
    """

    rows = _messages(messages)
    user_messages = [
        message
        for message in rows
        if str(message.get("role") or "") == "user"
    ]
    user_text = "\n".join(
        text
        for message in user_messages
        for text in _message_texts(message)
    )
    assistant_texts = [
        text.strip()
        for message in rows
        if str(message.get("role") or "") == "assistant"
        for text in _message_texts(message)
    ]
    if len(assistant_texts) != 1 or not re.fullmatch(r"[A-H]", assistant_texts[0]):
        return ""
    options: dict[str, str] = {}
    for line in user_text.splitlines():
        match = OPTION_LINE.fullmatch(line)
        if not match:
            continue
        label, option = match.groups()
        if label in options:
            raise RuntimeError(f"duplicate visible MCQ option: {label}")
        options[label] = unicodedata.normalize("NFKC", option).strip()
    expected = [chr(ord("A") + index) for index in range(len(options))]
    if not 2 <= len(options) <= 8 or list(options) != expected:
        return ""
    answer = assistant_texts[0]
    selected = options.get(answer, "")
    if not selected:
        return ""
    if re.fullmatch(r"\[\s*see option image labeled [A-H]\s*\]", selected, re.I):
        return _selected_option_image_identity(
            user_messages,
            answer=answer,
            option_labels=expected,
        )
    normalized = unicodedata.normalize("NFKC", selected).strip()
    return hashlib.sha256(("text-v1\0" + normalized).encode("utf-8")).hexdigest()


def correct_option_identity(row: Mapping[str, Any]) -> str:
    """Return a source-certified or actor-visible semantic answer identity."""

    value = row.get("trajectory_audit")
    if isinstance(value, Mapping):
        audit = value
    else:
        encoded = _text(value)
        if not encoded:
            return visible_selected_option_identity(row.get("messages"))
        try:
            decoded = json.loads(encoded)
        except (TypeError, ValueError):
            return visible_selected_option_identity(row.get("messages"))
        audit = decoded if isinstance(decoded, Mapping) else {}
    return _text(audit.get("correct_option_identity_sha256")) or (
        visible_selected_option_identity(row.get("messages"))
    )


def certified_nested_evidence_view(row: Mapping[str, Any]) -> str:
    """Return a view only for containment plus the intervention certificate.

    A pair name is not evidence of nesting.  The compiler certificate must
    bind the visible frame indices, prove that the minimal witness is a subset
    of the causal context, and bind source-hidden dual-view recovery with a
    failed question-only control. This field is optimizer-side provenance and
    never enters the model prompt.
    """

    value = row.get("data_certificate")
    if isinstance(value, Mapping):
        certificate = value
    else:
        encoded = _text(value)
        if not encoded:
            return ""
        try:
            decoded = json.loads(encoded)
        except (TypeError, ValueError):
            return ""
        certificate = decoded if isinstance(decoded, Mapping) else {}
    view = _text(certificate.get("view"))
    source_view = _text(row.get("source_view"))
    if view not in {"dense", "context"} or source_view not in {
        view,
        f"evidence_{view}",
    }:
        return ""
    required = {
        "minimal_witness_is_context_subset": True,
        "source_answer_unchanged": True,
        "teacher_defines_label": False,
        "teacher_response_is_training_target": False,
        "benchmark_instances_used": False,
    }
    if any(certificate.get(key) is not expected for key, expected in required.items()):
        return ""
    minimal = certificate.get("minimal_witness_indices")
    context = certificate.get("context_indices")
    visible = certificate.get("visible_indices")
    if not all(isinstance(value, list) and value for value in (minimal, context, visible)):
        return ""
    try:
        minimal_set = {int(item) for item in minimal}
        context_set = {int(item) for item in context}
        visible_set = {int(item) for item in visible}
    except (TypeError, ValueError):
        return ""
    if (
        len(minimal_set) != len(minimal)
        or len(context_set) != len(context)
        or len(visible_set) != len(visible)
        or not minimal_set.issubset(context_set)
        or visible_set != (minimal_set if view == "dense" else context_set)
    ):
        return ""
    recoverability = certificate.get("answer_recoverability")
    if not isinstance(recoverability, Mapping):
        return ""
    recoverability_required = {
        "format": "iclr_spatial.answer_recoverability_certificate.v2",
        "view": view,
        "view_answer_matches_source_gt": True,
        "view_visually_sufficient": True,
        "both_views_admitted": True,
        "question_only_control_passes": False,
        "visual_necessity_observed": True,
        "source_answer_sent_to_solver": False,
        "question_only_control_contains_images": False,
        "teacher_defines_label": False,
        "teacher_response_is_training_target": False,
    }
    if any(
        recoverability.get(key) != expected
        for key, expected in recoverability_required.items()
    ):
        return ""
    try:
        confidence = float(recoverability.get("view_confidence"))
        threshold = float(recoverability.get("min_confidence"))
    except (TypeError, ValueError):
        return ""
    if not 0 <= threshold <= confidence <= 1:
        return ""
    for key in (
        "verification_record_sha256",
        "verification_parent_sha256",
        "source_target_sha256",
        "solver_models_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", _text(recoverability.get(key))) is None:
            return ""
    return view


def certified_frontier_representation_view(row: Mapping[str, Any]) -> str:
    """Validate a source-utility frontier certificate for representation replay.

    These pairs are not claimed to be minimal-witness containment orbits. They
    form a separate, stricter capability frontier: both visible views are
    mastered by the mentor, the student remains weak, visual/temporal
    grounding is positive, and no benchmark/category field selected the row.
    """

    value = row.get("representation_orbit_certificate")
    if isinstance(value, Mapping):
        certificate = value
    else:
        encoded = _text(value)
        if not encoded:
            return ""
        try:
            decoded = json.loads(encoded)
        except (TypeError, ValueError):
            return ""
        certificate = decoded if isinstance(decoded, Mapping) else {}
    view = _text(certificate.get("view"))
    source_view = _text(row.get("source_view"))
    if view not in {"dense", "context"} or source_view not in {
        view,
        f"evidence_{view}",
    }:
        return ""
    if certificate.get("format") != (
        "iclr_spatial.frontier_representation_orbit_certificate.v1"
    ):
        return ""
    source_identity = _text(certificate.get("source_identity"))
    if source_identity != _text(row.get("semantic_source_sha256")):
        return ""
    required = {
        "mentor_full_correct_views": 2,
        "views": 2,
        "neutral_failure_required": True,
        "benchmark_instances_used": False,
        "benchmark_category_selection": False,
        "task_or_category_selection": False,
        "teacher_defines_label": False,
        "teacher_response_is_training_target": False,
    }
    if any(certificate.get(key) != expected for key, expected in required.items()):
        return ""
    try:
        full_margin = float(certificate.get("mentor_mean_full_margin"))
        grounding = float(certificate.get("mentor_min_grounding_advantage"))
        temporal = float(certificate.get("mentor_mean_temporal_advantage"))
        student_error = float(certificate.get("student_incorrect_view_fraction"))
    except (TypeError, ValueError):
        return ""
    if not (
        full_margin > 0
        and grounding > 0
        and temporal >= 0
        and 0 < student_error <= 1
    ):
        return ""
    return view


def certified_feature_evidence_orbit_role(feature: Mapping[str, Any]) -> int:
    """Map a dataset-validated optimizer-only view to its orbit role."""

    view = _text(feature.get("certified_evidence_orbit_view"))
    if view == "dense":
        return 0
    if view == "context":
        return 1
    return -1
