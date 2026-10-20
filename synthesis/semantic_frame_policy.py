"""Visible-language temporal routing for causal video questions.

The policy distinguishes questions about the immediately visible state from
questions that require accumulated history.  It consumes only the user-visible
question string; benchmark categories, answers, predictions, and confidence
are not inputs.
"""

from __future__ import annotations

import os
import re
from typing import Mapping

from synthesis.spatial_query_compiler import compile_spatial_query


_LOCAL_SCOPE = re.compile(
    r"(?:\bcurrent(?:ly)?\b|\bat this moment\b|\bright now\b|"
    r"\bat the moment\b|\bthis segment\b|\bin this clip\b)",
    re.IGNORECASE,
)
_HISTORY_SCOPE = re.compile(
    r"(?:\bsame (?:place|scene)\b|\breferenced moment\b|\blast time\b|"
    r"\bhow many times\b|\bso far\b|\bbefore or after\b|"
    r"\bvisited before\b|\bshown for a longer\b|\bpassed by\b|"
    r"\bin the video\b|\bfull video\b|\bthroughout\b|\broute\b|\bpath\b|\bmap\b|"
    r"\breturn\w*\b|\brevisit\w*\b)",
    re.IGNORECASE,
)
_EGO_MOTION_SCOPE = re.compile(
    r"(?:\bcamera\b.{0,48}\b(?:yawing|rotating|moving (?:backward|forward))\b|"
    r"\b(?:yawing|rotating)\b.{0,48}\bcamera\b)",
    re.IGNORECASE,
)

_CHRONOLOGICAL_SCOPE = re.compile(
    r"(?:\bchronological (?:order|sequence)\b|\bcorrect (?:temporal )?order\b|"
    r"\bin what order\b|\b(?:tour |visit )?sequence\b|\bbefore or after\b|"
    r"\bimmediately (?:before|after|beforehand)\b|"
    r"\b(?:first|second|next|last)\b.{0,80}\b"
    r"(?:visit\w*|enter\w*|leave\w*|pass\w*|shown|seen|encounter\w*)\b|"
    r"\b(?:visit\w*|enter\w*|leave\w*|pass\w*|go|went|shown|seen|encounter\w*)\b"
    r".{0,80}\b(?:before|after|next|first|second|last)\b|"
    r"\bwhat (?:came|happened|was shown) (?:first|next|last)\b)",
    re.IGNORECASE,
)
_TEMPORAL_DERIVATIVE_SCOPE = re.compile(
    r"(?:\b(?:opening|closing|rotating|stationary|scrolling|ascending|descending)\b|"
    r"\b(?:jumping|moving|walking) (?:upward|downward|upstairs|downstairs|"
    r"forward|backward)\b|\byawing (?:left|right)\b)",
    re.IGNORECASE,
)
_STATIC_LAYOUT_SCOPE = re.compile(
    r"(?:\btopolog\w*\b|\bspatial (?:arrangement|structure|relationship)\b|"
    r"\bprimary connection\b|\b(?:space|room|area)\b.{0,64}\bconnect\w*\b|"
    r"\b(?:shortest|most reasonable) route\b|\bplausible detour\b|"
    r"\bbased on the observed layout\b)",
    re.IGNORECASE,
)

_EXPLICIT_ENDPOINT_REFERENCE_SCOPE = re.compile(
    r"(?:\bturn\s*\d+\b|\b(?:the\s+)?\d+\s*(?:st|nd|rd|th)?\s*turn\b|"
    r"\bprevious (?:position|orientation|pose|location|view|turn)\b|"
    r"\bas a reference\b|\brelative to (?:that|the) (?:position|orientation|pose|location)\b|"
    r"\bfrom where (?:you|i|we) (?:were|was)\b)",
    re.IGNORECASE,
)


def select_causal_frame_policy(question: str) -> tuple[str, str]:
    """Return a frame policy and an auditable visible-semantic reason."""

    text = " ".join(str(question or "").split())
    if _HISTORY_SCOPE.search(text):
        return "uniform", "explicit_historical_scope"
    # Instantaneous ego motion is defined by a temporal derivative rather than
    # a single local scene state. Preserve the longer causal flow context even
    # when the wording also contains “current/currently”.
    if _EGO_MOTION_SCOPE.search(text):
        return "uniform", "explicit_ego_motion_scope"
    if _LOCAL_SCOPE.search(text):
        return "recent", "explicit_current_scope"
    return "uniform", "unspecified_or_global_scope"


def select_temporal_support_policy(question: str) -> tuple[str, str]:
    """Select one causal-support horizon from visible semantics alone.

    This benchmark-independent operator applies to any ordered observation
    stream.  Video evaluators instantiate the selected horizon over frames;
    interactive evaluators instantiate it over observation turns.  Static
    local-state questions receive recent support.  Temporal derivatives and
    event-order queries receive logarithmic support.  Other historical/global
    queries retain the full causal prefix.
    """

    text = " ".join(str(question or "").split())
    # An explicitly named historical endpoint dominates incidental words such
    # as "current".  Coordinate differences and pose rotations require both
    # endpoints even when the requested quantity is phrased at the current
    # time.
    if _EXPLICIT_ENDPOINT_REFERENCE_SCOPE.search(text):
        return "uniform", "explicit_endpoint_reference_scope"
    # A named entity expressed relative to the current observer is not a
    # purely local appearance query. It requires the entity's persistent
    # world location together with the current observer pose. Bind this to
    # the same benchmark-independent signed-heading operator used by the
    # deterministic executor rather than letting incidental words such as
    # “now” collapse the support to the latest frame/turn.
    plan = compile_spatial_query(text)
    if plan.operator == "query_relative_pose":
        return "uniform", "persistent_relative_pose_operator"
    if _TEMPORAL_DERIVATIVE_SCOPE.search(text):
        return "log_decay", "explicit_temporal_derivative_scope"
    # Static connectivity language can contain incidental temporal words such
    # as “after entering” or “first connection”. Unless the question asks for
    # the current view, this is a layout query and needs broad prefix coverage.
    if _STATIC_LAYOUT_SCOPE.search(text) and not _LOCAL_SCOPE.search(text):
        return "uniform", "explicit_static_layout_scope"
    if _CHRONOLOGICAL_SCOPE.search(text):
        return "log_decay", "explicit_chronological_scope"
    if _HISTORY_SCOPE.search(text):
        return "uniform", "explicit_historical_scope"
    if _LOCAL_SCOPE.search(text):
        return "recent", "explicit_current_scope"
    return "uniform", "unspecified_or_global_scope"


def select_adaptive_causal_frame_policy(question: str) -> tuple[str, str]:
    """Backward-compatible frame-domain entry for temporal support."""

    if os.getenv("OVOS_FROZEN_TEMPORAL_SUPPORT_20260824", "0") == "1":
        return select_frozen_adaptive_causal_frame_policy_20260824(question)
    return select_temporal_support_policy(question)


def select_frozen_adaptive_causal_frame_policy_20260824(
    question: str,
) -> tuple[str, str]:
    """Exact query-only selector used by the frozen 2026-08-24 OVO policy."""

    text = " ".join(str(question or "").split())
    if _TEMPORAL_DERIVATIVE_SCOPE.search(text):
        return "log_decay", "explicit_temporal_derivative_scope"
    if _STATIC_LAYOUT_SCOPE.search(text) and not _LOCAL_SCOPE.search(text):
        return "uniform", "explicit_static_layout_scope"
    if _CHRONOLOGICAL_SCOPE.search(text):
        return "log_decay", "explicit_chronological_scope"
    if _HISTORY_SCOPE.search(text):
        return "uniform", "explicit_historical_scope"
    if _LOCAL_SCOPE.search(text):
        return "recent", "explicit_current_scope"
    return "uniform", "unspecified_or_global_scope"


def temporal_support_turn_indices(
    total_turns: int,
    policy: str,
    *,
    recent_turns: int = 1,
) -> list[int]:
    """Instantiate the shared support horizon over ordered interaction turns.

    ``uniform`` keeps the complete prefix. ``recent`` keeps the bounded tail.
    ``log_decay`` keeps the current turn and power-of-two historical offsets,
    including the causal origin.  The operation is deterministic and depends
    on no benchmark metadata, answer, prediction, score, or confidence.
    """

    if total_turns < 1:
        raise ValueError("total_turns must be positive")
    if recent_turns < 1:
        raise ValueError("recent_turns must be positive")
    if policy == "uniform":
        return list(range(total_turns))
    if policy == "recent":
        return list(range(max(0, total_turns - recent_turns), total_turns))
    if policy != "log_decay":
        raise ValueError(f"unsupported temporal support policy: {policy}")
    current = total_turns - 1
    selected = {0, current}
    offset = 1
    while current - offset >= 0:
        selected.add(current - offset)
        offset *= 2
    return sorted(selected)


def select_temporal_dereference_frame_policy(
    question: str, options: Mapping[str, str] | None = None
) -> tuple[str, str]:
    """Add only the independently validated temporal-dereference executor.

    All non-applicable questions retain the frozen semantic-adaptive policy.
    The query compiler consumes visible language only.  In particular, this
    selector has no task/category, answer, annotation, prediction, or score
    input.
    """

    plan = compile_spatial_query(question, options)
    if plan.operator == "compare_referenced_state":
        return "referenced_pair", "validated_explicit_temporal_dereference"
    return select_adaptive_causal_frame_policy(question)


def select_uniform_temporal_dereference_frame_policy(
    question: str, options: Mapping[str, str] | None = None
) -> tuple[str, str]:
    """Preserve uniform evidence except for an explicit timestamp query."""

    plan = compile_spatial_query(question, options)
    if plan.operator == "compare_referenced_state":
        return "referenced_pair", "validated_explicit_temporal_dereference"
    return "uniform", "uniform_fallback_no_temporal_dereference"
