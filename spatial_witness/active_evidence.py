"""Budgeted, causal tool protocol for active spatial evidence seeking.

The protocol is deliberately smaller than the historical free-form agent.  A
controller sees the question, options, query time, and an initial chronological
overview.  It may issue at most two generic evidence requests before answering.
Every requested frame is charged against the same 128-frame budget used by the
OVO-S uniform baseline, including overlapping requests.

This module plans timestamps only.  It does not use benchmark task IDs,
evidence annotations, answers, model confidence, or an external model.  A
benchmark adapter is responsible for decoding the planned timestamps and for
presenting the returned frames to the same SFT model that chose the action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Mapping, Sequence


PROTOCOL = "active_spatial_evidence128_v1"
FRAME_BUDGET = 128
OVERVIEW_FRAMES = 32
MAX_TOOL_CALLS = 2
SYSTEM_PROMPT = (
    "Act as a causal spatial evidence seeker. Inspect only the visible video "
    "prefix, use the smallest useful generic evidence action, and answer with "
    "one JSON action. Never infer from benchmark metadata or option position."
)
ALLOWED_TOOLS = frozenset(
    {"inspect_window", "compare_moments", "trace_motion", "answer"}
)
ANSWER_LABELS = frozenset("ABCDEFGH")


class ProtocolError(ValueError):
    """Raised when an action violates the causal or budget contract."""


@dataclass(frozen=True)
class EvidenceRequest:
    action: str
    nframes: int = 0
    center_fraction: float | None = None
    radius_fraction: float | None = None
    moment_fractions: tuple[float, float] | None = None
    interval_fractions: tuple[float, float] | None = None
    answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class ActiveEvidenceState:
    """Mutable accounting state for one query."""

    query_time: float
    frame_budget: int = FRAME_BUDGET
    overview_frames: int = OVERVIEW_FRAMES
    max_tool_calls: int = MAX_TOOL_CALLS
    tool_calls: list[EvidenceRequest] = field(default_factory=list)
    frames_charged: int = 0
    answered: bool = False

    def __post_init__(self) -> None:
        # A streaming request can legally arrive at the initial video instant.
        # In that boundary state the only causal observation is timestamp zero;
        # rejecting it turns a valid state into an evaluator failure.
        if not math.isfinite(self.query_time) or self.query_time < 0:
            raise ProtocolError("query_time must be finite and non-negative")
        if not 0 < self.overview_frames <= self.frame_budget:
            raise ProtocolError("invalid overview frame allocation")
        if self.max_tool_calls <= 0:
            raise ProtocolError("max_tool_calls must be positive")
        self.frames_charged = self.overview_frames

    @property
    def frames_remaining(self) -> int:
        return self.frame_budget - self.frames_charged

    def public_state(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "query_time": self.query_time,
            "frame_budget": self.frame_budget,
            "frames_charged": self.frames_charged,
            "frames_remaining": self.frames_remaining,
            "tool_calls_used": len(self.tool_calls),
            "tool_calls_remaining": self.max_tool_calls - len(self.tool_calls),
        }

    def execute(self, request: EvidenceRequest) -> list[float]:
        """Validate an action, charge it, and return causal frame timestamps."""

        if self.answered:
            raise ProtocolError("no action is allowed after answer")
        validate_request(request)
        if request.action == "answer":
            self.answered = True
            return []
        if len(self.tool_calls) >= self.max_tool_calls:
            raise ProtocolError("tool-call limit exceeded")
        if request.nframes > self.frames_remaining:
            raise ProtocolError("frame budget exceeded")
        times = request_times(request, query_time=self.query_time)
        # Charge the request, not the deduplicated return, so repeated or
        # overlapping calls cannot manufacture extra observations for free.
        self.frames_charged += request.nframes
        self.tool_calls.append(request)
        return times


def _finite_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ProtocolError(f"{name} must be in [0, 1]")
    return result


def _positive_fraction(value: Any, name: str) -> float:
    result = _finite_fraction(value, name)
    if result <= 0:
        raise ProtocolError(f"{name} must be positive")
    return result


def _frame_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError("nframes must be an integer")
    if not 8 <= value <= 48 or value % 8:
        raise ProtocolError("nframes must be a multiple of 8 in [8, 48]")
    return value


def parse_action(value: str | Mapping[str, Any]) -> EvidenceRequest:
    """Parse one flat controller action and reject unknown fields."""

    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise ProtocolError("action is not valid JSON") from error
    else:
        payload = dict(value)
    if not isinstance(payload, dict):
        raise ProtocolError("action must be one JSON object")
    action = str(payload.get("action") or "")
    if action not in ALLOWED_TOOLS:
        raise ProtocolError(f"unsupported action: {action!r}")
    allowed_fields = {
        "action",
        "nframes",
        "center_fraction",
        "radius_fraction",
        "moment_fractions",
        "interval_fractions",
        "answer",
    }
    unknown = sorted(set(payload) - allowed_fields)
    if unknown:
        raise ProtocolError(f"unknown action fields: {unknown}")

    def pair(name: str) -> tuple[float, float] | None:
        raw = payload.get(name)
        if raw is None:
            return None
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ProtocolError(f"{name} must contain exactly two fractions")
        return (
            _finite_fraction(raw[0], f"{name}[0]"),
            _finite_fraction(raw[1], f"{name}[1]"),
        )

    request = EvidenceRequest(
        action=action,
        nframes=(
            _frame_count(payload.get("nframes")) if action != "answer" else 0
        ),
        center_fraction=(
            _finite_fraction(payload["center_fraction"], "center_fraction")
            if "center_fraction" in payload
            else None
        ),
        radius_fraction=(
            _positive_fraction(payload["radius_fraction"], "radius_fraction")
            if "radius_fraction" in payload
            else None
        ),
        moment_fractions=pair("moment_fractions"),
        interval_fractions=pair("interval_fractions"),
        answer=(str(payload.get("answer") or "").strip().upper() or None),
    )
    validate_request(request)
    return request


def validate_request(request: EvidenceRequest) -> None:
    if request.action not in ALLOWED_TOOLS:
        raise ProtocolError(f"unsupported action: {request.action!r}")
    if request.action == "answer":
        if request.answer not in ANSWER_LABELS:
            raise ProtocolError("answer must be one letter A-H")
        if any(
            value is not None
            for value in (
                request.center_fraction,
                request.radius_fraction,
                request.moment_fractions,
                request.interval_fractions,
            )
        ) or request.nframes:
            raise ProtocolError("answer action contains tool arguments")
        return
    _frame_count(request.nframes)
    if request.answer is not None:
        raise ProtocolError("tool request cannot contain an answer")
    if request.action == "inspect_window":
        if request.center_fraction is None or request.radius_fraction is None:
            raise ProtocolError("inspect_window requires center and radius")
        if request.moment_fractions is not None or request.interval_fractions is not None:
            raise ProtocolError("inspect_window contains unrelated arguments")
    elif request.action == "compare_moments":
        if request.moment_fractions is None or request.radius_fraction is None:
            raise ProtocolError("compare_moments requires moments and radius")
        if request.moment_fractions[0] == request.moment_fractions[1]:
            raise ProtocolError("compare_moments requires distinct moments")
        if request.center_fraction is not None or request.interval_fractions is not None:
            raise ProtocolError("compare_moments contains unrelated arguments")
    elif request.action == "trace_motion":
        if request.interval_fractions is None:
            raise ProtocolError("trace_motion requires one interval")
        if request.interval_fractions[0] >= request.interval_fractions[1]:
            raise ProtocolError("trace_motion interval must be increasing")
        if any(
            value is not None
            for value in (
                request.center_fraction,
                request.radius_fraction,
                request.moment_fractions,
            )
        ):
            raise ProtocolError("trace_motion contains unrelated arguments")


def _linspace(start: float, end: float, count: int) -> list[float]:
    if count == 1:
        return [start]
    step = (end - start) / (count - 1)
    return [start + index * step for index in range(count)]


def overview_times(query_time: float, nframes: int = OVERVIEW_FRAMES) -> list[float]:
    if not math.isfinite(query_time) or query_time < 0 or nframes <= 0:
        raise ProtocolError("invalid overview request")
    return _linspace(0.0, query_time, nframes)


def request_times(request: EvidenceRequest, *, query_time: float) -> list[float]:
    validate_request(request)
    if request.action == "answer":
        return []
    if request.action == "inspect_window":
        assert request.center_fraction is not None and request.radius_fraction is not None
        start = max(0.0, request.center_fraction - request.radius_fraction)
        end = min(1.0, request.center_fraction + request.radius_fraction)
        fractions = _linspace(start, end, request.nframes)
    elif request.action == "compare_moments":
        assert request.moment_fractions is not None and request.radius_fraction is not None
        first_count = request.nframes // 2
        second_count = request.nframes - first_count
        fractions = []
        for moment, count in zip(request.moment_fractions, (first_count, second_count)):
            start = max(0.0, moment - request.radius_fraction)
            end = min(1.0, moment + request.radius_fraction)
            fractions.extend(_linspace(start, end, count))
    else:
        assert request.interval_fractions is not None
        fractions = _linspace(*request.interval_fractions, request.nframes)
    times = [min(query_time, max(0.0, fraction * query_time)) for fraction in fractions]
    if len(times) != request.nframes or any(time > query_time for time in times):
        raise ProtocolError("tool produced a non-causal frame plan")
    return times


def controller_prompt(
    question: str,
    options: Mapping[str, str],
    state: ActiveEvidenceState,
) -> str:
    """Build the actor-visible controller prompt without hidden routing data."""

    option_lines = [f"{label}. {text}" for label, text in sorted(options.items())]
    return "\n".join(
        [
            "You are viewing a causal chronological overview of a video prefix.",
            f"The {state.overview_frames} overview frames span normalized time 0 to 1.",
            f"You have {state.frames_remaining} frame credits and at most "
            f"{state.max_tool_calls - len(state.tool_calls)} evidence calls remaining.",
            "Choose one JSON action: inspect_window, compare_moments, "
            "trace_motion, or answer. Use only normalized times in [0,1].",
            "Do not answer until the visible evidence is sufficient.",
            "",
            str(question).strip(),
            *option_lines,
        ]
    )


def evidence_packet_times(
    query_time: float, requests: Sequence[EvidenceRequest]
) -> dict[str, Any]:
    """Execute a predeclared action sequence for tests and offline materialization."""

    state = ActiveEvidenceState(query_time=query_time)
    packets: list[dict[str, Any]] = [
        {"kind": "overview", "times": overview_times(query_time)}
    ]
    for request in requests:
        times = state.execute(request)
        if request.action != "answer":
            packets.append(
                {"kind": request.action, "request": request.to_dict(), "times": times}
            )
    return {"state": state.public_state(), "packets": packets, "answered": state.answered}
