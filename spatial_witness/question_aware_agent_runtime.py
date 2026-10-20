"""Question-aware, benchmark-blind runtime for causal spatial video agents.

``task-blind`` means every sample receives the same action ABI, budgets, and
environment contract.  It does *not* mean that the controller is forbidden to
read its question.  The controller may actively seek evidence and invoke
generic memory/geometry tools, while each tool remains answer-blind and sees
no benchmark category, annotation, option label, or score.

This module unifies the previously separate active-evidence and causal-state
interfaces.  It also exposes a hash-bound provider boundary for local tracking,
depth, pose, or mapping implementations; providers cannot be generative answer
models and their returned evidence is checked against the causal prefix.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Protocol

from PIL import Image

from spatial_witness.active_evidence import (
    ActiveEvidenceState,
    EvidenceRequest,
    ProtocolError,
    overview_times,
    parse_action as parse_evidence_action,
)
from spatial_witness.agent_harness import AgentModelIdentity
from spatial_witness.causal_state_builder import pixel_sha256
from spatial_witness.causal_workspace import (
    BoundedWorkspaceSession,
    CausalSpatialWorkspace,
    EvidenceRef,
    SUPPORTED_EXECUTORS,
    WorkspaceBudget,
    WorkspaceError,
)


class CausalFrameLike(Protocol):
    frame_index: int
    timestamp: float
    image: Image.Image


FORMAT = "spatial_witness.question_aware_spatial_agent.v1"
TRANSITION_FORMAT = "state_t_action_t_environment_state_t_plus_1"
EVIDENCE_TOOLS = frozenset({"inspect_window", "compare_moments", "trace_motion"})
GEOMETRY_TOOLS = frozenset({"transform_2d_vector", "simulate_rotation"})
BUILTIN_TOOLS = EVIDENCE_TOOLS | SUPPORTED_EXECUTORS | GEOMETRY_TOOLS
WORKSPACE_PACKET_MODES = frozenset({"state_and_evidence", "evidence_only"})
BUILTIN_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "inspect_window": {
        "purpose": "inspect a dense local interval around one normalized causal time",
        "arguments": {
            "nframes": "integer multiple of 8 in [8,48]",
            "center_fraction": "number in [0,1]",
            "radius_fraction": "number in (0,1]",
        },
    },
    "compare_moments": {
        "purpose": "compare two distinct moments with matched local evidence",
        "arguments": {
            "nframes": "integer multiple of 8 in [8,48]",
            "moment_fractions": "two distinct numbers in [0,1]",
            "radius_fraction": "number in (0,1]",
        },
    },
    "trace_motion": {
        "purpose": "sample one increasing causal interval for motion evidence",
        "arguments": {
            "nframes": "integer multiple of 8 in [8,48]",
            "interval_fractions": "two increasing numbers in [0,1]",
        },
    },
    "retrieve_last_seen": {
        "purpose": "retrieve the latest causal observation of one visible-language entity",
        "arguments": {"subject": "nonempty entity name"},
    },
    "count_place_visits": {
        "purpose": "count causally recorded entries into one place",
        "arguments": {"subject": "nonempty place name"},
    },
    "query_current_place_state": {
        "purpose": "read the current causally observed place state",
        "arguments": {},
    },
    "query_topology_neighbors": {
        "purpose": "list observed direct neighbors of one place",
        "arguments": {"subject": "nonempty place name"},
    },
    "query_topology_path": {
        "purpose": "find an observed topology path between two places",
        "arguments": {
            "start": "nonempty start place name",
            "goal": "nonempty goal place name",
        },
    },
    "query_relative_pose": {
        "purpose": "compute relative SE(2) pose between two observed frame indices",
        "arguments": {
            "from_frame": "nonnegative integer frame index",
            "to_frame": "nonnegative integer frame index",
        },
    },
    "query_camera_pose": {
        "purpose": "summarize the causal camera trajectory from a reference pose to now",
        "arguments": {
            "since": "optional already-written snapshot timestamp, not a question field",
        },
    },
    "query_entity_relative_pose": {
        "purpose": "transform a remembered metric entity position into current ego coordinates",
        "arguments": {"subject": "nonempty entity name"},
    },
    "transform_2d_vector": {
        "purpose": "transform a 2-D vector between ego and world reference frames",
        "arguments": {
            "x": "finite number",
            "y": "finite number",
            "yaw_radians": "finite number",
            "mode": "ego_to_world or world_to_ego",
        },
    },
    "simulate_rotation": {
        "purpose": "rotate a 2-D vector by a hypothetical yaw change",
        "arguments": {
            "x": "finite number",
            "y": "finite number",
            "delta_yaw_radians": "finite number",
        },
    },
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_LABEL = re.compile(r"^[A-H]$")
_FORBIDDEN_PROVIDER_KEYS = frozenset(
    {"answer", "correct_answer", "correct_option", "label", "prediction", "score"}
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _text(value: Any, name: str) -> str:
    output = " ".join(str(value or "").strip().split())
    if not output:
        raise WorkspaceError(f"{name} must be nonempty")
    return output


def _options(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise WorkspaceError("visible options must be an object")
    output = {
        str(label).strip().upper(): _text(text, "option text")
        for label, text in value.items()
    }
    labels = [chr(ord("A") + index) for index in range(len(output))]
    if not 2 <= len(labels) <= 8 or sorted(output) != labels:
        raise WorkspaceError("visible option labels must be contiguous from A")
    return {label: output[label] for label in labels}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkspaceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise WorkspaceError(f"{name} must be finite")
    return result


def _contains_forbidden_provider_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).strip().casefold() in _FORBIDDEN_PROVIDER_KEYS:
                return True
            if _contains_forbidden_provider_key(item):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_forbidden_provider_key(item) for item in value)
    return False


@dataclass(frozen=True)
class RuntimeBudget:
    frame_budget: int = 128
    overview_frames: int = 32
    maximum_tool_calls: int = 4
    maximum_workspace_evidence_refs: int = 96

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_budget, bool)
            or not isinstance(self.frame_budget, int)
            or self.frame_budget < 8
        ):
            raise WorkspaceError("frame_budget must be an integer of at least 8")
        if not 0 < self.overview_frames <= self.frame_budget:
            raise WorkspaceError("overview_frames must fit inside frame_budget")
        if not 1 <= self.maximum_tool_calls <= 8:
            raise WorkspaceError("maximum_tool_calls must be in [1, 8]")
        if self.maximum_workspace_evidence_refs < 1:
            raise WorkspaceError("maximum_workspace_evidence_refs must be positive")


@dataclass(frozen=True)
class RuntimeCapabilityGrant:
    tools: tuple[str, ...]
    promotion_manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.tools or tuple(sorted(set(self.tools))) != self.tools:
            raise WorkspaceError("runtime tools must be nonempty, unique, and sorted")
        if not all(_TOOL_NAME.fullmatch(name) for name in self.tools):
            raise WorkspaceError("runtime grant contains an invalid tool name")
        if not _SHA256.fullmatch(self.promotion_manifest_sha256):
            raise WorkspaceError("runtime promotion manifest must be a SHA-256")


@dataclass(frozen=True)
class ProviderContext:
    source_id: str
    query_time: float
    causal_frames: tuple[CausalFrameLike, ...]


@dataclass(frozen=True)
class ProviderResult:
    status: str
    facts: tuple[dict[str, Any], ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"ok", "unavailable"}:
            raise WorkspaceError("provider status must be ok or unavailable")
        if _contains_forbidden_provider_key(self.facts):
            raise WorkspaceError("provider attempted to return an answer-like field")


ProviderExecute = Callable[[Mapping[str, Any], ProviderContext], ProviderResult]


@dataclass(frozen=True)
class LocalSpatialToolProvider:
    """A local, answer-blind implementation such as tracking, pose, or depth."""

    name: str
    implementation_sha256: str
    execute: ProviderExecute
    external_generative_model_or_api: bool = False
    consumes_question_options_or_benchmark_metadata: bool = False

    def __post_init__(self) -> None:
        if not _TOOL_NAME.fullmatch(self.name) or self.name in BUILTIN_TOOLS:
            raise WorkspaceError("provider name is invalid or shadows a builtin")
        if not _SHA256.fullmatch(self.implementation_sha256):
            raise WorkspaceError("provider implementation must be hash-bound")
        if self.external_generative_model_or_api is not False:
            raise WorkspaceError("external generative providers are forbidden")
        if self.consumes_question_options_or_benchmark_metadata is not False:
            raise WorkspaceError("spatial providers must remain question and benchmark blind")
        if not callable(self.execute):
            raise WorkspaceError("provider execute must be callable")


@dataclass(frozen=True)
class RuntimeDecision:
    kind: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    answer: str | None = None


def parse_runtime_decision(
    value: str | Mapping[str, Any], *, labels: set[str], allowed_tools: set[str]
) -> RuntimeDecision:
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise WorkspaceError("runtime decision is not valid JSON") from error
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise WorkspaceError("runtime decision must be one JSON object")
    if not isinstance(payload, dict):
        raise WorkspaceError("runtime decision must be one JSON object")
    kind = str(payload.get("kind") or "")
    if kind == "answer":
        if set(payload) != {"kind", "answer"}:
            raise WorkspaceError("answer decision contains unrelated fields")
        answer = str(payload.get("answer") or "").strip().upper()
        if not _LABEL.fullmatch(answer) or answer not in labels:
            raise WorkspaceError("answer is not one visible option label")
        return RuntimeDecision("answer", answer=answer)
    if kind == "tool":
        if set(payload) != {"kind", "tool", "arguments"}:
            raise WorkspaceError("tool decision must contain tool and arguments only")
        tool = str(payload.get("tool") or "")
        arguments = payload.get("arguments")
        if tool not in allowed_tools:
            raise WorkspaceError("tool is absent from the runtime capability grant")
        if not isinstance(arguments, Mapping):
            raise WorkspaceError("tool arguments must be an object")
        return RuntimeDecision("tool", tool=tool, arguments=dict(arguments))
    raise WorkspaceError(f"unsupported runtime decision kind: {kind!r}")


ModelStep = Callable[[dict[str, Any]], str | Mapping[str, Any]]


class QuestionAwareSpatialAgentRuntime:
    """Execute ``state_t -> action_t -> environment -> state_(t+1)``."""

    def __init__(
        self,
        source_id: str,
        frames: Sequence[CausalFrameLike],
        *,
        query_time: float,
        question: str,
        options: Mapping[str, Any],
        option_images: Mapping[str, Image.Image] | None = None,
        model_identity: AgentModelIdentity,
        capability_grant: RuntimeCapabilityGrant,
        workspace: CausalSpatialWorkspace | None = None,
        providers: Sequence[LocalSpatialToolProvider] = (),
        budget: RuntimeBudget | None = None,
        workspace_packet_mode: str = "state_and_evidence",
    ) -> None:
        self.source_id = _text(source_id, "source_id")
        self.query_time = _finite(query_time, "query_time")
        if self.query_time < 0:
            raise WorkspaceError("query_time must be nonnegative")
        self.question = _text(question, "question")
        self.options = _options(options)
        self.option_images = dict(option_images or {})
        if not set(self.option_images) <= set(self.options):
            raise WorkspaceError("option images contain a non-visible option label")
        if any(not isinstance(image, Image.Image) for image in self.option_images.values()):
            raise WorkspaceError("option images must be PIL images")
        self.model_identity = model_identity
        self.capability_grant = capability_grant
        self.budget = budget or RuntimeBudget()
        if workspace_packet_mode not in WORKSPACE_PACKET_MODES:
            raise WorkspaceError("unsupported runtime workspace packet mode")
        self.frames = tuple(frames)
        self._validate_frames()
        self._frame_by_identity = {
            self._identity(frame): frame for frame in self.frames
        }
        provider_map = {provider.name: provider for provider in providers}
        if len(provider_map) != len(providers):
            raise WorkspaceError("provider names must be unique")
        unknown = set(capability_grant.tools) - BUILTIN_TOOLS - set(provider_map)
        if unknown:
            raise WorkspaceError(f"runtime grant has no implementation for: {sorted(unknown)}")
        self.providers = provider_map
        self.tool_schemas = {
            tool: BUILTIN_TOOL_SCHEMAS.get(
                tool,
                {
                    "purpose": "hash-bound local spatial provider",
                    "arguments": "provider-defined JSON object",
                },
            )
            for tool in capability_grant.tools
        }
        needs_workspace = bool(set(capability_grant.tools) & SUPPORTED_EXECUTORS)
        if needs_workspace and workspace is None:
            raise WorkspaceError("workspace tools require one causal workspace")
        self.workspace_session = None
        if workspace is not None:
            self.workspace_session = BoundedWorkspaceSession(
                workspace,
                query_time=self.query_time,
                budget=WorkspaceBudget(
                    maximum_executor_calls=self.budget.maximum_tool_calls,
                    maximum_evidence_refs=self.budget.maximum_workspace_evidence_refs,
                ),
                packet_mode=workspace_packet_mode,
            )
        self.workspace_packet_mode = workspace_packet_mode
        try:
            self.evidence_state = ActiveEvidenceState(
                query_time=self.query_time,
                frame_budget=self.budget.frame_budget,
                overview_frames=self.budget.overview_frames,
                max_tool_calls=self.budget.maximum_tool_calls,
            )
        except ProtocolError as error:
            raise WorkspaceError(str(error)) from error
        self.visible_evidence: list[EvidenceRef] = self._resolve_times(
            overview_times(self.query_time, self.budget.overview_frames)
        )
        self.tool_calls_used = 0
        self.transcript: list[dict[str, Any]] = []
        self.transitions: list[dict[str, Any]] = []

    def _validate_frames(self) -> None:
        previous_index, previous_time = -1, -1.0
        for frame in self.frames:
            if frame.frame_index <= previous_index or frame.timestamp < previous_time:
                raise WorkspaceError("causal frames must be chronological")
            if frame.timestamp > self.query_time:
                raise WorkspaceError("causal prefix contains a future frame")
            previous_index, previous_time = frame.frame_index, frame.timestamp
        if not self.frames:
            raise WorkspaceError("runtime requires at least one causal frame")

    def _identity(self, frame: CausalFrameLike) -> tuple[str, int, float, str]:
        return (
            self.source_id,
            frame.frame_index,
            float(frame.timestamp),
            pixel_sha256(frame.image),
        )

    def _reference(self, frame: CausalFrameLike) -> EvidenceRef:
        return EvidenceRef(*self._identity(frame))

    def _resolve_times(self, times: Sequence[float]) -> list[EvidenceRef]:
        output = []
        for requested in times:
            frame = min(
                self.frames,
                key=lambda item: (abs(float(item.timestamp) - requested), item.frame_index),
            )
            output.append(self._reference(frame))
        return output

    def _known_reference(self, reference: EvidenceRef) -> None:
        identity = (
            reference.source_id,
            reference.frame_index,
            float(reference.timestamp),
            reference.pixel_sha256,
        )
        if identity not in self._frame_by_identity or reference.timestamp > self.query_time:
            raise WorkspaceError("tool returned evidence outside the causal prefix")

    def _charge_and_show(self, references: Sequence[EvidenceRef], *, precharged: bool) -> None:
        for reference in references:
            self._known_reference(reference)
        if not precharged:
            if len(references) > self.evidence_state.frames_remaining:
                raise WorkspaceError("tool evidence exceeds the unified frame budget")
            self.evidence_state.frames_charged += len(references)
        self.visible_evidence.extend(references)

    def _public_state(self) -> dict[str, Any]:
        output = {
            "tool_calls_used": self.tool_calls_used,
            "tool_calls_remaining": self.budget.maximum_tool_calls - self.tool_calls_used,
            "frames_charged": self.evidence_state.frames_charged,
            "frames_remaining": self.evidence_state.frames_remaining,
            "visible_evidence": [asdict(item) for item in self.visible_evidence],
            "transcript": self.transcript,
        }
        if self.workspace_session is not None:
            output["workspace_index"] = self.workspace_session.public_inventory()
        else:
            output["workspace_index"] = {"status": "unavailable"}
        return output

    def _model_images(self) -> list[dict[str, Any]]:
        output, seen = [], set()
        for reference in self.visible_evidence:
            identity = (
                reference.source_id,
                reference.frame_index,
                float(reference.timestamp),
                reference.pixel_sha256,
            )
            if identity in seen:
                continue
            seen.add(identity)
            output.append(
                {"reference": asdict(reference), "image": self._frame_by_identity[identity].image}
            )
        return output

    def _model_option_images(self) -> list[dict[str, Any]]:
        return [
            {
                "option_label": label,
                "image": self.option_images[label],
                "pixel_sha256": pixel_sha256(self.option_images[label]),
            }
            for label in self.options
            if label in self.option_images
        ]

    @staticmethod
    def _geometry(tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if tool == "transform_2d_vector":
            required = {"x", "y", "yaw_radians", "mode"}
            if set(arguments) != required:
                raise WorkspaceError("transform_2d_vector has invalid arguments")
            x, y = _finite(arguments["x"], "x"), _finite(arguments["y"], "y")
            yaw = _finite(arguments["yaw_radians"], "yaw_radians")
            mode = str(arguments["mode"])
            if mode == "ego_to_world":
                out_x = math.cos(yaw) * x - math.sin(yaw) * y
                out_y = math.sin(yaw) * x + math.cos(yaw) * y
            elif mode == "world_to_ego":
                out_x = math.cos(yaw) * x + math.sin(yaw) * y
                out_y = -math.sin(yaw) * x + math.cos(yaw) * y
            else:
                raise WorkspaceError("invalid reference-frame transform mode")
            return {"status": "ok", "facts": [{"x": out_x, "y": out_y, "mode": mode}], "evidence": []}
        if set(arguments) != {"x", "y", "delta_yaw_radians"}:
            raise WorkspaceError("simulate_rotation has invalid arguments")
        x, y = _finite(arguments["x"], "x"), _finite(arguments["y"], "y")
        yaw = _finite(arguments["delta_yaw_radians"], "delta_yaw_radians")
        return {
            "status": "ok",
            "facts": [{
                "x": math.cos(yaw) * x - math.sin(yaw) * y,
                "y": math.sin(yaw) * x + math.cos(yaw) * y,
                "delta_yaw_radians": yaw,
            }],
            "evidence": [],
        }

    def _execute_tool(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if tool in EVIDENCE_TOOLS:
            try:
                request: EvidenceRequest = parse_evidence_action(
                    {"action": tool, **dict(arguments)}
                )
                times = self.evidence_state.execute(request)
            except ProtocolError as error:
                raise WorkspaceError(str(error)) from error
            references = self._resolve_times(times)
            self._charge_and_show(references, precharged=True)
            return {
                "status": "ok",
                "facts": [{"requested_timestamps": times}],
                "evidence": [asdict(item) for item in references],
            }
        if tool in SUPPORTED_EXECUTORS:
            assert self.workspace_session is not None
            packet = self.workspace_session.execute({"executor": tool, **dict(arguments)})
            references = tuple(EvidenceRef(**item) for item in packet["evidence"])
            self._charge_and_show(references, precharged=False)
            return packet
        if tool in GEOMETRY_TOOLS:
            return self._geometry(tool, arguments)
        provider = self.providers[tool]
        result = provider.execute(
            dict(arguments),
            ProviderContext(self.source_id, self.query_time, self.frames),
        )
        if not isinstance(result, ProviderResult):
            raise WorkspaceError("provider must return ProviderResult")
        self._charge_and_show(result.evidence, precharged=False)
        packet = {
            "status": result.status,
            "facts": list(result.facts),
            "evidence": [asdict(item) for item in result.evidence],
            "provider_implementation_sha256": provider.implementation_sha256,
        }
        if result.reason is not None:
            packet["reason"] = result.reason
        return packet

    def run(self, model_step: ModelStep) -> dict[str, Any]:
        raw_hashes, answer = [], None
        for turn in range(self.budget.maximum_tool_calls + 1):
            before = self._public_state()
            context = {
                "protocol": FORMAT,
                "turn": turn,
                "visible_question": self.question,
                "visible_options": self.options,
                "allowed_tools": list(self.capability_grant.tools),
                "allowed_tool_schemas": self.tool_schemas,
                "runtime_budget": {
                    key: before[key]
                    for key in (
                        "tool_calls_used", "tool_calls_remaining",
                        "frames_charged", "frames_remaining",
                    )
                },
                "workspace_index": before["workspace_index"],
                "tool_transcript": self.transcript,
                "model_visible_evidence": self._model_images(),
                "model_visible_option_images": self._model_option_images(),
                "action_schema": {
                    "tool": {"kind": "tool", "tool": "allowed tool", "arguments": {}},
                    "answer": {"kind": "answer", "answer": "visible option label"},
                },
            }
            raw = model_step(context)
            raw_hashes.append(_canonical_sha256(raw))
            decision = parse_runtime_decision(
                raw,
                labels=set(self.options),
                allowed_tools=set(self.capability_grant.tools),
            )
            if decision.kind == "answer":
                answer = decision.answer
                break
            if self.tool_calls_used >= self.budget.maximum_tool_calls:
                raise WorkspaceError("runtime tool-call budget exceeded")
            assert decision.tool is not None and decision.arguments is not None
            packet = self._execute_tool(decision.tool, decision.arguments)
            self.tool_calls_used += 1
            step = {
                "turn": turn,
                "state_t_sha256": _canonical_sha256(before),
                "action_t": {"tool": decision.tool, "arguments": decision.arguments},
                "environment_packet": packet,
            }
            self.transcript.append({"tool": decision.tool, "arguments": decision.arguments, **packet})
            step["state_t_plus_1_sha256"] = _canonical_sha256(self._public_state())
            self.transitions.append(step)
        if answer is None:
            raise WorkspaceError("runtime exhausted its tool budget without answering")
        body = {
            "format": FORMAT,
            "transition_protocol": TRANSITION_FORMAT,
            "answer": answer,
            "model_identity": asdict(self.model_identity),
            "capability_grant": asdict(self.capability_grant),
            "provider_manifests": {
                name: provider.implementation_sha256
                for name, provider in sorted(self.providers.items())
            },
            "same_checkpoint_all_model_turns": True,
            "controller_question_aware": True,
            "tools_question_option_and_benchmark_blind": True,
            "causal_horizon": self.query_time,
            "runtime_budget": asdict(self.budget),
            "workspace_packet_mode": self.workspace_packet_mode,
            "option_image_pixel_sha256": {
                label: pixel_sha256(image)
                for label, image in sorted(self.option_images.items())
            },
            "final_public_state": self._public_state(),
            "transitions": self.transitions,
            "raw_decision_sha256": raw_hashes,
            "external_generative_model_calls": 0,
            "benchmark_metadata_answers_predictions_or_scores_consumed": False,
        }
        return {**body, "run_sha256": _canonical_sha256(body)}
