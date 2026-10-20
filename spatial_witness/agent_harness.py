"""Single-checkpoint agent loop over the causal spatial workspace.

The orchestration boundary is intentionally model-agnostic: a caller injects
one model callback, backed by one declared checkpoint and decoding contract,
for every controller and answer turn.  The workspace tools never emit an
answer and have no access to question/options or benchmark metadata.

This is experimental infrastructure.  A concrete video-derived extractor and
complete paired benchmark measurements are still required before promotion.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any

from PIL import Image

from spatial_witness.causal_workspace import (
    BoundedWorkspaceSession,
    CausalSpatialWorkspace,
    EvidenceRef,
    SUPPORTED_EXECUTORS,
    WorkspaceBudget,
    WorkspaceError,
    parse_action,
)
from spatial_witness.causal_state_builder import pixel_sha256


FORMAT = "spatial_witness.single_checkpoint_agent_harness.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LABEL = re.compile(r"^[A-H]$")


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _text(value: Any, name: str) -> str:
    result = " ".join(str(value or "").strip().split())
    if not result:
        raise WorkspaceError(f"{name} must be nonempty")
    return result


def _visible_options(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise WorkspaceError("visible options must be an object")
    output = {str(key).strip().upper(): _text(text, "option text") for key, text in value.items()}
    expected = [chr(ord("A") + index) for index in range(len(output))]
    if not 2 <= len(output) <= 8 or sorted(output) != expected:
        raise WorkspaceError("visible option labels must be contiguous from A")
    return {label: output[label] for label in expected}


@dataclass(frozen=True)
class AgentModelIdentity:
    checkpoint_sha256: str
    decoding_contract_sha256: str
    external_generative_model_or_api: bool = False

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.checkpoint_sha256):
            raise WorkspaceError("agent checkpoint identity must be a SHA-256")
        if not _SHA256.fullmatch(self.decoding_contract_sha256):
            raise WorkspaceError("agent decoding contract must be a SHA-256")
        if self.external_generative_model_or_api is not False:
            raise WorkspaceError("external generative agent model or API is forbidden")


@dataclass(frozen=True)
class AgentCapabilityGrant:
    """Hash-bound allowlist of executors that passed the promotion gate."""

    executors: tuple[str, ...]
    promotion_manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.executors:
            raise WorkspaceError("agent capability grant must not be empty")
        if tuple(sorted(set(self.executors))) != self.executors:
            raise WorkspaceError("granted executors must be unique and sorted")
        unknown = sorted(set(self.executors) - SUPPORTED_EXECUTORS)
        if unknown:
            raise WorkspaceError(f"capability grant contains unknown executors: {unknown}")
        if not _SHA256.fullmatch(self.promotion_manifest_sha256):
            raise WorkspaceError("capability promotion manifest must be a SHA-256")


@dataclass(frozen=True)
class AgentDecision:
    kind: str
    action: dict[str, Any] | None = None
    answer: str | None = None


def parse_agent_decision(
    value: str | Mapping[str, Any], *, option_labels: set[str]
) -> AgentDecision:
    """Parse a strict tool-or-answer decision and reject hidden fields."""

    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise WorkspaceError("agent decision is not valid JSON") from error
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise WorkspaceError("agent decision must be one JSON object")
    if not isinstance(payload, dict):
        raise WorkspaceError("agent decision must be one JSON object")
    kind = str(payload.get("kind") or "")
    if kind == "answer":
        if set(payload) != {"kind", "answer"}:
            raise WorkspaceError("answer decision contains hidden or unrelated fields")
        answer = str(payload.get("answer") or "").strip().upper()
        if not _LABEL.fullmatch(answer) or answer not in option_labels:
            raise WorkspaceError("answer is not a visible option label")
        return AgentDecision("answer", answer=answer)
    if kind == "tool":
        if set(payload) != {"kind", "action"} or not isinstance(
            payload.get("action"), Mapping
        ):
            raise WorkspaceError("tool decision must contain exactly one action object")
        parsed = parse_action(payload["action"])
        action = {
            key: item for key, item in asdict(parsed).items() if item is not None
        }
        return AgentDecision("tool", action=action)
    raise WorkspaceError(f"unsupported agent decision kind: {kind!r}")


ModelStep = Callable[[dict[str, Any]], str | Mapping[str, Any]]
EvidenceResolver = Callable[[Mapping[str, Any]], Image.Image]


def validate_evidence_references(
    value: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Normalize an ordered, duplicate-free public evidence list."""

    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WorkspaceError("baseline evidence must be a sequence")
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int, float, str]] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise WorkspaceError("baseline evidence reference must be an object")
        reference = EvidenceRef(**dict(raw))
        identity = (
            reference.source_id,
            reference.frame_index,
            reference.timestamp,
            reference.pixel_sha256,
        )
        if identity in seen:
            raise WorkspaceError("baseline evidence contains a duplicate reference")
        seen.add(identity)
        output.append(asdict(reference))
    return output


class SingleCheckpointAgentHarness:
    """Run bounded state queries and one final answer with one model callback."""

    def __init__(
        self,
        workspace: CausalSpatialWorkspace,
        *,
        query_time: float,
        question: str,
        options: Mapping[str, Any],
        model_identity: AgentModelIdentity,
        capability_grant: AgentCapabilityGrant,
        budget: WorkspaceBudget | None = None,
        packet_mode: str = "state_and_evidence",
        baseline_evidence: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.question = _text(question, "visible question")
        self.options = _visible_options(options)
        self.model_identity = model_identity
        self.capability_grant = capability_grant
        self.baseline_evidence = validate_evidence_references(baseline_evidence)
        self.session = BoundedWorkspaceSession(
            workspace,
            query_time=query_time,
            budget=budget,
            packet_mode=packet_mode,
        )

    def _resolved_evidence(
        self, evidence_resolver: EvidenceResolver
    ) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        seen: set[tuple[str, int, float, str]] = set()
        references = list(self.baseline_evidence)
        references.extend(
            reference
            for packet in self.session.transcript
            for reference in packet["evidence"]
        )
        for reference in references:
            identity = (
                str(reference["source_id"]),
                int(reference["frame_index"]),
                float(reference["timestamp"]),
                str(reference["pixel_sha256"]),
            )
            if identity in seen:
                continue
            seen.add(identity)
            image = evidence_resolver(reference)
            if not isinstance(image, Image.Image):
                raise WorkspaceError("evidence resolver must return a PIL image")
            actual = pixel_sha256(image)
            if actual != reference["pixel_sha256"]:
                raise WorkspaceError("resolved evidence pixel SHA-256 mismatch")
            resolved.append({"reference": dict(reference), "image": image})
        return resolved

    def _model_context(
        self, turn: int, evidence_resolver: EvidenceResolver
    ) -> dict[str, Any]:
        return {
            "protocol": FORMAT,
            "turn": turn,
            "visible_question": self.question,
            "visible_options": self.options,
            "public_budget": self.session.public_budget(),
            "tool_transcript": self.session.transcript,
            # The concrete evaluator serializes these PIL images through the
            # same processor as ordinary video frames. They are deliberately
            # excluded from the persisted JSON audit; their exact pixel hashes
            # are already bound in each transcript reference.
            "model_visible_evidence": self._resolved_evidence(evidence_resolver),
            "allowed_output": {
                "tool": {
                    "kind": "tool",
                    "action": "WorkspaceAction",
                    "granted_executors": list(self.capability_grant.executors),
                },
                "answer": {"kind": "answer", "answer": "visible option label"},
            },
        }

    def run(
        self, model_step: ModelStep, *, evidence_resolver: EvidenceResolver
    ) -> dict[str, Any]:
        """Call one injected model until it answers or violates the budget."""

        raw_decision_hashes: list[str] = []
        answer: str | None = None
        maximum_turns = self.session.budget.maximum_executor_calls + 1
        for turn in range(maximum_turns):
            context = self._model_context(turn, evidence_resolver)
            raw = model_step(context)
            raw_decision_hashes.append(_sha256(raw))
            decision = parse_agent_decision(
                raw, option_labels=set(self.options)
            )
            if decision.kind == "answer":
                answer = decision.answer
                break
            assert decision.action is not None
            if decision.action["executor"] not in self.capability_grant.executors:
                raise WorkspaceError(
                    "agent requested an executor absent from the promotion grant"
                )
            self.session.execute(decision.action)
        if answer is None:
            raise WorkspaceError("agent exhausted its tool budget without answering")
        body: dict[str, Any] = {
            "format": FORMAT,
            "answer": answer,
            "model_identity": asdict(self.model_identity),
            "capability_grant": asdict(self.capability_grant),
            "same_checkpoint_all_turns": True,
            "model_turns": len(raw_decision_hashes),
            "raw_decision_sha256": raw_decision_hashes,
            "packet_mode": self.session.packet_mode,
            "baseline_evidence": self.baseline_evidence,
            "baseline_evidence_sha256": _sha256(self.baseline_evidence),
            "tool_transcript": self.session.transcript,
            "final_budget": self.session.public_budget(),
            "question_and_options_consumed_by_workspace": False,
            "benchmark_metadata_answers_predictions_or_scores_consumed_by_policy": False,
            "external_generative_model_calls": 0,
            "evidence_pixels_resolved_and_hash_verified": True,
        }
        return {**body, "run_sha256": _sha256(body)}
