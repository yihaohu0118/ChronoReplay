"""Deterministic-planner control over the causal spatial workspace.

This harness is the deployment fallback for a pure-VLM checkpoint that has
not learned a reliable JSON action policy.  A frozen compiler consumes only
visible question text and emits at most one high-precision workspace action.
The exact same answer checkpoint then receives either the resulting state and
evidence or a matched evidence-only packet and produces the option label.

The compiler never emits an answer and a capability grant still controls
which executors may run.  This is therefore an agent harness with a
deterministic planner, not benchmark-category routing or another answer model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
import re
from typing import Any

from PIL import Image

from spatial_witness.agent_harness import (
    AgentCapabilityGrant,
    AgentModelIdentity,
    EvidenceResolver,
    ModelStep,
    parse_agent_decision,
    validate_evidence_references,
)
from spatial_witness.causal_state_builder import pixel_sha256
from spatial_witness.causal_workspace import (
    BoundedWorkspaceSession,
    CausalSpatialWorkspace,
    WorkspaceBudget,
    WorkspaceError,
)
from spatial_witness.deterministic_reader import (
    resolve_deterministic_workspace_answer,
)
from synthesis.spatial_query_compiler import (
    compile_spatial_query,
    compile_workspace_action,
)


FORMAT = "spatial_witness.compiled_agent_harness.v1"
_LABEL = re.compile(r"^[A-H]$")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _text(value: Any, name: str) -> str:
    result = " ".join(str(value or "").strip().split())
    if not result:
        raise WorkspaceError(f"{name} must be nonempty")
    return result


def _options(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise WorkspaceError("visible options must be an object")
    normalized = {
        str(key).strip().upper(): _text(text, "option text")
        for key, text in value.items()
    }
    labels = [chr(ord("A") + index) for index in range(len(normalized))]
    if not 2 <= len(labels) <= 8 or sorted(normalized) != labels:
        raise WorkspaceError("visible option labels must be contiguous from A")
    return {label: normalized[label] for label in labels}


def _answer(value: str | Mapping[str, Any], labels: set[str]) -> str:
    # Pure VLM SFT is trained to emit a label.  The concrete adapter may also
    # return the ordinary strict agent JSON object; both representations carry
    # exactly the same answer and no hidden state.
    if isinstance(value, str):
        label = value.strip().upper()
        if _LABEL.fullmatch(label) and label in labels:
            return label
    decision = parse_agent_decision(value, option_labels=labels)
    if decision.kind != "answer" or decision.answer is None:
        raise WorkspaceError("compiled harness answerer attempted another tool call")
    return decision.answer


class CompiledWorkspaceHarness:
    """Execute at most one granted compiled action, then answer once."""

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
        self.options = _options(options)
        self.model_identity = model_identity
        self.capability_grant = capability_grant
        self.baseline_evidence = validate_evidence_references(baseline_evidence)
        self.session = BoundedWorkspaceSession(
            workspace,
            query_time=query_time,
            budget=budget or WorkspaceBudget(maximum_executor_calls=1),
            packet_mode=packet_mode,
        )

    def _resolved_evidence(
        self, evidence_resolver: EvidenceResolver
    ) -> list[dict[str, Any]]:
        resolved = []
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
            if pixel_sha256(image) != reference["pixel_sha256"]:
                raise WorkspaceError("resolved evidence pixel SHA-256 mismatch")
            resolved.append({"reference": dict(reference), "image": image})
        return resolved

    def run(
        self, model_step: ModelStep, *, evidence_resolver: EvidenceResolver
    ) -> dict[str, Any]:
        plan = compile_spatial_query(self.question, self.options)
        compiled = compile_workspace_action(self.question, self.options)
        action = None
        status = "no_high_precision_action"
        if compiled is not None:
            if compiled["executor"] in self.capability_grant.executors:
                action = compiled
                self.session.execute(action)
                status = "executed"
            else:
                status = "not_in_capability_grant"
        context = {
            "protocol": FORMAT,
            "visible_question": self.question,
            "visible_options": self.options,
            "compiled_query_plan": plan.to_dict(),
            "compiled_workspace_action": action,
            "compiled_action_status": status,
            "public_budget": self.session.public_budget(),
            "tool_transcript": self.session.transcript,
            "model_visible_evidence": self._resolved_evidence(evidence_resolver),
            "allowed_output": "one visible option label",
        }
        raw = model_step(context)
        model_answer = _answer(raw, set(self.options))
        deterministic = resolve_deterministic_workspace_answer(
            options=self.options,
            action=action,
            transcript=self.session.transcript,
        )
        answer = deterministic.label if deterministic is not None else model_answer
        body: dict[str, Any] = {
            "format": FORMAT,
            "answer": answer,
            "model_answer": model_answer,
            "deterministic_answer_resolution": (
                deterministic.to_dict() if deterministic is not None else None
            ),
            "model_identity": asdict(self.model_identity),
            "capability_grant": asdict(self.capability_grant),
            "same_checkpoint_answerer": True,
            "planner": "frozen_visible_language_compiler",
            "compiled_query_plan": plan.to_dict(),
            "compiled_workspace_action": action,
            "compiled_action_status": status,
            "packet_mode": self.session.packet_mode,
            "baseline_evidence": self.baseline_evidence,
            "baseline_evidence_sha256": _canonical_sha256(
                self.baseline_evidence
            ),
            "tool_transcript": self.session.transcript,
            "raw_answer_sha256": _canonical_sha256(raw),
            "question_and_options_consumed_by_workspace": False,
            "benchmark_metadata_answers_predictions_or_scores_consumed": False,
            "external_generative_model_calls": 0,
        }
        return {**body, "run_sha256": _canonical_sha256(body)}
