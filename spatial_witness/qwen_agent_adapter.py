"""Concrete same-checkpoint Qwen callback for the streaming agent system.

The adapter owns no model of its own.  It serializes observer, controller, and
answer contexts into prompts for one already-loaded Qwen checkpoint.  Observer
prompts are question-blind; answer prompts receive the frozen baseline images
plus hash-bound tool evidence and structured workspace facts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any

from PIL import Image

from spatial_witness.agent_harness import FORMAT as AGENT_FORMAT
from spatial_witness.causal_workspace import WorkspaceError
from spatial_witness.compiled_agent_harness import FORMAT as COMPILED_FORMAT
from spatial_witness.question_aware_agent_runtime import (
    FORMAT as QUESTION_AWARE_RUNTIME_FORMAT,
)
from spatial_witness.same_checkpoint_observer import (
    FORMAT as OBSERVER_FORMAT,
    observer_prompt,
)
from synthesis.vlm_inference import predict


FORMAT = "spatial_witness.qwen_same_checkpoint_adapter.v1"
_LABEL = re.compile(r"^[A-H]$")
_ANSWER = re.compile(
    r"(?i)(?:answer\s*(?::|=|is)\s*|\"answer\"\s*:\s*\")([A-H])"
)
Generate = Callable[[Any, Any, list[Image.Image], str, int], str]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _default_generate(
    model: Any,
    processor: Any,
    images: list[Image.Image],
    prompt: str,
    max_new_tokens: int,
) -> str:
    return predict(
        model,
        processor,
        images,
        prompt,
        max_new_tokens=max_new_tokens,
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Recover the first unambiguous JSON object from model-facing text.

    The action ABI remains strict after parsing.  This boundary only tolerates
    harmless prose or a Markdown fence around one object, because ordinary
    instruction-tuned checkpoints often append a short explanation even when
    asked for JSON.  A second object remains ambiguous and is rejected.
    """

    stripped = str(text or "").strip()
    candidates = [stripped]
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        candidates.append("\n".join(lines[1:-1]).strip())
    brace = stripped.find("{")
    if brace >= 0:
        try:
            value, consumed = json.JSONDecoder().raw_decode(stripped[brace:])
            if isinstance(value, dict):
                tail = stripped[brace + consumed :].strip().strip("`").strip()
                if "{" in tail:
                    raise WorkspaceError("model emitted more than one JSON object")
                return value
        except json.JSONDecodeError:
            pass
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise WorkspaceError("same-checkpoint model did not emit one JSON object")


def extract_option_label(text: str, labels: set[str]) -> str:
    stripped = str(text or "").strip().upper()
    if _LABEL.fullmatch(stripped) and stripped in labels:
        return stripped
    match = _ANSWER.search(str(text or ""))
    if match and match.group(1).upper() in labels:
        return match.group(1).upper()
    raise WorkspaceError("same-checkpoint answer is not one visible option label")


def compiled_answer_prompt(context: Mapping[str, Any]) -> str:
    """Serialize the public state-reader ABI used at train and inference time.

    Keeping this formatter outside the model callback lets the source-GT data
    compiler build reader supervision against the exact deployed prompt.  The
    context is already question-aware; only the persistent writer is required
    to remain question blind.
    """

    options = context.get("visible_options") or {}
    option_text = "\n".join(
        f"{label}. {text}" for label, text in options.items()
    )
    state = {
        "compiled_query_plan": context.get("compiled_query_plan"),
        "compiled_workspace_action": context.get("compiled_workspace_action"),
        "compiled_action_status": context.get("compiled_action_status"),
        "workspace_index": context.get("workspace_index"),
        "tool_transcript": context.get("tool_transcript") or [],
    }
    return f"""Answer the multiple-choice spatial video question using the ordered causal frames and the optional audited workspace state below.
Workspace state may say unavailable; never invent a missing fact. Frame indices and timestamps are evidence provenance, not answer labels.

Question: {context.get('visible_question')}
Options:
{option_text}

Audited workspace packet:
{json.dumps(state, ensure_ascii=False, sort_keys=True)}

Return only the option letter."""


@dataclass(frozen=True)
class QwenAdapterConfig:
    observer_max_new_tokens: int = 192
    controller_max_new_tokens: int = 128
    answer_max_new_tokens: int = 8

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise WorkspaceError(f"{name} must be a positive integer")


class QwenSameCheckpointModelStep:
    """Serialize every agent-system turn through one loaded checkpoint."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        config: QwenAdapterConfig | None = None,
        generate: Generate = _default_generate,
    ) -> None:
        self.model = model
        self.processor = processor
        self.config = config or QwenAdapterConfig()
        self.generate = generate
        self.calls = {"observer": 0, "controller": 0, "answer": 0}
        self.observer_parse_failures = 0
        self.controller_parse_fallbacks = 0

    def _call(
        self, images: list[Image.Image], prompt: str, maximum_tokens: int
    ) -> str:
        return self.generate(
            self.model, self.processor, images, prompt, maximum_tokens
        )

    def _observe(self, context: Mapping[str, Any]) -> dict[str, Any]:
        image = context.get("model_visible_observation")
        if not isinstance(image, Image.Image):
            raise WorkspaceError("observer context has no PIL observation")
        maximum = int(context.get("maximum_entities") or 0)
        prompt = observer_prompt(maximum)
        self.calls["observer"] += 1
        raw = self._call(
            [image], prompt, self.config.observer_max_new_tokens
        )
        try:
            return extract_json_object(raw)
        except WorkspaceError:
            # A malformed question-blind write is missing evidence, not a
            # reason to discard the downstream QA example.  Fail closed on
            # state mutation by converting only this frame to the observer's
            # existing ``unavailable`` state.  The counter remains auditable.
            self.observer_parse_failures += 1
            return {"status": "unavailable", "reason": "unreadable"}

    @staticmethod
    def _images(context: Mapping[str, Any]) -> list[Image.Image]:
        images = []
        for item in context.get("model_visible_evidence") or []:
            image = item.get("image") if isinstance(item, Mapping) else None
            if not isinstance(image, Image.Image):
                raise WorkspaceError("answer evidence contains a non-PIL image")
            images.append(image)
        for item in context.get("model_visible_option_images") or []:
            image = item.get("image") if isinstance(item, Mapping) else None
            label = item.get("option_label") if isinstance(item, Mapping) else None
            if not isinstance(image, Image.Image) or not _LABEL.fullmatch(str(label or "")):
                raise WorkspaceError("visible option evidence is malformed")
            images.append(image)
        return images

    @staticmethod
    def _answer_prompt(context: Mapping[str, Any]) -> str:
        return compiled_answer_prompt(context)

    def _compiled_answer(self, context: Mapping[str, Any]) -> str:
        labels = {str(label).upper() for label in (context.get("visible_options") or {})}
        self.calls["answer"] += 1
        raw = self._call(
            self._images(context),
            self._answer_prompt(context),
            self.config.answer_max_new_tokens,
        )
        return extract_option_label(raw, labels)

    def _controller(self, context: Mapping[str, Any]) -> dict[str, Any]:
        prompt = f"""You are a bounded spatial-video controller. Use only the visible question, options, public budget, prior tool transcript, and ordered causal frames.
Return exactly one JSON object and no markdown. Either return {{"kind":"answer","answer":"A"}} with a visible option letter, or return {{"kind":"tool","action":{{"executor":"one granted executor","subject":"visible entity or place when required"}}}}.
Granted executors: {json.dumps(context.get('allowed_output', {}).get('tool', {}).get('granted_executors', []))}
Question: {context.get('visible_question')}
Options: {json.dumps(context.get('visible_options'), ensure_ascii=False, sort_keys=True)}
Public budget: {json.dumps(context.get('public_budget'), sort_keys=True)}
Tool transcript: {json.dumps(context.get('tool_transcript') or [], ensure_ascii=False, sort_keys=True)}"""
        self.calls["controller"] += 1
        raw = self._call(
            self._images(context), prompt, self.config.controller_max_new_tokens
        )
        return extract_json_object(raw)

    def _question_aware_runtime_controller(
        self, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Choose a generic evidence/state/geometry action or answer.

        The action set is identical across samples.  Question awareness is
        intentional: task blindness forbids benchmark-category routing, not
        active evidence selection from the visible question.
        """

        prompt = f"""You are a causal spatial-video agent. Use the visible question and options to decide whether the current evidence is sufficient. If it is not, invoke one generic tool from the shared allowlist.
Return exactly one JSON object and no markdown.
Tool form: {{"kind":"tool","tool":"one allowed tool","arguments":{{...}}}}
Answer form: {{"kind":"answer","answer":"A"}}
Never use benchmark categories, hidden annotations, future frames, or option position as evidence.
When option images are present, they follow the causal video evidence in visible option-label order.
Before answering, compare the requested operation with the generic tool purposes. If a tool can return a relevant fact that is absent from the transcript, call the single most relevant tool. Never repeat an identical tool call; answer when the visible evidence and returned facts are sufficient or the relevant state is unavailable.
Allowed tools: {json.dumps(context.get('allowed_tools') or [])}
Typed tool schemas: {json.dumps(context.get('allowed_tool_schemas') or {}, ensure_ascii=False, sort_keys=True)}
Runtime budget: {json.dumps(context.get('runtime_budget') or {}, sort_keys=True)}
Question-blind causal workspace index: {json.dumps(context.get('workspace_index') or {}, ensure_ascii=False, sort_keys=True)}
Question: {context.get('visible_question')}
Options: {json.dumps(context.get('visible_options'), ensure_ascii=False, sort_keys=True)}
Tool transcript: {json.dumps(context.get('tool_transcript') or [], ensure_ascii=False, sort_keys=True)}"""
        self.calls["controller"] += 1
        raw = self._call(
            self._images(context), prompt, self.config.controller_max_new_tokens
        )
        try:
            decision = extract_json_object(raw)
        except WorkspaceError:
            # Syntax failure is not evidence that the visual decision is
            # invalid.  Fall back to the checkpoint's native direct-answer
            # interface; semantic action validation remains strict whenever a
            # tool object was recovered.
            self.controller_parse_fallbacks += 1
            decision = {"kind": "answer", "answer": ""}
        if str(decision.get("kind") or "") == "answer":
            self.calls["answer"] += 1
            answer_raw = self._call(
                self._images(context),
                self._answer_prompt(context),
                self.config.answer_max_new_tokens,
            )
            labels = {
                str(label).upper()
                for label in (context.get("visible_options") or {})
            }
            return {
                "kind": "answer",
                "answer": extract_option_label(answer_raw, labels),
            }
        return decision

    def __call__(self, context: dict[str, Any]) -> str | Mapping[str, Any]:
        protocol = str(context.get("protocol") or "")
        if protocol == OBSERVER_FORMAT:
            return self._observe(context)
        if protocol == COMPILED_FORMAT:
            return self._compiled_answer(context)
        if protocol == AGENT_FORMAT:
            return self._controller(context)
        if protocol == QUESTION_AWARE_RUNTIME_FORMAT:
            return self._question_aware_runtime_controller(context)
        raise WorkspaceError(f"unsupported same-checkpoint context: {protocol!r}")

    def audit(self) -> dict[str, Any]:
        body = {
            "format": FORMAT,
            "config": asdict(self.config),
            "calls": dict(self.calls),
            "observer_parse_failures": self.observer_parse_failures,
            "controller_parse_fallbacks": self.controller_parse_fallbacks,
            "one_loaded_checkpoint_for_all_calls": True,
            "observer_question_blind": True,
            "runtime_controller_question_aware": True,
            "runtime_tools_task_and_benchmark_blind": True,
            "external_generative_model_or_api": False,
        }
        return {**body, "audit_sha256": _canonical_sha256(body)}
