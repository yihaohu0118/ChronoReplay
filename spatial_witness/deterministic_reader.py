"""Conservative readers for executable spatial workspace facts.

The state writer remains question blind.  These readers run only after a
question-derived, task-blind executor has returned a typed fact.  They resolve
an answer only when the visible options provide one unambiguous semantic
mapping; every other case falls back to the same-checkpoint VLM answerer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import re
from typing import Any


@dataclass(frozen=True)
class DeterministicAnswer:
    label: str
    executor: str
    fact: str
    value: bool | int | float | str
    rule: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _revisit_polarity(option: str) -> bool | None:
    text = _normalized(option)
    negative = (
        r"\bnew\b",
        r"\bfirst (?:visit|time|appearance)\b",
        r"\bnot (?:previously )?(?:seen|visited|shown)\b",
        r"\bnot .*\b(?:seen|visited|shown) before\b",
        r"\bnever (?:seen|visited|shown)\b",
    )
    positive = (
        r"\brevisit(?:ed)?\b",
        r"\bpreviously (?:seen|visited|shown)\b",
        r"\b(?:seen|visited|shown) (?:earlier|before)\b",
        r"\breturn(?:ed|ing)?\b",
    )
    if any(re.search(pattern, text) for pattern in negative):
        return False
    if any(re.search(pattern, text) for pattern in positive):
        return True
    return None


def resolve_deterministic_workspace_answer(
    *,
    options: Mapping[str, str],
    action: Mapping[str, Any] | None,
    transcript: Sequence[Mapping[str, Any]],
) -> DeterministicAnswer | None:
    """Resolve a typed fact only when its option semantics are unambiguous."""

    if not action or action.get("executor") != "query_current_place_state":
        return None
    facts = [
        fact
        for packet in transcript
        for fact in (packet.get("facts") or [])
        if isinstance(fact, Mapping)
    ]
    if len(facts) != 1 or not isinstance(facts[0].get("is_revisit"), bool):
        return None
    polarities = {
        str(label): _revisit_polarity(text) for label, text in options.items()
    }
    positive = [label for label, value in polarities.items() if value is True]
    negative = [label for label, value in polarities.items() if value is False]
    if len(positive) != 1 or len(negative) != 1 or len(options) != 2:
        return None
    value = bool(facts[0]["is_revisit"])
    return DeterministicAnswer(
        label=positive[0] if value else negative[0],
        executor="query_current_place_state",
        fact="is_revisit",
        value=value,
        rule="unambiguous_visible_revisit_polarity",
    )
