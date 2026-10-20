"""Compile visible spatial language into a benchmark-agnostic data plan.

The compiler is used while constructing and auditing SFT examples and by the
minimal deterministic harness. It accepts only actor-visible question and
option text; benchmark levels, task IDs, answers, evidence annotations,
predictions, and scores are not inputs. Only operators that independently pass
a paired harness gate may affect release-time observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class SpatialQueryPlan:
    operator: str
    state_views: tuple[str, ...]
    temporal_scope: str
    evidence_policy: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _match(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def _captured_subject(pattern: str, question: str) -> str | None:
    match = re.search(pattern, str(question or ""), re.IGNORECASE)
    if match is None:
        return None
    subject = " ".join(str(match.group("subject") or "").strip().split())
    subject = re.sub(r"^[\s'\"`]*(?:the|a|an|my|our)\s+", "", subject, flags=re.I)
    subject = re.sub(
        r"\s+(?:in (?:this|the) video|so far|in total|again)$",
        "",
        subject,
        flags=re.I,
    )
    subject = subject.strip(" \t\r\n?!.,;:'\"`")
    # Fail closed on clause-sized or malformed captures.  Entity resolution is
    # not an excuse to send the full benchmark question into a state key.
    if not subject or len(subject) > 80 or len(subject.split()) > 10:
        return None
    return subject


def observation_chunk_reference(text: str) -> int | None:
    """Parse a visible temporal reference to an earlier observation chunk.

    The writer keys snapshots by timestamp. The reader may recover the chunk
    index from visible language such as "end of turn 4" or "end of
    observation chunk 4". Benchmark category fields are not inputs.
    """

    match = re.search(
        r"\bend of(?: the)? (?:turn|observation chunk|chunk) (?P<a>\d+)\b|"
        r"\bend of the (?P<b>\d+) (?:turn|observation chunk|chunk)\b",
        str(text or ""),
        re.IGNORECASE,
    )
    if match is None:
        return None
    value = int(match.group("a") or match.group("b"))
    return value if value >= 1 else None


def _uncalibrated_units_cannot_answer(text: str) -> bool:
    return _match(
        r"\b(in meters|how many degrees|"
        r"in what (?:clockwise|counterclockwise) direction \(in degrees\))\b",
        text,
    )


def compile_spatial_query(
    question: str, options: Mapping[str, str] | None = None
) -> SpatialQueryPlan:
    """Return one auditable plan derived from model-visible language only."""

    option_text = " ".join(str(value) for value in (options or {}).values())
    text = " ".join((str(question or "") + " " + option_text).split())

    if _match(r"\b(map|trajectory|bird'?s[- ]eye|floor ?plan|path shape)\b", text):
        return SpatialQueryPlan(
            "match_trajectory_graph",
            ("event_ledger", "pose_graph", "topology_graph"),
            "full_prefix",
            "causal_pyramid",
            "visible_trajectory_or_map_language",
        )
    if _match(
        r"\b(shortest route|detour|reachab|connect(?:ed|ion|s)?|neighbor|"
        r"adjacent|topolog)\b",
        text,
    ):
        return SpatialQueryPlan(
            "query_topology_graph",
            ("event_ledger", "topology_graph"),
            "full_prefix",
            "causal_pyramid",
            "visible_connectivity_or_route_language",
        )
    if _match(
        r"\b(relative direction|cardinal direction|north|south|east|west|"
        r"initial viewing direction|facing direction|behind (?:the )?camera|"
        r"to (?:your|my|our) (?:left|right)(?: now)?|"
        r"(?:clockwise|counterclockwise) direction(?: \(in degrees\))?.{0,80}"
        r"from (?:your|my|our) (?:position|orientation))\b",
        text,
    ):
        return SpatialQueryPlan(
            "query_relative_pose",
            ("object_memory", "pose_graph"),
            "full_prefix",
            "causal_pyramid",
            "visible_coordinate_frame_language",
        )
    if _match(
        r"\b(?:same|different) (?:place|scene|room|object).{0,48}"
        r"(?:referenced moment|at \d+(?:\.\d+)? seconds?)\b",
        text,
    ):
        return SpatialQueryPlan(
            "compare_referenced_state",
            ("local_buffer", "event_ledger", "object_memory"),
            "referenced_pair",
            "referenced_pair",
            "visible_explicit_reference_time_language",
        )
    if _match(
        r"\b(how many times|revisit|return(?:ed|ing)?|been here before|"
        r"has (?:this|the current) (?:place|scene|room|location|area) "
        r"(?:appeared|been (?:seen|visited)|shown) (?:earlier|before)|"
        r"how many .{0,48}(?:so far|in total|throughout|have (?:you|we) seen)|"
        r"which visit .{0,48}(?:is this|was this))\b",
        text,
    ):
        return SpatialQueryPlan(
            "count_place_events",
            ("event_ledger", "object_memory"),
            "full_prefix",
            "uniform",
            "visible_revisit_or_event_count_language",
        )
    if _match(
        r"\b(last (?:see|saw|seen|time)|previously|earlier|no longer visible|"
        r"before or after|immediately (?:before|after|beforehand)|"
        r"visit(?:ed)? before|before (?:entering|the current)|"
        r"after leaving|just (?:opened|passed)|in what order|chronological|"
        r"what came (?:first|next))\b",
        text,
    ):
        return SpatialQueryPlan(
            "retrieve_ordered_event",
            ("event_ledger", "object_memory"),
            "historical",
            "log_decay",
            "visible_historical_binding_language",
        )
    if _match(
        r"\b(rotat(?:e|ed|ing)|yaw(?:ing)?|moving (?:forward|backward)|"
        r"relative speed|camera motion|object motion|opening|closing|"
        r"ascending|descending)\b",
        text,
    ):
        return SpatialQueryPlan(
            "estimate_temporal_derivative",
            ("local_buffer", "event_ledger"),
            "recent_motion",
            "log_decay",
            "visible_motion_or_state_change_language",
        )
    if _match(
        r"\b(current(?:ly)?|right now|at (?:this|the) moment|this segment|"
        r"distance|size|clearance|occlu|support(?:ed)?|contain(?:s|ed)?)\b",
        text,
    ):
        return SpatialQueryPlan(
            "inspect_local_state",
            ("local_buffer",),
            "current",
            "recent",
            "visible_local_state_language",
        )
    return SpatialQueryPlan(
        "answer_from_causal_prefix",
        ("local_buffer", "event_ledger", "object_memory"),
        "unspecified",
        "uniform",
        "no_specialized_visible_semantics",
    )


def compile_workspace_action(
    question: str, options: Mapping[str, str] | None = None
) -> dict[str, Any] | None:
    """Compile only universally valid anonymous-state executor actions.

    Named-place and named-object actions require semantic entity resolution and
    therefore remain decisions for the same VLM controller. Revisit/event-count
    questions can safely query the current anonymous visual place: the tool
    returns state rather than selecting an option, and the answerer may ignore
    it when the question refers to another location.
    """

    text = " ".join(
        (str(question or "") + " " + " ".join(str(value) for value in (options or {}).values())).split()
    )
    # These phrases describe an ordering relation, not the yes/no sense of
    # "have we been here before". Keep them out of the anonymous revisit
    # executor even if another token happens to match the broad compiler.
    if _match(
        r"\b(before or after|in what order|which .{0,64} visited first|"
        r"visited (?:immediately )?(?:before|after) .{1,64}|"
        r"last .{0,64} before (?:going|entering|leaving))\b",
        text,
    ):
        return None
    if not _uncalibrated_units_cannot_answer(text):
        since_chunk = observation_chunk_reference(text)
        if since_chunk is not None and _match(
            r"\b(rotated clockwise or counterclockwise|"
            r"orientation rotated|"
            r"move(?:d)? a certain distance "
            r"(?:left or right|forward or backward)|"
            r"did you move a certain distance)\b",
            text,
        ):
            return {"executor": "query_camera_pose"}
    plan = compile_spatial_query(question, options)
    # Anonymous visual loop closure is valid only for deictic scene identity.
    # A request to count visits to a named room, object, or semantic region
    # needs entity/region tracking and cannot be answered by viewpoint IDs.
    # Keep the distinction at the interface instead of asking the answerer to
    # paper over an executor whose fact has the wrong semantics.
    anonymous_current_scene = _match(
        r"\b(?:this same(?: [a-z-]+)? scene|this place|this scene|"
        r"the current (?:place|scene|view)|current view|been here before)\b",
        str(question or ""),
    )
    if plan.operator == "count_place_events" and anonymous_current_scene:
        return {"executor": "query_current_place_state"}

    if plan.operator == "count_place_events":
        place = None
        for pattern in (
            r"\bhow many times (?:did|have) (?:i|we|you) "
            r"(?:enter(?:ed)?|visit(?:ed)?|return(?:ed)? to) "
            r"(?P<subject>[^?]+?)(?:\?|$)",
            r"\bwhich visit to (?P<subject>[^?]+?) (?:is|was) this(?:\?|$)",
        ):
            place = _captured_subject(pattern, question)
            if place is not None:
                break
        if place is not None:
            return {"executor": "count_place_visits", "subject": place}

    if plan.operator == "retrieve_ordered_event":
        entity = None
        for pattern in (
            r"\bwhere did (?:i|we|you) last (?:see|notice|encounter) "
            r"(?P<subject>[^?]+?)(?:\?|$)",
            r"\bwhere was (?P<subject>[^?]+?) last seen(?:\?|$)",
            r"\b(?P<subject>[^?]+?) was last seen where(?:\?|$)",
        ):
            entity = _captured_subject(pattern, question)
            if entity is not None:
                break
        if entity is not None:
            return {"executor": "retrieve_last_seen", "subject": entity}

    if plan.operator == "match_trajectory_graph" or (
        plan.operator == "query_relative_pose"
        and _match(
            r"\b(map|trajectory|bird'?s[- ]eye|floor ?plan|path shape|"
            r"initial viewing direction|facing direction|"
            r"destination relative to the start|relative to the start|"
            r"camera heading|path taken)\b",
            text,
        )
        and not _match(
            r"\b(?:located|where is) .{0,80}relative to the camera\b",
            text,
        )
    ):
        return {"executor": "query_camera_pose"}
    return None
