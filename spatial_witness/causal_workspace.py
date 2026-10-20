"""Benchmark-blind causal state workspace for a spatial agent harness.

The trained VLM remains responsible for visual semantics and the final answer.
This module provides stateful operations that cannot be recovered reliably in
one answer forward pass: persistent object memory, place/event history,
observed topology, and deterministic SE(2) pose integration.

Every state update is chronological and bound to source evidence.  Query
actions expose neither benchmark metadata nor answer labels, and executors
return facts rather than multiple-choice answers.  The module is experimental
until each executor passes the paired promotion gates in
``docs/agent_harness.md``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


PROTOCOL = "spatial_witness.causal_workspace.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_EXECUTORS = frozenset(
    {
        "retrieve_last_seen",
        "count_place_visits",
        "query_current_place_state",
        "query_topology_neighbors",
        "query_topology_path",
        "query_relative_pose",
        "query_entity_relative_pose",
        "query_camera_pose",
    }
)
_PACKET_MODES = frozenset({"state_and_evidence", "evidence_only"})


class WorkspaceError(ValueError):
    """Raised when an update or query violates the causal tool contract."""


def uncalibrated_motion_labels(
    local_dx: float, local_dy: float, delta_yaw: float
) -> dict[str, str]:
    """Map an uncalibrated ego delta into qualitative labels, never MCQ letters."""

    if delta_yaw > 0.35:
        heading = "net_left_in_optical_flow_coordinates"
    elif delta_yaw < -0.35:
        heading = "net_right_in_optical_flow_coordinates"
    else:
        heading = "mostly_straight_in_optical_flow_coordinates"
    if abs(local_dy) >= abs(local_dx):
        translation = (
            "left_of_reference_heading"
            if local_dy > 0
            else "right_of_reference_heading"
        )
    else:
        translation = (
            "forward_of_reference_pose"
            if local_dx >= 0
            else "backward_of_reference_pose"
        )
    return {
        "uncalibrated_heading_change": heading,
        "uncalibrated_ego_translation": translation,
        "yaw_sign_convention": (
            "positive_yaw_is_counterclockwise_in_workspace_coordinates"
        ),
    }


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkspaceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise WorkspaceError(f"{name} must be finite")
    return result


def _key(value: Any, name: str) -> str:
    result = " ".join(str(value or "").strip().casefold().split())
    if not result:
        raise WorkspaceError(f"{name} must be nonempty")
    return result


@dataclass(frozen=True)
class EvidenceRef:
    """One immutable reference to actor-visible causal evidence."""

    source_id: str
    frame_index: int
    timestamp: float
    pixel_sha256: str

    def __post_init__(self) -> None:
        if not str(self.source_id).strip():
            raise WorkspaceError("evidence source_id must be nonempty")
        if isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise WorkspaceError("evidence frame_index must be nonnegative")
        _finite(self.timestamp, "evidence timestamp")
        if self.timestamp < 0:
            raise WorkspaceError("evidence timestamp must be nonnegative")
        if not _SHA256.fullmatch(str(self.pixel_sha256)):
            raise WorkspaceError("evidence pixel_sha256 must be lowercase SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectObservation:
    entity: str
    attributes: tuple[tuple[str, str], ...]
    world_position: tuple[float, float] | None
    place: str | None
    evidence: EvidenceRef


@dataclass(frozen=True)
class PlaceVisit:
    place: str
    ordinal: int
    evidence: EvidenceRef


@dataclass(frozen=True)
class Transition:
    source: str
    target: str
    evidence: EvidenceRef


@dataclass(frozen=True)
class PoseNode:
    frame_index: int
    timestamp: float
    x: float
    y: float
    yaw: float
    evidence: EvidenceRef


@dataclass(frozen=True)
class WorkspaceAction:
    executor: str
    subject: str | None = None
    start: str | None = None
    goal: str | None = None
    from_frame: int | None = None
    to_frame: int | None = None
    as_of: float | None = None
    since: float | None = None


@dataclass(frozen=True)
class WorkspaceBudget:
    maximum_executor_calls: int = 2
    maximum_evidence_refs: int = 32

    def __post_init__(self) -> None:
        if self.maximum_executor_calls < 1 or self.maximum_evidence_refs < 1:
            raise WorkspaceError("workspace budgets must be positive")


@dataclass(frozen=True)
class ExecutorResult:
    status: str
    executor: str
    facts: tuple[dict[str, Any], ...]
    evidence: tuple[EvidenceRef, ...]
    state_version: int
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "protocol": PROTOCOL,
            "status": self.status,
            "executor": self.executor,
            "facts": list(self.facts),
            "evidence": [item.to_dict() for item in self.evidence],
            "state_version": self.state_version,
            "answer": None,
        }
        if self.reason is not None:
            output["reason"] = self.reason
        return output


def parse_action(value: str | Mapping[str, Any]) -> WorkspaceAction:
    """Parse one flat, benchmark-blind executor request and reject extras."""

    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise WorkspaceError("workspace action is not valid JSON") from error
    else:
        payload = dict(value)
    if not isinstance(payload, dict):
        raise WorkspaceError("workspace action must be one JSON object")
    allowed = {
        "executor",
        "subject",
        "start",
        "goal",
        "from_frame",
        "to_frame",
        "as_of",
        "since",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise WorkspaceError(f"unknown workspace action fields: {unknown}")
    executor = str(payload.get("executor") or "")
    if executor not in SUPPORTED_EXECUTORS:
        raise WorkspaceError(f"unsupported workspace executor: {executor!r}")

    def optional_key(name: str) -> str | None:
        return _key(payload[name], name) if payload.get(name) is not None else None

    def optional_frame(name: str) -> int | None:
        raw = payload.get(name)
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise WorkspaceError(f"{name} must be a nonnegative integer")
        return raw

    action = WorkspaceAction(
        executor=executor,
        subject=optional_key("subject"),
        start=optional_key("start"),
        goal=optional_key("goal"),
        from_frame=optional_frame("from_frame"),
        to_frame=optional_frame("to_frame"),
        as_of=(
            _finite(payload["as_of"], "as_of")
            if payload.get("as_of") is not None
            else None
        ),
        since=(
            _finite(payload["since"], "since")
            if payload.get("since") is not None
            else None
        ),
    )
    required: dict[str, tuple[str, ...]] = {
        "retrieve_last_seen": ("subject",),
        "count_place_visits": ("subject",),
        "query_current_place_state": (),
        "query_topology_neighbors": ("subject",),
        "query_topology_path": ("start", "goal"),
        "query_relative_pose": ("from_frame", "to_frame"),
        "query_entity_relative_pose": ("subject",),
        "query_camera_pose": (),
    }
    missing = [name for name in required[executor] if getattr(action, name) is None]
    if missing:
        raise WorkspaceError(f"{executor} is missing fields: {missing}")
    relevant = {
        "retrieve_last_seen": {"executor", "subject", "as_of"},
        "count_place_visits": {"executor", "subject", "as_of"},
        "query_current_place_state": {"executor", "as_of"},
        "query_topology_neighbors": {"executor", "subject", "as_of"},
        "query_topology_path": {"executor", "start", "goal", "as_of"},
        "query_relative_pose": {
            "executor", "from_frame", "to_frame", "as_of"
        },
        "query_entity_relative_pose": {"executor", "subject", "as_of"},
        "query_camera_pose": {"executor", "as_of", "since"},
    }
    unrelated = sorted(
        key for key in payload if key not in relevant[executor]
    )
    if unrelated:
        raise WorkspaceError(f"{executor} contains unrelated fields: {unrelated}")
    return action


class CausalSpatialWorkspace:
    """Persistent external state updated only in chronological order."""

    def __init__(self) -> None:
        self.current_time = -1.0
        self.state_version = 0
        self._objects: dict[str, list[ObjectObservation]] = defaultdict(list)
        self._visits: dict[str, list[PlaceVisit]] = defaultdict(list)
        self._transitions: list[Transition] = []
        self._current_place: str | None = None
        self._poses: dict[int, PoseNode] = {}
        self._last_pose: PoseNode | None = None

    def _advance(self, evidence: EvidenceRef) -> None:
        if evidence.timestamp < self.current_time:
            raise WorkspaceError(
                "state updates must be chronological: "
                f"{evidence.timestamp} < {self.current_time}"
            )
        self.current_time = evidence.timestamp
        self.state_version += 1

    def advance_causal_horizon(self, timestamp: float) -> None:
        """Advance the visible prefix clock without inventing a state fact.

        A fixed-stride observer will normally process its last frame before
        the question boundary.  Frames can also be rejected as unavailable.
        Neither case should make a later, still-causal question time invalid,
        but the clock advance must not increment ``state_version`` or create
        evidence.  Facts remain bounded by their own evidence timestamps.
        """

        value = _finite(timestamp, "causal horizon")
        if value < 0 or value < self.current_time:
            raise WorkspaceError(
                "causal horizon must advance monotonically from "
                f"{self.current_time}"
            )
        self.current_time = value

    def observe_entity(
        self,
        entity: str,
        evidence: EvidenceRef,
        *,
        attributes: Mapping[str, Any] | None = None,
        world_position: Sequence[float] | None = None,
        place: str | None = None,
    ) -> None:
        entity = _key(entity, "entity")
        normalized = tuple(
            sorted(
                (_key(key, "attribute name"), str(value))
                for key, value in (attributes or {}).items()
            )
        )
        position = None
        if world_position is not None:
            if (
                isinstance(world_position, (str, bytes))
                or not isinstance(world_position, Sequence)
                or len(world_position) != 2
            ):
                raise WorkspaceError("entity world_position must contain x and y")
            position = (
                _finite(world_position[0], "entity world x"),
                _finite(world_position[1], "entity world y"),
            )
        normalized_place = _key(place, "entity place") if place is not None else None
        self._advance(evidence)
        self._objects[entity].append(
            ObjectObservation(entity, normalized, position, normalized_place, evidence)
        )

    def observe_place(self, place: str, evidence: EvidenceRef) -> None:
        place = _key(place, "place")
        self._advance(evidence)
        if place == self._current_place:
            return
        if self._current_place is not None:
            self._transitions.append(Transition(self._current_place, place, evidence))
        visits = self._visits[place]
        visits.append(PlaceVisit(place, len(visits) + 1, evidence))
        self._current_place = place

    def integrate_pose(
        self,
        evidence: EvidenceRef,
        *,
        local_dx: float = 0.0,
        local_dy: float = 0.0,
        delta_yaw: float = 0.0,
    ) -> PoseNode:
        """Compose a local SE(2) delta into the allocentric pose workspace."""

        if evidence.frame_index in self._poses:
            raise WorkspaceError(f"duplicate pose frame: {evidence.frame_index}")
        dx = _finite(local_dx, "local_dx")
        dy = _finite(local_dy, "local_dy")
        dyaw = _finite(delta_yaw, "delta_yaw")
        previous = self._last_pose
        if previous is None:
            x, y, yaw = 0.0, 0.0, 0.0
            if any(abs(value) > 1e-12 for value in (dx, dy, dyaw)):
                raise WorkspaceError("the first pose must initialize the origin")
        else:
            x = previous.x + math.cos(previous.yaw) * dx - math.sin(previous.yaw) * dy
            y = previous.y + math.sin(previous.yaw) * dx + math.cos(previous.yaw) * dy
            yaw = math.atan2(
                math.sin(previous.yaw + dyaw), math.cos(previous.yaw + dyaw)
            )
        self._advance(evidence)
        node = PoseNode(
            evidence.frame_index, evidence.timestamp, x, y, yaw, evidence
        )
        self._poses[node.frame_index] = node
        self._last_pose = node
        return node

    def _query_time(self, as_of: float | None) -> float:
        result = self.current_time if as_of is None else _finite(as_of, "as_of")
        if result < 0 or result > self.current_time:
            raise WorkspaceError(
                f"query time must be inside the causal prefix [0,{self.current_time}]"
            )
        return result

    def _unavailable(self, executor: str, reason: str) -> ExecutorResult:
        return ExecutorResult(
            "unavailable", executor, (), (), self.state_version, reason
        )

    def public_inventory(
        self,
        *,
        as_of: float | None = None,
        maximum_entities: int = 64,
        maximum_places: int = 32,
    ) -> dict[str, Any]:
        """Return a bounded, answer-blind index over causal workspace state.

        A controller cannot reliably call a keyed memory executor unless it
        knows the canonical entity and place names written by the independent
        observer.  This index exposes only names, coverage and provenance
        coordinates already present in the causal workspace.  It contains no
        question, benchmark metadata, option, prediction or answer field.
        """

        if (
            isinstance(maximum_entities, bool)
            or not isinstance(maximum_entities, int)
            or maximum_entities < 1
            or isinstance(maximum_places, bool)
            or not isinstance(maximum_places, int)
            or maximum_places < 1
        ):
            raise WorkspaceError("inventory limits must be positive integers")
        cutoff = self._query_time(as_of)
        entities = []
        for name, observations in self._objects.items():
            causal = [
                item for item in observations if item.evidence.timestamp <= cutoff
            ]
            if not causal:
                continue
            latest = causal[-1]
            entities.append(
                {
                    "name": name,
                    "observation_count": len(causal),
                    "last_seen_frame": latest.evidence.frame_index,
                    "last_seen_timestamp": latest.evidence.timestamp,
                    "has_metric_position": latest.world_position is not None,
                }
            )
        entities.sort(
            key=lambda item: (
                -float(item["last_seen_timestamp"]),
                str(item["name"]),
            )
        )

        places = []
        for name, visits in self._visits.items():
            causal = [item for item in visits if item.evidence.timestamp <= cutoff]
            if not causal:
                continue
            latest = causal[-1]
            places.append(
                {
                    "name": name,
                    "visit_count": len(causal),
                    "last_entry_frame": latest.evidence.frame_index,
                    "last_entry_timestamp": latest.evidence.timestamp,
                }
            )
        places.sort(
            key=lambda item: (
                -float(item["last_entry_timestamp"]),
                str(item["name"]),
            )
        )
        causal_transitions = [
            item
            for item in self._transitions
            if item.evidence.timestamp <= cutoff
        ]
        causal_poses = [
            item for item in self._poses.values() if item.timestamp <= cutoff
        ]
        return {
            "status": "available",
            "as_of": cutoff,
            "state_version": self.state_version,
            "entities": entities[:maximum_entities],
            "entity_count": len(entities),
            "entities_truncated": len(entities) > maximum_entities,
            "places": places[:maximum_places],
            "place_count": len(places),
            "places_truncated": len(places) > maximum_places,
            "topology_transition_count": len(causal_transitions),
            "pose_node_count": len(causal_poses),
        }

    def retrieve_last_seen(
        self, entity: str, *, as_of: float | None = None
    ) -> ExecutorResult:
        entity = _key(entity, "entity")
        cutoff = self._query_time(as_of)
        candidates = [
            item for item in self._objects.get(entity, [])
            if item.evidence.timestamp <= cutoff
        ]
        if not candidates:
            return self._unavailable("retrieve_last_seen", "entity_not_observed")
        item = candidates[-1]
        return ExecutorResult(
            "ok",
            "retrieve_last_seen",
            ({
                "entity": entity,
                "last_seen_timestamp": item.evidence.timestamp,
                "last_seen_frame": item.evidence.frame_index,
                "attributes": dict(item.attributes),
                "world_position": (
                    list(item.world_position)
                    if item.world_position is not None else None
                ),
                "place": item.place,
            },),
            (item.evidence,),
            self.state_version,
        )

    def count_place_visits(
        self, place: str, *, as_of: float | None = None
    ) -> ExecutorResult:
        place = _key(place, "place")
        cutoff = self._query_time(as_of)
        visits = [
            item for item in self._visits.get(place, [])
            if item.evidence.timestamp <= cutoff
        ]
        if not visits:
            return self._unavailable("count_place_visits", "place_not_observed")
        return ExecutorResult(
            "ok",
            "count_place_visits",
            ({"place": place, "visit_count": len(visits)},),
            tuple(item.evidence for item in visits),
            self.state_version,
        )

    def query_current_place_state(
        self, *, as_of: float | None = None
    ) -> ExecutorResult:
        """Return anonymous current-place identity and its causal visit count.

        This executor is useful with question-blind visual place recognition:
        the answer model need not know the observer's internal place ID in
        order to ask whether the currently visible location was revisited.
        """

        cutoff = self._query_time(as_of)
        causal_visits = sorted(
            (
                visit
                for visits in self._visits.values()
                for visit in visits
                if visit.evidence.timestamp <= cutoff
            ),
            key=lambda item: (
                item.evidence.timestamp,
                item.evidence.frame_index,
                item.place,
            ),
        )
        if not causal_visits:
            return self._unavailable(
                "query_current_place_state", "no_place_observation"
            )
        current = causal_visits[-1]
        entries = [item for item in causal_visits if item.place == current.place]
        return ExecutorResult(
            "ok",
            "query_current_place_state",
            ({
                "anonymous_place_id": current.place,
                "visit_count": len(entries),
                "current_visit_ordinal": current.ordinal,
                "current_entry_frame": current.evidence.frame_index,
                "current_entry_timestamp": current.evidence.timestamp,
                "is_revisit": len(entries) > 1,
            },),
            tuple(item.evidence for item in entries),
            self.state_version,
        )

    def query_topology_neighbors(
        self, place: str, *, as_of: float | None = None
    ) -> ExecutorResult:
        place = _key(place, "place")
        cutoff = self._query_time(as_of)
        transitions = [
            item for item in self._transitions
            if (item.source == place or item.target == place)
            and item.evidence.timestamp <= cutoff
        ]
        if not transitions:
            return self._unavailable(
                "query_topology_neighbors", "no_observed_connectivity_transition"
            )
        first_by_target: dict[str, Transition] = {}
        for item in transitions:
            neighbor = item.target if item.source == place else item.source
            first_by_target.setdefault(neighbor, item)
        ordered_neighbors = sorted(first_by_target)
        ordered = [first_by_target[key] for key in ordered_neighbors]
        return ExecutorResult(
            "ok",
            "query_topology_neighbors",
            ({"place": place, "observed_neighbors": ordered_neighbors},),
            tuple(item.evidence for item in ordered),
            self.state_version,
        )

    def query_topology_path(
        self, start: str, goal: str, *, as_of: float | None = None
    ) -> ExecutorResult:
        start, goal = _key(start, "start"), _key(goal, "goal")
        cutoff = self._query_time(as_of)
        # A traversal through an indoor doorway establishes physical
        # connectivity in both directions even if only one traversal direction
        # has been observed.  Event chronology remains attached to the single
        # causal transition evidence.
        adjacency: dict[str, list[tuple[str, EvidenceRef]]] = defaultdict(list)
        for item in self._transitions:
            if item.evidence.timestamp <= cutoff:
                adjacency[item.source].append((item.target, item.evidence))
                adjacency[item.target].append((item.source, item.evidence))
        for edges in adjacency.values():
            edges.sort(key=lambda item: (item[0], item[1].timestamp))
        queue: deque[tuple[str, tuple[str, ...], tuple[EvidenceRef, ...]]] = deque(
            [(start, (start,), ())]
        )
        visited = {start}
        while queue:
            node, path, evidence = queue.popleft()
            if node == goal:
                return ExecutorResult(
                    "ok",
                    "query_topology_path",
                    ({"start": start, "goal": goal, "path": list(path), "hops": len(path) - 1},),
                    evidence,
                    self.state_version,
                )
            for target, edge_evidence in adjacency.get(node, []):
                if target not in visited:
                    visited.add(target)
                    queue.append(
                        (target, path + (target,), evidence + (edge_evidence,))
                    )
        return self._unavailable("query_topology_path", "no_observed_connectivity_path")

    def query_relative_pose(
        self,
        from_frame: int,
        to_frame: int,
        *,
        as_of: float | None = None,
    ) -> ExecutorResult:
        cutoff = self._query_time(as_of)
        first, second = self._poses.get(from_frame), self._poses.get(to_frame)
        if (
            first is None
            or second is None
            or first.timestamp > cutoff
            or second.timestamp > cutoff
        ):
            return self._unavailable("query_relative_pose", "pose_node_unavailable")
        world_dx, world_dy = second.x - first.x, second.y - first.y
        dx = math.cos(first.yaw) * world_dx + math.sin(first.yaw) * world_dy
        dy = -math.sin(first.yaw) * world_dx + math.cos(first.yaw) * world_dy
        dyaw = math.atan2(
            math.sin(second.yaw - first.yaw), math.cos(second.yaw - first.yaw)
        )
        lower, upper = sorted((first.timestamp, second.timestamp))
        provenance = tuple(
            node.evidence
            for node in sorted(self._poses.values(), key=lambda item: item.timestamp)
            if lower <= node.timestamp <= upper
        )
        return ExecutorResult(
            "ok",
            "query_relative_pose",
            ({
                "from_frame": from_frame,
                "to_frame": to_frame,
                "local_dx": dx,
                "local_dy": dy,
                "delta_yaw_radians": dyaw,
            },),
            provenance,
            self.state_version,
        )

    def query_camera_pose(
        self, *, as_of: float | None = None, since: float | None = None
    ) -> ExecutorResult:
        """Return the causal camera trajectory from a reference pose to now.

        Frame indices are owned by the writer. The query does not take a
        question, option, or named entity. Facts are uncalibrated SE(2)
        summaries, not multiple-choice answers. ``since`` selects an already
        written snapshot timestamp; it is not a benchmark turn id.
        """

        cutoff = self._query_time(as_of)
        poses = [
            item for item in self._poses.values() if item.timestamp <= cutoff
        ]
        if len(poses) < 2:
            return self._unavailable("query_camera_pose", "pose_graph_too_sparse")
        poses.sort(key=lambda item: (item.timestamp, item.frame_index))
        origin = poses[0]
        if since is not None:
            since_ts = _finite(since, "since")
            if since_ts > cutoff:
                return self._unavailable("query_camera_pose", "since_after_query_time")
            referenced = [
                item for item in poses if item.timestamp <= since_ts
            ]
            if not referenced:
                return self._unavailable("query_camera_pose", "since_pose_unavailable")
            origin = referenced[-1]
            poses = [item for item in poses if item.timestamp >= origin.timestamp]
            if len(poses) < 2:
                return self._unavailable("query_camera_pose", "pose_graph_too_sparse")
        latest = poses[-1]
        relative = self.query_relative_pose(
            origin.frame_index, latest.frame_index, as_of=cutoff
        )
        if relative.status != "ok":
            return relative
        path_length = 0.0
        net_abs_yaw = 0.0
        turns = 0
        previous = origin
        for node in poses[1:]:
            step = math.hypot(node.x - previous.x, node.y - previous.y)
            path_length += step
            dyaw = abs(
                math.atan2(
                    math.sin(node.yaw - previous.yaw),
                    math.cos(node.yaw - previous.yaw),
                )
            )
            net_abs_yaw += dyaw
            if dyaw > 0.2:
                turns += 1
            previous = node
        fact = dict(relative.facts[0])
        labels = uncalibrated_motion_labels(
            float(fact["local_dx"]),
            float(fact["local_dy"]),
            float(fact["delta_yaw_radians"]),
        )
        curvature = (
            "many_turns"
            if turns >= 4
            else "some_turns"
            if turns >= 1
            else "few_or_no_turns"
        )
        fact.update(
            {
                "pose_node_count": len(poses),
                "path_length": path_length,
                "net_abs_yaw_radians": net_abs_yaw,
                "turn_count": turns,
                "latest_x": latest.x,
                "latest_y": latest.y,
                "latest_yaw_radians": latest.yaw,
                "reference_timestamp": origin.timestamp,
                "uncalibrated_path_shape": curvature,
                "metric_pose_claimed": False,
                **labels,
            }
        )
        return ExecutorResult(
            "ok",
            "query_camera_pose",
            (fact,),
            relative.evidence,
            self.state_version,
        )

    def query_entity_relative_pose(
        self, entity: str, *, as_of: float | None = None
    ) -> ExecutorResult:
        """Transform one persistent allocentric object position into ego SE(2).

        The writer, not this executor, owns semantic identity and metric
        position estimation.  The executor only performs a deterministic
        reference-frame transform over causal state and returns provenance for
        both the object observation and camera pose.
        """

        entity = _key(entity, "entity")
        cutoff = self._query_time(as_of)
        objects = [
            item
            for item in self._objects.get(entity, [])
            if item.evidence.timestamp <= cutoff and item.world_position is not None
        ]
        poses = [
            item for item in self._poses.values() if item.timestamp <= cutoff
        ]
        if not objects:
            return self._unavailable(
                "query_entity_relative_pose", "entity_metric_position_unavailable"
            )
        if not poses:
            return self._unavailable(
                "query_entity_relative_pose", "camera_pose_unavailable"
            )
        item = objects[-1]
        pose = max(poses, key=lambda value: (value.timestamp, value.frame_index))
        assert item.world_position is not None
        world_dx = item.world_position[0] - pose.x
        world_dy = item.world_position[1] - pose.y
        local_x = math.cos(pose.yaw) * world_dx + math.sin(pose.yaw) * world_dy
        local_y = -math.sin(pose.yaw) * world_dx + math.cos(pose.yaw) * world_dy
        bearing = math.atan2(local_x, local_y)
        if -math.pi / 4 <= bearing < math.pi / 4:
            sector = "front"
        elif math.pi / 4 <= bearing < 3 * math.pi / 4:
            sector = "right"
        elif -3 * math.pi / 4 <= bearing < -math.pi / 4:
            sector = "left"
        else:
            sector = "behind"
        provenance = (item.evidence, pose.evidence)
        if item.evidence == pose.evidence:
            provenance = (item.evidence,)
        return ExecutorResult(
            "ok",
            "query_entity_relative_pose",
            ({
                "entity": entity,
                "camera_frame": pose.frame_index,
                "local_x": local_x,
                "local_y": local_y,
                "distance": math.hypot(local_x, local_y),
                "bearing_radians": bearing,
                "egocentric_sector": sector,
                "source_place": item.place,
            },),
            provenance,
            self.state_version,
        )

    def execute(self, value: str | Mapping[str, Any]) -> ExecutorResult:
        action = parse_action(value)
        if action.executor == "retrieve_last_seen":
            assert action.subject is not None
            return self.retrieve_last_seen(action.subject, as_of=action.as_of)
        if action.executor == "count_place_visits":
            assert action.subject is not None
            return self.count_place_visits(action.subject, as_of=action.as_of)
        if action.executor == "query_current_place_state":
            return self.query_current_place_state(as_of=action.as_of)
        if action.executor == "query_topology_neighbors":
            assert action.subject is not None
            return self.query_topology_neighbors(action.subject, as_of=action.as_of)
        if action.executor == "query_topology_path":
            assert action.start is not None and action.goal is not None
            return self.query_topology_path(
                action.start, action.goal, as_of=action.as_of
            )
        if action.executor == "query_entity_relative_pose":
            assert action.subject is not None
            return self.query_entity_relative_pose(
                action.subject, as_of=action.as_of
            )
        if action.executor == "query_camera_pose":
            return self.query_camera_pose(as_of=action.as_of, since=action.since)
        assert action.executor == "query_relative_pose"
        assert action.from_frame is not None and action.to_frame is not None
        return self.query_relative_pose(
            action.from_frame, action.to_frame, as_of=action.as_of
        )


def evidence(
    source_id: str, frame_index: int, timestamp: float, pixel_sha256: str
) -> EvidenceRef:
    """Small public constructor used by adapters and tests."""

    return EvidenceRef(source_id, frame_index, timestamp, pixel_sha256)


class BoundedWorkspaceSession:
    """Budgeted query interface between one VLM policy and a workspace.

    A session never predicts an answer.  It exposes compact executor packets,
    charges every returned evidence reference including repeated references,
    and binds each packet to canonical bytes.  This makes action-policy and
    answerer experiments auditable without granting the tool an answer path.
    """

    def __init__(
        self,
        workspace: CausalSpatialWorkspace,
        *,
        query_time: float,
        budget: WorkspaceBudget | None = None,
        packet_mode: str = "state_and_evidence",
    ) -> None:
        self.workspace = workspace
        self.query_time = _finite(query_time, "query_time")
        if self.query_time < 0 or self.query_time > workspace.current_time:
            raise WorkspaceError("session query_time is outside available causal state")
        self.budget = budget or WorkspaceBudget()
        if packet_mode not in _PACKET_MODES:
            raise WorkspaceError(
                f"unsupported workspace packet mode: {packet_mode!r}"
            )
        self.packet_mode = packet_mode
        self.executor_calls = 0
        self.evidence_refs_charged = 0
        self.transcript: list[dict[str, Any]] = []

    def public_budget(self) -> dict[str, int]:
        return {
            "maximum_executor_calls": self.budget.maximum_executor_calls,
            "executor_calls_used": self.executor_calls,
            "executor_calls_remaining": (
                self.budget.maximum_executor_calls - self.executor_calls
            ),
            "maximum_evidence_refs": self.budget.maximum_evidence_refs,
            "evidence_refs_charged": self.evidence_refs_charged,
            "evidence_refs_remaining": (
                self.budget.maximum_evidence_refs - self.evidence_refs_charged
            ),
        }

    def public_inventory(self) -> dict[str, Any]:
        """Expose the causal memory index only in the state treatment arm."""

        if self.packet_mode == "evidence_only":
            return {
                "status": "withheld_by_evidence_only_control",
                "as_of": self.query_time,
                "state_version": self.workspace.state_version,
            }
        return self.workspace.public_inventory(as_of=self.query_time)

    @staticmethod
    def _packet_hash(packet: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def execute(self, value: str | Mapping[str, Any]) -> dict[str, Any]:
        if self.executor_calls >= self.budget.maximum_executor_calls:
            raise WorkspaceError("workspace executor-call budget exceeded")
        action = parse_action(value)
        # A policy cannot query beyond the time at which its question arrived.
        if action.as_of is None:
            payload = {key: item for key, item in asdict(action).items() if item is not None}
            payload["as_of"] = self.query_time
            action = parse_action(payload)
        elif action.as_of > self.query_time:
            raise WorkspaceError("workspace action queries after session query_time")
        result = self.workspace.execute(
            {key: item for key, item in asdict(action).items() if item is not None}
        )
        charge = len(result.evidence)
        if self.evidence_refs_charged + charge > self.budget.maximum_evidence_refs:
            raise WorkspaceError("workspace evidence-reference budget exceeded")
        self.executor_calls += 1
        self.evidence_refs_charged += charge
        body = result.to_dict()
        # The evidence-only mode is a causal mechanism control: it preserves
        # the exact executor call, evidence identities, frame charge, and
        # budget while withholding the deterministic state computation.  A
        # state-and-evidence gain over this control therefore cannot be
        # attributed merely to seeing different frames.
        if self.packet_mode == "evidence_only":
            body["facts"] = []
        body["packet_mode"] = self.packet_mode
        body["query_time"] = self.query_time
        body["budget"] = self.public_budget()
        packet = {**body, "packet_sha256": self._packet_hash(body)}
        self.transcript.append(packet)
        return packet
