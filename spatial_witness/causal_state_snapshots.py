"""Query-independent persistent spatial-state snapshots.

The writer-facing API intentionally has no question, option, label, task, or
benchmark fields.  A new causal observation interval is committed once per
``(video_uid, query_timestamp)``.  Any number of later questions can read the
same immutable snapshot without invoking the writer again.

This module stores and validates state; it does not decide how a model turns
pixels into a state delta.  That callback receives only the previous state,
new observations, and their causal provenance.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from spatial_witness.causal_workspace import EvidenceRef, WorkspaceError


FORMAT = "spatial_witness.causal_state_snapshots.v1"
SNAPSHOT_FORMAT = "spatial_witness.causal_state_snapshot.v1"
_FORBIDDEN_WRITER_KEYS = frozenset(
    {
        "answer",
        "answer_index",
        "benchmark",
        "benchmark_category",
        "category",
        "correct_answer",
        "label",
        "option",
        "options",
        "question",
        "query",
        "score",
        "task_id",
    }
)
StateUpdate = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[EvidenceRef]],
    Mapping[str, Any],
]


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _clone(value: Any) -> Any:
    return json.loads(_canonical(value))


def _timestamp(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkspaceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise WorkspaceError(f"{name} must be finite and nonnegative")
    return result


def _video_uid(value: Any) -> str:
    result = str(value or "").strip()
    if not result:
        raise WorkspaceError("video_uid must be nonempty")
    return result


def _assert_writer_blind(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if key in _FORBIDDEN_WRITER_KEYS:
                raise WorkspaceError(
                    f"forbidden query/benchmark field in writer input/state: {path}.{key}"
                )
            _assert_writer_blind(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_writer_blind(child, f"{path}[{index}]")


def _chunk_sha256(
    observations: Sequence[Mapping[str, Any]], evidence: Sequence[EvidenceRef]
) -> str:
    return _sha256(
        {
            "observations": list(observations),
            "evidence": [item.to_dict() for item in evidence],
        }
    )


@dataclass(frozen=True)
class CausalStateSnapshot:
    video_uid: str
    query_timestamp: float
    state_version: int
    prior_state_sha256: str
    state_sha256: str
    observation_batch_sha256: str
    writer_implementation_sha256: str
    writer_invoked: bool
    evidence_provenance: tuple[dict[str, Any], ...]
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": SNAPSHOT_FORMAT,
            "video_uid": self.video_uid,
            "query_timestamp": self.query_timestamp,
            "state_version": self.state_version,
            "prior_state_sha256": self.prior_state_sha256,
            "state_sha256": self.state_sha256,
            "observation_batch_sha256": self.observation_batch_sha256,
            "writer_implementation_sha256": self.writer_implementation_sha256,
            "writer_invoked": self.writer_invoked,
            "evidence_provenance": _clone(list(self.evidence_provenance)),
            "state": _clone(self.state),
            "question_or_options_visible_to_writer": False,
        }

    def clone(self) -> "CausalStateSnapshot":
        return CausalStateSnapshot(
            video_uid=self.video_uid,
            query_timestamp=self.query_timestamp,
            state_version=self.state_version,
            prior_state_sha256=self.prior_state_sha256,
            state_sha256=self.state_sha256,
            observation_batch_sha256=self.observation_batch_sha256,
            writer_implementation_sha256=self.writer_implementation_sha256,
            writer_invoked=self.writer_invoked,
            evidence_provenance=tuple(_clone(list(self.evidence_provenance))),
            state=_clone(self.state),
        )


class CausalStateSnapshotStore:
    """Append-only per-video causal state with immutable query snapshots."""

    def __init__(
        self,
        *,
        writer_implementation_sha256: str,
        initial_state: Mapping[str, Any] | None = None,
    ) -> None:
        if len(writer_implementation_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in writer_implementation_sha256
        ):
            raise WorkspaceError("writer implementation hash must be lowercase SHA-256")
        initial = dict(initial_state or {})
        _assert_writer_blind(initial, "initial_state")
        _canonical(initial)
        self.writer_implementation_sha256 = writer_implementation_sha256
        self._initial_state = _clone(initial)
        self._snapshots: dict[tuple[str, float], CausalStateSnapshot] = {}
        self._timestamps: dict[str, list[float]] = {}
        self._latest_state: dict[str, dict[str, Any]] = {}
        self._versions: dict[str, int] = {}
        self.writer_calls = 0

    def advance(
        self,
        *,
        video_uid: str,
        query_timestamp: float,
        observations: Sequence[Mapping[str, Any]],
        evidence_provenance: Sequence[EvidenceRef],
        update_state: StateUpdate,
    ) -> CausalStateSnapshot:
        """Commit one new causal interval; an identical retry is idempotent."""

        uid = _video_uid(video_uid)
        timestamp = _timestamp(query_timestamp, "query_timestamp")
        if not observations:
            raise WorkspaceError("a new snapshot requires new observations")
        normalized_observations = [dict(item) for item in observations]
        if len(normalized_observations) != len(evidence_provenance):
            raise WorkspaceError("each observation requires one evidence reference")
        _assert_writer_blind(normalized_observations, "observations")
        _canonical(normalized_observations)
        evidence = tuple(evidence_provenance)
        if any(not isinstance(item, EvidenceRef) for item in evidence):
            raise WorkspaceError("evidence_provenance must contain EvidenceRef values")
        chunk_sha256 = _chunk_sha256(normalized_observations, evidence)
        key = (uid, timestamp)
        existing = self._snapshots.get(key)
        if existing is not None:
            if existing.observation_batch_sha256 != chunk_sha256:
                raise WorkspaceError(
                    "snapshot retry changed observations at an existing causal key"
                )
            return existing.clone()

        prior_timestamps = self._timestamps.get(uid, [])
        previous_timestamp = prior_timestamps[-1] if prior_timestamps else -1.0
        if timestamp <= previous_timestamp:
            raise WorkspaceError(
                "snapshot timestamps must advance strictly within each video"
            )
        for item in evidence:
            if item.timestamp <= previous_timestamp or item.timestamp > timestamp:
                raise WorkspaceError(
                    "new evidence must lie in the unconsumed causal interval"
                )

        prior_state = self._latest_state.get(uid, self._initial_state)
        prior_state = _clone(prior_state)
        prior_sha256 = _sha256(prior_state)
        updated = update_state(
            _clone(prior_state),
            tuple(_clone(normalized_observations)),
            evidence,
        )
        if not isinstance(updated, Mapping):
            raise WorkspaceError("state update callback must return one mapping")
        state = dict(updated)
        _assert_writer_blind(state, "state")
        _canonical(state)
        state_sha256 = _sha256(state)
        version = self._versions.get(uid, 0) + 1
        snapshot = CausalStateSnapshot(
            video_uid=uid,
            query_timestamp=timestamp,
            state_version=version,
            prior_state_sha256=prior_sha256,
            state_sha256=state_sha256,
            observation_batch_sha256=chunk_sha256,
            writer_implementation_sha256=self.writer_implementation_sha256,
            writer_invoked=True,
            evidence_provenance=tuple(item.to_dict() for item in evidence),
            state=_clone(state),
        )
        self._snapshots[key] = snapshot
        self._timestamps.setdefault(uid, []).append(timestamp)
        self._latest_state[uid] = _clone(state)
        self._versions[uid] = version
        self.writer_calls += 1
        return snapshot.clone()

    def advance_horizon(
        self, *, video_uid: str, query_timestamp: float
    ) -> CausalStateSnapshot:
        """Create a read snapshot when no new decoder-visible frame exists."""

        uid = _video_uid(video_uid)
        timestamp = _timestamp(query_timestamp, "query_timestamp")
        key = (uid, timestamp)
        existing = self._snapshots.get(key)
        if existing is not None:
            return existing.clone()
        prior_timestamps = self._timestamps.get(uid, [])
        previous_timestamp = prior_timestamps[-1] if prior_timestamps else -1.0
        if timestamp <= previous_timestamp:
            raise WorkspaceError(
                "snapshot timestamps must advance strictly within each video"
            )
        state = _clone(self._latest_state.get(uid, self._initial_state))
        state_sha256 = _sha256(state)
        snapshot = CausalStateSnapshot(
            video_uid=uid,
            query_timestamp=timestamp,
            state_version=self._versions.get(uid, 0),
            prior_state_sha256=state_sha256,
            state_sha256=state_sha256,
            observation_batch_sha256=_chunk_sha256([], []),
            writer_implementation_sha256=self.writer_implementation_sha256,
            writer_invoked=False,
            evidence_provenance=(),
            state=state,
        )
        self._snapshots[key] = snapshot
        self._timestamps.setdefault(uid, []).append(timestamp)
        return snapshot.clone()

    def read_exact(
        self, *, video_uid: str, query_timestamp: float
    ) -> CausalStateSnapshot:
        uid = _video_uid(video_uid)
        timestamp = _timestamp(query_timestamp, "query_timestamp")
        try:
            return self._snapshots[(uid, timestamp)].clone()
        except KeyError as error:
            raise WorkspaceError("causal snapshot does not exist") from error

    def read_latest(
        self, *, video_uid: str, at_or_before: float | None = None
    ) -> CausalStateSnapshot:
        uid = _video_uid(video_uid)
        timestamps = self._timestamps.get(uid, [])
        if not timestamps:
            raise WorkspaceError("video has no causal snapshots")
        if at_or_before is None:
            timestamp = timestamps[-1]
        else:
            horizon = _timestamp(at_or_before, "at_or_before")
            index = bisect_right(timestamps, horizon) - 1
            if index < 0:
                raise WorkspaceError("no causal snapshot exists at or before horizon")
            timestamp = timestamps[index]
        return self._snapshots[(uid, timestamp)].clone()

    def evidence_history(
        self,
        *,
        video_uid: str,
        at_or_before: float,
        maximum_refs: int,
    ) -> tuple[EvidenceRef, ...]:
        """Return a bounded recent provenance packet from frozen snapshots."""

        uid = _video_uid(video_uid)
        horizon = _timestamp(at_or_before, "at_or_before")
        if maximum_refs <= 0:
            raise WorkspaceError("maximum_refs must be positive")
        timestamps = self._timestamps.get(uid, [])
        end = bisect_right(timestamps, horizon)
        by_identity: dict[tuple[int, str], EvidenceRef] = {}
        for timestamp in timestamps[:end]:
            snapshot = self._snapshots[(uid, timestamp)]
            for raw in snapshot.evidence_provenance:
                evidence = EvidenceRef(**dict(raw))
                by_identity[(evidence.frame_index, evidence.pixel_sha256)] = evidence
        ordered = sorted(
            by_identity.values(),
            key=lambda item: (item.timestamp, item.frame_index, item.pixel_sha256),
        )
        return tuple(ordered[-maximum_refs:])

    def to_manifest(self) -> dict[str, Any]:
        snapshots = [
            self._snapshots[key].to_dict()
            for key in sorted(self._snapshots, key=lambda item: (item[0], item[1]))
        ]
        body = {
            "format": FORMAT,
            "writer_implementation_sha256": self.writer_implementation_sha256,
            "initial_state": _clone(self._initial_state),
            "writer_calls": self.writer_calls,
            "snapshot_count": len(snapshots),
            "snapshots": snapshots,
            "state_writer_inputs": [
                "prior_state", "new_observations", "evidence_provenance"
            ],
            "forbidden_writer_inputs": sorted(_FORBIDDEN_WRITER_KEYS),
        }
        return {**body, "manifest_sha256": _sha256(body)}

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "CausalStateSnapshotStore":
        """Restore an immutable snapshot index and revalidate its hash chain."""

        manifest = dict(payload)
        claimed_hash = manifest.pop("manifest_sha256", None)
        if manifest.get("format") != FORMAT or claimed_hash != _sha256(manifest):
            raise WorkspaceError("snapshot manifest format or hash is invalid")
        store = cls(
            writer_implementation_sha256=str(
                manifest.get("writer_implementation_sha256") or ""
            ),
            initial_state=manifest.get("initial_state") or {},
        )
        snapshots = manifest.get("snapshots")
        if not isinstance(snapshots, list):
            raise WorkspaceError("snapshot manifest snapshots must be a list")
        expected_writer_calls = 0
        for raw in snapshots:
            if not isinstance(raw, Mapping):
                raise WorkspaceError("snapshot manifest contains a non-object")
            item = dict(raw)
            if item.pop("format", None) != SNAPSHOT_FORMAT:
                raise WorkspaceError("snapshot entry format is invalid")
            if item.pop("question_or_options_visible_to_writer", None) is not False:
                raise WorkspaceError("snapshot writer-blind certificate is missing")
            uid = _video_uid(item.get("video_uid"))
            timestamp = _timestamp(item.get("query_timestamp"), "query_timestamp")
            if str(item.get("writer_implementation_sha256")) != (
                store.writer_implementation_sha256
            ):
                raise WorkspaceError("snapshot writer implementation changed")
            state = item.get("state")
            if not isinstance(state, Mapping):
                raise WorkspaceError("snapshot state must be a mapping")
            state = dict(state)
            _assert_writer_blind(state, "state")
            if item.get("state_sha256") != _sha256(state):
                raise WorkspaceError("snapshot state hash is invalid")
            previous = store._latest_state.get(uid, store._initial_state)
            if item.get("prior_state_sha256") != _sha256(previous):
                raise WorkspaceError("snapshot prior-state hash chain is invalid")
            timestamps = store._timestamps.get(uid, [])
            if timestamps and timestamp <= timestamps[-1]:
                raise WorkspaceError("snapshot manifest is not chronological")
            evidence_raw = item.get("evidence_provenance")
            if not isinstance(evidence_raw, list):
                raise WorkspaceError("snapshot evidence provenance must be a list")
            evidence = tuple(EvidenceRef(**dict(value)) for value in evidence_raw)
            writer_invoked = item.get("writer_invoked")
            if not isinstance(writer_invoked, bool):
                raise WorkspaceError("snapshot writer_invoked flag is invalid")
            if writer_invoked:
                expected_writer_calls += 1
                previous_version = store._versions.get(uid, 0)
                if item.get("state_version") != previous_version + 1:
                    raise WorkspaceError("writer snapshot state version is invalid")
                if not evidence:
                    raise WorkspaceError("writer snapshot has no evidence")
                store._versions[uid] = previous_version + 1
            elif item.get("state_version") != store._versions.get(uid, 0):
                raise WorkspaceError("horizon snapshot changed state version")
            snapshot = CausalStateSnapshot(
                video_uid=uid,
                query_timestamp=timestamp,
                state_version=int(item["state_version"]),
                prior_state_sha256=str(item["prior_state_sha256"]),
                state_sha256=str(item["state_sha256"]),
                observation_batch_sha256=str(item["observation_batch_sha256"]),
                writer_implementation_sha256=store.writer_implementation_sha256,
                writer_invoked=writer_invoked,
                evidence_provenance=tuple(value.to_dict() for value in evidence),
                state=_clone(state),
            )
            store._snapshots[(uid, timestamp)] = snapshot
            store._timestamps.setdefault(uid, []).append(timestamp)
            store._latest_state[uid] = _clone(state)
        if manifest.get("snapshot_count") != len(snapshots):
            raise WorkspaceError("snapshot manifest count is invalid")
        if manifest.get("writer_calls") != expected_writer_calls:
            raise WorkspaceError("snapshot manifest writer call count is invalid")
        store.writer_calls = expected_writer_calls
        return store


def merge_snapshot_manifests(
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and merge disjoint video shards into one snapshot manifest."""

    if not manifests:
        raise WorkspaceError("at least one snapshot manifest is required")
    restored = [CausalStateSnapshotStore.from_manifest(value) for value in manifests]
    writer_hash = restored[0].writer_implementation_sha256
    initial_state = restored[0]._initial_state
    snapshots: list[dict[str, Any]] = []
    seen_videos: set[str] = set()
    writer_calls = 0
    for store in restored:
        if store.writer_implementation_sha256 != writer_hash:
            raise WorkspaceError("snapshot shards use different writer implementations")
        if _canonical(store._initial_state) != _canonical(initial_state):
            raise WorkspaceError("snapshot shards use different initial states")
        manifest = store.to_manifest()
        videos = {str(row["video_uid"]) for row in manifest["snapshots"]}
        overlap = seen_videos & videos
        if overlap:
            raise WorkspaceError(
                f"snapshot shards overlap videos: {sorted(overlap)[:3]}"
            )
        seen_videos.update(videos)
        snapshots.extend(manifest["snapshots"])
        writer_calls += int(manifest["writer_calls"])
    snapshots.sort(key=lambda row: (str(row["video_uid"]), row["query_timestamp"]))
    body = {
        "format": FORMAT,
        "writer_implementation_sha256": writer_hash,
        "initial_state": _clone(initial_state),
        "writer_calls": writer_calls,
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
        "state_writer_inputs": [
            "prior_state", "new_observations", "evidence_provenance"
        ],
        "forbidden_writer_inputs": sorted(_FORBIDDEN_WRITER_KEYS),
    }
    output = {**body, "manifest_sha256": _sha256(body)}
    CausalStateSnapshotStore.from_manifest(output)
    return output
