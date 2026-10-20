"""Question-blind observation-chunk pose timeline and qualitative reader.

The writer consumes each new causal observation chunk at most once and never
reads a question, option, label, or score. The reader may use visible language
to retrieve an already written snapshot. The model-visible packet is
qualitative and uncalibrated; it never emits a multiple-choice letter or a
meter/degree claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image

from spatial_witness.classical_trajectory_writer import ClassicalTrajectoryWriter
from synthesis.spatial_query_compiler import (
    compile_workspace_action,
    observation_chunk_reference,
)

FORMAT = "spatial_witness.observation_chunk_pose_packet.v1"
_QUALITATIVE_KEYS = (
    "metric_pose_claimed",
    "reference_timestamp",
    "uncalibrated_heading_change",
    "uncalibrated_ego_translation",
    "uncalibrated_path_shape",
    "yaw_sign_convention",
    "pose_node_count",
)


def render_qualitative_pose_packet(fact: Mapping[str, Any]) -> str | None:
    if fact.get("metric_pose_claimed") is not False:
        return None
    lines = [
        "Uncalibrated camera-trajectory state from optical flow. "
        "These are not meters, not degrees, and not multiple-choice answers."
    ]
    for key in _QUALITATIVE_KEYS:
        if key not in fact:
            continue
        lines.append(f"- {key}: {fact[key]}")
    packet = "\n".join(lines)
    if any(token in packet for token in ("answer:", "correct_option", "\nA.", " B.")):
        return None
    return packet


class ObservationChunkPoseTimeline:
    """Persist uncalibrated pose across observation chunks of one source."""

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self.writer = ClassicalTrajectoryWriter(source_id)
        self.frame_index = 0
        self.chunk_end_timestamp: dict[int, float] = {}

    def ingest_chunk(self, chunk_index: int, images: Sequence[Image.Image]) -> None:
        if int(chunk_index) < 1:
            raise ValueError("observation chunk indices are 1-based")
        for image in images:
            self.frame_index += 1
            timestamp = float(self.frame_index)
            self.writer.ingest(
                image,
                frame_index=self.frame_index,
                timestamp=timestamp,
            )
        if images:
            self.chunk_end_timestamp[int(chunk_index)] = float(self.frame_index)

    def retrieve_qualitative_packet(
        self,
        question: str,
        options: Mapping[str, str] | None = None,
    ) -> str | None:
        plan = compile_workspace_action(question, options)
        if not plan or plan.get("executor") != "query_camera_pose":
            return None
        visible = " ".join(
            [str(question or "")]
            + [str(value) for value in (options or {}).values()]
        )
        action: dict[str, Any] = {"executor": "query_camera_pose"}
        since_chunk = observation_chunk_reference(visible)
        if since_chunk is not None:
            timestamp = self.chunk_end_timestamp.get(since_chunk)
            if timestamp is None:
                return None
            action["since"] = timestamp
        result = self.writer.builder.workspace.execute(action)
        if result.status != "ok" or not result.facts:
            return None
        return render_qualitative_pose_packet(result.facts[0])
