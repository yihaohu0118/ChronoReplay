"""Question-blind classical trajectory writer.

The writer consumes chronological RGB frames only. It never reads a question,
option, task id, label, prediction, or score. Pairwise image similarities are
integrated into the existing SE(2) workspace as an uncalibrated optical-flow
trajectory: a generic spatial state that later queries may retrieve.

This is not metric visual odometry and not a benchmark-leaf matcher. A sparse
or failed track simply omits that pose node.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from PIL import Image

from spatial_witness.causal_state_builder import (
    ExtractorIdentity,
    QuestionBlindStateBuilder,
    pixel_sha256,
)
from spatial_witness.causal_workspace import CausalSpatialWorkspace, EvidenceRef
from spatial_witness.opencv_spatial_tools import estimate_image_similarity


FORMAT = "spatial_witness.classical_optical_flow_trajectory_writer.v1"
_MINIMUM_INLIER_FRACTION = 0.25


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TrajectoryWriterConfig:
    minimum_inlier_fraction: float = _MINIMUM_INLIER_FRACTION
    maximum_step_translation: float = 0.45
    maximum_step_rotation: float = 1.2

    def identity(self) -> ExtractorIdentity:
        payload = asdict(self)
        return ExtractorIdentity(
            name="classical-optical-flow-trajectory",
            version="1",
            kind="classical_vision",
            configuration_sha256=_sha256(payload),
        )


class ClassicalTrajectoryWriter:
    """Append uncalibrated SE(2) nodes from causal RGB frames."""

    def __init__(
        self,
        source_id: str,
        *,
        config: TrajectoryWriterConfig | None = None,
        builder: QuestionBlindStateBuilder | None = None,
    ) -> None:
        self.source_id = source_id
        self.config = config or TrajectoryWriterConfig()
        self.builder = builder or QuestionBlindStateBuilder(
            source_id, self.config.identity()
        )
        self._previous: Image.Image | None = None
        self.accepted_steps = 0
        self.skipped_steps = 0

    def apply_pose(
        self,
        workspace: CausalSpatialWorkspace,
        image: Image.Image,
        *,
        frame_index: int,
        timestamp: float,
    ) -> Mapping[str, Any]:
        """Integrate one uncalibrated pose node into an existing workspace.

        This path is for writers that share a workspace with a semantic
        observer. It never calls the VLM and never reads a question.
        """

        ref = EvidenceRef(
            self.source_id, frame_index, float(timestamp), pixel_sha256(image)
        )
        if self._previous is None:
            workspace.integrate_pose(ref)
            self._previous = image.copy()
            self.accepted_steps += 1
            return {"status": "origin", "metric_pose_claimed": False}

        similarity = estimate_image_similarity(self._previous, image)
        self._previous = image.copy()
        if (
            similarity is None
            or float(similarity["ransac_inlier_fraction"])
            < self.config.minimum_inlier_fraction
            or abs(float(similarity["image_dx_fraction"]))
            > self.config.maximum_step_translation
            or abs(float(similarity["image_dy_fraction"]))
            > self.config.maximum_step_translation
            or abs(float(similarity["image_rotation_radians"]))
            > self.config.maximum_step_rotation
        ):
            self.skipped_steps += 1
            return {"status": "skipped", "metric_pose_claimed": False}

        workspace.integrate_pose(
            ref,
            local_dx=float(similarity["image_dx_fraction"]),
            local_dy=float(similarity["image_dy_fraction"]),
            delta_yaw=float(similarity["image_rotation_radians"]),
        )
        self.accepted_steps += 1
        return {
            "status": "ok",
            "metric_pose_claimed": False,
            "ransac_inlier_fraction": similarity["ransac_inlier_fraction"],
        }

    def ingest(
        self,
        image: Image.Image,
        *,
        frame_index: int,
        timestamp: float,
    ) -> Mapping[str, Any]:
        return self.apply_pose(
            self.builder.workspace,
            image,
            frame_index=frame_index,
            timestamp=timestamp,
        )

    def ingest_prefix(
        self,
        frames: Sequence[Any],
    ) -> dict[str, Any]:
        for frame in frames:
            self.ingest(
                frame.image,
                frame_index=int(frame.frame_index),
                timestamp=float(frame.timestamp),
            )
        audit = self.builder.audit()
        inventory = self.builder.workspace.public_inventory()
        return {
            "format": FORMAT,
            "ok": True,
            "accepted_steps": self.accepted_steps,
            "skipped_steps": self.skipped_steps,
            "pose_node_count": inventory.get("pose_node_count"),
            "question_or_options_read": False,
            "benchmark_metadata_read": False,
            "metric_pose_claimed": False,
            "builder_audit": audit,
        }
