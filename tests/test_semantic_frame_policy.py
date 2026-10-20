from __future__ import annotations

from synthesis.semantic_frame_policy import (
    select_adaptive_causal_frame_policy,
    select_causal_frame_policy,
    select_temporal_dereference_frame_policy,
    select_uniform_temporal_dereference_frame_policy,
)


def test_routes_explicit_current_state_to_short_dense_window() -> None:
    assert select_causal_frame_policy("What is visible in the current frame?")[0] == "recent"
    assert select_causal_frame_policy("What changes in this segment?")[0] == "recent"


def test_ego_camera_motion_retains_temporal_flow_context() -> None:
    assert select_causal_frame_policy(
        "Based on the visual flow, is the camera currently yawing left or right?"
    ) == ("uniform", "explicit_ego_motion_scope")
    assert select_causal_frame_policy(
        "In the current moment, is the camera moving backward or forward?"
    ) == ("uniform", "explicit_ego_motion_scope")
    # Relative object motion still benefits from the local recent window.
    assert select_causal_frame_policy(
        "Is the bicycle currently moving faster than our camera?"
    )[0] == "recent"


def test_history_semantics_override_current_wording() -> None:
    assert select_causal_frame_policy(
        "At the current moment, is this the same place as at 20 seconds?"
    )[0] == "uniform"
    assert select_causal_frame_policy(
        "Including the current view, how many times has this scene appeared so far?"
    )[0] == "uniform"
    assert select_causal_frame_policy(
        "Was the room visited before or after the street in the video?"
    )[0] == "uniform"


def test_unspecified_scope_fails_to_global_context() -> None:
    assert select_causal_frame_policy("Which object is red?") == (
        "uniform",
        "unspecified_or_global_scope",
    )


def test_adaptive_policy_separates_local_motion_chronology_and_layout() -> None:
    assert select_adaptive_causal_frame_policy(
        "In the current frame, what directly supports the dog?"
    ) == ("recent", "explicit_current_scope")
    assert select_adaptive_causal_frame_policy(
        "In the current moment, is the door opening or closing?"
    ) == ("log_decay", "explicit_temporal_derivative_scope")
    assert select_adaptive_causal_frame_policy(
        "Was the hall visited before or after the street?"
    ) == ("log_decay", "explicit_chronological_scope")
    assert select_adaptive_causal_frame_policy(
        "What is the second room entered after going upstairs?"
    ) == ("log_decay", "explicit_chronological_scope")
    assert select_adaptive_causal_frame_policy(
        "Tracing the tour sequence, which place came immediately before this one?"
    ) == ("log_decay", "explicit_chronological_scope")
    assert select_adaptive_causal_frame_policy(
        "Based on the full video, what is the topology of the rooms?"
    ) == ("uniform", "explicit_static_layout_scope")
    assert select_adaptive_causal_frame_policy(
        "After entering, through which structure does the space first connect?"
    ) == ("uniform", "explicit_static_layout_scope")
    assert select_adaptive_causal_frame_policy(
        "What is the spatial arrangement in the current frame?"
    ) == ("recent", "explicit_current_scope")
    assert select_adaptive_causal_frame_policy(
        "Is the rectangular mirror to your left/right now?"
    ) == ("uniform", "persistent_relative_pose_operator")
    assert select_adaptive_causal_frame_policy(
        "Based on your current orientation, in what clockwise direction "
        '(in degrees) is "the red bucket" from your position?'
    ) == ("uniform", "persistent_relative_pose_operator")


def test_temporal_dereference_adds_one_validated_executor_only() -> None:
    assert select_temporal_dereference_frame_policy(
        "At the current moment, is this the same or a different place "
        "relative to the referenced moment at 12.5 seconds?",
        {"A": "Same place", "B": "Different place"},
    ) == ("referenced_pair", "validated_explicit_temporal_dereference")
    assert select_temporal_dereference_frame_policy(
        "Was the hall visited before or after the street?"
    ) == ("log_decay", "explicit_chronological_scope")


def test_uniform_temporal_dereference_preserves_uniform_fallback() -> None:
    assert select_uniform_temporal_dereference_frame_policy(
        "At the current moment, is this the same or a different place "
        "relative to the referenced moment at 12.5 seconds?",
        {"A": "Same place", "B": "Different place"},
    ) == ("referenced_pair", "validated_explicit_temporal_dereference")
    assert select_uniform_temporal_dereference_frame_policy(
        "What is currently next to the chair?",
        {"A": "Table", "B": "Door"},
    ) == ("uniform", "uniform_fallback_no_temporal_dereference")
