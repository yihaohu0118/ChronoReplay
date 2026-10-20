from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "trajectory_order_control",
    ROOT / "data_pipeline" / "stages" / "build_causal_trajectory_order_control.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def block(index: int) -> list[dict]:
    return [
        {"role": "user", "content": [{"type": "text", "text": f"q{index}"}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"a{index}"}]},
    ]


def test_order_is_deterministic_nonidentity() -> None:
    first = MODULE.nonidentity_order("scene-a", 4)
    second = MODULE.nonidentity_order("scene-a", 4)
    assert first == second
    assert first != [0, 1, 2, 3]
    assert sorted(first) == [0, 1, 2, 3]


def test_reorder_preserves_atomic_blocks_and_turn_weights() -> None:
    messages = [message for index in range(4) for message in block(index)]
    row = {
        "uid": "scene-a",
        "messages": messages,
        "supervised_assistant_ordinals": [0, 2, 3],
        "assistant_turn_loss_weights": [1.0, 3.0, 4.0],
    }
    actual = MODULE.reorder_trajectory(row)
    order = actual["trajectory_original_turn_indices"]
    expected_messages = [message for index in order for message in block(index)]
    assert actual["messages"] == expected_messages
    weight_by_original = {0: 1.0, 2: 3.0, 3: 4.0}
    expected_supervised = [
        new for new, original in enumerate(order) if original in weight_by_original
    ]
    assert actual["supervised_assistant_ordinals"] == expected_supervised
    assert actual["assistant_turn_loss_weights"] == [
        weight_by_original[order[new]] for new in expected_supervised
    ]


def test_single_turn_is_retained_without_false_permutation() -> None:
    row = {
        "uid": "single",
        "messages": block(0),
        "supervised_assistant_ordinals": [0],
        "assistant_turn_loss_weights": [1.0],
    }
    actual = MODULE.reorder_trajectory(row)
    assert actual["messages"] == row["messages"]
    assert actual["trajectory_order_control"] is False
    assert actual["trajectory_original_turn_indices"] == [0]
