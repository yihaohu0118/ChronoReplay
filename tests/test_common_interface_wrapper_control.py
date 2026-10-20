from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "common_interface_control",
    ROOT / "data_pipeline" / "stages" / "build_common_interface_wrapper_control.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_wrap_preserves_target_value_and_nonassistant_context() -> None:
    raw = [
        {"role": "user", "content": [{"type": "text", "text": "distance?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "1.25 m"}]},
    ]
    transformed, targets = MODULE.transform_messages(raw)
    assert targets == ["1.25 m"]
    assert transformed[0] == raw[0]
    assert json.loads(transformed[1]["content"][0]["text"]) == {"value": "1.25 m"}
    assert raw[1]["content"][0]["text"] == "1.25 m"


def test_wrapper_is_canonical_and_information_preserving() -> None:
    value = 'A "quoted" value'
    assert json.loads(MODULE.wrapped(value))["value"] == value
    assert MODULE.wrapped(value) == '{"value":"A \\"quoted\\" value"}'
