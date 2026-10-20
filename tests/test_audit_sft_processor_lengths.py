from types import SimpleNamespace

from data_pipeline.processor_contract import configure_processor_pixel_budget
from scripts.audit_sft_processor_lengths import select_probe_rows


def _row(uid: str, component: str, images: int, message_bytes: int = 1):
    return {
        "uid": uid,
        "component": component,
        "num_images": images,
        "num_turns": 1,
        "messages": [{"role": "user", "content": "x" * message_bytes}],
    }


def test_probe_keeps_maximum_cost_row_from_every_component():
    rows = [_row(f"large-{index}", "large", 32) for index in range(10)]
    rows += [
        _row("small-short", "small", 8),
        _row("small-long", "small", 16),
    ]

    selected = select_probe_rows(rows, 4)

    assert "small-long" in {row["uid"] for row in selected}


def test_probe_fails_closed_when_component_count_exceeds_budget():
    rows = [_row(f"row-{index}", f"component-{index}", 1) for index in range(3)]

    try:
        select_probe_rows(rows, 2)
    except ValueError as error:
        assert "number of source components" in str(error)
    else:
        raise AssertionError("expected component coverage failure")


def test_pixel_budget_is_applied_without_mutating_the_model_files():
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            size={"shortest_edge": 65536, "longest_edge": 16777216}
        )
    )
    configured = configure_processor_pixel_budget(processor, 200704)
    assert configured == {"shortest_edge": 65536, "longest_edge": 200704}
    assert processor.image_processor.size == configured
