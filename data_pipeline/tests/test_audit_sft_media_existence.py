from data_pipeline.stages.audit_sft_media_existence import image_paths


def test_image_paths_reads_only_model_visible_image_blocks() -> None:
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/a.jpg"},
                    {"type": "text", "text": "question"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "A"}],
            },
        ]
    }
    assert image_paths(row) == ["/tmp/a.jpg"]
