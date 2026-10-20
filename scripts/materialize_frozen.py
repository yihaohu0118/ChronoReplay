"""Reconstruct frozen replay order/weights from locally obtained converted source rows."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from data_pipeline.pure_vlm_utils import PARQUET_ENCODING, sha256
from scripts.prepare_data import read_rows, messages_and_media

ROOT = Path(__file__).resolve().parents[1]


def text_sha256(messages: list[dict]) -> str:
    text = [{"role": m["role"],
             "text": [p.get("text", "") for p in m["content"] if p.get("type") == "text"]}
            for m in messages]
    return hashlib.sha256(json.dumps(text, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def select(candidates: list[dict], selection: dict) -> list[dict]:
    if selection.get("format") != "chronoreplay.frozen_training_selection.v1":
        raise ValueError("Unrecognized frozen selection format.")
    targets = selection["rows"]
    expected = [r["uid"] for r in targets]
    if len(set(expected)) != len(expected) or not expected:
        raise ValueError("Empty or duplicate selection IDs.")
    wanted = set(expected)
    source = {}
    for row in candidates:
        uid = row.get("uid")
        if uid in wanted:
            if uid in source:
                raise ValueError(f"Duplicate selected candidate: {uid}")
            source[uid] = row
    missing = wanted - set(source)
    if missing:
        raise ValueError(f"Missing {len(missing)} selected source rows; first IDs: {sorted(missing)[:3]}")
    output = []
    for i, target in enumerate(targets):
        if target["row_index"] != i:
            raise ValueError("Selection row order changed.")
        row = deepcopy(source[target["uid"]])
        if text_sha256(row["messages"]) != target["text_sha256"]:
            raise ValueError(f"Native prompt/target text mismatch: {target['uid']}")
        images = sum(p.get("type") == "image" for m in row["messages"] for p in m["content"])
        if images != target["num_images"]:
            raise ValueError(f"Observation count mismatch: {target['uid']}")
        for key in ("component", "target_contract", "workfirst_family"):
            if row.get(key) != target.get(key):
                raise ValueError(f"Source field mismatch: {target['uid']} / {key}")
        if (row.get("supervised_assistant_ordinals") or []) != (target.get("supervised_assistant_ordinals") or []):
            raise ValueError(f"Supervised-turn mask mismatch: {target['uid']}")
        row["sample_loss_weight"] = target["sample_loss_weight"]
        output.append(row)
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, action="append", required=True,
                   help="Native converted-source JSONL or Parquet; may be repeated.")
    p.add_argument("--selection", type=Path,
                   default=ROOT / "data_manifests/replay_selection.json")
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Choose a new output directory.")
    selection = json.loads(a.selection.read_text())
    candidates = [row for path in a.candidate for row in read_rows(path)]
    rows = select(candidates, selection)
    media: set[Path] = set()
    for row in rows:
        row["messages"] = messages_and_media(row["messages"], a.media_root.resolve(), media)
    from PIL import Image
    for path in sorted(media):
        with Image.open(path) as image:
            image.load()
    import pandas as pd
    a.output.mkdir(parents=True)
    parquet = a.output / "train.parquet"
    pd.DataFrame.from_records(rows).to_parquet(parquet, index=False, **PARQUET_ENCODING)
    report = {
        "format": "chronoreplay.materialized_frozen_selection.v1",
        "ok": True, "rows": len(rows), "unique_media": len(media),
        "train_sha256": sha256(parquet), "selection_sha256": sha256(a.selection),
        "source_pool_sha256": selection["source_train_sha256"],
        "native_text_and_masks_verified": True,
        "frozen_ids_order_and_weights_verified": True,
        "all_media_decoded": True, "pixel_identity_verified": False,
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.output / "MEDIA_DECODE_READY").write_text("ok\n")
    (a.output / "DATA_READY").write_text("ok\n")
    print(json.dumps({"rows": len(rows), "unique_media": len(media)}))


if __name__ == "__main__":
    main()
