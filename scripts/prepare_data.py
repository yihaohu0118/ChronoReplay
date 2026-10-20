"""Build a portable, family-balanced replay pool from documented native records."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from data_pipeline.pure_vlm_utils import PARQUET_ENCODING, builtin, sha256


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd
        return [builtin(r) for r in pd.read_parquet(path).to_dict("records")]
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if path.suffix == ".json" else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError(f"Expected a list of records: {path}")
    return rows


def local_image(value: str, root: Path) -> Path:
    if "://" in value:
        raise ValueError("Use local media paths, not URLs.")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Image escapes --media-root: {value}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def messages_and_media(messages: Any, root: Path, media: set[Path]) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("Each row needs native chat messages.")
    out = deepcopy(messages)
    for i, message in enumerate(out):
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("Unsupported message role.")
        if message["role"] == "assistant" and (i == 0 or out[i - 1]["role"] != "user"):
            raise ValueError("Every assistant turn must follow its user observation.")
        content = message.get("content")
        if not isinstance(content, list) or not content:
            raise ValueError("Content must be a nonempty list of text/image blocks.")
        for part in content:
            if part.get("type") == "image":
                if message["role"] != "user":
                    raise ValueError("Images belong to observation/user turns.")
                path = local_image(str(part["image"]), root)
                part["image"] = str(path)
                media.add(path)
            elif part.get("type") == "text" and isinstance(part.get("text"), str):
                pass
            else:
                raise ValueError("Only native text and local image blocks are supported.")
    if out[-1]["role"] != "assistant":
        raise ValueError("The final message must be a supervised native answer.")
    return out


def build(preservation: list[dict], scenes: list[dict], media_root: Path,
          heldout: set[str], alpha: float = 0.5) -> tuple[list[dict], dict, set[Path]]:
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Trajectory mass ratio must be finite and positive.")
    media: set[Path] = set()
    categorical: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    numeric: list[dict] = []
    rows: list[dict] = []
    seen: set[str] = set()
    contents: set[str] = set()
    train_scenes: set[str] = set()
    for raw in preservation:
        row = deepcopy(raw)
        for key in ("uid", "source_id", "scene_id", "semantic_id", "family", "source_split"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Preservation row lacks {key}.")
        if row["source_split"] not in {"train", "training"}:
            raise ValueError("Preservation records must use the source training split.")
        if row["scene_id"] in heldout:
            raise ValueError(f"Held-out scene in training: {row['scene_id']}")
        if row["uid"] in seen:
            raise ValueError(f"Duplicate UID: {row['uid']}")
        seen.add(row["uid"])
        train_scenes.add(row["scene_id"])
        row["messages"] = messages_and_media(row["messages"], media_root, media)
        answers = [m for m in row["messages"] if m["role"] == "assistant"]
        if len(answers) != 1:
            raise ValueError("Preservation examples must be single-turn.")
        fingerprint = json.dumps(row["messages"], sort_keys=True, ensure_ascii=False)
        if fingerprint in contents:
            raise ValueError("Exact duplicate preservation content.")
        contents.add(fingerprint)
        row["supervised_assistant_ordinals"] = [0]
        row["num_images"] = sum(p["type"] == "image" for m in row["messages"] for p in m["content"])
        row["component"] = "public_preservation_" + row["family"]
        if row["family"] == "categorical":
            categorical[(row["source_id"], row["scene_id"], row["semantic_id"])].append(row)
            row["workfirst_family"] = "L"
        elif row["family"] == "numeric":
            numeric.append(row)
            row["workfirst_family"] = "N"
        else:
            raise ValueError("family must be categorical or numeric.")
        rows.append(row)
    if not categorical or not numeric:
        raise ValueError("Both categorical and numeric preservation records are required.")
    for group in categorical.values():
        views = [r.get("view") for r in group]
        if len(group) not in (1, 3) or len(set(views)) != len(views) or any(not v for v in views):
            raise ValueError("Categorical groups need one view or three distinct views.")
        if len(group) == 3 and set(views) != {"dense", "context", "long"}:
            raise ValueError("Three-view groups must use dense/context/long.")
        targets = {json.dumps(r["messages"][-1]["content"], sort_keys=True) for r in group}
        if len(targets) != 1:
            raise ValueError("Matched views must retain the same source target.")

    trajectories = []
    all_scene_ids: set[str] = set()
    context_only = 0
    total_turns = 0
    for scene in scenes:
        sid = str(scene["scan_id"])
        if sid in all_scene_ids or sid in heldout:
            raise ValueError(f"Duplicate or held-out OST scene: {sid}")
        all_scene_ids.add(sid)
        train_scenes.add(sid)
        messages = []
        supervised = []
        for i, turn in enumerate(scene["user_message"]):
            obs = [{"type": "text", "text": str(scene["system_prompt"])}]
            obs += [{"type": "image", "image": str(v)} for v in turn.get("image_paths", [])]
            obs += [{"type": "text", "text": str(turn["prompt"])}]
            messages += [{"role": "user", "content": obs},
                         {"role": "assistant", "content": [{"type": "text", "text": str(turn["answer"])}]}]
            if str(turn.get("type") or "None") != "None":
                supervised.append(i)
        total_turns += len(scene["user_message"])
        messages = messages_and_media(messages, media_root, media)
        if not supervised:
            context_only += 1
            continue
        uid = "ost-train-scene-" + hashlib.sha256(sid.encode()).hexdigest()[:32]
        if uid in seen:
            raise ValueError(f"Duplicate trajectory UID: {uid}")
        seen.add(uid)
        trajectories.append({
            "uid": uid, "source_id": "OST-Bench", "source_uid": sid,
            "scene_id": sid, "source_split": "train",
            "source_group_sha256": hashlib.sha256(sid.encode()).hexdigest(),
            "component": "official_ost_train_multiturn_scene",
            "target_contract": "causal_multiturn_mixed_spatial_answer",
            "workfirst_family": "OST", "messages": messages,
            "supervised_assistant_ordinals": supervised,
            "num_turns": len(scene["user_message"]),
            "num_images": sum(p["type"] == "image" for m in messages for p in m["content"]),
        })
    turns = sum(len(r["supervised_assistant_ordinals"]) for r in trajectories)
    if not turns:
        raise ValueError("At least one supervised trajectory turn is required.")
    n = len(rows) + len(trajectories)
    parent_mass = n / (1 + alpha)
    for group in categorical.values():
        for row in group:
            row["sample_loss_weight"] = parent_mass / (2 * len(categorical) * len(group))
    for row in numeric:
        row["sample_loss_weight"] = parent_mass / (2 * len(numeric))
    for row in trajectories:
        row["sample_loss_weight"] = parent_mass * alpha * len(row["supervised_assistant_ordinals"]) / turns
    rows += trajectories
    masses = Counter()
    for row in rows:
        masses[row["workfirst_family"]] += row["sample_loss_weight"]
    report = {
        "format": "chronoreplay.public_data.v1", "ok": True,
        "rows": n, "categorical_rows": sum(map(len, categorical.values())),
        "categorical_semantic_groups": len(categorical), "numeric_rows": len(numeric),
        "trajectory_rows": len(trajectories), "source_ost_scenes": len(scenes),
        "context_only_scenes_excluded": context_only,
        "trajectory_turns_including_context": total_turns,
        "trajectory_supervised_turns": turns, "family_loss_mass": dict(masses),
        "alpha": alpha, "mean_sample_weight": sum(masses.values()) / n,
        "unique_media": len(media), "train_scene_ids": sorted(train_scenes),
        "heldout_scene_ids": sorted(heldout), "scene_overlap": 0,
        "source_targets_preserved": True,
        "paper_membership_verified": False,
    }
    return rows, report, media


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preservation", type=Path, required=True)
    p.add_argument("--ost-train", type=Path, required=True)
    p.add_argument("--heldout-scenes", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--expect-paper-counts", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        p.error("Choose a new output directory.")
    heldout = json.loads(a.heldout_scenes.read_text())
    if not isinstance(heldout, list) or not heldout or not all(isinstance(v, str) and v for v in heldout):
        p.error("--heldout-scenes must contain a nonempty JSON list of canonical scene IDs.")
    rows, report, media = build(read_rows(a.preservation), read_rows(a.ost_train),
                                a.media_root.resolve(), set(heldout), a.alpha)
    expected = {
        "rows": 39561, "categorical_rows": 30635, "categorical_semantic_groups": 12635,
        "numeric_rows": 2545, "trajectory_rows": 6381, "source_ost_scenes": 6390,
        "context_only_scenes_excluded": 9, "trajectory_supervised_turns": 45006,
        "trajectory_turns_including_context": 47096,
    }
    if a.expect_paper_counts and any(report[k] != v for k, v in expected.items()):
        p.error("Input counts differ from the declared paper recipe.")
    from PIL import Image
    for path in sorted(media):
        with Image.open(path) as image:
            image.load()
    import pandas as pd
    a.output.mkdir(parents=True)
    parquet = a.output / "train.parquet"
    pd.DataFrame.from_records(rows).to_parquet(parquet, index=False, **PARQUET_ENCODING)
    report.update({
        "train_sha256": sha256(parquet), "all_media_decoded": True,
        "source_file_sha256": {k: sha256(getattr(a, k)) for k in ("preservation", "ost_train", "heldout_scenes")},
    })
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.output / "MEDIA_DECODE_READY").write_text("ok\n")
    (a.output / "DATA_READY").write_text("ok\n")
    print(json.dumps({k: report[k] for k in ("rows", "unique_media", "family_loss_mass")}))


if __name__ == "__main__":
    main()
