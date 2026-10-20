#!/usr/bin/env python3
"""Merge complete, video-disjoint UCS state-writer shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_witness.causal_state_snapshots import merge_snapshot_manifests


INPUT_FORMAT = "iclr_spatial.ucs_query_independent_state_builder.v1"
FORMAT = "iclr_spatial.ucs_query_independent_state_release.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--expected-videos", type=int, default=532)
    parser.add_argument("--expected-prefixes", type=int, default=7287)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payloads = []
    reports = []
    traces = []
    for path in args.shard:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("format") != INPUT_FORMAT:
            raise ValueError(f"invalid UCS state shard: {path}")
        payloads.append(value["snapshot_store"])
        reports.append(value["report"])
        traces.extend(value.get("trace") or [])
    snapshot_store = merge_snapshot_manifests(payloads)
    videos = {row["video_uid"] for row in snapshot_store["snapshots"]}
    prefixes = len(snapshot_store["snapshots"])
    if len(videos) != args.expected_videos or prefixes != args.expected_prefixes:
        raise ValueError(
            "incomplete UCS state coverage: "
            f"videos={len(videos)}/{args.expected_videos} "
            f"prefixes={prefixes}/{args.expected_prefixes}"
        )
    output = {
        "format": FORMAT,
        "complete": True,
        "videos": len(videos),
        "unique_prefixes": prefixes,
        "question_or_options_visible_to_writer": False,
        "snapshot_store": snapshot_store,
        "shard_reports": reports,
        "trace": sorted(
            traces,
            key=lambda row: (str(row["video_uid"]), row["query_timestamp"]),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"videos": len(videos), "unique_prefixes": prefixes},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
