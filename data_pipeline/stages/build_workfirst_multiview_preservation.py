#!/usr/bin/env python3
"""Restore matched context views without increasing semantic L mass."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import pandas as pd

from data_pipeline.pure_vlm_utils import PARQUET_ENCODING, builtin, sha256
from data_pipeline.stages.compose_packet_bearing_sft import absolute_media, content_key


FORMAT = "iclr_spatial.workfirst_multiview_preservation.v1"
CDROP_FORMAT = "iclr_spatial.workfirst_lnc_joint_cdrop.v1"
PARENT_FORMAT = "iclr_spatial.state_reader_writer_sft_pool.v1"
VIEWS = {
    "pure_vlm_main_evidence_dense",
    "pure_vlm_main_evidence_context",
    "long_context_direct_answer",
}


def records(path: Path) -> list[dict]:
    return [
        {str(key): builtin(value) for key, value in row.items()}
        for row in pd.read_parquet(path).to_dict("records")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cdrop-dir", type=Path, required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    cdrop_train, parent_train = args.cdrop_dir / "train.parquet", args.parent_dir / "train.parquet"
    cdrop_report = json.loads((args.cdrop_dir / "report.json").read_text())
    parent_report = json.loads((args.parent_dir / "report.json").read_text())
    if not (
        cdrop_report.get("format") == CDROP_FORMAT
        and cdrop_report.get("ok") is True
        and cdrop_report.get("rows") == 15180
        and cdrop_report.get("family_rows") == {"L": 12635, "N": 2545, "C": 0}
        and cdrop_report.get("train_sha256") == sha256(cdrop_train)
    ):
        raise ValueError("C-drop source drift")
    if not (
        parent_report.get("format") == PARENT_FORMAT
        and parent_report.get("ok") is True
        and parent_report.get("rows") == 40339
        and parent_report.get("benchmark_inputs_read") is False
        and parent_report.get("benchmark_predictions_or_scores_read") is False
        and parent_report.get("train_sha256") == sha256(parent_train)
    ):
        raise ValueError("40,339 source parent drift")

    cdrop = records(cdrop_train)
    dense = [
        row for row in cdrop
        if str(row.get("workfirst_family") or "") == "L"
        and str(row.get("component") or "") == "pure_vlm_main_evidence_dense"
    ]
    if len(dense) != 9000:
        raise ValueError(f"selected dense row drift: {len(dense)}")
    semantic_ids = {str(row.get("semantic_source_sha256") or "") for row in dense}
    if "" in semantic_ids or len(semantic_ids) != 9000:
        raise ValueError("selected dense semantic IDs drift")
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for raw_row in records(parent_train):
        component = str(raw_row.get("component") or "")
        semantic = str(raw_row.get("semantic_source_sha256") or "")
        if component in VIEWS and semantic in semantic_ids:
            row = absolute_media(raw_row, parent_train.parent)
            if component in groups[semantic]:
                raise ValueError("duplicate source view")
            groups[semantic][component] = row
    if len(groups) != 9000 or any(set(views) != VIEWS for views in groups.values()):
        raise ValueError("selected semantics lack exact three-view closure")

    original_l_weight = float(cdrop_report["family_sample_loss_weights"]["L"])
    raw: list[dict] = []
    for row in cdrop:
        value = dict(row)
        if (
            str(value.get("workfirst_family") or "") == "L"
            and str(value.get("component") or "") == "pure_vlm_main_evidence_dense"
        ):
            value["sample_loss_weight"] = original_l_weight / 3.0
        raw.append(value)
    for semantic in sorted(semantic_ids):
        for component in ("pure_vlm_main_evidence_context", "long_context_direct_answer"):
            value = dict(groups[semantic][component])
            value["workfirst_family"] = "P"
            value["sample_loss_weight"] = original_l_weight / 3.0
            raw.append(value)
    if len(raw) != 33180:
        raise ValueError(len(raw))
    raw_mass = sum(float(row["sample_loss_weight"]) for row in raw)
    scale = len(raw) / raw_mass
    for row in raw:
        row["sample_loss_weight"] = float(row["sample_loss_weight"]) * scale
    mean_weight = sum(float(row["sample_loss_weight"]) for row in raw) / len(raw)
    if abs(mean_weight - 1.0) > 1e-12:
        raise ValueError(mean_weight)
    keys = [content_key(row) for row in raw]
    if len(keys) != len(set(keys)):
        raise ValueError("multiview pool contains exact-content duplicate")
    family_rows = {"L": 12635, "N": 2545, "C": 0, "P": 18000}
    masses = Counter()
    for row in raw:
        masses[str(row.get("workfirst_family") or "")] += float(row["sample_loss_weight"])
    if abs((masses["L"] + masses["P"]) - masses["N"]) > 1e-8:
        raise ValueError("visual/numeric mass drift")
    components = Counter(str(row.get("component") or "") for row in raw)
    contracts = Counter(str(row.get("target_contract") or "") for row in raw)

    args.output_dir.mkdir(parents=True)
    train = args.output_dir / "train.parquet"
    pd.DataFrame.from_records(raw).to_parquet(train, index=False, **PARQUET_ENCODING)
    report = {
        **cdrop_report,
        "format": FORMAT,
        "rows": len(raw),
        "train": str(train.resolve()),
        "train_sha256": sha256(train),
        "source_cdrop_train_sha256": sha256(cdrop_train),
        "source_parent_train_sha256": sha256(parent_train),
        "family_rows": family_rows,
        "family_sample_loss_weights": {
            name: (float(masses[name]) / count if count else 0.0)
            for name, count in family_rows.items()
        },
        "family_total_gradient_mass": {name: float(masses[name]) for name in family_rows},
        "sample_loss_weight_mean": mean_weight,
        "components": dict(sorted(components.items())),
        "target_contracts": dict(sorted(contracts.items())),
        "semantic_groups": 9000,
        "views_per_semantic_group": 3,
        "matched_views_added": 18000,
        "view_group_weight_policy": "divide original dense L mass equally across dense/context/long, then globally normalize mean weight to one",
        "n_total_mass_equals_l": False,
        "visual_total_mass_equals_numeric": True,
        "benchmark_inputs_read": False,
        "benchmark_predictions_or_scores_read": False,
        "single_factor_change": "semantic-mass-preserving multiview replay",
        "promotion_status": "prospective current-view preservation control",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "DATA_READY").write_text("ok\n")
    print(json.dumps({"rows":len(raw),"family_rows":family_rows,"masses":dict(masses),"scale":scale},sort_keys=True))


if __name__ == "__main__":
    main()
