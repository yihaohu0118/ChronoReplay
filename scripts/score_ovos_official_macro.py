#!/usr/bin/env python3
"""Score OVO-S eval outputs with the paper-style subcategory macro average.

This is for OVO-S-Bench only (task_subcategory like 1.1.1, 4.3).  It must not
be used for unrelated online-video benchmark outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


FAMILIES = [
    "1.1", "1.2", "1.3",
    "2.1", "2.2", "2.3",
    "3.1", "3.2", "3.3",
    "4.1", "4.2", "4.3",
]
SUBCATEGORIES = [
    "1.1.1", "1.1.2", "1.1.3", "1.1.4",
    "1.2.1", "1.2.2", "1.2.3", "1.2.4",
    "1.3.1", "1.3.2", "1.3.3", "1.3.4", "1.3.5",
    "2.1.1", "2.1.2",
    "2.2.1", "2.2.2", "2.2.3",
    "2.3.1", "2.3.2",
    "3.1.1", "3.1.2", "3.1.3",
    "3.2.1",
    "3.3.1", "3.3.2", "3.3.3",
    "4.1", "4.2", "4.3",
]
SUBCATEGORIES_BY_LEVEL = {
    level: [task for task in SUBCATEGORIES if task.startswith(f"{level}.")]
    for level in ("1", "2", "3", "4")
}


def _family(task: str) -> str:
    parts = str(task).split(".")
    if len(parts) >= 2 and parts[0] in {"1", "2", "3", "4"}:
        return ".".join(parts[:2])
    return str(task)


def load_results(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        data = json.loads(path.read_text())
        for row in data.get("results", []):
            uid = str(row.get("id", ""))
            if uid and uid in seen:
                continue
            if uid:
                seen.add(uid)
            rows.append(row)
    return rows


def score(rows: list[dict]) -> dict:
    by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_subcategory: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    correct = 0
    for row in rows:
        task = str(row.get("task") or row.get("task_subcategory") or "")
        fam = _family(task)
        is_correct = int(bool(row.get("correct", False)))
        correct += is_correct
        by_family[fam][0] += is_correct
        by_family[fam][1] += 1
        by_subcategory[task][0] += is_correct
        by_subcategory[task][1] += 1

    family_scores = {}
    for fam in FAMILIES:
        c, n = by_family[fam]
        family_scores[fam] = {
            "correct": c,
            "total": n,
            "accuracy": c / n if n else None,
        }

    subcategory_scores = {}
    for task in SUBCATEGORIES:
        c, n = by_subcategory[task]
        subcategory_scores[task] = {
            "correct": c,
            "total": n,
            "accuracy": c / n if n else None,
        }

    level_scores = {}
    for level, subcategories in SUBCATEGORIES_BY_LEVEL.items():
        accs = [
            subcategory_scores[task]["accuracy"]
            for task in subcategories
            if subcategory_scores[task]["accuracy"] is not None
        ]
        level_scores[level] = {
            "accuracy": sum(accs) / len(accs) if accs else None,
            "subcategories": subcategories,
        }

    complete_families = all(family_scores[fam]["total"] > 0 for fam in FAMILIES)
    complete_subcategories = all(
        subcategory_scores[task]["total"] > 0 for task in SUBCATEGORIES
    )
    subcategory_accs = [
        subcategory_scores[task]["accuracy"]
        for task in SUBCATEGORIES
        if subcategory_scores[task]["accuracy"] is not None
    ]
    observed_macro = (
        sum(float(a) for a in subcategory_accs) / len(subcategory_accs)
        if subcategory_accs else None
    )
    prediction_sources: dict[str, int] = defaultdict(int)
    observed_nframes: dict[str, int] = defaultdict(int)
    observed_prompt_styles: dict[str, int] = defaultdict(int)
    observed_frame_policies: dict[str, int] = defaultdict(int)
    observed_requested_frame_policies: dict[str, int] = defaultdict(int)
    observed_frame_policy_reasons: dict[str, int] = defaultdict(int)
    observed_media_serializations: dict[str, int] = defaultdict(int)
    observed_native_video_fps_protocols: dict[str, int] = defaultdict(int)
    observed_processor_paths: dict[str, int] = defaultdict(int)
    for row in rows:
        prediction_sources[str(row.get("prediction_source") or "legacy_unspecified")] += 1
        observed_nframes[str(row.get("nframes", "legacy_unspecified"))] += 1
        observed_prompt_styles[
            str(row.get("prompt_style") or "legacy_unspecified")
        ] += 1
        observed_frame_policies[
            str(row.get("frame_policy") or "legacy_unspecified")
        ] += 1
        observed_requested_frame_policies[
            str(row.get("requested_frame_policy") or "legacy_unspecified")
        ] += 1
        observed_frame_policy_reasons[
            str(row.get("frame_policy_reason") or "legacy_unspecified")
        ] += 1
        observed_media_serializations[
            str(row.get("media_serialization") or "ordered_image_blocks")
        ] += 1
        observed_native_video_fps_protocols[
            str(row.get("native_video_fps_protocol") or "not_applicable")
        ] += 1
        observed_processor_paths[
            str(row.get("processor_path") or "legacy_unspecified")
        ] += 1
    return {
        "total": len(rows),
        "correct": correct,
        "raw_accuracy": correct / max(len(rows), 1),
        # Backward-compatible numeric field used by already-running fixed-gate
        # watchers.  It is an official 30-subcategory macro only when
        # official_macro_valid is true; subset gates must be described by the
        # explicit observed_subcategory_macro_accuracy field below.
        "official_macro_accuracy": observed_macro,
        "official_macro_valid": complete_subcategories,
        "observed_subcategory_macro_accuracy": observed_macro,
        "observed_subcategory_count": len(subcategory_accs),
        "complete_all_families": complete_families,
        "complete_all_subcategories": complete_subcategories,
        "missing_families": [
            fam for fam in FAMILIES if family_scores[fam]["total"] == 0
        ],
        "missing_subcategories": [
            task
            for task in SUBCATEGORIES
            if subcategory_scores[task]["total"] == 0
        ],
        "prediction_sources": dict(sorted(prediction_sources.items())),
        "observed_nframes": dict(sorted(observed_nframes.items())),
        "observed_prompt_styles": dict(sorted(observed_prompt_styles.items())),
        "observed_frame_policies": dict(sorted(observed_frame_policies.items())),
        "observed_requested_frame_policies": dict(
            sorted(observed_requested_frame_policies.items())
        ),
        "observed_frame_policy_reasons": dict(
            sorted(observed_frame_policy_reasons.items())
        ),
        "observed_media_serializations": dict(
            sorted(observed_media_serializations.items())
        ),
        "observed_native_video_fps_protocols": dict(
            sorted(observed_native_video_fps_protocols.items())
        ),
        "observed_processor_paths": dict(sorted(observed_processor_paths.items())),
        "by_level": level_scores,
        "by_family": family_scores,
        "by_subcategory": subcategory_scores,
    }


def load_expected_counts(annotation: Path) -> dict[str, int]:
    # Keep pandas optional so the scorer still works on plain result JSON files.
    import pandas as pd

    df = pd.read_parquet(annotation)
    counts = {fam: 0 for fam in FAMILIES}
    for task in df["task_subcategory"].astype(str):
        fam = _family(task)
        if fam in counts:
            counts[fam] += 1
    return counts


def _id_set_sha256(values: list[str]) -> str:
    payload = "\n".join(sorted(values)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--annotation", type=Path, default=None,
                        help="OVO-S parquet for full-coverage checks")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--processor-path",
        type=Path,
        default=None,
        help="Pinned processor source used for every prediction row.",
    )
    parser.add_argument(
        "--inference-protocol",
        choices=(
            "legacy_unspecified",
            "model_only",
            "same_model_active_evidence",
            "deterministic_agent_harness",
            "deterministic_categorical_executor",
            "verified_state_projection",
            "historical_l43_matcher",
        ),
        default="legacy_unspecified",
        help="Record whether any external L4.3 matcher was enabled.",
    )
    parser.add_argument(
        "--frame-sampling-protocol",
        choices=(
            "uniform_128",
            "uniform_fixed_causal",
            "uniform_subsample128_fixed_causal",
            "cyclic_fixed_causal",
            "recent_window_fixed_causal",
            "log_decay_fixed_causal",
            "hybrid_global_recent_fixed_causal",
            "referenced_pair_fixed_causal",
            "causal_pyramid_fixed_causal",
            "semantic_visible_scope_causal",
            "semantic_adaptive_causal",
            "causal_origin_stable_support_causal",
            "semantic_temporal_dereference_causal",
            "uniform_temporal_dereference_causal",
            "active_spatial_evidence128_fixed_causal",
        ),
        default="uniform_128",
        help="Record the deterministic frame preprocessing used for inference.",
    )
    parser.add_argument(
        "--requested-default-nframes",
        type=int,
        default=None,
        help="Record the global frame request before any explicit route override.",
    )
    parser.add_argument("--semantic-uniform-nframes", type=int, default=128)
    parser.add_argument("--semantic-recent-nframes", type=int, default=16)
    parser.add_argument("--semantic-reference-nframes", type=int, default=8)
    parser.add_argument(
        "--prompt-protocol",
        default="unspecified",
        help="Record the prompt contract without changing predictions.",
    )
    parser.add_argument(
        "--media-serialization-protocol",
        choices=("ordered_image_blocks", "native_video"),
        default="ordered_image_blocks",
    )
    parser.add_argument(
        "--native-video-fps-protocol",
        choices=("not_applicable", "fixed_config", "effective_causal_span"),
        default="not_applicable",
    )
    args = parser.parse_args()

    rows = load_results(args.results)
    report = score(rows)
    if args.model_path:
        model_path = args.model_path.resolve()
        config_path = model_path / "config.json"
        index_path = model_path / "model.safetensors.index.json"
        report["model_path"] = str(model_path)
        report["adapter_config_present"] = (model_path / "adapter_config.json").is_file()
        report["model_config_sha256"] = (
            hashlib.sha256(config_path.read_bytes()).hexdigest()
            if config_path.is_file() else None
        )
        report["model_index_sha256"] = (
            hashlib.sha256(index_path.read_bytes()).hexdigest()
            if index_path.is_file() else None
        )
    if args.processor_path:
        processor_path = args.processor_path.resolve()
        preprocessor_config = processor_path / "preprocessor_config.json"
        if not preprocessor_config.is_file():
            raise SystemExit(f"missing pinned processor config: {preprocessor_config}")
        expected_processors = {str(processor_path): len(rows)}
        if report["observed_processor_paths"] != expected_processors:
            raise SystemExit(
                "processor source drift: expected "
                f"{expected_processors}, observed {report['observed_processor_paths']}"
            )
        report["processor_path"] = str(processor_path)
        report["processor_config_sha256"] = hashlib.sha256(
            preprocessor_config.read_bytes()
        ).hexdigest()
    report["inference_protocol"] = args.inference_protocol
    report["external_l43_matcher"] = (
        None
        if args.inference_protocol == "legacy_unspecified"
        else args.inference_protocol == "historical_l43_matcher"
    )
    report["frame_sampling_protocol"] = args.frame_sampling_protocol
    if args.requested_default_nframes is not None:
        if args.requested_default_nframes < 1:
            raise SystemExit("--requested-default-nframes must be positive")
        report["requested_default_nframes"] = args.requested_default_nframes
    report["prompt_protocol"] = args.prompt_protocol
    report["media_serialization_protocol"] = args.media_serialization_protocol
    expected_media = {args.media_serialization_protocol: len(rows)}
    if report["observed_media_serializations"] != expected_media:
        raise SystemExit(
            "media serialization does not match prediction rows: "
            f"{report['observed_media_serializations']}"
        )
    report["native_video_fps_protocol"] = args.native_video_fps_protocol
    expected_fps_protocol = {args.native_video_fps_protocol: len(rows)}
    if report["observed_native_video_fps_protocols"] != expected_fps_protocol:
        raise ValueError(
            "native-video FPS protocol drift: expected "
            f"{expected_fps_protocol}, observed "
            f"{report['observed_native_video_fps_protocols']}"
        )
    if args.inference_protocol == "model_only":
        non_model = {
            source: count
            for source, count in report["prediction_sources"].items()
            if source != "model" and count
        }
        if non_model:
            raise SystemExit(
                "model_only scoring requires auditable model predictions; "
                f"found non-model or legacy sources: {non_model}"
            )
    if args.inference_protocol == "verified_state_projection":
        invalid_sources = {
            source: count
            for source, count in report["prediction_sources"].items()
            if source not in {"model", "verified_state_projection"} and count
        }
        if invalid_sources or not report["prediction_sources"].get(
            "verified_state_projection", 0
        ):
            raise SystemExit(
                "verified_state_projection scoring requires a nonempty, auditable "
                f"state selection with model fallback; invalid={invalid_sources}"
            )
    if args.frame_sampling_protocol in {
        "uniform_fixed_causal",
        "uniform_subsample128_fixed_causal",
        "recent_window_fixed_causal",
        "log_decay_fixed_causal",
        "hybrid_global_recent_fixed_causal",
        "referenced_pair_fixed_causal",
    }:
        if args.requested_default_nframes is None:
            raise SystemExit(
                f"{args.frame_sampling_protocol} scoring requires "
                "--requested-default-nframes"
            )
        expected_frames = str(args.requested_default_nframes)
        if report["observed_nframes"] != {expected_frames: len(rows)}:
            raise SystemExit(
                "fixed causal frame protocol does not match prediction rows: "
                f"{report['observed_nframes']}"
            )
        report["causal_prefix_only"] = True
        report["long_query_time_cap_seconds"] = None
        report["benchmark_metadata_actor_visible"] = False
    if args.frame_sampling_protocol == "causal_pyramid_fixed_causal":
        if args.requested_default_nframes is None:
            raise SystemExit(
                "causal_pyramid_fixed_causal scoring requires "
                "--requested-default-nframes"
            )
        expected_frames = str(args.requested_default_nframes)
        if report["observed_nframes"] != {expected_frames: len(rows)}:
            raise SystemExit(
                "causal pyramid budget does not match prediction rows: "
                f"{report['observed_nframes']}"
            )
        invalid = [
            {
                "id": row.get("id"),
                "requested": row.get("requested_frame_policy"),
                "resolved": row.get("frame_policy"),
                "reason": row.get("frame_policy_reason"),
            }
            for row in rows
            if row.get("requested_frame_policy") != "causal_pyramid"
            or row.get("frame_policy") != "causal_pyramid"
            or row.get("frame_policy_reason") != "fixed_policy"
        ]
        if invalid:
            raise SystemExit(
                "causal pyramid protocol does not match prediction rows: "
                f"{invalid[:5]}"
            )
        report["causal_prefix_only"] = True
        report["long_query_time_cap_seconds"] = None
        report["benchmark_metadata_actor_visible"] = False
        report["frame_sampler_inputs"] = [
            "causal_prefix_end_time",
            "video_fps",
        ]
        report["question_text_frame_router"] = False
        report["causal_pyramid_allocation"] = {
            "global_prefix": 0.50,
            "log_history": 0.25,
            "dense_recent": 0.25,
        }
    if args.frame_sampling_protocol == "semantic_visible_scope_causal":
        if min(
            args.semantic_uniform_nframes,
            args.semantic_recent_nframes,
        ) < 1:
            raise SystemExit("semantic frame budgets must be positive")
        expected = {
            "uniform": args.semantic_uniform_nframes,
            "recent": args.semantic_recent_nframes,
        }
        invalid = []
        for row in rows:
            resolved = str(row.get("frame_policy") or "")
            requested = str(row.get("requested_frame_policy") or "")
            try:
                budget = int(row.get("nframes") or 0)
            except (TypeError, ValueError):
                budget = 0
            if (
                requested != "semantic_current"
                or resolved not in expected
                or budget != expected.get(resolved)
            ):
                invalid.append(
                    {
                        "id": row.get("id"),
                        "requested": requested,
                        "resolved": resolved,
                        "nframes": budget,
                    }
                )
        if invalid:
            raise SystemExit(
                "semantic visible-scope protocol does not match prediction rows: "
                f"{invalid[:5]}"
            )
        report["causal_prefix_only"] = True
        report["long_query_time_cap_seconds"] = None
        report["benchmark_metadata_actor_visible"] = False
        report["semantic_uniform_nframes"] = args.semantic_uniform_nframes
        report["semantic_recent_nframes"] = args.semantic_recent_nframes
        report["semantic_router_inputs"] = ["visible_question_text"]
    if args.frame_sampling_protocol == "semantic_adaptive_causal":
        if min(
            args.semantic_uniform_nframes,
            args.semantic_recent_nframes,
        ) < 1:
            raise SystemExit("semantic frame budgets must be positive")
        expected = {
            "uniform": args.semantic_uniform_nframes,
            "log_decay": args.semantic_uniform_nframes,
            "recent": args.semantic_recent_nframes,
        }
        invalid = []
        for row in rows:
            resolved = str(row.get("frame_policy") or "")
            requested = str(row.get("requested_frame_policy") or "")
            try:
                budget = int(row.get("nframes") or 0)
            except (TypeError, ValueError):
                budget = 0
            if (
                requested != "semantic_adaptive"
                or resolved not in expected
                or budget != expected.get(resolved)
            ):
                invalid.append(
                    {
                        "id": row.get("id"),
                        "requested": requested,
                        "resolved": resolved,
                        "nframes": budget,
                    }
                )
        if invalid:
            raise SystemExit(
                "semantic adaptive protocol does not match prediction rows: "
                f"{invalid[:5]}"
            )
        report["causal_prefix_only"] = True
        report["long_query_time_cap_seconds"] = None
        report["benchmark_metadata_actor_visible"] = False
        report["semantic_uniform_nframes"] = args.semantic_uniform_nframes
        report["semantic_recent_nframes"] = args.semantic_recent_nframes
        report["semantic_router_inputs"] = ["visible_question_text"]
    if args.frame_sampling_protocol in {
        "semantic_temporal_dereference_causal",
        "uniform_temporal_dereference_causal",
    }:
        if min(
            args.semantic_uniform_nframes,
            args.semantic_recent_nframes,
            args.semantic_reference_nframes,
        ) < 1:
            raise SystemExit("semantic dereference frame budgets must be positive")
        uniform_fallback = (
            args.frame_sampling_protocol
            == "uniform_temporal_dereference_causal"
        )
        requested_policy = (
            "uniform_temporal_dereference"
            if uniform_fallback else "semantic_dereference"
        )
        expected = {
            "uniform": args.semantic_uniform_nframes,
            "referenced_pair": args.semantic_reference_nframes,
        }
        if not uniform_fallback:
            expected.update({
                "log_decay": args.semantic_uniform_nframes,
                "recent": args.semantic_recent_nframes,
            })
        invalid = []
        invalid_traces = []
        executor_calls = 0
        for row in rows:
            resolved = str(row.get("frame_policy") or "")
            requested = str(row.get("requested_frame_policy") or "")
            try:
                budget = int(row.get("nframes") or 0)
            except (TypeError, ValueError):
                budget = 0
            if (
                requested != requested_policy
                or resolved not in expected
                or budget != expected.get(resolved)
            ):
                invalid.append(
                    {
                        "id": row.get("id"),
                        "requested": requested,
                        "resolved": resolved,
                        "nframes": budget,
                    }
                )
            trace = row.get("harness_state")
            trace_ok = (
                isinstance(trace, dict)
                and trace.get("format")
                == "iclr_spatial.causal_state_harness_trace.v1"
                and trace.get("benchmark_metadata_actor_visible") is False
                and trace.get("external_generative_model_or_api") is False
                and isinstance(trace.get("compiled_query"), dict)
            )
            if trace_ok and resolved == "referenced_pair":
                state = trace.get("executor_result")
                indices = state.get("extracted_frame_indices") if isinstance(state, dict) else None
                timestamps = (
                    state.get("extracted_frame_timestamps_seconds")
                    if isinstance(state, dict) else None
                )
                query_time = state.get("query_time_seconds") if isinstance(state, dict) else None
                evidence_hashes = (
                    state.get("evidence_frame_sha256")
                    if isinstance(state, dict)
                    else None
                )
                frame_count = (
                    state.get("video_frame_count")
                    if isinstance(state, dict)
                    else None
                )
                source_metadata_valid = bool(
                    isinstance(state, dict)
                    and (
                        (
                            int(state.get("video_size_bytes") or 0) > 0
                            and int(state.get("video_mtime_ns") or 0) > 0
                            and state.get("source_metadata_from_cache_only") is not True
                        )
                        or (
                            state.get("source_metadata_from_cache_only") is True
                            and state.get("video_size_bytes") is None
                            and state.get("video_mtime_ns") is None
                        )
                    )
                )
                trace_ok = (
                    trace.get("executor_status") == "ok"
                    and isinstance(state, dict)
                    and state.get("format")
                    == "iclr_spatial.temporal_dereference_state.v1"
                    and state.get("executor") == "explicit_temporal_dereference"
                    and state.get("status") == "ok"
                    and state.get("causal_prefix_verified") is True
                    and state.get("source_fields")
                    == ["visible_question_timestamp", "causal_video_prefix"]
                    and isinstance(state.get("canonical_video_path_sha256"), str)
                    and len(state["canonical_video_path_sha256"]) == 64
                    and source_metadata_valid
                    and int(state.get("requested_frame_budget") or 0)
                    == args.semantic_reference_nframes
                    and isinstance(indices, list)
                    and 1 < len(indices) <= args.semantic_reference_nframes
                    and indices == sorted(set(indices))
                    and isinstance(frame_count, int)
                    and frame_count > 0
                    and all(0 <= int(value) < frame_count for value in indices)
                    and isinstance(timestamps, list)
                    and len(timestamps) == len(indices)
                    and isinstance(evidence_hashes, list)
                    and len(evidence_hashes) == len(indices)
                    and all(
                        isinstance(value, str)
                        and len(value) == 64
                        and all(character in "0123456789abcdef" for character in value)
                        for value in evidence_hashes
                    )
                    and state.get("evidence_hash_protocol")
                    == "sha256(pil_mode||pil_size||pixel_bytes)"
                    and isinstance(state.get("executor_wall_time_ms"), (int, float))
                    and float(state["executor_wall_time_ms"]) >= 0
                    and isinstance(query_time, (int, float))
                    and all(
                        isinstance(value, (int, float))
                        and 0 <= float(value) <= float(query_time) + 1e-9
                        for value in timestamps
                    )
                )
                executor_calls += int(trace_ok)
            elif trace_ok:
                trace_ok = (
                    trace.get("executor_status")
                    == "fallback_no_promoted_executor"
                    and trace.get("executor_result") is None
                )
            if not trace_ok:
                invalid_traces.append({"id": row.get("id"), "resolved": resolved})
        if invalid:
            raise SystemExit(
                "semantic temporal-dereference protocol does not match rows: "
                f"{invalid[:5]}"
            )
        if invalid_traces:
            raise SystemExit(
                "semantic temporal-dereference harness trace is missing or invalid: "
                f"{invalid_traces[:5]}"
            )
        report["causal_prefix_only"] = True
        report["long_query_time_cap_seconds"] = None
        report["benchmark_metadata_actor_visible"] = False
        report["semantic_uniform_nframes"] = args.semantic_uniform_nframes
        report["semantic_recent_nframes"] = args.semantic_recent_nframes
        report["semantic_reference_nframes"] = args.semantic_reference_nframes
        report["temporal_dereference_fallback_policy"] = (
            "uniform" if uniform_fallback else "semantic_adaptive"
        )
        report["semantic_router_inputs"] = [
            "visible_question_text",
            "visible_option_text",
        ]
        report["agent_executors"] = ["explicit_temporal_dereference"]
        report["external_generative_model_or_api"] = False
        report["harness_trace_format"] = "iclr_spatial.causal_state_harness_trace.v1"
        report["harness_trace_rows"] = len(rows)
        report["executor_call_counts"] = {
            "explicit_temporal_dereference": executor_calls,
            "fallback_no_promoted_executor": len(rows) - executor_calls,
        }
        report["executor_state_provenance_verified"] = True
    if args.frame_sampling_protocol == "adaptive_causal_agent":
        invalid_counts = {
            value: count
            for value, count in report["observed_nframes"].items()
            if not value.isdigit() or int(value) < 1
        }
        if invalid_counts:
            raise SystemExit(
                "adaptive_causal_agent requires positive audited frame counts: "
                f"{invalid_counts}"
            )
        report["causal_prefix_only"] = True
        report["benchmark_metadata_actor_visible"] = False
    if args.prompt_protocol == "taskblind_default":
        if report["observed_prompt_styles"] != {"taskblind_default": len(rows)}:
            raise SystemExit(
                "task-blind prompt protocol does not match prediction rows: "
                f"{report['observed_prompt_styles']}"
            )
    if args.annotation:
        annotation_path = args.annotation.resolve()
        report["annotation_path"] = str(annotation_path)
        report["annotation_sha256"] = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
        import pandas as pd
        annotation_frame = pd.read_parquet(annotation_path)
        if "id" in annotation_frame:
            expected_ids = annotation_frame["id"].astype(str).tolist()
            prediction_ids = [str(row.get("id", "")) for row in rows]
            report["annotation_id_set_sha256"] = _id_set_sha256(expected_ids)
            report["prediction_id_set_sha256"] = _id_set_sha256(prediction_ids)
            report["prediction_ids_match_annotation"] = (
                len(prediction_ids) == len(expected_ids)
                and len(set(prediction_ids)) == len(prediction_ids)
                and set(prediction_ids) == set(expected_ids)
            )
        expected = load_expected_counts(args.annotation)
        report["expected_total"] = sum(expected.values())
        report["expected_by_family"] = expected
        undercovered = {
            fam: {
                "have": report["by_family"][fam]["total"],
                "expected": n,
            }
            for fam, n in expected.items()
            if report["by_family"][fam]["total"] != n
        }
        report["undercovered_families"] = undercovered
        report["full_coverage"] = (
            not undercovered
            and report["total"] == sum(expected.values())
            and report.get("prediction_ids_match_annotation", True)
        )
        # With an annotation supplied, an official macro is valid only for an
        # exact prediction set.  Merely retaining one example per category can
        # otherwise hide skipped/corrupt media behind a plausible macro score.
        report["official_macro_valid"] = bool(
            report["complete_all_subcategories"] and report["full_coverage"]
        )

    macro_label = (
        "official_macro"
        if report["official_macro_valid"]
        else "observed_subcategory_macro"
    )
    print(
        f"OVO-S rows={report['total']} raw={_pct(report['raw_accuracy'])} "
        f"{macro_label}={_pct(report['official_macro_accuracy'])} "
        f"complete={report['complete_all_subcategories']}"
    )
    if args.annotation:
        print(
            f"coverage={report['total']}/{report['expected_total']} "
            f"full_coverage={report['full_coverage']}"
        )
    print("levels:", {
        f"L{level}": _pct(data["accuracy"])
        for level, data in report["by_level"].items()
    })
    if report["missing_families"]:
        print("missing_families:", ",".join(report["missing_families"]))
    if report["missing_subcategories"]:
        print("missing_subcategories:", ",".join(report["missing_subcategories"]))
    if args.annotation and report["undercovered_families"]:
        preview = ", ".join(
            f"{fam}:{v['have']}/{v['expected']}"
            for fam, v in sorted(report["undercovered_families"].items())[:12]
        )
        print("undercovered_families:", preview)
    for fam in FAMILIES:
        data = report["by_family"][fam]
        suffix = ""
        if args.annotation:
            suffix = f"/{report['expected_by_family'].get(fam, 0)} expected"
        print(
            f"  {fam}: {data['correct']}/{data['total']}="
            f"{_pct(data['accuracy'])}{suffix}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
