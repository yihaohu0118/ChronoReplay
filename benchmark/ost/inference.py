#!/usr/bin/env python3
"""Streaming multi-turn OST-Bench inference for Qwen3-VL under vLLM.

The upstream harness replays one conversation at a time with
``transformers.generate``.  The protocol is reproduced exactly here -- same
grouping by ``scan_id``, same turn order, the system prompt repeated inside
every user turn, ``add_vision_id`` enabled, and the model's own reply appended
to the history -- but turns are issued round by round so that every
conversation in a chunk shares one batched forward pass.  Round t of all
conversations is generated together, then appended, then round t+1; the
sequential dependency inside a conversation is preserved.

Outputs are written per ``scan_id`` in the upstream response format so the
official evaluator can read them unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from synthesis.spatial_query_compiler import compile_spatial_query

FORMAT = "iclr_spatial.ost_bench_inference.v1"

# Upstream QwenVL_baseline.py pixel budget, kept so the reported numbers stay
# comparable with the published OST-Bench table.
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 256 * 28 * 28


def _consensus_response(completions: list[str]) -> str:
    """Return a deterministic semantic consensus without consulting labels."""
    if len(completions) == 1:
        return completions[0]
    from benchmark.ost.score import process_answer

    answers = [process_answer(text) for text in completions]
    numbers = []
    for value in answers:
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            numbers = []
            break
    if numbers:
        ordered = sorted(numbers)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle])
        )
        return json.dumps(
            {
                "answer": round(median, 6),
                "reason": "Median of independent low-temperature causal answers.",
            }
        )
    normalized = [str(value).strip().lower() for value in answers]
    counts = Counter(normalized)
    winner = max(counts, key=lambda item: (counts[item], -normalized.index(item)))
    return completions[normalized.index(winner)]


def conversations(annotation: Path) -> dict[str, dict[str, Any]]:
    """Group raw rows into one ordered conversation per scan."""

    payload = json.loads(annotation.read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, Any]] = {}
    for row in payload:
        scan_id = str(row["scan_id"])
        entry = grouped.setdefault(
            scan_id,
            {
                "scan_id": scan_id,
                "system_prompt": row["system_prompt"],
                "user_message": [],
            },
        )
        if entry["system_prompt"] != row["system_prompt"]:
            raise ValueError(f"{scan_id} has an inconsistent system prompt")
        turn = dict(row)
        turn["image_paths"] = list(turn.pop("new_observations"))
        turn["prompt"] = turn.pop("user_message")
        entry["user_message"].append(turn)
    for entry in grouped.values():
        entry["user_message"].sort(key=lambda item: int(item["turn_id"]))
        expected = list(range(1, len(entry["user_message"]) + 1))
        actual = [int(item["turn_id"]) for item in entry["user_message"]]
        if actual != expected:
            raise ValueError(f"{entry['scan_id']} has non-contiguous turn ids")
    return grouped


def candidate_responses(root: Path) -> dict[tuple[str, int], str]:
    """Load one complete OST prediction directory as verifier candidates."""
    output: dict[tuple[str, int], str] = {}
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scan_id = str(payload["scan_id"])
        for turn in payload["user_message"]:
            key = (scan_id, int(turn["turn_id"]))
            if key in output:
                raise ValueError(f"duplicate candidate response: {key}")
            output[key] = str(turn.get("response") or "")
    return output


def _admissible(manifest: Path | None) -> set[str] | None:
    if manifest is None:
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("format") != "iclr_spatial.scene_disjoint_eval_manifest.v1":
        raise ValueError("not a scene-disjoint evaluation manifest")
    return {str(item) for item in payload["admissible_scene_ids"]}


def _user_turn(prompt: str, images: list[str], system_prompt: str) -> dict[str, Any]:
    # Upstream repeats the system prompt as the first text block of every user
    # turn rather than using a system role.  Reproduced verbatim.
    content: list[dict[str, Any]] = [{"type": "text", "text": system_prompt}]
    content.extend({"type": "image", "image": path} for path in images)
    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def causal_support_turn_indices(
    total_turns: int, view: str, *, key: str = ""
) -> list[int]:
    """Instantiate the question-independent spatial view over OST turns."""

    from synthesis.semantic_frame_policy import temporal_support_turn_indices

    if view == "cyclic":
        # Keep the current query block last and rotate only the completed
        # historical blocks.  This preserves a valid answer-generation
        # frontier while destroying the history's global causal origin.
        history_turns = total_turns - 1
        if history_turns < 2:
            return list(range(total_turns))
        digest = int.from_bytes(
            hashlib.sha256(
                f"ost-causal-origin-v1|{key}|{total_turns}".encode()
            ).digest()[:8],
            "big",
        )
        shift = 1 + digest % (history_turns - 1)
        return (
            list(range(shift, history_turns))
            + list(range(shift))
            + [total_turns - 1]
        )
    policy = {
        "full": "uniform",
        "recent": "recent",
        "event": "log_decay",
    }.get(view)
    if policy is None:
        raise ValueError(f"unsupported OST causal-support view: {view}")
    return temporal_support_turn_indices(total_turns, policy)


def select_causal_turn_support(
    history: list[dict[str, Any]],
    all_images: list[Any],
    image_counts: list[int],
    *,
    view: str,
    key: str = "",
) -> tuple[list[dict[str, Any]], list[Any], list[int]]:
    """Select complete user/assistant turn blocks and aligned images.

    The final/current block may contain only its user message because its
    answer has not yet been generated.  Earlier blocks must be complete.
    """

    if not image_counts:
        raise ValueError("causal support requires at least one turn")
    expected_messages = 2 * len(image_counts) - 1
    if len(history) != expected_messages:
        raise ValueError(
            f"history/turn geometry drifted: messages={len(history)} "
            f"turns={len(image_counts)}"
        )
    if sum(image_counts) != len(all_images) or any(value < 1 for value in image_counts):
        raise ValueError("turn image accounting drifted")
    selected = causal_support_turn_indices(len(image_counts), view, key=key)
    selected_messages: list[dict[str, Any]] = []
    selected_images: list[Any] = []
    image_offsets = [0]
    for count in image_counts:
        image_offsets.append(image_offsets[-1] + count)
    for turn_index in selected:
        selected_messages.append(history[2 * turn_index])
        assistant_index = 2 * turn_index + 1
        if assistant_index < len(history):
            selected_messages.append(history[assistant_index])
        selected_images.extend(
            all_images[image_offsets[turn_index] : image_offsets[turn_index + 1]]
        )
    return selected_messages, selected_images, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--causal-support-view",
        choices=("full", "recent", "event", "cyclic"),
        default="full",
        help=(
            "Question-independent causal view over observation turns. full "
            "keeps the prefix, recent keeps the current turn, and event keeps "
            "the current/origin plus power-of-two historical turns. cyclic "
            "is a content-matched origin-shifted reference used only for scoring."
        ),
    )
    parser.add_argument(
        "--classical-trajectory",
        action="store_true",
        help=(
            "Ingest each observation chunk with the question-blind optical-flow "
            "writer and append a qualitative pose packet when the visible "
            "question compiles to query_camera_pose."
        ),
    )
    parser.add_argument(
        "--self-refine",
        action="store_true",
        help=(
            "after each first response, run one task-blind verify-and-correct "
            "pass over the identical causal history and images"
        ),
    )
    parser.add_argument(
        "--spatial-ledger-prompt",
        action="store_true",
        help="append one task-blind causal spatial-ledger instruction to every turn",
    )
    parser.add_argument(
        "--persistent-state-writer",
        action="store_true",
        help=(
            "run one question-independent state update per new observation chunk, "
            "then expose the frozen state to the question-dependent reader"
        ),
    )
    parser.add_argument(
        "--compact-state-writer",
        action="store_true",
        help="use a bounded one-line schema for the query-independent persistent writer",
    )
    parser.add_argument(
        "--append-only-state-writer",
        action="store_true",
        help="write a bounded question-independent delta per chunk and append it to an immutable ledger",
    )
    parser.add_argument(
        "--writer-max-tokens",
        type=int,
        default=0,
        help="override persistent-writer generation budget; zero selects the style default",
    )
    parser.add_argument(
        "--writer-arbitration",
        action="store_true",
        help="globally arbitrate writer memory against raw visual geometry in the reader",
    )
    parser.add_argument("--self-consistency-samples", type=int, default=1)
    parser.add_argument("--self-consistency-temperature", type=float, default=0.3)
    parser.add_argument(
        "--proposal-logprobs",
        action="store_true",
        help="record length-normalized generation log-prob for view selection",
    )
    parser.add_argument("--candidate-a-dir", type=Path, default=None)
    parser.add_argument("--candidate-b-dir", type=Path, default=None)
    parser.add_argument("--candidate-choice-only", action="store_true")
    parser.add_argument(
        "--operator-grounded-candidate-choice",
        action="store_true",
        help="bind candidate arbitration to a visible-language spatial operator",
    )
    parser.add_argument(
        "--candidate-answer-only",
        action="store_true",
        help="show normalized candidate answers without their generated rationales",
    )
    parser.add_argument(
        "--candidate-order-flip",
        action="store_true",
        help="invert the deterministic candidate order for an equivariance control",
    )
    parser.add_argument(
        "--candidate-independent-verification",
        action="store_true",
        help="score each candidate in a separate symmetric compatibility prompt",
    )
    args = parser.parse_args()

    writer_modes = sum(
        int(value)
        for value in (
            args.persistent_state_writer,
            args.compact_state_writer,
            args.append_only_state_writer,
        )
    )
    if writer_modes > 1:
        raise SystemExit("choose exactly one persistent writer style")
    if args.writer_max_tokens < 0:
        raise SystemExit("writer-max-tokens must be non-negative")
    if args.writer_arbitration and writer_modes == 0:
        raise SystemExit("writer-arbitration requires a persistent writer mode")
    if args.self_consistency_samples < 1:
        raise SystemExit("self-consistency-samples must be positive")
    if args.self_consistency_samples > 1 and args.self_consistency_temperature <= 0:
        raise SystemExit("multi-sample self-consistency requires positive temperature")
    if (args.candidate_a_dir is None) != (args.candidate_b_dir is None):
        raise SystemExit("candidate verifier requires both candidate directories")
    if args.candidate_choice_only and args.candidate_a_dir is None:
        raise SystemExit("candidate-choice-only requires candidate directories")
    if args.operator_grounded_candidate_choice and not args.candidate_choice_only:
        raise SystemExit(
            "operator-grounded candidate choice requires candidate-choice-only"
        )
    if (
        args.candidate_answer_only
        or args.candidate_order_flip
        or args.candidate_independent_verification
    ) and not args.candidate_choice_only:
        raise SystemExit(
            "candidate answer/order controls require candidate-choice-only"
        )
    if args.candidate_independent_verification and any(
        (
            args.classical_trajectory,
            args.spatial_ledger_prompt,
            writer_modes,
            args.self_refine,
            args.self_consistency_samples > 1,
            args.proposal_logprobs,
        )
    ):
        raise SystemExit(
            "independent candidate verification requires the plain deterministic reader"
        )
    if args.causal_support_view != "full" and any(
        (
            args.classical_trajectory,
            args.self_refine,
            writer_modes,
            args.writer_arbitration,
            args.self_consistency_samples > 1,
            args.candidate_a_dir is not None,
        )
    ):
        raise SystemExit(
            "causal-support view isolation requires the plain single-proposal reader"
        )

    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard specification")
    for path in (args.annotation, args.model / "config.json"):
        if not path.is_file():
            raise SystemExit(f"missing artifact: {path}")
    if not args.image_root.is_dir():
        raise SystemExit(f"missing image root: {args.image_root}")

    grouped = conversations(args.annotation)
    candidate_a = (
        candidate_responses(args.candidate_a_dir)
        if args.candidate_a_dir is not None
        else None
    )
    candidate_b = (
        candidate_responses(args.candidate_b_dir)
        if args.candidate_b_dir is not None
        else None
    )
    admissible = _admissible(args.manifest)
    scan_ids = sorted(grouped)
    if admissible is not None:
        scan_ids = [scan_id for scan_id in scan_ids if scan_id in admissible]
    scan_ids = scan_ids[args.shard_index :: args.shard_count]
    args.output.mkdir(parents=True, exist_ok=True)
    pending = [
        scan_id
        for scan_id in scan_ids
        if not (args.output / f"{scan_id}.json").is_file()
    ]
    print(
        json.dumps(
            {
                "shard": [args.shard_index, args.shard_count],
                "assigned_scans": len(scan_ids),
                "pending_scans": len(pending),
            }
        ),
        flush=True,
    )
    if not pending:
        return

    empty: list[str] = []
    for scan_id in pending:
        for turn in grouped[scan_id]["user_message"]:
            for relative in turn["image_paths"]:
                path = args.image_root / relative
                if not path.is_file():
                    raise SystemExit(f"missing observation: {path}")
                if path.stat().st_size == 0:
                    empty.append(str(path))
    if empty:
        raise SystemExit(
            f"empty observation files: {empty[:5]} n={len(empty)}"
        )

    # vLLM forks its engine core, which cannot inherit an initialized CUDA
    # context.  Set this before the import rather than relying on the caller.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from PIL import Image, ImageFile
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    # The public OST image tree contains a small number of PNG files whose
    # final stream marker is truncated although their decoded pixels are
    # available.  PIL's standard tolerant mode keeps those observations while
    # preserving the same pixels and protocol for every compared checkpoint.
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    processor = AutoProcessor.from_pretrained(
        str(args.model), min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    llm = LLM(
        model=str(args.model),
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.max_images},
        mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=False,
        trust_remote_code=True,
        seed=0,
    )
    sampling = SamplingParams(
        temperature=(
            args.self_consistency_temperature
            if args.self_consistency_samples > 1
            else 0.0
        ),
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        n=args.self_consistency_samples,
        logprobs=1 if args.proposal_logprobs else None,
    )
    writer_max_tokens = args.writer_max_tokens or (
        128
        if args.append_only_state_writer
        else 192
        if args.compact_state_writer
        else 256
    )
    writer_sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=writer_max_tokens,
    )
    binary_sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=4,
    )

    truncated = 0
    timelines: dict[str, Any] = {}
    if args.classical_trajectory:
        from spatial_witness.observation_chunk_pose import (
            ObservationChunkPoseTimeline,
        )
    for start in range(0, len(pending), args.chunk_size):
        chunk = pending[start : start + args.chunk_size]
        histories: dict[str, list[dict[str, Any]]] = {scan: [] for scan in chunk}
        turn_image_counts: dict[str, list[int]] = {scan: [] for scan in chunk}
        persistent_states: dict[str, str] = {scan: "{}" for scan in chunk}
        images: dict[str, list[Any]] = defaultdict(list)
        rounds = max(len(grouped[scan]["user_message"]) for scan in chunk)
        for index in range(rounds):
            active: list[str] = []
            round_paths: dict[str, list[str]] = {}
            round_images: dict[str, list[Any]] = {}
            for scan_id in chunk:
                turns = grouped[scan_id]["user_message"]
                if index >= len(turns):
                    continue
                turn = turns[index]
                paths = []
                for relative in turn["image_paths"]:
                    path = args.image_root / relative
                    if not path.is_file():
                        raise SystemExit(f"missing observation: {path}")
                    paths.append(str(path))
                    images[scan_id].append(Image.open(path).convert("RGB"))
                active.append(scan_id)
                round_paths[scan_id] = paths
                round_images[scan_id] = images[scan_id][-len(paths) :]
            if not active:
                continue
            if (
                args.persistent_state_writer
                or args.compact_state_writer
                or args.append_only_state_writer
            ):
                writer_prompts = []
                for scan_id in active:
                    writer_instruction = (
                        "You are a query-independent persistent spatial-state writer. "
                        f"Previous state: {persistent_states[scan_id]}\n"
                        "Update the state using only the following new observation "
                        "chunk. Do not answer or anticipate any question. Preserve "
                        "camera pose changes, persistent entities and relations, "
                        "visible counts, temporal order, confidence, and invalidated "
                        "facts. Return one compact JSON object."
                    )
                    if args.compact_state_writer:
                        writer_instruction = (
                            "You are a query-independent persistent spatial-state writer. "
                            f"Previous state: {persistent_states[scan_id]}\n"
                            f"Update it from observation chunk {index + 1} only. Return one "
                            "valid single-line JSON object with this bounded schema: "
                            '{"turn":int,"camera":"short delta","entities":'
                            '["type|count|last_turn|relation"],"events":["short"],'
                            '"topology":["short"],"invalidated":["short"]}. '
                            "Keep at most 12 total list entries, retain old facts unless "
                            "contradicted, use no prose, and never answer or anticipate a question."
                        )
                    if args.append_only_state_writer:
                        writer_instruction = (
                            "You are a query-independent spatial event writer. Use only "
                            f"observation chunk {index + 1}; do not read, answer, or "
                            "anticipate any question. Return one valid single-line JSON "
                            "delta with schema "
                            '{"turn":int,"camera_delta":"short","seen":["entity|relation"],'
                            '"counts":["entity=count"],"events":["short"]}. '
                            "Keep at most 10 total list entries, state only visible or "
                            "geometrically supported facts, and use no prose."
                        )
                    content: list[dict[str, Any]] = [
                        {
                            "type": "text",
                            "text": writer_instruction,
                        }
                    ]
                    content.extend(
                        {"type": "image", "image": path}
                        for path in round_paths[scan_id]
                    )
                    writer_text = processor.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=False,
                        add_generation_prompt=True,
                        add_vision_id=True,
                    )
                    writer_prompts.append(
                        {
                            "prompt": writer_text,
                            "multi_modal_data": {"image": list(round_images[scan_id])},
                        }
                    )
                writer_outputs = llm.generate(
                    writer_prompts, writer_sampling, use_tqdm=False
                )
                for scan_id, output in zip(active, writer_outputs, strict=True):
                    state = output.outputs[0].text
                    if args.append_only_state_writer:
                        persistent_states[scan_id] = (
                            f"{persistent_states[scan_id]}\nchunk_{index + 1}:{state}"
                        ).strip()
                        grouped[scan_id]["user_message"][index]["state_delta"] = state
                    else:
                        persistent_states[scan_id] = state
                    grouped[scan_id]["user_message"][index]["persistent_state"] = persistent_states[scan_id]

            prompts = []
            independent_prompts = []
            independent_alternative_text: dict[str, str] = {}
            round_full_candidates: dict[str, str] = {}
            round_candidate_pairs: dict[str, tuple[str, str]] = {}
            for scan_id in active:
                turn = grouped[scan_id]["user_message"][index]
                paths = round_paths[scan_id]
                prompt = turn["prompt"]
                if candidate_a is not None and candidate_b is not None:
                    key = (scan_id, int(turn["turn_id"]))
                    if key not in candidate_a or key not in candidate_b:
                        raise RuntimeError(f"candidate coverage miss: {key}")
                    full_candidate = candidate_a[key]
                    left, right = full_candidate, candidate_b[key]
                    swapped = int(
                        hashlib.sha256(f"ost-verifier-v1|{scan_id}|{turn['turn_id']}".encode()).hexdigest(),
                        16,
                    ) % 2 == 1
                    swapped = bool(swapped) ^ bool(args.candidate_order_flip)
                    if swapped:
                        left, right = right, left
                    display_left, display_right = left, right
                    if args.candidate_answer_only:
                        from benchmark.ost.score import process_answer

                        display_left = process_answer(left)
                        display_right = process_answer(right)
                    verifier_instruction = (
                        "Return exactly one character: Y or N. Do not rewrite the answer.\n"
                        if args.candidate_independent_verification
                        else "Return exactly one character: A or B. Do not rewrite the answer.\n"
                        if args.candidate_choice_only
                        else "Return only the required JSON object with answer and reason.\n"
                    )
                    operator_instruction = ""
                    if args.operator_grounded_candidate_choice:
                        visible_question = str(
                            turn.get("origin_question") or turn.get("prompt") or ""
                        )
                        plan = compile_spatial_query(visible_question)
                        operator_instruction = (
                            "The visible question compiles to the generic spatial "
                            f"operator `{plan.operator}` over state views "
                            f"{', '.join(plan.state_views)}. Execute that operator "
                            "with the stated temporal scope and reference frame. "
                            "Select the candidate whose relation is consistent with "
                            "the causal visual support; do not choose by wording or "
                            "candidate position.\n"
                        )
                        turn["candidate_operator"] = plan.operator
                        turn["candidate_operator_state_views"] = list(plan.state_views)
                        turn["candidate_operator_temporal_scope"] = plan.temporal_scope
                    if args.candidate_independent_verification:
                        prefix = (
                            f"{prompt}\n\nAssess one candidate independently against all "
                            "causal images and conversation history. "
                            f"{operator_instruction}{verifier_instruction}Candidate: "
                        )
                        prompt = f"{prefix}{display_left}"
                        independent_alternative_text[scan_id] = f"{prefix}{display_right}"
                        round_full_candidates[scan_id] = full_candidate
                    else:
                        prompt = (
                            f"{prompt}\n\nTwo independent candidate answers follow. Verify both "
                            "against all causal images and conversation history. Select, correct, "
                            "or recompute the answer without preferring either position. "
                            f"{operator_instruction}{verifier_instruction}"
                            f"Candidate A: {display_left}\nCandidate B: {display_right}"
                        )
                    round_candidate_pairs[scan_id] = (left, right)
                    turn["candidate_order_swapped"] = swapped
                if args.spatial_ledger_prompt:
                    prompt = (
                        f"{prompt}\n\n"
                        "Before answering, update a compact causal spatial ledger from "
                        "the observations and conversation available so far: camera "
                        "position/orientation changes, persistent object relations, "
                        "visible counts, and relevant timestamps. Resolve egocentric "
                        "versus allocentric reference frames and perform any needed "
                        "numeric comparison before returning the required JSON answer."
                    )
                if args.classical_trajectory:
                    timeline = timelines.setdefault(
                        scan_id, ObservationChunkPoseTimeline(scan_id)
                    )
                    new_images = images[scan_id][-len(paths) :]
                    timeline.ingest_chunk(int(turn["turn_id"]), new_images)
                    packet = timeline.retrieve_qualitative_packet(prompt)
                    if packet:
                        prompt = f"{prompt}\n\n{packet}"
                        turn["classical_trajectory_applied"] = True
                if (
                    args.persistent_state_writer
                    or args.compact_state_writer
                    or args.append_only_state_writer
                ):
                    prompt = (
                        f"{prompt}\n\nPersistent causal spatial state (written without "
                        f"access to this question):\n{persistent_states[scan_id]}"
                    )
                    if args.writer_arbitration:
                        prompt += (
                            "\nUse this state as a memory aid for persistent existence, "
                            "co-occurrence, and temporal facts. For direction, distance, "
                            "camera pose, or reference-frame geometry, recompute from the "
                            "raw causal images and conversation; never trust approximate "
                            "coordinates in the state when they conflict with visual evidence."
                        )
                histories[scan_id].append(
                    _user_turn(
                        prompt, paths, grouped[scan_id]["system_prompt"]
                    )
                )
                turn_image_counts[scan_id].append(len(paths))
                support_history, support_images, selected_turns = select_causal_turn_support(
                    histories[scan_id],
                    images[scan_id],
                    turn_image_counts[scan_id],
                    view=args.causal_support_view,
                    key=scan_id,
                )
                turn["causal_support_view"] = args.causal_support_view
                turn["causal_support_selected_turns"] = [
                    value + 1 for value in selected_turns
                ]
                turn["causal_support_total_turns"] = index + 1
                text = processor.apply_chat_template(
                    support_history,
                    tokenize=False,
                    add_generation_prompt=True,
                    add_vision_id=True,
                )
                prompts.append(
                    {
                        "prompt": text,
                        "multi_modal_data": {"image": list(support_images)},
                    }
                )
                if args.candidate_independent_verification:
                    alternate_history = list(support_history)
                    alternate_history[-1] = _user_turn(
                        independent_alternative_text[scan_id],
                        paths,
                        grouped[scan_id]["system_prompt"],
                    )
                    alternate_text = processor.apply_chat_template(
                        alternate_history,
                        tokenize=False,
                        add_generation_prompt=True,
                        add_vision_id=True,
                    )
                    independent_prompts.append(
                        {
                            "prompt": alternate_text,
                            "multi_modal_data": {"image": list(support_images)},
                        }
                    )
            active_sampling = (
                binary_sampling if args.candidate_independent_verification else sampling
            )
            outputs = llm.generate(prompts, active_sampling, use_tqdm=False)
            independent_outputs = (
                llm.generate(independent_prompts, binary_sampling, use_tqdm=False)
                if args.candidate_independent_verification
                else None
            )
            initial_responses: dict[str, str] = {}
            for output_index, (scan_id, output) in enumerate(
                zip(active, outputs, strict=True)
            ):
                candidates = [completion.text for completion in output.outputs]
                initial_responses[scan_id] = _consensus_response(candidates)
                if args.proposal_logprobs:
                    if len(output.outputs) != 1:
                        raise RuntimeError(
                            "proposal log-prob requires one deterministic completion"
                        )
                    completion = output.outputs[0]
                    token_count = len(completion.token_ids)
                    cumulative = completion.cumulative_logprob
                    if token_count < 1 or cumulative is None:
                        raise RuntimeError("vLLM did not return proposal log-prob")
                    turn = grouped[scan_id]["user_message"][index]
                    turn["response_cumulative_logprob"] = float(cumulative)
                    turn["response_token_count"] = token_count
                    turn["response_mean_logprob"] = float(cumulative) / token_count
                truncated += sum(
                    completion.finish_reason == "length"
                    for completion in output.outputs
                )
                if args.candidate_independent_verification:
                    assert independent_outputs is not None
                    raw_left = initial_responses[scan_id].strip().upper()
                    raw_right = independent_outputs[output_index].outputs[0].text.strip().upper()
                    left_yes = raw_left.startswith("Y")
                    right_yes = raw_right.startswith("Y")
                    if left_yes != right_yes:
                        selected = round_candidate_pairs[scan_id][0 if left_yes else 1]
                    else:
                        selected = round_full_candidates[scan_id]
                    initial_responses[scan_id] = selected
                    turn = grouped[scan_id]["user_message"][index]
                    turn["candidate_independent_left_raw"] = raw_left
                    turn["candidate_independent_right_raw"] = raw_right
                    turn["candidate_independent_decisive"] = left_yes != right_yes
                elif args.candidate_choice_only:
                    raw_choice = initial_responses[scan_id].strip().upper()
                    choice = "B" if re.search(r"(?:^|\b)B(?:\b|$)", raw_choice) else "A"
                    initial_responses[scan_id] = round_candidate_pairs[scan_id][
                        1 if choice == "B" else 0
                    ]
                    grouped[scan_id]["user_message"][index]["candidate_choice_raw"] = raw_choice
                    grouped[scan_id]["user_message"][index]["candidate_choice"] = choice
            final_responses = dict(initial_responses)
            if args.self_refine:
                refine_prompts = []
                for scan_id in active:
                    verify_history = list(histories[scan_id]) + [
                        {"role": "assistant", "content": initial_responses[scan_id]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Verify the previous answer against all causal images "
                                        "and conversation history available so far. Correct any "
                                        "spatial, temporal, counting, or reference-frame mistake. "
                                        "Do not assume future observations. Return only the required "
                                        "JSON object with answer and reason."
                                    ),
                                }
                            ],
                        },
                    ]
                    text = processor.apply_chat_template(
                        verify_history,
                        tokenize=False,
                        add_generation_prompt=True,
                        add_vision_id=True,
                    )
                    refine_prompts.append(
                        {
                            "prompt": text,
                            "multi_modal_data": {"image": list(images[scan_id])},
                        }
                    )
                refined = llm.generate(refine_prompts, sampling, use_tqdm=False)
                for scan_id, output in zip(active, refined, strict=True):
                    candidates = [completion.text for completion in output.outputs]
                    final_responses[scan_id] = _consensus_response(candidates)
                    truncated += sum(
                        completion.finish_reason == "length"
                        for completion in output.outputs
                    )
            for scan_id in active:
                response = final_responses[scan_id]
                if args.self_refine:
                    grouped[scan_id]["user_message"][index]["initial_response"] = initial_responses[scan_id]
                grouped[scan_id]["user_message"][index]["response"] = response
                histories[scan_id].append(
                    {"role": "assistant", "content": response}
                )
        for scan_id in chunk:
            for image in images.pop(scan_id, []):
                image.close()
            record = dict(grouped[scan_id])
            record["inference_format"] = FORMAT
            record["classical_trajectory"] = bool(args.classical_trajectory)
            record["self_refine"] = bool(args.self_refine)
            record["spatial_ledger_prompt"] = bool(args.spatial_ledger_prompt)
            record["persistent_state_writer"] = bool(args.persistent_state_writer)
            record["compact_state_writer"] = bool(args.compact_state_writer)
            record["append_only_state_writer"] = bool(args.append_only_state_writer)
            record["writer_max_tokens"] = writer_max_tokens
            record["writer_arbitration"] = bool(args.writer_arbitration)
            record["self_consistency_samples"] = args.self_consistency_samples
            record["self_consistency_temperature"] = (
                args.self_consistency_temperature
                if args.self_consistency_samples > 1
                else 0.0
            )
            record["candidate_verifier"] = bool(candidate_a is not None)
            record["candidate_choice_only"] = bool(args.candidate_choice_only)
            record["operator_grounded_candidate_choice"] = bool(
                args.operator_grounded_candidate_choice
            )
            record["candidate_answer_only"] = bool(args.candidate_answer_only)
            record["candidate_order_flip"] = bool(args.candidate_order_flip)
            record["candidate_independent_verification"] = bool(
                args.candidate_independent_verification
            )
            record["causal_support_view"] = args.causal_support_view
            record["proposal_logprobs"] = bool(args.proposal_logprobs)
            destination = args.output / f"{scan_id}.json"
            temporary = destination.with_suffix(".json.partial")
            temporary.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        print(
            json.dumps(
                {
                    "completed_scans": min(start + args.chunk_size, len(pending)),
                    "pending_scans": len(pending),
                    "length_truncated_turns": truncated,
                }
            ),
            flush=True,
        )
    print(json.dumps({"shard_complete": True, "length_truncated_turns": truncated}))


if __name__ == "__main__":
    main()
