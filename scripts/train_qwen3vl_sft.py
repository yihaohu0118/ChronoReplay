#!/usr/bin/env python3
"""Full-parameter Qwen3-VL SFT over audited multimodal conversations.

The Spatial-Witness release uses the ordinary assistant-only cross-entropy
path with unit sample weights. Extra research hooks remain disabled unless an
explicit non-release launcher opts into them.
"""

from __future__ import annotations

import argparse
import collections
from functools import partial
import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import seed_worker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.sft.media import (
    MultimodalSFTDataset,
    encode_processor_item,
    find_subsequence,
)
from synthesis.sft.evidence_orbit import (
    certified_feature_evidence_orbit_role,
    certified_frontier_representation_view,
    certified_nested_evidence_view,
    correct_option_identity,
)
from data_pipeline.processor_contract import configure_processor_pixel_budget


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optimizer_index(index: Any) -> tuple[int, bool]:
    """Decode a real row index or a zero-loss optimizer padding reference."""

    if isinstance(index, tuple):
        if len(index) != 2 or index[1] is not True:
            raise RuntimeError(f"invalid optimizer padding index: {index!r}")
        return int(index[0]), True
    return int(index), False


class SpatialSFTDataset(MultimodalSFTDataset):
    """Expose optional batching metadata while keeping it out of the prompt."""

    def __getitem__(self, index: Any) -> Dict[str, Any]:
        row_index, is_optimizer_padding = optimizer_index(index)
        feature = super().__getitem__(row_index)
        row = self.rows[row_index]
        if is_optimizer_padding:
            # ZeRO-3 requires every rank to execute the same parameterized
            # image/video branches. Padding references align those branches
            # without adding supervision or changing any real row's mass.
            feature["sample_loss_weight"] = 0.0
            feature["optimizer_padding"] = True
        feature.update(
            {
                "pair_id": row.get("pair_id"),
                "component": row.get("component"),
                "gradient_route_family": (
                    row.get("joint_family") or row.get("workfirst_family")
                ),
                "action_types": row.get("action_types"),
                # Pure-VLM evidence-orbit metadata is optimizer-side only. It
                # groups independently serialized dense/context views
                # of one source question without adding any actor-visible tag.
                "semantic_source_sha256": row.get("semantic_source_sha256"),
                "source_view": row.get("source_view"),
                "target_contract": row.get("target_contract"),
                "correct_option_identity_sha256": correct_option_identity(row),
                "certified_evidence_orbit_view": (
                    certified_nested_evidence_view(row)
                    or certified_frontier_representation_view(row)
                ),
                # Intervention recovery rows keep a rejected assistant action
                # in the causal context, but supervise only the verified
                # repair.  This row metadata never enters the actor prompt.
                "supervised_assistant_ordinals": row.get(
                    "supervised_assistant_ordinals"
                ),
                # Optional optimizer-only weights allocate a multi-turn
                # sample's fixed loss mass across its assistant turns. They
                # never enter the prompt or alter the causal transcript.
                "assistant_turn_loss_weights": row.get(
                    "assistant_turn_loss_weights"
                ),
            }
        )
        return feature


EVIDENCE_ORBIT_ROLE = {
    "evidence_dense": 0,
    "evidence_context": 1,
    "dense": 0,
    "context": 1,
    "long_context_semantic_aux": 2,
}
OPTION_MARKER = re.compile(r"(?:^|\n)\s*([A-H])\.\s+", re.MULTILINE)


def nonempty_string(value: Any) -> str:
    """Return a stripped string while treating heterogeneous Parquet NaN as null."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def evidence_orbit_role(row: Mapping[str, Any]) -> int:
    """Map one direct-VLM view to its source-orbit role, or ``-1``.

    Component names are deliberately ignored: the role is defined only by the
    generic evidence view and target contract, never by a benchmark task.
    """

    source_view = nonempty_string(row.get("source_view"))
    role = EVIDENCE_ORBIT_ROLE.get(source_view, -1)
    contract = nonempty_string(row.get("target_contract"))
    if role in {0, 1} and contract != "letter_only":
        return -1
    if role == 2 and contract != "answer_first_selected_option":
        return -1
    return role


def visible_option_count(messages: Any) -> int:
    """Recover the contiguous A--H option cardinality from the user prompt."""

    texts: List[str] = []
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        for part in message.get("content", []):
            if isinstance(part, Mapping) and part.get("type") == "text":
                texts.append(str(part.get("text") or ""))
    labels = OPTION_MARKER.findall("\n".join(texts))
    expected = [chr(ord("A") + index) for index in range(len(labels))]
    if not 2 <= len(labels) <= 8 or labels != expected:
        raise RuntimeError(f"non-contiguous direct-VLM options: {labels}")
    return len(labels)


def canonical_letter_token_ids(tokenizer: Any) -> List[int]:
    """Resolve the one-token direct-answer alphabet used by the MCQ contract."""

    output: List[int] = []
    for letter in "ABCDEFGH":
        encoded = tokenizer(letter, add_special_tokens=False)
        ids = list(encoded.get("input_ids") or [])
        if len(ids) != 1:
            raise RuntimeError(f"option letter is not one token: {letter} -> {ids}")
        output.append(int(ids[0]))
    if len(set(output)) != len(output):
        raise RuntimeError(f"option letter token ids are not unique: {output}")
    return output

ACTION_FAMILY = {
    "stream_observe": "observe",
    # The accepted 52.38 recipe down-weighted only the cheap prefix overview;
    # keep it distinct from evidence-bearing inspect/stream observations.
    "observe_prefix": "observe_prefix",
    "inspect_interval": "observe",
    "inspect_window": "observe",
    "compare_moments": "observe",
    "trace_motion": "observe",
    "bind_relation": "evidence",
    "bind_landmark": "evidence",
    "log_room_visit": "evidence",
    "patch_map": "evidence",
    "simulate_step": "evidence",
    "align_trajectory": "evidence",
    "update_hypotheses": "evidence",
    "write_evidence": "evidence",
    "revise_evidence": "evidence",
    "recall_buffer": "memory",
    "recall_evidence": "memory",
    "answer": "answer",
    "answer_mcq": "answer",
}


class StopAfterStepCallback(TrainerCallback):
    """Stop cleanly at an intermediate step without changing the LR horizon.

    ``max_steps`` remains the original scheduler horizon. This diagnostic
    callback only requests a final save and graceful stop after the requested
    global step; its artifact is not a release candidate.
    """

    def __init__(
        self,
        stop_after_step: int,
        force_save_every_steps: int = 0,
        *,
        save_on_stop: bool = True,
    ) -> None:
        self.stop_after_step = int(stop_after_step)
        self.force_save_every_steps = int(force_save_every_steps)
        self.save_on_stop = bool(save_on_stop)

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        if (
            self.force_save_every_steps > 0
            and int(state.global_step) > 0
            and int(state.global_step) % self.force_save_every_steps == 0
        ):
            control.should_save = True
        if self.stop_after_step > 0 and int(state.global_step) >= self.stop_after_step:
            if self.save_on_stop:
                control.should_save = True
            control.should_training_stop = True
        return control


def action_family(text: str) -> str:
    stripped = text.strip()
    if len(stripped) == 1 and stripped.upper() in "ABCDEFGH":
        # Keep deployment-format letters separate from optional structured
        # targets in heterogeneous research corpora.
        return "deployment_answer"
    try:
        value = json.loads(stripped)
    except (TypeError, json.JSONDecodeError):
        return "other"
    if not isinstance(value, Mapping):
        return "other"
    action = value.get("action", value.get("type", ""))
    if isinstance(action, Mapping):
        action = action.get("type", action.get("action", ""))
    return ACTION_FAMILY.get(str(action), "other")


def _list_value(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def optional_text(value: Any) -> str:
    """Return blank for Arrow/Pandas null scalars in heterogeneous pools."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def optional_count(value: Any) -> int:
    """Normalize an optional Parquet numeric scalar used only for auditing."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    count = int(value)
    if count < 0:
        raise RuntimeError(f"count metadata must be non-negative, got {value!r}")
    return count


def direct_length_proxy(row: Mapping[str, Any]) -> int:
    """Estimate multimodal context cost without reading labels or task metadata."""

    actor_frames = optional_count(row.get("num_actor_frames"))
    images = optional_count(row.get("num_images"))
    turns = optional_count(row.get("num_turns"))
    messages = row.get("messages")
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    text_bytes = len(json.dumps(messages or [], default=str).encode("utf-8"))
    # Coefficients only establish a monotonic bucket order. The actual Qwen
    # processor remains the fail-closed authority for the 8192-token limit.
    return actor_frames * 64 + images * 256 + turns * 128 + text_bytes // 4


def decision_role(row: Mapping[str, Any]) -> int:
    """Return 0 for insufficient/observe, 1 for sufficient/answer, else -1."""

    actions = [str(value) for value in _list_value(row.get("action_types"))]
    explicit_role = optional_text(row.get("paired_decision_role"))
    if explicit_role:
        expected = {
            "insufficient": ("stream_observe", 0),
            "sufficient": ("answer_mcq", 1),
        }.get(explicit_role)
        if expected is None or actions != [expected[0]]:
            raise RuntimeError(
                "explicit paired-decision role/action mismatch: "
                f"role={explicit_role!r} actions={actions} uid={row.get('uid')}"
            )
        return expected[1]
    component = str(row.get("component") or "")
    decision_component = (
        component.endswith("_insufficient_decision")
        or component.endswith("_sufficient_decision")
        or component.endswith("_strict_insufficient")
        or component.endswith("_strict_sufficient")
        or component
        in {
            "v43_neighbor_insufficient_policy",
            "v43_neighbor_sufficient_policy",
        }
    )
    if len(actions) != 1 or not decision_component:
        return -1
    if actions[0] == "stream_observe":
        return 0
    if actions[0] == "answer_mcq":
        return 1
    return -1


EvidenceProbeSpec = bool


def evidence_status_probe_spec(tokenizer: Any) -> EvidenceProbeSpec:
    """Verify that the tokenizer exposes offsets for the commitment field.

    The release canonicalizer sorts JSON keys, so current action targets place
    ``action`` before ``evidence_status``.  Older actor traces sometimes put
    the self-audit field first.  JSON object order is semantically irrelevant;
    offsets let the paired objective locate the named field in either position
    without relying on a tokenized prefix (whose first token can merge with a
    preceding comma or brace).
    """

    encoded = tokenizer(
        '{"evidence_status":"insufficient"}',
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if not offsets or len(offsets) != len(encoded.get("input_ids") or []):
        raise RuntimeError("tokenizer does not expose evidence-status offsets")
    return True


def assistant_labels_and_weights(
    tokenizer: Any,
    input_ids: torch.Tensor,
    messages: List[Dict[str, Any]],
    family_weights: Mapping[str, float],
    *,
    normalize_by_turn: bool,
    evidence_probe_spec: EvidenceProbeSpec | None = None,
    supervised_assistant_ordinals: set[int] | None = None,
    assistant_turn_loss_weights: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, int], List[tuple[int, int, int]]]:
    labels = torch.full_like(input_ids, -100)
    weights = torch.zeros_like(input_ids, dtype=torch.float32)
    ids = input_ids.tolist()
    cursor = 0
    counts: Dict[str, int] = {}
    evidence_probes: List[tuple[int, int, int]] = []
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    marker = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    if not marker or im_end_id is None:
        raise RuntimeError("Qwen chat-template assistant delimiters are unavailable")
    assistant_ordinal = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        text_items = [
            item
            for item in message.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if len(text_items) != 1:
            raise RuntimeError(
                "every assistant turn must contain exactly one text action"
            )
        supervise = (
            supervised_assistant_ordinals is None
            or assistant_ordinal in supervised_assistant_ordinals
        )
        for item in text_items:
            text = str(item.get("text", ""))
            marker_start = find_subsequence(ids, marker, cursor)
            if marker_start < 0:
                raise RuntimeError(
                    f"assistant marker alignment failed at ordinal {assistant_ordinal}"
                )
            start = marker_start + len(marker)
            try:
                end = ids.index(im_end_id, start)
            except ValueError:
                raise RuntimeError(
                    f"assistant end alignment failed at ordinal {assistant_ordinal}"
                )
            stop = min(end + 1, labels.numel())
            # The role marker makes this span unambiguous even for one-token
            # anchor answers such as "C", which often also occur in user text.
            decoded = tokenizer.decode(ids[start:end], skip_special_tokens=False).strip()
            if decoded != text.strip():
                raise RuntimeError(
                    f"assistant content alignment failed at ordinal {assistant_ordinal}"
                )
            cursor = stop
            if not supervise:
                continue
            family = action_family(text)
            active_tokens = max(stop - start, 1)
            turn_weight = float(family_weights.get(family, family_weights["other"]))
            if assistant_turn_loss_weights is not None:
                if assistant_ordinal >= len(assistant_turn_loss_weights):
                    raise RuntimeError(
                        "assistant turn loss weights are shorter than the transcript"
                    )
                turn_weight *= assistant_turn_loss_weights[assistant_ordinal]
            token_weight = turn_weight / active_tokens if normalize_by_turn else turn_weight
            labels[start:stop] = input_ids[start:stop]
            weights[start:stop] = token_weight
            counts[family] = counts.get(family, 0) + 1
            if evidence_probe_spec is not None:
                try:
                    payload = json.loads(text)
                except (TypeError, json.JSONDecodeError):
                    payload = None
                if isinstance(payload, Mapping) and payload.get("evidence_status") in {
                    "insufficient",
                    "sufficient",
                }:
                    status = str(payload["evidence_status"])
                    target_ids = ids[start:end]
                    matches = list(
                        re.finditer(
                            r'"evidence_status":"(insufficient|sufficient)"',
                            text,
                        )
                    )
                    if len(matches) != 1 or matches[0].group(1) != status:
                        raise RuntimeError(
                            "evidence-status target must contain exactly one strict "
                            f"compact-JSON commitment field: {text}"
                        )
                    encoded = tokenizer(
                        text,
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                    )
                    encoded_ids = list(encoded.get("input_ids") or [])
                    offsets = list(encoded.get("offset_mapping") or [])
                    token_offset = find_subsequence(target_ids, encoded_ids, 0)
                    if token_offset < 0 or len(offsets) != len(encoded_ids):
                        raise RuntimeError(
                            "evidence-status offset tokenization does not align with "
                            f"the assistant span: {text}"
                        )
                    value_start, value_end = matches[0].span(1)
                    value_tokens = [
                        index
                        for index, (left, right) in enumerate(offsets)
                        if int(left) < value_end and int(right) > value_start
                    ]
                    if not value_tokens:
                        raise RuntimeError(
                            f"evidence-status value has no aligned token: {text}"
                        )
                    local_position = value_tokens[0]
                    if token_offset + local_position <= 0:
                        raise RuntimeError(
                            "evidence-status token has no causal predecessor logit"
                        )
                    role = 0 if status == "insufficient" else 1
                    # logits[position] predicts the first token overlapping the
                    # status value.  Record its actual id, because byte-level
                    # tokenization can merge the opening quote with that token.
                    evidence_probes.append(
                        (
                            start + token_offset + local_position - 1,
                            role,
                            int(encoded_ids[local_position]),
                        )
                    )
        assistant_ordinal += 1
    if supervised_assistant_ordinals is not None:
        valid = set(range(assistant_ordinal))
        if (
            not supervised_assistant_ordinals
            or not supervised_assistant_ordinals.issubset(valid)
        ):
            raise RuntimeError(
                "supervised assistant ordinals are empty or outside the transcript: "
                f"requested={sorted(supervised_assistant_ordinals)} "
                f"assistant_turns={assistant_ordinal}"
            )
    if (
        assistant_turn_loss_weights is not None
        and len(assistant_turn_loss_weights) != assistant_ordinal
    ):
        raise RuntimeError(
            "assistant turn loss weight count does not match the transcript: "
            f"weights={len(assistant_turn_loss_weights)} "
            f"assistant_turns={assistant_ordinal}"
        )
    return labels, weights, counts, evidence_probes


def supervised_ordinals(value: Any) -> set[int] | None:
    """Normalize an optional Arrow/list mask without interpreting NaN as one."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    values = _list_value(value)
    if not values:
        raise RuntimeError("supervised_assistant_ordinals must be non-empty")
    ordinals = {int(item) for item in values}
    if len(ordinals) != len(values) or any(item < 0 for item in ordinals):
        raise RuntimeError(
            "supervised_assistant_ordinals must be unique non-negative integers"
        )
    return ordinals


def assistant_turn_weights(value: Any) -> list[float] | None:
    """Normalize optional positive per-assistant optimizer-only weights."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    values = _list_value(value)
    if not values:
        raise RuntimeError("assistant_turn_loss_weights must be non-empty")
    weights = [float(item) for item in values]
    if any(not math.isfinite(item) or item <= 0.0 for item in weights):
        raise RuntimeError(
            "assistant_turn_loss_weights must be finite positive scalars"
        )
    return weights


@dataclass
class TransitionCollator:
    processor: Any
    max_length: int
    family_weights: Dict[str, float]
    normalize_by_turn: bool = True
    evidence_pair_margin_enabled: bool = False
    evidence_orbit_enabled: bool = False
    causal_gradient_firewall_enabled: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded = []
        labels_list = []
        weights_list = []
        probes_list: List[List[tuple[int, int, int]]] = []
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        option_letter_token_ids = (
            canonical_letter_token_ids(tokenizer)
            if self.evidence_orbit_enabled
            else []
        )
        probe_spec = (
            evidence_status_probe_spec(tokenizer)
            if self.evidence_pair_margin_enabled
            else None
        )
        for feature in features:
            messages = feature["messages"]
            supervised = supervised_ordinals(
                feature.get("supervised_assistant_ordinals")
            )
            per_turn_weights = assistant_turn_weights(
                feature.get("assistant_turn_loss_weights")
            )
            item = encode_processor_item(self.processor, messages)
            length = item["input_ids"].numel()
            if length > self.max_length:
                raise RuntimeError(
                    f"train-ready contract violated: uid={feature['uid']} "
                    f"tokens={length} max_length={self.max_length}"
                )
            labels, weights, counts, probes = assistant_labels_and_weights(
                tokenizer,
                item["input_ids"],
                messages,
                self.family_weights,
                normalize_by_turn=self.normalize_by_turn,
                evidence_probe_spec=probe_spec,
                supervised_assistant_ordinals=supervised,
                assistant_turn_loss_weights=per_turn_weights,
            )
            expected = (
                len(supervised)
                if supervised is not None
                else sum(
                    1
                    for message in messages
                    if message.get("role") == "assistant"
                    for part in message.get("content", [])
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            )
            if not torch.any(labels != -100) or sum(counts.values()) != expected:
                raise RuntimeError(
                    f"assistant span alignment failed: uid={feature['uid']} "
                    f"expected={expected} aligned={sum(counts.values())}"
                )
            encoded.append(item)
            labels_list.append(labels)
            weights_list.append(weights)
            probes_list.append(probes)

        pad_id = tokenizer.pad_token_id
        max_len = max(item["input_ids"].numel() for item in encoded)
        batch: Dict[str, Any] = {key: [] for key in ("input_ids", "attention_mask", "labels", "loss_weights")}
        pixel_values = []
        image_grid_thw = []
        pixel_values_videos = []
        video_grid_thw = []
        for item, labels, weights in zip(encoded, labels_list, weights_list):
            pad = max_len - item["input_ids"].numel()
            batch["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, pad), value=pad_id))
            batch["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0))
            batch["labels"].append(torch.nn.functional.pad(labels, (0, pad), value=-100))
            batch["loss_weights"].append(torch.nn.functional.pad(weights, (0, pad), value=0.0))
            if "pixel_values" in item:
                pixel_values.append(item["pixel_values"])
            if "image_grid_thw" in item:
                image_grid_thw.append(item["image_grid_thw"])
            if "pixel_values_videos" in item:
                pixel_values_videos.append(item["pixel_values_videos"])
            if "video_grid_thw" in item:
                video_grid_thw.append(item["video_grid_thw"])
        out = {key: torch.stack(value) for key, value in batch.items()}
        out["sample_loss_weights"] = torch.tensor(
            [float(feature["sample_loss_weight"]) for feature in features],
            dtype=torch.float32,
        )
        if pixel_values:
            out["pixel_values"] = torch.cat(pixel_values, dim=0)
        if image_grid_thw:
            out["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
        if pixel_values_videos:
            out["pixel_values_videos"] = torch.cat(pixel_values_videos, dim=0)
        if video_grid_thw:
            out["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)
        if self.evidence_pair_margin_enabled:
            roles = [decision_role(feature) for feature in features]
            pair_ids = [str(feature.get("pair_id") or "") for feature in features]
            pair_valid = bool(
                len(features) == 2
                and pair_ids[0]
                and pair_ids[0] == pair_ids[1]
                and sorted(roles) == [0, 1]
            )
            positions: List[int] = []
            status_tokens: Dict[int, int] = {}
            for feature, role, probes in zip(features, roles, probes_list):
                if role >= 0:
                    if len(probes) != 1 or probes[0][1] != role:
                        raise RuntimeError(
                            "counterfactual decision row lacks one matching canonical "
                            f"evidence-status probe: uid={feature.get('uid')} role={role} "
                            f"probes={probes}"
                        )
                    positions.append(int(probes[0][0]))
                    status_tokens[role] = int(probes[0][2])
                else:
                    positions.append(-1)
            if probe_spec is None or (pair_valid and set(status_tokens) != {0, 1}):
                raise RuntimeError("missing evidence-status probe specification")
            out["evidence_pair_roles"] = torch.tensor(roles, dtype=torch.long)
            out["evidence_pair_positions"] = torch.tensor(positions, dtype=torch.long)
            out["evidence_pair_valid"] = torch.tensor(
                [int(pair_valid)], dtype=torch.long
            )
            out["evidence_status_token_ids"] = torch.tensor(
                [status_tokens.get(0, -1), status_tokens.get(1, -1)],
                dtype=torch.long,
            )
        if self.evidence_orbit_enabled:
            roles = [
                certified_feature_evidence_orbit_role(feature)
                for feature in features
            ]
            group_ids = [
                nonempty_string(feature.get("semantic_source_sha256"))
                for feature in features
            ]
            option_identities = [
                nonempty_string(feature.get("correct_option_identity_sha256"))
                for feature in features
            ]
            direct_option_identities = [
                identity
                for identity, role in zip(option_identities, roles)
                if role in {0, 1}
            ]
            orbit_valid = bool(
                len(features) == 2
                and group_ids[0]
                and len(set(group_ids)) == 1
                and len(direct_option_identities) == 2
                and direct_option_identities[0]
                and len(set(direct_option_identities)) == 1
                and sorted(roles) == [0, 1]
            )
            answer_positions: List[int] = []
            answer_token_ids: List[int] = []
            option_counts: List[int] = []
            direct_letter_ids = set(option_letter_token_ids)
            for feature, role, labels in zip(features, roles, labels_list):
                if role not in {0, 1}:
                    option_counts.append(0)
                    answer_positions.append(-1)
                    answer_token_ids.append(-1)
                    continue
                option_counts.append(visible_option_count(feature["messages"]))
                active = torch.nonzero(labels.ne(-100), as_tuple=False).flatten()
                # The canonical chat template may additionally supervise its
                # assistant end marker.  The first active token is still the
                # independently audited one-letter answer; ordinary CE keeps
                # supervising every later active token unchanged.
                if active.numel() < 1 or int(active[0]) <= 0:
                    raise RuntimeError(
                        "direct evidence-orbit row must start with one option "
                        f"letter: uid={feature.get('uid')} active={active.tolist()}"
                    )
                position = int(active[0])
                target_id = int(labels[position])
                if target_id not in direct_letter_ids:
                    raise RuntimeError(
                        "evidence-orbit target is outside the one-token option alphabet: "
                        f"uid={feature.get('uid')} token_id={target_id}"
                    )
                if target_id not in set(
                    option_letter_token_ids[: option_counts[-1]]
                ):
                    raise RuntimeError(
                        "evidence-orbit target is outside the visible option range: "
                        f"uid={feature.get('uid')} options={option_counts[-1]}"
                    )
                answer_positions.append(position - 1)
                answer_token_ids.append(target_id)
            out["evidence_orbit_roles"] = torch.tensor(roles, dtype=torch.long)
            out["evidence_orbit_valid"] = torch.tensor(
                [int(orbit_valid)], dtype=torch.long
            )
            out["evidence_orbit_answer_positions"] = torch.tensor(
                answer_positions, dtype=torch.long
            )
            out["evidence_orbit_answer_token_ids"] = torch.tensor(
                answer_token_ids, dtype=torch.long
            )
            out["evidence_orbit_option_counts"] = torch.tensor(
                option_counts, dtype=torch.long
            )
            out["evidence_orbit_letter_token_ids"] = torch.tensor(
                option_letter_token_ids, dtype=torch.long
            )
        if self.causal_gradient_firewall_enabled:
            route_names = [
                nonempty_string(feature.get("gradient_route_family"))
                for feature in features
            ]
            route_map = {"EXACTPACKET": 0, "OST": 1}
            if not route_names or any(name not in route_map for name in route_names):
                raise RuntimeError(
                    f"gradient-firewall batch has invalid routes: {route_names}"
                )
            route_ids = [route_map[name] for name in route_names]
            if len(set(route_ids)) != 1:
                raise RuntimeError(
                    f"gradient-firewall micro-batch mixes routes: {route_names}"
                )
            out["gradient_route_ids"] = torch.tensor(route_ids, dtype=torch.long)
        return out


def reduce_weighted_token_loss(
    token_loss: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    normalize_by_sample: bool,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce token CE with an explicit, testable sample-normalization mode."""
    active = labels.ne(-100)
    if sample_weights is None:
        sample_weights = torch.ones(
            token_loss.shape[0],
            dtype=token_loss.dtype,
            device=token_loss.device,
        )
    sample_weights = sample_weights.to(
        device=token_loss.device, dtype=token_loss.dtype
    )
    if (
        sample_weights.ndim != 1
        or sample_weights.numel() != token_loss.shape[0]
        or not torch.all(torch.isfinite(sample_weights))
        or torch.any(sample_weights < 0)
    ):
        raise RuntimeError(
            "sample_loss_weights must be finite non-negative scalars"
        )
    if normalize_by_sample:
        per_sample_denominator = (weights * active.to(weights.dtype)).sum(dim=-1)
        valid_samples = per_sample_denominator.gt(0)
        if not torch.any(valid_samples):
            raise RuntimeError("SFT batch contains no supervised assistant tokens")
        per_sample_numerator = (token_loss * weights).sum(dim=-1)
        per_sample_loss = (
            per_sample_numerator[valid_samples]
            / per_sample_denominator[valid_samples].clamp_min(1e-8)
        )
        # Keep this as an absolute multiplier, not a within-microbatch
        # normalized importance weight.  With per-device batch size one,
        # dividing by the sample-weight sum would cancel the weight before
        # gradient accumulation and make trajectory-mass conservation inert.
        positive_count = sample_weights[valid_samples].gt(0).sum().to(
            dtype=per_sample_loss.dtype
        )
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                positive_count, op=torch.distributed.ReduceOp.SUM
            )
            world_size = torch.distributed.get_world_size()
        if float(positive_count.item()) <= 0:
            raise RuntimeError("global SFT batch contains only optimizer padding")
        # DDP averages rank gradients. Dividing each local sum by the global
        # real-row count / world size therefore produces the exact global
        # mean even when a rank receives only zero-loss alignment rows.
        denominator = positive_count / float(world_size)
        return (
            per_sample_loss * sample_weights[valid_samples]
        ).sum() / denominator
    scaled_weights = weights * sample_weights.unsqueeze(-1)
    positive_tokens = active & sample_weights.unsqueeze(-1).gt(0)
    denominator = weights[positive_tokens].sum()
    if float(denominator.item()) <= 0:
        return (token_loss * scaled_weights).sum()
    return (token_loss * scaled_weights).sum() / denominator


def sparse_supervised_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute next-token CE only at supervised SFT target positions.

    Agent trajectories retain long user/environment contexts with label
    ``-100``. Converting the full ``batch x sequence x vocabulary`` tensor to
    fp32 before applying ``ignore_index`` wastes tens of GiB while contributing
    exactly zero loss. Selecting supervised positions first is mathematically
    identical and keeps fp32 CE restricted to target tokens.
    """

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise RuntimeError(
            f"invalid sparse CE shapes: logits={tuple(logits.shape)} "
            f"labels={tuple(labels.shape)}"
        )
    shifted_logits = logits[..., :-1, :]
    shifted_labels = labels[..., 1:].contiguous()
    supervised = shifted_labels.ne(-100)
    if not torch.any(supervised):
        raise RuntimeError("SFT micro-batch contains no supervised target token")
    selected_loss = torch.nn.functional.cross_entropy(
        shifted_logits[supervised].float(),
        shifted_labels[supervised],
        reduction="none",
    )
    token_loss = torch.zeros_like(shifted_labels, dtype=selected_loss.dtype)
    token_loss = token_loss.masked_scatter(supervised, selected_loss)
    return token_loss, shifted_labels


def supervised_predictor_positions(labels: torch.Tensor) -> torch.Tensor:
    """Return the sequence positions whose logits predict any supervised token."""

    if labels.ndim != 2 or labels.shape[1] < 2:
        raise RuntimeError(f"invalid SFT label shape: {tuple(labels.shape)}")
    positions = torch.nonzero(labels[:, 1:].ne(-100).any(dim=0), as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("SFT micro-batch contains no supervised target token")
    return positions


def sparse_selected_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    predictor_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute CE when the model projects only selected hidden-state positions."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"invalid selected CE shapes: logits={tuple(logits.shape)} "
            f"labels={tuple(labels.shape)}"
        )
    if predictor_positions.ndim != 1 or logits.shape[1] != predictor_positions.numel():
        raise RuntimeError("selected logits and predictor-position counts differ")
    if torch.any(predictor_positions < 0) or torch.any(
        predictor_positions + 1 >= labels.shape[1]
    ):
        raise RuntimeError("selected predictor position is outside the label sequence")
    selected_labels = labels.index_select(1, predictor_positions + 1).contiguous()
    supervised = selected_labels.ne(-100)
    if not torch.any(supervised):
        raise RuntimeError("selected SFT logits contain no supervised target")
    selected_loss = torch.nn.functional.cross_entropy(
        logits[supervised].float(),
        selected_labels[supervised],
        reduction="none",
    )
    token_loss = torch.zeros_like(selected_labels, dtype=selected_loss.dtype)
    token_loss = token_loss.masked_scatter(supervised, selected_loss)
    return token_loss, selected_labels


def remap_predictor_positions(
    positions: torch.Tensor,
    selected: torch.Tensor,
    *,
    sequence_length: int,
) -> torch.Tensor:
    """Map full-sequence logit positions into the selected-logit axis."""

    if positions.ndim != 1 or selected.ndim != 1:
        raise RuntimeError("predictor positions must be one-dimensional")
    lookup = torch.full(
        (sequence_length,), -1, dtype=torch.long, device=selected.device
    )
    lookup[selected] = torch.arange(selected.numel(), device=selected.device)
    mapped = torch.full_like(positions, -1)
    valid = positions.ge(0)
    if torch.any(positions[valid] >= sequence_length):
        raise RuntimeError("evidence predictor position is outside the sequence")
    mapped[valid] = lookup[positions[valid]]
    if torch.any(mapped[valid] < 0):
        raise RuntimeError("evidence predictor position was not selected for logits")
    return mapped


def evidence_pair_margin_loss(
    logits: torch.Tensor,
    roles: torch.Tensor,
    positions: torch.Tensor,
    status_token_ids: torch.Tensor,
    pair_valid: torch.Tensor,
    *,
    target_margin: float,
) -> torch.Tensor:
    """Rank evidence sufficiency by pixels, invariant to global action prior.

    For the insufficient state, ``logit(insufficient)-logit(sufficient)``
    should exceed the same quantity in the sufficient state by the requested
    margin.  Ordinary token CE still supplies the absolute labels; this term
    only enforces the paired visual delta that global action rebalancing
    cannot express.
    """

    if pair_valid.numel() != 1 or int(pair_valid.item()) == 0:
        return logits.sum() * 0.0
    if logits.ndim != 3 or roles.ndim != 1 or positions.ndim != 1:
        raise RuntimeError("invalid evidence-pair margin tensor ranks")
    if roles.numel() != 2 or positions.numel() != 2 or status_token_ids.numel() != 2:
        raise RuntimeError("evidence-pair margin requires exactly one two-state pair")
    insufficient_rows = torch.nonzero(roles.eq(0), as_tuple=False).flatten()
    sufficient_rows = torch.nonzero(roles.eq(1), as_tuple=False).flatten()
    if insufficient_rows.numel() != 1 or sufficient_rows.numel() != 1:
        raise RuntimeError(f"invalid evidence-pair roles: {roles.tolist()}")
    if torch.any(positions.lt(0)) or torch.any(positions.ge(logits.shape[1])):
        raise RuntimeError(
            f"invalid evidence-pair positions={positions.tolist()} sequence={logits.shape[1]}"
        )
    insufficient_token, sufficient_token = status_token_ids.unbind()
    row_indices = torch.arange(2, device=logits.device)
    # Index before widening to fp32.  Casting the full [B,T,V] tensor would
    # allocate several unnecessary GiB for a loss that uses four scalars.
    state_logits = logits[row_indices, positions].float()
    status_margin = (
        state_logits[:, insufficient_token] - state_logits[:, sufficient_token]
    )
    visual_delta = (
        status_margin[insufficient_rows[0]] - status_margin[sufficient_rows[0]]
    )
    return torch.nn.functional.softplus(
        logits.new_tensor(float(target_margin)) - visual_delta
    )


def evidence_pair_absolute_margin_loss(
    logits: torch.Tensor,
    roles: torch.Tensor,
    positions: torch.Tensor,
    status_token_ids: torch.Tensor,
    pair_valid: torch.Tensor,
    *,
    target_margin: float,
) -> torch.Tensor:
    """Place both paired states on the correct side of the stop threshold."""

    if pair_valid.numel() != 1 or int(pair_valid.item()) == 0:
        return logits.sum() * 0.0
    if logits.ndim != 3 or roles.ndim != 1 or positions.ndim != 1:
        raise RuntimeError("invalid evidence-pair absolute-margin tensor ranks")
    if roles.numel() != 2 or positions.numel() != 2 or status_token_ids.numel() != 2:
        raise RuntimeError("absolute evidence margin requires exactly one two-state pair")
    insufficient_rows = torch.nonzero(roles.eq(0), as_tuple=False).flatten()
    sufficient_rows = torch.nonzero(roles.eq(1), as_tuple=False).flatten()
    if insufficient_rows.numel() != 1 or sufficient_rows.numel() != 1:
        raise RuntimeError(f"invalid absolute-margin roles: {roles.tolist()}")
    if torch.any(positions.lt(0)) or torch.any(positions.ge(logits.shape[1])):
        raise RuntimeError(
            f"invalid absolute-margin positions={positions.tolist()} sequence={logits.shape[1]}"
        )
    insufficient_token, sufficient_token = status_token_ids.unbind()
    row_indices = torch.arange(2, device=logits.device)
    state_logits = logits[row_indices, positions].float()
    status_margin = (
        state_logits[:, insufficient_token] - state_logits[:, sufficient_token]
    )
    insufficient_margin = status_margin[insufficient_rows[0]]
    sufficient_margin = status_margin[sufficient_rows[0]]
    target = logits.new_tensor(float(target_margin))
    return 0.5 * (
        torch.nn.functional.softplus(target - insufficient_margin)
        + torch.nn.functional.softplus(target + sufficient_margin)
    )


def evidence_pair_insufficient_margin_loss(
    logits: torch.Tensor,
    roles: torch.Tensor,
    positions: torch.Tensor,
    status_token_ids: torch.Tensor,
    pair_valid: torch.Tensor,
    *,
    target_margin: float,
) -> torch.Tensor:
    """Correct only the under-confident insufficient side of a state pair.

    The held-out V47 audit placed sufficient states safely on the Answer side
    while insufficient states remained just below the decision boundary.  A
    symmetric loss keeps moving the already-correct side and changes the
    global answer prior.  This one-sided term instead raises
    ``logit(insufficient)-logit(sufficient)`` only for the causal state that
    lacks evidence.
    """

    if pair_valid.numel() != 1 or int(pair_valid.item()) == 0:
        return logits.sum() * 0.0
    if logits.ndim != 3 or roles.ndim != 1 or positions.ndim != 1:
        raise RuntimeError("invalid one-sided evidence-margin tensor ranks")
    if roles.numel() != 2 or positions.numel() != 2 or status_token_ids.numel() != 2:
        raise RuntimeError("one-sided evidence margin requires exactly one two-state pair")
    insufficient_rows = torch.nonzero(roles.eq(0), as_tuple=False).flatten()
    sufficient_rows = torch.nonzero(roles.eq(1), as_tuple=False).flatten()
    if insufficient_rows.numel() != 1 or sufficient_rows.numel() != 1:
        raise RuntimeError(f"invalid one-sided evidence roles: {roles.tolist()}")
    if torch.any(positions.lt(0)) or torch.any(positions.ge(logits.shape[1])):
        raise RuntimeError(
            f"invalid one-sided margin positions={positions.tolist()} sequence={logits.shape[1]}"
        )
    insufficient_token, sufficient_token = status_token_ids.unbind()
    row = insufficient_rows[0]
    state_logits = logits[row, positions[row]].float()
    insufficient_margin = (
        state_logits[insufficient_token] - state_logits[sufficient_token]
    )
    target = state_logits.new_tensor(float(target_margin))
    return torch.nn.functional.softplus(target - insufficient_margin)


def evidence_orbit_losses(
    logits: torch.Tensor,
    roles: torch.Tensor,
    positions: torch.Tensor,
    answer_token_ids: torch.Tensor,
    option_counts: torch.Tensor,
    letter_token_ids: torch.Tensor,
    orbit_valid: torch.Tensor,
    *,
    target_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return view-consistency and worst-view answer-margin objectives.

    Dense and context views may use independently permuted option letters, so
    comparing raw A--H distributions would be wrong.  We instead compare the
    permutation-invariant correct-vs-all-distractors margin in each view.  The
    floor term optimizes a smooth lower bound over both views: a source orbit
    is not considered learned when only one evidence rendering is easy.
    """

    zero = logits.sum() * 0.0
    if orbit_valid.numel() != 1 or int(orbit_valid.item()) == 0:
        return zero, zero
    if logits.ndim != 3:
        raise RuntimeError(f"invalid evidence-orbit logits: {tuple(logits.shape)}")
    if any(value.ndim != 1 for value in (
        roles,
        positions,
        answer_token_ids,
        option_counts,
        letter_token_ids,
    )):
        raise RuntimeError("evidence-orbit metadata must be one-dimensional")
    if roles.numel() != 2 or positions.numel() != 2:
        raise RuntimeError("evidence orbit requires one nested two-view atomic batch")
    direct_rows = torch.nonzero(roles.lt(2), as_tuple=False).flatten()
    if direct_rows.numel() != 2 or sorted(roles[direct_rows].tolist()) != [0, 1]:
        raise RuntimeError(f"invalid evidence-orbit roles: {roles.tolist()}")
    margins: List[torch.Tensor] = []
    for row_tensor in direct_rows:
        row = int(row_tensor)
        position = int(positions[row])
        count = int(option_counts[row])
        target_id = int(answer_token_ids[row])
        if not 0 <= position < logits.shape[1] or not 2 <= count <= 8:
            raise RuntimeError(
                f"invalid evidence-orbit position/count: row={row} "
                f"position={position} count={count}"
            )
        visible_ids = letter_token_ids[:count]
        matches = torch.nonzero(visible_ids.eq(target_id), as_tuple=False).flatten()
        if matches.numel() != 1:
            raise RuntimeError(
                f"evidence-orbit target token is not one visible option: {target_id}"
            )
        state_logits = logits[row, position].float().index_select(0, visible_ids)
        target_index = int(matches[0])
        distractor_mask = torch.ones(count, dtype=torch.bool, device=logits.device)
        distractor_mask[target_index] = False
        target_logit = state_logits[target_index]
        distractor_logsumexp = torch.logsumexp(
            state_logits[distractor_mask], dim=0
        )
        margins.append(target_logit - distractor_logsumexp)
    margin_tensor = torch.stack(margins)
    consistency = torch.nn.functional.smooth_l1_loss(
        margin_tensor[0], margin_tensor[1], reduction="mean"
    )
    # log(2) normalizes the soft minimum so equal margins remain unchanged.
    smooth_worst_margin = (
        -torch.logsumexp(-margin_tensor, dim=0)
        + math.log(float(margin_tensor.numel()))
    )
    floor = torch.nn.functional.softplus(
        logits.new_tensor(float(target_margin)) - smooth_worst_margin
    )
    return consistency, floor


class TurnWeightedTrainer(Trainer):
    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights")
        sample_loss_weights = inputs.pop("sample_loss_weights", None)
        gradient_route_ids = inputs.pop("gradient_route_ids", None)
        if bool(getattr(self, "causal_gradient_firewall_enabled", False)):
            if gradient_route_ids is None or gradient_route_ids.numel() < 1:
                raise RuntimeError("gradient-firewall route metadata is missing")
            unique_routes = torch.unique(gradient_route_ids.detach())
            if unique_routes.numel() != 1 or int(unique_routes[0]) not in {0, 1}:
                raise RuntimeError(
                    "gradient-firewall micro-batch route is not homogeneous"
                )
            route = int(unique_routes[0])
            micro_index = int(
                getattr(self, "causal_gradient_firewall_micro_index", 0)
            )
            if micro_index % 2 == 0:
                self.causal_gradient_firewall_first_route = route
            else:
                first_route = int(
                    getattr(self, "causal_gradient_firewall_first_route", -1)
                )
                if route != 0 or first_route not in {0, 1}:
                    raise RuntimeError(
                        "gradient-firewall optimizer step must be OST->EXACTPACKET "
                        "or EXACTPACKET->EXACTPACKET"
                    )
            self.causal_gradient_firewall_micro_index = micro_index + 1
            self.causal_gradient_firewall_current_route = route
        elif gradient_route_ids is not None:
            raise RuntimeError("unexpected gradient-firewall route metadata")
        pair_roles = inputs.pop("evidence_pair_roles", None)
        pair_positions = inputs.pop("evidence_pair_positions", None)
        pair_valid = inputs.pop("evidence_pair_valid", None)
        status_token_ids = inputs.pop("evidence_status_token_ids", None)
        orbit_roles = inputs.pop("evidence_orbit_roles", None)
        orbit_valid = inputs.pop("evidence_orbit_valid", None)
        orbit_positions = inputs.pop("evidence_orbit_answer_positions", None)
        orbit_answer_token_ids = inputs.pop(
            "evidence_orbit_answer_token_ids", None
        )
        orbit_option_counts = inputs.pop("evidence_orbit_option_counts", None)
        orbit_letter_token_ids = inputs.pop(
            "evidence_orbit_letter_token_ids", None
        )
        predictor_positions = supervised_predictor_positions(labels)
        # Qwen3-VL otherwise applies its 151k-way LM head to every visual
        # context position. Direct SFT supervises only the final answer span;
        # projecting the exact union of predictor positions is mathematically
        # identical and avoids materializing multi-GiB unused logits.
        inputs["logits_to_keep"] = predictor_positions
        outputs = model(**inputs)
        logits = outputs.logits
        token_loss, shift_labels = sparse_selected_token_loss(
            logits, labels, predictor_positions
        )
        shift_weights = loss_weights.index_select(
            1, predictor_positions + 1
        ).contiguous().float()
        # With per-device batch > 1, a single batch-global denominator silently
        # gives long JSON trajectories more mass than short MCQ anchors.  The
        # optional mode reproduces the historical micro-batch=1 objective while
        # retaining a larger, GPU-efficient micro-batch.
        loss = reduce_weighted_token_loss(
            token_loss,
            shift_labels,
            shift_weights,
            normalize_by_sample=bool(
                getattr(self, "sample_normalized_loss", False)
            ),
            sample_weights=sample_loss_weights,
        )
        pair_weight = float(getattr(self, "evidence_pair_margin_weight", 0.0))
        absolute_weight = float(
            getattr(self, "evidence_pair_absolute_margin_weight", 0.0)
        )
        insufficient_weight = float(
            getattr(self, "evidence_pair_insufficient_margin_weight", 0.0)
        )
        if pair_weight > 0 or absolute_weight > 0 or insufficient_weight > 0:
            if any(
                value is None
                for value in (
                    pair_roles,
                    pair_positions,
                    pair_valid,
                    status_token_ids,
                )
            ):
                raise RuntimeError("evidence-pair margin metadata is missing")
            pair_positions = remap_predictor_positions(
                pair_positions,
                predictor_positions,
                sequence_length=labels.shape[1],
            )
            if pair_weight > 0:
                pair_loss = evidence_pair_margin_loss(
                    logits,
                    pair_roles,
                    pair_positions,
                    status_token_ids,
                    pair_valid,
                    target_margin=float(
                        getattr(self, "evidence_pair_margin_target", 2.0)
                    ),
                )
                loss = loss + pair_weight * pair_loss
            if absolute_weight > 0:
                absolute_loss = evidence_pair_absolute_margin_loss(
                    logits,
                    pair_roles,
                    pair_positions,
                    status_token_ids,
                    pair_valid,
                    target_margin=float(
                        getattr(self, "evidence_pair_absolute_margin_target", 1.0)
                    ),
                )
                loss = loss + absolute_weight * absolute_loss
            if insufficient_weight > 0:
                insufficient_loss = evidence_pair_insufficient_margin_loss(
                    logits,
                    pair_roles,
                    pair_positions,
                    status_token_ids,
                    pair_valid,
                    target_margin=float(
                        getattr(self, "evidence_pair_insufficient_margin_target", 1.0)
                    ),
                )
                loss = loss + insufficient_weight * insufficient_loss
        orbit_consistency_weight = float(
            getattr(self, "evidence_orbit_consistency_weight", 0.0)
        )
        orbit_floor_weight = float(
            getattr(self, "evidence_orbit_floor_weight", 0.0)
        )
        if orbit_consistency_weight > 0 or orbit_floor_weight > 0:
            orbit_metadata = (
                orbit_roles,
                orbit_valid,
                orbit_positions,
                orbit_answer_token_ids,
                orbit_option_counts,
                orbit_letter_token_ids,
            )
            if any(value is None for value in orbit_metadata):
                raise RuntimeError("evidence-orbit loss metadata is missing")
            mapped_orbit_positions = remap_predictor_positions(
                orbit_positions,
                predictor_positions,
                sequence_length=labels.shape[1],
            )
            consistency_loss, floor_loss = evidence_orbit_losses(
                logits,
                orbit_roles,
                mapped_orbit_positions,
                orbit_answer_token_ids,
                orbit_option_counts,
                orbit_letter_token_ids,
                orbit_valid,
                target_margin=float(
                    getattr(self, "evidence_orbit_margin_target", 1.0)
                ),
            )
            loss = (
                loss
                + orbit_consistency_weight * consistency_loss
                + orbit_floor_weight * floor_loss
            )
        return (loss, outputs) if return_outputs else loss


def pair_atomic_partition(dataset: Any) -> tuple[List[List[int]], List[int]]:
    """Partition matched causal decisions into two-row atomic blocks.

    Each block stores one insufficient/observe state and one matched
    sufficient/answer state under one ``pair_id``.  Execution-recovery rows
    declare ``paired_decision_role`` because their provenance component remains
    an intervention recovery rather than being relabelled as an initial
    decision.  Keeping both roles in the same micro-batch removes early-step
    action-prior noise without changing sample weights or provenance.
    """

    rows = getattr(dataset, "rows", None)
    if not isinstance(rows, list):
        raise RuntimeError("pair-atomic batches require dataset.rows metadata")
    groups: Dict[str, List[tuple[int, str]]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        role = decision_role(row)
        if role < 0:
            continue
        action = "stream_observe" if role == 0 else "answer_mcq"
        pair_id = str(row.get("pair_id") or "")
        if not pair_id:
            raise RuntimeError(f"counterfactual decision row lacks pair_id: index={index}")
        groups[pair_id].append((index, action))

    blocks: List[List[int]] = []
    paired_indices: set[int] = set()
    for pair_id, members in sorted(groups.items()):
        actions = {action for _, action in members}
        if len(members) != 2 or actions != {"stream_observe", "answer_mcq"}:
            raise RuntimeError(
                f"invalid counterfactual pair {pair_id}: members={members}"
            )
        block = [index for index, _ in members]
        blocks.append(block)
        paired_indices.update(block)
    if not blocks:
        raise RuntimeError("pair-atomic batches found no counterfactual pairs")
    singles = [index for index in range(len(dataset)) if index not in paired_indices]
    if len(paired_indices) + len(singles) != len(dataset):
        raise RuntimeError("pair-atomic partition does not cover the dataset exactly once")
    return blocks, singles


def pair_atomic_execution_scope(
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
) -> str:
    """Return the supported scope that keeps every two-row unit atomic."""

    if per_device_batch_size >= 2 and per_device_batch_size % 2 == 0:
        return "micro_batch"
    if per_device_batch_size == 1 and gradient_accumulation_steps == 2:
        return "optimizer_step"
    raise ValueError(
        "pair-atomic batches require an even micro-batch size or "
        "micro-batch=1 with gradient-accumulation=2"
    )


def evidence_orbit_partition(dataset: Any) -> tuple[List[List[int]], List[int]]:
    """Partition complete dense/context source orbits and singletons."""

    rows = getattr(dataset, "rows", None)
    if not isinstance(rows, list):
        raise RuntimeError("evidence-orbit batches require dataset.rows metadata")
    groups: Dict[str, List[tuple[int, int, str]]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        role = evidence_orbit_role(row)
        source_id = nonempty_string(row.get("semantic_source_sha256"))
        certified_view = certified_nested_evidence_view(
            row
        ) or certified_frontier_representation_view(row)
        expected_view = "dense" if role == 0 else "context" if role == 1 else ""
        if not source_id or not certified_view:
            continue
        if certified_view != expected_view:
            raise RuntimeError(
                f"nested evidence certificate/view mismatch: {source_id}"
            )
        option_identity = correct_option_identity(row)
        if not option_identity:
            raise RuntimeError(
                f"evidence orbit lacks certified answer identity: {source_id}"
            )
        groups[source_id].append((index, role, option_identity))
    blocks: List[List[int]] = []
    orbit_indices: set[int] = set()
    for source_id, members in sorted(groups.items()):
        roles = [role for _, role, _ in members]
        direct_identities = {
            identity for _, role, identity in members if role in {0, 1}
        }
        if len(members) != 2 or sorted(roles) != [0, 1] or (
            "" in direct_identities or len(direct_identities) != 1
        ):
            raise RuntimeError(
                f"incomplete or duplicated evidence orbit {source_id}: {members}"
            )
        by_role = {role: index for index, role, _ in members}
        block = [by_role[0], by_role[1]]
        blocks.append(block)
        orbit_indices.update(block)
    if not blocks:
        raise RuntimeError("evidence-orbit batching found no complete source orbits")
    singles = [index for index in range(len(rows)) if index not in orbit_indices]
    if len(orbit_indices) + len(singles) != len(rows):
        raise RuntimeError("evidence-orbit partition does not cover the dataset")
    return blocks, singles


class EvidenceOrbitSampler(torch.utils.data.Sampler[int]):
    """Shuffle nested source-view pairs as indivisible micro-batches."""

    def __init__(
        self,
        orbit_blocks: List[List[int]],
        singleton_indices: List[int],
        *,
        seed: int,
    ) -> None:
        if any(len(block) != 2 for block in orbit_blocks):
            raise ValueError("every evidence-orbit block must contain two rows")
        self.orbit_blocks = [list(block) for block in orbit_blocks]
        self.singleton_indices = list(singleton_indices)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = 2 * len(self.orbit_blocks) + len(self.singleton_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):  # type: ignore[no-untyped-def]
        rng = random.Random(self.seed + self.epoch)
        blocks = [list(block) for block in self.orbit_blocks]
        singles = list(self.singleton_indices)
        rng.shuffle(singles)
        tail_count = len(singles) % 2
        tail = singles[-tail_count:] if tail_count else []
        if tail_count:
            singles = singles[:-tail_count]
        units = blocks + [
            singles[index : index + 2] for index in range(0, len(singles), 2)
        ]
        rng.shuffle(units)
        for unit in units:
            rng.shuffle(unit)
        order = [index for unit in units for index in unit] + tail
        if len(order) != self.num_samples or len(set(order)) != self.num_samples:
            raise RuntimeError("evidence-orbit sampler produced duplicate/missing rows")
        return iter(order)


class PairAtomicSampler(torch.utils.data.Sampler[int]):
    """Shuffle two-row causal pairs as indivisible micro-batches."""

    def __init__(
        self,
        pair_blocks: List[List[int]],
        singleton_indices: List[int],
        *,
        seed: int,
        stratified_schedule: bool = False,
    ) -> None:
        self.pair_blocks = [list(block) for block in pair_blocks]
        self.singleton_indices = list(singleton_indices)
        self.seed = int(seed)
        self.stratified_schedule = bool(stratified_schedule)
        self.epoch = 0
        self.num_samples = 2 * len(self.pair_blocks) + len(self.singleton_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):  # type: ignore[no-untyped-def]
        rng = random.Random(self.seed + self.epoch)
        pair_batches = [list(block) for block in self.pair_blocks]
        singles = list(self.singleton_indices)
        rng.shuffle(singles)
        # An odd corpus necessarily leaves one incomplete final micro-batch.
        # Keep that singleton as the epoch tail rather than interleaving it:
        # otherwise it shifts the two-row grid and can split every later
        # causal pair across DataLoader batches.  No row is dropped, padded or
        # duplicated, so this remains a one-pass, unit-mass schedule.
        tail_singleton = [singles.pop()] if len(singles) % 2 else []
        singleton_batches = [
            singles[index : index + 2] for index in range(0, len(singles), 2)
        ]
        rng.shuffle(pair_batches)
        rng.shuffle(singleton_batches)
        if self.stratified_schedule:
            # Proportional fair interleaving keeps every prefix within one
            # pair batch of the epoch-level ratio. This removes long local
            # runs of endpoint-heavy singleton trajectories while preserving
            # one-pass coverage and unit sample weights.
            batches = []
            pair_index = singleton_index = 0
            total = len(pair_batches) + len(singleton_batches)
            for position in range(total):
                expected_pairs = round((position + 1) * len(pair_batches) / total)
                choose_pair = (
                    pair_index < expected_pairs
                    and pair_index < len(pair_batches)
                ) or singleton_index >= len(singleton_batches)
                if choose_pair:
                    batches.append(pair_batches[pair_index])
                    pair_index += 1
                else:
                    batches.append(singleton_batches[singleton_index])
                    singleton_index += 1
            if pair_index != len(pair_batches) or singleton_index != len(singleton_batches):
                raise RuntimeError("stratified pair schedule did not consume every batch")
        else:
            batches = pair_batches + singleton_batches
            rng.shuffle(batches)
        if tail_singleton:
            batches.append(tail_singleton)
        for batch in batches:
            rng.shuffle(batch)
        order = [index for batch in batches for index in batch]
        if len(order) != self.num_samples or len(set(order)) != self.num_samples:
            raise RuntimeError("pair-atomic sampler produced duplicate or missing rows")
        return iter(order)


def media_signature(row: Mapping[str, Any]) -> str:
    """Return the actor-visible media branches exercised by one row."""

    messages = row.get("messages")
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    kinds: set[str] = set()
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, Mapping):
            continue
        for part in _list_value(message.get("content")):
            if not isinstance(part, Mapping):
                continue
            kind = str(part.get("type") or "")
            if kind in {"image", "video"}:
                kinds.add(kind)
    return "+".join(sorted(kinds)) if kinds else "text"


class CausalGradientFirewallSampler(torch.utils.data.Sampler[Any]):
    """Pair every causal global micro-batch with a categorical micro-batch.

    The first micro-batch in each paired optimizer step is OST and the second
    is EXACTPACKET.  Remaining categorical rows are emitted as categorical
    pairs.  Real rows appear exactly once; tuple references are zero-loss
    alignment padding understood by ``SpatialSFTDataset``.
    """

    def __init__(
        self,
        rows: List[Mapping[str, Any]],
        *,
        local_batch_size: int,
        world_size: int,
        gradient_accumulation_steps: int,
        length_bucketing: bool,
        seed: int,
    ) -> None:
        if local_batch_size not in {2, 4} or gradient_accumulation_steps != 2:
            raise ValueError(
                "gradient firewall requires micro-batch 2 or 4 and GAS 2"
            )
        if world_size < 1:
            raise ValueError("gradient firewall requires a positive world size")
        self.rows = rows
        self.local_batch_size = int(local_batch_size)
        self.world_size = int(world_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.length_bucketing = bool(length_bucketing)
        self.global_micro_batch = self.local_batch_size * self.world_size
        self.seed = int(seed)
        self.epoch = 0
        groups: Dict[str, List[int]] = collections.defaultdict(list)
        for index, row in enumerate(rows):
            family = nonempty_string(
                row.get("joint_family") or row.get("workfirst_family")
            )
            if family not in {"EXACTPACKET", "OST"}:
                raise RuntimeError(
                    f"gradient firewall found unsupported family at {index}: {family}"
                )
            groups[family].append(index)
        if set(groups) != {"EXACTPACKET", "OST"}:
            raise RuntimeError(f"gradient firewall family coverage drift: {groups.keys()}")
        if len(groups["EXACTPACKET"]) < len(groups["OST"]):
            raise RuntimeError("gradient firewall needs at least as many anchors as causal rows")
        self.groups = {key: list(value) for key, value in sorted(groups.items())}
        self.lengths = [direct_length_proxy(row) for row in rows]
        self._cached_order: List[Any] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cached_order = None

    def _chunks(self, indices: List[int]) -> List[List[Any]]:
        chunks: List[List[Any]] = []
        for start in range(0, len(indices), self.global_micro_batch):
            real = list(indices[start : start + self.global_micro_batch])
            padded: List[Any] = list(real)
            for offset in range(self.global_micro_batch - len(real)):
                padded.append((real[offset % len(real)], True))
            chunks.append(padded)
        return chunks

    def _order(self) -> List[Any]:
        if self._cached_order is not None:
            return list(self._cached_order)
        rng = random.Random(self.seed + self.epoch)
        exact = list(self.groups["EXACTPACKET"])
        causal = list(self.groups["OST"])
        rng.shuffle(exact)
        rng.shuffle(causal)
        if self.length_bucketing:
            exact.sort(key=lambda index: self.lengths[index], reverse=True)
            causal.sort(key=lambda index: self.lengths[index], reverse=True)
        exact_chunks = self._chunks(exact)
        causal_chunks = self._chunks(causal)
        if len(exact_chunks) < len(causal_chunks):
            raise RuntimeError("categorical chunk coverage is smaller than causal coverage")

        optimizer_steps: List[List[List[Any]]] = []
        for causal_chunk in causal_chunks:
            optimizer_steps.append([causal_chunk, exact_chunks.pop()])
        # Pair all remaining exact chunks.  If one full chunk remains, split
        # it into two positive, padded micro-batches so no all-padding loss is
        # ever constructed and every real row still appears exactly once.
        if len(exact_chunks) % 2:
            last = exact_chunks.pop()
            real = [value for value in last if not isinstance(value, tuple)]
            midpoint = max(1, len(real) // 2)
            halves = [real[:midpoint], real[midpoint:]]
            if not halves[1]:
                halves = [[real[0]], real[1:]]
            split_chunks: List[List[Any]] = []
            for half in halves:
                padded: List[Any] = list(half)
                for offset in range(self.global_micro_batch - len(half)):
                    padded.append((half[offset % len(half)], True))
                split_chunks.append(padded)
            optimizer_steps.append(split_chunks)
        while exact_chunks:
            optimizer_steps.append([exact_chunks.pop(), exact_chunks.pop()])
        rng.shuffle(optimizer_steps)
        output = [
            index
            for optimizer_step in optimizer_steps
            for micro_batch in optimizer_step
            for index in micro_batch
        ]
        real = [
            int(value[0]) if isinstance(value, tuple) else int(value)
            for value in output
            if not isinstance(value, tuple)
        ]
        if sorted(real) != list(range(len(self.rows))):
            raise RuntimeError("gradient firewall duplicated or omitted a real row")
        if len(output) % (self.global_micro_batch * 2):
            raise RuntimeError("gradient firewall emitted an incomplete optimizer step")
        self._cached_order = output
        return list(output)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._order())

    def __len__(self) -> int:
        return len(self._order())

    def stats(self) -> Dict[str, Any]:
        order = self._order()
        padding = sum(isinstance(value, tuple) for value in order)
        return {
            "enabled": True,
            "real_rows": len(self.rows),
            "family_rows": {
                family: len(indices) for family, indices in self.groups.items()
            },
            "local_batch_size": self.local_batch_size,
            "world_size": self.world_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "global_micro_batch": self.global_micro_batch,
            "optimizer_steps": len(order) // (self.global_micro_batch * 2),
            "zero_loss_padding_rows": padding,
            "real_row_duplication": False,
            "causal_microbatch_precedes_anchor_microbatch": True,
            "length_bucketing_within_family": self.length_bucketing,
            "benchmark_metadata_read": False,
        }


class MediaHomogeneousBatchSampler(torch.utils.data.Sampler[List[Any]]):
    """Keep ZeRO-3 image/video branch calls identical on every rank.

    Each media group is padded to a complete global batch using references to
    real rows. ``SpatialSFTDataset`` marks those references with zero loss, so
    every source row still contributes exactly once while all ranks execute
    the same parameterized visual branch sequence at each optimizer step.
    """

    drop_last = False

    def __init__(
        self,
        rows: List[Mapping[str, Any]],
        *,
        local_batch_size: int,
        world_size: int,
        seed: int,
    ) -> None:
        if local_batch_size < 1 or world_size < 1:
            raise ValueError("batch size and world size must be positive")
        self.rows = rows
        self.batch_size = int(local_batch_size)
        self.world_size = int(world_size)
        self.global_batch_size = self.batch_size * self.world_size
        self.seed = int(seed)
        self.epoch = 0
        groups: Dict[str, List[int]] = collections.defaultdict(list)
        for index, row in enumerate(rows):
            groups[media_signature(row)].append(index)
        if not groups:
            raise RuntimeError("media-homogeneous batching found no rows")
        self.groups = {key: value for key, value in sorted(groups.items())}
        self.padding_by_signature = {
            key: (-len(indices)) % self.global_batch_size
            for key, indices in self.groups.items()
        }
        self.num_batches = sum(
            (len(indices) + self.padding_by_signature[key]) // self.batch_size
            for key, indices in self.groups.items()
        )
        if self.num_batches % self.world_size:
            raise RuntimeError("media batches do not form complete global steps")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "real_rows": len(self.rows),
            "real_rows_by_signature": {
                key: len(indices) for key, indices in self.groups.items()
            },
            "zero_loss_padding_by_signature": dict(self.padding_by_signature),
            "zero_loss_padding_rows": sum(self.padding_by_signature.values()),
            "local_batch_size": self.batch_size,
            "world_size": self.world_size,
            "global_batch_size": self.global_batch_size,
            "global_steps": self.num_batches // self.world_size,
            "real_sample_mass_preserved": True,
            "benchmark_metadata_read": False,
            "model_or_score_selection": False,
        }

    def __iter__(self):  # type: ignore[no-untyped-def]
        rng = random.Random(self.seed + self.epoch)
        global_units: List[List[List[Any]]] = []
        seen_real: List[int] = []
        padding_count = 0
        for signature, source_indices in self.groups.items():
            indices = list(source_indices)
            rng.shuffle(indices)
            seen_real.extend(indices)
            padding = self.padding_by_signature[signature]
            padded: List[Any] = list(indices)
            for offset in range(padding):
                padded.append((indices[offset % len(indices)], True))
                padding_count += 1
            batches = [
                padded[start : start + self.batch_size]
                for start in range(0, len(padded), self.batch_size)
            ]
            if any(len(batch) != self.batch_size for batch in batches):
                raise RuntimeError(f"incomplete local media batch: {signature}")
            units = [
                batches[start : start + self.world_size]
                for start in range(0, len(batches), self.world_size)
            ]
            if any(len(unit) != self.world_size for unit in units):
                raise RuntimeError(f"incomplete global media step: {signature}")
            global_units.extend(units)
        rng.shuffle(global_units)
        if sorted(seen_real) != list(range(len(self.rows))):
            raise RuntimeError("media sampler duplicated or omitted a real row")
        if padding_count != sum(self.padding_by_signature.values()):
            raise RuntimeError("media sampler padding count drift")
        for unit in global_units:
            # Accelerate's BatchSamplerShard assigns these adjacent batches to
            # the ranks in one distributed step, so their signatures match.
            for batch in unit:
                yield batch


class FamilyStratifiedLengthSampler(torch.utils.data.Sampler[int]):
    """Interleave capability families inside every global optimizer step.

    Epoch-level loss weights cannot balance gradients when length bucketing
    places a rare, heavily weighted family in a small number of pure steps.
    Weighted fair queuing preserves every real row exactly once, while sorting
    only *within* a global step keeps simultaneous ranks close in sequence
    cost without separating families again.
    """

    def __init__(
        self,
        rows: List[Mapping[str, Any]],
        *,
        global_step_rows: int,
        seed: int,
    ) -> None:
        if global_step_rows < 1:
            raise ValueError("global optimizer-step size must be positive")
        self.rows = rows
        self.global_step_rows = int(global_step_rows)
        self.seed = int(seed)
        self.epoch = 0
        groups: Dict[str, List[int]] = collections.defaultdict(list)
        for index, row in enumerate(rows):
            family = optional_text(row.get("workfirst_family"))
            if not family:
                raise RuntimeError(
                    f"family-stratified schedule lacks workfirst_family at {index}"
                )
            groups[family].append(index)
        if len(groups) < 2:
            raise RuntimeError("family-stratified schedule requires at least two families")
        # Every complete step can contain every family only when each family
        # has at least one row per step. Fail closed rather than over-sampling.
        complete_steps = len(rows) // self.global_step_rows
        too_small = {
            family: len(indices)
            for family, indices in groups.items()
            if len(indices) < complete_steps
        }
        if too_small:
            raise RuntimeError(
                "family cannot cover every complete optimizer step without duplication: "
                f"{too_small} steps={complete_steps}"
            )
        self.groups = {family: list(indices) for family, indices in sorted(groups.items())}
        self.lengths = [direct_length_proxy(row) for row in rows]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def _order(self) -> List[int]:
        rng = random.Random(self.seed + self.epoch)
        shuffled = {family: list(indices) for family, indices in self.groups.items()}
        for indices in shuffled.values():
            rng.shuffle(indices)
        used = {family: 0 for family in shuffled}
        # The next normalized position is a weighted-fair-queue priority. A
        # deterministic random tie break avoids a fixed family order.
        tie = {family: rng.random() for family in shuffled}
        order: List[int] = []
        while len(order) < len(self.rows):
            available = [
                family for family, indices in shuffled.items()
                if used[family] < len(indices)
            ]
            family = min(
                available,
                key=lambda name: (
                    (used[name] + 1) / len(shuffled[name]), tie[name], name
                ),
            )
            order.append(shuffled[family][used[family]])
            used[family] += 1
        steps = [
            order[start : start + self.global_step_rows]
            for start in range(0, len(order), self.global_step_rows)
        ]
        # Batch size is one in the work-first runs. Sorting inside the global
        # step lets the same distributed micro-step process similar lengths on
        # all ranks, but cannot alter family counts for that optimizer step.
        for step in steps:
            step.sort(key=lambda index: self.lengths[index], reverse=True)
        output = [index for step in steps for index in step]
        if len(output) != len(self.rows) or len(set(output)) != len(self.rows):
            raise RuntimeError("family-stratified schedule duplicated or omitted rows")
        return output

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._order())

    def stats(self) -> Dict[str, Any]:
        order = self._order()
        complete = [
            order[start : start + self.global_step_rows]
            for start in range(0, len(order), self.global_step_rows)
            if len(order[start : start + self.global_step_rows]) == self.global_step_rows
        ]
        family_sets = [
            {optional_text(self.rows[index].get("workfirst_family")) for index in step}
            for step in complete
        ]
        expected = set(self.groups)
        return {
            "enabled": True,
            "real_rows": len(self.rows),
            "family_rows": {
                family: len(indices) for family, indices in self.groups.items()
            },
            "global_step_rows": self.global_step_rows,
            "complete_optimizer_steps": len(complete),
            "all_family_step_fraction": (
                sum(value == expected for value in family_sets) / len(family_sets)
            ),
            "real_sample_mass_preserved": True,
            "row_duplication": False,
            "sample_reweighting_changed": False,
            "benchmark_metadata_read": False,
        }


class PairAtomicTrainer(TurnWeightedTrainer):
    def get_train_dataloader(self):  # type: ignore[no-untyped-def]
        if not bool(getattr(self, "media_homogeneous_batches", False)):
            return super().get_train_dataloader()
        if any(
            bool(getattr(self, name, False))
            for name in (
                "length_grouped_batches",
                "pair_atomic_batches",
                "evidence_orbit_atomic_batches",
            )
        ):
            raise RuntimeError(
                "media-homogeneous batching is exclusive with other batch samplers"
            )
        dataset = self.train_dataset
        rows = getattr(dataset, "rows", None)
        if not isinstance(rows, list) or len(rows) != len(dataset):
            raise RuntimeError(
                "media-homogeneous batching requires complete dataset.rows metadata"
            )
        batch_sampler = MediaHomogeneousBatchSampler(
            rows,
            local_batch_size=int(self._train_batch_size),
            world_size=int(self.args.world_size),
            seed=int(getattr(self, "media_homogeneous_seed")),
        )
        if int(self.args.process_index) == 0:
            print({"media_homogeneous_batch_contract": batch_sampler.stats()})
        data_collator = self._get_collator_with_removed_columns(
            self.data_collator, description="Training"
        )
        params: Dict[str, Any] = {
            "batch_sampler": batch_sampler,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "worker_init_fn": partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            ),
        }
        if self.args.dataloader_num_workers > 0:
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return self.accelerator.prepare(torch.utils.data.DataLoader(dataset, **params))

    def _get_train_sampler(self, train_dataset: Any = None):  # type: ignore[no-untyped-def]
        if bool(getattr(self, "causal_gradient_firewall_batches", False)):
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            rows = getattr(dataset, "rows", None)
            if not isinstance(rows, list) or len(rows) != len(dataset):
                raise RuntimeError("gradient-firewall schedule requires dataset.rows")
            sampler = CausalGradientFirewallSampler(
                rows,
                local_batch_size=int(self.args.per_device_train_batch_size),
                world_size=int(self.args.world_size),
                gradient_accumulation_steps=int(
                    self.args.gradient_accumulation_steps
                ),
                length_bucketing=bool(
                    getattr(
                        self,
                        "causal_gradient_firewall_length_bucketing",
                        False,
                    )
                ),
                seed=int(getattr(self, "causal_gradient_firewall_seed", 0)),
            )
            if int(self.args.process_index) == 0:
                print({"causal_gradient_firewall_batch_contract": sampler.stats()})
            return sampler
        if bool(getattr(self, "family_stratified_length_batches", False)):
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            rows = getattr(dataset, "rows", None)
            if not isinstance(rows, list) or len(rows) != len(dataset):
                raise RuntimeError("family-stratified schedule requires dataset.rows")
            sampler = FamilyStratifiedLengthSampler(
                rows,
                global_step_rows=(
                    int(self.args.world_size)
                    * int(self.args.per_device_train_batch_size)
                    * int(self.args.gradient_accumulation_steps)
                    * int(getattr(self, "family_stratified_window_steps", 1))
                ),
                seed=int(getattr(self, "family_stratified_length_seed", 0)),
            )
            if int(self.args.process_index) == 0:
                print({"family_stratified_length_contract": sampler.stats()})
            return sampler
        if bool(getattr(self, "length_grouped_batches", False)):
            if bool(getattr(self, "pair_atomic_batches", False)):
                raise RuntimeError(
                    "length grouping and pair-atomic scheduling are mutually exclusive"
                )
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            rows = getattr(dataset, "rows", None)
            if not isinstance(rows, list) or len(rows) != len(dataset):
                raise RuntimeError("length grouping requires complete dataset.rows metadata")
            generator = torch.Generator()
            generator.manual_seed(int(getattr(self, "length_grouped_seed", 0)))
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=[direct_length_proxy(row) for row in rows],
                generator=generator,
            )
        if bool(getattr(self, "evidence_orbit_atomic_batches", False)):
            if bool(getattr(self, "pair_atomic_batches", False)):
                raise RuntimeError(
                    "evidence-orbit and counterfactual-pair schedules are exclusive"
                )
            return EvidenceOrbitSampler(
                getattr(self, "evidence_orbit_blocks"),
                getattr(self, "evidence_orbit_singles"),
                seed=int(getattr(self, "evidence_orbit_seed")),
            )
        if not bool(getattr(self, "pair_atomic_batches", False)):
            return super()._get_train_sampler(train_dataset)
        return PairAtomicSampler(
            getattr(self, "pair_atomic_blocks"),
            getattr(self, "pair_atomic_singles"),
            seed=int(getattr(self, "pair_atomic_seed")),
            stratified_schedule=bool(
                getattr(self, "pair_atomic_stratified_schedule", False)
            ),
        )


def causal_firewall_parameters(
    model: Any, last_layers: int
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return the high-level language decision parameters protected from OST."""

    if last_layers < 1:
        return []
    indexed: list[tuple[int, str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        match = re.search(r"(?:^|\.)language_model\.layers\.(\d+)\.", name)
        if match is not None:
            indexed.append((int(match.group(1)), name, parameter))
    if not indexed:
        raise RuntimeError("gradient firewall found no language blocks")
    total_layers = max(index for index, _, _ in indexed) + 1
    if last_layers >= total_layers:
        raise RuntimeError(
            f"gradient firewall must leave an unprotected state stack: "
            f"last={last_layers} total={total_layers}"
        )
    threshold = total_layers - last_layers
    protected: list[tuple[str, torch.nn.Parameter]] = [
        (name, parameter)
        for index, name, parameter in indexed
        if index >= threshold and parameter.requires_grad
    ]
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized = name.lower()
        if (
            "language_model.norm" in normalized
            or normalized.endswith("lm_head.weight")
        ):
            protected.append((name, parameter))
    deduplicated = {name: parameter for name, parameter in protected}
    if not deduplicated:
        raise RuntimeError("gradient firewall protected no trainable parameters")
    return sorted(deduplicated.items())


def causal_state_path_firewall_parameters(
    model: Any,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Protect every trainable parameter outside the visual state path.

    With the default full-SFT contract the vision tower is frozen and the
    visual merger remains trainable.  Relation-route gradients can therefore
    update the merger that writes visual state into the language stream, while
    answer-route gradients continue to update the complete trainable model.
    The definition is architectural rather than layer-depth selected.
    """

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    state_path = [
        (name, parameter)
        for name, parameter in trainable
        if "visual" in name.lower()
    ]
    if not state_path:
        raise RuntimeError(
            "state-path firewall found no trainable visual state parameters"
        )
    protected = [
        (name, parameter)
        for name, parameter in trainable
        if "visual" not in name.lower()
    ]
    if not protected:
        raise RuntimeError(
            "state-path firewall found no non-visual reader parameters to protect"
        )
    return sorted(protected)


def apply_causal_gradient_firewall(
    trainer: Any, gradient: torch.Tensor
) -> torch.Tensor:
    """Remove only causal gradients from the protected decision parameters."""

    route = int(trainer.causal_gradient_firewall_current_route)
    if route == 1:
        return torch.zeros_like(gradient)
    if route == 0:
        return gradient
    raise RuntimeError(f"gradient-firewall backward lacks a valid route: {route}")


def configure_full_finetune(
    model: Any,
    freeze_vision_encoder: bool,
    freeze_visual_merger: bool,
    freeze_language_layers_before: int = 0,
) -> Dict[str, int]:
    trainable = frozen = 0
    for name, parameter in model.named_parameters():
        is_visual = "visual" in name.lower()
        is_merger = "merger" in name.lower()
        freeze_visual_parameter = (
            freeze_vision_encoder
            and is_visual
            and (freeze_visual_merger or not is_merger)
        )
        layer = re.search(r"(?:^|\.)language_model\.layers\.(\d+)\.", name)
        freeze_lower_language = bool(
            freeze_language_layers_before > 0
            and (
                (layer is not None and int(layer.group(1)) < freeze_language_layers_before)
                or "language_model.embed_tokens" in name
            )
        )
        parameter.requires_grad = not (freeze_visual_parameter or freeze_lower_language)
        count = parameter.numel()
        if parameter.requires_grad:
            trainable += count
        else:
            frozen += count
    if trainable == 0:
        raise RuntimeError("full-finetune configuration left no trainable parameters")
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def training_args(args: argparse.Namespace) -> TrainingArguments:
    trainer_seed = (
        args.seed if args.trainer_seed is None else args.trainer_seed
    )
    kwargs: Dict[str, Any] = {
        "output_dir": str(args.output),
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": 1.0,
        "bf16": True,
        "tf32": args.tf32,
        "logging_steps": args.logging_steps,
        "disable_tqdm": True,
        "save_steps": args.save_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": "none",
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "deepspeed": str(args.deepspeed) if args.deepspeed else None,
        # Keep initialization and Trainer RNG identities explicit. Historical
        # runs before 2026-08-13 called set_seed(args.seed) before loading but
        # then let TrainingArguments reset training/data RNGs to its default
        # 42. ``--trainer-seed`` makes that legacy execution contract
        # reproducible without silently lying in the materialized manifest.
        "seed": trainer_seed,
        "data_seed": trainer_seed,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if args.save_only_model:
        if "save_only_model" not in signature.parameters:
            raise RuntimeError(
                "installed Transformers does not support model-only checkpoints"
            )
        kwargs["save_only_model"] = True
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[eval_key] = "no"
    return TrainingArguments(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deepspeed", type=Path, default=None)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=0,
        help=(
            "gracefully save and stop at this global step while retaining "
            "--max-steps as the scheduler horizon"
        ),
    )
    parser.add_argument(
        "--force-save-every-steps",
        type=int,
        default=0,
        help="callback-level save cadence that remains active after trainer-state restore",
    )
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=0,
        help="per-image longest-edge pixel budget; 0 preserves the base processor",
    )
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=625)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("linear", "cosine"),
        default="cosine",
        help=(
            "Explicit scheduler contract. The immutable 52.38 checkpoint used "
            "linear or cosine learning-rate schedule."
        ),
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=125)
    parser.add_argument(
        "--save-strategy",
        choices=("no", "steps", "epoch"),
        default="steps",
        help="Disable intermediate checkpoints when only final_model is required.",
    )
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--save-only-model",
        action="store_true",
        help=(
            "omit optimizer/scheduler/RNG state from diagnostic checkpoints"
        ),
    )
    parser.add_argument(
        "--skip-final-save",
        action="store_true",
        help=(
            "diagnostic-only memory/throughput probe: stop after the declared "
            "step and write per-rank CUDA telemetry without materializing a "
            "checkpoint; requires --stop-after-step"
        ),
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--dry-run-samples", type=int, default=0)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument(
        "--dataloader-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pin DataLoader CPU tensors before H2D; disable on runtimes whose mapped CUDA device rejects pin_memory(device)",
    )
    parser.add_argument(
        "--length-grouped-batches",
        action="store_true",
        help=(
            "randomly bucket examples by a media/text cost proxy to reduce "
            "padding and cross-rank stragglers without changing row weights"
        ),
    )
    parser.add_argument(
        "--family-stratified-length-batches",
        action="store_true",
        help=(
            "preserve one-pass row coverage while proportionally mixing every "
            "workfirst_family inside each distributed optimizer step; length "
            "sorting is restricted to rows already assigned to that step"
        ),
    )
    parser.add_argument(
        "--family-stratified-window-steps",
        type=int,
        default=1,
        help=(
            "number of adjacent optimizer steps in one proportional family "
            "window; values above one trade exact per-step coverage for better "
            "cross-rank length efficiency without changing row coverage"
        ),
    )
    parser.add_argument(
        "--media-homogeneous-batches",
        action="store_true",
        help=(
            "synchronize image/video branch signatures across every ZeRO-3 "
            "rank; alignment rows have zero loss and do not change real-row mass"
        ),
    )
    parser.add_argument(
        "--causal-gradient-firewall-batches",
        action="store_true",
        help=(
            "pair each homogeneous OST micro-batch with an EXACTPACKET "
            "micro-batch in one optimizer step; requires batch=2 and GAS=2"
        ),
    )
    parser.add_argument(
        "--causal-gradient-firewall-last-layers",
        type=int,
        default=0,
        help=(
            "number of final language blocks whose OST gradients are set to "
            "zero while categorical gradients remain unchanged"
        ),
    )
    parser.add_argument(
        "--causal-gradient-firewall-state-path-only",
        action="store_true",
        help=(
            "zero relation-route gradients for every non-visual trainable "
            "parameter; answer-route gradients remain unchanged"
        ),
    )
    parser.add_argument(
        "--causal-gradient-firewall-length-bucketing",
        action="store_true",
        help=(
            "sort rows by the generic sequence/media cost proxy inside each "
            "firewall family before forming distributed micro-batches"
        ),
    )
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "trade activation memory for recomputation; disable only after a "
            "measured memory probe demonstrates sufficient device headroom"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--trainer-seed",
        type=int,
        default=None,
        help=(
            "explicit Trainer/data RNG seed; defaults to --seed. Set this "
            "separately only to replay an audited historical mixed-seed run"
        ),
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TF32 matmuls; pass --no-tf32 for the historical SFT contract.",
    )
    parser.add_argument("--train-vision-encoder", action="store_true")
    parser.add_argument("--freeze-visual-merger", action="store_true")
    parser.add_argument(
        "--freeze-language-layers-before",
        type=int,
        default=0,
        help="freeze token embeddings and language blocks with index below this value",
    )
    parser.add_argument("--token-normalized-loss", action="store_true")
    parser.add_argument(
        "--sample-normalized-loss",
        action="store_true",
        help=(
            "normalize weighted assistant-token CE inside each sample before "
            "averaging the micro-batch; this preserves the legacy batch=1 "
            "objective at larger micro-batches"
        ),
    )
    parser.add_argument(
        "--pair-atomic-batches",
        action="store_true",
        help=(
            "place each exact insufficient/sufficient counterfactual pair in "
            "one micro-batch or gradient-accumulation step; ordering only, "
            "with no reweighting"
        ),
    )
    parser.add_argument(
        "--expected-pair-blocks",
        type=int,
        default=0,
        help="fail closed unless pair-atomic discovery finds this many pairs",
    )
    parser.add_argument(
        "--pair-stratified-schedule",
        action="store_true",
        help=(
            "proportionally interleave counterfactual-pair and singleton "
            "batches so every training prefix stays globally representative"
        ),
    )
    parser.add_argument(
        "--evidence-orbit-atomic-batches",
        action="store_true",
        help=(
            "place the nested dense and context views of one source in a "
            "single two-row micro-batch; no row duplication or reweighting"
        ),
    )
    parser.add_argument(
        "--expected-evidence-orbits",
        type=int,
        default=0,
        help="fail closed unless exactly this many nested two-view orbits are found",
    )
    parser.add_argument(
        "--evidence-orbit-consistency-weight",
        type=float,
        default=0.0,
        help="coefficient for dense/context correct-margin consistency",
    )
    parser.add_argument(
        "--evidence-orbit-floor-weight",
        type=float,
        default=0.0,
        help="coefficient for the smooth worst-view correct-margin floor",
    )
    parser.add_argument(
        "--evidence-orbit-margin-target",
        type=float,
        default=1.0,
        help="target correct-vs-distractor logit margin for every orbit view",
    )
    parser.add_argument(
        "--evidence-pair-margin-weight",
        type=float,
        default=0.0,
        help=(
            "non-negative coefficient for the paired visual evidence-status "
            "margin; positive values require true two-row atomic micro-batches"
        ),
    )
    parser.add_argument(
        "--evidence-pair-margin-target",
        type=float,
        default=2.0,
        help="target logit-delta separation for an insufficient/sufficient pair",
    )
    parser.add_argument(
        "--evidence-pair-absolute-margin-weight",
        type=float,
        default=0.0,
        help=(
            "coefficient for placing insufficient and sufficient states on "
            "opposite sides of the evidence-status decision threshold"
        ),
    )
    parser.add_argument(
        "--evidence-pair-absolute-margin-target",
        type=float,
        default=1.0,
        help="per-state evidence-status logit margin around the stop threshold",
    )
    parser.add_argument(
        "--evidence-pair-insufficient-margin-weight",
        type=float,
        default=0.0,
        help=(
            "coefficient for a one-sided margin that moves only insufficient "
            "states onto the Observe side of the commitment threshold"
        ),
    )
    parser.add_argument(
        "--evidence-pair-insufficient-margin-target",
        type=float,
        default=1.0,
        help="target insufficient-vs-sufficient token margin for insufficient states",
    )
    parser.add_argument("--observe-weight", type=float, default=1.0)
    parser.add_argument(
        "--observe-prefix-weight",
        type=float,
        default=None,
        help="optional legacy-specific weight for observe_prefix only",
    )
    parser.add_argument("--evidence-weight", type=float, default=1.0)
    parser.add_argument("--memory-weight", type=float, default=1.25)
    parser.add_argument("--answer-weight", type=float, default=3.0)
    parser.add_argument("--deployment-answer-weight", type=float, default=3.0)
    parser.add_argument("--other-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_length <= 0 or args.max_pixels < 0:
        raise ValueError("max_length must be positive and max_pixels non-negative")
    if args.freeze_language_layers_before < 0:
        raise ValueError("freeze-language-layers-before must be non-negative")
    if args.causal_gradient_firewall_last_layers < 0:
        raise ValueError("causal-gradient-firewall-last-layers must be non-negative")
    if (
        args.causal_gradient_firewall_state_path_only
        and args.causal_gradient_firewall_last_layers > 0
    ):
        raise ValueError(
            "state-path-only and last-layer gradient firewalls are mutually exclusive"
        )
    firewall_enabled = bool(
        args.causal_gradient_firewall_last_layers > 0
        or args.causal_gradient_firewall_state_path_only
    )
    if firewall_enabled != bool(args.causal_gradient_firewall_batches):
        raise ValueError(
            "gradient firewall requires its paired batch schedule and exactly "
            "one parameter-routing mode"
        )
    if args.causal_gradient_firewall_length_bucketing and not firewall_enabled:
        raise ValueError(
            "gradient-firewall length bucketing requires the firewall"
        )
    if firewall_enabled and not (
        args.per_device_batch_size in {2, 4}
        and args.gradient_accumulation_steps == 2
    ):
        raise ValueError(
            "gradient firewall requires per-device batch 2 or 4 and GAS 2"
        )
    if args.skip_final_save and args.stop_after_step <= 0:
        raise ValueError("--skip-final-save requires --stop-after-step")
    if (
        args.evidence_pair_margin_weight < 0
        or args.evidence_pair_absolute_margin_weight < 0
        or args.evidence_pair_insufficient_margin_weight < 0
        or args.evidence_pair_margin_target <= 0
        or args.evidence_pair_absolute_margin_target <= 0
        or args.evidence_pair_insufficient_margin_target <= 0
    ):
        raise ValueError("evidence-pair margin weights/targets must be non-negative/positive")
    if (
        args.evidence_orbit_consistency_weight < 0
        or args.evidence_orbit_floor_weight < 0
        or args.evidence_orbit_margin_target <= 0
    ):
        raise ValueError(
            "evidence-orbit weights/target must be non-negative/positive"
        )
    if (
        args.evidence_pair_margin_weight > 0
        or args.evidence_pair_absolute_margin_weight > 0
        or args.evidence_pair_insufficient_margin_weight > 0
    ) and not (
        args.pair_atomic_batches and args.per_device_batch_size == 2
    ):
        raise ValueError(
            "evidence-pair margin requires --pair-atomic-batches and "
            "--per-device-batch-size 2 so both visual states share one forward pass"
        )
    if args.length_grouped_batches and args.pair_atomic_batches:
        raise ValueError(
            "--length-grouped-batches cannot be combined with --pair-atomic-batches"
        )
    if args.family_stratified_length_batches and any(
        (
            args.length_grouped_batches,
            args.pair_atomic_batches,
            args.evidence_orbit_atomic_batches,
            args.media_homogeneous_batches,
            args.causal_gradient_firewall_batches,
        )
    ):
        raise ValueError(
            "--family-stratified-length-batches is exclusive with other batch samplers"
        )
    if args.family_stratified_window_steps < 1 or (
        args.family_stratified_window_steps != 1
        and not args.family_stratified_length_batches
    ):
        raise ValueError(
            "--family-stratified-window-steps requires a positive value and "
            "--family-stratified-length-batches"
        )
    if args.media_homogeneous_batches and any(
        (
            args.length_grouped_batches,
            args.family_stratified_length_batches,
            args.pair_atomic_batches,
            args.evidence_orbit_atomic_batches,
            args.causal_gradient_firewall_batches,
        )
    ):
        raise ValueError(
            "--media-homogeneous-batches is exclusive with other batch samplers"
        )
    if args.causal_gradient_firewall_batches and any(
        (
            args.length_grouped_batches,
            args.family_stratified_length_batches,
            args.media_homogeneous_batches,
            args.pair_atomic_batches,
            args.evidence_orbit_atomic_batches,
        )
    ):
        raise ValueError(
            "--causal-gradient-firewall-batches is exclusive with other batch samplers"
        )
    orbit_loss_enabled = bool(
        args.evidence_orbit_consistency_weight > 0
        or args.evidence_orbit_floor_weight > 0
    )
    if orbit_loss_enabled and not (
        args.evidence_orbit_atomic_batches and args.per_device_batch_size == 2
    ):
        raise ValueError(
            "evidence-orbit losses require --evidence-orbit-atomic-batches "
            "and --per-device-batch-size 2"
        )
    if args.evidence_orbit_atomic_batches and args.pair_atomic_batches:
        raise ValueError(
            "evidence-orbit and counterfactual-pair schedules are mutually exclusive"
        )
    if args.length_grouped_batches and args.evidence_orbit_atomic_batches:
        raise ValueError(
            "length grouping and evidence-orbit scheduling are mutually exclusive"
        )
    set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    processor_image_size = configure_processor_pixel_budget(
        processor, args.max_pixels
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = SpatialSFTDataset(args.train, processor, max_samples=args.max_train_samples)
    source_sample_weights = [
        float(
            row.get(
                "sample_loss_weight",
                row.get("sample_weight", 1.0),
            )
        )
        for row in dataset.rows
    ]
    source_sample_reweighting = any(
        not math.isclose(weight, 1.0, rel_tol=0.0, abs_tol=1e-12)
        for weight in source_sample_weights
    )
    source_sample_weight_distribution = dict(
        sorted(
            collections.Counter(str(weight) for weight in source_sample_weights).items()
        )
    )
    pair_blocks: List[List[int]] = []
    pair_singles: List[int] = []
    pair_atomic_stats: Dict[str, Any] = {
        "enabled": False,
        "pair_blocks": 0,
        "paired_rows": 0,
        "singleton_rows": len(dataset),
    }
    if args.pair_atomic_batches:
        atomic_scope = pair_atomic_execution_scope(
            args.per_device_batch_size,
            args.gradient_accumulation_steps,
        )
        pair_blocks, pair_singles = pair_atomic_partition(dataset)
        if args.expected_pair_blocks > 0 and len(pair_blocks) != args.expected_pair_blocks:
            raise RuntimeError(
                f"pair-block count drift: expected={args.expected_pair_blocks} "
                f"actual={len(pair_blocks)}"
            )
        pair_atomic_stats = {
            "enabled": True,
            "pair_blocks": len(pair_blocks),
            "paired_rows": 2 * len(pair_blocks),
            "singleton_rows": len(pair_singles),
            "tail_singleton_rows": len(pair_singles) % 2,
            "micro_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "atomic_scope": atomic_scope,
            "schedule": (
                "proportional_stratified"
                if args.pair_stratified_schedule
                else "random_batch"
            ),
            "sample_reweighting": False,
        }
        if int(os.environ.get("RANK", "0")) == 0:
            print({"pair_atomic_contract": pair_atomic_stats})
    elif args.pair_stratified_schedule:
        raise ValueError("--pair-stratified-schedule requires --pair-atomic-batches")
    orbit_blocks: List[List[int]] = []
    orbit_singles: List[int] = []
    orbit_stats: Dict[str, Any] = {
        "enabled": False,
        "orbit_blocks": 0,
        "orbit_rows": 0,
        "singleton_rows": len(dataset),
    }
    if args.evidence_orbit_atomic_batches:
        orbit_blocks, orbit_singles = evidence_orbit_partition(dataset)
        if (
            args.expected_evidence_orbits > 0
            and len(orbit_blocks) != args.expected_evidence_orbits
        ):
            raise RuntimeError(
                "evidence-orbit count drift: "
                f"expected={args.expected_evidence_orbits} actual={len(orbit_blocks)}"
            )
        orbit_stats = {
            "enabled": True,
            "orbit_blocks": len(orbit_blocks),
            "orbit_rows": 2 * len(orbit_blocks),
            "singleton_rows": len(orbit_singles),
            "tail_singleton_rows": len(orbit_singles) % 2,
            "micro_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "sample_reweighting": False,
        }
        if int(os.environ.get("RANK", "0")) == 0:
            print({"evidence_orbit_contract": orbit_stats})
    family_weights = {
        "observe": args.observe_weight,
        "observe_prefix": (
            args.observe_weight
            if args.observe_prefix_weight is None
            else args.observe_prefix_weight
        ),
        "evidence": args.evidence_weight,
        "memory": args.memory_weight,
        "answer": args.answer_weight,
        "deployment_answer": args.deployment_answer_weight,
        "other": args.other_weight,
    }
    if any(value < 0 for value in family_weights.values()):
        raise ValueError(f"family weights must be non-negative: {family_weights}")
    collator = TransitionCollator(
        processor=processor,
        max_length=args.max_length,
        family_weights=family_weights,
        normalize_by_turn=not args.token_normalized_loss,
        evidence_pair_margin_enabled=(
            args.evidence_pair_margin_weight > 0
            or args.evidence_pair_absolute_margin_weight > 0
            or args.evidence_pair_insufficient_margin_weight > 0
        ),
        evidence_orbit_enabled=orbit_loss_enabled,
        causal_gradient_firewall_enabled=firewall_enabled,
    )
    if args.dry_run_samples > 0:
        count = min(args.dry_run_samples, len(dataset))
        indices = {
            round(i * (len(dataset) - 1) / max(count - 1, 1))
            for i in range(count)
        }
        # The parquet is deterministically shuffled, so an evenly spaced probe
        # alone can miss a rare long/many-image trajectory.  Add extrema and
        # one representative for every component, protocol, and task.
        rows = getattr(dataset, "rows", [])
        if (
            args.evidence_pair_margin_weight > 0
            or args.evidence_pair_absolute_margin_weight > 0
            or args.evidence_pair_insufficient_margin_weight > 0
        ):
            if not pair_blocks:
                raise RuntimeError("evidence-pair dry run found no atomic pair")
            pair_batch = collator([dataset[index] for index in pair_blocks[0]])
            if (
                int(pair_batch["evidence_pair_valid"].item()) != 1
                or sorted(pair_batch["evidence_pair_roles"].tolist()) != [0, 1]
                or torch.any(pair_batch["evidence_pair_positions"].lt(0))
            ):
                raise RuntimeError("evidence-pair dry-run metadata contract failed")
            print(
                json.dumps(
                    {
                    "evidence_pair_margin_dry_run": "passed",
                    "weight": args.evidence_pair_margin_weight,
                    "target": args.evidence_pair_margin_target,
                    "absolute_weight": args.evidence_pair_absolute_margin_weight,
                    "absolute_target": args.evidence_pair_absolute_margin_target,
                    "insufficient_weight": args.evidence_pair_insufficient_margin_weight,
                    "insufficient_target": args.evidence_pair_insufficient_margin_target,
                    "roles": pair_batch["evidence_pair_roles"].tolist(),
                    "positions": pair_batch["evidence_pair_positions"].tolist(),
                    "status_token_ids": pair_batch[
                        "evidence_status_token_ids"
                    ].tolist(),
                    },
                    sort_keys=True,
                )
            )
        if orbit_loss_enabled:
            if not orbit_blocks:
                raise RuntimeError("evidence-orbit dry run found no atomic orbit")
            orbit_batch = collator([dataset[index] for index in orbit_blocks[0]])
            if (
                int(orbit_batch["evidence_orbit_valid"].item()) != 1
                or sorted(orbit_batch["evidence_orbit_roles"].tolist())
                != [0, 1]
                or torch.sum(
                    orbit_batch["evidence_orbit_answer_positions"].ge(0)
                ).item()
                != 2
            ):
                raise RuntimeError("evidence-orbit dry-run metadata contract failed")
            print(
                json.dumps(
                    {
                        "evidence_orbit_dry_run": "passed",
                        "consistency_weight": args.evidence_orbit_consistency_weight,
                        "floor_weight": args.evidence_orbit_floor_weight,
                        "margin_target": args.evidence_orbit_margin_target,
                        "roles": orbit_batch["evidence_orbit_roles"].tolist(),
                        "positions": orbit_batch[
                            "evidence_orbit_answer_positions"
                        ].tolist(),
                    },
                    sort_keys=True,
                )
            )
        if rows:
            indices.add(
                max(range(len(rows)), key=lambda i: optional_count(rows[i].get("num_images")))
            )
            indices.add(
                max(range(len(rows)), key=lambda i: optional_count(rows[i].get("num_turns")))
            )
            for key in ("component", "protocol_family"):
                groups: Dict[str, List[int]] = collections.defaultdict(list)
                for index, row in enumerate(rows):
                    groups[str(row.get(key) or "")].append(index)
                for group in groups.values():
                    indices.add(group[0])
                    indices.add(
                        max(group, key=lambda i: optional_count(rows[i].get("num_images")))
                    )
        indices = sorted(indices)
        max_tokens = 0
        max_images = 0
        max_token_uid = ""
        max_image_uid = ""
        token_lengths: List[int] = []
        sampled_components: collections.Counter[str] = collections.Counter()
        sampled_protocols: collections.Counter[str] = collections.Counter()
        for index in indices:
            batch = collator([dataset[index]])
            tokens = int(batch["attention_mask"].sum().item())
            images = int(batch.get("image_grid_thw", torch.empty(0, 3)).shape[0])
            token_lengths.append(tokens)
            if tokens > max_tokens:
                max_tokens = tokens
                max_token_uid = str(rows[index].get("uid") if rows else index)
            if images > max_images:
                max_images = images
                max_image_uid = str(rows[index].get("uid") if rows else index)
            if rows:
                sampled_components[str(rows[index].get("component") or "")] += 1
                sampled_protocols[str(rows[index].get("protocol_family") or "")] += 1
        sorted_lengths = sorted(token_lengths)
        def percentile(fraction: float) -> int:
            position = round(fraction * (len(sorted_lengths) - 1))
            return sorted_lengths[position]
        print(
            json.dumps(
                {
                "dry_run": "passed",
                "samples": len(indices),
                "max_tokens": max_tokens,
                "max_token_uid": max_token_uid,
                "max_images": max_images,
                "max_image_uid": max_image_uid,
                "token_length_p50": percentile(0.50),
                "token_length_p95": percentile(0.95),
                "token_length_p99": percentile(0.99),
                "sampled_components": dict(sorted(sampled_components.items())),
                "sampled_protocols": dict(sorted(sampled_protocols.items())),
                "turn_normalized": not args.token_normalized_loss,
                "sample_normalized": args.sample_normalized_loss,
                "sample_loss_weight_semantics": (
                    "absolute_gradient_multiplier_v1"
                ),
                "family_weights": family_weights,
                },
                sort_keys=True,
            )
        )
        return
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    text_config = getattr(model.config, "text_config", model.config)
    language_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)
    if args.freeze_language_layers_before > language_layers:
        raise ValueError(
            f"cannot freeze before layer {args.freeze_language_layers_before}; model has {language_layers}"
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    stats = configure_full_finetune(
        model,
        not args.train_vision_encoder,
        args.freeze_visual_merger,
        args.freeze_language_layers_before,
    )
    finetune = (
        "full_model"
        if args.train_vision_encoder
        else "full_language_and_merger"
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            {
                "parameter_contract": stats,
                "finetune": finetune,
                "vision_encoder_frozen": not args.train_vision_encoder,
                "visual_merger_frozen": args.freeze_visual_merger,
                "freeze_language_layers_before": args.freeze_language_layers_before,
            }
        )

    callbacks = (
        [
            StopAfterStepCallback(
                args.stop_after_step,
                args.force_save_every_steps,
                save_on_stop=not args.skip_final_save,
            )
        ]
        if args.stop_after_step > 0 or args.force_save_every_steps > 0
        else None
    )
    trainer = PairAtomicTrainer(
        model=model,
        args=training_args(args),
        train_dataset=dataset,
        data_collator=collator,
        callbacks=callbacks,
        # Transformers saves the processing class into every checkpoint.  A
        # model-only checkpoint cannot be evaluated causally because
        # AutoProcessor then lacks its image processor/chat template and may
        # silently fall back to unrelated files.  Supplying the exact training
        # processor keeps intermediate gates self-contained.
        processing_class=processor,
    )
    trainer.sample_normalized_loss = bool(args.sample_normalized_loss)
    trainer.pair_atomic_batches = bool(args.pair_atomic_batches)
    trainer.pair_atomic_blocks = pair_blocks
    trainer.pair_atomic_singles = pair_singles
    trainer.pair_atomic_seed = int(args.seed)
    trainer.pair_atomic_stratified_schedule = bool(args.pair_stratified_schedule)
    trainer.evidence_orbit_atomic_batches = bool(
        args.evidence_orbit_atomic_batches
    )
    trainer.evidence_orbit_blocks = orbit_blocks
    trainer.evidence_orbit_singles = orbit_singles
    trainer.evidence_orbit_seed = int(args.seed)
    trainer.length_grouped_batches = bool(args.length_grouped_batches)
    trainer.length_grouped_seed = int(args.seed)
    trainer.family_stratified_length_batches = bool(
        args.family_stratified_length_batches
    )
    trainer.family_stratified_length_seed = int(args.seed)
    trainer.family_stratified_window_steps = int(
        args.family_stratified_window_steps
    )
    trainer.media_homogeneous_batches = bool(args.media_homogeneous_batches)
    trainer.media_homogeneous_seed = int(args.seed)
    trainer.causal_gradient_firewall_batches = bool(
        args.causal_gradient_firewall_batches
    )
    trainer.causal_gradient_firewall_seed = int(args.seed)
    trainer.causal_gradient_firewall_length_bucketing = bool(
        args.causal_gradient_firewall_length_bucketing
    )
    trainer.causal_gradient_firewall_enabled = firewall_enabled
    trainer.causal_gradient_firewall_current_route = -1
    trainer.causal_gradient_firewall_micro_index = 0
    trainer.causal_gradient_firewall_first_route = -1
    trainer.evidence_pair_margin_weight = float(args.evidence_pair_margin_weight)
    trainer.evidence_pair_margin_target = float(args.evidence_pair_margin_target)
    trainer.evidence_pair_absolute_margin_weight = float(
        args.evidence_pair_absolute_margin_weight
    )
    trainer.evidence_pair_absolute_margin_target = float(
        args.evidence_pair_absolute_margin_target
    )
    trainer.evidence_pair_insufficient_margin_weight = float(
        args.evidence_pair_insufficient_margin_weight
    )
    trainer.evidence_pair_insufficient_margin_target = float(
        args.evidence_pair_insufficient_margin_target
    )
    trainer.evidence_orbit_consistency_weight = float(
        args.evidence_orbit_consistency_weight
    )
    trainer.evidence_orbit_floor_weight = float(
        args.evidence_orbit_floor_weight
    )
    trainer.evidence_orbit_margin_target = float(
        args.evidence_orbit_margin_target
    )
    protected_firewall_parameters: list[tuple[str, torch.nn.Parameter]] = []
    firewall_parameter_mode: str | None = None
    protected_firewall_tensor_count = 0
    protected_firewall_parameter_count = 0
    if firewall_enabled:
        if args.causal_gradient_firewall_state_path_only:
            protected_firewall_parameters = causal_state_path_firewall_parameters(
                model
            )
            firewall_parameter_mode = "visual_state_path_only"
        else:
            protected_firewall_parameters = causal_firewall_parameters(
                model, int(args.causal_gradient_firewall_last_layers)
            )
            firewall_parameter_mode = "last_language_layers"
        # Capture global counts before DeepSpeed replaces full parameters with
        # rank-local partitions. Recomputing numel after trainer.train() would
        # report only the local shard and is not a model-level quantity.
        protected_firewall_tensor_count = len(protected_firewall_parameters)
        protected_firewall_parameter_count = sum(
            parameter.numel() for _, parameter in protected_firewall_parameters
        )

        trainer.causal_gradient_firewall_hook_handles = [
            parameter.register_hook(
                lambda gradient: apply_causal_gradient_firewall(
                    trainer, gradient
                )
            )
            for _, parameter in protected_firewall_parameters
        ]
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                {
                    "causal_gradient_firewall_parameter_contract": {
                        "last_language_layers": int(
                            args.causal_gradient_firewall_last_layers
                        ),
                        "parameter_mode": firewall_parameter_mode,
                        "protected_tensors": protected_firewall_tensor_count,
                        "protected_parameters": protected_firewall_parameter_count,
                        "causal_route": "OST",
                        "anchor_route": "EXACTPACKET",
                        "optimizer_step_pair": "OST_then_EXACTPACKET",
                    }
                }
            )
    train_result = trainer.train()
    if args.skip_final_save:
        rank = int(os.environ.get("RANK", "0"))
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        telemetry = {
            "format": "iclr_spatial.sft_memory_probe.v1",
            "diagnostic_only": True,
            "checkpoint_materialized": False,
            "rank": rank,
            "global_step": int(trainer.state.global_step),
            "stop_after_step": int(args.stop_after_step),
            "per_device_batch_size": int(args.per_device_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "length_grouped_batches": bool(args.length_grouped_batches),
            "family_stratified_length_batches": bool(
                args.family_stratified_length_batches
            ),
            "family_stratified_window_steps": int(
                args.family_stratified_window_steps
            ),
            "cuda_device": device,
            "max_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device is not None else None
            ),
            "max_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device is not None else None
            ),
            "train_metrics": dict(train_result.metrics or {}),
            "causal_gradient_firewall": {
                "enabled": firewall_enabled,
                "parameter_mode": firewall_parameter_mode,
                "protected_tensors": protected_firewall_tensor_count,
                "protected_parameters": protected_firewall_parameter_count,
                "relation_route": "OST" if firewall_enabled else None,
                "answer_route": "EXACTPACKET" if firewall_enabled else None,
            },
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"memory_probe_rank{rank}.json").write_text(
            json.dumps(telemetry, indent=2, sort_keys=True) + "\n"
        )
        print({"memory_probe": telemetry})
        return
    final_dir = args.output / "final_model"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    if int(os.environ.get("RANK", "0")) == 0:
        (final_dir / "sft_contract.json").write_text(
            json.dumps(
                {
                    "finetune": finetune,
                    "base_model": str(args.model),
                    "base_model_identity": (
                        json.loads(
                            (args.model / "MODEL_IDENTITY.json").read_text()
                        )
                        if (args.model / "MODEL_IDENTITY.json").is_file()
                        else None
                    ),
                    "train_data": {
                        "path": str(args.train),
                        "sha256": file_sha256(args.train),
                        "rows": len(dataset),
                        "benchmark_category_selection": False,
                        "sample_reweighting": source_sample_reweighting,
                        "sample_loss_weight_distribution": (
                            source_sample_weight_distribution
                        ),
                    },
                    "optimization": {
                        "epochs": args.epochs,
                        "max_steps": args.max_steps,
                        "max_length": args.max_length,
                        "max_pixels": args.max_pixels,
                        "processor_image_size": processor_image_size,
                        "per_device_batch_size": args.per_device_batch_size,
                        "gradient_accumulation_steps": (
                            args.gradient_accumulation_steps
                        ),
                        "learning_rate": args.lr,
                        "lr_scheduler_type": args.lr_scheduler_type,
                        "warmup_ratio": args.warmup_ratio,
                        "weight_decay": args.weight_decay,
                        "initialization_seed": args.seed,
                        "trainer_seed": (
                            args.seed
                            if args.trainer_seed is None
                            else args.trainer_seed
                        ),
                        "data_seed": (
                            args.seed
                            if args.trainer_seed is None
                            else args.trainer_seed
                        ),
                        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                        "attn_implementation": args.attn_implementation,
                        "gradient_checkpointing": args.gradient_checkpointing,
                        "length_grouped_batches": args.length_grouped_batches,
                        "family_stratified_length_batches": (
                            args.family_stratified_length_batches
                        ),
                        "family_stratified_window_steps": (
                            args.family_stratified_window_steps
                        ),
                        "causal_gradient_firewall_batches": (
                            args.causal_gradient_firewall_batches
                        ),
                        "causal_gradient_firewall_last_layers": (
                            args.causal_gradient_firewall_last_layers
                        ),
                        "causal_gradient_firewall_state_path_only": (
                            args.causal_gradient_firewall_state_path_only
                        ),
                        "causal_gradient_firewall_length_bucketing": (
                            args.causal_gradient_firewall_length_bucketing
                        ),
                        "tf32": args.tf32,
                        "resume_from_checkpoint": None,
                    },
                    "weighted_adapter_fusion": False,
                    "lora": None,
                    "vision_encoder_frozen": not args.train_vision_encoder,
                    "visual_merger_frozen": args.freeze_visual_merger,
                    "freeze_language_layers_before": args.freeze_language_layers_before,
                    "assistant_only": True,
                    "turn_normalized": not args.token_normalized_loss,
                    "sample_normalized": args.sample_normalized_loss,
                    "sample_loss_weight_semantics": (
                        "absolute_gradient_multiplier_v1"
                    ),
                    "causal_gradient_firewall": {
                        "enabled": firewall_enabled,
                        "causal_family": "OST" if firewall_enabled else None,
                        "anchor_family": (
                            "EXACTPACKET" if firewall_enabled else None
                        ),
                        "last_language_layers": int(
                            args.causal_gradient_firewall_last_layers
                        ),
                        "parameter_mode": (
                            "visual_state_path_only"
                            if args.causal_gradient_firewall_state_path_only
                            else (
                                "last_language_layers"
                                if firewall_enabled
                                else None
                            )
                        ),
                        "protected_tensors": protected_firewall_tensor_count,
                        "protected_parameters": protected_firewall_parameter_count,
                        "optimizer_step_pair": (
                            "OST_then_EXACTPACKET" if firewall_enabled else None
                        ),
                        "length_bucketing_within_family": bool(
                            args.causal_gradient_firewall_length_bucketing
                        ),
                        "real_rows_duplicated": False,
                        "benchmark_category_routing": False,
                    },
                    "pair_atomic_batches": pair_atomic_stats,
                    "evidence_orbit_batches": orbit_stats,
                    "evidence_orbit_objective": {
                        "enabled": orbit_loss_enabled,
                        "consistency_weight": (
                            args.evidence_orbit_consistency_weight
                        ),
                        "floor_weight": args.evidence_orbit_floor_weight,
                        "margin_target": args.evidence_orbit_margin_target,
                        "alignment": (
                            "permutation_invariant_correct_vs_distractor_margin"
                        ),
                        "selection": False,
                        "sample_reweighting": False,
                    },
                    "evidence_pair_margin": {
                        "enabled": (
                            args.evidence_pair_margin_weight > 0
                            or args.evidence_pair_absolute_margin_weight > 0
                            or args.evidence_pair_insufficient_margin_weight > 0
                        ),
                        "weight": args.evidence_pair_margin_weight,
                        "target": args.evidence_pair_margin_target,
                        "absolute_weight": args.evidence_pair_absolute_margin_weight,
                        "absolute_target": args.evidence_pair_absolute_margin_target,
                        "insufficient_weight": args.evidence_pair_insufficient_margin_weight,
                        "insufficient_target": args.evidence_pair_insufficient_margin_target,
                        "status": "insufficient_vs_sufficient_first_divergent_token",
                        "requires_same_forward_pass": True,
                    },
                    "family_weights": family_weights,
                    "parameter_counts": stats,
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
        )
        print({"final_model": str(final_dir)})


if __name__ == "__main__":
    main()
