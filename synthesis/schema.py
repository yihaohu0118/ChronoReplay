"""Data schema for synthetic spatial-video training items.

Task taxonomy fields are opaque metadata used for analysis. Training policies,
action availability, sampling and rewards must use explicit semantic contract
fields rather than values from that taxonomy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SynthItem:
    # --- portable annotation fields ---
    id: str = ""
    source_dataset: str = ""          # e.g. "synth_floorplan", "habitat_hm3d", "arkitscenes"
    video_id: str = ""
    video_path: str = ""              # relative to video_base_dir at train time
    level: int = 0                    # 1..4
    task_main_category: str = ""
    task_subcategory: str = ""
    task_type_name: str = ""
    question: str = ""
    options: Dict[str, str] = field(default_factory=dict)   # letter -> text or data-URI image
    answers: List[str] = field(default_factory=list)        # one letter per query_time
    query_times: List[float] = field(default_factory=list)  # seconds
    evidence_times: List[List[float]] = field(default_factory=list)  # [[start, end], ...]

    # --- synthesis-only metadata (dropped in strict export) ---
    synth_meta: Dict[str, Any] = field(default_factory=dict)
    # Common synth_meta keys:
    #   "generator":        which generator produced the item
    #   "world_state":      serialized ground-truth world state used for verification
    #   "grounded_cot":     verified reasoning trace for SFT (str)
    #   "cot_verified":     bool, whether every spatial claim passed GT verification
    #   "text_probe":       result dict of the text-shortcut filter
    #   "difficulty":       generator-specific difficulty knobs

    def finalize_id(self) -> None:
        """Derive a stable content-hash id and main category if unassigned."""
        if not self.task_main_category:
            parts = self.task_subcategory.split(".")
            self.task_main_category = ".".join(parts[:2])
        if self.id:
            return
        payload = json.dumps(
            [self.source_dataset, self.video_id, self.task_subcategory,
             self.question, sorted(self.options.items()), self.query_times],
            sort_keys=True, default=str,
        )
        digest = hashlib.sha1(payload.encode()).hexdigest()[:12]
        self.id = f"synth_{self.task_subcategory}_{digest}"

    def validate(self) -> List[str]:
        """Return a list of problems; empty list means the item is well-formed."""
        problems = []
        if not self.question.strip():
            problems.append("empty question")
        if len(self.options) < 2:
            problems.append("fewer than 2 options")
        if not self.query_times:
            problems.append("no query_times")
        if len(self.answers) != len(self.query_times):
            problems.append("answers/query_times length mismatch")
        letters = set(self.options.keys())
        for ans in self.answers:
            if ans not in letters:
                problems.append(f"answer {ans!r} not among options {sorted(letters)}")
        if self.level not in (1, 2, 3, 4):
            problems.append(f"bad level {self.level}")
        for span in self.evidence_times:
            if len(span) != 2 or span[0] > span[1]:
                problems.append(f"bad evidence span {span}")
            elif self.query_times and span[1] > max(self.query_times) + 1e-6:
                problems.append(f"evidence span {span} ends after query_time (violates streaming)")
        return problems

    def to_annotation(self, strict: bool = False) -> Dict[str, Any]:
        """Dict in the exact shape annotation_utils/load_annotations expects."""
        d = asdict(self)
        # Drop null/empty option slots before export. Parquet struct/map unification
        # otherwise pads shorter MCQs (e.g. binary A/B) with C/D=None.
        opts = d.get("options") or {}
        if isinstance(opts, dict):
            d["options"] = {
                str(k): v for k, v in opts.items()
                if v is not None and str(v).strip() and str(v).strip().lower() != "none"
            }
        if strict:
            d.pop("synth_meta", None)
        else:
            # parquet map columns cannot hold arbitrary nesting reliably; store as JSON
            d["synth_meta"] = json.dumps(d.get("synth_meta", {}), default=str, ensure_ascii=False)
            # Same for options: variable-key dicts get schema-padded with nulls in parquet.
            d["options"] = json.dumps(d.get("options", {}), default=str, ensure_ascii=False)
        return d


def items_to_records(items: List[SynthItem], strict: bool = False) -> List[Dict[str, Any]]:
    records = []
    for item in items:
        item.finalize_id()
        problems = item.validate()
        if problems:
            raise ValueError(f"invalid item {item.id}: {problems}")
        records.append(item.to_annotation(strict=strict))
    return records
