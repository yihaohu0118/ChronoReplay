"""Portable inference and scoring entry point for the three ChronoReplay benchmarks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def commands(a: argparse.Namespace) -> list[list[str]]:
    py = [sys.executable, "-m"]
    predictions = a.output / "predictions"
    shard = predictions / f"part-{a.shard_index:05d}.json"
    if a.stage == "infer":
        if a.benchmark == "ost":
            return [py + ["benchmark.ost.inference", "--model", str(a.model),
                "--annotation", str(a.annotation), "--image-root", str(a.media_root),
                "--manifest", str(a.manifest), "--output", str(predictions),
                "--shard-index", str(a.shard_index), "--shard-count", str(a.shard_count),
                "--chunk-size", "48", "--max-new-tokens", "512",
                "--max-model-len", "32768", "--max-images", "64",
                "--tensor-parallel-size", "1"]]
        cmd = py + ["synthesis.eval_ovos_uniform", "--model", str(a.model),
            "--processor", str(a.processor), "--annotation", str(a.annotation),
            "--video-base", str(a.media_root), "--out", str(shard),
            "--nframes", "128", "--frame-size", "448", "--prompt-style", "default",
            "--max-new-tokens", "8", "--batch-size", "2" if a.benchmark == "ovo" else "1",
            "--frame-policy", "semantic_adaptive" if a.policy == "adaptive" else "uniform",
            "--semantic-uniform-nframes", "128", "--semantic-recent-nframes", "16",
            "--semantic-reference-nframes", "8", "--state-compiler", "none",
            "--answer-parser", "native_letter", "--decode-mode", "generate",
            "--shard-rank", str(a.shard_index), "--shard-world-size", str(a.shard_count),
            "--save-raw-reply"]
        if a.resume:
            cmd.append("--resume")
        return [cmd]
    parts = [predictions / f"part-{i:05d}.json" for i in range(a.shard_count)]
    if a.benchmark == "ovo":
        return [py + ["scripts.score_ovos_official_macro", *map(str, parts),
            "--annotation", str(a.annotation), "--out", str(a.output / "scores.json"),
            "--model-path", str(a.model), "--processor-path", str(a.processor),
            "--inference-protocol", "model_only",
            "--frame-sampling-protocol", "semantic_adaptive_causal" if a.policy == "adaptive" else "uniform_128",
            "--requested-default-nframes", "128", "--prompt-protocol", "default"]]
    if a.benchmark == "ucs":
        merge = py + ["scripts.merge_ucs_causal_shards"]
        for part in parts:
            merge += ["--shard", str(part)]
        merge += ["--expected-rows", "8114", "--output", str(a.output / "merged.json")]
        return [merge, py + ["scripts.evaluate_ucs_bench",
            "--predictions", str(a.output / "merged.json"), "--questions", str(a.annotation),
            "--protocol", "timestamp_causal_model_only", "--model", str(a.model),
            "--expected-total", "8114", "--output", str(a.output / "scores.json")]]
    return [py + ["benchmark.ost.score", "--predictions", str(predictions),
        "--annotation", str(a.annotation), "--manifest", str(a.manifest),
        "--model-path", str(a.model), "--output", str(a.output / "scores.json")]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["ovo", "ost", "ucs"], required=True)
    p.add_argument("--stage", choices=["infer", "score"], required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--processor", type=Path, help="Pin the same official-base processor for matched runs.")
    p.add_argument("--annotation", type=Path, required=True)
    p.add_argument("--media-root", type=Path)
    p.add_argument("--manifest", type=Path, help="OST scene-disjoint evaluation manifest.")
    p.add_argument("--ost-evaluator", type=Path,
                   default=os.environ.get("CHRONOREPLAY_OST_EVALUATOR"),
                   help="Local official OST evaluation/OST_evaluator.py.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--policy", choices=["uniform", "adaptive"], default="uniform")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not 0 <= a.shard_index < a.shard_count:
        p.error("Require 0 <= shard-index < shard-count.")
    if a.policy == "adaptive" and a.benchmark != "ovo":
        p.error("Adaptive access is a separate OVO-S protocol.")
    if a.benchmark == "ost" and not a.manifest:
        p.error("--manifest is required for OST.")
    if a.stage == "infer" and not a.media_root:
        p.error("--media-root is required for inference.")
    for key in ("model", "processor", "annotation", "media_root", "manifest", "output", "ost_evaluator"):
        value = getattr(a, key)
        if value is not None:
            setattr(a, key, value.expanduser().resolve())
    a.processor = a.processor or a.model
    cmds = commands(a)
    for cmd in cmds:
        print(shlex.join(cmd), flush=True)
    if a.dry_run:
        return
    if a.benchmark == "ost" and a.stage == "score":
        if not a.ost_evaluator or not a.ost_evaluator.is_file():
            p.error("OST scoring requires --ost-evaluator; see docs/evaluation.md.")
    for path in [a.annotation] + ([a.manifest] if a.manifest else []):
        if not path.is_file():
            p.error(f"Missing input: {path}")
    if a.stage == "infer":
        if not (a.model / "config.json").is_file() or not a.media_root.is_dir():
            p.error("Missing local checkpoint or media root.")
        if not (a.processor / "preprocessor_config.json").is_file():
            p.error("Missing processor files; pass --processor with a complete snapshot.")
    predictions = a.output / "predictions"
    manifest = a.output / f"run-{a.shard_index:05d}.json"
    protocol = {
        "benchmark": a.benchmark, "policy": a.policy, "model": str(a.model),
        "processor": str(a.processor), "annotation": str(a.annotation),
        "media_root": str(a.media_root), "manifest": str(a.manifest),
        "shard_index": a.shard_index, "shard_count": a.shard_count,
    }
    if a.stage == "infer":
        if manifest.exists():
            if not a.resume or json.loads(manifest.read_text()) != protocol:
                p.error("Existing run: use --resume with identical inputs and settings.")
        predictions.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(protocol, indent=2) + "\n")
    else:
        for i in range(a.shard_count):
            stored = a.output / f"run-{i:05d}.json"
            if not stored.is_file():
                p.error(f"Missing inference manifest: {stored}")
            old = json.loads(stored.read_text())
            for key in ("benchmark", "policy", "model", "processor", "annotation", "shard_count"):
                if old.get(key) != protocol[key]:
                    p.error(f"Scoring/inference mismatch: {key}")
    env = dict(os.environ)
    if a.ost_evaluator:
        env["CHRONOREPLAY_OST_EVALUATOR"] = str(a.ost_evaluator)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for cmd in cmds:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
