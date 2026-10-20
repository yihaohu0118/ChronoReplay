"""Launch the documented ChronoReplay recipe from a local official-base snapshot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(model: Path, data: Path, output: Path, gpus: int,
            config: dict, port: int = 29500) -> list[str]:
    if gpus not in (4, 8):
        raise ValueError("The documented recipe uses four or eight GPUs.")
    micro = config["per_device_batch_size"]
    batch = config["global_batch_size"]
    if batch % (gpus * micro):
        raise ValueError("Global batch must be divisible by GPUs × microbatch.")
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
        "--node-rank=0", "--master-addr=127.0.0.1", f"--master-port={port}",
        f"--nproc-per-node={gpus}", str(ROOT / "scripts/train_qwen3vl_sft.py"),
        "--model", str(model), "--train", str(data / "train.parquet"),
        "--output", str(output), "--deepspeed",
        str(ROOT / "configs/deepspeed_zero3_sft.json"),
        "--gradient-accumulation-steps", str(batch // (gpus * micro)),
    ]
    metadata = {"global_batch_size", "model_id", "model_revision"}
    for key, value in config.items():
        if key in metadata:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            elif key in {"dataloader_pin_memory", "gradient_checkpointing"}:
                cmd.append("--no-" + key.replace("_", "-"))
        else:
            cmd.extend([flag, str(value)])
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", type=int, choices=[4, 8], default=8)
    p.add_argument("--config", type=Path, default=ROOT / "configs/chronoreplay_train.json")
    p.add_argument("--port", type=int, default=29500)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command without importing torch or starting training.")
    p.add_argument("--allow-custom-data", action="store_true",
                   help="Permit a new pool whose counts differ from the paper recipe.")
    p.add_argument("--processor-audit", type=Path,
                   help="Defaults to DATA/processor-audit.json.")
    a = p.parse_args()
    if not 1024 <= a.port <= 65535:
        p.error("--port must be between 1024 and 65535")
    config = json.loads(a.config.read_text())
    model, data, output = (v.expanduser().resolve() for v in (a.model, a.data, a.output))
    cmd = command(model, data, output, a.gpus, config, a.port)
    print(shlex.join(cmd), flush=True)
    if a.dry_run:
        return
    for path in (model / "config.json", data / "train.parquet",
                 data / "report.json", data / "DATA_READY", data / "MEDIA_DECODE_READY"):
        if not path.is_file():
            p.error(f"Missing input: {path}")
    if output.exists():
        p.error(f"Choose a new output directory: {output}")
    report = json.loads((data / "report.json").read_text())
    if not report.get("ok"):
        p.error("Data preparation did not complete successfully.")
    from data_pipeline.pure_vlm_utils import sha256
    digest = sha256(data / "train.parquet")
    if report.get("train_sha256") != digest:
        p.error("Training data changed since its preparation report.")
    audit_path = a.processor_audit or data / "processor-audit.json"
    if not audit_path.is_file():
        p.error("Run the processor-length audit in docs/training.md first.")
    audit = json.loads(audit_path.read_text())
    if not audit.get("ok") or audit.get("train_sha256") != digest or audit.get("violations"):
        p.error("Processor audit must pass and refer to the same training data.")
    if not a.allow_custom_data and report.get("rows") != 39561:
        p.error("Expected 39,561 paper-recipe rows; use --allow-custom-data for a new pool.")
    ds = json.loads((ROOT / "configs/deepspeed_zero3_sft.json").read_text())
    zero = ds["zero_optimization"]
    if zero["stage"] != 3 or any(zero.get(k) for k in ("offload_param", "offload_optimizer")):
        p.error("This recipe requires GPU-only ZeRO-3.")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
