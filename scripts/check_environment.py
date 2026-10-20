"""Check installed dependencies and, when requested, local CUDA execution."""
from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def pins(path: Path) -> dict[str, str]:
    found = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("-r "):
            found.update(pins(path.parent / line[3:].strip()))
        elif line and not line.startswith("#") and "==" in line:
            name, version = line.split("==")
            found[name.strip()] = version.strip()
    return found


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=["cpu", "model", "train", "ost"], default="cpu")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--strict-versions", action="store_true")
    a = p.parse_args()
    filename = "requirements.txt" if a.profile == "cpu" else f"requirements-{a.profile}.txt"
    expected = pins(ROOT / filename)
    if a.profile in {"model", "train"}:
        expected["flash-attn"] = "2.8.3"
    installed, errors, differences = {}, [], []
    for name, version in expected.items():
        try:
            current = metadata.version(name)
            installed[name] = current
            if current.split("+")[0] != version:
                differences.append(f"{name}: installed={current}, recipe={version}")
        except metadata.PackageNotFoundError:
            errors.append(f"Missing package: {name}")
    try:
        for name in ("numpy", "pandas", "pyarrow", "PIL.Image"):
            importlib.import_module(name)
        if a.profile != "cpu":
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            importlib.import_module("torchvision")
        if a.profile in {"model", "train"}:
            importlib.import_module("flash_attn")
        if a.profile == "train":
            importlib.import_module("deepspeed")
        if a.profile == "ost":
            importlib.import_module("vllm")
    except (ImportError, OSError, RuntimeError) as exc:
        errors.append(f"Import check: {exc}")
    cuda = None
    if a.require_cuda:
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA is unavailable.")
            else:
                cuda = {"runtime": torch.version.cuda, "device": torch.cuda.get_device_name()}
                x = torch.ones((16, 16), dtype=torch.bfloat16, device="cuda")
                assert torch.isfinite(x @ x).all().item()
                if a.profile in {"model", "train"}:
                    from flash_attn import flash_attn_func
                    q = torch.zeros((1, 8, 2, 64), dtype=torch.bfloat16, device="cuda")
                    assert torch.isfinite(flash_attn_func(q, q, q)).all().item()
                torch.cuda.synchronize()
        except (ImportError, RuntimeError, AssertionError) as exc:
            errors.append(f"CUDA kernel check: {exc}")
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if check.returncode:
        errors.append("pip check: " + (check.stdout + check.stderr).strip())
    if a.strict_versions:
        errors += differences
        if sys.version_info[:2] != (3, 11):
            errors.append("The pinned release recipe targets Python 3.11.")
    print(json.dumps({
        "profile": a.profile, "python": platform.python_version(),
        "platform": platform.platform(), "packages": installed,
        "version_differences": differences, "cuda": cuda, "errors": errors,
        "ok": not errors,
    }, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
