# Release validation

This document records software checks, not benchmark results.

## Homepage publication checks (2026-10-20)

- Twenty-six targeted public-entry-point and data-pipeline tests passed locally
  on Python 3.11.5. The full CPU suite is also configured in GitHub Actions.
- The homepage was checked in an isolated browser at 1440-pixel desktop and
  390-pixel mobile widths, with no horizontal overflow or JavaScript errors.
- Result switching, figure enlargement, BibTeX copying, local links, and the
  JavaScript-disabled results table passed browser checks.
- The displayed main scores were checked against the manuscript's numeric macros.
- The linked manuscript was recompiled as a 38-page PDF. The first page and
  reproducibility statement contain verified Code and Project Homepage links.
- Publication files were scanned for credential patterns before uploading.

## Completed in the packaging review

- Original archive contents were inspected and compared with the unpacked sources.
- Missing UCS prediction/state-snapshot merge implementations were restored.
- The OST adapter delegates metrics to a local official evaluator rather
  than redistributing the upstream implementation. 252 synthetic conformance
  checks agreed with the existing project adapter, including category
  aggregation and its zero-target convention.
- Historical experiment result JSON, paper-result tables, and their numerical
  alignment script were removed.
- A synthetic CPU pipeline decoded local images, preserved native conversation
  targets and context-only turns, and produced a family-balanced Parquet pool.
- Eight-GPU and four-GPU training command construction was checked for the
  same global batch and explicitly specified recipe.
- All three public benchmark entry points were checked in inference/scoring
  dry-run mode without weights.
- The current test suite passed locally; see `PACKAGE_AUDIT.json` for the
  packaged test count and validation status.
- The combined CPU/model/training/OST requirement pins resolved successfully
  with an isolated Linux/Python 3.11 `pip install --dry-run --ignore-installed`.
  FlashAttention is a separate build/install step and was checked by import
  in the existing project environment.
- Frozen training membership was exported from the original pool: 39,561
  unique ordered IDs, positive weights with mean one, and native text hashes.
- An existing Linux/Python 3.11 project environment was inspected:
  PyTorch 2.8.0+cu126, Transformers 4.57.3, DeepSpeed 0.19.2,
  FlashAttention 2.8.3, vLLM 0.11.0, Qwen-VL-Utils 0.0.14, and
  Accelerate 1.14.0. Qwen3-VL, Trainer, DeepSpeed, FlashAttention, and vLLM
  imported successfully.

## Separate runtime acceptance

The pinned clean-install recipe has not been used here to retrain an 8B
model or rerun the three full benchmarks. Import success is not a CUDA
kernel or full training test. On the target Linux GPU host, run
`scripts.check_environment --strict-versions --require-cuda` with the
appropriate profile before full inference/training.

Exact numerical replication additionally requires the selected preservation
records, archived benchmark revisions, and frozen scene manifests described
in [data.md](data.md) and [evaluation.md](evaluation.md).
