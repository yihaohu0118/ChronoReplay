# Training

Prepare the [GPU environment](installation.md) and [data](data.md). All
commands run at the repository root.

## Configuration

[`configs/chronoreplay_train.json`](../configs/chronoreplay_train.json)
explicitly sets the published recipe:

| Setting | Value |
|---|---|
| Initialization | Qwen3-VL-8B-Instruct, pinned official revision |
| Trainable parameters | Language model and visual merger; frozen visual encoder |
| Epochs / step limit | One epoch; `max_steps=-1` |
| Global batch | 32 |
| Microbatch | Two examples per GPU |
| Learning rate / scheduler | `2e-6`, cosine, 3% warmup |
| Weight decay | `0.01` |
| Maximum text / visual item budget | 32,768 tokens / 200,704 processed pixels |
| Precision / memory | bf16, gradient checkpointing, GPU-only ZeRO-3 |
| Seed / Trainer seed | 42 / 42 |
| Loss | Sample-normalized weighted native supervision |

The wrapper passes these options explicitly to the historical trainer; it
does not rely on that trainer's research defaults. Four GPUs use gradient
accumulation four; eight GPUs use accumulation two.

## Prepare token-length checks

After decoding all images during data preparation, check the model processor:

```bash
python -m scripts.audit_sft_processor_lengths \
  --train data/replay/train.parquet \
  --model models/Qwen3-VL-8B-Instruct \
  --max-length 32768 --max-pixels 200704 --all-rows \
  --report data/replay/processor-audit.json
```

This may take time on a full corpus but does not load model weights.

## Launch

Inspect the exact command first:

```bash
python -m scripts.train \
  --model models/Qwen3-VL-8B-Instruct \
  --data data/replay --output outputs/chronoreplay \
  --gpus 8 --dry-run
```

Then allocate the GPUs to this job and run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m scripts.train \
  --model models/Qwen3-VL-8B-Instruct \
  --data data/replay --output outputs/chronoreplay --gpus 8
```

The trainer saves a self-contained `outputs/chronoreplay/final_model/`.
Use `--gpus 4` with four visible GPUs for the four-GPU topology.
Use `--allow-custom-data` explicitly for a new pool with different row counts.
Choose a fresh output directory; these commands do not resume from a prior
fine-tuned checkpoint.

## Controlled variants

The native recipe retains chronological episode order. The existing
`data_pipeline.stages.build_causal_trajectory_order_control` and
`data_pipeline.stages.build_common_interface_wrapper_control` modules expose
their input contracts with `--help`. Run each control from the same official
base, with its own output directory and the same budget. The code archive
does not include historical control scores.
