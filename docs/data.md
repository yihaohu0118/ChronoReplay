# Data preparation

The public entry point consumes **native preservation records** and the
**released OST training conversations**. It constructs the weighted replay
pool directly, without requiring historical `cdrop-dir`, `parent-dir`, or
internally named readiness reports.

## Data availability

OST episodes are available from the
[official release](https://huggingface.co/datasets/rbler/OST-Bench).
Obtain original media from its sources under the relevant terms. The
preservation pool combines source-annotated categorical questions with
ARKitScenes camera-rotation and corrected-camera geometry targets.

The original preservation selection involved training-source visual-utility
filtering and matched-view reconstruction. The original training pool's
selected row IDs, order, weights, supervision masks, and native text hashes
are included in [the frozen membership manifest](../data_manifests/README.md).
**Converted source records and media are not bundled.** Downloading source
datasets alone does not perform those conversions. The manifest can verify
and reconstruct the frozen pool from converted records; the portable
adapter below also supports a new explicitly specified source pool.

The historical converters are retained for users who already have those
inputs:

1. `build_workfirst_multiview_preservation.py`: a 15,180-row C-drop pool plus
   the original 40,339-row parent, with matching source reports, reconstruct
   the three matched views and the 33,180 preservation rows.
2. `build_ost_official_train_multiturn_sft.py`: the native OST train
   annotations, decoded training media, and held-out scene identifiers
   produce causal episodes.
3. `compose_ost_official_train_release.py`: those two pools produce the
   original format consumed by the trainer.

These are input contracts, not public download links.

## Reconstruct the frozen pool

Given locally obtained converted-source records with their original IDs:

```bash
python -m scripts.materialize_frozen \
  --candidate data/converted/preservation.parquet \
  --candidate data/converted/ost_episodes.parquet \
  --media-root data/media \
  --output data/replay
```

Inputs may be JSONL or Parquet. Paths in native messages must resolve within
the local media root. The script checks all selected IDs, text hashes,
supervision masks, and image counts, restores original row order and weights,
decodes every referenced image, and writes the trainer's ready data format.
It does not regenerate missing source records or infer their annotations.

## Prepare a new source pool

Use the portable route below for a supplied/exported source pool or a new
controlled study.

## Directory layout

```text
data/
  media/                          # shared root for relative image paths
    preservation/...
    ost_train/...
  preservation.jsonl
  OST_bench_train.json
  heldout_scenes.json
  replay/                         # generated; must not already exist
```

All `image` and OST `image_paths` entries resolve underneath `--media-root`.
Extract/mount media so those paths match the native records. URL fetching and
paths outside that root are rejected.

## Preservation schema

Each JSONL line (or Parquet row) contains:

| Field | Meaning |
|---|---|
| `uid` | Unique serialized-example identifier |
| `source_id` | Original dataset/release identifier |
| `scene_id` | Canonical physical scene ID used across all train/evaluation sources |
| `source_split` | `train` or `training` |
| `semantic_id` | Source question identity shared by its matched visual views |
| `family` | `categorical` or `numeric` |
| `view` | `dense`, `context`, `long`, or a distinct single-view identifier |
| `messages` | Native `user`/`assistant` messages, with typed text/image blocks |

Example structure:

```json
{
  "uid": "source-question-dense",
  "source_id": "your-licensed-source",
  "scene_id": "canonical-train-scene",
  "source_split": "train",
  "semantic_id": "source-question",
  "family": "categorical",
  "view": "dense",
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "image": "preservation/frame.jpg"},
      {"type": "text", "text": "The original source question and options."}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": "The original source answer."}
    ]}
  ]
}
```

The example describes the schema; it is not an experimental record.
Preserve source questions, answers, units, candidate order, and observed
images when exporting real records. A categorical source question has one
view or the three `dense/context/long` views, all with the same target.
The builder divides its mass across the views. Numeric examples retain
their native numeric answer rather than an option letter.

## OST conversations and scene separation

Use the source `OST_bench_train.json` list unchanged. Each scene has
`scan_id`, `system_prompt`, and ordered `user_message` turns containing
`image_paths`, `prompt`, `answer`, and `type`.

The builder repeats the source system prompt at each user turn, keeps the
original order, and excludes `type=None` observation turns from direct loss
while retaining them in context. Context-only episodes are excluded.

`heldout_scenes.json` is a nonempty JSON list of canonical physical-scene IDs.
Build it from the evaluation manifests **before** selecting training data.
Resolve source aliases to the same physical identity; equal strings are
checked by the adapter, not inferred from visual content.

## Build the pool

```bash
python -m scripts.prepare_data \
  --preservation data/preservation.jsonl \
  --ost-train data/OST_bench_train.json \
  --heldout-scenes data/heldout_scenes.json \
  --media-root data/media \
  --alpha 0.5 \
  --output data/replay
```

The output contains `train.parquet`, an input/hash/count `report.json`,
`MEDIA_DECODE_READY`, and `DATA_READY`. All unique images are decoded before
readiness is written.

`--expect-paper-counts` additionally requires 30,635 categorical rows
(12,635 semantic groups), 2,545 numeric rows, and 6,381 supervised OST
episodes / 45,006 supervised trajectory turns. Matching counts does not
establish matching source membership.

At `alpha=0.5`, categorical, numeric, and trajectory supervision each receive
one third of the total loss mass. Categorical mass is normalized over
semantic groups, numeric mass over examples, and trajectory mass over
supervised turns. The final mean serialized-example weight is one.
