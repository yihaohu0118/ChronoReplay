# Evaluation

`python -m scripts.evaluate` separates prediction from scoring and uses the
existing causal readers and benchmark scorers. Run commands from the
repository root. `--dry-run` prints all commands without loading weights or
starting a GPU process.

## Prepare benchmark inputs

| Benchmark | Official source | Required local inputs |
|---|---|---|
| OVO-S | [OVO-S-Bench](https://github.com/InternLM/OVO-S-Bench) | Annotation JSON/Parquet, source videos, image options if present |
| OST | [OST-Bench](https://github.com/InternRobotics/OST-Bench) | `OST_bench.json`, image tree, scene-disjoint manifest |
| UCS | [UCS-Bench](https://github.com/cocowy1/UCS-Bench) | Per-video `*-mcq.json` files and videos |

Keep benchmark releases fixed. The paper used an archived 1,695-question
OVO-S population; an upstream revision with a different number of rows is a
different evaluation. This code package does not contain that annotation
snapshot or the exact paper OST scene manifest. For numerical paper
replication, use the archived identities rather than silently substituting a
current download.

Flatten UCS annotations with:

```bash
python -m scripts.materialize_ucs_bench_questions \
  --qa-dir data/UCS/qa \
  --output data/UCS/questions.jsonl \
  --report data/UCS/input-report.json
```

For a new training corpus, construct an OST scene exclusion manifest before
inference:

```bash
python -m data_pipeline.stages.build_scene_disjoint_eval_manifest \
  --benchmark OST=data/OST/OST_bench.json=scan_id \
  --training-source replay=data/replay/train.parquet=scene_id \
  --output data/OST/manifest.json
```

The scene identifiers must resolve the same physical sources across datasets.
A manifest created from a new pool does not establish the paper's exact test
population.

## OVO-S

Use the Transformers environment. Pin the same base processor for matched
model comparisons:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate \
  --benchmark ovo --stage infer \
  --model outputs/chronoreplay/final_model \
  --processor models/Qwen3-VL-8B-Instruct \
  --annotation data/OVO-S/annotations.parquet \
  --media-root data/OVO-S/videos --output outputs/eval-ovo

python -m scripts.evaluate \
  --benchmark ovo --stage score \
  --model outputs/chronoreplay/final_model \
  --processor models/Qwen3-VL-8B-Instruct \
  --annotation data/OVO-S/annotations.parquet \
  --output outputs/eval-ovo
```

This uses uniform 128-frame access, greedy native answers, an eight-token
ceiling, and batch size two. Add `--policy adaptive` to **both** commands
and choose a different output directory for the adaptive-access condition.
The scorer reports the official family-macro aggregation.

## OST

Use the OST/vLLM environment. Obtain the official evaluator directly from
the benchmark repository; its metric implementation is loaded locally and
is not redistributed in this archive:

```bash
git clone https://github.com/InternRobotics/OST-Bench.git external/OST-Bench
```

Record and retain the upstream commit for repeatable scoring. The score
report records the evaluator file's SHA-256.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate \
  --benchmark ost --stage infer \
  --model outputs/chronoreplay/final_model \
  --annotation data/OST/OST_bench.json --media-root data/OST/images \
  --manifest data/OST/manifest.json --output outputs/eval-ost

python -m scripts.evaluate \
  --benchmark ost --stage score \
  --model outputs/chronoreplay/final_model \
  --annotation data/OST/OST_bench.json \
  --manifest data/OST/manifest.json --output outputs/eval-ost \
  --ost-evaluator external/OST-Bench/evaluation/OST_evaluator.py
```

The reader preserves multi-turn causal order and writes per-scene response
files. Decoding is greedy, with 512 output tokens and one sample.
For the historical shell launcher, set `CHRONOREPLAY_OST_EVALUATOR` to the
same official evaluator path.

## UCS

Use the Transformers environment:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate \
  --benchmark ucs --stage infer \
  --model outputs/chronoreplay/final_model \
  --processor models/Qwen3-VL-8B-Instruct \
  --annotation data/UCS/questions.jsonl --media-root data/UCS/videos \
  --output outputs/eval-ucs

python -m scripts.evaluate \
  --benchmark ucs --stage score \
  --model outputs/chronoreplay/final_model \
  --processor models/Qwen3-VL-8B-Instruct \
  --annotation data/UCS/questions.jsonl --output outputs/eval-ucs
```

This uses timestamp-causal uniform 128-frame access, batch size one, and
eight-token greedy native answers. Scoring requires all 8,114 questions.
The merger rejects duplicate or incomplete IDs. It is included together
with the historical state-snapshot merger used by the older launcher.

## Shards, resume, and outputs

For independent inference shards, use the same output root and
`--shard-count N`, with a distinct `--shard-index 0 ... N-1` on disjoint
allocated GPUs. Pass `--shard-count N` to scoring after every shard finishes.
`--resume` requires the same stored configuration.

```text
outputs/eval-<benchmark>/
  run-00000.json          # input/protocol identity
  predictions/           # generated responses or per-scene files
  merged.json            # UCS only
  scores.json            # generated by the scoring stage
```

These files are generated by the user. Historical experiment outputs are
excluded from the code archive.
