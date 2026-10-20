# Third-party material

## OVO-S

The OVO-S adapter files in `benchmark/ovo/` retain the included
[MIT license](benchmark/ovo/LICENSE). Upstream:
[InternLM/OVO-S-Bench](https://github.com/InternLM/OVO-S-Bench).
Local adapters add causal-access and prediction-format integration.

## OST

`benchmark/ost/score.py` validates prediction coverage and calls a
user-supplied local `evaluation/OST_evaluator.py` from
[InternRobotics/OST-Bench](https://github.com/InternRobotics/OST-Bench).
The upstream metric implementation is not redistributed. Obtain it from
the original distributor under its applicable terms, following
[the evaluation guide](docs/evaluation.md#ost).

## Model and data resources

Qwen weights, source media, benchmark annotations, and converted training
records are not redistributed in this code archive. Obtain each resource
from its original distributor and retain its attribution and terms.

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [OST-Bench data](https://huggingface.co/datasets/rbler/OST-Bench)
- [UCS-Bench / DirectMe](https://github.com/cocowy1/UCS-Bench)
- [ARKitScenes](https://github.com/apple/ARKitScenes)

## Documentation structure

The overview/preparation/evaluation organization was informed by
[OneStreamer](https://github.com/MCG-NJU/OneStreamer). No OneStreamer
model source or benchmark output was copied into this package.
