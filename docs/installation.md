# Installation

Run commands from the repository root. The pinned GPU recipe targets **Linux
x86_64, Python 3.11, CUDA 12.8, and Ampere or newer NVIDIA GPUs**.

## CPU data preparation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.smoke_test --output outputs/cpu-demo
```

The full unit-test suite also imports PyTorch:

```bash
python -m pip install torch==2.8.0
python -m pytest -q -p no:cacheprovider
```

## Training and Transformers inference

Use a separate environment from OST/vLLM to keep optional dependencies isolated.
Install PyTorch before compiling FlashAttention or DeepSpeed.

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install packaging==25.0 ninja==1.13.0 setuptools wheel
python -m pip install -r requirements-train.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip check
python -m scripts.check_environment --profile train --strict-versions
```

OVO-S and UCS inference need `requirements-model.txt` plus FlashAttention,
without DeepSpeed. FlashAttention compilation requires a compatible CUDA
development toolkit and host compiler; match its wheel to the installed
Python, PyTorch, CUDA, and C++ ABI.

On a GPU allocated to your job, check actual kernels before loading an 8B model:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.check_environment \
  --profile train --strict-versions --require-cuda
```

## OST / vLLM

```bash
python3.11 -m venv .venv-ost
source .venv-ost/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-ost.txt
python -m pip check
CUDA_VISIBLE_DEVICES=0 python -m scripts.check_environment \
  --profile ost --strict-versions --require-cuda
```

vLLM 0.11.0 declares PyTorch 2.8.0 and torchvision 0.23.0 in its
[CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt).
Transformers 4.57.3 supplies the Qwen3-VL implementation. FlashAttention 2.8.3
installation requirements are documented by its
[maintainers](https://pypi.org/project/flash-attn/2.8.3/).

The model-library versions were also inspected in an existing project
environment, where Qwen3-VL, Trainer, DeepSpeed, FlashAttention, and vLLM
imports succeeded. That environment used PyTorch's CUDA 12.6 build; the
commands above use the CUDA 12.8 wheel family associated with the standard
vLLM installation. These are pinned release installation recipes, not an
exported historical training environment. CPU validation and GPU acceptance are reported
separately in [validation.md](validation.md); do not infer that a GPU training
run was repeated when only a command or CPU test was checked.

## Model initialization

Obtain the official base snapshot, including processor and tokenizer files:

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir models/Qwen3-VL-8B-Instruct
```

The `hf` CLI is installed with `huggingface-hub` through Transformers.
Model downloads and source datasets remain subject to their original terms.
Training starts fresh from this base. Evaluation accepts a separate local
fine-tuned checkpoint and an explicitly pinned base processor.
