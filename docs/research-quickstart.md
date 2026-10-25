# Research quickstart

Run these commands from the Relay Lab repository root after completing
[local setup](../README.md#run-locally), including the running SpacetimeDB
service, published module and built subscription bridge. Use new output paths
for each experiment; keep failed runs and their original evidence.

## Prepare the local Qwen runtime

The tested inference host is an Apple Silicon Mac. Python 3.13 and the isolated
MLX environment below are separate from Torch/LeRobot. These instructions do
not install a Linux Qwen inference service. Latent-only diagnostics do not need
Qwen; `--planner llm` and Studio's text modes do.

```sh
uv venv --python 3.13 .venv-llm
uv pip install --python .venv-llm/bin/python -r requirements-llm.txt
export HF_HOME="$PWD/.cache/huggingface"
uv run --extra learned --extra deploy python - <<'PY'
from relay_lab.llm import LocalLLM
with LocalLLM(startup_timeout=900) as model:
    print(model.generate([{"role": "user", "content": "Reply with READY."}]).text)
PY
```

The warm-up downloads and loads the pinned `Qwen/Qwen3-1.7B-MLX-4bit` revision
from [`llm.py`](../src/relay_lab/llm.py), then makes one real generation. Allow
network access and disk space for the initial download. Studio's folding worker
runs offline, so complete this before launching a task. Missing dependencies or
weights produce an error rather than a fallback planner.

## Run the task-trained cloth baseline

This path uses the original state-assisted CNN/attention controllers, with
Qwen phase review and SpacetimeDB coordination. It does not execute either
foundation VLA's native action head. See the [cloth guide](cloth-coordination.md)
for observation boundaries, fixtures, success criteria and recording formats.

```sh
uv run --extra learned relay-lab cloth-run --seed 101 \
  --communication structured_text --scenario delayed \
  --coordination spacetime --planner llm --max-ticks 1800 \
  --output runs/cloth-task

uv run --extra learned relay-lab cloth-compare --seeds 101 102 \
  --scenarios clear delayed --coordination spacetime --planner llm \
  --max-ticks 1800 --output runs/cloth-comparison

uv run --extra learned relay-lab cloth --directory runs/cloth-comparison
```

Open <http://127.0.0.1:8769/>. The comparison pairs communication modes on the
same seeds, scenes and tick budgets. Its embeddings are compressed measured
state; they are separate from the frozen foundation-feature experiment below.
For dashboard dispatch, follow the [Studio task walkthrough](studio.md).

## Frozen SmolVLA / X-VLA communication

This experiment trains small encoders, receivers and replacement action heads
on frozen foundation features. It does not fine-tune native VLA action heads.
The comparison is a recorded-state communication diagnostic, not a completed
fold. Read the [method and limitations](latent-coordination.md) and
[packaged evidence](../examples/cloth-coordination/learned-latent/README.md).

Install the VLA extra, then download the exact model and tokenizer assets used
by the extractors. This step downloads several gigabytes but performs no model
inference. Run it once with network access before the offline cache commands.
X-VLA's loader expects the repository-local cache shown here.

```sh
uv sync --python 3.13 --extra learned --extra deploy --extra vla
export HF_HOME="$PWD/.cache/huggingface"
uv run --extra learned --extra vla python - <<'PY'
from huggingface_hub import snapshot_download
from relay_lab.smolvla import (
    DEFAULT_MODEL_ID, DEFAULT_REVISION, VLM_MODEL_ID, VLM_REVISION,
)
from relay_lab.cloth.pretrained import XVLA_ID, XVLA_REVISION, BART_ID, BART_REVISION

assets = [
    (DEFAULT_MODEL_ID, DEFAULT_REVISION, ["*.json", "*.safetensors", "README.md"]),
    (VLM_MODEL_ID, VLM_REVISION, ["*.json", "*.txt"]),
    (XVLA_ID, XVLA_REVISION, ["*.json", "model.safetensors"]),
    (BART_ID, BART_REVISION, ["config.json", "vocab.json", "merges.txt",
                            "tokenizer.json", "tokenizer_config.json",
                            "special_tokens_map.json"]),
]
for model_id, revision, patterns in assets:
    snapshot_download(model_id, revision=revision, allow_patterns=patterns,
                      cache_dir=".cache/huggingface/hub")
PY

uv run --extra learned --extra vla python -m relay_lab.cloth.latent_dataset \
  --workers 4 --output runs/latent-data
HF_HUB_OFFLINE=1 uv run --extra learned --extra vla \
  python -m relay_lab.cloth.latent_training cache \
  --dataset runs/latent-data --output runs/latent-features --family smolvla
HF_HUB_OFFLINE=1 uv run --extra learned --extra vla \
  python -m relay_lab.cloth.latent_training cache \
  --dataset runs/latent-data --output runs/latent-features --family xvla
uv run --extra learned --extra vla python -m relay_lab.cloth.latent_training train \
  --dataset runs/latent-data --cache runs/latent-features \
  --output runs/latent-adapters --epochs 400
uv run --extra learned --extra vla python -m relay_lab.cloth.latent_experiment compare \
  --dataset runs/latent-data --cache runs/latent-features \
  --checkpoints runs/latent-adapters --output runs/latent-transport --backend spacetime
```

Run the two backbone cache jobs sequentially on memory-constrained machines.
Retain the dataset split, feature hashes, model identities, training reports and
all comparison outcomes. The packaged evidence also includes aligned inputs and
caches for retraining adapters without extracting foundation features again.

## Earlier parcel and factory experiments

The parcel path uses the original task-trained Relay VLA, local Qwen and
SpacetimeDB. It is separate from both cloth controller families above.

```sh
uv run --extra learned relay-lab run --seed 7 --policy relay-vla \
  --planner llm --coordination spacetime --output runs/local-mediator
uv run --extra learned relay-lab serve --directory runs --port 8766
```

Open <http://127.0.0.1:8766/local-mediator/replay.html>. Training and model scope
are in the [Relay VLA guide](relay-vla.md). Earlier factory code and recordings
remain available: [factory setup](factory.md), [learned factory control](factory-vla.md),
and the [private-information comparison](../examples/factory-communication/README.md).
These guides distinguish controlled channel tests from actual Qwen/Spacetime
integration runs and retain unsuccessful conditions.
