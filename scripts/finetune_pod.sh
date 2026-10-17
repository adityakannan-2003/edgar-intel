#!/usr/bin/env bash
# The GPU half of the fine-tune benchmark, run ON a rented pod: one RTX 4090,
# a PyTorch/CUDA image, everything under /workspace. Every stage logs to
# /workspace/logs/.
#
# From the Mac first (the pod needs the code and the data, not the database):
#   rsync -az --exclude .venv src pyproject.toml README.md scripts POD:/workspace/edgar-intel/
#   rsync -az data/finetune/train.jsonl data/finetune/held_out.jsonl POD:/workspace/data/
#
# Then on the pod:
#   bash scripts/finetune_pod.sh setup                       # two environments, once
#   bash scripts/finetune_pod.sh train pilot --max-examples 300 --epochs 1
#   bash scripts/finetune_pod.sh serve pilot                 # Qwen3-8B + adapter "extract" on :8000
#
# And from the Mac, through a tunnel so nothing is exposed publicly:
#   ssh -N -L 8000:localhost:8000 POD
#   edgar-intel finetune predict --arm qwen3-8b-lora --base-url http://localhost:8000/v1 ...
#
# Training and serving get separate environments because vLLM pins its own
# torch and transformers. The training versions are the ones the CPU smoke
# test passed on (reports/finetune/train-smoke-cpu.json).
set -euo pipefail

W=/workspace
REPO=$W/edgar-intel
export HF_HOME=$W/hf            # model downloads land on the pod's disk, once
mkdir -p "$W/logs" "$W/adapters"

stage=${1:?usage: finetune_pod.sh setup|train NAME [ARGS]|serve NAME}
shift

case "$stage" in
setup)
  nvidia-smi | tee "$W/logs/nvidia-smi.txt"
  python -m pip install -q uv
  uv venv -q --python 3.12 "$W/train-env"
  VIRTUAL_ENV=$W/train-env uv pip install -q \
    torch==2.14.1 transformers==5.20.1 peft==0.21.2 trl==1.16.0 datasets==5.1.0 \
    accelerate bitsandbytes -e "$REPO"
  uv venv -q --python 3.12 "$W/serve-env"
  VIRTUAL_ENV=$W/serve-env uv pip install -q vllm
  "$W/train-env/bin/python" -c "import torch; assert torch.cuda.is_available(); print('train env: cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
  "$W/serve-env/bin/python" -c "import vllm; print('serve env: vllm', vllm.__version__)"
  ;;
train)
  name=${1:?adapter name}; shift
  cd "$REPO"
  "$W/train-env/bin/edgar-intel" finetune train \
    --train-file "$W/data/train.jsonl" --output-dir "$W/adapters/$name" "$@" \
    2>&1 | tee "$W/logs/train-$name.log"
  ;;
serve)
  name=${1:?adapter name}
  pkill -f "vllm serve" || true
  nohup "$W/serve-env/bin/vllm" serve Qwen/Qwen3-8B \
    --enable-lora --lora-modules "extract=$W/adapters/$name" --max-lora-rank 16 \
    --max-model-len 8192 --gpu-memory-utilization 0.90 --seed 0 --port 8000 \
    > "$W/logs/serve-$name.log" 2>&1 &
  pid=$!
  # The first start downloads 16 GB of weights; give it 20 minutes, and stop
  # waiting the moment the server dies rather than polling a dead port.
  for _ in $(seq 240); do
    if curl -sf localhost:8000/v1/models >/dev/null; then
      curl -s localhost:8000/v1/models
      exit 0
    fi
    kill -0 "$pid" 2>/dev/null || { tail -40 "$W/logs/serve-$name.log"; exit 1; }
    sleep 5
  done
  echo "vLLM not ready after 20 minutes" >&2
  tail -40 "$W/logs/serve-$name.log"
  exit 1
  ;;
*)
  echo "unknown stage: $stage" >&2
  exit 2
  ;;
esac
