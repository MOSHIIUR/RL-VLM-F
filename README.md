# Five-Goal Observable MetaWorld: Full Qwen/vLLM Runs

This is the minimal setup for the five goal-observable MetaWorld experiments using the full preference-based run with `vlm=qwen_local` and vLLM.

The five tasks are:

- `metaworld_button-press-topdown-v2-goal-observable`
- `metaworld_door-open-v2-goal-observable`
- `metaworld_drawer-open-v2-goal-observable`
- `metaworld_push-v2-goal-observable`
- `metaworld_window-open-v2-goal-observable`

## Setup

From the repo root:

```bash
conda env create -f conda_env.yml
conda activate rlvlmf
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install numpy==1.26.0
pip install \
  'setuptools<81' \
  'protobuf==6.33.6' \
  'wandb==0.25.1' \
  'vllm==0.7.3' \
  'qwen-vl-utils==0.0.14' \
  'git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef'
```

## Run All Five Experiments

Use the provided launcher:

```bash
bash run_observable_qwen_full.sh
```

To use a different seed:

```bash
bash run_observable_qwen_full.sh 1
```

The script does the following for you:

- runs from the repo root
- sets the same environment variables needed for the MetaWorld / MuJoCo run
- sets the recommended Qwen/vLLM environment variables
- launches the five full runs sequentially

Outputs go under `exp/observable_qwen_full/...`.

## Notes

- These runs use `Qwen/Qwen2.5-VL-3B-Instruct`.
- The script defaults to `QWEN_VL_BACKEND=vllm`.
- No Gemini or OpenAI setup is needed for this workflow.
