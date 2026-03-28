#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYFLEXROOT="${ROOT_DIR}/softgym/PyFlex"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/softgym:${PYFLEXROOT}/bindings/build${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MUJOCO_GL=egl

SEED="${1:-0}"

export QWEN_VL_MODEL="${QWEN_VL_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export QWEN_VL_BACKEND="${QWEN_VL_BACKEND:-vllm}"
export QWEN_VL_BATCH_SIZE="${QWEN_VL_BATCH_SIZE:-4}"
export QWEN_VL_DTYPE="${QWEN_VL_DTYPE:-float16}"
export QWEN_VL_MAX_SIDE="${QWEN_VL_MAX_SIDE:-224}"
export QWEN_VL_MAX_NEW_TOKENS="${QWEN_VL_MAX_NEW_TOKENS:-4}"
export QWEN_VL_MAX_MODEL_LEN="${QWEN_VL_MAX_MODEL_LEN:-4096}"
export QWEN_VL_MAX_NUM_SEQS="${QWEN_VL_MAX_NUM_SEQS:-4}"
export QWEN_VL_GPU_MEMORY_UTILIZATION="${QWEN_VL_GPU_MEMORY_UTILIZATION:-0.55}"
export QWEN_VL_LIMIT_MM_PER_PROMPT="${QWEN_VL_LIMIT_MM_PER_PROMPT:-{\"image\": 2}}"

TASKS=(
  metaworld_button-press-topdown-v2-goal-observable
  metaworld_door-open-v2-goal-observable
  metaworld_drawer-open-v2-goal-observable
  metaworld_push-v2-goal-observable
  metaworld_window-open-v2-goal-observable
)

for ENV in "${TASKS[@]}"; do
  echo "Starting ${ENV} with seed ${SEED}"
  python train_PEBBLE.py \
    env="${ENV}" \
    seed="${SEED}" \
    exp_name=observable_qwen_full \
    gradient_update=1 \
    activation=tanh \
    num_eval_episodes=1 \
    agent.params.actor_lr=0.0003 \
    agent.params.critic_lr=0.0003 \
    double_q_critic.params.hidden_dim=256 \
    double_q_critic.params.hidden_depth=3 \
    diag_gaussian_actor.params.hidden_dim=256 \
    diag_gaussian_actor.params.hidden_depth=3 \
    reward=learn_from_preference \
    vlm_label=1 \
    vlm=qwen_local \
    image_reward=1 \
    eval_frequency=10000 \
    save_interval=20000 \
    segment=1 \
    reward_batch=40 \
    reward_update=10 \
    num_interact=4000 \
    max_feedback=20000 \
    reward_lr=1e-4 \
    num_seed_steps=1000 \
    num_unsup_steps=9000 \
    num_train_steps=1000000 \
    feed_type=0 \
    teacher_beta=-1 \
    teacher_gamma=1 \
    teacher_eps_mistake=0 \
    teacher_eps_skip=0 \
    teacher_eps_equal=0 \
    cached_label_path=null \
    agent.params.batch_size=512
done
