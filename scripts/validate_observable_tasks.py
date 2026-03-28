#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from observable_tasks import OBSERVABLE_TASKS, canonicalize_env_name
from prompt import clip_env_prompts, goal_env_prompts
import utils


def _to_serializable(obs):
    obs = np.asarray(obs)
    return {
        "shape": list(obs.shape),
        "goal_tail": obs[-3:].tolist(),
    }


def validate_task(env_name, seed, output_dir):
    cfg = SimpleNamespace(env=canonicalize_env_name(env_name), seed=seed)
    env = utils.make_metaworld_env(cfg)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    step_output = env.step(action)
    if len(step_output) == 4:
        next_obs, reward, done, info = step_output
    else:
        next_obs, reward, terminated, truncated, info = step_output
        done = terminated or truncated

    frame = env.render()
    task_dir = output_dir / env_name.replace("metaworld_", "")
    task_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(task_dir / "render.png")

    summary = {
        "env": env_name,
        "canonical_env": canonicalize_env_name(env_name),
        "camera_name": env.env.camera_name,
        "partially_observable": bool(getattr(env.env, "_partially_observable", True)),
        "initial_obs": _to_serializable(obs),
        "next_obs": _to_serializable(next_obs),
        "reward": float(reward),
        "done": bool(done),
        "success": bool(info.get("success", False)),
        "clip_prompt": clip_env_prompts[env_name],
        "goal_prompt": goal_env_prompts[env_name],
    }
    with open(task_dir / "summary.json", "w", encoding="ascii") as f:
        json.dump(summary, f, indent=2)

    env.close()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-aliases", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_names = [task["canonical"] for task in OBSERVABLE_TASKS]
    if args.include_aliases:
        task_names.extend(task["alias"] for task in OBSERVABLE_TASKS)

    results = [validate_task(env_name, args.seed, output_dir) for env_name in task_names]
    with open(output_dir / "manifest.json", "w", encoding="ascii") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
