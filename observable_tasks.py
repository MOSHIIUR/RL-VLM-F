OBSERVABLE_TASKS = [
    {
        "canonical": "metaworld_button-press-topdown-v2-goal-observable",
        "alias": "metaworld_button-press-topdown-observable",
        "slug": "button-press-topdown-observable",
    },
    {
        "canonical": "metaworld_door-open-v2-goal-observable",
        "alias": "metaworld_door-open-v2-observable",
        "slug": "door-open-v2-observable",
    },
    {
        "canonical": "metaworld_drawer-open-v2-goal-observable",
        "alias": "metaworld_drawer-open-v2-observable",
        "slug": "drawer-open-v2-observable",
    },
    {
        "canonical": "metaworld_push-v2-goal-observable",
        "alias": "metaworld_push-v2-observable",
        "slug": "push-v2-observable",
    },
    {
        "canonical": "metaworld_window-open-v2-goal-observable",
        "alias": "metaworld_window-open-v2-observable",
        "slug": "window-open-v2-observable",
    },
]


CANONICAL_TO_TASK = {task["canonical"]: task for task in OBSERVABLE_TASKS}
ALIASES_TO_CANONICAL = {task["alias"]: task["canonical"] for task in OBSERVABLE_TASKS}
ALIASES_TO_CANONICAL.update({task["canonical"]: task["canonical"] for task in OBSERVABLE_TASKS})


def canonicalize_env_name(env_name):
    if env_name not in ALIASES_TO_CANONICAL:
        raise KeyError(f"Unsupported observable task: {env_name}")
    return ALIASES_TO_CANONICAL[env_name]


def get_observable_tasks(include_aliases=False):
    task_names = [task["canonical"] for task in OBSERVABLE_TASKS]
    if include_aliases:
        task_names.extend(task["alias"] for task in OBSERVABLE_TASKS)
    return task_names


def common_smoke_overrides(env_name, seed=0, exp_name="observable_modal_smoke"):
    canonical_env = canonicalize_env_name(env_name)
    return [
        f"env={canonical_env}",
        f"seed={seed}",
        f"exp_name={exp_name}",
        "gradient_update=1",
        "activation=tanh",
        "num_eval_episodes=1",
        "eval_frequency=5000",
        "save_interval=5000",
        "agent.params.actor_lr=0.0003",
        "agent.params.critic_lr=0.0003",
        "agent.params.batch_size=256",
        "double_q_critic.params.hidden_dim=256",
        "double_q_critic.params.hidden_depth=3",
        "diag_gaussian_actor.params.hidden_dim=256",
        "diag_gaussian_actor.params.hidden_depth=3",
    ]


def stage_overrides(stage):
    if stage == "gt":
        return [
            "reward=gt_task_reward",
            "vlm_label=0",
            "image_reward=0",
            "num_seed_steps=1000",
            "num_unsup_steps=0",
            "num_train_steps=20000",
        ]
    if stage == "clip":
        return [
            "reward=clip_image_text_matching",
            "vlm_label=0",
            "image_reward=0",
            "num_seed_steps=1000",
            "num_unsup_steps=0",
            "num_train_steps=20000",
        ]
    if stage == "qwen_pref":
        return [
            "reward=learn_from_preference",
            "vlm_label=1",
            "vlm=qwen_local",
            "image_reward=1",
            "segment=1",
            "reward_batch=4",
            "reward_update=1",
            "num_interact=1000",
            "max_feedback=40",
            "reward_lr=1e-4",
            "num_seed_steps=500",
            "num_unsup_steps=500",
            "num_train_steps=6000",
            "feed_type=0",
            "teacher_beta=-1",
            "teacher_gamma=1",
            "teacher_eps_mistake=0",
            "teacher_eps_skip=0",
            "teacher_eps_equal=0",
            "cached_label_path=null",
            "agent.params.batch_size=128",
        ]
    if stage == "qwen_full":
        return [
            "reward=learn_from_preference",
            "vlm_label=1",
            "vlm=qwen_local",
            "image_reward=1",
            "eval_frequency=10000",
            "save_interval=20000",
            "segment=1",
            "reward_batch=40",
            "reward_update=10",
            "num_interact=4000",
            "max_feedback=20000",
            "reward_lr=1e-4",
            "num_seed_steps=1000",
            "num_unsup_steps=9000",
            "num_train_steps=1000000",
            "feed_type=0",
            "teacher_beta=-1",
            "teacher_gamma=1",
            "teacher_eps_mistake=0",
            "teacher_eps_skip=0",
            "teacher_eps_equal=0",
            "cached_label_path=null",
            "agent.params.batch_size=512",
        ]
    raise KeyError(f"Unsupported stage: {stage}")


def build_stage_overrides(env_name, stage, seed=0, exp_name="observable_modal_smoke", run_dir=None):
    overrides = common_smoke_overrides(env_name, seed=seed, exp_name=exp_name)
    overrides.extend(stage_overrides(stage))
    if run_dir is not None:
        overrides.append(f"hydra.run.dir={run_dir}")
    return overrides
