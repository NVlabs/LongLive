# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: experiments/robotwin/fastwam_policy/deploy_policy.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/benchmarks/robotwin/longwam_policy/deploy_policy.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

"""Thin upstream evaluator callbacks; all prediction/execution is shared runtime."""
import atexit
from longwam.benchmarks.robotwin.adapter import policy_batch


def get_model(usr_args):
    from longwam.evaluation import read_settings
    from longwam.runtime import create_policy

    if not usr_args.get("runtime_config"):
        raise ValueError("runtime_config is required; launch longwam eval robotwin2/domino")
    policy = create_policy(read_settings(usr_args["runtime_config"]))
    atexit.register(policy.close)
    return policy


def eval(task_env, model, observation):
    if observation is None:
        raise ValueError("The runtime requires an observation every control step")
    action = model.select_action(policy_batch(observation, task_env.get_instruction()))
    task_env.take_action(action[0].numpy(), action_type="qpos")


def reset_model(model):
    model.reset()
