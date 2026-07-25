# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .flat_env_cfg import DogFlatEnvCfg


##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Dog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "cfg": DogFlatEnvCfg(),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DogFlatPPORunnerCfg",
    },
)
