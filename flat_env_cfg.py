# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.config.dog.rough_env_cfg import DogRoughEnvCfg


@configclass
class DogFlatEnvCfg(DogRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg()

    def __post_init__(self):
        super().__post_init__()

        # override rewards for flat terrain
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        
        # 修复足端名称（小写匹配）
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [".*_foot"]
        
        # 修复 undesired_contacts 的正则表达式（小写匹配）
        # 注意：这里需要修改 rough_env_cfg.py 中的配置，但我们可以直接在 flat_env_cfg.py 中覆盖
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_hip", "base"]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_hip", "base"]
        
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        
        # no terrain curriculum
        self.curriculum.terrain_levels = None
