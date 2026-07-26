# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.config.dog.flat_env_cfg import DogFlatEnvCfg


@configclass
class DogFlatEnvCfgSimple(DogFlatEnvCfg):
    sim: SimulationCfg = SimulationCfg()

    def __post_init__(self):
        super().__post_init__()

        # ===== 核心修改 =====
        # 1. 速度跟踪奖励（大幅提高，让"走起来"成为最划算的事）
        self.rewards.track_lin_vel_xy_exp.weight = 5.0   # 提高到 5.0
        
        # 2. 对"静止不动"的额外惩罚
        # 注意：我们需要在 rewards 中新增一个项，但这里我们直接修改现有项的逻辑
        
        # 3. 惩罚项（保持适度）
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.dof_torques_l2.weight = -1e-6
        self.rewards.dof_acc_l2.weight = -1e-7
        self.rewards.action_rate_l2.weight = -0.001
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.feet_air_time.weight = 0.0
