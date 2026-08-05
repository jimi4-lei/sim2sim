# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass
from isaaclab.managers import TerminationTermCfg

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_assets.robots.dog import DOG_CFG


@configclass
class DogRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # ============================================================
        # 🔥 在调用 super() 之前设置
        # ============================================================
        
        # 先禁用终止条件
        if hasattr(self, "terminations"):
            # 临时存储，防止被覆盖
            pass
        
        super().__post_init__()
        
        # 切换机器人为 dog
        self.scene.robot = DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # ============================================================
        # 🔥🔥🔥 在 super() 之后再次禁用
        # ============================================================
        
        # 禁用所有终止条件
        if hasattr(self.terminations, "base_contact"):
            self.terminations.base_contact.time_out = False
            print("✅ DogRoughEnvCfg: 禁用 base_contact (time_out=False)")
        
        if hasattr(self.terminations, "base_height"):
            self.terminations.base_height.time_out = False
            print("✅ DogRoughEnvCfg: 禁用 base_height (time_out=False)")
        
        if hasattr(self.terminations, "time_out"):
            self.terminations.time_out.time_out = False
            print("✅ DogRoughEnvCfg: 禁用 time_out (time_out=False)")
        
        # 确保 episode length 很长
        self.episode_length_s = 1000.0
        print(f"✅ DogRoughEnvCfg: episode_length_s = {self.episode_length_s}")
