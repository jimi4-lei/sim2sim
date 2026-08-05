# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.managers import RewardTermCfg
from isaaclab_tasks.core.velocity.rewards import alive_reward
from isaaclab_tasks.core.velocity.config.dog.rough_env_cfg import DogRoughEnvCfg


@configclass
class DogFlatEnvCfg(DogRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg()

    def __post_init__(self):
        # ============================================================
        # 在调用 super() 之前先禁用终止条件
        # ============================================================
        
        if hasattr(self, "terminations") and hasattr(self.terminations, "base_contact"):
            if hasattr(self.terminations.base_contact, "params"):
                self.terminations.base_contact.params["threshold"] = 999999.0
            self.terminations.base_contact.time_out = False
        
        # 调用父类初始化
        super().__post_init__()

        # ============================================================
        # 🔥 再次确保禁用终止条件
        # ============================================================
        
        if hasattr(self.terminations, "base_contact"):
            if hasattr(self.terminations.base_contact, "params"):
                self.terminations.base_contact.params["threshold"] = 999999.0
            self.terminations.base_contact.time_out = False
        
        if hasattr(self.terminations, "time_out"):
            self.terminations.time_out.time_out = False

        # ============================================================
        # 修正正则表达式
        # ============================================================
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [".*_foot"]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_hip", "base"]
        
        # ============================================================
        # 🔥 添加存活奖励（必须在访问之前创建）
        # ============================================================
        
        if not hasattr(self.rewards, "alive"):
            self.rewards.alive = RewardTermCfg(
                weight=0.3,
                params={},
                func=alive_reward,
            )
            print("✅ 添加存活奖励: 0.3")
        else:
            self.rewards.alive.weight = 0.3
            print("✅ 更新存活奖励: 0.3")

        # ============================================================
        # 🔥🔥🔥 重新平衡奖励权重
        # ============================================================
        
        # 1. 行走奖励
        self.rewards.track_lin_vel_xy_exp.weight = 80.0
        
        # 2. 惩罚（大幅降低）
        self.rewards.dof_acc_l2.weight = -5e-8
        self.rewards.dof_torques_l2.weight = -5e-7
        self.rewards.action_rate_l2.weight = -0.0005
        
        # 3. 姿态惩罚
        self.rewards.lin_vel_z_l2.weight = -0.1
        self.rewards.ang_vel_xy_l2.weight = -0.005
        self.rewards.flat_orientation_l2.weight = -0.2
        
        # 4. 接触惩罚
        self.rewards.undesired_contacts.weight = -0.05
        
        # 5. 禁用足端离地时间
        self.rewards.feet_air_time.weight = 0.0
        
        print("✅ 奖励权重已重新平衡:")
        print(f"   track_lin_vel_xy_exp: 80.0")
        print(f"   dof_acc_l2: -5e-8")
        print(f"   alive: 0.3")

        # ============================================================
        # 设置机器人的初始关节角度
        # ============================================================
        
        self.scene.robot.init_state.joint_pos = {
            "FL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "RL_hip_joint": 0.1,
            "FL_thigh_joint": 0.4,
            "FR_thigh_joint": -0.4,
            "RR_thigh_joint": -0.4,
            "RL_thigh_joint": 0.4,
            "FL_calf_joint": -1.2,
            "FR_calf_joint": -1.2,
            "RR_calf_joint": -1.2,
            "RL_calf_joint": -1.2,
        }
        print("✅ 设置机器人初始站立姿态")
        
        # ============================================================
        # 设置 episode 长度
        # ============================================================
        self.episode_length_s = 1000.0

        # 减小命令速度范围
        if hasattr(self.commands, "range_lin_vel_x"):
            self.commands.range_lin_vel_x = [-0.3, 0.3]
            self.commands.range_lin_vel_y = [-0.2, 0.2]
        
        # ============================================================
        # 地形设置
        # ============================================================
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        
        print("\n" + "="*60)
        print("✅ 配置完成! 奖励权重已重新平衡")
        print("="*60 + "\n")
