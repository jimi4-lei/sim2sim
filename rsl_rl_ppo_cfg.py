# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DogFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96                     # 从 24 → 96（关键！）
    max_iterations = 3000
    save_interval = 50
    experiment_name = "dog_flat"
    empirical_normalization = True            # 开启归一化（关键！）
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],          # 从 [128,128,128] 扩充
        activation="elu",
        obs_normalization=False,              # 由 runner 层面控制，这里保持 False
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.4,                     # 从 1.0 → 0.6
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],          # 与 actor 保持一致
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,                    # 从 0.005 → 0.02
        num_learning_epochs=15,               # 从 5 → 15
        num_mini_batches=8,                   # 从 4 → 8
        learning_rate=5.0e-4,                 # 从 1e-3 → 5e-4
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
