import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--max_iterations", type=int, default=3000, help="Max training iterations")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.core.velocity.config.dog.flat_env_cfg import DogFlatEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

def main():
    print("Setting up environment...")
    
    env_cfg = DogFlatEnvCfg()
    if hasattr(args, 'num_envs') and args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    else:
        env_cfg.scene.num_envs = 4096
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"Base environment created with {env.num_envs} envs")
    
    env = RslRlVecEnvWrapper(env)
    print("Environment wrapped with RslRlVecEnvWrapper")
    
    # ===== 配置中添加 save_interval =====
    agent_cfg = {
        "seed": 42,
        "num_steps_per_env": 24,
        "max_iterations": args.max_iterations,
        "save_interval": 50,  # 每 50 轮保存一次模型
        "experiment_name": "dog_flat",
        "empirical_normalization": False,
        "init_noise_std": 2.0,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128, 128],
            "activation": "elu",
            "obs_normalization": False,
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128, 128],
            "activation": "elu",
            "obs_normalization": False,
        },
        "runner": {
            "class_name": "OnPolicyRunner",
            "max_iterations": args.max_iterations,
        },
    }
    
    runner = OnPolicyRunner(env, agent_cfg, log_dir="./logs/rsl_rl/dog_flat")
    
    print("Starting training...")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    print("Training completed!")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
