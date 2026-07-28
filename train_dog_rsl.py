import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--max_iterations", type=int, default=3000, help="Max training iterations")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.core.velocity.config.dog.flat_env_cfg import DogFlatEnvCfg
from rsl_rl.runners import OnPolicyRunner

# ===== 字典包装器，支持 .to() 方法 =====
class DictWithTo:
    def __init__(self, data):
        self._data = data
        
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
    
    def __contains__(self, key):
        return key in self._data
    
    def __iter__(self):
        return iter(self._data)
    
    def items(self):
        return self._data.items()
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def to(self, device):
        """将字典中的所有张量移动到指定设备"""
        new_data = {}
        for key, value in self._data.items():
            if hasattr(value, 'to'):
                new_data[key] = value.to(device)
            else:
                new_data[key] = value
        return DictWithTo(new_data)
    
    # 让 isinstance(obs, dict) 返回 True
    def __class__(self):
        return dict

# ===== 环境适配器 =====
class RslRlEnvAdapter:
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.cfg = {"env_name": "dog_flat", "num_envs": self.num_envs}
        self._cached_obs = None
        self.num_actions = 12
        self.num_obs = 48
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.max_episode_length = 1000
        
    def reset(self):
        obs_dict, _ = self.env.reset()
        self._cached_obs = obs_dict["policy"]
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        return DictWithTo({"policy": self._cached_obs})
    
    def step(self, actions):
        obs_dict, reward, terminated, truncated, info = self.env.step(actions)
        self._cached_obs = obs_dict["policy"]
        self.episode_length_buf += 1
        self.episode_length_buf[terminated | truncated] = 0
        dones = terminated | truncated
        return DictWithTo({"policy": self._cached_obs}), reward, dones, {}
    
    def get_observations(self):
        if self._cached_obs is None:
            obs_dict, _ = self.env.reset()
            self._cached_obs = obs_dict["policy"]
        return DictWithTo({"policy": self._cached_obs})
    
    def close(self):
        self.env.close()

def main():
    print("Setting up environment...")
    
    env_cfg = DogFlatEnvCfg()
    if hasattr(args, 'num_envs') and args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    else:
        env_cfg.scene.num_envs = 4096
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"Environment created with {env.num_envs} envs")
    
    env_adapter = RslRlEnvAdapter(env)
    
    agent_cfg = {
        "seed": 42,
        "num_steps_per_env": 16,
        "empirical_normalization": False,
        "init_noise_std": 1.0,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 8,
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
    
    runner = OnPolicyRunner(env_adapter, agent_cfg, log_dir="./logs/rsl_rl/dog_flat")
    
    print("Starting training...")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    print("Training completed!")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
