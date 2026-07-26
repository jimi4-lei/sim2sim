import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=100, help="Max training iterations")
parser.add_argument("--num_steps", type=int, default=24, help="Steps per environment per iteration")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.core.velocity.config.dog.flat_env_cfg_simple import DogFlatEnvCfgSimple
from torch.utils.tensorboard import SummaryWriter

class SimpleEnvAdapter:
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.num_obs = env.observation_space["policy"].shape[0]
        self.num_actions = 12
        self._cached_obs = None
        self.cfg = {"env_name": "anymal_d_flat", "num_envs": self.num_envs}
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.max_episode_length = 1000
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        
    def reset(self):
        obs_dict, _ = self.env.reset()
        self._cached_obs = obs_dict["policy"]
        self.obs_buf = self._cached_obs
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        return self._cached_obs
    
    def step(self, actions):
        obs_dict, reward, terminated, truncated, info = self.env.step(actions)
        self._cached_obs = obs_dict["policy"]
        self.obs_buf = self._cached_obs
        self.episode_length_buf += 1
        self.episode_length_buf[terminated | truncated] = 0
        return self._cached_obs, reward, terminated, truncated, info
    
    def get_observations(self):
        if self._cached_obs is None:
            obs_dict, _ = self.env.reset()
            self._cached_obs = obs_dict["policy"]
            self.obs_buf = self._cached_obs
        return self._cached_obs

class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ELU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.mlp = nn.Sequential(*layers)
        self.log_std = nn.Parameter(
            torch.ones(action_dim)*-1
        )
        
    def forward(self, obs):
        mean = self.mlp(obs)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0))
        return mean, std
    
    def get_action(self, obs):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, mean, std
    
    def evaluate(self, obs, action):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

class Value(nn.Module):
    def __init__(self, obs_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ELU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, obs):
        return self.mlp(obs)

def main():
    print("Setting up environment...")
    
    env_cfg = DogFlatEnvCfgSimple()
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"Environment created with {env.num_envs} envs")
    
    env_adapter = SimpleEnvAdapter(env)
    device = env_adapter.device
    num_envs = env_adapter.num_envs
    num_obs = env_adapter.num_obs
    num_actions = env_adapter.num_actions
    num_steps = args.num_steps
    max_iterations = args.max_iterations
    
    # 创建网络
    policy = Policy(num_obs, num_actions).to(device)
    value = Value(num_obs).to(device)
    policy_optimizer = optim.Adam(policy.parameters(), lr=5e-4)
    value_optimizer = optim.Adam(value.parameters(), lr=5e-4)
    
    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir="./logs/tensorboard")
    print("TensorBoard logs will be saved to ./logs/tensorboard")
    
    # 存储rollout数据
    obs_buffer = torch.zeros(num_steps, num_envs, num_obs, device=device)
    actions_buffer = torch.zeros(num_steps, num_envs, num_actions, device=device)
    log_probs_buffer = torch.zeros(num_steps, num_envs, 1, device=device)
    rewards_buffer = torch.zeros(num_steps, num_envs, 1, device=device)
    values_buffer = torch.zeros(num_steps, num_envs, 1, device=device)
    terminated_buffer = torch.zeros(num_steps, num_envs, 1, device=device, dtype=torch.bool)
    
    print("Starting PPO training...")
    
    obs = env_adapter.reset()
    
    for iteration in range(max_iterations):
        # 收集数据
        for step in range(num_steps):
            obs_buffer[step] = obs
            with torch.no_grad():
                action, log_prob, mean, std = policy.get_action(obs)
                value_val = value(obs)
            
            actions_buffer[step] = action
            log_probs_buffer[step] = log_prob
            values_buffer[step] = value_val
            
            obs, reward, terminated, truncated, _ = env_adapter.step(action)
            rewards_buffer[step] = reward.unsqueeze(1)
            terminated_buffer[step] = terminated.unsqueeze(1)
        
        # 计算GAE
        with torch.no_grad():
            last_value = value(obs)
            advantages = torch.zeros(num_steps, num_envs, 1, device=device)
            returns = torch.zeros(num_steps, num_envs, 1, device=device)
            gae = 0
            gamma = 0.99
            lam = 0.95
            for step in reversed(range(num_steps)):
                if step == num_steps - 1:
                    next_value = last_value
                else:
                    next_value = values_buffer[step + 1]
                delta = rewards_buffer[step] + gamma * next_value * (1 - terminated_buffer[step].float()) - values_buffer[step]
                gae = delta + gamma * lam * (1 - terminated_buffer[step].float()) * gae
                advantages[step] = gae
                returns[step] = advantages[step] + values_buffer[step]
        
        # 更新策略
        batch_size = num_envs * num_steps
        indices = torch.randperm(batch_size)
        num_mini_batches = 4
        mini_batch_size = batch_size // num_mini_batches
        
        for epoch in range(5):
            for mini_batch_start in range(0, batch_size, mini_batch_size):
                mini_batch_end = mini_batch_start + mini_batch_size
                mini_indices = indices[mini_batch_start:mini_batch_end]
                
                mini_obs = obs_buffer.view(-1, num_obs)[mini_indices]
                mini_actions = actions_buffer.view(-1, num_actions)[mini_indices]
                mini_log_probs = log_probs_buffer.view(-1, 1)[mini_indices]
                mini_advantages = advantages.view(-1, 1)[mini_indices]
                mini_returns = returns.view(-1, 1)[mini_indices]
                mini_values = values_buffer.view(-1, 1)[mini_indices]
                
                # 归一化优势
                mini_advantages = (mini_advantages - mini_advantages.mean()) / (mini_advantages.std() + 1e-8)
                
                # 更新策略
                new_log_probs, entropy = policy.evaluate(mini_obs, mini_actions)
                ratio = torch.exp(new_log_probs - mini_log_probs)
                clip_param = 0.2
                surr1 = ratio * mini_advantages
                surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * mini_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
                
                policy_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                policy_optimizer.step()
                
                # 更新价值函数
                value_pred = value(mini_obs)
                value_loss = nn.MSELoss()(value_pred, mini_returns)
                
                value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(value.parameters(), 1.0)
                value_optimizer.step()
        
        if iteration % 10 == 0:
            avg_reward = rewards_buffer.mean().item()
            avg_value = values_buffer.mean().item()
            print(f"Iteration {iteration}: Avg Reward: {avg_reward:.4f}, Avg Value: {avg_value:.4f}")
            # 记录到 TensorBoard
            writer.add_scalar("Reward/Average", avg_reward, iteration)
            writer.add_scalar("Value/Average", avg_value, iteration)
            writer.add_scalar("Loss/Policy", policy_loss.item(), iteration)
            writer.add_scalar("Loss/Value", value_loss.item(), iteration)
    
    print("Training completed!")
    
    # 关闭 TensorBoard writer
    writer.close()
    print("TensorBoard logs saved. Run: tensorboard --logdir ./logs/tensorboard")
    
    # 保存模型
    import os
    os.makedirs("./logs", exist_ok=True)
    torch.save(policy.state_dict(), "./logs/dog_policy.pth")
    print("Model saved to ./logs/dog_policy.pth")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
