我看了你的 train_dog_rsl.py 和报错信息，这个错误基本可以确定不是 rollout_storage.py 的问题，而是你的 RSL-RL 配置里面 num_steps_per_env 和环境采样数据维度不匹配导致的。

你的核心报错：

IndexError: index 22988 is out of bounds for dimension 0 with size 16

位置：

rsl_rl/storage/rollout_storage.py

old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params)

意思是：

PPO更新的时候，它认为你的 rollout buffer 里面有很多数据（22988），但是保存旧策略分布参数的数组只有16个。

也就是说：

你的 observation/action/reward buffer长度和 actor 输出保存长度不一致。

1. 先看你的配置

你的 agent_cfg：

"num_steps_per_env": 16,

这里就是你问的：

num_steps_per_env在哪里检查？

就在这里。

你的文件：

train_dog_rsl.py

里面：

agent_cfg = {
    "seed": 42,
    "num_steps_per_env": 16,

这个就是。

2. 为什么会出这个错误？

你的运行：

./isaaclab.sh -p train_dog_rsl.py --headless --num_envs 2048 --max_iterations 3000

但是你的 argparse里面：

parser.add_argument("--max_iterations", type=int, default=3000)

你没有定义：

--num_envs

所以：

args.num_envs

实际上不存在。

你的代码：

if hasattr(args, 'num_envs') and args.num_envs is not None:
    env_cfg.scene.num_envs = args.num_envs
else:
    env_cfg.scene.num_envs = 4096

所以你以为：

--num_envs 2048

生效了。

实际上：

没有！

最后环境：

Environment created with 4096 envs

证明了。

3. 第一个必须修改

你的 argparse 改成：

parser.add_argument(
    "--num_envs",
    type=int,
    default=4096,
    help="Number of environments"
)

放这里：

parser = argparse.ArgumentParser()

parser.add_argument("--max_iterations",
                    type=int,
                    default=3000)

parser.add_argument("--num_envs",
                    type=int,
                    default=4096)

AppLauncher.add_app_launcher_args(parser)
4. 第二个问题：2048环境 + 16步太小

你的配置：

num_envs = 4096
num_steps_per_env = 16

一次 PPO rollout 数据：

4096 × 16
=
65536 samples

你的 mini batch:

num_mini_batches=8

每个batch：

65536 / 8
=
8192

这个没问题。

但是你现在报：

size 16

说明 RSL-RL认为：

old_distribution_params.shape
=
[16,...]

而不是：

[4096*16,...]
5. 真正原因在你的 Adapter

这里：

class RslRlEnvAdapter:

你自己包装 IsaacLab 环境。

问题在这里：

def reset(self):

    obs_dict, _ = self.env.reset()

    self._cached_obs = obs_dict["policy"]

    return DictWithTo({"policy": self._cached_obs})

还有：

def step(self, actions):

    obs_dict, reward, terminated, truncated, info = self.env.step(actions)

    self._cached_obs = obs_dict["policy"]

    return DictWithTo({"policy": self._cached_obs}), reward, dones, {}

IsaacLab原生的 ManagerBasedRLEnv 已经适配 RSL-RL。

你现在：

ManagerBasedRLEnv
        |
        |
你的Adapter
        |
        |
OnPolicyRunner

相当于重复包装。

6. 官方结构应该是：

IsaacLab:

ManagerBasedRLEnv
        |
        |
RslRlVecEnvWrapper
        |
        |
OnPolicyRunner

不是：

ManagerBasedRLEnv
        |
        |
自己写adapter
        |
        |
OnPolicyRunner
7. 你应该删除这个：

整个：

class RslRlEnvAdapter:

不要自己写。

换成官方：

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

然后：

env = ManagerBasedRLEnv(cfg=env_cfg)

env = RslRlVecEnvWrapper(env)

然后：

runner = OnPolicyRunner(
    env,
    agent_cfg,
    log_dir="./logs/rsl_rl/dog_flat"
)
8. 第三个问题：num_steps_per_env

你的：

num_steps_per_env":16

对于四足机器人：

偏小。

推荐：

RTX5060：

4096环境：

num_steps_per_env = 24

如果显存不够：

2048 env
24 steps

比较合理。

例如：

env_cfg.scene.num_envs=2048

然后：

"num_steps_per_env":24,
9. 你的训练命令改

现在：

./isaaclab.sh -p train_dog_rsl.py --headless --num_envs 2048 --max_iterations 3000

修改 argparse 后：

运行：

./isaaclab.sh -p train_dog_rsl.py \
--headless \
--num_envs 2048 \
--max_iterations 3000

启动应该显示：

Environment created with 2048 envs

不是：

4096 envs
