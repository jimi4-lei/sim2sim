从你贴出的 IsaacLab + RSL-RL 训练日志来看，这个情况不太正常。尤其是你说“最开始 reward 约 -33，现在到了 -21000”，这通常不是正常的训练波动，而是说明奖励函数某一项发生了严重恶化，或者 reward 计算尺度出现了问题。

我们先看你现在的数据：

Mean reward: -21064.80

主要负奖励：
Episode_Reward/dof_acc_l2: -26.4676
Metrics/base_velocity/error_vel_yaw: 6.5867
Mean value loss: 157
success_rate: 0
但是注意：

Episode_Reward/dof_torques_l2: -0.0006
Episode_Reward/action_rate_l2: -0.0000
Episode_Reward/undesired_contacts: -0.0244
这些都正常。

真正异常的是：

Mean reward 从 -33 → -21000
这个数量级变化，99%不是机器人学不会，而是 reward scaling 出问题或者某个 reward 项爆炸。

1. 最大可能原因：某个 reward 项没有做归一化（最常见）
你现在：

Episode_Reward/dof_acc_l2: -26
看起来不算巨大。

但是最终：

Mean reward = -21000
说明一定还有一个隐藏项没有打印出来，或者 reward term 被累计了。

例如：

reward = -torch.sum(joint_acc ** 2)
如果：

joint_acc = 100
那么：

100² = 10000
12个关节：

120000
一次 episode 累积：

-21000
非常容易出现。

正常 IsaacLab 四足训练：

类似：

alive                 +0.1
track_lin_vel         +1~5
orientation           -0.1~-2
torque                -0.001~-0.1
action_rate           -0.01~-1
reward 一般：

-5 ~ +10
或者：

-100 ~ +100
不会：

-20000
2. 你的 value loss 也说明 critic 崩了
你的：

Mean value loss: 158
这个其实已经比较高。

如果 reward 从：

-33
突然变成：

-21000
那么 PPO 的 value network 会认为：

下一步价值突然下降几万个单位

于是：

critic 学不动。

表现就是：

surrogate loss ≈0
entropy loss=0
success_rate=0
你这里：

Mean surrogate loss: -0.0000
Mean entropy loss:0
Mean action std:1.00
说明策略基本没有有效更新。

3. 检查 reward term 的权重
重点检查你的 cfg：

例如：

rewards = RewTermCfg(
    func=xxx,
    weight=-1.0
)
有没有类似：

weight=-100
weight=-1000
特别是：

角速度
track_ang_vel_z_exp
加速度
dof_acc_l2
姿态
flat_orientation_l2
速度误差
track_lin_vel_xy_exp
这些最容易爆。

4. 你的 episode length 很奇怪
你这里：

Mean episode length:10887
如果你的：

decimation=4
dt=0.005
那么：

10887 step≈217秒

四足机器人训练一般：

20~100秒
比较多。

如果 episode 太长：

reward 是累积的：

比如：

每一步：

-2 reward
episode：

10000 step
最后：

-20000
所以需要看：

你的 reward 是：

每 step reward
还是

episode sum reward
RSL-RL 的：

Mean reward
通常是 episode accumulated reward。

5. 为什么刚开始 -33，现在 -21000？
可能过程：

初始：

机器人很快摔倒：

episode:

100 step
reward:

-33
后来：

机器人没有摔倒：

episode:

10000 step
虽然每一步 reward 稍微负：

-2
累计：

-20000
所以：

reward 变负，不一定代表变差。

你需要看：

reward / episode length
而不是总 reward。

比如：

以前：

reward=-33
length=50

平均：
-0.66 step
现在：

reward=-21000
length=10800

平均：
-1.94 step
那才是真的变差。

6. 建议你马上检查三个东西
(1) 打印单步 reward
在环境里加：

print(torch.mean(reward))
不要看 episode reward。

(2) 看所有 reward term
你的日志里面没有全部列出来。

打开：

env.cfg.rewards
把所有：

Episode_Reward/xxx
打印出来。

找有没有：

-100
-1000
-10000
的项。

(3) 看机器人是否真的在学
不要只看 reward。

播放 checkpoint：

观察：

是否站起来？

是否保持身体水平？

是否能跟踪速度？

是否疯狂抖腿？

如果机器人：

✅ 能站
✅ 能走
✅ reward越来越负

那是 reward 设计问题。

如果：

❌ 摔倒
❌ 原地震动
❌ 动作越来越乱

那是 PPO 崩了。

根据你这份日志，我的判断排序：
概率：

原因	可能性
episode reward 累积导致数值变大	⭐⭐⭐⭐⭐
dof_acc 或某个 reward term 爆炸	⭐⭐⭐⭐
episode length 太长	⭐⭐⭐⭐
PPO critic 崩	⭐⭐⭐
机器人策略退化	⭐⭐
你可以再发我两个东西，我基本能直接定位：

你的 rewards 配置（cfg.py 里面 rewards 那部分）

训练最开始 iteration 0~20 的日志（尤其 reward、episode length）

另外最好告诉我：

你用的是 IsaacLab 哪个版本？

机器人是 Unitree Go1 / A1 / 自定义？

是训练 velocity tracking 还是 rough terrain locomotion？



方案1（最推荐）：Isaac Lab + rsl_rl ⭐⭐⭐⭐⭐
适合你目前情况

你的硬件：

RTX5060
Ubuntu
Isaac Lab
四足URDF

非常匹配。

目前很多四足论文路线：

Isaac Gym / Isaac Lab
        |
        |
       PPO
        |
        |
    policy.pt
        |
        |
    sim2sim
        |
        |
    真机
训练流程
第一步：准备机器人模型

你的：

dog.urdf
meshes/

转换：

URDF
 |
 |
Isaac Lab Articulation
 |
 |
RobotCfg

例如：

dog_cfg.py

里面定义：

关节
电机
PD参数
质量
摩擦
初始姿态
第二步：创建环境

Isaac Lab:

ManagerBasedRLEnv

结构：

dog_flat_env_cfg.py

里面定义：

Observation

例如：

机器人状态：

base linear velocity
base angular velocity

joint position
joint velocity

gravity projection

command velocity

48维类似你现在：

obs=48
Action

四足12自由度：

12 joints

action:
[
hip1,
thigh1,
calf1,

...
]
Reward

平地走：

一般：

前进奖励
reward += vx
速度跟踪

希望：

command:

vx=1m/s

机器人：

actual vx=1

奖励：

exp(-(vx-command)^2)
姿态稳定

防止翻：

penalty:

roll
pitch
能耗

限制：

joint torque
摔倒惩罚
termination
第三步 PPO训练

rsl_rl:

配置：

num_envs=4096

num_steps_per_env=24

训练：

./isaaclab.sh \
-p train.py \
--headless

得到：

model_*.pt
第四步 sim2sim

训练环境：

Isaac Lab

验证环境：

例如：

Gazebo / MuJoCo

流程：

Isaac Lab
    |
    |
policy
    |
    |
export ONNX
    |
    |
Gazebo
    |
    |
ROS2 controller
