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


未选择任何文件




################################################################################
                          Learning iteration 218/3000                            

                            Total steps: 21528576 
                       Steps per second: 2691 
                        Collection time: 1.469s 
                          Learning time: 35.054s 
                        Mean value loss: 205.3018
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -17542.06
                    Mean episode length: 9391.27
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0004
     Episode_Reward/track_ang_vel_z_exp: 0.0069
      Episode_Reward/undesired_contacts: -0.0211
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2129
    Episode_Reward/track_lin_vel_xy_exp: 2.6200
     Metrics/base_velocity/error_vel_xy: 1.2822
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -17.7155
                   Episode_Reward/alive: 0.0042
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1297
    Metrics/base_velocity/error_vel_yaw: 5.5350
           Episode_Reward/feet_air_time: -0.0019
           Episode_Reward/ang_vel_xy_l2: -1.1631
     Episode_Reward/flat_orientation_l2: -0.0497
--------------------------------------------------------------------------------
                         Iteration time: 36.52s
                           Time elapsed: 2:13:30
                                    ETA: 1 day, 4:15:26

################################################################################
                          Learning iteration 219/3000                            

                            Total steps: 21626880 
                       Steps per second: 2682 
                        Collection time: 1.440s 
                          Learning time: 35.211s 
                        Mean value loss: 189.3216
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -17542.06
                    Mean episode length: 9391.27
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0072
      Episode_Reward/undesired_contacts: -0.0211
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2139
    Episode_Reward/track_lin_vel_xy_exp: 2.8408
     Metrics/base_velocity/error_vel_xy: 1.2333
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -20.8717
                   Episode_Reward/alive: 0.0042
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1073
    Metrics/base_velocity/error_vel_yaw: 5.6437
           Episode_Reward/feet_air_time: -0.0016
           Episode_Reward/ang_vel_xy_l2: -1.2944
     Episode_Reward/flat_orientation_l2: -0.0502
--------------------------------------------------------------------------------
                         Iteration time: 36.65s
                           Time elapsed: 2:14:07
                                    ETA: 1 day, 4:14:50

################################################################################
                          Learning iteration 220/3000                            

                            Total steps: 21725184 
                       Steps per second: 2697 
                        Collection time: 1.430s 
                          Learning time: 35.010s 
                        Mean value loss: 191.0393
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -17596.49
                    Mean episode length: 9414.61
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0063
      Episode_Reward/undesired_contacts: -0.0212
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2141
    Episode_Reward/track_lin_vel_xy_exp: 2.4975
     Metrics/base_velocity/error_vel_xy: 1.3201
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -20.4121
                   Episode_Reward/alive: 0.0042
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1257
    Metrics/base_velocity/error_vel_yaw: 6.2887
           Episode_Reward/feet_air_time: -0.0016
           Episode_Reward/ang_vel_xy_l2: -1.4586
     Episode_Reward/flat_orientation_l2: -0.0516
--------------------------------------------------------------------------------
                         Iteration time: 36.44s
                           Time elapsed: 2:14:43
                                    ETA: 1 day, 4:14:12

################################################################################
                          Learning iteration 221/3000                            

                            Total steps: 21823488 
                       Steps per second: 2692 
                        Collection time: 1.619s 
                          Learning time: 34.893s 
                        Mean value loss: 186.2815
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -17817.14
                    Mean episode length: 9461.86
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0083
      Episode_Reward/undesired_contacts: -0.0214
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2150
    Episode_Reward/track_lin_vel_xy_exp: 2.2841
     Metrics/base_velocity/error_vel_xy: 1.3061
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -20.8072
                   Episode_Reward/alive: 0.0042
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.0986
    Metrics/base_velocity/error_vel_yaw: 6.0399
           Episode_Reward/feet_air_time: -0.0013
           Episode_Reward/ang_vel_xy_l2: -1.1974
     Episode_Reward/flat_orientation_l2: -0.0466
--------------------------------------------------------------------------------
                         Iteration time: 36.51s
                           Time elapsed: 2:15:20
                                    ETA: 1 day, 4:13:35

################################################################################
                          Learning iteration 222/3000                            

                            Total steps: 21921792 
                       Steps per second: 2705 
                        Collection time: 1.449s 
                          Learning time: 34.889s 
                        Mean value loss: 180.5593
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -17867.39
                    Mean episode length: 9486.23
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0084
      Episode_Reward/undesired_contacts: -0.0214
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2154
    Episode_Reward/track_lin_vel_xy_exp: 2.0275
     Metrics/base_velocity/error_vel_xy: 1.3878
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -21.1483
                   Episode_Reward/alive: 0.0043
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1016
    Metrics/base_velocity/error_vel_yaw: 6.8740
           Episode_Reward/feet_air_time: -0.0010
           Episode_Reward/ang_vel_xy_l2: -1.1564
     Episode_Reward/flat_orientation_l2: -0.0452
--------------------------------------------------------------------------------
                         Iteration time: 36.34s
                           Time elapsed: 2:15:56
                                    ETA: 1 day, 4:12:55

################################################################################
                          Learning iteration 223/3000                            

                            Total steps: 22020096 
                       Steps per second: 2680 
                        Collection time: 1.563s 
                          Learning time: 35.108s 
                        Mean value loss: 180.7261
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -17867.39
                    Mean episode length: 9486.23
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0004
     Episode_Reward/track_ang_vel_z_exp: 0.0077
      Episode_Reward/undesired_contacts: -0.0215
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2158
    Episode_Reward/track_lin_vel_xy_exp: 2.6424
     Metrics/base_velocity/error_vel_xy: 1.1339
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -21.0339
                   Episode_Reward/alive: 0.0043
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.0554
    Metrics/base_velocity/error_vel_yaw: 4.1879
           Episode_Reward/feet_air_time: -0.0012
           Episode_Reward/ang_vel_xy_l2: -0.6941
     Episode_Reward/flat_orientation_l2: -0.0469
--------------------------------------------------------------------------------
                         Iteration time: 36.67s
                           Time elapsed: 2:16:33
                                    ETA: 1 day, 4:12:20

################################################################################
                          Learning iteration 224/3000                            

                            Total steps: 22118400 
                       Steps per second: 2685 
                        Collection time: 1.599s 
                          Learning time: 35.007s 
                        Mean value loss: 188.0860
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -17797.49
                    Mean episode length: 9560.16
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0004
     Episode_Reward/track_ang_vel_z_exp: 0.0104
      Episode_Reward/undesired_contacts: -0.0216
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2171
    Episode_Reward/track_lin_vel_xy_exp: 2.3349
     Metrics/base_velocity/error_vel_xy: 1.2102
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -18.6696
                   Episode_Reward/alive: 0.0043
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.0905
    Metrics/base_velocity/error_vel_yaw: 4.3736
           Episode_Reward/feet_air_time: -0.0016
           Episode_Reward/ang_vel_xy_l2: -0.8682
     Episode_Reward/flat_orientation_l2: -0.0393
--------------------------------------------------------------------------------
                         Iteration time: 36.61s
                           Time elapsed: 2:17:10
                                    ETA: 1 day, 4:11:43

################################################################################
                          Learning iteration 225/3000                            

                            Total steps: 22216704 
                       Steps per second: 2707 
                        Collection time: 1.500s 
                          Learning time: 34.808s 
                        Mean value loss: 177.4607
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -17797.49
                    Mean episode length: 9560.16
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0004
     Episode_Reward/track_ang_vel_z_exp: 0.0110
      Episode_Reward/undesired_contacts: -0.0216
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2173
    Episode_Reward/track_lin_vel_xy_exp: 2.2887
     Metrics/base_velocity/error_vel_xy: 1.1986
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -18.2845
                   Episode_Reward/alive: 0.0043
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.0878
    Metrics/base_velocity/error_vel_yaw: 4.2246
           Episode_Reward/feet_air_time: -0.0016
           Episode_Reward/ang_vel_xy_l2: -0.7614
     Episode_Reward/flat_orientation_l2: -0.0371
--------------------------------------------------------------------------------
                         Iteration time: 36.31s
                           Time elapsed: 2:17:46
                                    ETA: 1 day, 4:11:04

################################################################################
                          Learning iteration 226/3000                            

                            Total steps: 22315008 
                       Steps per second: 2702 
                        Collection time: 1.484s 
                          Learning time: 34.896s 
                        Mean value loss: 169.3776
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -17880.80
                    Mean episode length: 9633.04
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0070
      Episode_Reward/undesired_contacts: -0.0218
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2179
    Episode_Reward/track_lin_vel_xy_exp: 2.6154
     Metrics/base_velocity/error_vel_xy: 1.2569
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -20.3250
                   Episode_Reward/alive: 0.0043
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.0966
    Metrics/base_velocity/error_vel_yaw: 6.0251
           Episode_Reward/feet_air_time: -0.0016
           Episode_Reward/ang_vel_xy_l2: -0.9126
     Episode_Reward/flat_orientation_l2: -0.0450
--------------------------------------------------------------------------------
                         Iteration time: 36.38s
                           Time elapsed: 2:18:22
                                    ETA: 1 day, 4:10:25

################################################################################
                          Learning iteration 227/3000                            

                            Total steps: 22413312 
                       Steps per second: 2699 
                        Collection time: 1.494s 
                          Learning time: 34.926s 
                        Mean value loss: 168.1151
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -17976.77
                    Mean episode length: 9679.40
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0004
     Episode_Reward/track_ang_vel_z_exp: 0.0075
      Episode_Reward/undesired_contacts: -0.0219
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2193
    Episode_Reward/track_lin_vel_xy_exp: 3.1877
     Metrics/base_velocity/error_vel_xy: 1.1011
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -20.5698
                   Episode_Reward/alive: 0.0044
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.0687
    Metrics/base_velocity/error_vel_yaw: 4.6867
           Episode_Reward/feet_air_time: -0.0012
           Episode_Reward/ang_vel_xy_l2: -0.8489
     Episode_Reward/flat_orientation_l2: -0.0421
--------------------------------------------------------------------------------
                         Iteration time: 36.42s
                           Time elapsed: 2:18:59
                                    ETA: 1 day, 4:09:46

################################################################################
                          Learning iteration 228/3000                            

                            Total steps: 22511616 
                       Steps per second: 2704 
                        Collection time: 1.408s 
                          Learning time: 34.946s 
                        Mean value loss: 177.9000
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -18011.10
                    Mean episode length: 9724.61
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0088
      Episode_Reward/undesired_contacts: -0.0220
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2204
    Episode_Reward/track_lin_vel_xy_exp: 2.9807
     Metrics/base_velocity/error_vel_xy: 1.1357
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -20.6922
                   Episode_Reward/alive: 0.0044
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.0723
    Metrics/base_velocity/error_vel_yaw: 5.1789
           Episode_Reward/feet_air_time: -0.0014
           Episode_Reward/ang_vel_xy_l2: -0.9003
     Episode_Reward/flat_orientation_l2: -0.0460
--------------------------------------------------------------------------------
                         Iteration time: 36.35s
                           Time elapsed: 2:19:35
                                    ETA: 1 day, 4:09:07

################################################################################
                          Learning iteration 229/3000                            

                            Total steps: 22609920 
                       Steps per second: 2696 
                        Collection time: 1.519s 
                          Learning time: 34.938s 
                        Mean value loss: 184.6150
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -18181.72
                    Mean episode length: 9792.18
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0070
      Episode_Reward/undesired_contacts: -0.0221
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2216
    Episode_Reward/track_lin_vel_xy_exp: 1.8501
     Metrics/base_velocity/error_vel_xy: 1.7229
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -24.5663
                   Episode_Reward/alive: 0.0044
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.2371
    Metrics/base_velocity/error_vel_yaw: 6.5055
           Episode_Reward/feet_air_time: -0.0013
           Episode_Reward/ang_vel_xy_l2: -2.5201
     Episode_Reward/flat_orientation_l2: -0.0546
--------------------------------------------------------------------------------
                         Iteration time: 36.46s
                           Time elapsed: 2:20:11
                                    ETA: 1 day, 4:08:29

################################################################################
                          Learning iteration 230/3000                            

                            Total steps: 22708224 
                       Steps per second: 2690 
                        Collection time: 1.550s 
                          Learning time: 34.980s 
                        Mean value loss: 165.8273
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -18242.58
                    Mean episode length: 9837.22
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0082
      Episode_Reward/undesired_contacts: -0.0222
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2229
    Episode_Reward/track_lin_vel_xy_exp: 2.6044
     Metrics/base_velocity/error_vel_xy: 1.1575
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -21.0464
                   Episode_Reward/alive: 0.0044
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.0727
    Metrics/base_velocity/error_vel_yaw: 4.9431
           Episode_Reward/feet_air_time: -0.0017
           Episode_Reward/ang_vel_xy_l2: -0.9178
     Episode_Reward/flat_orientation_l2: -0.0421
--------------------------------------------------------------------------------
                         Iteration time: 36.53s
                           Time elapsed: 2:20:48
                                    ETA: 1 day, 4:07:52

################################################################################
                          Learning iteration 231/3000                            

                            Total steps: 22806528 
                       Steps per second: 2696 
                        Collection time: 1.464s 
                          Learning time: 34.990s 
                        Mean value loss: 171.8232
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -18322.75
                    Mean episode length: 9882.00
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0088
      Episode_Reward/undesired_contacts: -0.0223
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2240
    Episode_Reward/track_lin_vel_xy_exp: 2.8654
     Metrics/base_velocity/error_vel_xy: 1.2164
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -22.3461
                   Episode_Reward/alive: 0.0044
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.0968
    Metrics/base_velocity/error_vel_yaw: 4.7871
           Episode_Reward/feet_air_time: -0.0016
           Episode_Reward/ang_vel_xy_l2: -1.1227
     Episode_Reward/flat_orientation_l2: -0.0498
--------------------------------------------------------------------------------
                         Iteration time: 36.45s
                           Time elapsed: 2:21:24
                                    ETA: 1 day, 4:07:14

################################################################################
                          Learning iteration 232/3000                            

                            Total steps: 22904832 
                       Steps per second: 2685 
                        Collection time: 1.607s 
                          Learning time: 35.005s 
                        Mean value loss: 166.0550
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -18376.13
                    Mean episode length: 9904.94
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0086
      Episode_Reward/undesired_contacts: -0.0223
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2242
    Episode_Reward/track_lin_vel_xy_exp: 2.9405
     Metrics/base_velocity/error_vel_xy: 1.2196
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -22.2682
                   Episode_Reward/alive: 0.0044
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1030
    Metrics/base_velocity/error_vel_yaw: 4.6219
           Episode_Reward/feet_air_time: -0.0017
           Episode_Reward/ang_vel_xy_l2: -1.2130
     Episode_Reward/flat_orientation_l2: -0.0512
--------------------------------------------------------------------------------
                         Iteration time: 36.61s
                           Time elapsed: 2:22:01
                                    ETA: 1 day, 4:06:38
################################################################################
                          Learning iteration 233/3000                            

                            Total steps: 23003136 
                       Steps per second: 2700 
                        Collection time: 1.488s 
                          Learning time: 34.908s 
                        Mean value loss: 167.9399
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -18524.56
                    Mean episode length: 9951.13
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0065
      Episode_Reward/undesired_contacts: -0.0224
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2250
    Episode_Reward/track_lin_vel_xy_exp: 2.0662
     Metrics/base_velocity/error_vel_xy: 2.0271
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -24.8429
                   Episode_Reward/alive: 0.0045
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.4026
    Metrics/base_velocity/error_vel_yaw: 7.7683
           Episode_Reward/feet_air_time: -0.0011
           Episode_Reward/ang_vel_xy_l2: -3.0021
     Episode_Reward/flat_orientation_l2: -0.0603
--------------------------------------------------------------------------------
                         Iteration time: 36.40s
                           Time elapsed: 2:22:37
                                    ETA: 1 day, 4:05:59

################################################################################
                          Learning iteration 234/3000                            

                            Total steps: 23101440 
                       Steps per second: 2692 
                        Collection time: 1.422s 
                          Learning time: 35.083s 
                        Mean value loss: 174.1616
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -18522.25
                    Mean episode length: 9974.50
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0075
      Episode_Reward/undesired_contacts: -0.0227
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2259
    Episode_Reward/track_lin_vel_xy_exp: 2.8682
     Metrics/base_velocity/error_vel_xy: 1.1326
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -21.8329
                   Episode_Reward/alive: 0.0045
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.0650
    Metrics/base_velocity/error_vel_yaw: 5.2330
           Episode_Reward/feet_air_time: -0.0012
           Episode_Reward/ang_vel_xy_l2: -0.7925
     Episode_Reward/flat_orientation_l2: -0.0449
--------------------------------------------------------------------------------
                         Iteration time: 36.51s
                           Time elapsed: 2:23:14
                                    ETA: 1 day, 4:05:22

################################################################################
                          Learning iteration 235/3000                            

                            Total steps: 23199744 
                       Steps per second: 2688 
                        Collection time: 1.609s 
                          Learning time: 34.962s 
                        Mean value loss: 169.9118
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -18679.75
                    Mean episode length: 10021.70
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0073
      Episode_Reward/undesired_contacts: -0.0227
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2268
    Episode_Reward/track_lin_vel_xy_exp: 2.2835
     Metrics/base_velocity/error_vel_xy: 1.4621
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -22.8458
                   Episode_Reward/alive: 0.0045
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1365
    Metrics/base_velocity/error_vel_yaw: 7.4291
           Episode_Reward/feet_air_time: -0.0011
           Episode_Reward/ang_vel_xy_l2: -1.2263
     Episode_Reward/flat_orientation_l2: -0.0486
--------------------------------------------------------------------------------
                         Iteration time: 36.57s
                           Time elapsed: 2:23:51
                                    ETA: 1 day, 4:04:45

################################################################################
                          Learning iteration 236/3000                            

                            Total steps: 23298048 
                       Steps per second: 2698 
                        Collection time: 1.408s 
                          Learning time: 35.024s 
                        Mean value loss: 170.4579
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -18916.21
                    Mean episode length: 10068.99
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0054
      Episode_Reward/undesired_contacts: -0.0227
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2279
    Episode_Reward/track_lin_vel_xy_exp: 1.2488
     Metrics/base_velocity/error_vel_xy: 2.2239
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -27.2366
                   Episode_Reward/alive: 0.0045
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.3966
    Metrics/base_velocity/error_vel_yaw: 8.4040
           Episode_Reward/feet_air_time: -0.0010
           Episode_Reward/ang_vel_xy_l2: -3.8725
     Episode_Reward/flat_orientation_l2: -0.0630
--------------------------------------------------------------------------------
                         Iteration time: 36.43s
                           Time elapsed: 2:24:27
                                    ETA: 1 day, 4:04:07

################################################################################
                          Learning iteration 237/3000                            

                            Total steps: 23396352 
                       Steps per second: 2690 
                        Collection time: 1.483s 
                          Learning time: 35.049s 
                        Mean value loss: 169.3289
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -19298.16
                    Mean episode length: 10138.30
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0053
      Episode_Reward/undesired_contacts: -0.0228
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2284
    Episode_Reward/track_lin_vel_xy_exp: 1.3599
     Metrics/base_velocity/error_vel_xy: 2.2670
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -27.8495
                   Episode_Reward/alive: 0.0045
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.4615
    Metrics/base_velocity/error_vel_yaw: 7.9938
           Episode_Reward/feet_air_time: -0.0011
           Episode_Reward/ang_vel_xy_l2: -3.9979
     Episode_Reward/flat_orientation_l2: -0.0638
--------------------------------------------------------------------------------
                         Iteration time: 36.53s
                           Time elapsed: 2:25:04
                                    ETA: 1 day, 4:03:30

################################################################################
                          Learning iteration 238/3000                            

                            Total steps: 23494656 
                       Steps per second: 2690 
                        Collection time: 1.539s 
                          Learning time: 34.993s 
                        Mean value loss: 168.1225
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -19582.40
                    Mean episode length: 10253.14
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0076
      Episode_Reward/undesired_contacts: -0.0230
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2303
    Episode_Reward/track_lin_vel_xy_exp: 2.7405
     Metrics/base_velocity/error_vel_xy: 1.2568
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -21.7903
                   Episode_Reward/alive: 0.0046
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.0972
    Metrics/base_velocity/error_vel_yaw: 5.2636
           Episode_Reward/feet_air_time: -0.0017
           Episode_Reward/ang_vel_xy_l2: -1.2969
     Episode_Reward/flat_orientation_l2: -0.0473
--------------------------------------------------------------------------------
                         Iteration time: 36.53s
                           Time elapsed: 2:25:40
                                    ETA: 1 day, 4:02:53

################################################################################
                          Learning iteration 239/3000                            

                            Total steps: 23592960 
                       Steps per second: 2685 
                        Collection time: 1.382s 
                          Learning time: 35.221s 
                        Mean value loss: 181.1042
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -19582.40
                    Mean episode length: 10253.14
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0088
      Episode_Reward/undesired_contacts: -0.0231
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2319
    Episode_Reward/track_lin_vel_xy_exp: 2.7539
     Metrics/base_velocity/error_vel_xy: 1.1152
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -23.2380
                   Episode_Reward/alive: 0.0046
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.0573
    Metrics/base_velocity/error_vel_yaw: 4.3149
           Episode_Reward/feet_air_time: -0.0016
           Episode_Reward/ang_vel_xy_l2: -0.7825
     Episode_Reward/flat_orientation_l2: -0.0444
--------------------------------------------------------------------------------
                         Iteration time: 36.60s
                           Time elapsed: 2:26:17
                                    ETA: 1 day, 4:02:17

################################################################################
                          Learning iteration 240/3000                            

                            Total steps: 23691264 
                       Steps per second: 2683 
                        Collection time: 1.443s 
                          Learning time: 35.185s 
                        Mean value loss: 162.2141
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -19615.23
                    Mean episode length: 10298.91
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0085
      Episode_Reward/undesired_contacts: -0.0232
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2325
    Episode_Reward/track_lin_vel_xy_exp: 3.3045
     Metrics/base_velocity/error_vel_xy: 1.1940
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -20.5219
                   Episode_Reward/alive: 0.0046
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.0998
    Metrics/base_velocity/error_vel_yaw: 4.8638
           Episode_Reward/feet_air_time: -0.0017
           Episode_Reward/ang_vel_xy_l2: -1.2027
     Episode_Reward/flat_orientation_l2: -0.0494
--------------------------------------------------------------------------------
                         Iteration time: 36.63s
                           Time elapsed: 2:26:53
                                    ETA: 1 day, 4:01:41

################################################################################
                          Learning iteration 241/3000                            

                            Total steps: 23789568 
                       Steps per second: 2686 
                        Collection time: 1.440s 
                          Learning time: 35.155s 
                        Mean value loss: 163.6630
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -19851.61
                    Mean episode length: 10345.66
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0078
      Episode_Reward/undesired_contacts: -0.0232
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2331
    Episode_Reward/track_lin_vel_xy_exp: 3.4380
     Metrics/base_velocity/error_vel_xy: 1.2297
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -21.1482
                   Episode_Reward/alive: 0.0046
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1229
    Metrics/base_velocity/error_vel_yaw: 5.5730
           Episode_Reward/feet_air_time: -0.0018
           Episode_Reward/ang_vel_xy_l2: -1.3869
     Episode_Reward/flat_orientation_l2: -0.0516
--------------------------------------------------------------------------------
                         Iteration time: 36.59s
                           Time elapsed: 2:27:30
                                    ETA: 1 day, 4:01:04

################################################################################
                          Learning iteration 242/3000                            

                            Total steps: 23887872 
                       Steps per second: 2691 
                        Collection time: 1.462s 
                          Learning time: 35.059s 
                        Mean value loss: 163.2455
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -19918.27
                    Mean episode length: 10368.91
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0121
      Episode_Reward/undesired_contacts: -0.0234
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2343
    Episode_Reward/track_lin_vel_xy_exp: 2.4811
     Metrics/base_velocity/error_vel_xy: 1.2052
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -22.9855
                   Episode_Reward/alive: 0.0046
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.0903
    Metrics/base_velocity/error_vel_yaw: 4.5155
           Episode_Reward/feet_air_time: -0.0012
           Episode_Reward/ang_vel_xy_l2: -0.8478
     Episode_Reward/flat_orientation_l2: -0.0400
--------------------------------------------------------------------------------
                         Iteration time: 36.52s
                           Time elapsed: 2:28:06
                                    ETA: 1 day, 4:00:27

################################################################################
                          Learning iteration 243/3000                            

                            Total steps: 23986176 
                       Steps per second: 2697 
                        Collection time: 1.415s 
                          Learning time: 35.032s 
                        Mean value loss: 161.1878
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -20095.79
                    Mean episode length: 10440.44
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0087
      Episode_Reward/undesired_contacts: -0.0234
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2349
    Episode_Reward/track_lin_vel_xy_exp: 2.6153
     Metrics/base_velocity/error_vel_xy: 1.2762
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -23.3016
                   Episode_Reward/alive: 0.0047
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1218
    Metrics/base_velocity/error_vel_yaw: 5.1973
           Episode_Reward/feet_air_time: -0.0013
           Episode_Reward/ang_vel_xy_l2: -1.2018
     Episode_Reward/flat_orientation_l2: -0.0444
--------------------------------------------------------------------------------
                         Iteration time: 36.45s
                           Time elapsed: 2:28:43
                                    ETA: 1 day, 3:59:49

################################################################################
                          Learning iteration 244/3000                            

                            Total steps: 24084480 
                       Steps per second: 2692 
                        Collection time: 1.468s 
                          Learning time: 35.040s 
                        Mean value loss: 169.1965
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -20095.79
                    Mean episode length: 10440.44
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0066
      Episode_Reward/undesired_contacts: -0.0235
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2358
    Episode_Reward/track_lin_vel_xy_exp: 2.4149
     Metrics/base_velocity/error_vel_xy: 1.4858
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -24.5649
                   Episode_Reward/alive: 0.0047
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1510
    Metrics/base_velocity/error_vel_yaw: 9.1042
           Episode_Reward/feet_air_time: -0.0012
           Episode_Reward/ang_vel_xy_l2: -1.4743
     Episode_Reward/flat_orientation_l2: -0.0562
--------------------------------------------------------------------------------
                         Iteration time: 36.51s
                           Time elapsed: 2:29:19
                                    ETA: 1 day, 3:59:12

################################################################################
                          Learning iteration 245/3000                            

                            Total steps: 24182784 
                       Steps per second: 2671 
                        Collection time: 1.510s 
                          Learning time: 35.282s 
                        Mean value loss: 164.5078
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -20180.27
                    Mean episode length: 10465.09
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0068
      Episode_Reward/undesired_contacts: -0.0236
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2360
    Episode_Reward/track_lin_vel_xy_exp: 2.3169
     Metrics/base_velocity/error_vel_xy: 1.4678
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -24.5537
                   Episode_Reward/alive: 0.0047
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1462
    Metrics/base_velocity/error_vel_yaw: 8.8412
           Episode_Reward/feet_air_time: -0.0013
           Episode_Reward/ang_vel_xy_l2: -1.4204
     Episode_Reward/flat_orientation_l2: -0.0553
--------------------------------------------------------------------------------
                         Iteration time: 36.79s
                           Time elapsed: 2:29:56
                                    ETA: 1 day, 3:58:38

################################################################################
                          Learning iteration 246/3000                            

                            Total steps: 24281088 
                       Steps per second: 2677 
                        Collection time: 1.498s 
                          Learning time: 35.211s 
                        Mean value loss: 159.5327
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -20180.27
                    Mean episode length: 10465.09
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0071
      Episode_Reward/undesired_contacts: -0.0237
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2363
    Episode_Reward/track_lin_vel_xy_exp: 2.1207
     Metrics/base_velocity/error_vel_xy: 1.4318
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -24.5311
                   Episode_Reward/alive: 0.0047
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1368
    Metrics/base_velocity/error_vel_yaw: 8.3152
           Episode_Reward/feet_air_time: -0.0014
           Episode_Reward/ang_vel_xy_l2: -1.3125
     Episode_Reward/flat_orientation_l2: -0.0537
--------------------------------------------------------------------------------
                         Iteration time: 36.71s
                           Time elapsed: 2:30:33
                                    ETA: 1 day, 3:58:03

################################################################################
                          Learning iteration 247/3000                            

                            Total steps: 24379392 
                       Steps per second: 2724 
                        Collection time: 1.487s 
                          Learning time: 34.599s 
                        Mean value loss: 163.4116
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -20048.68
                    Mean episode length: 10515.74
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0099
      Episode_Reward/undesired_contacts: -0.0238
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2369
    Episode_Reward/track_lin_vel_xy_exp: 2.6327
     Metrics/base_velocity/error_vel_xy: 1.3938
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -20.5519
                   Episode_Reward/alive: 0.0047
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1699
    Metrics/base_velocity/error_vel_yaw: 5.9509
           Episode_Reward/feet_air_time: -0.0018
           Episode_Reward/ang_vel_xy_l2: -1.4296
     Episode_Reward/flat_orientation_l2: -0.0589
--------------------------------------------------------------------------------
                         Iteration time: 36.09s
                           Time elapsed: 2:31:09
                                    ETA: 1 day, 3:57:21

################################################################################
                          Learning iteration 248/3000                            

                            Total steps: 24477696 
                       Steps per second: 2717 
                        Collection time: 1.433s 
                          Learning time: 34.742s 
                        Mean value loss: 160.5909
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -20075.87
                    Mean episode length: 10541.51
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0087
      Episode_Reward/undesired_contacts: -0.0239
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2373
    Episode_Reward/track_lin_vel_xy_exp: 2.5596
     Metrics/base_velocity/error_vel_xy: 1.6677
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -19.6365
                   Episode_Reward/alive: 0.0048
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.2846
    Metrics/base_velocity/error_vel_yaw: 6.3177
           Episode_Reward/feet_air_time: -0.0015
           Episode_Reward/ang_vel_xy_l2: -2.0730
     Episode_Reward/flat_orientation_l2: -0.0687
--------------------------------------------------------------------------------
                         Iteration time: 36.17s
                           Time elapsed: 2:31:45
                                    ETA: 1 day, 3:56:40

################################################################################
                          Learning iteration 249/3000                            

                            Total steps: 24576000 
                       Steps per second: 2709 
                        Collection time: 1.493s 
                          Learning time: 34.786s 
                        Mean value loss: 187.3772
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -20191.65
                    Mean episode length: 10593.22
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0079
      Episode_Reward/undesired_contacts: -0.0240
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2385
    Episode_Reward/track_lin_vel_xy_exp: 2.6460
     Metrics/base_velocity/error_vel_xy: 1.3286
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -23.6847
                   Episode_Reward/alive: 0.0048
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1258
    Metrics/base_velocity/error_vel_yaw: 6.4679
           Episode_Reward/feet_air_time: -0.0016
           Episode_Reward/ang_vel_xy_l2: -1.3414
     Episode_Reward/flat_orientation_l2: -0.0554
--------------------------------------------------------------------------------
                         Iteration time: 36.28s
                           Time elapsed: 2:32:21
                                    ETA: 1 day, 3:56:00

################################################################################
                          Learning iteration 250/3000                            

                            Total steps: 24674304 
                       Steps per second: 2702 
                        Collection time: 1.609s 
                          Learning time: 34.772s 
                        Mean value loss: 161.0086
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -20497.68
                    Mean episode length: 10645.47
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0080
      Episode_Reward/undesired_contacts: -0.0241
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2389
    Episode_Reward/track_lin_vel_xy_exp: 2.5561
     Metrics/base_velocity/error_vel_xy: 1.5644
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -26.1375
                   Episode_Reward/alive: 0.0048
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.2483
    Metrics/base_velocity/error_vel_yaw: 5.7912
           Episode_Reward/feet_air_time: -0.0016
           Episode_Reward/ang_vel_xy_l2: -2.4219
     Episode_Reward/flat_orientation_l2: -0.0581
--------------------------------------------------------------------------------
                         Iteration time: 36.38s
                           Time elapsed: 2:32:58
                                    ETA: 1 day, 3:55:22

################################################################################
                          Learning iteration 251/3000                            

                            Total steps: 24772608 
                       Steps per second: 2714 
                        Collection time: 1.460s 
                          Learning time: 34.760s 
                        Mean value loss: 162.6044
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -20879.20
                    Mean episode length: 10724.29
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0077
      Episode_Reward/undesired_contacts: -0.0242
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2405
    Episode_Reward/track_lin_vel_xy_exp: 2.5098
     Metrics/base_velocity/error_vel_xy: 1.4275
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -25.7060
                   Episode_Reward/alive: 0.0048
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1739
    Metrics/base_velocity/error_vel_yaw: 5.7419
           Episode_Reward/feet_air_time: -0.0016
           Episode_Reward/ang_vel_xy_l2: -1.6680
     Episode_Reward/flat_orientation_l2: -0.0543
--------------------------------------------------------------------------------
                         Iteration time: 36.22s
                           Time elapsed: 2:33:34
                                    ETA: 1 day, 3:54:41

################################################################################
                          Learning iteration 252/3000                            

                            Total steps: 24870912 
                       Steps per second: 2702 
                        Collection time: 1.404s 
                          Learning time: 34.974s 
                        Mean value loss: 154.8549
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -20879.20
                    Mean episode length: 10724.29
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0006
     Episode_Reward/track_ang_vel_z_exp: 0.0079
      Episode_Reward/undesired_contacts: -0.0242
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2412
    Episode_Reward/track_lin_vel_xy_exp: 1.9991
     Metrics/base_velocity/error_vel_xy: 1.7031
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -29.4665
                   Episode_Reward/alive: 0.0048
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.2761
    Metrics/base_velocity/error_vel_yaw: 5.6148
           Episode_Reward/feet_air_time: -0.0016
           Episode_Reward/ang_vel_xy_l2: -2.5720
     Episode_Reward/flat_orientation_l2: -0.0607
--------------------------------------------------------------------------------
                         Iteration time: 36.38s
                           Time elapsed: 2:34:10
                                    ETA: 1 day, 3:54:03

################################################################################
                          Learning iteration 253/3000                            

                            Total steps: 24969216 
                       Steps per second: 2722 
                        Collection time: 1.426s 
                          Learning time: 34.679s 
                        Mean value loss: 159.3726
                    Mean surrogate loss: 0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -21007.93
                    Mean episode length: 10777.57
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0005
     Episode_Reward/track_ang_vel_z_exp: 0.0078
      Episode_Reward/undesired_contacts: -0.0244
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2416
    Episode_Reward/track_lin_vel_xy_exp: 2.5165
     Metrics/base_velocity/error_vel_xy: 1.4665
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -24.6624
                   Episode_Reward/alive: 0.0049
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1845
    Metrics/base_velocity/error_vel_yaw: 5.7994
           Episode_Reward/feet_air_time: -0.0018
           Episode_Reward/ang_vel_xy_l2: -1.8012
     Episode_Reward/flat_orientation_l2: -0.0603
--------------------------------------------------------------------------------
                         Iteration time: 36.11s
                           Time elapsed: 2:34:46
                                    ETA: 1 day, 3:53:21

################################################################################
                          Learning iteration 254/3000                            

                            Total steps: 25067520 
                       Steps per second: 2701 
                        Collection time: 1.481s 
                          Learning time: 34.907s 
                        Mean value loss: 154.9543
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -21064.80
                    Mean episode length: 10804.14
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0006
     Episode_Reward/track_ang_vel_z_exp: 0.0080
      Episode_Reward/undesired_contacts: -0.0244
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2427
    Episode_Reward/track_lin_vel_xy_exp: 3.0434
     Metrics/base_velocity/error_vel_xy: 1.2012
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -26.4676
                   Episode_Reward/alive: 0.0049
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1011
    Metrics/base_velocity/error_vel_yaw: 6.5867
           Episode_Reward/feet_air_time: -0.0017
           Episode_Reward/ang_vel_xy_l2: -1.3488
     Episode_Reward/flat_orientation_l2: -0.0569
--------------------------------------------------------------------------------
                         Iteration time: 36.39s
                           Time elapsed: 2:35:23
                                    ETA: 1 day, 3:52:43

################################################################################
                          Learning iteration 255/3000                            

                            Total steps: 25165824 
                       Steps per second: 2721 
                        Collection time: 1.521s 
                          Learning time: 34.598s 
                        Mean value loss: 157.4728
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -21064.80
                    Mean episode length: 10804.14
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0006
     Episode_Reward/track_ang_vel_z_exp: 0.0080
      Episode_Reward/undesired_contacts: -0.0244
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2427
    Episode_Reward/track_lin_vel_xy_exp: 3.0434
     Metrics/base_velocity/error_vel_xy: 1.2012
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -26.4676
                   Episode_Reward/alive: 0.0049
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1011
    Metrics/base_velocity/error_vel_yaw: 6.5867
           Episode_Reward/feet_air_time: -0.0017
           Episode_Reward/ang_vel_xy_l2: -1.3488
     Episode_Reward/flat_orientation_l2: -0.0569
--------------------------------------------------------------------------------
                         Iteration time: 36.12s
                           Time elapsed: 2:35:59
                                    ETA: 1 day, 3:52:01

################################################################################
                          Learning iteration 256/3000                            

                            Total steps: 25264128 
                       Steps per second: 2725 
                        Collection time: 1.454s 
                          Learning time: 34.618s 
                        Mean value loss: 158.6105
                    Mean surrogate loss: -0.0000
                      Mean entropy loss: 0.0000
                            Mean reward: -21281.91
                    Mean episode length: 10887.09
                        Mean action std: 1.00
          Episode_Reward/dof_pos_limits: 0.0000
          Episode_Reward/dof_torques_l2: -0.0006
     Episode_Reward/track_ang_vel_z_exp: 0.0087
      Episode_Reward/undesired_contacts: -0.0245
                   Metrics/success_rate: 0.0000
           Episode_Termination/time_out: 0.2429
    Episode_Reward/track_lin_vel_xy_exp: 3.0855
     Metrics/base_velocity/error_vel_xy: 1.1964
          Episode_Reward/action_rate_l2: -0.0000
              Episode_Reward/dof_acc_l2: -26.3127
                   Episode_Reward/alive: 0.0049
       Episode_Termination/base_contact: 0.0000
            Episode_Reward/lin_vel_z_l2: -0.1012
    Metrics/base_velocity/error_vel_yaw: 6.4390
           Episode_Reward/feet_air_time: -0.0016
           Episode_Reward/ang_vel_xy_l2: -1.2953
     Episode_Reward/flat_orientation_l2: -0.0549
--------------------------------------------------------------------------------
                         Iteration time: 36.07s
                           Time elapsed: 2:36:35
                                    ETA: 1 day, 3:51:20




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
