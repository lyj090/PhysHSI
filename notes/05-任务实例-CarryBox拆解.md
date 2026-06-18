---
tags:
  - type/concept
  - area/robotics
  - proj/amp-reproduction
status: active
created: 2026-06-17
---

# 05 任务实例：CarryBox 全流程拆解

> **文档定位**：以 CarryBox（最复杂任务）为例，完整拆解任务设计——观测构建、奖励工程、终止条件、多阶段训练、噪声模型。
>
> **前置**：[[01-Projects/AMP-Reproduction/PhysHSI/04-环境架构与运动库|04 环境架构]]
> **后续**：[[01-Projects/AMP-Reproduction/PhysHSI/06-机器人配置与动作空间|06 机器人配置]]
>
> **源码**：`legged_gym/legged_gym/envs/g1/carrybox.py`（~2074 行）、`carrybox_config.py`

---

## 1. CarryBox 任务概述

### 任务定义

> 机器人从初始位置出发，靠近箱子 → 抱起箱子 → 搬运到目标位置 → 放下箱子。

```
┌──────────────────────────────────────────────────┐
│                    CarryBox 任务流程                │
│                                                   │
│  🤖                    📦              🎯         │
│  机器人                 箱子             目标      │
│   │                     │                │        │
│   ├─ 阶段1: loco ──────▶│  靠近箱子       │        │
│   │                     │                │        │
│   ├─ 阶段2: pickUp ────▶📦 抱起箱子       │        │
│   │                     │                │        │
│   ├─ 阶段3: carryWith ──▶📦 抱箱搬运 ────▶│        │
│   │                     │                │        │
│   └─ 阶段4: putDown ────▶📦 放下箱子 ────▶🎯      │
│                                                   │
└──────────────────────────────────────────────────┘
```

### 四阶段配置

```python
# carrybox_config.py
class box:
    skill = ["loco", "pickUp", "carryWith", "putDown"]
    skill_init_prob = [1.0, 0.0, 0.0, 0.0]  # 初始全为 loco
```

---

## 2. 观测空间设计

### 2.1 观测维度

```python
# carrybox_config.py
num_envs = 1024
num_actions = 29       # G1: 29 DOF
num_dofs = 29
num_proprio_obs = 6 + num_dofs*2 + num_actions + 3*5  # = 6+58+29+15 = 108
num_task_obs = 15      # 箱子+目标信息
num_actor_history = 6  # 6帧历史
num_actor_obs = num_actor_history * (num_proprio_obs + num_task_obs)  # 6*123=738
num_privileged_obs = num_proprio_obs + 3 + num_task_obs  # = 126
```

### 2.2 本体感知观测（Proprioceptive）

```python
# carrybox.py: compute_observations()

current_actor_obs = torch.cat([
    base_ang_vel * obs_scales.ang_vel,          # [3]  基座角速度 (×0.25)
    projected_gravity,                           # [3]  投影重力方向
    (dof_pos - default_dof_pos) * obs_scales.dof_pos,  # [29] 关节位置偏差 (×1.0)
    dof_vel * obs_scales.dof_vel,                # [29] 关节速度 (×0.05)
    end_effector_pos,                            # [15] 末端位置(手×2+脚×2+头)
    actions                                      # [29] 上一时刻动作
], dim=-1)  # 总计 108 维
```

**末端执行器位置细节**：
```python
end_effector_pos = torch.concat([
    left_hand_pos,      # [3] 左手掌
    right_hand_pos,     # [3] 右手掌
    left_foot_pos,      # [3] 左脚
    right_foot_pos,     # [3] 右脚
    head_pos            # [3] 头部
], dim=-1)  # 在机器人局部系下 [15]
```

### 2.3 任务观测（Task）

```python
# carrybox.py: compute_task_observations()

task_obs_critic = torch.cat([
    box_pos_local,       # [3]  箱子相对机器人位置
    box_rot_6d_local,    # [6]  箱子姿态（6D 表示）
    box_size,            # [3]  箱子尺寸
    goal_pos_local       # [3]  目标相对机器人位置
], dim=-1)  # 总计 15 维
```

**观测噪声模型**（模拟真实感知）：
```python
if self.add_noise:
    # 粗粒度感知（距离远或看不到标签）
    is_coarse = (robot2object_dist >= thresh_tag) | ((~can_see_tag) & (~has_seen_tag))
    is_mask = (~can_see_tag) & has_seen_tag & (robot2object_dist < 0.65)

    # 远距离：使用偏移后的默认位置
    box_pos[is_coarse] += far_pos_offset

    # 加入高斯噪声
    box_pos += torch_rand_float(-pos_noise_scale, pos_noise_scale, ...)
    box_rot += quat_noise(ang_noise_scale=5°)

    # 视野遮挡：重置为默认
    box_pos_local[is_mask] = default_zero_pos
    box_rot_local[is_mask] = default_quat
```

> **三种感知模式**：粗粒度（>thresh_tag）→ 偏移估计；精粒度（能看到）→ 真实值+噪声；遮挡 → 用最后已知位置。

### 2.4 特权观测（Privileged）

```python
# 特权观测（仅 Critic 使用，含"作弊"信息）
privileged_obs = torch.cat([
    current_obs,           # 108 维（含 base_lin_vel）
    task_obs_critic,       # 15 维（无噪声）
], dim=-1)  # 126 维
```

**Actor-Critic 不对称设计**：
- Actor 输入：6 帧历史 × 123 维（含噪声本体感受 + 含噪声任务信息）
- Critic 输入：当前帧 × 126 维（含速度真值 + 无噪声任务信息）

---

## 3. 奖励工程

### 3.1 奖励函数全景

```python
# carrybox_config.py
class scales:
    # === 正则化奖励（惩罚） ===
    dof_acc = -1e-7           # 关节加速度
    action_rate = -0.03       # 动作变化率
    torques = -1e-4           # 力矩
    dof_vel = -2e-4           # 关节速度
    dof_pos_limits = -5.0     # 关节限位
    dof_vel_limits = -1e-3    # 关节速度限位
    torque_limits = -0.03     # 力矩限位

    # === 任务奖励 ===
    walk_task = 1.0           # 走近箱子（阶段1）
    carryup_task = 1.0        # 抱起箱子（阶段2）
    relocation_task = 1.5     # 搬运箱子（阶段3）← 最高权重
    standup_task = 0.2        # 放下后站直（阶段4）
```

### 3.2 阶段 1: 靠近箱子（walk_task）

```python
# walk 子奖励
robot2object_pos = 0.0       # 距离箱子（关闭）
robot2object_vel = 1.0       # 朝向箱子速度 ⭐
start_heading = 0.5          # 朝向箱子方向对齐

def _reward_robot2object_vel(self):
    # 奖励机器人朝向箱子的速度分量
    dir_to_object = normalize(robot2object_dir[:, :2])
    vel_toward_object = dot(base_lin_vel[:, :2], dir_to_object)
    return vel_toward_object  # 正值：朝箱子走；负值：远离

def _reward_start_heading(self):
    # 奖励机器人朝向对准箱子
    heading_alignment = dot(base_forward, dir_to_object)
    return heading_alignment
```

### 3.3 阶段 2: 抱起箱子（carryup_task）

```python
# carryup 子奖励
hand_pos = 0.7              # 手靠近箱子抓取点
hand_contact = 0.0           # 手接触（关闭，用位置代替）
box_height = 2.0             # 箱子离地高度

def _reward_hand_pos(self):
    # 双手靠近箱侧
    left_dist = norm(left_hand_pos - left_target_pos)
    right_dist = norm(right_hand_pos - right_target_pos)
    return exp(-5.0 * (left_dist + right_dist))

def _reward_box_height(self):
    # 奖励箱子被抱起（离地高度）
    box_height_from_ground = box_states[:, 2] - ground_height
    target_height = 0.72  # 搬运高度
    return exp(-10.0 * (box_height_from_ground - target_height)**2)
```

### 3.4 阶段 3: 搬运箱子（relocation_task）

```python
# relocation 子奖励
relocation_heading = 0.5        # 朝向目标
relocation_heading_vel = 0.0    # 朝向目标速度（关闭）
robot2goal_pos = 0.0            # 距目标距离（关闭）
robot2goal_vel = 1.0            # 朝向目标速度 ⭐
object2goal_pos = 1.0           # 箱子距目标距离
put_box = 2.0                   # 箱子放到目标上 ⭐⭐

def _reward_object2goal_pos(self):
    # 箱子越接近目标，奖励越大
    dist = norm(box_states[:, :3] - goal_pos)
    return exp(-5.0 * dist / thresh_object2goal)

def _reward_put_box(self):
    # 箱子在目标位置正上方（准备放置）
    is_above = box_pos[:, 2] > goal_pos[:, 2] + 0.05
    xy_close = norm(box_pos[:, :2] - goal_pos[:, :2]) < 0.05
    return is_above.float() * xy_close.float() * 1.0
```

### 3.5 阶段 4: 放下后站直（standup_task）

```python
# standup 子奖励
base_height = 0.0              # 基座高度（关闭）
head_height = 0.5              # 头部高度
stand_still = 1.0              # 站立不动 ⭐
hand_free = 0.5                # 手离开箱子

def _reward_stand_still(self):
    # 放下箱子后，惩罚身体/手部动作
    return -torch.sum(torch.abs(dof_vel), dim=-1)

def _reward_hand_free(self):
    # 手远离箱子（确认已放下）
    left_dist = norm(left_hand_pos - box_pos)
    right_dist = norm(right_hand_pos - box_pos)
    return torch.min(left_dist, right_dist)  # 至少一只手远离
```

### 3.6 CarryBox 特殊设计：stage 阶段掩码

```python
# 奖励组合：各阶段使用不同的 stage 权重
rewards = amp_reward * amp_coef * stage.to(device) + task_reward * (1 - amp_coef)
```

```python
# 阶段切换逻辑（基于进度条件）
# stage=0 (loco):     robot2object_dist > 0.7
# stage=1 (pickUp):   robot2object_dist < 0.7 AND box_height < 0.1
# stage=2 (carryWith): box_height > 0.5
# stage=3 (putDown):  object2goal_dist < 0.2
```

---

## 4. 终止条件

```python
# carrybox.py: check_termination()

def check_termination(self):
    # 1. 非期望接触（头部/躯干撞地）
    self.reset_buf |= torch.any(
        torch.norm(contact_forces[:, termination_contact_indices, :], dim=-1) > 10.0,
        dim=1)

    # 2. 超时
    self.time_out_buf = episode_length_buf > max_episode_length  # 20s
    self.reset_buf |= self.time_out_buf

    # 3. 重力方向异常（摔倒）
    self.reset_buf |= torch.any(
        torch.norm(projected_gravity[:, :2], dim=-1) > 0.8, dim=1)

    # 4. 头部过低
    self.reset_buf |= rigid_body_states[:, head_index, 2] < 0.6

    # 5. 基座过低
    self.reset_buf |= root_states[:, 2] < 0.2

    # 6. 身体倾斜过大
    self.reset_buf |= torch.logical_or(
        torch.abs(roll) > 0.5, torch.abs(pitch) > 1.1)

    # 7. 基座水平速度过大（滑倒）
    self.reset_buf |= torch.norm(base_lin_vel[:, :2], dim=-1) > 3.0

    # 8. 髋关节低于阈值
    self.reset_buf |= torch.any(
        rigid_body_states[:, hip_yaw_indices, 2] < 0.15, dim=1)

    # 9. 箱子倾斜/倒置（仅在精调阶段）
    if box_termination:
        self.reset_buf |= (projected_gravity_box[:, 2] > -0.05)

    # 成功条件（不终止，仅标记）
    self.success_buf = non_tilt_box & (object2goal_dist < thresh)
```

---

## 5. 伪代码：CarryBox 环境 step 全流程

```
Procedure: CarryBox.step(actions)

Input: actions [N, 29]  ← 策略输出（目标关节位置）

1. Clip actions to [-clip_actions, clip_actions]

2. For decimation_steps = 1 to 4:        # PD 控制循环
     torques = Kp * (action - dof_pos) + Kd * (-dof_vel)
     torques += motor_strength * torques + actuation_offset
     gym.set_dof_actuation_force(torques)
     gym.simulate()                       # 5ms 物理步进
     gym.refresh_dof_states()

3. post_physics_step():
     # 3a. 刷新物理状态
     refresh all tensors (dof, root, contact, rigid_body)

     # 3b. 计算派生量
     base_quat, roll, pitch, yaw = euler_from_quat(root_states)
     base_lin_vel = quat_rotate_inv(base_quat, global_vel)
     base_ang_vel = quat_rotate_inv(base_quat, global_ang_vel)
     projected_gravity = quat_rotate_inv(base_quat, [0,0,-1])
     end_effector_pos = compute_ee_positions()
     robot2object_dist = compute_distance()
     ...

     # 3c. 检测标签可见性（模拟视觉）
     _can_see_tag()

     # 3d. 检测终止
     check_termination()

     # 3e. 计算奖励
     for each reward_function:
         task_reward += scale * reward_fn()
     amp_reward = discriminator.predict_reward(amp_obs)
     total_reward = amp_coef * amp_reward + (1-amp_coef) * task_reward

     # 3f. 计算 AMP 观测
     amp_obs = compute_amp_observations()

     # 3g. 重置终止环境
     reset_idx(env_ids)

     # 3h. 计算观测
     compute_observations()

     # 3i. 更新历史缓冲
     last_actions, last_dof_vel, last_root_vel = ...

Output: obs, privileged_obs, rewards, dones, extras, termination_ids, termination_obs, amp_obs
```

---

## 6. 箱子物理与随机化

```python
# carrybox_config.py
class box:
    base_size = [0.3, 0.3, 0.25]    # 基准尺寸 (m)
    use_random = True

    # 尺寸随机化
    scale_range_x = [0.7, 1.3]      # 21cm ~ 39cm
    scale_range_y = [0.7, 1.3]
    scale_range_z = [0.6, 1.4]      # 15cm ~ 35cm

    # 密度随机化
    density_range = [10.0, 100.0]   # kg/m³（影响质量）
    density_default = 50.0

    # 重置模式
    reset_mode = 'default'          # 或 'random', 'hybrid'
    hybrid_init_prob = 0.8

    # 视觉噪声参数
    far_pos_offset = 0.2            # 远距离位置偏移
    pos_noise_scale = 0.05          # 位置噪声标准差 (5cm)
    ang_noise_scale = deg2rad(5)    # 姿态噪声标准差 (5°)
```

---

## 7. 训练命令与流程

```bash
# 阶段 1: 初始训练（20k iters）
python legged_gym/scripts/train.py --task carrybox --headless

# 阶段 2: 精调（30-40k iters，增大 amp_coef）
python legged_gym/scripts/train.py --task carrybox_resume \
    --resume --resume_path logs/amp_carrybox/carrybox_coef0.25/model_20000.pt \
    --headless

# 可视化
python legged_gym/scripts/play.py --task carrybox \
    --resume_path logs/.../model_XXXXX.pt

# 查看参考运动
python legged_gym/scripts/play.py --task carrybox --play_dataset
```

---

## 8. 对迁移的启示

CarryBox 的设计模式可直接应用到轮式机械臂搬运任务：

| CarryBox 设计 | 轮式单臂适配 |
|--------------|------------|
| 4 阶段技能链 | 简化为 2-3 阶段：导航→抓取→放置 |
| 双手抱箱 | 单臂末端抓取（五指手或吸盘） |
| 双足行走靠近 | 轮式差速导航靠近 |
| 箱子 6D 位姿观测 | 物体位姿观测（含目标架位姿） |
| 感知噪声模型 | 直接复用（远/近/遮挡三模式） |
| 奖励：robot2object_vel | 改为 base_to_object 导航奖励 |
| 奖励：box_height | 改为 gripper_to_object 距离 |

---

## 关联

- [[01-Projects/AMP-Reproduction/PhysHSI/04-环境架构与运动库|04 环境架构]]
- [[01-Projects/AMP-Reproduction/PhysHSI/06-机器人配置与动作空间|06 机器人配置]]
- [[01-Projects/AMP-Reproduction/PhysHSI/08-迁移指南-轮式底盘单臂|08 迁移指南]]
