---
tags:
  - type/concept
  - area/robotics
  - proj/amp-reproduction
status: active
created: 2026-06-17
---

# 02 核心算法：AMP 判别器详解

> **文档定位**：逐行拆解 PhysHSI 中 AMP（Adversarial Motion Prior）判别器的实现，包含数学原理、代码实现、伪代码和用法指导。
>
> **前置**：[[01-Projects/AMP-Reproduction/PhysHSI/01-PhysHSI总览与论文对照|01 PhysHSI 总览]]
> **后续**：[[01-Projects/AMP-Reproduction/PhysHSI/03-核心算法-HIM-PPO详解|03 HIM PPO 详解]]
>
> **源码**：`rsl_rl/rsl_rl/modules/amp.py`

---

## 1. AMP 判别器是什么

### 1.1 直觉理解

AMP 判别器的核心思想来自 GAN（生成对抗网络）：

```
生成器 = 策略网络（Policy）
判别器 = AMP Discriminator

策略生成运动 → 判别器判断"这像不像专家数据"
判别器给出分数 → 策略用它作为"风格奖励"
```

**与传统 GAN 的区别**：

| 维度 | 传统 GAN | AMP |
|------|---------|-----|
| 生成器 | 学习生成以假乱真的样本 | 学习产生"像专家"的运动 |
| 判别器目标 | 真假二分类 | 区分专家运动 vs 策略运动 |
| 奖励信号 | 判别器输出概率 | `max(0, 1 - 0.25*(D-1)²)` 映射为正值 |
| 输入 | 完整图像/数据 | 状态转移对 (s_t, s_{t+1}) |
| 训练方式 | 同步对抗 | PPO 更新策略 → 判别器从经验回滚学习 |

### 1.2 数学公式（对应论文）

**判别器训练目标**（LSGAN loss）：

$$\mathcal{L}_D = \mathbb{E}_{s \sim \pi_E}[(D(s) - 1)^2] + \mathbb{E}_{s \sim \pi_\theta}[(D(s) + 1)^2] + w^{gp} \cdot \mathcal{L}_{gp}$$

其中：
- $D(s) \in \mathbb{R}$ 是判别器输出的 logit（不是 [0,1] 概率）
- $\pi_E$ 是专家策略（参考运动数据），$\pi_\theta$ 是当前策略
- $w^{gp} = 0.1 \times 5$（gradient penalty 权重）
- $\mathcal{L}_{gp}$ 是 gradient penalty（1-GP 形式）

**Style Reward 函数**：

$$r^s(s_t, s_{t+1}) = \max\left(0,\; 1 - \frac{1}{4}(D(s_t, s_{t+1}) - 1)^2\right)$$

**与任务奖励融合**：

$$r_{total} = (1 - \alpha) \cdot r^{task} + \alpha \cdot r^s \cdot \mathbb{1}_{stage}$$

其中 $\alpha = 0.25$（`amp_coef`），`stage` 是任务阶段指示器（CarryBox 多阶段）。

---

## 2. 代码逐行拆解

### 2.1 判别器网络结构

```python
# rsl_rl/rsl_rl/modules/amp.py, class AMP(nn.Module)

class AMP(nn.Module):
    def __init__(self, num_obs, amp_coef,
                 hidden_dims=[512, 256],  # 2层MLP
                 activation='relu',
                 init_noise_std=1.0,
                 device='cuda:0', **kwargs):
```

**网络结构**：
```
输入: num_obs (10帧 × 每帧obs)
  │
  ├─ Linear(num_obs → 512) + ReLU    ← trunk[0:2]
  ├─ Linear(512 → 256) + ReLU        ← trunk[2:4]
  ├─ Linear(256 → 1)                 ← amp_linear (输出logit)
  │
输出: logit ∈ ℝ
```

**关键代码**：

```python
# 构建 trunck（隐藏层序列）
disc_layers = []
disc_layers.append(nn.Linear(mlp_input_dim, hidden_dims[0]))
disc_layers.append(activation)  # ReLU

for l in range(len(hidden_dims)):
    if l == len(hidden_dims) - 1:
        ln = nn.Linear(hidden_dims[l], mlp_output_dim)  # → 1
        torch.nn.init.uniform_(ln.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)  # ±1.0
        torch.nn.init.zeros_(ln.bias)
        self.amp_linear = ln       # 最后一层单独存为 amp_linear
    else:
        ln = nn.Linear(hidden_dims[l], hidden_dims[l+1])
        torch.nn.init.uniform_(ln.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
        torch.nn.init.zeros_(ln.bias)
        disc_layers.append(ln)
        disc_layers.append(activation)
self.trunk = nn.Sequential(*disc_layers)
```

> **设计要点**：最后输出层 `amp_linear` 单独保存（与 `trunk` 分离），因为两类参数使用不同的 weight decay（`amp_linear` 用 `10e-2`，`trunk` 用 `10e-4`）。

### 2.2 前向传播

```python
def forward(self, x):
    disc_demo_logit = self.trunk(x)
    disc_demo_logit = self.amp_linear(disc_demo_logit)
    return disc_demo_logit
```

极其简单：`x → trunk → amp_linear → logit`。

### 2.3 判别器损失函数

这是最核心的方法：

```python
def compute_loss(self, agent_obs, expert_obs):
    # agent_obs: 策略产生的观测，形状 [batch, num_obs]
    # expert_obs: 参考运动的观测，形状 [batch, num_obs]

    policy_d = self.amp_linear(self.trunk(agent_obs))   # D(策略运动)
    expert_d = self.amp_linear(self.trunk(expert_obs))   # D(专家运动)

    # LSGAN 损失
    expert_loss = (expert_d - 1).pow(2).mean()  # 专家 → +1
    policy_loss = (policy_d + 1).pow(2).mean()  # 策略 → -1

    gail_loss = expert_loss + policy_loss

    # Gradient Penalty（仅对专家数据）
    grad_pen = self.compute_grad_pen(expert_obs, agent_obs) * 0.1

    loss = gail_loss + grad_pen

    return loss, expert_loss, policy_loss
```

**数学展开**：

$$\mathcal{L}_{expert} = \mathbb{E}[(D(s_E) - 1)^2] \quad \text{拉向 +1}$$
$$\mathcal{L}_{policy} = \mathbb{E}[(D(s_\pi) + 1)^2] \quad \text{拉向 -1}$$
$$\mathcal{L}_{gp} = 0.1 \times \|\nabla_{s_E} D(s_E)\|^2$$

### 2.4 Gradient Penalty 实现

```python
def compute_grad_pen(self, expert_state, policy_state, lambda_=5):
    # 使用 DRAGAN-style gradient penalty：仅对专家数据
    expert_state = expert_state.detach().requires_grad_(True)
    disc_demo_logit = self.trunk(expert_state)
    disc_demo_logit = self.amp_linear(disc_demo_logit)

    # 计算 D 对 expert_state 的梯度
    disc_demo_grad = torch.autograd.grad(
        disc_demo_logit, expert_state,
        grad_outputs=torch.ones_like(disc_demo_logit),
        create_graph=True, retain_graph=True, only_inputs=True
    )
    disc_demo_grad = disc_demo_grad[0]
    # 对每个样本：梯度的 L2 范数平方
    disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
    disc_grad_penalty = torch.mean(disc_demo_grad)

    grad_pen = disc_grad_penalty * lambda_  # lambda_=5
    return grad_pen
```

> **与 WGAN-GP 的区别**：WGAN-GP 在真实/生成样本间插值后计算 GP；PhysHSI 使用 **DRAGAN-style**（仅对真实数据），且目标是惩罚大梯度（不是 $\|\nabla D\|-1$ 形式）。

### 2.5 Style Reward 计算

```python
def predict_reward(self, agent_obs, normalizer):
    with torch.no_grad():
        self.eval()
        if normalizer is not None:
            agent_obs = normalizer.normalize(agent_obs)
        d = self.amp_linear(self.trunk(agent_obs))
        self.train()
        # 核心公式：r = max(0, 1 - 0.25*(D-1)²)
        return torch.clamp(1 - 0.25 * torch.square(d - 1), min=0)
```

**可视化**：

```
reward
 1.0 ┤       ╱‾‾‾╲
     │      ╱      ╲
 0.5 │     ╱        ╲
     │    ╱          ╲
 0.0 ├───╯            ╰───
     └───┬────────────┬───▶ D (logit)
        -1            1   3
```

- 当 $D(s) = 1$（专家水平）：$r = 1.0$（满分）
- 当 $D(s) = -1$ 或 $3$：$r = 0$（不给分）
- 本质是一个以 1 为中心的平滑高斯型奖励

### 2.6 与任务奖励融合

```python
def combine_reward(self, amp_reward, task_reward, stage=None):
    if stage is not None:
        rewards = amp_reward * self.amp_coef * stage.to(self.device) + \
                  task_reward * (1 - self.amp_coef)
    else:
        rewards = amp_reward * self.amp_coef + \
                  task_reward * (1 - self.amp_coef)
    return rewards
```

- `amp_coef = 0.25`：风格奖励权重 25%，任务奖励 75%
- `stage`：多阶段训练时，技能阶段指示器（本文档第 5 节详述）

---

## 3. 伪代码：完整判别器训练流程

```
Algorithm: AMP Discriminator Training

Input:
  - D: 判别器网络 (trunk + amp_linear)
  - B_policy:  策略产生的 AMP 观测批次 [B, num_obs]
  - B_expert:  参考运动 AMP 观测批次 [B, num_obs]（从 MotionLib 采样）
  - optimizer: Adam（分类参数权重衰减）

Procedure:
  # 1. 前向传播
  D_policy = D(B_policy)      # 策略 logit
  D_expert = D(B_expert)      # 专家 logit

  # 2. LSGAN 损失
  L_expert = mean((D_expert - 1)²)   # 专家应该输出 +1
  L_policy = mean((D_policy + 1)²)   # 策略应该输出 -1
  L_gail = L_expert + L_policy

  # 3. Gradient Penalty (DRAGAN-style)
  B_expert.requires_grad = True
  D_expert_grad = D(B_expert)
  grad = ∂D_expert_grad / ∂B_expert
  L_gp = 0.1 * mean(||grad||²)

  # 4. 总损失
  L_total = L_gail + L_gp

  # 5. 反向传播
  optimizer.zero_grad()
  L_total.backward()
  optimizer.step()

  # 6. (可选) 更新 AMP 观测归一化器
  normalizer.update(B_policy)
  normalizer.update(B_expert)

Output: L_total, L_expert, L_policy
```

---

## 4. AMP 观测空间

### 4.1 观测组成

在 PhysHSI 中，AMP 判别器看的不是原始图像或传感器数据，而是**状态转移特征**。

**CarryBox 的 AMP 观测（单帧）**：

```python
# carrybox.py: compute_amp_observations()
num_one_step_obs = 1 + 29 + 5*3 + 3 + 6 + 6  # = 60

current_amp_obs = torch.cat([
    base_height,           # [1]  基座离地高度
    dof_pos,              # [29] 关节位置（仅 amp_obs_joint_id）
    end_effector_pos,     # [15] 末端执行器相对位置（双手+双脚+头）
    box_pos_local,        # [3]  箱子在机器人局部系下位置
    base_lin_vel,         # [3]  基座线速度
    base_ang_vel,         # [3]  基座角速度
    root_rot_obs,         # [6]  基座旋转（切线-法线 6D 表示）
], dim=-1)  # 总计 60 维
```

**AMP 观测是多帧堆叠**：

```python
# carrybox_config.py
window_length = 10  # 10帧历史
num_obs = 60 * 10 = 600  # 判别器输入维度
```

```python
# 堆叠方式：滑动窗口
self.amp_obs_buf = torch.cat(
    (self.amp_obs_buf[:, num_one_step_obs:], current_amp_obs),
    dim=-1
)  # 丢弃最旧帧，追加新帧
```

### 4.2 归一化

```python
# amp.py / actor_critic.py
class RunningMeanStd:
    def __init__(self, shape, device):
        self.n = 1e-4
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)

    def update(self, x):
        # 增量更新均值/方差（Welford 算法变体）
        ...
        self.mean = old_mean + delta * batch_count / tot_count
        self.var = M2 / tot_count
        self.n = tot_count

class Normalization:
    def __call__(self, x, update=False):
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (torch.sqrt(self.running_ms.var) + 1e-4)
        return x
```

> **注意**：PhysHSI 默认 `use_normalizer = False`（配置中关闭）。原因是 AMP 观测已经具备自然尺度，且归一化可能引入不必要的噪声。

---

## 5. 多阶段训练与 Stage 机制

### 5.1 为什么需要多阶段

CarryBox 有 4 个子技能（loco → pickUp → carryWith → putDown）。如果一开始就用严格的 AMP 约束，策略无法探索到有效行为。

### 5.2 阶段定义

```python
# carrybox_config.py
class box:
    skill = ["loco", "pickUp", "carryWith", "putDown"]
    skill_init_prob = [1.0, 0.0, 0.0, 0.0]  # 初始只训练 loco
```

```python
# carrybox.py: combine_reward() 使用 stage 掩码
rewards = amp_reward * amp_coef * stage + task_reward * (1 - amp_coef)
```

- 初始阶段 `stage=0` → 只有任务奖励（靠近箱子）
- 随着训练进行，`stage` 逐步变成 1 → AMP 风格约束生效

### 5.3 两阶段训练策略

```
第一阶段 (20k iters):
  - skill_init_prob = [1.0, 0, 0, 0]  # 只做 loco
  - amp_coef = 0.25
  - box_termination = False  # 宽松终止条件

第二阶段 (30-40k iters, from checkpoint):
  - 增加 amp_coef 加强风格约束
  - 逐步开启 pickUp/carryWith/putDown 技能
  - box_termination = True  # 严格终止条件
```

---

## 6. 配置速查表

```yaml
# carrybox_config.py 中 AMP 相关配置
amp:
  amp_coef: 0.25               # 风格奖励权重 (α)
  num_one_step_obs: 60         # 单帧观测维度
  window_length: 10            # 历史帧数
  num_obs: 600                 # 判别器输入维度 = 60 × 10
  ratio_random_range: [0.95, 1.05]  # RSI 时间比例随机范围
  use_normalizer: False        # 是否启用 AMP 观测归一化

# 判别器网络（在 him_ppo.py 中创建）
discriminator:
  hidden_dims: [512, 256]      # trunk 隐藏层
  activation: relu             # 激活函数

# 损失权重
loss:
  gail: 1.0                    # LSGAN 损失
  gp_lambda: 5.0               # gradient penalty lambda
  gp_coef: 0.1                 # gradient penalty 系数
```

---

## 7. 用法示例

### 7.1 加载预训练判别器并评估

```python
import torch
from rsl_rl.modules.amp import AMP

# 1. 创建判别器
amp = AMP(
    num_obs=600,        # 10帧 × 60维
    amp_coef=0.25,
    hidden_dims=[512, 256],
    activation='relu',
    device='cuda:0'
)

# 2. 加载权重
ckpt = torch.load('resources/ckpt/carrybox.pt')
amp.load_state_dict(ckpt['amp_state_dict'])

# 3. 评估：给一组观测打分
agent_obs = torch.randn(1024, 600, device='cuda:0')  # 模拟策略运动
expert_obs = torch.randn(1024, 600, device='cuda:0')  # 模拟专家运动

with torch.no_grad():
    amp.eval()
    agent_score = amp(agent_obs)   # 应接近 -1（被判定为非专家）
    expert_score = amp(expert_obs) # 应接近 +1（被判定为专家）

# 4. 计算 style reward
style_reward = amp.predict_reward(agent_obs, normalizer=None)
# shape: [1024, 1], 值范围 [0, 1]
```

### 7.2 独立训练判别器（概念演示）

```python
# 伪代码：演示判别器如何独立训练
amp = AMP(num_obs=600, amp_coef=0.25, device='cuda:0')
optimizer = torch.optim.Adam(amp.parameters(), lr=1e-3)

for epoch in range(1000):
    # 从 MotionLib 采样专家数据
    expert_obs = motion_buffer.get_expert_obs(batch_size=1024)

    # 从策略回滚采样（训练中由 HIM PPO 收集）
    agent_obs = rollout_storage.get_amp_obs(batch_size=1024)

    loss, expert_loss, policy_loss = amp.compute_loss(agent_obs, expert_obs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 8. 关键设计决策与洞察

### 8.1 为什么用 LSGAN 而不是标准 GAN

1. **梯度更稳定**：LSGAN 用平方损失，避免 sigmoid 饱和区梯度消失
2. **reward 映射更平滑**：$r = \max(0, 1 - 0.25*(D-1)^2)$ 给出平滑的奖励信号
3. **与 PPO 更兼容**：PPO 对奖励尺度敏感，LSGAN 的二次损失天然控制 logit 范围

### 8.2 为什么判别器看状态转移而非单帧

AMP 判别器输入 $s_t$ 实际上包含多帧历史（window_length=10），隐含了状态转移信息：

$$D(s_{t-9}, s_{t-8}, ..., s_t) \approx D((s_{t-1}, s_t) \text{ 类型特征})$$

这比显式给 $(s_t, s_{t+1})$ 更稳定，因为滑动窗口平滑了单步噪声。

### 8.3 为什么 DRAGAN-style GP

WGAN-GP 需要在真假样本间插值，但 AMP 的运动数据分布差异大，插值样本无物理意义。DRAGAN-style 仅约束专家数据区域的梯度，避免了无意义的插值。

---

## 关联

- [[01-Projects/AMP-Reproduction/PhysHSI/01-PhysHSI总览与论文对照|01 PhysHSI 总览]]
- [[01-Projects/AMP-Reproduction/PhysHSI/03-核心算法-HIM-PPO详解|03 HIM PPO 详解]]
- [[01-Projects/AMP-Reproduction/PhysHSI/04-环境架构与运动库|04 环境架构与运动库]]
- [[01-Projects/AMP-Reproduction/PhysHSI/08-迁移指南-轮式底盘单臂|08 迁移指南]]
