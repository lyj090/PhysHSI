---
tags:
  - type/concept
  - area/robotics
  - proj/amp-reproduction
status: active
created: 2026-06-17
---

# 03 核心算法：HIM PPO 详解

> **文档定位**：逐行拆解 PhysHSI 中的 HIM PPO（Hybrid Internal Model PPO）算法实现——这是对整个训练管线的核心改造。
>
> **前置**：[[01-Projects/AMP-Reproduction/PhysHSI/02-核心算法-AMP判别器详解|02 AMP 判别器]]
> **后续**：[[01-Projects/AMP-Reproduction/PhysHSI/04-环境架构与运动库|04 环境架构]]
>
> **源码**：`rsl_rl/rsl_rl/algorithms/him_ppo.py`（285 行）

---

## 1. HIM PPO 是什么

HIM PPO 是 PhysHSI 对标准 PPO 的三重增强版本：

```
标准 PPO
  + AMP 判别器联合训练      → style reward 注入
  + Smoothness Regularization → 策略/价值函数平滑性约束
  + Muon Optimizer（可选）   → 更高效的权重更新
─────────────────────────────────────
= HIM PPO
```

### 与标准 PPO 的核心差异

| 组件 | 标准 PPO | HIM PPO |
|------|---------|---------|
| **Actor 更新** | PPO clip loss | PPO clip loss + policy smooth loss |
| **Critic 更新** | MSE(V, return) | clipped MSE + value smooth loss |
| **额外损失** | 无 | AMP 判别器损失（联合训练） |
| **学习率调度** | 固定/线性 | 自适应 KL（含 Muon 分组） |
| **优化器** | Adam | Adam 或 Muon+Adam 混合 |
| **Rollout 存储** | 标准 | 扩展 amp_obs 字段 |

---

## 2. 代码逐行拆解

### 2.1 初始化

```python
class HIMPPO:
    def __init__(self,
                 actor_critic,              # Actor-Critic 网络
                 num_learning_epochs=1,     # PPO epochs per update
                 num_mini_batches=1,        # mini-batch 数
                 clip_param=0.2,            # PPO clip ε
                 gamma=0.998,               # 折扣因子
                 lam=0.95,                  # GAE λ
                 value_loss_coef=1.0,
                 entropy_coef=0.0,          # PhysHSI: entropy=0
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",          # 或 "adaptive" KL
                 desired_kl=0.01,           # 自适应 KL 目标
                 device='cpu',
                 use_muon_optim=False,       # 是否使用 Muon
                 value_smoothness_coef=0.1,  # 价值平滑系数 ⭐
                 smoothness_upper_bound=1.0,
                 smoothness_lower_bound=0.1,
                 amp=None,                   # AMP 判别器 ⭐
                 amp_normalizer=None,        # AMP 观测归一化
                 motion_buffer=None,         # 运动库缓冲区
                 ):
```

### 2.2 优化器配置：Adam vs Muon

PhysHSI 支持两种优化器模式：

#### Adam 模式（默认）

```python
# 分组参数：不同 weight decay
params = [
    {'params': self.actor_critic.parameters(), 'name': 'actor_critic'},
    {'params': self.amp.trunk.parameters(),
     'weight_decay': 10e-4, 'name': 'amp_trunk'},       # 轻正则
    {'params': self.amp.amp_linear.parameters(),
     'weight_decay': 10e-2, 'name': 'amp_head'}          # 较重正则
]
self.optimizer = optim.Adam(params, lr=learning_rate)
```

> **设计意图**：判别器最后一个线性层（`amp_linear`）用更大的 weight decay，防止过拟合到当前策略；trunk 用较小 decay，保持特征表达能力。

#### Muon 模式（`use_muon_optim=True`）

```python
# Muon: 对 ≥2D 参数使用动量更新（类似 SGD with momentum on steroids）
# Adam: 对 <2D 参数（bias/gain）使用 Adam

ac_hidden_weights = [p for p in self.actor_critic.parameters() if p.ndim >= 2]
amp_trunk_hidden_weights = [p for p in self.amp.trunk.parameters() if p.ndim >= 2]
amp_linear_hidden_weights = [p for p in self.amp.amp_linear.parameters() if p.ndim >= 2]

param_groups = [
    dict(params=ac_hidden_weights, use_muon=True, lr=learning_rate),
    dict(params=ac_hidden_gains_biases + amp_linear_hidden_gains_biases
         + amp_linear_hidden_weights + amp_trunk_hidden_gains_biases
         + amp_trunk_hidden_weights,
         use_muon=False, lr=learning_rate, betas=(0.9, 0.95)),
]
self.optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
```

> **Muon 优势**：对权重矩阵（≥2D）使用 Newton-Schulz 迭代近似梯度正交化，收敛更快。

### 2.3 动作生成：`act()`

```python
def act(self, obs, critic_obs):
    # 防止 NaN（环境重置时的边界情况）
    if obs.isnan().any():
        obs = torch.zeros(...)
        critic_obs = torch.zeros(...)

    # 采样动作
    self.transition.actions = self.actor_critic.act(obs).detach()
    # 评估价值
    self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
    # log π(a|s)
    self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
        self.transition.actions).detach()
    # 记录分布参数（用于后续 KL 计算）
    self.transition.action_mean = self.actor_critic.action_mean.detach()
    self.transition.action_sigma = self.actor_critic.action_std.detach()
    # 存储观测
    self.transition.observations = obs
    self.transition.critic_observations = critic_obs
    return self.transition.actions
```

### 2.4 环境步处理：`process_env_step()`

```python
def process_env_step(self, rewards, dones, infos, next_critic_obs):
    self.transition.next_critic_observations = next_critic_obs.clone()
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones

    # Timeout bootstrapping: 超时不算"终止"，用价值估计代替
    if 'time_outs' in infos:
        self.transition.rewards += self.gamma * torch.squeeze(
            self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

    self.storage.add_transitions(self.transition)
    self.transition.clear()
    self.actor_critic.reset(dones)
```

> **Timeout bootstrapping**：如果 episode 因为达到最大步数结束（非真正失败），用当前价值估计 $V(s)$ 作为后续回报的近似。这避免了将"没失败但时间到了"的状态标记为价值 0。

### 2.5 回报计算：`compute_returns()`

```python
def compute_returns(self, last_critic_obs):
    last_values = self.actor_critic.evaluate(last_critic_obs).detach()
    self.storage.compute_returns(last_values, self.gamma, self.lam)
    # 内部使用 GAE: A_t = δ_t + γλ A_{t+1}
    # δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

### 2.6 核心更新：`update()` —— 完整拆解

这是整个算法最核心的方法。以下是**逐段拆解**：

```python
def update(self):
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_amp_loss = 0
    mean_expert_loss = 0
    mean_policy_loss = 0

    generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs)
```

#### 阶段 1：PPO 标准更新

```python
    for obs_batch, next_obs_batch, critic_obs_batch, actions_batch, \
        next_critic_obs_batch, cont_batch, target_values_batch, \
        advantages_batch, returns_batch, shortreturns_batch, \
        old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, \
        amp_obs_batch in generator:

        # --- 重新评估当前策略 ---
        self.actor_critic.act(obs_batch)
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
        value_batch = self.actor_critic.evaluate(critic_obs_batch)
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        entropy_batch = self.actor_critic.entropy
```

#### 阶段 2：自适应 KL 调度

```python
        # --- 自适应学习率（可选） ---
        if self.desired_kl is not None and self.schedule == 'adaptive':
            with torch.inference_mode():
                # KL(π_old || π_new) 的近似计算
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1e-5)
                    + (torch.square(old_sigma_batch)
                       + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                # KL 太大 → 降 lr；KL 太小 → 升 lr
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
```

> **两层 KL 阈值设计**：`desired_kl * 2.0` 和 `desired_kl / 2.0`，形成 "dead zone" 避免频繁调整。

#### 阶段 3：PPO 损失

```python
        # --- PPO Surrogate Loss ---
        ratio = torch.exp(actions_log_prob_batch
                         - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) \
            * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # --- Clipped Value Loss ---
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch \
                + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        loss = surrogate_loss + self.value_loss_coef * value_loss \
               - self.entropy_coef * entropy_batch.mean()
```

> **注意**：PhysHSI 中 `entropy_coef = 0.01`，远小于典型 RL 设置（0.001-0.01）。这是因为 AMP 风格奖励已经提供了足够的探索引导。

#### 阶段 4：Smoothness Regularization ⭐

```python
        # --- Smoothness Loss ---
        # 调整 smoothness 系数使其与 clip_param 协调
        epsilon = self.smoothness_lower_bound \
            / (self.smoothness_upper_bound - self.smoothness_lower_bound)
        policy_smooth_coef = self.smoothness_upper_bound * epsilon
        value_smooth_coef = self.value_smoothness_coef * policy_smooth_coef

        # 在相邻状态间随机插值
        mix_weights = cont_batch * (torch.rand_like(cont_batch) - 0.5) * 2.0
        mix_obs_batch = obs_batch + mix_weights * (next_obs_batch - obs_batch)
        mix_critic_obs_batch = critic_obs_batch \
            + mix_weights * (next_critic_obs_batch - critic_obs_batch)

        # Policy smooth: 相近输入 → 相近动作
        policy_smooth_loss = torch.square(
            torch.norm(mu_batch - self.actor_critic.act_inference(mix_obs_batch), dim=-1)
        ).mean()

        # Value smooth: 相近输入 → 相近价值
        value_smooth_loss = torch.square(
            torch.norm(value_batch
                      - self.actor_critic.evaluate(mix_critic_obs_batch), dim=-1)
        ).mean()

        smooth_loss = policy_smooth_coef * policy_smooth_loss \
                      + value_smooth_coef * value_smooth_loss
        loss += smooth_loss
```

**数学表述**：

$$\mathcal{L}_{smooth} = w_{ps} \cdot \|\pi(s_t) - \pi(s_t + \delta \cdot (s_{t+1} - s_t))\|^2 + w_{vs} \cdot \|V(s_t) - V(s_t + \delta \cdot (s_{t+1} - s_t))\|^2$$

其中 $\delta \sim U(-1, 1)$，仅在 $s_t$ 和 $s_{t+1}$ 之间连续时才进行插值（`cont_batch > 0`）。

**为什么需要**：人形机器人动作空间大，状态转移连续——约束局部 Lipschitz 平滑性可以防止高频震荡动作。

#### 阶段 5：AMP 判别器损失

```python
        # --- AMP Loss ---
        if self.amp is not None:
            # 从 MotionLib 采样专家观测
            amp_expert_obs_batch = self.motion_buffer.get_expert_obs(
                batch_size=obs_batch.shape[0]).to(self.device)

            if self.amp_normalizer is not None:
                amp_expert_obs_batch = self.amp_normalizer.normalize(amp_expert_obs_batch)
                amp_obs_batch = self.amp_normalizer.normalize(amp_obs_batch)

            amp_loss, expert_loss, policy_loss = self.amp.compute_loss(
                amp_obs_batch, amp_expert_obs_batch)
            loss += amp_loss

            # 更新 AMP 归一化统计量
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(amp_obs_batch)
                self.amp_normalizer.update(amp_expert_obs_batch)
```

#### 阶段 6：梯度更新

```python
        # --- Gradient Step ---
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # 累积统计
        mean_value_loss += value_loss.item()
        mean_surrogate_loss += surrogate_loss.item()
        mean_amp_loss += amp_loss.item()
        mean_expert_loss += expert_loss.item()
        mean_policy_loss += policy_loss.item()
```

---

## 3. 完整损失函数总结

$$\mathcal{L}_{HIM-PPO} = \underbrace{\mathcal{L}_{PPO}^{clip}}_{actor} + \underbrace{c_v \cdot \mathcal{L}_{VF}^{clip}}_{critic} - \underbrace{c_e \cdot \mathcal{H}(\pi)}_{entropy} + \underbrace{\mathcal{L}_{smooth}}_{smoothness} + \underbrace{\mathcal{L}_{AMP}}_{discriminator}$$

其中：

| 项 | 公式 | 含义 |
|----|------|------|
| $\mathcal{L}_{PPO}^{clip}$ | $\max(-A \cdot r, -A \cdot clip(r, 1-\epsilon, 1+\epsilon))$ | PPO clip loss |
| $\mathcal{L}_{VF}^{clip}$ | $\max((V-R)^2, (V_{clip}-R)^2)$ | Clipped value loss |
| $\mathcal{H}(\pi)$ | $\sum \log \sigma$ | 熵正则（coefficient=0.01） |
| $\mathcal{L}_{smooth}$ | $w_{ps}\|\pi(s) - \pi(s')\|^2 + w_{vs}\|V(s) - V(s')\|^2$ | 平滑性约束 |
| $\mathcal{L}_{AMP}$ | $(D(s_E)-1)^2 + (D(s_\pi)+1)^2 + 0.1 \cdot GP$ | LSGAN + gradient penalty |

---

## 4. 伪代码：完整 HIM PPO 训练循环

```
Algorithm: HIM PPO Training Loop

Hyperparameters:
  γ = 0.998, λ = 0.95, ε = 0.2, K = 1 (epochs)
  w_amp = 0.25, c_v = 1.0, c_e = 0.01
  w_smooth_policy = computed, w_smooth_value = 0.1 × w_smooth_policy

Initialize:
  π_θ, V_φ    ← ActorCritic networks
  D_ψ         ← AMP discriminator
  Optimizer    ← Adam (with parameter-group weight decay)
  MotionBuffer ← pre-loaded reference motion data
  Storage      ← HIMRolloutStorage

For iteration = 1 to max_iterations:

  # ===== 1. Rollout (收集经验) =====
  For t = 1 to num_steps_per_env:
    obs, critic_obs = env.get_observations()
    action ~ π_θ(obs)                     # 采样动作
    next_obs, reward, done, info = env.step(action)

    # 融合 AMP style reward
    amp_obs = env.compute_amp_observations()
    style_reward = D_ψ.predict_reward(amp_obs)
    total_reward = w_amp × style_reward + (1 - w_amp) × task_reward

    storage.add(obs, action, total_reward, done, ..., amp_obs)

  # ===== 2. Compute Returns (GAE) =====
  last_value = V_φ(last_critic_obs)
  storage.compute_returns(last_value, γ, λ)
  # GAE: A_t = Σ (γλ)^k · δ_{t+k}
  # δ_t = r_t + γ·V(s_{t+1}) - V(s_t)

  # ===== 3. PPO Update =====
  For epoch = 1 to K:
    For each mini_batch:

      # 3a. Re-evaluate current policy
      π_new = π_θ(obs); V_new = V_φ(obs)

      # 3b. Adaptive LR (optional)
      KL(π_old || π_new); adjust lr if needed

      # 3c. PPO losses
      L_policy = clip_ppo_loss(π_new, π_old, A)
      L_value  = clip_value_loss(V_new, V_target)
      L_entropy = -H(π_new)

      # 3d. Smoothness regularization
      s' = s + δ(s_next - s), δ ~ U[-1, 1]
      L_smooth = w_ps·||π(s) - π(s')||² + w_vs·||V(s) - V(s')||²

      # 3e. AMP discriminator loss
      expert_obs = MotionBuffer.sample()
      L_amp = (D_ψ(expert_obs)-1)² + (D_ψ(agent_obs)+1)² + 0.1·GP

      # 3f. Total loss & step
      L = L_policy + c_v·L_value - c_e·L_entropy + L_smooth + L_amp
      optimizer.zero_grad(); L.backward(); clip_grad(); optimizer.step()

  # ===== 4. Logging & Checkpoint =====
  If iteration % save_interval == 0:
    save(π_θ, V_φ, D_ψ, optimizer, normalizer)
```

---

## 5. Actor-Critic 网络架构

```python
# rsl_rl/rsl_rl/modules/actor_critic.py

class ActorCritic(nn.Module):
    def __init__(self,
                 num_actor_obs,        # 策略观测维度
                 num_critic_obs,       # 价值观测维度（可包含特权信息）
                 actor_history_length, # 观测历史帧数
                 num_actions=29,       # G1: 29 DOF
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu'):
```

**网络结构**：
```
Actor:                        Critic:
  obs [B, num_actor_obs]        critic_obs [B, num_critic_obs]
    │                              │
    ├─ Linear→512 + ELU            ├─ Linear→512 + ELU
    ├─ Linear→256 + ELU            ├─ Linear→256 + ELU
    ├─ Linear→128 + ELU            ├─ Linear→128 + ELU
    ├─ Linear→29 (action mean)     ├─ Linear→1 (value)
    │                              │
    └─ σ (learned param, [29])     └─ V(s)
       ↓
    π = N(μ, σ²)
```

**关键实现细节**：

```python
# 动作分布：独立高斯
self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
# 注意：σ 是可学习参数，不随状态变化（对角高斯）

def update_distribution(self, observations):
    mean = self.actor(observations)
    self.distribution = Normal(mean, mean * 0. + self.std)
    # σ 固定（可学习但不依赖输入）

def act(self, observations):
    self.update_distribution(observations)
    return self.distribution.sample()  # 训练时采样

def act_inference(self, observations):
    actions_mean = self.actor(observations)
    return actions_mean  # 推理时用均值
```

---

## 6. 运行统计归一化：`RunningMeanStd`

```python
class RunningMeanStd:
    def __init__(self, shape, device):
        self.n = 1e-4           # 初始计数（小值避免除零）
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)

    def update(self, x):
        # Welford 风格增量更新
        count = self.n
        batch_count = x.size(0)
        tot_count = count + batch_count

        old_mean = self.mean.clone()
        delta = torch.mean(x, dim=0) - old_mean

        self.mean = old_mean + delta * batch_count / tot_count
        m_a = self.var * count
        m_b = x.var(dim=0) * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        self.var = M2 / tot_count
        self.n = tot_count
```

> **用于**：AMP 观测归一化（可选）、观测噪声注入时保持尺度一致。

---

## 7. 关键超参数速查

```python
# HIM PPO 默认超参数
class algorithm:
    clip_param = 0.2           # PPO clip ε
    gamma = 0.998              # 折扣因子
    lam = 0.95                 # GAE λ
    value_loss_coef = 1.0      # VF loss 权重
    entropy_coef = 0.01        # 熵系数
    learning_rate = 1e-3
    max_grad_norm = 1.0
    use_clipped_value_loss = True
    desired_kl = 0.01          # 自适应 KL 目标
    value_smoothness_coef = 0.1
    use_muon_optim = False     # 默认用 Adam

class runner:
    num_steps_per_env = 100    # 每次 rollout 步数
    max_iterations = 20000     # 总迭代数
    num_learning_epochs = 5    # PPO epochs（内置默认）
    num_mini_batches = 4       # Mini-batch 数（内置默认）
    save_interval = 500        # 每 500 iter 保存
```

---

## 8. 与标准 PPO 的对齐检查

如果你已经实现过标准 PPO，以下是需要修改的检查清单：

- [ ] 在 rollout 存储中添加 `amp_obs` 字段
- [ ] `process_env_step` 中处理 `time_outs` bootstrapping
- [ ] `update()` 中计算判别器损失 + 梯度
- [ ] `update()` 中添加 smoothness regularization
- [ ] 优化器参数分组（判别器不同 weight decay）
- [ ] 在 action 采样和 inference 间切换（训练时采样，推理时均值）
- [ ] `compute_returns` 使用 GAE（含 timeout 处理）
- [ ] AMP 观测归一化（可选，默认关闭）

---

## 关联

- [[01-Projects/AMP-Reproduction/PhysHSI/02-核心算法-AMP判别器详解|02 AMP 判别器]]
- [[01-Projects/AMP-Reproduction/PhysHSI/04-环境架构与运动库|04 环境架构]]
- [[01-Projects/AMP-Reproduction/PhysHSI/05-任务实例-CarryBox拆解|05 CarryBox 拆解]]
- [[01-Projects/AMP-Reproduction/Isaac-Sim/RL基础-PPO与Sim-to-Real|PPO 基础]]
