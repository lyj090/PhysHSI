---
tags:
  - type/project
  - area/robotics
  - proj/amp-reproduction
status: active
created: 2026-06-17
---

# PhysHSI 总览与论文对照

> **文档定位**：PhysHSI 学习笔记入口，建立项目全局认知。本系列面向从四足 AMP 扩展到人形场景交互（最终迁移到轮式底盘+单臂）的学习路径。
>
> **前置**：[[01-Projects/AMP-Reproduction/AMP-Reproduction.md|AMP-Reproduction 项目入口]]、AMP 原理论文笔记
> **后续**：[[01-Projects/AMP-Reproduction/PhysHSI/02-核心算法-AMP判别器详解|02 AMP 判别器详解]]

---

## 1. PhysHSI 是什么

**PhysHSI**（**Phys**ics-Based **H**umanoid-**S**cene **I**nteraction）是上海 AI Lab + 港科大联合开发的人形机器人场景交互系统。论文发表于 arXiv（2025.10），代码开源于 [GitHub](https://github.com/InternRobotics/PhysHSI)。

### 核心贡献

| 维度 | 内容 |
|------|------|
| **任务覆盖** | 6 项任务：搬运箱子（CarryBox）、坐下（SitDown）、躺下（LieDown）、站起（StandUp）、恐龙步态（StyleLoco-Dinosaur）、高抬膝步态（StyleLoco-Highknee） |
| **算法融合** | HIM（Hybrid Internal Model）PPO + AMP（Adversarial Motion Prior）+ RSI（Randomized State Initialization） |
| **机器人平台** | Unitree G1（29 DOF 人形机器人） |
| **仿真引擎** | NVIDIA Isaac Gym Preview 4（GPU 加速并行仿真） |
| **学术意义** | 首次将 AMP 从 locomotion 拓展到长期程（long-horizon）全身场景交互 |

---

## 2. 系统架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                        PhysHSI 系统架构                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ MotionLib    │    │ AMP          │    │ HIM PPO          │  │
│  │ (参考运动库)  │───▶│ (判别器)     │───▶│ (策略训练)        │  │
│  │              │    │              │    │                  │  │
│  │ • 运动帧数据  │    │ • LSGAN 判别 │    │ • Actor-Critic   │  │
│  │ • 技能分类    │    │ • Style      │    │ • Smoothness Reg │  │
│  │ • RSI 范围   │    │   Reward     │    │ • Muon Optimizer │  │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘  │
│                                                    │             │
│  ┌──────────────────────────────────────────────────▼─────────┐ │
│  │              Isaac Gym 并行仿真环境 (≤4096 envs)            │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐      ┌─────────┐  │ │
│  │  │ Env 0   │  │ Env 1   │  │ Env 2   │ ...  │ Env N   │  │ │
│  │  │ G1+Box  │  │ G1+Box  │  │ G1+Box  │      │ G1+Box  │  │ │
│  │  └─────────┘  └─────────┘  └─────────┘      └─────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              任务层 (Task-Specific)                         │ │
│  │  CarryBox │ SitDown │ LieDown │ StandUp │ StyleLoco × 2   │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 数据流

```
MotionLib 参考帧 ──▶ AMP_Discriminator(policy_state) ──▶ style_reward
                                                            │
                               ┌────────────────────────────┘
                               ▼
Environment ──▶ obs_buf ──▶ Actor(obs) ──▶ action ──▶ PD Controller ──▶ Isaac Gym
    ▲                                                           │
    │                     ┌─────────────────────────────────────┘
    │                     ▼
    └─────────── reward = task_reward * (1-w) + style_reward * w
```

---

## 3. 论文对应关系

PhysHSI 的算法设计对应以下论文谱系：

```
SIGGRAPH 2020 (Peng et al.)
  AMP 原始版本：局部状态转移判别器
  └─▶ IROS 2022 (Escontrela et al.)
        AMP for Hardware：验证 sim-to-real
        └─▶ PhysHSI (Wang et al., 2025)
              将 AMP 从 locomotion 扩展到场景交互
              新增：HIM 混合训练、RSI 课程、多阶段训练
```

### 与 AMP-IROS22（本项目核心论文）的关系

| 对比维度 | AMP-IROS22 | PhysHSI |
|----------|-----------|---------|
| **机器人** | Unitree A1 四足 (12 DOF) | Unitree G1 人形 (29 DOF) |
| **任务** | 纯 locomotion（速度跟踪） | 全身交互（搬箱、坐、躺、站） |
| **判别器输入** | 足端/关节状态转移 | 全身关节 + 末端执行器 + 物体位置 |
| **PPO 变体** | 标准 PPO | HIM PPO（smoothness reg + Muon） |
| **训练策略** | 单阶段 | 多阶段（预训练 → 精调） |
| **物体交互** | 无 | 有（箱子位置/姿态纳入观测） |
| **RSI** | 无 | 有（随机状态初始化课程） |

---

## 4. 代码仓库结构

```
PhysHSI/
├── legged_gym/                     # 主框架（基于 ETH legged_gym 魔改）
│   ├── legged_gym/
│   │   ├── envs/
│   │   │   ├── base/
│   │   │   │   ├── base_task.py        # 基类：仿真管理、环境操作
│   │   │   │   ├── base_config.py      # 配置递归实例化
│   │   │   │   └── legged_robot_config.py  # 完整配置定义
│   │   │   ├── g1/                     # G1 人形机器人任务
│   │   │   │   ├── carrybox.py         # 搬箱子（最复杂，~2074 行）
│   │   │   │   ├── carrybox_config.py  # 搬箱子配置
│   │   │   │   ├── carrybox_resume_config.py  # 精调阶段配置
│   │   │   │   ├── sitdown.py / sitdown_config.py
│   │   │   │   ├── liedown.py / liedown_config.py
│   │   │   │   ├── standup.py / standup_config.py
│   │   │   │   └── styleloco.py / *_config.py
│   │   │   └── motionlib/              # 运动库
│   │   │       ├── motionlib_carrybox.py
│   │   │       ├── motionlib_sitdown.py
│   │   │       ├── motionlib_liedown.py
│   │   │       ├── motionlib_standup.py
│   │   │       └── motionlib_styleloco.py
│   │   ├── utils/
│   │   │   ├── task_registry.py        # 任务注册、环境/算法创建
│   │   │   ├── math.py                 # 四元数、欧拉角、旋转工具
│   │   │   ├── terrain.py              # 地形生成
│   │   │   ├── helpers.py              # ONNX/JIT 导出等辅助
│   │   │   └── torch_utils.py          # PyTorch 工具函数
│   │   └── scripts/
│   │       ├── train.py                # 训练入口 (20 行)
│   │       └── play.py                 # 可视化/测试入口
│   └── resources/
│       ├── robots/g1/urdf/g1_29dof.urdf  # G1 机器人 URDF
│       ├── dataset/                       # 参考运动数据 (.pt)
│       ├── config/                        # 运动配置文件 (.yaml)
│       └── ckpt/                          # 预训练权重
├── rsl_rl/                           # RL 算法库（基于 RSL-RL 魔改）
│   └── rsl_rl/
│       ├── algorithms/
│       │   └── him_ppo.py            # HIM PPO 实现（核心算法文件）⭐
│       ├── modules/
│       │   ├── actor_critic.py       # Actor-Critic 网络架构
│       │   └── amp.py                # AMP 判别器模块 ⭐
│       └── storage/
│           └── him_rollout_storage.py # 经验回滚存储
├── docs/                             # 项目文档
│   ├── MOBILE_MANIPULATOR_MIGRATION_PLAN.md  # 迁移到移动操作的计划
│   └── reproduction/                 # 复现指南
├── requirements.txt
└── README.md
```

**⭐ 标记为核心算法文件**，建议优先精读。

---

## 5. 六大任务概览

| 任务 | 命令参数 | 难度 | 关键挑战 | 训练阶段 |
|------|----------|------|----------|----------|
| **CarryBox** | `carrybox` | 🔴 最难 | 长时程、多阶段（靠近→拿起→搬运→放下）、物体感知 | 2 阶段（预训练 + 精调） |
| **SitDown** | `sitdown` | 🟡 中等 | 接触物理、椅子碰撞 | 1 阶段 |
| **LieDown** | `liedown` | 🟡 中等 | 全身倒地控制、多接触点 | 1 阶段 |
| **StandUp** | `standup` | 🟡 中等 | 从地面站起的力矩需求 | 1 阶段 |
| **StyleLoco-Dinosaur** | `styleloco_dinosaur` | 🟢 较易 | 纯 locomotion 风格 | 1 阶段 |
| **StyleLoco-Highknee** | `styleloco_highknee` | 🟢 较易 | 纯 locomotion 风格 | 1 阶段 |

### 重点关注 CarryBox

CarryBox 是最复杂的任务，包含四个子技能的顺序衔接：

```
阶段 1: loco (靠近箱子) → 阶段 2: pickUp (拿起箱子)
    → 阶段 3: carryWith (抱箱搬运) → 阶段 4: putDown (放下箱子)
```

这是与「轮式底盘+单臂搬运」最相关的任务，后续拆解文档将以此为重点。

---

## 6. 核心概念速览

在深入代码之前，先理解以下概念在 PhysHSI 中的具体含义：

| 概念 | 在 PhysHSI 中的实现 |
|------|-------------------|
| **AMP 判别器** | 2 层 MLP（512→256→1），对策略运动帧输出 logit，目标是专家=+1、策略=-1 |
| **Style Reward** | `max(0, 1 - 0.25*(D-1)²)`，D 是判别器输出，越接近 1 奖励越大 |
| **AMP 观测窗口** | 10 帧历史（window_length=10），每帧含关节角、末端位置、基座速度等 |
| **HIM PPO** | PPO 变体：增加了 value/action smoothness regularization + Muon 优化器 |
| **RSI** | Randomized State Initialization：从参考运动的随机帧初始化状态（课程学习） |
| **多阶段训练** | CarryBox 先训 loco 靠近，再训 pick+carry+put（AMP coef 逐步增大） |

---

## 7. 与本项目的关联

### 你的目标

> 在 Isaac Sim/Lab 中，基于**轮式底盘+单机械臂**实现 Physics-based 操作策略，AMP 风格约束使动作自然流畅。

### PhysHSI 提供什么

| PhysHSI 组件   | 可迁移到你的项目                      |
| ------------ | ----------------------------- |
| AMP 判别器架构    | 直接复用：用操作动作数据训练判别器，约束策略"像专家"   |
| HIM PPO 训练流程 | 直接复用：smoothness reg 对操作任务同样重要 |
| 运动库系统        | 可适配：将 G1 全身帧 → 机械臂末端轨迹帧       |
| 观测噪声模型       | 直接复用：相机感知噪声建模                 |
| 域随机化配置       | 直接复用：摩擦、质量、延迟等                |
| 多阶段训练策略      | 可适配：先训底盘导航，再训抓取操作             |

### 差异与挑战

| 维度 | PhysHSI (G1 人形) | 你的平台（轮式底盘+单臂） |
|------|------------------|------------------------|
| 自由度 | 29 DOF（双腿+双臂+腰） | ~8-10 DOF（底盘+单臂） |
| 移动方式 | 双足行走 | 轮式差速/全向 |
| AMP 参考数据 | 全身 mocap 帧 | 机械臂末端轨迹/关节序列 |
| 物体交互 | 双手抱箱 | 单臂抓取/放置 |
| 仿真引擎 | Isaac Gym | Isaac Sim/Lab（推荐） |

关键迁移策略见 [[01-Projects/AMP-Reproduction/PhysHSI/08-迁移指南-轮式底盘单臂|08 迁移指南]]。

---

## 8. 文档阅读路线

建议按以下顺序阅读本系列文档：

```
01-PhysHSI总览与论文对照.md  ← 你在这里
 │
 ├── 02-核心算法-AMP判别器详解.md    ← 先理解"style reward 怎么来"
 ├── 03-核心算法-HIM-PPO详解.md      ← 然后理解"策略怎么训"
 ├── 04-环境架构与运动库.md           ← 理解仿真和数据怎么组织
 ├── 05-任务实例-CarryBox拆解.md     ← 实战：最复杂任务全流程
 ├── 06-机器人配置与动作空间.md       ← 理解 G1 的控制接口
 ├── 07-训练部署与可视化.md           ← 跑起来的实际操作
 └── 08-迁移指南-轮式底盘单臂.md      ← 将知识应用到你的项目
```

**如果时间有限**：先读 02 → 03 → 05 → 08，这四篇覆盖核心算法 + 实战 + 迁移。

---

## 9. 关键代码入口

| 想了解的内容 | 文件路径 | 行数参考 |
|-------------|---------|---------|
| 训练主循环 | `legged_gym/legged_gym/scripts/train.py` | ~48 行（很简洁） |
| HIM PPO 完整 update() | `rsl_rl/rsl_rl/algorithms/him_ppo.py` | ~L182-L284 |
| AMP 判别器 forward/loss/reward | `rsl_rl/rsl_rl/modules/amp.py` | ~L94-L131 |
| Actor-Critic 网络 | `rsl_rl/rsl_rl/modules/actor_critic.py` | ~L92-L195 |
| CarryBox 环境 step/post_physics_step | `legged_gym/legged_gym/envs/g1/carrybox.py` | ~L94-L282 |
| CarryBox 奖励函数 | 同上 `compute_reward()` | ~L417-L442 |
| CarryBox 观测构建 | 同上 `compute_observations()` | ~L486-L516 |
| 运动库加载与采样 | `legged_gym/legged_gym/envs/motionlib/motionlib_carrybox.py` | ~L16-L93 |
| 任务注册 | `legged_gym/legged_gym/utils/task_registry.py` | - |
| 配置类继承链 | `legged_gym/legged_gym/envs/base/legged_robot_config.py` | - |

---

## 关联

- [[01-Projects/AMP-Reproduction/AMP-Reproduction|AMP 复现项目入口]]
- [[01-Projects/AMP-Reproduction/PhysHSI/02-核心算法-AMP判别器详解|02 AMP 判别器]]
- [[01-Projects/AMP-Reproduction/PhysHSI/08-迁移指南-轮式底盘单臂|08 迁移指南]]
