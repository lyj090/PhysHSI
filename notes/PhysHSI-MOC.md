---
tags:
  - type/project
  - area/robotics
  - proj/amp-reproduction
status: active
created: 2026-06-17
---

# PhysHSI 学习笔记 MOC

> **Map of Content**：PhysHSI 项目学习笔记索引。面向「从零学习 PhysHSI → 迁移到轮式底盘+单臂」的完整路径。

---

## 文档导航

| 序号 | 文档 | 定位 | 预计阅读时间 |
|------|------|------|------------|
| 01 | [[01-Projects/AMP-Reproduction/PhysHSI/01-PhysHSI总览与论文对照|01 PhysHSI 总览与论文对照]] | 入口：项目架构、论文谱系、代码地图 | 20 min |
| 02 | [[01-Projects/AMP-Reproduction/PhysHSI/02-核心算法-AMP判别器详解|02 AMP 判别器详解]] | 核心：LSGAN 判别器、gradient penalty、style reward | 30 min |
| 03 | [[01-Projects/AMP-Reproduction/PhysHSI/03-核心算法-HIM-PPO详解|03 HIM PPO 详解]] | 核心：PPO + smoothness reg + Muon + AMP 联合训练 | 40 min |
| 04 | [[01-Projects/AMP-Reproduction/PhysHSI/04-环境架构与运动库|04 环境架构与运动库]] | 环境：Isaac Gym 集成、MotionLib、RSI、地形 | 25 min |
| 05 | [[01-Projects/AMP-Reproduction/PhysHSI/05-任务实例-CarryBox拆解|05 CarryBox 拆解]] | 实战：最复杂任务的观测/奖励/终止/多阶段训练 | 35 min |
| 06 | [[01-Projects/AMP-Reproduction/PhysHSI/06-机器人配置与动作空间|06 机器人配置与动作空间]] | 工程：G1 29 DOF、PD 控制、域随机化、噪声模型 | 25 min |
| 07 | [[01-Projects/AMP-Reproduction/PhysHSI/07-训练部署与可视化|07 训练部署与可视化]] | 操作：训练命令、checkpoint、日志、ONNX 导出 | 20 min |
| 08 | [[01-Projects/AMP-Reproduction/PhysHSI/08-迁移指南-轮式底盘单臂|08 迁移指南]] ⭐ | 迁移：完整适配方案、代码模板、检查清单 | 40 min |

---

## 推荐阅读顺序

### 快速上手（最小路径，~90 min）
```
01 → 02 → 05 → 08
```
适用于：想快速理解 AMP 怎么用到操作任务上。

### 完整学习（全路径，~4h）
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

### 按需查阅
- 想改算法 → 02 + 03
- 想改环境 → 04 + 05
- 想改机器人 → 06
- 想部署 → 07
- 想迁移到自己的项目 → 08

---

## 代码速查

| 想看什么 | 文件 |
|---------|------|
| AMP 判别器 | `rsl_rl/rsl_rl/modules/amp.py` |
| HIM PPO 更新 | `rsl_rl/rsl_rl/algorithms/him_ppo.py:L182-L284` |
| Actor-Critic 网络 | `rsl_rl/rsl_rl/modules/actor_critic.py` |
| CarryBox 环境 | `legged_gym/legged_gym/envs/g1/carrybox.py` |
| CarryBox 配置 | `legged_gym/legged_gym/envs/g1/carrybox_config.py` |
| MotionLib | `legged_gym/legged_gym/envs/motionlib/motionlib_carrybox.py` |
| 训练入口 | `legged_gym/legged_gym/scripts/train.py` |
| 可视化 | `legged_gym/legged_gym/scripts/play.py` |
| 数学工具 | `legged_gym/legged_gym/utils/math.py` |
| 任务注册 | `legged_gym/legged_gym/utils/task_registry.py` |

---

## 关联

- [[01-Projects/AMP-Reproduction/AMP-Reproduction|AMP 复现项目入口]]
- [[01-Projects/AMP-Reproduction/PhysHSI/01-PhysHSI总览与论文对照|PhysHSI 总览 →]]
