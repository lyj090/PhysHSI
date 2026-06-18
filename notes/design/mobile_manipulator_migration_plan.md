# PhysHSI 架构迁移计划：移动抓取 (Mobile Grasping) MVP

## 1. 迁移目标与核心方法论

**当前目标**：为了降低初期迁移的风险，我们将复杂的“拿-搬-放”全流程缩小为**“移动抓取” (Mobile Grasping / Pick Task)** 的最小可行性验证 (MVP)。重点验证底盘移动与机械臂抓取的协同控制。

**保留的核心方法论**：
1. **AMP (Adversarial Motion Priors)**：使用对抗性先验，约束机器人生成平滑、协同的（非生硬的）控制策略。
2. **阶段性状态初始化 (Stage-wise Initialization / RSI)**：通过在任务的不同阶段进行初始化，解决长程 (Long-horizon) 任务梯度稀疏的问题。

---

## 2. 模块化需求清单 (Requirements by Module)

为了启动此 MVP，你需要准备并提供以下 4 个模块的材料和代码实现：

### 模块 A：仿真资产 (Assets Module)
这是整个环境的基础，定义了物理世界的实体。
- [ ] **机器人 URDF/MJCF 文件**：
  - 必须包含：**移动底盘**（如差速轮/全向轮模型，或用于简化仿真的虚拟位移节点 `Virtual Root`）。
  - 必须包含：**多自由度机械臂**（如 6-DOF UR5）。
  - 必须包含：**末端夹爪**（如两指 Gripper，需带有明确的开合驱动关节）。
- [ ] **目标物体资产**：
  - 一个简单的 `box.urdf` 或 `cylinder.urdf`，用于被抓取。
- [ ] **关节信息列表**：
  - 整理出底盘控制关节、机械臂控制关节、夹爪控制关节的具体名称列表（后续将写入 Config 文件）。

### 模块 B：专家数据 (Expert Data Module)
用于支撑 AMP 算法，为机器人提供“像样”的抓取参考。
- [ ] **轨迹数据集 (Trajectory Dataset)**：
  - 形式：`.pt` 或 `.npy` 格式的张量文件。
  - 数量：约 50-100 条短轨迹。
  - 内容：记录一段完整的 **“底盘驶近 -> 机械臂伸出 -> 夹爪闭合并抬起物体”** 的过程。
  - 数据维度要求：必须包含每一帧的 `[底盘线速度/角速度, 机械臂所有关节位置/速度, 夹爪关节位置, 目标物体的世界坐标]`。
- [ ] **数据获取工具**：
  - 需要你准备一个简单的脚本（基于 ROS/MoveIt 或人工遥操作），在外部环境（如 PyBullet 或真实世界）中录制上述动作并导出数据。

### 模块 C：环境核心逻辑 (Environment Module)
需要在 `legged_gym/envs/` 下创建一个新的子目录（如 `mobile_pick/`），并实现以下 Python 类。
- [ ] **`MobilePickTask` 类 (继承自 `BaseTask`)**：
  - **解耦操作**：彻底删除所有 `legged_robot.py` 中关于“空中时间 (Air Time)”、“脚部接触”、“双腿对称性”的硬编码逻辑。
  - **`_compute_torques` (核心控制)**：实现混合控制域。将网络输出的 action 拆分：一部分映射为底盘的目标速度指令 (CMD_VEL)，另一部分映射为机械臂/夹爪的目标 PD 位置指令。
  - **`compute_observations` (观测空间)**：拼接底盘速度、机械臂关节状态、**夹爪与物体的相对空间偏差向量**、夹爪开合状态。
  - **`compute_amp_observations` (AMP 判别器输入)**：提取底盘速度、手臂关节速度，用于评估运动的平滑与协同度。

### 模块 D：奖励与训练配置 (Config Module)
编写 `mobile_pick_config.py`，配置奖励函数和 RSI 初始化策略。
- [ ] **任务奖励函数 (Task Rewards)**：
  - `approach_reward`：底盘与物体距离的指数奖励。
  - `reaching_reward`：夹爪末端与物体距离的指数奖励。
  - `grasp_reward`：核心稀疏奖励。当夹爪闭合且物体 Z 轴高度提升（成功抓起）时给予巨大分值。
- [ ] **惩罚函数 (Penalties)**：
  - `dof_vel_limits`、`torque_limits`：防止机械臂抽搐或电机过载。
  - `singularity_penalty`：防止机械臂进入死锁姿态。
- [ ] **阶段性初始化分布 (RSI Probabilities)**：
  - 设计合理的 `skill_init_prob`：
    - `loco` (远处出发，练底盘导航) -> 建议 40%
    - `approach` (距离 0.5m，练伸臂) -> 建议 30%
    - `pre_grasp` (夹爪已触碰物体两侧，专练闭合并发力提起) -> 建议 30%

---

## 3. 实施路线图 (MVP Roadmap)

| 阶段 | 任务 | 验证标准 | 预期用时 |
| :--- | :--- | :--- | :--- |
| **Milestone 1** | 环境与资产导入 | 轮式机械臂在 Isaac Gym 中加载成功，底盘和手臂能分别响应随机 action。 | 1 天 |
| **Milestone 2** | 短程专家数据采集 | 能够通过 `play_dataset.py` 重放 50 条“移动并抓取”的规划轨迹。 | 1-2 天 |
| **Milestone 3** | 纯 RL 抓取训练 | 暂时关闭 AMP。开启 30% `pre_grasp` 初始化，机器人能够大概率抓起物体。 | 2 天 |
| **Milestone 4** | AMP 风格对齐 | 开启 AMP。机器人的抓取动作变得连贯、无剧烈抖动，且底盘移动时不发生明显甩臂。 | 2 天 |