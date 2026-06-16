# Raigor 移动抓取 Demo (Navigation & Grasp) 仿真实现与训练方案

本方案旨在结合 **PhysHSI** 的算法精髓（RSI 参考状态初始化、AMP 对抗风格奖励、两阶段训练）与 **raigor_amp** 已有的轮式机械臂仿真环境，设计并训练一个“底盘导航过去并平滑抓住目标物体”的 Demo。

---

## 1. PhysHSI 的核心精髓及其在移动操作中的映射

PhysHSI 解决复杂全身交互的核心在于三个“精髓”：

*   **精髓一：参考状态初始化 (Reference State Initialization, RSI) — 解决“探索稀疏”**
    *   *原理*：对于“开过去 -> 伸臂 -> 抓取”的长时程任务，随机探索几乎不可能成功。RSI 让机器人在 Episode 开始时，有概率（如 40%）直接从参考抓取轨迹的中间帧（如手臂已触碰物体、或夹爪已包络物体）初始化。
    *   *映射*：机器人先在物体前学会“合拢夹爪提起物体”；随着训练进行，成功率反向传播，逼迫策略学会前置动作（底盘如何导航对齐、机械臂如何展开）。
*   **精髓二：对抗运动先验 (Adversarial Motion Priors, AMP) — 解决“协同与画风”**
    *   *原理*：手动设计奖励很难协调“底盘平移”与“手臂关节转动”的相对节奏。AMP 引入判别器，输入底盘与手臂的联合状态转移 `[s_t, s_{t+1}]`，并与参考轨迹对比。
    *   *映射*：避免出现“底盘先开到桌前，停稳后再慢吞吞伸臂”的分步僵硬动作，自然训练出“边走边展开手臂，到达瞬间直接闭合夹爪”的动态协同风格。
*   **精髓三：两阶段训练范式 (Two-Step Training) — 解决“难易解耦”**
    *   *原理*：第一阶段放宽约束（如忽略轻微穿透），主攻“抓到目标”；第二阶段加载权重，调高 AMP 判别器权重，收紧碰撞和关节限幅，精细化打磨动作。

---

## 2. 现有资源组合与如何开始训练

您无需从零编写代码，`raigor_amp` 已经打通了 `isaaclab` 下的 AMP + RSL-RL 训练链路，且内置了 `raigor` 机器人的动作转换器和 `Raigor-Amp-v0` 任务。

### 步骤 1：激活环境并进入工作区
```bash
conda activate physhsi  # 或您的 isaaclab 虚拟环境
cd /home/mzy/rl_ws/src/raigor_amp/
```

### 步骤 2：冒烟测试（验证动作与环境渲染）
在后台训练前，先启动带 GUI 的界面，验证机器人的轮子驱动与机械臂 PD 响应：
```bash
python -m mm_sim_lab.scripts.play --task Raigor-Amp-Play-v0
```

### 步骤 3：启动训练 (Train)
使用 RSL-RL 与内置的 `AMPLoader` 载入 `datasets/raigor_amp` 下的协同携带轨迹进行训练：
```bash
python -m mm_sim_lab.scripts.train --task Raigor-Amp-v0 --num_envs 64 --max_iterations 1000 --headless
```
训练过程中的 Tensorboard 会输出 `mean_amp_loss`（判别器损失）与 `mean_style_reward`（风格奖励），您可以据此监控底盘与机械臂的协同程度。

---

## 3. “移动过去抓住东西” (Move & Grab) Demo 优化计划

为了针对性实现“移动过去抓住一个特定物体（如杯子/方块）”这一 Demo，需要在 `raigor_amp` 环境中进行如下优化：

### A. 搭建物体交互场景 (USD Scene)
1.  在 `mm_sim_lab/envs/raigor/raigor_amp_env_cfg.py` 的 `RaigorSceneCfg` 中添加一个桌子（Table）和一个目标物体（Object，如刚体方块）：
    ```python
    # 示例配置：在环境中生成刚体方块
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(usd_path=".../box.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(7.0, 1.0, 0.8)),
    )
    ```
2.  在机械臂末端（`wrist_3_link`）或夹爪内部配置 `ContactSensor` 接触传感器，以输出稳定的抓取接触信号。

### B. 录制抓取协同轨迹 (Reference Trajectory)
1.  录制一条包含以下三个节点的短轨迹：
    *   *Node 1*：底盘距离物体 3 米，机械臂从收拢位置开始抬起，底盘起步。
    *   *Node 2*：底盘接近桌子，速度减慢，机械臂末端对准物体，夹爪张开。
    *   *Node 3*：夹爪合拢闭合，检测到接触力，机械臂向上抬起 10 厘米。
2.  通过 `convert_amp_dataset.py` 将该轨迹转换并存为 `datasets/raigor_amp/grab_demo.npy`，更新 `weights.json` 让判别器以此为基准。

### C. 观测与奖励微调
1.  **新增 Policy 观测**：
    *   末端执行器相对物体的相对位置 `ee_to_object_pos_rel`。
    *   夹爪接触传感器力矩/状态 `gripper_contact_force`。
2.  **微调任务奖励 ($R_{task}$)**：
    *   接近奖励：末端贴近物体的指数距离惩罚。
    *   抓取奖励：当夹爪接触力大于阈值，且物体高度被提升时，给予高额的阶段性 Task Reward。
3.  **AMP 判别器协同**：
    *   设置判别器只关注底盘线速度、偏航角速度以及机械臂关节角速度的联合关系，确保在加速/减速过程中机械臂展开是平滑的。

### D. 单阶段快速训练实施 (Grab & Lift)

由于该任务的目标非常明确，即“开过去、抓住并拿起来”（对应 `carrybox` 任务的前半段），物理约束与长时程搬运任务相比要简单很多，因此**完全可以通过单阶段直接训练完成收敛**：

1.  **参数与权重分配**：
    *   设置适中的风格权重平衡（如 $w^g = 0.6, w^s = 0.4$），无需分步调整。
    *   保证任务奖励（EE 靠近物体、夹爪合拢力、物体上升高度）具有足够的梯度，引导底盘和手臂快速接触物体。
2.  **RSI 的高效区间截取**：
    *   Motion Library 不需要覆盖搬运后的长距离移动。
    *   重点截取“**靠近物体前 1 秒**”到“**物体被抓取并抬升 10 厘米后 1 秒**”的轨迹区间。
    *   将 RSI 重置概率设为 40%~50%，使机器人有极高概率直接在“手在物体旁”或“夹爪已抱紧”的时刻初始化。这会使策略在极少迭代内学会“合拢与抬起”，然后将梯度反向传导给导航段，一次性跑通全链路。
3.  **Episode 提早终止 (Early Termination)**：
    *   在 `TerminationsCfg` 中加入一个判定：一旦物体高度被抬升到指定高度（如 `target_height`）并保持稳定接触 0.5 秒，即视作任务成功，直接触发 Episode 重置。这能有效防止机器人抓起物体后，因为多余的随机探索而导致动作发散。
