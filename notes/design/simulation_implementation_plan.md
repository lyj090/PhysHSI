# 移动操作机器人（轮式底盘 + 单机械臂）仿真实现与任务开发计划

本计划针对 **移动操作机器人（Mobile Manipulator，如轮式底盘 + 单机械臂）** 的运动控制与协同交互任务进行优化，旨在利用 **PhysHSI** 的核心思想（RSI 与 AMP 对抗风格奖励）以及 **raigor_amp** 的工程结构（差速底盘控制与模块化环境），实现底盘与手臂高动态协同交互。

---

## 1. 轮式移动操作系统的资源与挑战差异 (Humanoid vs. Mobile Manipulator)

| 维度 | 人形机器人 (G1, PhysHSI 默认) | 轮式移动操作机器人 (Raigor, 优化目标) |
| --- | --- | --- |
| **自由度与树级拓扑** | 29+ 自由度 floating-base，高维高耦。 | SE(2) 底盘 + 6 DoF 机械臂 (如 UR5e)，相对低维。 |
| **动作空间 (Action)** | 关节 PD 目标位置 (全机力矩/角度控制)。 | 8 维 `[v_x, w_z, arm_delta_pos * 6]` (底盘线/角速度 + 关节增量)。 |
| **稳定性与平衡** | 必须控制重心平衡与防跌倒，终止条件严格。 | 底盘天然稳定，终止条件主要为碰撞、自锁或限位。 |
| **参考数据源** | AMASS/SAMP 等人体 Mocap 数据集。 | 遥操作 (Teleop)、MoveIt/OMPL 规划轨迹或关键帧插值动作。 |
| **主要协同挑战** | 双脚与地面的接触力、全身平衡控制。 | **基座-手臂的动态协同**（“边走边抓”、预先调整朝向、惯性防倾）。 |

---

## 2. 仿真核心组件设计 (Core Simulation Components)

为了在 Isaac Lab 中训练轮式机械臂的协同交互任务，核心组件优化如下：

### 2.1 动作空间：底盘-手臂混合控制 (Chassis-Arm Action)
采用 `raigor_amp` 中使用的差速-关节混合动作控制器：
*   **输入 Action**：8 维张量 `[v_x, w_z, delta_q_1 ... delta_q_6]`。
*   **内部映射**：
    *   `v_x`（底盘前向线速度）与 `w_z`（底盘偏航角速度）直接转换为轮子的转速或履带的驱动输入。
    *   `delta_q_1...6` 累加到机械臂当前的关节目标位置上，经由关节级高频 PD 控制器（Stiffness ~ 3000, Damping ~ 50）驱动机械臂。

### 2.2 观测空间 (Observation Space)
*   **Policy 观测**：
    *   底盘线速度与角速度 (Base linear & angular velocity)。
    *   机械臂各关节的当前位置与速度（相对于初始关节姿态）。
    *   末端执行器 (EE) 当前位姿（相对于机器人基座）。
    *   末端执行器目标位姿或目标物体的相对位姿。
    *   历史动作信息 (Last action)。
*   **AMP 观测 (判别器输入)**：
    *   提取关键连杆（如底盘 root、机械臂肘部、末端执行器）在相邻两个物理步骤间的状态转移张量 `[s_t, s_{t+1}]`。
    *   用于判别底盘的行进速度与手臂的拉伸幅度是否自然协调。

### 2.3 动作库与参考状态初始化 (RSI)
对于“底盘移动到物体前并将物体拿走（类似 CarryBox）”的长时程交互：
1.  **轨迹收集**：使用 OMPL/MoveIt 离线生成 20~50 组平滑的协同抓取移动轨迹（底盘平动、转动，手臂同步伸出、收回），将其转换为 PhysHSI 可读取的 `.npy` 数据格式。
2.  **RSI 随机初始化**：在强化学习每个 Episode 启动时，有 30%~50% 的概率直接将底盘与手臂初始化在“接近抓取物体的瞬间”或“抓取后的协同运送路径中”。这消除了强化学习探索“基座必须对齐物体才能抓取”的冷启动障碍。

### 2.4 协同奖励函数优化 (Optimized Rewards)
$$R_{total} = w^g R_{task} + w^s R_{style}$$

*   **任务奖励 ($R_{task}$)**：
    *   末端位姿跟踪 (EE Pose tracking)：促使机械臂向目标点移动。
    *   避障与碰撞惩罚 (Collision penalty)：减少底盘与手臂和环境物体的硬碰撞。
    *   可达性/可操作度惩罚 (Manipulability penalty)：惩罚机械臂拉伸至极限位置（奇异点），促使底盘主动向目标靠近。
*   **对抗协同风格奖励 ($R_{style}$)**：
    *   基于 LSGAN 的判别器判分。如果机器人底盘不移动而只拉伸机械臂去够物体（不协调运动），判别器将给出极低评分。
    *   迫使网络学出：在底盘接近物体的同时，手臂以平滑的轨迹同步抬起，实现流畅的 Mobile Manipulation 效果。

---

## 3. 开发实施步骤 (Implementation Steps)

### 阶段一：基础模型与差速控制器验证
1.  在 Isaac Lab 中加载您的轮式/履带机械臂 USD 资产。
2.  编写极简的测试脚本验证驱动逻辑：
    ```bash
    python legged_gym/scripts/test_action.py --task [your_mm_task]
    ```
    验证给定的 `[v_x, w_z]` 动作输入是否能正确驱动底盘，并且机械臂的 6 个轴转动方向是否符合预期。

### 阶段二：参考数据集录制与转换
1.  规划 10 秒左右的底盘移动 + 手臂抓取交互路径。
2.  利用转换脚本将位置坐标序列导出为 `[your_mm_task].npy` 字典文件。
3.  放入 `legged_gym/resources/dataset/`，并在 `legged_gym/envs/motionlib/` 中注册专属动作库。

### 阶段三：搭建仿真任务环境
1.  在 `legged_gym/envs/g1/` 目录下（或者新建 `mm/` 目录）创建 `your_mm_task.py` 与 `your_mm_task_config.py`。
2.  设置底盘的刚体属性和轮地摩擦力，设置手臂与夹爪的碰撞传感器。

### 阶段四：两阶段式强化学习训练 (Train)
1.  **引导训练 (Stage 1)**：放宽机械臂拉伸极限惩罚与避障条件，将 AMP 权重设为较小值（如 `0.2`），重点在于让机械臂末端快速接触到目标。
    *   命令：`python legged_gym/scripts/train.py --task your_mm_task --headless`
2.  **精细化训练 (Stage 2)**：收紧碰撞判定与关节极限，将 AMP 权重提高至 `0.65`（引入强协同约束），加载阶段一的 checkpoint 进行微调。
    *   命令：`python legged_gym/scripts/train.py --task your_mm_task_resume --resume --resume_path [stage1_ckpt_path] --headless`

### 阶段五：真机对接与 ONNX 部署
1.  使用 `play.py` 评估策略，观察底盘与手臂是否出现抖动。
2.  确认无误后将 Policy 网络导出为 ONNX。
3.  通过 [raigor_sim2any](file:///home/mzy/rl_ws/src/raigor_sim2any) 部署到 ROS 控制层，对接底盘差速话题（如 `/cmd_vel`）和机械臂关节控制话题（如 `/joint_speed_command`）。
