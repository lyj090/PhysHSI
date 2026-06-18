# Raigor 移动抓取 Demo (Navigation & Grasp) 详细开发与训练计划

本方案是在 **PhysHSI** 的核心框架下，为 Raigor 轮式双臂/单臂移动操作机器人设计的“导航到桌前、平滑伸臂并抓取抬起目标物体”任务的详细开发与训练计划。

---

## 1. 任务设计与目标定义

本 Demo 的目标是训练一个统一的策略，完成以下串联任务：
1. **底盘导航 (Navigation)**：机器人底盘从随机初始位置（如距离桌子 1.5 ~ 3 米）移动并对齐到桌子前方。
2. **末端逼近 (Approach)**：机械臂在移动过程中逐步展开，将末端执行器（EE，基于 `wrist_3_link`）精确对准桌面上的目标物体。
3. **虚拟抓取 (Virtual Grasp)**：当末端执行器与物体的距离小于阈值（$d < 0.05\text{ m}$）时，触发吸附/抓取锁定（磁力/吸盘虚拟抓取模型），将物体固定在末端。
4. **稳定抬升 (Lift)**：抓取锁定后，机械臂将物体向上抬升至少 10 厘米，并保持底盘稳定。

### 成功指标
* 最终物体高度比初始桌面高度高出 $\ge 0.1\text{ m}$。
* 整个动作流畅、无剧烈抖动，底盘无碰撞翻车。
* Episode 结束时，物体被成功提起。


## 1.5 PhysHSI 精髓与任务契合度分析

虽然虚拟吸附简化了抓取瞬间的复杂指尖接触力学，但 PhysHSI 的三大支柱依然是跑通该移动操作任务的核心关键：
1. **RSI 解决稀疏探索**：导航到桌前并伸手抓取具有极长的时程。若完全从 3 米外随机探索，成功率接近零。RSI 让机器人频繁从“手在物体旁”或“物体已被抓起”的中间状态初始化，以极快速度训练出“抓取与抬升”，再通过价值梯度反传教会底盘“如何导航对准”。
2. **AMP 促成底盘-双臂并行动作**：常规 RL 会导致严重的“分步式僵硬行为”（如开到桌前完全停稳，再抬臂伸手）。AMP 学习协同运动分布，强迫策略实现“边导航边展臂，到达桌面瞬间完成抓取”的并行动画画风。
3. **两阶段训练解耦控制限幅**：第一阶段降低碰撞惩罚以让策略快速学到“凑近并捏起”的动作；第二阶段加载权重，加大碰撞和关节限幅惩罚，迫使机器人在接近桌子时平滑减速，实现无撞优雅操作。

---


## 2. 详细实现计划与步骤

我们通过 5 个开发闸门 (Gates) 逐步实现和测试：

### Gate 1: 场景搭建 (USD Scene Setup)
修改 [raigor_amp_env_cfg.py](file:///home/mzy/rl_ws/src/raigor_amp/mm_sim_lab/mm_sim_lab/envs/raigor/raigor_amp_env_cfg.py)，在 `RaigorSceneCfg` 中添加桌面（Table）和目标方块（Object）：
* **桌子配置**：位置为 `(2.5, 0.0, 0.2)`，大小为 `(0.8, 1.2, 0.4)` 米。
* **物体配置**：定义为 `RigidObjectCfg`，初始放置在桌面中心 `(2.5, 0.0, 0.43)`（桌面高度 0.4m + 方块半高 0.03m），大小为 `(0.06, 0.06, 0.06)` 米的刚体 Cuboid。

### Gate 2: 虚拟抓取逻辑实现 (Virtual Grasping)
由于机器人 USD 中未包含物理夹爪关节，我们采用 **吸附式虚拟抓取** 模型：
1. 新建一个自定义的环境类 `RaigorGraspEnv`，继承自 `ManagerBasedRLEnv`。
2. 在 `__init__` 中初始化 `self.grasp_attached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)`。
3. 在 `step()` 方法中：
   * 计算 `wrist_3_link` 修正后的 EE 位置与 Object 质心位置的三维距离 $d$。
   * 当 $d < 0.05\text{ m}$ 且未重置时，设置对应环境的 `self.grasp_attached` 为 `True`。
   * 对于 `self.grasp_attached == True` 的环境，直接修改仿真中 Object 的 Root State，使其位置与 EE 位置保持固定的抓取相对偏移，速度与 EE 对齐。
   * 在 `reset` 事件触发时，重置对应环境的 `self.grasp_attached` 为 `False`。

### Gate 3: 观测与奖励函数设计
在 [rewards.py](file:///home/mzy/rl_ws/src/raigor_amp/mm_sim_lab/mm_sim_lab/envs/base/rewards.py) 中新增或修改以下奖励项：
1. **EE 逼近物体奖励 ($R_{\text{approach}}$)**：
   $$r_{\text{approach}} = \exp\left(-\frac{\|p_{\text{ee}} - p_{\text{obj}}\|}{\sigma_p}\right)$$
   引导末端向物体靠近。
2. **物体抬升奖励 ($R_{\text{lift}}$)**：
   $$r_{\text{lift}} = \text{clamp}(z_{\text{obj}} - z_{\text{table\_top}}, 0.0, 0.2) \times w_{\text{lift}}$$
   当物体被抬起时，提供正向的 Task Reward 梯度。
3. **底盘速度平抑惩罚 ($R_{\text{chassis}}$)**：
   当末端距离物体极近或已抓取时，限制底盘的线速度与角速度，防止推倒桌子或动作过冲。
4. **抓取成功阶段性奖励 ($R_{\text{grasp}}$)**：
   一旦 `grasp_attached` 首次转为 `True`，给予高额的一次性奖励。

### Gate 4: RSI 重置状态生成与 AMP 数据集转换
1. **截取协同轨迹**：从现有的协同携带数据集（如 `carry0.txt` 等）中截取前部“靠近并举起”的帧。
2. 使用 [convert_amp_dataset.py](file:///home/mzy/rl_ws/src/raigor_amp/mm_sim_lab/mm_sim_lab/scripts/convert_amp_dataset.py) 转换为 `grab_demo.npy`。
3. 修改 [raigor_amp_env_cfg.py](file:///home/mzy/rl_ws/src/raigor_amp/mm_sim_lab/mm_sim_lab/envs/raigor/raigor_amp_env_cfg.py) 的 `reset_ref` 事件，将 RSI 状态重置概率提升至 40% ~ 50%，使策略能够直接从“手处于物体旁”的状态开始探索抓取和抬升。

### Gate 5: 训练与调优流程
1. **第一阶段（快速单阶段训练）**：
   * 采用 $w_{\text{task}} = 0.7, w_{\text{style}} = 0.3$。
   * 设置 Episode 提前终止：当物体抬升高度达到 15 厘米并保持稳定 0.5 秒时，判定成功并结束 episode。
2. **测试与可视化**：
   * 运行测试脚本验证 `RaigorGraspEnv` 能否正常 Reset 并运行：
     ```bash
     python -m mm_sim_lab.test.test_env --task Raigor-Grasp-v0 --num_envs 1 --steps 100 --headless
     ```
   * 启动训练：
     ```bash
     python -m mm_sim_lab.scripts.train --task Raigor-Grasp-v0 --num_envs 64 --max_iterations 1000 --headless
     ```
   * 启动 Play 验证训练效果：
     ```bash
     python -m mm_sim_lab.scripts.play --task Raigor-Grasp-v0
     ```

---

## 3. 具体代码结构调整设计

### A. 注册新环境任务
在 [envs/\_\_init\_\_.py](file:///home/mzy/rl_ws/src/raigor_amp/mm_sim_lab/mm_sim_lab/envs/__init__.py) 中注册 `Raigor-Grasp-v0`：
```python
_register_once(
    "Raigor-Grasp-v0",
    entry_point="mm_sim_lab.envs.raigor.raigor_grasp_env:RaigorGraspEnv",
    kwargs={
        "env_cfg_entry_point": "mm_sim_lab.envs.raigor.raigor_amp_env_cfg:RaigorGraspEnvCfg",
    },
)
```

### B. 核心抓取类实现方案
创建新文件 `mm_sim_lab/envs/raigor/raigor_grasp_env.py`，实现 `RaigorGraspEnv` 类。在每一步的 `step()` 结束前，如果 `grasp_attached` 为真，则使用 Isaac Lab API 重置物体的物理位置与姿态：
```python
import torch
from isaaclab.envs import ManagerBasedRLEnv

class RaigorGraspEnv(ManagerBasedRLEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.grasp_attached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_asset = self.scene["object"]
        
    def step(self, actions):
        # 正常物理仿真步进
        obs, rewards, terminated, truncated, info = super().step(actions)
        
        # 物理吸附状态检测与更新
        ee_pos = info.get("ee_pos", None) # 或通过观测器获取
        if ee_pos is not None:
            obj_pos = self.object_asset.data.root_pos_w
            dist = torch.norm(ee_pos - obj_pos, dim=-1)
            new_attach = (dist < 0.05) & (~self.grasp_attached)
            self.grasp_attached = self.grasp_attached | new_attach
            
        # 若吸附，锁定物体的姿态
        if self.grasp_attached.any():
            # 获取 EE 姿态并直接写入物体的刚体属性
            pass
            
        return obs, rewards, terminated, truncated, info
```

### C. 终止条件优化
在 [raigor_amp_env_cfg.py](file:///home/mzy/rl_ws/src/raigor_amp/mm_sim_lab/mm_sim_lab/envs/raigor/raigor_amp_env_cfg.py) 中，对 `TerminationsCfg` 引入任务成功终止判定，防止学到抓起后再丢下的发散行为。
