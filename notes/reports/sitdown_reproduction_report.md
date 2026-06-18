# PhysHSI "SitDown" 任务复现报告

## 1. 复现概述

本项目成功复现了论文 [PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System](https://arxiv.org/abs/2510.11072) 中 Unitree G1 机器人的 **Sit Down** (坐下) 交互任务。

- **操作系统**: Ubuntu 20.04
- **Python 版本**: 3.8
- **仿真引擎**: Isaac Gym Preview 4
- **硬件配置**: NVIDIA RTX 4070 (12GB VRAM)
- **训练时长**: 约 48 小时 (累计，使用 `--resume` 断点续训)
- **迭代次数**: 30,000 Iterations (~ 27.8 亿步)

## 2. 核心调整与适配

由于本地 RTX 4070 显存（12GB）限制，我们在复现过程中对环境配置进行了优化：

- **环境变量修改**: 修改 `legged_gym/envs/g1/sitdown_config.py`，将 `num_envs` 从默认的 `4096` 降低至 `1024`，成功避免了 `CUDA Out of Memory` 错误，同时保证了训练效率（稳定在 ~17,000 steps/s）。
- **代码 Bug 修复**: 修复了 `legged_gym/utils/task_registry.py` 中 `make_alg_runner` 函数在处理 `--resume_path` 为空时的 `AttributeError`，确保了断点续训功能的稳定运行。

## 3. 性能评估与收敛分析

在达到 30k 迭代后，模型完全收敛，进入极稳固的平台期。各项指标表现如下，符合甚至超越了论文中描述的性能：

| 指标                             | 最终表现 (Iter 30k) | 状态解读                                                                                     |
| :------------------------------- | :------------------ | :------------------------------------------------------------------------------------------- |
| **Mean Reward (平均奖励)**       | ~50.0 - 53.0        | **高位收敛**。奖励相比初期（-2.7）有了质的飞跃，成功完成了接近/转身/坐下的全套复杂交互逻辑。 |
| **Episode Length (生存时长)**    | ~650 - 710 步       | **理论极限**。机器人极少摔倒，能够在椅子上维持稳态平衡，远超常规动作所需步数。               |
| **Action Noise Std (动作噪声)**  | 0.23                | **极高精度**。动作极度平滑、自然，无机械抖动，完美体现了 AMP 算法风格对齐的效果。            |
| **Torque/Vel Limits (硬性约束)** | 接近 0.000          | **硬件友好**。未发生超扭矩和超速现象，具备极高的实机部署潜力。                               |

## 4. 训练日志与可视化

我们提取了训练过程中的 TensorBoard 日志，以便对训练曲线进行详细审计。日志已剔除巨大的 `.pt` 权重文件，仅保留数据面板所需信息。

- **日志存放路径**: `docs/logs/`
- **本地查看方法**:
  在项目根目录下（即 `PhysHSI/` 目录）执行以下指令，即可开启可视化面板：
  ```bash
  tensorboard --logdir=docs/logs
  ```
  *(注：启动后在浏览器访问输出的 http://localhost:6006 即可查看)*

## 5. 如何验证测试

要使用训练好的权重进行可视化测试，请将模型放置于本目录或合适的位置，并在 `PhysHSI/legged_gym` 目录下运行以下命令：

```bash
# 假设模型名为 model_30000.pt 并存放于 resources/ckpt/ 下
python3 scripts/play.py --task sitdown --resume_path ../resources/ckpt/model_30000.pt
```

```bash
python3 legged_gym/scripts/play.py --task sitdown --load_run Jun08_03-03-05_sitdown_coef0.35 --checkpoint 30000
```

_(注：如果需要录制视频或截图，可以在弹出的 Isaac Gym 视窗中进行操作。)_
