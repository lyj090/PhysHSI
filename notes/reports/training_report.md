# Raigor Grasp Task Training Progress Report (Overnight Run)

This file tracks the long-term training status of the non-intrusive `Raigor-Grasp-v0` task on Isaac Lab, monitored hourly.

## Training Configuration
* **Task Name**: `Raigor-Grasp-v0`
* **Environments**: 64
* **Max Iterations**: 71000 (Expected to finish around 2026-06-18 08:46:00)
* **Device**: `cuda:0`

## Monitoring Log

| Timestamp | Iteration / Step | Mean Reward | Mean Ep Len | Approach Reward | Lift Reward | Grasp Success | AMP Loss | Style Rew/Step | Task Rew/Step | Status / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-06-17 02:26:33 | 13 / 26624 | 20.88 | 187.14 | 0.0067 | 1.5020 | 0.0000 | 0.1284 | 0.047 | 0.038 | Running. Started overnight training run. |
| 2026-06-17 03:00:00 | 2000 / 4098048 | 212.39 | 500.00 | 0.0272 | 1.9200 | 0.0000 | 0.6541 | 1.331 | 0.041 | Running. Metrics stable. |
| 2026-06-17 04:00:00 | 4000 / 8194048 | 200.17 | 500.00 | 0.0493 | 1.3200 | 0.0000 | 0.6377 | 1.259 | 0.031 | Running. Stable performance. |
| 2026-06-17 05:00:00 | 7700 / 15771648 | 192.43 | 500.00 | 0.0002 | 0.0000 | 0.0000 | 0.6190 | 1.256 | 0.011 | Running. Policy style tuning. |
| 2026-06-17 06:00:00 | 10670 / 21854208 | 193.28 | 500.00 | 0.0019 | 0.0000 | 0.0000 | 0.6358 | 1.269 | 0.003 | Running. Stable performance. |
| 2026-06-17 07:00:00 | 13440 / 27527168 | 199.67 | 500.00 | 0.0000 | 0.0000 | 0.0000 | 0.6712 | 1.330 | 0.003 | Running. Stable performance. |
| 2026-06-17 08:00:00 | 16300 / 33384448 | 200.94 | 500.00 | 0.0000 | 0.0000 | 0.0000 | 0.6744 | 1.351 | 0.001 | Running. Policy stable. |
| 2026-06-17 09:00:00 | 19200 / 39323648 | 202.22 | 500.00 | 0.0007 | 0.0000 | 0.0000 | 0.6774 | 1.346 | 0.001 | Running. Steady training. |
| 2026-06-17 09:22:13 | 20346 / 41670656 | 201.73 | 500.00 | 0.0008 | 0.0300 | 0.0000 | 0.6737 | 1.342 | 0.001 | Stopped. Checkpoint saved at logs/Raigor-Grasp-v0/20260617_022612/model_20200.pt. |

*Last updated: 2026-06-17 09:22:13*
