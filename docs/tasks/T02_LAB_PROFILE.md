# 任务 T02：EPFL 冲印店扫描档案校准（执行：Codex；规格与验收：Claude）

## 背景

T01 已证明：物理范围内的胶片参数改不了主要问题（留出改善仅 0.11，未通过，见 `T01_ACCEPTANCE.md`）。追踪结果指向扫描层：
预言机定光下残差约 5 ΔE，而实际自动定光约 12 ΔE；灯芯周围被印成近白、夜景中间调偏亮（`CRITERIA.md`「光源轮廓与黄白泛光」）。
本任务只校准**扫描 / 冲印店过程**的现有设置，形成一个**只代表 EPFL 这家 Frontier 冲印店**的可选档案。胶片参数一律不动。

## 参数与范围（不得放宽）

| 参数 | 代码位置 | 范围 | 现值 |
|---|---|---|---|
| `latd_normal` | `engine.LATD_NORMAL`（正常场景平均密度偏移） | −0.40 … +0.20 | −0.14 |
| `latd_limit` | `Variation.latd_limit`（逐帧亮度校正上限，D） | 0.02 … 0.50 | 0.12 |
| `highlight_aim` | `timing.HIGHLIGHT_AIM`（高光侧目标反射率） | 0.45 … 0.90 | 0.65 |
| `shoulder_white` | `timing.SHOULDER_WHITE`（额外肩部渐近白） | 0.85 … 0.995 | 0.93 |
| `color_strength` | `Params.scan_color_strength`（逐帧颜色校正力度） | 0 … 1 | 标称 0.925 |
| `conditioning` | 输出调整：`"lab"`（现值）或 `"none"` | 离散二选一 | `"lab"` |

不得改动：胶片 / 显影 / 光晕参数、`SATURATION_MATRIX` 与 `LAB_TINT` 的数值、印放矩阵 P、LED、`current/next/legacy` 的任何默认值。

## 数据与目标函数

- 拆分与 T01 相同（`accept_calibration.TRAIN` / `HOLDOUT`）。**只能使用 TRAIN 11 个场景**；留出不得查看或用于任何选择。
- 评估必须直接调用 `tools/eval/accept_t02.py` 的 `apply()` 与 `scene_job()`（实际自动定光、roll=None、grain=0，对齐到真实扫描），不要另写一套。
- 目标：`mean_TRAIN(全图 ΔE00 中位) + 0.5 × mean(有光源的 TRAIN 场景的光源旁 ΔE00 中位)`，越小越好。
- `conditioning` 两个取值分别拟合连续参数，取训练目标更好的一个；两者结果都记录。
- 粗网格 + 有界局部优化，总评估 ≤ 400 次，全部记录，固定随机种子。

## 停止规则

- 任一连续参数贴边界（距边界 < 2% 范围）：停止并报告，不放宽、不换参数。
- 训练改善而留出不通过：如实报告失败，不换拆分、不重试。

## 交付物（新分支 `codex/t02-lab-profile`，基于 `handoff/recovered-research-2026-10-23` 最新提交）

1. `tools/eval/calibrate_lab.py`、`tools/eval/t02_profile_params.json`（验收用参数，键名同上表）、`tools/eval/t02_result.json`（全部评估记录）。
2. 可选档案的正式接入：给 `Params` 增加对应的可选覆盖字段（默认 `None` = 沿用现有模块常数），并新增
   `recipe("cinestill_800t_epfl_profile")`。`current/next/legacy` 及现有配方输出必须逐位不变；附测试。
3. `docs/T02_REPORT.md`：参数、是否贴边界、训练目标改善；不要自报留出结果。

## 验收（Claude 独立执行）

`FILMLAB_DATA=… python3 tools/eval/accept_t02.py t02_profile_params.json 输出.json`。基准（现值，2026-10-24 本机测定）：
训练 全图 **12.308**、光源旁 **14.873**；EPFL 留出 全图 **11.387**、光源旁 **17.504**；3200K 灰卡彩度 0.212。
通过条件（全部满足）：参数不贴边界；灰卡彩度 < 0.5；EPFL 留出全图平均改善 ≥ 0.5、光源旁平均改善 ≥ 0.5；
任何留出分组变差 ≤ 0.5、任何留出场景变差 ≤ 1.5。咖啡店（他店）只报告不判定。另由 Claude 跑三代际严格回归与全部测试，
并把留出场景的完整画面与光源局部对比图交用户目视——**最终由用户目视决定是否采用**。

（自检：试探参数 `{"conditioning":"none","latd_normal":-0.25}` 留出全图变差 1.81、钨丝灯箱组变差 7.3，被正确判为不通过。）
