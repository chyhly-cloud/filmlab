# 任务 T01：胶片参数校准（执行：Codex；规格与验收：Claude）

## 背景与原则

用户确定：拟合只能纠正显影链里**已有**的物理量，不新增任何 RGB 调色层。依据是 `docs/CALIBRATION_PARAMETER_INVENTORY.md`：
在物理范围内可辨识、且对冲印店无法补偿的误差有显著影响的胶片参数只有三个。其余参数（染料峰移、光晕的其他子参数、扫描 LED / 印放矩阵 P）
**保持现值，不得拟合**。冲印店差异（亮度定光、相纸肩部）属于后续任务 T02，本任务不碰。

## 参数与范围（不得放宽）

| 参数 | 含义 | 范围 | 现值 |
|---|---|---|---|
| `tau` | C-41 冲洗 ECN-2 乳剂的显影程度 τ | 1.0–1.3 | 1/0.85 ≈ 1.176 |
| `dir` | DIR 层间抑制强度倍数（`chemistry.develop_image` 的 inhib_amount 乘数） | 0.6–1.4，且见下方物理检查 | 1.0 |
| `diffuse_fraction` | 光晕总强度（`Variation.diffuse_fraction`；它代表整个光晕方向，其余光晕子参数保持现值） | 0.45–0.75 | 0.6 |

DIR 物理检查（必做）：拟合值下，C-41 中 500T 的"分色曝光 gamma / 中性曝光 gamma"必须落在 Kodak US 6,210,870 的 C-41 负片实测范围内：
R 1.15–1.49，G 1.42–1.64，B 1.29–1.50（见 `chemistry._dir_matrix` 注释）。超出即不合格。

## 数据与目标函数

- 评估配置固定：`generation.use("next")`、`Params(recipe("cinestill_800t"), timing="frame", roll=None, grain=0)`；
  每个场景用定光预言机 O1（只给三个印放光号）后，按格子取 ΔE00 中位。实现直接调用
  `tools/eval/param_sensitivity.py: evaluate_scene(sid, roll=None)`，参数注入方式与 `tools/eval/accept_calibration.py: apply()` 完全一致
  （不要另写一套注入，以免与验收不一致）。
- **只能使用训练集**（`accept_calibration.TRAIN`，11 个 EPFL 场景）。留出集（`HOLDOUT`，10 个场景，含色卡与他店咖啡店）
  不得用于拟合、调参、选择初值或提前查看。
- 目标：训练集各场景 ΔE00 中位的平均值最小。
- 方法建议：先在范围内做粗网格（例如 5×5×4），再用有界局部优化细化；总评估次数控制在约 300 次以内并全部记录。

## 禁止事项

- 不改 `current`、`next`、`legacy` 及任何现有默认值；不改 `SATURATION_MATRIX`、`LAB_TINT`、扫描参数。
- 不新增参数、不放宽范围、不加 RGB 后处理。
- 拟合值贴住边界（距边界 < 2% 范围）时：**停止并报告**，不要放宽或换参数。
- 训练变好但留出不通过：如实报告为失败，不要换拆分或反复重试。

## 交付物（推到新分支 `codex/t01-film-calibration`，基于 `handoff/recovered-research-2026-10-23` 最新提交）

1. `tools/eval/calibrate_film.py`：拟合程序（可复现，固定随机种子）。
2. `tools/eval/t01_calibration_result.json`：最终参数、全部评估记录（参数 → 训练平均 ΔE）、DIR 分色 gamma 比检查值。
3. 一个**可选**代际 `next_cal`（在 `filmlab/generation.py` 中，next + 校准值；实现方式应让 `current/next/legacy` 输出逐位不变），以及测试。
4. `docs/T01_REPORT.md`：简短报告——最终参数、是否贴边界、训练集改善、DIR 物理检查结果；不要自报留出结果。

## 验收（由 Claude 独立执行）

`FILMLAB_DATA=… python3 tools/eval/accept_calibration.py 参数.json 输出.json`，基准（`{}`，现值，2026-10-23 测定）：
训练平均 ΔE00 **4.879**，留出平均 **5.679**，3200K 灰卡彩度 0.21。通过条件（全部满足）：

1. 参数在范围内且不贴边界；DIR 分色 gamma 比在专利范围内；
2. 3200K 灰卡彩度 < 0.5；ECN-2 原生中性灰阶 Status M 相对现行变化 ≤ 0.01 D；
3. 留出平均改善 ≥ 0.2；任何留出分组变差 ≤ 0.3；任何留出场景变差 ≤ 0.8；
4. 另由 Claude 跑 `current/legacy/next` 严格逐位回归（42/42）与全部测试；并出对比图交用户目视。

（验收工具已用试探参数 τ=1.28、DIR×1.3 自检：留出变差 0.69，被正确判为不通过。）
