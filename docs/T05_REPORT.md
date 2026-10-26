# T05 固定印放光号偏差修正

基于 `324ac5d510a0b52ff35fae37d11db9115681319e`，分支 `codex/t05-lab-balance`。

## 固定值与来源

三个值逐位照抄 `docs/tasks/T05_LAB_BALANCE.md`，作为 `filmlab.engine` 模块常数：

| 常数 | 值 |
|---|---:|
| LAB_BALANCE_DELTA_K | -0.11437142396955352 |
| LAB_BALANCE_DELTA_R | -0.12100152025600766 |
| LAB_BALANCE_DELTA_B | -0.008793146365340334 |

来源是 T04 固定TRAIN 11帧 O1 偏移的中位数，数据提交 `663232d16005e2ba44ebe47310509217c52332f1` 的 `tools/eval/t04_result.json`；固定值由规格提交 `324ac5d` 给定。本实现没有重新估计、拟合或按ΔE调整这些值，没有读取留出光号来决定参数。

## 施加位置与作用范围

新增可选标志 `lab_balance`（列入 `Params.corrections` 注释）。新增 `next_balance`，它的白平衡、修正集及材料版本等于 `next` 加 `lab_balance`；未修改 `current`、`next`、`legacy` 或现有配方。

在 `Engine.scan` 的 `timing="frame"` 分支中，`Scanner.balance_frame` 完成之后、写入trace及相纸成像之前，应用 `_lab_balance`：

- K三个通道加相同ΔK。
- M的off加 `(ΔR, 0, ΔB)`；原M为None时先按off=0、r=1处理。
- 用 `dataclasses.replace` 保留r、s、x_knee、x_white等其余字段，不修改原PrintTiming。

此处与 `oracle_timing.fit(1)` 的K/M施加位置和单位一一对应。每次扫描先重新自动定光，再加固定Δ，不会随重复扫描累积。不增加逐帧或按场景的规则。

标志关闭时仅跳过新分支，不执行新算术。gray、standard定光及物理参考配方不进入该分支，输出不变；实际光学印放路径同样没有改动。没有RGB调色层、额外系数或其他常数修改。

## 合成验证

- 精确常数、含M与M=None两种光号映射；K与off偏移误差<1e-12；绿偏移、r、s和肩部字段不变。
- 实际引擎frame扫描trace：Dn不变，自动定光后恰好加Δ，重复扫描不累积。
- gray、standard、`cinestill_800t_reference` 开关标志后的像素逐位不变。
- `current`、`legacy`、`next` 默认修正集不含新标志；关闭时不调用新算术。
- 将基准324ac5d导出到隔离目录，在独立进程渲染所有现有RECIPES及额外frame合成输入：三代共 **102** 项像素与基准逐位一致。
- 全量测试：**375 passed, 3 warnings in 131.09s (0:02:11)**。详细记录在 `tools/eval/t05_test_result.json`。

全部验证只用固定种子的合成数组，没有运行或导入 `accept_t05.py`，没有运行真实场景或报告真实场景结果。可选代际仍需Claude在验收环境独立验收并由用户目视，当前不声称实景改善或验收通过。
