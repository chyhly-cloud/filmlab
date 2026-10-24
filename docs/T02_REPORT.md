# T02 EPFL 冲印店扫描档案校准

- 基线：`handoff/recovered-research-2026-10-23` / `c2f2df7`。
- 执行状态：`complete`；只使用固定 TRAIN 11 场景，22/22 输入文件已重新核验 SHA256。
- 只拟合 latd_normal、latd_limit、highlight_aim、shoulder_white、color_strength，conditioning=lab/none 分别拟合，按训练目标选较小者；不依据留出结果做选择。
- 目标 = 11 场景全图 ΔE00 中位的均值 + 0.5 × 有有效光源邻域的训练场景 ΔE00 中位的均值。无有效邻域记 null、不当成零；邻域场景集合必须与该组基准一致。
- 评估直接调用 accept_t02.scene_job()，由其内部恰好调用一次 apply()；实际自动定光，next、roll=None、grain=0。每场景新进程，不使用 O1 预言机。
- 固定种子 20261024；每组：现值基准 + 43 点稀疏内部网格（中心、各轴四分位点、内部角点），再做原范围内有界 Powell。每组最多 200 次，总计 ≤400；所有场景和候选原子存盘、可续跑。
- 任一已评估候选触及 2% 边界区域，不论目标是否改善，立即停止该组；仍独立完成另一个 conditioning。不扩大范围、不换参数、不重启已终止搜索。

| conditioning | 状态 | 候选数 | 最佳全图 | 最佳光源旁 | 最佳目标 |
|---|---|---:|---:|---:|---:|
| lab | stopped_boundary | 61 | 9.478259 | 15.204246 | 17.080381 |
| none | stopped_boundary | 62 | 9.637086 | 16.430177 | 17.852174 |

每组最佳参数与停止事件：

- lab 最佳参数：`{"latd_normal": -0.046633217005056515, "latd_limit": 0.26000000000000006, "highlight_aim": 0.675, "shoulder_white": 0.9225, "color_strength": 0.25, "conditioning": "lab"}`；最佳点是否贴边：False。
- lab 触边停止事件：`{"index": 60, "keys": ["latd_limit"], "params": {"latd_normal": -0.046633217005056515, "latd_limit": 0.4936853036017161, "highlight_aim": 0.675, "shoulder_white": 0.9225, "color_strength": 0.25, "conditioning": "lab"}}`。
- none 最佳参数：`{"latd_normal": -0.058239894875685316, "latd_limit": 0.13999999999999999, "highlight_aim": 0.675, "shoulder_white": 0.9225, "color_strength": 0.5, "conditioning": "none"}`；最佳点是否贴边：False。
- none 触边停止事件：`{"index": 61, "keys": ["latd_limit"], "params": {"latd_normal": -0.058239894875685316, "latd_limit": 0.49368530360171603, "highlight_aim": 0.675, "shoulder_white": 0.9225, "color_strength": 0.5, "conditioning": "none"}}`。

- 本机 lab 现值基准：全图 12.281277，光源旁 15.008418，目标 19.785486。
- 规格的训练基准为全图 12.308、光源旁 14.873；不同环境/配准引起的差异应由 Claude 独立复核，没有为贴合基准修改数据、评估器或参数。
- 训练选择：lab，来源拟合状态 `stopped_boundary`；参数 `t02_profile_params.json` 是该组最佳已评估点，不保证该组已正常收敛或没有触边事件。
- 训练目标相对 lab 现值改善：2.705105。
- 可选配方 `cinestill_800t_epfl_profile` 的 enabled=False。触边或未正常完成的来源组不得启用，即使它的最佳点在内部且训练目标更小。最终采用仍需 Claude 独立验收和用户目视批准。
- 正式接入使用 Params 的四个可选扫描覆盖字段，默认 None；颜色力度与 conditioning 使用现有字段。不修改全局常数，不污染其他引擎；胶片/显影/光晕及矩阵/tint/P/LED 数值保持原样。
- 改动前后合成输入逐位回归退出码：0；三代际各 42 项，共126项，这是6幅合成输入×7种配置，不冒称 Claude 的真实回归夹具。
- 全量测试退出码：0；完整输出见 `t02_test_result.json`。
- 未查看或运行留出数据、缓存或验收结果，未自行报告留出指标。

## 复现与独立验收

拟合命令（已终止的完整日志不会自动重开搜索）：

```bash
FILMLAB_DATA=… OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/eval/calibrate_lab.py
```

由 Claude 独立执行：

```bash
FILMLAB_DATA=… python tools/eval/accept_t02.py tools/eval/t02_profile_params.json 输出.json
```

请先审查两组停止状态、边界事件和完整评估记录，再核验灰卡、默认输出逐位回归、留出验收及他店隔离。完整画面与光源局部对比图应交用户目视；不放宽范围、不根据留出结果重新拟合。
