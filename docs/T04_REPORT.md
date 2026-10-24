# T04 冲印店逐帧校色力度：前置闭合失败，按规则停止

基线：`a43c2af1162eff92f72513ed0d349e9cae6a2235`，分支 `codex/t04-lab-timing`。**T04测量未完成；没有k估计、置信区间或冲印店校色结论。**

## 停止事件

按固定TRAIN∪HOLDOUT顺序，第一场景 `R1-00-1` 直接调用 `accept_t02.scene_job(({}, sid))`，未另写渲染、配准、自动定光或ΔE计算：

| 项目 | 值 |
|---|---:|
| 提交基准 t03_baseline.json | 10.025349766370226 |
| 本机scene_job auto全图ΔE00中位数 | 10.681290489485985 |
| 绝对差 | 0.655940723115759 |
| 要求 | 严格小于1e-6 |

本次观察来自唯一一次真实场景前置计算（持久任务 `t04-preflight`）；它完成后即停止，没有运行其他20个场景，没有重复重渲染以寻找较接近基准的结果。原始完成事件已转录到 `tools/eval/t04_result.json` 的metadata.auto_checks。

![auto闭合诊断](figures/t04/auto_closure.png)

## 已确认与未确认的原因

- R1-00-1的RAW与扫描SHA256均与提交的epfl_800t_manifest.csv一致。文件名、实际与预期哈希见结果JSON。
- 使用现有 `filmlab-t01-data/oracle/geo_R1-00-1.npz` 固定配准缓存；缓存SHA256为 `2dee66510ac7a8dd25d5539c204d515c3f72db7ad34c197eed48fb7714ca41b5`，未重建或更换。
- 与先前T02本机现值基准的该场景结果一致（10.681290489485985）；这支持它不是本次T04生产改动引入的问题，但不能证明它与验收方的配准和数值运行时相同。
- 提交基准只提供结果值，没有可对照的几何缓存指纹、LibRaw构建和完整运行时指纹；**尚不能将差异确定归因于配准还是环境**。不以“可能是环境差异”为由忽略停止规则。
- 本次Python/NumPy/SciPy/colour/rawpy/scikit-image/Pillow版本均记录于JSON；LibRaw为0.22.1。闭合测量完成后另安装Matplotlib 3.11.2用于报告绘图，未更新上述测量依赖，也未重算场景。未为匹配结果更换版本、缓存、默认值或基准。

## 未产生的测量结果

| 交付测量 | 状态 |
|---|---|
| 20帧O1、逐帧颜色/密度光号 | 未运行 |
| 每帧≥200次24px格子bootstrap | 未运行 |
| k、≥2000次帧级bootstrap、逐组剔除 | 未运行 |
| 三类灯箱组内/组间离散 | 未运行 |
| 21场景×auto/O1/O2×区域的误差账本 | 未运行 |
| 单场景/全量的x与账本数值完全一致 | 尚无法验证，未产生这些数值 |
| c_lab对c_auto散点与完整账本热图 | 无合法数据，未伪造；下图仅显示停止状态 |

![阶段停止状态（并非误差账本）](figures/t04/stage_status.png)

结果JSON保留frames/k/lightbox/ledger空对象、verdict=null，metadata.measurement_complete=false。没有将前置失败伪装为四条科学结论中的“数据不足以区分”；四条规则只适用于完成后的k置信区间。

## 工具与合成测试

`tools/eval/lab_timing.py` **当前仅交付前置闭合入口与测量单位核对原语，并非已完成的全量测量程序**。支持以下单场景/全量前置命令，结构一致；任一闭合失败写停止结果并退出码2。若所有闭合通过仍会标记measurement_pending（退出码3），不声称已完成后续尚未实现的测量。

```bash
FILMLAB_DATA=… python tools/eval/lab_timing.py R1-00-1 output.json
FILMLAB_DATA=… python tools/eval/lab_timing.py output.json
```

合成测试从oracle_timing.job提取并执行其原始嵌套fit() AST，使用已知x的可逆合成响应，恢复误差<1e-6，再核对 `c_lab=c_auto+(xR,xB)` 和 `K_lab=K_auto+xK`。另测单位不一致停止、严格闭合容差及超过3帧触边的规则。没有在真实数据上使用这些拟合原语，没有修改O1定义或边界。

全量测试：**348 passed、3条已有colour警告**，退出码0。完整输出：`tools/eval/t04_test_result.json`。`filmlab/` 相对a43c2af逐字节不变，tree为 `081dac42993d4f9b8b4780d1d643421ad96c6632`；`t03_baseline.json` 同样未改。未运行或导入accept_t04.py。

## 恢复所需证据

请验收方提供与已提交基准对应的R1-00-1固定几何缓存/指纹、运行时和精确复现命令，以查明差异。不得靠选择更接近目标的缓存、放宽1e-6或重写基准继续。闭合问题解决并确认恢复后，才继续实现和执行O1/O2、bootstrap、账本及k结论。
