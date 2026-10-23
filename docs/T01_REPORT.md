# T01 胶片参数校准

- 基线：`handoff/recovered-research-2026-10-23` / `fa68cb1`。
- 搜索状态：`stopped_boundary`；共记录 83 个候选（含物理约束拒绝项），完整训练评估 33 个。
- 只使用 `accept_calibration.TRAIN` 的固定 11 场景；没有读取留出图像、缓存或结果。
- 注入直接调用 `accept_calibration.apply()`；每场景直接调用 `param_sensitivity.evaluate_scene(sid, roll=None)`，新进程隔离补丁，O1、ΔE00 中位再取场景平均。
- 固定种子 0；粗搜索 3×5×3 个内部点，再在原始范围内 Powell 细化；总候选不超过 300，物理不合格候选记为拒绝，不计算训练误差；改进候选贴边时立即停止。
- 训练文件由 `epfl_800t_manifest.csv` 指定，22/22 文件 SHA256 校验通过。
- 本机训练基准均值：4.845756048003865（规格记录 4.879；环境/重算配准差异应独立复核）。
- 候选参数：`{"tau": 1.1618891807415646, "dir": 1.0235444182395073, "diffuse_fraction": 0.45394668524892745}`。
- 训练平均 ΔE00：4.763226；相对本机基准改善：0.082530。
- 是否贴边：True（任一参数距边界 ≤ 范围 2%）；没有放宽范围或更换拆分。
- DIR 分色/中性 gamma 比：`{"R": 1.292601316513579, "G": 1.6194593779415538, "B": 1.2914128206753455}`；物理检查通过：True。
- 中性 gamma：[0.6261446193958429, 0.5263022761939412, 0.6805965016209095]；分色 gamma：[0.8093553593589604, 0.8523251568142639, 0.8789310479000311]。
- DIR 定义：C-41 实际化学 `run()` 生成的染料密度，对库存 logH 全轴 4097 点求最大斜率；中性时三层同步曝光，分色时仅目标层曝光、其余固定库存曝光轴最低值。专利范围来自规格/现有 `_dir_matrix` 注释；此定义在规格中未细化，供 Claude 独立审查。
- `next_cal` 为显式可选代际，当前 enabled=False；只有搜索正常完成、物理合格、未贴边且训练改善时启用。其存在不表示独立验收已通过。
- `next_cal` 直接复用同一个 `apply()`；退出该代际恢复原函数并清缓存。`current/next/legacy` 默认值与成像算法保持，合成输入验证逐位恢复；真实 42/42 回归留给验收方。
- 全部测试退出码：0；完整测试摘要见 `tools/eval/t01_test_result.json`。
- 没有新增 RGB 调色、扫描拟合或额外物理参数。未自行运行留出验收。

## 复现与独立验收

拟合：

```bash
FILMLAB_DATA=… OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/eval/calibrate_film.py
```

拟合日志在结果 JSON 中，重跑按参数和已完成场景恢复；已停止的边界搜索不会自动重启。

由 Claude 独立执行（仅当上面确有候选）：

```bash
FILMLAB_DATA=… python tools/eval/accept_calibration.py tools/eval/t01_calibration_params.json 输出.json
```

请独立检查 DIR 定义、灰卡、ECN-2 中性密度、留出改善与分组退步、三代际严格回归，并制作目视对比图。
