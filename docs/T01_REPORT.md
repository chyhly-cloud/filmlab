# T01 胶片参数校准：运行中

基于 `fa68cb1`，分支 `codex/t01-film-calibration`。这是一份进行中的交付，尚无最终拟合或验收结论。

- 训练所需 22 个 RAW/扫描文件已按清单 SHA256 验证；只下载固定 11 个 TRAIN 场景，没有读取留出图像或缓存。
- `calibrate_film.py` 直接调用验收的 `apply()` 与敏感度程序的 `evaluate_scene(sid, roll=None)`。每个场景独立新进程，固定种子 0，仅搜索 tau、dir、diffuse_fraction。
- 参数范围保持规格，粗搜索 3×5×3 内部点，再做有界 Powell；候选总数不超过 300。全部候选与完成的场景写入可续跑 JSON。改进的拟合候选贴边时立即结束。
- DIR 检查采用 C-41 实际化学 `run()` 的染料密度曲线，在库存 logH 全轴 4097 点上取最大斜率；中性时三层同步曝光，分色时仅目标层曝光，其余在轴最低值。规格没有进一步细化 gamma 定义，需 Claude 独立审查。物理不合格候选不运行训练图像。
- `next_cal` 已实现，当前禁用；只有正常完成、DIR 合格、未贴边并有训练改善才启用。其注入也复用 `apply()`；切回其他代际恢复原函数。
- 当前完整测试：343 passed，3 个已有 colour 光谱对齐警告。测试包含注入一致、重复切代际后逐位恢复、贴边停搜、日志恢复及交付启用条件；未冒称真实 42/42 回归。

持久后台任务 `t01-film-calibration` 正在计算。`t01-film-delivery` 将在其结束后自动生成本报告最终版、写出完整结果和扁平参数 JSON、重跑全量测试，先生成本地提交，再由本任务的定时完成检查通过 GitHub 连接器上传此分支并核对远端树 SHA（本机命令行无写入凭据）。失败也会如实生成报告，校准不会自动晋升为 current/next。

完成前请勿将本文件或 `status=running` 的结果 JSON 当作拟合完成。所有留出验收由 Claude 独立执行：

```bash
FILMLAB_DATA=… python tools/eval/accept_calibration.py tools/eval/t01_calibration_params.json 输出.json
```

请审查最终状态、DIR 定义与结果，并独立运行灰卡、ECN-2 中性密度、留出验收和三代际 42/42 严格回归，制作目视对比。拟合方没有自行运行留出验收。
