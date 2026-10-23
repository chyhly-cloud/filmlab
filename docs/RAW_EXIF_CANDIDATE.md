# 明确启用的曝光元数据补齐候选

## 输入错误与修正范围

已有G81 RAW有快门和光圈，却缺少既有解析器支持的ISO字段；同文件内嵌JPEG记录ISO200。
原`_exif`在发现ExposureTime后直接返回，因而未查缺失ISO，令亮度先验退到固定档数分支。
实际成像影响与逆拟合证据见[源端反演](INVERSE_SOURCE_RADIANCE.md)。

本修正只补已记录的曝光元数据，不推断ISO、不估计灯芯辐射，不修改工程亮度先验、MAX_STOPS、
随机种子、测光、扫描、显影或光晕核。默认关闭，current/next/next_sensor均不自动启用。

## 接口

```python
rgb = rawio.load_raw(path, complete_exif=True)
exposure = rawio.exif_exposure(path, complete_exif=True)  # t, N, ISO
iso = rawio.exif_iso(path, complete_exif=True)
nominal_clip = rawio.clip_luminance(path, complete_exif=True)

trace = {}
tags = rawio._exif_complete(path, trace=trace)
```

已有调用不传参数时保留原读取路径；也不向原有单参数测试hook传新kwargs。
`trace`记录各字段来自RAW或embedded JPEG、原分数字符串及冲突/失败原因。
合并保留原Tag对象，不用浮点重写EXIF。

候选先检查RAW曝光字段：完整则直接使用，不读取JPEG或覆盖RAW。仅当缺失字段时读取同文件内嵌JPEG。
共有快门/光圈须在相对1e-4内一致，以容纳JPEG十进制舍入；共有ISO须精确一致。
共有相机厂商/型号/原始拍摄时间也须一致。冲突时保留原RAW、不借用缺失字段；
预览读取失败或不是JPEG时同样保留已有字段。无法完成的字段继续为未知。

## 已完成验证

- 十项单元测试：默认早返回、单参数hook兼容、缺ISO补齐、原Tag/分数/来源保留、快门/光圈/相机冲突拒绝、
  ISO冲突拒绝、读取失败保留字段、完整RAW优先、无效ISO别名不抢占有效字段、JPEG舍入容差。
- 两实际RAW（G81 P1480512、用户Sony DSC03906）在current、next、next_sensor的修改前/后默认解码，
  六组输出均逐位一致。测试使用固定原路径、seed=0、max_size=1400，`HIGHLIGHT_RECON="prior"`。
- G81候选得到 `(0.5, 5.6, 200)` 与名义Lclip=37.632，输出有限且与此前独立Lclip覆盖实验逐位一致。
- Sony原记录完整，得到 `(1/30, 2.8, 1600)` 与名义Lclip≈17.64；候选解码与默认逐位一致。

新增十项测试连同已有实测通道与高光追踪测试，共19项通过。
实际回归记录、原文件快照和候选数组在仓库外`/workspace/filmlab-audit/exif-candidate/`：
`before.json`、`verification.json`、`capture_before.py`、`verify_after.py`。

## 证据边界

名义截断亮度公式依赖ISO定义及镜头透过率代理，不是实际灯芯辐射标定。
候选使元数据读取更完整，但其后的随机亮度分布仍未经本灯实测。
后续[真实失测区域验收](EXIF_PRIOR_HOLDOUT.md)发现，补齐ISO后既有工程先验在两个透光孔
明显过亮，且所有抽样触发MAX_STOPS上限；保留读取能力，不推广这条先验输出。
既有全幅元数据完成对照追回灯旁橙红，却使外缘退步；不因输入记录修正成立就认定光晕整体验收通过。
本候选只新增一个生产模块的显式选项，默认逐位回归通过；不声称生产文件摘要仍全部未变。
