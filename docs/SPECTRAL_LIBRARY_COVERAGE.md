# 十源谱库的覆盖范围审查

## 研究决策

**原十谱库拟合失败不能代表常见实际灯谱均失败。** 在本次固定FilmLab名义实例内，
原库虽已占满六维线性/仿射空间，非负混合范围仍没有覆盖多种已公开的标准参考谱与
命名灯谱。下一步可以使用预声明的完整公开集合进行条件性验证，不能把十库边界当成
胶片后端的根本限制，也不能据此宣布已识别实际灯谱。

这项审查不读取RAW、真实扫描或最终RGB，不按照片分数选择新谱，不拟合相机SSF、
材料系数、显影或扫描。扩展集合与几何边界只保存于仓库外，生产管线不变。

## 预声明资料全集与来源层级

使用本环境`colour 0.4.7`自带的全部59项`SDS_ILLUMINANTS`、56项`SDS_LIGHT_SOURCES`，
以及项目已经使用的2400/2700/3000/3300K四个Planck黑体。纳入类别取决于预定集合，
不取决于走廊颜色、拟合残差或光晕外观。

只保留原始覆盖完整380–780nm的数据。七项ISO 7589数据仅到690nm，其中打印机谱
仅到560nm，全部列为excluded；没有补零、延长端点或假装已测到780nm。
完整保留112条，含110种不同归一形状；两组重复为C/SC、Incandescent/60 A/W。
项目原有31个成员全部已涵盖，不添加任意单波长delta或温度扫描。

| 来源层级 | 纳入条目 | 能说明什么 |
| --- | ---: | --- |
| CIE等标准参考分布 | 52 | 包括标准荧光灯/LED、高压放电，以及A/B/C/D/E/ID参考条件；不能全称为某款灯具实测 |
| colour命名光源表 | 56 | RIT Pointer数据、NIST CQS传统灯/LED/Philips灯表，以及Kinoton资料；是公开灯谱资料，不是现场灯具测量 |
| 项目理论黑体 | 4 | 已存在的Planck模型，仅是条件光源，不是新材料测量 |

例如E是等能参考谱、D系列是日光模型，四个黑体也是理论模型；与命名汞灯、荧光灯
不能等量看待。NIST CQS等库中还有LED组合/模拟资料，不能因有产品名称便认定每条
均是具有独立不确定度说明的完整实测报告。

本地资料来源说明来自`colour/colorimetry/datasets/light_sources/sds.py`及
`.../illuminants/sds.py`，其引用包括：

- [NIST CQS simulation 7.4](https://drive.google.com/file/d/1PsuU6QjUJjCX6tQyCud6ul2Tbs8rYWW9/view)
- [RIT PointerData](http://www.cis.rit.edu/research/mcsl2/online/PointerData.xls)
- CIE 15:2004及[CIE 015:2018](https://doi.org/10.25039/TR.015.2018)

本轮没有访问网络补材料，直接审查已安装公开数组。

## 固定六维响应与归一方式

所有输入用保非负的线性插值到380–780nm、5nm网格；不外推。每条谱除以原有
`lights.camera_rgb(SPD) @ [.2126,.7152,.0722]`，使项目输入Y=1，维持原
`WB_REFERENCE=D55`。此Y为项目理想CIE相机代理下的数值定义，不是实际传感器或
绝对辐射标定。原十谱线性重采样与此前81-bin版本最大差异小于5e−15。

固定`next_sensor`、`cinestill_800t`配方与当前`incident.spectral_responses(engine)`，
同一谱同时计算direct三层、return三层，并保留原配方`2**exposure_ev`公共因子。
六个坐标依次为`direct_R/G/B, return_R/G/B`，其R/G/B指模型三感光层，不是扫描RGB。
没有改变两条路径的相对幅度。

**这些数值属于名义Engine实例，不是实际走廊实例。** 初始化顺序为：先在初始
`WB_REFERENCE=D55`下归一化谱库，再调用`generation.use('next_sensor')`（切换为D65），
随后构造默认`roll=0/frame=0`的Engine并积分六维响应。JSON原先的engine WB字段
记录的是切换后的D65，不能据此误认谱库曾按D65归一；现已分别记录归一与Engine状态。
原`normalized_Y_max_error`也在切换后计算，误称为归一误差；其3.215%数值现单列为
`post_generation_WB_Y_deviation`，真实D55归一检查单独记录。没有重算或替换名义LP结果。

实际走廊使用`engine_for()`的`roll=1/frame=15`及D65，含该实例的材料变化，不能直接
拿本报告`response6`作闭合或拟合。主线只复用112条**预声明谱形**，按实际WB重新
归一化、按实际Engine重新积分direct/return；重新积分后原十谱闭合误差3.55e−15。
场景专用库另存，不覆写这里的名义结果。本报告以下秩、凸包、比值和覆盖计数均限定
于该名义实例，不应标为实际两灯的响应边界。

两种范围分别检查：

- 凸包：系数非负且和为1，固定输入Y。
- 非负锥：同一六维系数向量非负，允许整体输入Y/幅度自由。

每轴除以原十库该坐标的最大响应，以免很小的return_B在数值优化中被忽略。
LP最小化六维最大绝对残差；归一坐标容差为1e−7，并独立检查实际重建残差。
这些距离是当前模型中的数值距离，不是ΔE，也不是每个目标坐标的相对百分比。

## 秩、凸包与非负锥

| 集合 | 线性秩 | 仿射秩 | 六维凸包顶点 |
| --- | ---: | ---: | ---: |
| 原十库 | 6 | 6 | 10 |
| 全集 | 6 | 6 | 66 |

满秩只说明允许正负系数时能够张成空间，不能保证非负光谱混合覆盖相同范围。
在相同六坐标尺度下，扩展凸包体积约为原库409.5倍；这不是现实灯具的发生概率、
物理光谱空间覆盖比例或成像质量提升倍数。

| 来源类别 | 数量 | 原固定Y凸包外 | 原自由幅度非负锥外 |
| --- | ---: | ---: | ---: |
| 标准参考分布 | 52 | 45 | 43 |
| 命名灯表 | 56 | 55 | 54 |
| 项目黑体 | 4 | 2 | 2 |
| 合计 | 112 | 102 | 99 |

这是严格有限库覆盖计数，不能将“外部”全部读作视觉上重要。按原坐标最大值缩放后，
命名灯表的锥外距离中位数约0.0353；LED-B2仅0.00577，3000K黑体仅0.00220，
而LED-RGB1为0.242、Mercury为0.171、HP2为0.117。原库失败至少可能有有限源谱
选择限制，但扩展库能否改善实际区域仍需独立验收。

## 不依赖LP的比值边界

对于非负混合且分母严格正的响应，比值是各成员比值的加权平均。因此任何非负系数
与任意总幅度都不能越过原成员的最小/最大比值。这是比满秩更直接的排除证据。

| 预声明诊断比值 | 原十库范围 | 全集范围 | 全集端点来源 |
| --- | --- | --- | --- |
| return_R / return_G | 3.199–12.539 | 0.562–15.651 | H38HT-100 Mercury / Neodimium Incandescent |
| direct_R / return_R | 3.557–4.544 | 3.504–5.936 | Neodimium Incandescent / LPS |
| direct_G / return_G | 24.306–40.857 | 12.820–43.130 | LPS / LED-RGB1标准参考 |
| direct_B / return_B | 32748–115446 | 35.49–257399 | LPS / Mercury |
| 三层direct总和 / return总和 | 7.581–27.142 | 6.556–51.443 | 2400K理论黑体 / H38HT-100 Mercury |

这些是固定模型曝光单位中的响应比，不是能量反射率、返回光子占比或真实800T材料实测。
尤其return_B很小，其比值会很大，不能据数值大小直接推断视觉重要性。

return_R/G超出原范围的**命名灯表**为：

| 公开灯表 | return_R/G |
| --- | ---: |
| Cool White FL | 3.087 |
| Daylight FL | 2.937 |
| Mercury | 2.322 |
| Neodimium Incandescent | 15.651 |
| F34/CW/RS/EW (Cool White FL) | 2.820 |
| F34T12/LW/RS/EW | 2.355 |
| F40/DX (Broadband FL) | 3.032 |
| F40/DXTP (Delux FL) | 2.627 |
| H38HT-100 (Mercury) | 0.562 |
| H38JA-100/DX (Mercury DX) | 2.920 |

另外FL1/FL5/FL6/FL3.3/FL3.11/LED-RGB1六个标准参考及项目2400K黑体也在外。
这说明新增合理光源确能改变原十库无法达到的三层返回比例；不能因此认为现场灯具
就是某个汞灯或把其中一条谱提前指定为修正。

## 可复用产物与复现

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/filmlab-env/bin/python tools/eval/spectral_library_coverage.py
/workspace/filmlab-env/bin/python -m pytest tests/test_spectral_library_coverage.py -q
```

仓库外`/workspace/filmlab-audit/spectral-library-coverage/`：

- `normalized_spectra.npz`：`names(112)`、`categories`、`spectra(112,81)`、
  `wavelengths_nm`、`response6(112,6)`、`original10_indices`、`coordinate_scale`、
  两条逐波长响应及新旧凸包的归一坐标半空间方程，均属roll0/frame0名义实例。
  实际场景只复用谱形，再以该场景WB和Engine重算，不能直接复用这里的`response6`。
- `results.json`：每条资料覆盖/来源/摘要、LP距离与混合系数、秩、体积、全部比值
  外部成员清单，以及七项排除记录。类别与原库标识不按照片结果筛选。

两项测试分别检查固定Y凸包不能混同自由幅度锥，以及满线性秩不能表示边界外的
非负比值。原十谱精确复现、Y归一与有限/非负检查也在工具执行中完成。
生产代码、正式默认和next未修改。
