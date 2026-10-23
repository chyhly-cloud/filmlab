# FilmLab → Claude Code：恢复后的完整交接

日期：2026-10-23。用户要求将全部研究交给 Claude Code。**旧工作区现已恢复**；此前聊天里“无法写入、只有历史记录”的说明保留为故障历史，不再代表文件取回状态。

## 1. 本次已实际核查的内容

- 仓库 `/workspace/filmlab`，恢复时 HEAD 为 `1fbac0d78c08cad0e7d389651bb5d6271cfde3cd`，分支 `film-physics-decoupling`。
- 找到15个已修改跟踪文件和242个未跟踪研究文件。这解释了为什么另一台机器仅克隆仓库看不到后续工作：此前没有提交推送，而不是研究文件已经丢失。
- `/workspace/filmlab-audit` 约9.7GB、`/workspace/filmlab-data` 约3.3GB、`/workspace/attachments` 约53MB均存在。是否可从Claude所在主机访问仍取决于实际数据迁移；Git不会同步这些目录。
- 本次对厨房TIFF、LICENSE和2026研究提要PDF重新计算SHA，均与历史记录一致。
- 本次对晚期冻结快照中59个生产文件重新计算SHA，全部一致。**这是相对该快照，不是声称源文件与1fbac0d相同**；下面列出的opt-in接口/诊断功能确有代码改动。
- 本次完整测试运行：`327 passed, 3 warnings in 30.99s`。警告为colour库的已有光谱对齐提示。此运行发生在恢复后的研究代码上，随后只引入Claude的regress.py修复及交接文档。
- 已从远端明确取得并检查 `a53e6d4fca3b12eb0bfafb694ffa72a51189bc99`：只修改regress.py。新交接分支以它为父基线，保护Claude的严格回归修复。

交接分支：`handoff/recovered-research-2026-10-23`。以实际远端和提交记录确认已推送，不把本地创建分支当作远端交付。

## 2. Claude首次接手：保留现有修复，先读文档

```bash
git status --short
git fetch origin
git log -3 --oneline origin/handoff/recovered-research-2026-10-23
git diff --stat a53e6d4..origin/handoff/recovered-research-2026-10-23
```

先保存自己的工作区，再合并/评审该分支；不要覆盖现有改动。分支已包含a53e6d4，无需把它回滚成旧脚本。若自己又有新提交，按通常Git合并评审，不强制重置。

阅读顺序：

1. 本文。
2. [完整按时间追加的研究交接](HANDOFF.md)，**尤其后半部的最新记录**。
3. [架构](ARCHITECTURE.md)与[实验准则/冻结记录](../tools/eval/CRITERIA.md)。
4. 本文末尾的完整报告索引及对应eval代码/机器摘要。

之前聊天中的长交接主要由会话恢复，遗漏了部分晚期结果。现在代码和详细记录可查，不能继续只依赖那份聊天摘要。以下给出两个重要更正：

- “局部a*方向相反”是**旧完整模型**的外部失败特征，不代表所有后续源反演候选都反号。独立天花板几何修正后旧模型仍为负，但后续112谱候选已为正、幅度不足且未晋升，见[局部配准](LOCAL_CEILING_REGISTRATION.md)及[几何后复拟合](GEOMETRY_SOURCE_REFIT.md)。
- 旧`next_halo`的0.1/0.3倍率证据因`halo_edge.py`重复解码已撤销；数值仅留历史复现，不是实测光晕强度。

## 3. 用户的持续决定

- `current`和`next`保持正式基线；`next_sensor`保留候选，尚未批准并入next。
- `paper_only`只作实验肩部候选，不推广，不重新拟合SHOULDER_WHITE、扫描肩部或颜色。
- 保留已知SPD/强度直接参与曝光的能力，但只有有依据的显式输入才启用，不能用亮像素分类代替测量。
- 保留逐区域验收的短曝光测量替换；未恢复灯芯不能标为实测。
- 不调整DIR、不加快慢子层、不接入新输运器。对同组最终JPEG的冻结参数搜索不重启。
- 原理反例、合成一致性、库存感光度下的条件参考、真实800T测量分开记录；不能通过增添模型复杂度填补不可辨识性。
- 用户曾授权数学反演和物理/留出核验；其产物仍是条件witness，不能改称实测SPD/亮度。后续明确冻结决定有效。
- **14帧方案只作备用，不执行、不扩拍摄帧数。优先现成真实负片资料。**
- 自主推进有判别目标的工作，减少碎片汇报。不要把每个常规步骤交给用户批准。

## 4. 恢复代码包含什么，哪些结论仍未被证明

|领域|保留内容与边界|
|---|---|
|高光与扫描|额外指数肩部、相纸和量化的分离审查；`paper_only`可选，真实验收有改善也有退步，不能认定正确扫描曲线|
|显式输入|`filmlab/incident.py`、`brackets.py`、`develop_negative(..., incident_sources=..., trace=...)`等；空输入保持原路径，部分实测覆盖不代表主要灯芯恢复|
|RAW/EXIF|记录完整性和显式解码候选；完整EXIF用于旧亮度prior会出现退步/触顶，不可因解析更完整就自动采用先验|
|光学传播|假想材料光子输运与角度/数值合同；不是实际800T的mu_a/mu_s/g测量|
|非中性显影|297探针与实际灯旁有限响应；其他染料对黄色方向抵消仅约4–5%，不支持DIR为主要根因|
|基底/可达性|有限搜索不能叫严格不可达；负曝光/全局蓝层补偿未采用，跨场景残差方向不同|
|源反演/几何|多种谱库、删失/PSF、径向源、留出与独立暗格线验证；部分指标改善，不等于完整橙红光晕已修好|
|全图颜色|五种固定扫描策略存在跨场景取舍，草地总ΔE好不等于色相好，不推广昼夜经验切换|
|曝光下限|已知谱色卡7/48旧映射非正曝光；平滑/非负谱不能通用替换，已知源约束也仍有正常样本退步|
|颗粒|扫描纹理和孔径统计审查；不把扫描噪声、锐化、采样全部当乳剂颗粒|

请按相应报告核对样本、单位、训练/留出和误差定义。CIELAB b*不是红色强度，分层曝光和不等于光子能量；局部导数不能直接外推完整返光。

研究配置也不能混用：旧三列真实/`exp_plain`/`paper_plain`采用`next_sensor`、film_metering、frame定光、无输出调整、grain=0。CLI默认current含auto_expose/gray定光/lab输出及颗粒，不是该研究候选图的复现配置。

## 5. 数据与复现

环境历史入口：`/workspace/filmlab-env/bin/python`，评估应设`FILMLAB_DATA=/workspace/filmlab-data`。默认使用单线程BLAS可减少资源占用。

```bash
cd /workspace/filmlab
source /workspace/filmlab-env/bin/activate
export FILMLAB_DATA=/workspace/filmlab-data
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m pytest -q
```

Claude那边已有101张rawcache、SillyStill、对齐缓存和两份42键基准，这是用户转来的Claude复核结果，不代表本机也有这些数据。本机原始EPFL和后续缓存另有清单。不要将另一环境缺缓存写成整个项目缺原始数据。

`regress.py`请沿用a53e6d4，明确`REPO`和`FILMLAB_DATA`。本次327测试不是当前/legacy各42项逐位回归，不能混为一谈。Claude此前42/42及有限值验证按其记录保留；合并后可利用其完整夹具再验证。

本机回归所需旧rawcache若缺失，不生成零项夹具、不用无关RAW替代。`baseline-legacy`不一定带regress.py，应使用明确脚本和REPO，不随意切换有改动的工作区。

原始EPFL入口：`tools/eval/epfl_800t_manifest.csv`和`download_epfl.py`；film目录的36张TIFF是RGB8正像，不是原始负片。

完整外部文件清单（路径/字节数，非自动许可转移）：[数据目录清单](../tools/eval/handoff_data_inventory.json)。大缓存、原图不进Git；Git源代码移交和外部数据搬运分别确认。绝对路径不是Claude另一台主机必然可读的路径。

## 6. 上传附件与许可证

### 实际取得的厨房负片

`SAMPLE-35mm-scan-primefilm-xa-cinestill800t-Analog.Cafe.tif`，21082840字节。

SHA256 `52d4045a89d4324b984ae8a92ab9405e99c57ce1c4e9e224a1f6e2835dd27129`。

LICENSE SHA256 `778cc0dc09ea509229ebe987d3ea6a3a3fbbaa0bedbf52bbbd5257d6c4e08ce9`。

单页RGB16、2000×1332、LZW、设备标签PrimeFilm XAs、Photoshop保存、非线性sRGB ICC。红通道.273386%像素在代码上限，不自动当作物理饱和。300DPI标签是文档分辨率，不是片面标定。厨房静物没有适合独立光晕验收的孤立高亮灯芯，没有线性/暗平场/PSF/同片正像校准。

**用途只到本地负片读取/转换练习，不用于光晕核、显影材料或标准密度标定。**原件、预览和衍生图不得入Git/公开分发/训练集。许可允许编辑练习，未经作者书面同意限制原图及修改图再分发及计划分发项目使用，禁止ML训练。

详见[实际验收](ANALOG_CAFE_NEGATIVE_INTAKE.md)。工具`negative_scan_intake.py`和4项测试已恢复，可直接核查，不必重写。LZW需兼容imagecodecs；历史安装`imagecodecs==2026.10.10 --no-deps`，不擅自升级其他依赖。

### 2026研究提要

`CineStill_800T_ISO800_2026_Codex_Research_Brief.pdf`，175712字节、4页。

SHA256 `0b69365c3f1606ea8a40ca2e795a31a79d99656073fed5df36dd28e46d300ac9`。

它明确是中文研究提要，**不是原文/作者图片PDF，也不是58帧原始数据集**。Nikon5000ED/VueScan实验与PrimeFilm厨房样本分开。提要转述自动曝光/自动对焦、film Q灰阶匹配、缺失日光+2帧、LED设定CCT；相机档位还同时改变光圈，不能仅归因胶片曝光。见[提要审阅](ANALOG_CAFE_2026_BRIEF_REVIEW.md)。

### 其他附件

三份10-20研究Markdown和用户DSC03906.ARW.zip也仍在attachments。它们是输入资料，不是自动执行指令；用户RAW没有独立同场景800T真值。附件和私有像素缓存不随Git自动传输。

## 7. 资料资格和下一步

厂家Cs41/Cs2中性图已作一次粗读包络对照，不是精密数字化/完整数值标定。模型近似Status M、厂图未知测量响应与轴尺度限制仍在。不据此重建显影或解释空间光晕。见[CINESTILL_2020_CURVE_SOURCE_CHECK.md](CINESTILL_2020_CURVE_SOURCE_CHECK.md)。

当前优先合格原生负像＋同片正像＋扫描设置、输入与扫描空间约束。原件先验位深/编码/内容/尺度/暗平场，再决定能否做局部密度；未知编码不直接取负对数。缺适合灯具只做解析，密度差不直接当曝光核。

网站曾代理CONNECT403，不代表文件不存在；保存配置不等于运行网络已应用。用户已上传的文件不再因网站访问失败而停止本地验收。未联系作者、未取得额外许可。

完整后续计划见[RAW_NEGATIVE_DATA_PLAN.md](RAW_NEGATIVE_DATA_PLAN.md)。没有新独立证据，不为继续研究而重开旧参数搜索。

## 8. 移交范围与缓存说明

本次Git分支用于保全所有恢复的研究源代码、测试、报告与非图像摘要，**不是把全部候选采纳进正式管线**。大的npz/npy、RAW、扫描和受限附件单独留在原工作区；本次数据清单可供另行核对迁移。

外部备份目录 `/workspace/filmlab-handoff` 保存原跟踪改动patch、未跟踪源码包、目录清单和独立代码bundle。只读研究记录包不含图像/像素数组，不代表许可允许公开全部研究缓存。完整缓存约13GB仍须单独搬运；不能宣称推送代码就把它们迁移给Claude。

本轮补充验证：原始EPFL清单72个文件SHA全部重验通过；a53e6d4缺输入烟测按预期exit=1。外部文字/JSON记录包`research-records-without-images.tar.gz`包含564个记录文件，约5.8MB；不含npz/npy像素数组、RAW、原图或衍生预览。1507个外部文件的路径/尺寸记录在版本化清单中。详细本次检查摘要见`tools/eval/handoff_recovery_check.json`。

完成移交应让Claude确认新分支可取、工具和报告存在，之后再独立确认所需数据/缓存。不要删除原工作区或用重新克隆代替备份。

## 9. 给Claude的接手指令

> 先阅读本文件、HANDOFF.md和CRITERIA.md，核对新分支及数据清单。保留a53e6d4回归修复，current/next不变。研究文件已实际取回，不再只靠聊天数字，也不必重写已恢复的验收工具。按详细报告较晚记录判断实验状态，注意旧a*反号不代表所有后续候选。厨房TIFF仅本地受限练习，2026实验仅有二手提要；14帧仍备用。新数据先验收独立判别能力，再决定候选。不要把文件恢复、测试通过或历史拟合改善说成真实800T光晕已修复。

## 10. 完整报告索引

以下索引由本次实际文件生成，详见列表；HANDOFF.md保存跨报告的按时间研究记录。

- [Analog.Cafe 2026 ISO800对比：离线研究提要审阅](ANALOG_CAFE_2026_BRIEF_REVIEW.md)
- [Analog.Cafe实际800T样本验收](ANALOG_CAFE_NEGATIVE_INTAKE.md)
- [角度、吸收位置与分层返光比例（2026-10-20）](ANGULAR_COLOUR_AUDIT.md)
- [FilmLab 成像链架构 (实验分支 `film-physics-decoupling`)](ARCHITECTURE.md)
- [独立参照下的并行研究](AUTONOMOUS_RESEARCH_ROUND.md)
- [正常区域的跨场景基底偏差与可达性收敛（2026-10-20）](BASE_COLOUR_SYSTEMATIC_AUDIT.md)
- [红色响应在哪个亮度区间丢失（2026-10-20）](BRIGHTNESS_RED_RESPONSE_AUDIT.md)
- [RAW自身能否约束同灯源色](CAMERA_SOURCE_CONSTRAINTS.md)
- [112谱库候选：两灯全幅联合重放](CATALOGUE_JOINT_RENDER.md)
- [标准谱库、近外联合验收与六响应放宽](CATALOGUE_SOURCE_INVERSE.md)
- [固定总输入的逐单元空间反演与象限对调](CENSORED_SOURCE_FIELD.md)
- [逐单元失测源场：独立审查](CENSORED_SOURCE_FIELD_REVIEW.md)
- [CineStill Cs41/Cs2 曲线：图面核查与一次有限中性对照](CINESTILL_2020_CURVE_SOURCE_CHECK.md)
- [基础成像链的受控约束核查](COLOUR_FOUNDATION_CONTRACTS.md)
- [固定负片的五场景色彩策略实验](COLOUR_POLICY_TRIAL.md)
- [走廊实际明亮外围：有限染料抵消与扫描压缩（2026-10-20）](CORRIDOR_DYE_RESPONSE.md)
- [Corridor small-light development audit](CORRIDOR_LIGHT_DEVELOPMENT_AUDIT.md)
- [走廊光晕的RAW源信息依赖（2026-10-20）](CORRIDOR_SOURCE_DEPENDENCY.md)
- [日间/蓝调整体色彩：逐帧灰平衡的有界核查](DAY_NIGHT_COLOUR_BALANCE_AUDIT.md)
- [同总吸收、不同深度：三记录充分性的最小检验](DEPTH_SUFFICIENCY_AUDIT.md)
- [DIR 自洽闭合候选（2026-10-20）](DIR_CONSISTENCY_CANDIDATE.md)
- [5.61与3.46的单位、有限变化及染料分配边界（2026-10-20）](DYE_CONTRIBUTION_UNITS.md)
- [EPFL 高光验证：预先固定的方法与证据边界](EPFL_HIGHLIGHT_VALIDATION.md)
- [完整EXIF候选的真实失测区域验收](EXIF_PRIOR_HOLDOUT.md)
- [输入曝光下限：受控光谱、真实输入与可辨识范围](EXPOSURE_FLOOR_DOMAIN_AUDIT.md)
- [独立几何后的源输入复核（2026-10-21）](GEOMETRY_SOURCE_REFIT.md)
- [天花板几何控制后的源谱重拟合：独立审查与幅度一致性](GEOMETRY_SOURCE_REFIT_REVIEW.md)
- [颗粒孔径平均的尺度一致性核查](GRAIN_APERTURE_SCALE_AUDIT.md)
- [草地基础色差局部诊断（2026-10-21）](GRASS_COLOUR_DIAGNOSTIC.md)
- [光源到负片的因果链审计（2026-10-20）](HALATION_CHAIN_AUDIT.md)
- [光晕目标：当前阻碍与停止条件](HALATION_GOAL_STATUS.md)
- [光晕机制的单因素对照（2026-10-20）](HALO_MECHANISM_EXPERIMENTS.md)
- [固定后端可达性与光谱—空间可分离性（2026-10-20）](HALO_REACHABILITY_AND_JOINT_TRANSPORT.md)
- [两处真实顶灯：无光晕基底与完整返光的残差方向（2026-10-20）](HALO_RESIDUAL_DIRECTION.md)
- [FilmLab CineStill 800T 研究交接 (2026-10-19)](HANDOFF.md)
- [高光信息与扫描影调审计](HIGHLIGHT_TONE_AUDIT.md)
- [历史乳剂原始文献与DLF来源核查（2026-10-20）](HISTORICAL_EMULSION_SOURCE_AUDIT.md)
- [照明推断与全局类别裁剪的独立复核](ILLUMINATION_INFERENCE_TRIAL.md)
- [独立测量资料核查与研究冻结](INDEPENDENT_EVIDENCE_SOURCE_AUDIT.md)
- [配准warp与物理传播坐标的尺度核查](INVERSE_GEOMETRY_SCALE_AUDIT.md)
- [未知灯芯输入的数学反演与独立验证](INVERSE_RADIANCE_RESEARCH.md)
- [源端反演的配准敏感性核查](INVERSE_REGISTRATION_AUDIT.md)
- [固定成像链的光源数学反演](INVERSE_SOURCE_RADIANCE.md)
- [固定测量约束边缘与HDR灯体后的近/外联合源逆解](JOINT_RADIUS_SOURCE_INVERSE.md)
- [走廊两灯：严格失测核心外的输入审计](LAMP_EDGE_INPUT_AUDIT.md)
- [灯具边缘的测量约束与完整成像链验证](LAMP_EDGE_SPECTRAL_CANDIDATE.md)
- [夜景光源局部验收与高光研究冻结](LIGHT_SOURCE_ACCEPTANCE.md)
- [同画幅天花板局部对照与研究分界（2026-10-20）](LOCAL_BACKGROUND_AND_RESEARCH_BOUNDARY.md)
- [局部天花板配准与灯体视差：独立审查](LOCAL_CEILING_GEOMETRY_REVIEW.md)
- [远场格线约束的近灯局部几何与光晕复验](LOCAL_CEILING_REGISTRATION.md)
- [全测灯体的浮点 RAW 颜色候选](MEASURED_BODY_DECODE_CANDIDATE.md)
- [显式入射光谱、逐灯 RAW 测量与完整成像链验证](MEASURED_SOURCE_VALIDATION.md)
- [多半径逆实验的测量孔径隔离](MEASUREMENT_APERTURE_AUDIT.md)
- [色彩域修复、已知谱合同与非中性显影链（2026-10-20）](NONNEUTRAL_CHAIN_AUDIT.md)
- [RAW + CineStill 800T paired-source recovery](PAIRED_DATA_RECOVERY.md)
- [独立分层光子输运验证（2026-10-20）](PHOTON_TRANSPORT_VALIDATION.md)
- [固定扫描密度矩阵与精确虚拟印放积分（2026-10-20）](PRINTING_DENSITY_AUDIT.md)
- [数学估计辐射的独立RAW留出验证](RADIANCE_CENSOR_HOLDOUT.md)
- [明确启用的曝光元数据补齐候选](RAW_EXIF_CANDIDATE.md)
- [RAW高光线性解码的独立数值审查](RAW_HDR_DECODE_AUDIT.md)
- [实际800T负片路线：来源验收与最小委托实验](RAW_NEGATIVE_DATA_PLAN.md)
- [青染料到扫描颜色的独立局部Jacobian（2026-10-20）](SCAN_CHROMA_JACOBIAN.md)
- [颗粒研究起点：扫描输出纹理的可辨识性](SCAN_TEXTURE_INITIAL_AUDIT.md)
- [实测通道保持：输入候选与实际退步](SENSOR_CONSTRAINED_INPUT_CANDIDATE.md)
- [指数肩部的依据、渐近值与独立消融](SHOULDER_ABLATION.md)
- [未知灯芯空间分布的条件边界](SOURCE_GEOMETRY_BOUNDS.md)
- [实际全幅高光重建的空间形状：短曝光留出核查](SOURCE_PROFILE_HOLDOUT.md)
- [失测灯芯空间分布：有限候选与半径一致性检查](SOURCE_SPATIAL_INVERSE.md)
- [十源谱库的覆盖范围审查](SPECTRAL_LIBRARY_COVERAGE.md)
- [非中性局部传递矩阵与新增结构证据（2026-10-20）](TRANSFER_MATRIX_AUDIT.md)
- [现行光晕核的数值域与离散化审查](TRANSPORT_NUMERICAL_CONTRACT_AUDIT.md)
- [DSC03906：规整红圈的实际链路追踪](USER_RAW_HALATION_FAILURE.md)
