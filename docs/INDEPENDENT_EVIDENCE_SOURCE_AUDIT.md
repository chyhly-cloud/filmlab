# 独立测量资料核查与研究冻结

后续更新（2026-10-21）：用户已补充厂家800T/Cs41中性曲线及Analog.Cafe真实负片/曝光实验的明确入口。前者已作限定粗读归档，见[CINESTILL_2020_CURVE_SOURCE_CHECK.md](CINESTILL_2020_CURVE_SOURCE_CHECK.md)；后者访问/验收状态与受控方案见[RAW_NEGATIVE_DATA_PLAN.md](RAW_NEGATIVE_DATA_PLAN.md)。新入口尚未取得TIFF，不改变本报告既有测量结论，但不再以“没有资料入口”概括当前状况。

日期：2026-10-20 UTC。仅核查公开来源、文件头和既有测量记录；未重新渲染、拟合参数或做最终 RGB 消融。

## 结论

**本轮未取得足以分离光源、负片与扫描响应的新独立测量，不开发新候选，结束这一阶段并冻结相关物理归因。**
这不表示这些数据不存在，也不表示原作者没有保留。能够核实的公开资料以既有正像扫描、部分 RAW 测量和其他材料/设备资料为主。
Drive、厂家、通用搜索与数据索引的部分入口在本环境被代理拦截，不能据此宣称不存在公开数据。

近顶灯的外部验收约束保留：同图参考扣除后，真实扫描的外侧减内侧 `Δa* = +6.84`，完整模型为 `−9.96`；
已有平移敏感性范围为 `−14.49…−5.35`，方向仍相反。它是**局部空间颜色走势失败**，不是测得的纯光晕核。
精确半径与数值受配准、材质/照明对照限制；中顶灯缺少内侧纯天花板支持，不提供第二条完整径向约束。
本轮没有对这些 RGB 进行新的参数消融，也不从走势反推材料系数。
完整条件见 [局部空间对照](LOCAL_BACKGROUND_AND_RESEARCH_BOUNDARY.md)。

## 1. 原项目是否公开了更原始的扫描？

### EPFL / cs-413-cinestill

核查 [主仓库](https://github.com/cs-413-cinestill/800t-emulation)，主分支固定到
`45248cf7e44892c5742e847536a855bc564ac397`；另有
`grain-detection-pycuda-test` 分支 `dad00771b0401f116fbb119084626f13742242f4`。

- 主分支 `dataset/film`、`dataset/digital` 的公开 Git 树仅有 `.gitignore`；README 链接的是已知的
  [Drive 数据集](https://drive.google.com/drive/folders/1o8s59JwpYBfTTskVF3c7DCLGVQAuGsFa)。
- 已查主页、相关目录、分支列表、最新提交页、releases 与 issues 页面，没有发现另一条高位深透射扫描或负片密度下载入口。
  检查时无 release、issue 查询为零结果；这不是所有历史提交或私人存储的穷尽调查。
- 颗粒分支的 `data/notes.txt` 写的是 “try to extract the non linearity of the tif by comparing to the raw sensor”。
  用数码相机输入和成片 TIFF 推断非线性，仍混合胶片、镜头和扫描过程，不是独立扫描仪标定。
- 对本地已恢复的 **36 张 film TIFF 全数重查文件头**：均为单页、RGB `uint8`、`BitsPerSample=(8,8,8)`，
  均带含 sRGB 标识的 ICC；保存软件为 Photoshop。第一张的设备字段是 Fujifilm eSystems / Digital Link。
  这些不是高位深负片透射文件，也没有隐藏的第二页高位深负片数据；设备标签不能证明具体 Frontier 型号及校色设置。
  Photoshop 保存记录也不能单独证明进行过何种调整。
- 本轮 Drive 重新列目录的请求在 HTTPS 代理 CONNECT 阶段返回 403。现有 36 对与六张备用 RAW 来自此前已成功恢复的数据，
  不将本轮访问失败误写成源文件消失。**是否另存了原始扫描、负片或实验记录：未知。**

这些文件直接约束的是已发布 800T 成片外观，而非负片密度或扫描映射。
已有白场/阶梯也只在最终输出域观测；缺少独立负片测量时，不能用它们隔离扫描仪响应。
即便另找到 16 位 TIFF，也必须核查它是否仍经过反相、自动定光、矩阵、曲线或局部处理，不能仅凭位深认定为透射测量。

### SillyStill

[仓库](https://github.com/mikasenghaas/sillystill)公开 head 仍为
`b1ebdb971207b403b7126b759c1d30d30954a722`，与此前核查一致；没有核实到新的原始扫描入口。
既有报告区分 41 对采集与 38 对处理数据，处理目标对胶片 Lab 亮度做直方图匹配；不能视为未校色设备响应。
已恢复咖啡馆 ARW/JPEG 的来源和限制见 [此前恢复记录](PAIRED_DATA_RECOVERY.md)。
本轮没有把报告中的 “raw dataset” 字样当作负片透射 RAW，也没有把只显示示例图当作完整公开原始数据。

## 2. HDR、灯芯与镜头眩光能否被独立约束？

| 来源 | 本轮核实的资料性质 | 能消除的歧义 | 当前不能消除的歧义 |
|---|---|---|---|
| 现有 EPFL 六张备用 RAW | 复核既有逐灯接收记录；677 个半尺寸 Bayer 单元，其中 671 是透光孔、6 是外墙灯边缘；没有恢复主要灯芯 | 仅在验收通过位置，约束相对相机通道信号，排除部分随机亮度先验 | 主要灯芯实际亮度/光谱、相机镜头眩光、胶片镜头差异、负片与扫描 |
| [Colour HDRI 示例数据](https://github.com/colour-science/colour-hdri-examples-datasets/tree/80dd236bcc42ea7b356f77556b3eb4923cd3dbc2) | 公开目录列出 `frobisher_001/IMG_2598…2602.CR2` 五个文件及一个 EXR 示例；仅读目录与项目文档，未下载/验证 RAW 本体、曝光元数据或标定 | 候选通用 RAW/HDR 算法夹具，使用前仍需核查文件与授权 | 没有已验证的同场景 800T、EPFL 灯具 SPD、绝对辐射或对应镜头眩光测量 |
| [Flare7K++](https://github.com/ykdai/Flare7K/tree/d1fb66ecb3c75fb3f4bfa715c49ef9265892d56f) | 作者说明：7000 合成图样和 962 实拍智能手机眩光图样；实拍使用暗室与镜头污染物，图样可叠到背景 | 可用于通用眩光算法研究，须遵守其设备/合成域限制；README 标明 S-Lab 非商业许可 | 不是 EPFL 相机/胶片镜头的 PSF，不是灯芯能量或 SPD 标定，也不是 800T 配对；未导入数据 |

现有逐灯覆盖、未恢复掩膜和时间/几何限制继续以
[逐区域记录](MEASURED_SOURCE_COVERAGE.json)与[验证报告](MEASURED_SOURCE_VALIDATION.md)为准。
没有重算或扩大 HDR 接收区域，也没有用透光孔的成功替代主要灯具测量。

未饱和包围曝光可以提供相对相机 RGB；它本身不能把相机镜头眩光从直接光中分离，也不能唯一恢复 SPD。
要分开这些因素，还需要对应镜头的独立点源/遮挡对照、稳定曝光序列、实际光谱或额外光谱约束。
无关场景的 HDR 或显示用 EXR 不能替代这组已知输入。

## 3. 实际 800T 材料与非中性显影测量

### 找到的“800T 数据”需要降级为 5219 代理

对 [Redlamp 的资料目录](https://github.com/pdcgomes/redlamp/tree/c5e28259818552790b950bd815c4387d7e41dee6/research/film-data)
进一步读取 `cinestill-800t.json` 和 `kodak-vision3-500t.json`，而非只看文件名或 README 宣称：

- 两个文件的 `source`、`characteristicCurves`、`spectralSensitivity`、`dyeDensity`、`granularity`、`mtf`、`interlayer`
  字段**完全相同**。逐字段比较结果和规范 JSON 摘要已保存。
- 800T 文件明确写明曲线复制自 Kodak H-1-5219，曲线处理为 ECN-2，而其标注的商品处理为 C-41。
  引用的 2026 Kodak 原 PDF 本轮未能取得；其日期、摘要与构造说明仍是第三方记录，未提升为本轮原厂验证。
- 因此它不是独立 800T/C-41 测量，更不是未显影乳剂吸收/散射或红色追加曝光实验。
  “文件名叫 800T”不能增加独立证据数量。

[iphone-to-film](https://github.com/mossein/iphone-to-film/tree/32563a3a0d7f484f2dd12518fb79490dff3a1b05)
README 也明确说明用 `sfl.KODAK_5219`、3200 K 白平衡与加强光晕表示 CineStill。
这属于建模选择，不是 800T 的材料标定。

[agx-emulsion 当前重定向到 spektrafilm](https://github.com/andreavolpato/spektrafilm/tree/3bb2c2d2801ff68b92019cf1dbcbb133d60832bc)。
本轮所查 profiles 列表包含 Kodak Vision3 500T 等，没有 CineStill 800T profile；README 说明从厂家资料、中性特性曲线及模拟耦合剂建模。
可用于理解方法，不能充当新的 800T 非中性测量。
其他搜索命中的模拟器 README 中的 “measured/physical” 宣称也未被当作实验表格；没有宣称穷尽它们的全部代码与数据。
FilmLab 自身出现在搜索结果中，亦不能作为独立外部证据。

### 厂家资料的访问边界

尝试 CineStill FAQ（两个域名）、800T 商品页、Kodak 5219 技术 PDF 与 AHU 文档，均在本环境代理 CONNECT 阶段被拒绝。
同样，Google/Bing、Zenodo 与 DataCite 本轮入口被拦截。没有收到原站内容，故不能作“厂家未发布”的确定断言。
附件和第三方关于“5219 去 remjet”、当前 AHU 构造或厂家没有感光曲线的说法仍须各自原始来源支持。
即使取得 5219 原厂曲线，其他片种/构造/ECN-2 的中性曝光也不等于实际批次 800T/C-41 的非中性响应。

感光谱是曝光—显影—密度测定的有效响应；显影后染料密度属于另一种材料状态。
两者均不能直接替代未显影乳剂的 `μa(λ), μs(λ), g(λ)`。

## 4. 确实可读的其他乳剂资料：只作算法参照

新核实了 [Rafal Lukawiecki 的 Delta 100 测试](https://github.com/RafalLukawiecki/film_tests/tree/bfc4ddf54bb770d3430120a12f0395cc00aa221e)：
CSV 有 21 行、13 列，包含相对 log 曝光与六种显影时间的两种曝光时长。
原实验说明记录了 4×5 Delta 100、绿色 Eseco SL-2 曝光、20°C DDX 1+4 与 Heiland TRD 2 绝对密度读数。
这是作者报告的实际测量，尚非本项目复测；仓库许可证为 GPLv2，未将表格或代码导入 FilmLab。

它可以作为通用 H–D 曲线分析、密度基准与曝光时长审查的候选材料，
**不能约束 800T 的彩色层间作用、染料、反向曝光或光晕传播**，本轮也未开展新的算法实验。

此前两篇历史乳剂光学论文仍未取得可核对的原表，状态不变，见
[历史资料核查](HISTORICAL_EMULSION_SOURCE_AUDIT.md)。
即便取得其 K/S 数据，也须先审查方程、测量几何与材料状态，不能直接当作 800T 微观系数。
[MCML 对照](PHOTON_TRANSPORT_VALIDATION.md)验证的是传播算法及参考/假想材料，不能填补实际材料测量。

## 5. 研究决策与解冻条件

本轮资料不足以在光源输入、胶片局部曝光/显影、扫描三个环节中唯一定位根因。
不接入新输运器、不使用负曝光补偿、不调整 DIR/肩部/颜色，不新增物理复杂度。
`current`、`next`、`next_sensor` 及 `paper_only` 的既有状态保持；已验证结果和候选不删除。

下一项测量先判断它能解除哪一种歧义，再决定实验：

| 独立数据包 | 能新增的判别能力 | 单独仍不能证明 |
|---|---|---|
| 同一负片高位深、未反相/未自动校正的扫描通道；曝光/暗场/平场、固定设置、负片及已知有色/中性透射标准，最好带空间光谱密度 | 区分负片已存在的空间/颜色差异与扫描映射造成的差异；标定需有几何和密度/透射定义 | 高位深文件本身不是绝对密度，也不能独立恢复入射光谱 |
| 对应灯具足够短的稳定 RAW 序列、逐灯曝光/配准/闪烁记录，灯具 SPD/辐射标定与对应镜头眩光对照 | 限制灯芯输入及镜头扩散，降低相机光谱歧义与随机高光先验的影响 | 不测负片仍不能唯一分开材料与扫描 |
| 明确实际 800T 批次、C-41 工艺的已知光谱/曝光矩阵：非中性及背景叠加，测负片光谱/标准密度；传播另需前/背照或点源空间测量 | 直接检查非中性显影响应；受控空间数据约束返回曝光传播 | 中性三条曲线不唯一给出深度结构、吸收/散射和量子效率 |

这些是解冻条件，不是要求立即自建完整实验室或联系他人的任务。
此次未联系作者。当前阶段以“没有获得足够的新独立数据”结束；不再对同一组最终 RGB 继续搜索物性。

## 6. 记录与验证

[可版本化来源账本](../tools/eval/independent_evidence_sources.json)保留 URL、固定提交、状态、时间、字节数、SHA256、
选定目录条目、36 张 TIFF 文件头及 800T/5219 逐字段比较。
本轮 65 次请求：48 次成功、16 次代理阻断；另一次颗粒分支提交号拼写错误导致 404，纠正为页面提供的完整提交后成功。
该 404 不构成数据不存在的证据。
公开网页和少量资料快照在 `/workspace/filmlab-audit/independent-evidence/`，属于可回收的本地缓存；账本已留在仓库。

检查包括 JSON 解析、成功响应摘要、目录来源、TIFF 元数据与代理/原站错误分类；未执行下载的外部代码。
本轮只更新来源记录和研究文档，未改生产 Python、未新增数值模型，故不重复跑渲染回归。
