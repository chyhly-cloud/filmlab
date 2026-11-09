# T55：银盐颗粒独立为 silvergrain 项目（Claude；重构，物理与数值不变）

## 0. 摘要

- **起因**：用户的指示是"银盐颗粒这个是单独的一个项目，你要单独去做"。颗粒应该独立开发，不再挂在 CineStill 800T 管线上验证。负责人补充了一项新要求：将来要能给**任意照片**加这种颗粒，既要从成像物理出发，也要看起来对。
- **做了什么**：
  - 在仓库顶层建了独立包 `silvergrain/`，有自己的 `pyproject.toml`、中文 README、测试、检验与示例。包内**零 `import filmlab`**（有测试检查），可以原样移到独立仓库。
  - filmlab 反过来依赖它：`filmlab/grain.py` 成为薄接口，`materials.py` 的通用常数与黑白数据从 `silvergrain/data` 读。
- **数值没有动**：
  - filmlab 已有输出 **238 组与 HEAD 逐位相同**：沿用 T53 的 180 组与 HEAD 比较，另 58 组显式设定，方法同 T46 §8 / T51 / T53。
  - silvergrain 自己的检验复现了 T46 / T47 / T47b / T50 / T53 的数：T46 留一、T47 四孔径、T47b、T53 闭式与参考结果**逐位相同**；T50 的 D-76 表与数据库参数的预测与文档一致。
- **新功能**：`apply_to_image`。两种模式：
  - "grain"（默认）：保留照片影调，只加物理颗粒；
  - "full"：该片的影调加颗粒。
  - 输入可以是 8 / 16 位 sRGB 或线性浮点、彩色或单色、任意尺寸。
  - 曝光走黑白片自己的光谱感光度，与 filmlab 同一套 RGB → 光谱方法，可加 Wratten 滤镜；测光按 ISO 约定 18% 灰 = 10/EI。
  - 可选 T51 实测的冲印店扫描。
  - 大图的黑白颗粒有快速路径 `closed_form`：用闭式协方差合成对数正态场，二阶统计与显式渲染相同。
- **测试**：filmlab 580 项通过，即原 568 项加新 12 项（`tests/test_t55_silvergrain.py`）。silvergrain 27 项通过，另有 1 项 slow：T46 全网格留一，单独运行通过。
- **检验中的新发现**（只报告，库里的数没有改）：
  1. **冲印店扫描光学的实现**。T51 的 Noritsu 光学 σ = 0.35 像素是按连续传递函数定的。filmlab 引擎用离散 `gaussian_filter` 实现它，几乎是恒等（邻点权重 0.017）。
     - 按连续 MTF 实现时，Tri-X 的胶片侧比值从 1.02–1.13 变为 1.17–1.30（真实 / 模型）；
     - 颗粒谱 P(0.48)/P(0.11) 从 2.36 变为 0.99（真实 1.42）。
     - silvergrain 用连续 MTF。filmlab 是否修改，留给负责人决定（第 6 节）。
  2. **同码值的颗粒强度主要由影调决定**。silvergrain 的 full 模式是按该片平均反差的直线反转，grain 模式用照片自己的影调；两者的中间调斜率（码值 / 底片密度）都比 RPL 的平。
     - 码值 68–138 时为 124–231，RPL 为 275–302；
     - 所以同码值的颗粒是真实的 1/1.5–1/2.9。
     - 胶片侧（除去影调）一致：Tri-X 1.08–1.30；HP5+ 1.44–1.79，T51b 已知 HP5+ 偏细。
     - 手机尺寸上，Tri-X 为 0.94–1.32（full）。

## 1. 包的结构

```
silvergrain/                     (仓库顶层; 本目录就是包: 扁平布局, pyproject 的 package-dir = {"silvergrain" = "."})
  __init__.py                    公共 API
  constants.py                   通用常数 (BAND_PHOTONS, ETA_REF, PHOTONS_R, AgBr / Ag 密度, GRAIN_SCALE_RANGE, check_grain_scale)
  dye.py                         T45 / T46 染料云与 κ 银量路径 (原 filmlab/grain.py 1–362 行 + scale_grain; apply → apply_dye_grain)
  silver.py                      T47–T53 显式银粒路径 (原 364–1097 行; apply_silver → silver_grain_density)
  database.py                    数据文件读取 (只读, 缓存)
  films.py                       Film / Developer / Curve, 胶片与显影液数据库
  emulsion.py                    derive(film, developer, grain_scale) → Emulsion; universal() (T46 通用常数)
  render.py                      render (explicit / closed_form / 染料云), granularity (闭式)
  spectra.py                     照片 RGB → 胶片每层曝光 (Jakob–Hanika 上采样 + 光源 + Wratten + 胶片光谱感光度)
  formats.py                     画幅 (T53 FILM_FORMATS), px_um_for, crop_to_format
  image.py                       apply_to_image = expose + render + finish; 读写、冲印店扫描
  data/                          universal.json, bw_t48.json, bw_t49_spectra.json, bw_t50_developers.json, wratten_filters.json,
                                 colour_films.json, official_rms.tsv, kodak_gran/, scanners.json, upsampling_xyY.npz
  validation/                    独立检验 (python -m silvergrain.validation) + data/ (Schmidt 读数、参考结果、真实扫描记录值) + results/report.json
  tests/                         pytest (27 项 + 1 项 slow)
  examples/apply_grain.py        示例 / 命令行 (合成测试图, 不需要第三方图片)
  README.md, pyproject.toml, conftest.py
```

- 物理核心按原代码**逐行搬移**：用脚本按行号切片，避免手抄出错。只做了两处结构改动：
  - `filmlab.grain.apply` 中与 FilmState 无关的部分成为 `dye.apply_dye_grain(spec, tgt, px_um, rng, reversal, strength, factor)`；
  - `apply_silver` 的核心成为 `silver.silver_grain_density(spec, d, px_um, seed, strength, render)`。
  - 随机数的调用顺序不变。
- 函数接受任何具有 `log_exposure`、`S_nom`、`design`、`stock.density_curves` 的曲线对象，即"鸭子类型"：filmlab 的 Emulsion 可以，silvergrain 的 `films.Curve` 也可以。

## 2. 公共 API

- `films()` / `developers()` / `film(key)` / `developer(key)`。
- `derive(film, developer=None, grain_scale=1)` → `Emulsion`，包含 `spec`、`mean_density(H)`、`exposure_axis()`、`describe()`。
- `render(emulsion, exposure=H | density=d, px_um=… | format=…, seed, method="auto", psf_um=0)` → `RenderResult`，包含 `density`、`mean`、`fluctuation`、`exposure`、`method`、`base_density`。
- `granularity(emulsion, density, aperture_um=48, shape="circle", net=True)`：闭式 RMS。黑白返回标量，彩色返回每层。
- `apply_to_image(image, film, developer, grain_scale, format, output_px, mode, exposure_ev, metering, filter, filter_compensate, scanner, seed, method)` → `GrainImage`，包含 `linear`、`encoded`、`info`、`save()`。拆开的步骤是 `expose()` 与 `finish()`。

彩色片的 `Developer` 是冲洗工艺（C-41 / ECN-2 / E-6）。黑白片的显影液是 d76 / xtol / rodinal / fx39。推冲不列，因为 T50 结论 3 认定它不可用。

## 3. 胶片与显影液数据库

- **黑白 10 种**：Tri-X 400、T-Max 100 / 400、Plus-X 125、HP5+、FP4+、Delta 100 / 400、Pan F+、Acros 100。
  - 用的是 filmlab 代际 t49_bw / t50_bw / t51_bw 的档案：T48 曲线、T49b 光谱，Tri-X 的曲线沿用 T47 的 Schmidt D-76。
  - 每种片都有感光度、晶体形态、曲线与冲洗口径、光谱、来源 URL，以及官方 RMS 与测量条件（取自 T46 的 films.tsv）。
- **彩色 17 种**：Vision3 4 种、Portra 3 种、Ektar、Gold、UltraMax、Pro 400H、X-TRA 400、C200、Provia 100F、Velvia 100、Ektachrome 100、Kodachrome 64。
  - 数据由 `tools/export_silvergrain_data.py` 从 filmlab 档案导出（spektrafilm，CC BY-SA 4.0）：曲线、染料光谱、光谱感光度，按 repr 写出，读回逐位相同。
  - T46 的公开物理量（EI、光源、形态）按 `materials.GRAIN_PHYS` 的同一规则解析。
  - Kodachrome 的 K-14 冲洗没有建模，没有进入检验。
- **移动的文件**（`git mv`，内容不变）：`filmlab/data/{bw_t48, bw_t49_spectra, bw_t50_developers, wratten_filters}.json` → `silvergrain/data/`。
  - `materials.bw_t48_data()` 等函数、`recipes.wratten_data()` 改为读新位置；两处测试与一个评估脚本的路径也已更新。
- **复制的数据**（git 按内容存，同一 blob 不增加仓库大小）：
  - `tools/eval/t46/films.tsv` → `data/official_rms.tsv`；
  - Kodak 颗粒度数字化 4 个 → `data/kodak_gran/`；
  - `filmlab/data/upsampling_xyY.npz`（filmlab 的颜色管线仍用自己的一份）；
  - Schmidt 与参考结果 → `validation/data/`。
- **通用常数**（`data/universal.json`，由 materials 的字面值导出）：5219 结构、η、β、a0、κ_Ag、T 颗粒厚度、冲洗工艺、T47 银丝形态、η_3D ×6.608337、Tri-X 曲线与 F-4017 光谱。
  - `materials.GRAIN_*`、`TRIX_*` 改为从这里读。读回的值与原字面值相同，有测试检查。
- **一致性**：silvergrain 推导的 57 个档案（黑白 10 片 × 4 种显影液，彩色 17 片）与 filmlab 的 `materials.silver_spec` / `grain_spec_t46` **相等**（dataclass 相等，含 k = 0.5 / 2）。黑白片的特性曲线与 `bw_silver_film` 逐位相同。

## 4. apply_to_image：物理定义

完整说明见 `silvergrain/README.md` 第 4 节与 `image.py` 模块说明。要点：

- **几何**：画幅长边对图像长边，µm/像素 = 画幅长边 × 1000 / 图像长边像素数，默认 35 mm。`output_px` 指在线性光里按面积缩小，即以输出像素为扫描孔径。
- **曝光**：
  - 把照片的线性值当作反射率，经 Jakob–Hanika 光谱上采样（与 `filmlab.spectral.spectral_response` 同一算法），在设计光源下照明（D55 / 3200 K），可加 Wratten 滤镜。
  - 按该片每层的光谱感光度积分：黑白用 T48 / T49b 的逐片全色光谱，彩色用三层光谱。
  - 18% 灰落在 H = 10/EI，再乘 2^EV。`metering="average"` 时以对数平均为中灰。
  - 带滤镜时默认按中性灰补偿，系数由模型算出（T49：与 Kodak 公布值 RMS 0.23 档）。
- **full**：
  - 负片：P = 0.18·10^((D − D_g)/γ̄)，γ̄ 为中灰 ±2 档的平均斜率；
  - 反转片：P = 10^−(D − D_min)。
  - 彩色片每层只在自己的通道里读，没有串色校正。
- **grain**（默认）。物理的部分：
  - 照片值 ↔ 底片密度的映射，就是该片特性曲线的反函数；
  - δ = D_px − D̄ 是显式银粒 / 染料云在该密度处的起伏，含空间结构与 −log⟨T⟩；
  - 输出 v' = v·10^(δ/γ_eff − b(H))：
    - 斜率取常数时，v·10^(δ/γ) 是反映射的精确形式，恒为正；一阶为 v·(1 + ln10·δ/γ)。
    - b(H) 是指数映射的二阶偏移，按同一曝光的全部像素算出、扣除（`image._exp_bias`）。
    - 输出平均 = 照片 × 10^(E[δ]/γ)，即只剩物理的 −log⟨T⟩。12 µm、18% 灰时：Tri-X k = 1 +1.1%，Rodinal +2.0%，k = 3 +8.8%，Portra ≈ 0。
    - 最初的实现用一阶形式，在 k ≥ 2 的整图上截断出黑点；只用指数形式又让 k = 3 的平均亮 29%。两者都是整图检查时发现的，现已改正。
  - γ_eff = max(γ, γ_速度点)：速度点以下胶片不记录可用影像，这是唯一的约定；
  - 黑白只作用于亮度，彩色每层进对应通道。
  - 假设照片线性值与景物亮度成正比。带 S 形曲线的相机 JPEG 中间调更陡，颗粒会被低估同样的倍数。
- **冲印店扫描**（可选）：
  - 光学用**连续**高斯 MTF（频域），孔径按画幅换算，亮度 USM，彩色另做色度降噪；
  - grain 模式只作用于颗粒部分：光学与孔径作用于 δ，USM 作用于编码后的差。

## 5. 检验（`python -m silvergrain.validation` → `silvergrain/validation/results/report.json`；只用颗粒数据）

| 项 | silvergrain | 原文档 / 参考 | 判定 |
|---|---|---|---|
| T46 官方 RMS 留一（38 种，13 β × 41 η 网格，187 s） | β 0.875、η 0.02478；留一 \|log\| 中位 0.170（×1.19）、RMS 0.208；电影负片 0.89–0.97；Tri-X 1.31、T-Max 100 0.67 … | `tools/eval/t46/loo.json`：逐片比值差 0.0 | **逐位相同** |
| T47 Schmidt Tri-X / D-76 四孔径 | +0.4% / +0.2% / −2.2% / +2.2%（门槛 ±15%）；19 级 \|ΔD\| ≤ 0.089；Kodak 48 µm 14.6 | `results.json` closed | **逐位相同** |
| T47b 官方 RMS | T-Max 100 1.00、T-Max 400 1.11（判定）；Tri-X 0.86、Tri-X 320 0.85、Plus-X 1.06、Acros 1.14（只报告） | `t47b_loo.json` | **逐位相同** |
| T47b Schmidt 6 种片 | 1.04（标定）、T-Max 100 1.14、T-Max 400 0.90、Delta 100 0.91、Delta 400 1.00、Pan F+ 0.70；判定 \|log\| 0.05 / 0.08，留出 0.10 / 0.18 | 同上 | **逐位相同** |
| T50 D-76 参照 | RMS_D 13.6 / 9.0 / 11.6 / 7.9 / 9.9 / 6.2；a_k 5.7 / 5.7 / 6.0 / 5.7 / 5.9 / 5.5 µm | T50 表 4.1 | 相同 |
| T50 显影液（数据库的全部片拟合参数） | XTOL：RMS 倍数 ±20% 内 5/6、a_k 6/6；Rodinal：Tri-X 1.42 / 1.58（19.2、a_k 6.5）、Delta 100 1.50 / 1.57；FX39：5/6、4/6 | T50 表 4.2 中参数相同的行（Tri-X、Delta 100、Pan F+ 等）数值相同；留一要用 `--full-t50`（约 30 分钟，未在本任务中重跑） | 一致 |
| T53 颗粒粗细（闭式） | σ√A(48 µm)/k 0.983–1.004（Tri-X），96 µm 0.98–1.00；曲线最大差 0.0061 D（T-Max 400）；涂布银 ∝ k；Portra 每层 σ48/k = 1.000 | `tools/eval/t53/validate.json` 的闭式量：差 0.0 | **逐位相同** |

**新的检验**

快速路径 closed_form 对 explicit（160² 像素，5 种组合 × 6 / 12 µm × EV −2 / 0 / +2）：

- 像素 σ_D 之比：k = 1 时 0.95–1.08，k = 2、6 µm（等效 3 µm）时 0.92–0.94；
- 相邻像素相关两者相近（0.01–0.08）；
- 平均密度差 ≤ 0.008 D（Rodinal 成团，闭式解本身的点足迹近似）。
- 据此，auto 规则定为：等效像素 px/k ≥ 3 µm 且图像大于 2 MP 时用 closed_form。

**真实扫描的记录值**（`validation/realscan.py`，不下载）。条件是 RPL 645 Noritsu、20 µm/像素，Tri-X 用 XTOL、HP5+ 用 D-76，均匀 18% 灰 × 2^EV（−3 … +3），silvergrain 两种模式，Noritsu 扫描开。

| | Tri-X（XTOL） | HP5+（D-76） |
|---|---|---|
| 胶片侧（真实 σ48 / 真实斜率）÷（silvergrain σ48 / silvergrain 斜率），码值 68–184 | full：1.17–1.30；grain：1.08–1.25；扫描高斯按 filmlab 的离散核：**1.02–1.13** | full：1.51–1.79；grain：1.44–1.57；离散核 1.31–1.54 |
| filmlab 模型同一量（T51b） | 0.98–1.12 | 1.28–1.42 |
| 影调斜率 dCode/dD，码值 68 / 99 / 138 / 184 | RPL 275 / 302 / 302 / 242；silvergrain full 124 / 171 / 231 / 301 | RPL 249 / 267 / 261 / 197；full 123 / 172 / 226 / 298 |
| 同码值 σ_48 真实 / silvergrain（full） | 2.88 / 2.14 / 1.55 / 0.94 | 3.12 / 2.46 / 1.74 / 1.18 |
| 原尺寸 σ_px（T52 锚）真实 / silvergrain，码值 99 / 138 / 184 | full 2.40 / 1.64 / 0.96；grain 1.90 / 1.77 / 1.10 | full 2.82 / 1.92 / 1.29 |
| 手机尺寸 σ_px（缩 2.33 倍），码值 148 / 184 | full 1.32 / 0.94；grain 1.52 / 1.07 | full 1.91 / 1.49 / 1.13（码值 102 / 147 / 181） |
| 谱形 P(f)/P(0.11)，f = 0.30 / 0.48 周/像素 | 真实 1.15 / 1.42；不经扫描 0.95 / 0.97（白）；Noritsu 连续 MTF 1.32 / 0.99；离散核 1.75 / 2.36 | 真实 0.89 / 0.99；不经扫描 1.00 / 1.04；连续 1.39 / 1.06；离散 1.84 / 2.51 |

**读法**

- **胶片物理与 filmlab 一致**。扫描按同一离散实现时，Tri-X 的胶片侧 1.02–1.13，对 filmlab 的 0.98–1.12，剩下的差别来自两者的扫描影调形状。HP5+ 偏细约 1.35 倍，T51b 已经记录。
- **同码值的差距来自影调**。silvergrain 的直线反转（系统反差 1），以及照片自己的影调（场景线性 → sRGB），在暗部与中间调的斜率只有 RPL 的 0.45–0.77，所以在同一码值上颗粒弱 1.5–2.9 倍。码值 184 处斜率相当，比值为 0.94–1.10。
  - 这不是颗粒物理的问题，没有按观感去调。
  - 用户若在意，应给出"照片的影调相对景物的反差"这一物理量，或用 RPL 的影调。这一项留给负责人决定，不在本任务中引入。
- **谱形**：
  - 不经扫描时，20 µm 像素上是白的，符合物理。
  - 连续 MTF 的 Noritsu 在中频上升（0.30 周/像素 1.32，真实 1.15），在 0.48 回落（0.99，真实 1.42）。离散核在高频过冲（2.36）。
  - 真实谱还含扫描仪的白噪声底与 JPEG（T51 的 N 项），silvergrain 没有建模，因为它不属于颗粒。

彩色（Portra 400 / C-41，RPL Noritsu）：同码值 σ_48 真实 / silvergrain(full) 在码值 71–229 上为 3.71 → 0.55，同样由影调主导（silvergrain 的直线反转对 RPL 的数字相纸影调）。silvergrain 的彩色物理与 filmlab 逐位相同，T54 已给出胶片侧：亮部 0.98，暗部 1.3–2.2 未归因。记录值里没有彩色的斜率，所以这里不再分解。

## 6. 发现：filmlab 引擎的扫描光学几乎没有生效（只报告）

- **现象**：`engine._scan_psf` 对 T51 的冲印店参数执行 `gaussian_filter(T, (0.35, 0.35, 0))`。σ = 0.35 像素的离散高斯核，中心权 1、邻点 0.017，几乎是恒等，与 T51 按连续传递函数定的 σ_o = 7 µm 不一致。
  - USM 半径 0.68 像素也用离散核（邻点权 0.34），偏差较小。
- **影响**：
  - t51_bw 的 Noritsu 输出在高频偏强：谱形 0.48 周/像素处 2.36 倍，真实 1.42。这与 T51b / T52 记录的"模型像素级颗粒略强于真实"（Tri-X 0.77–0.95）方向一致。
  - 改为连续 MTF 后，Tri-X 胶片侧的真实 / 模型从约 1.05 升到约 1.2。
- **处理**：silvergrain 用连续 MTF（`image._gauss`，频域）。filmlab 的代码与输出**没有改**，以保证逐位相同。是否修正、修正后要不要重新拟合 T51 的 σ_o，由负责人决定。

## 7. 性能（24 MP，6064 × 4040，35 mm，5.94 µm/像素；单进程，`ulimit -v 9000000`；人像帧 2× 放大）

| 胶片 / 显影液 | 模式 | 渲染法 | 时间 | 峰值内存 |
|---|---|---|---|---|
| Tri-X / D-76，k = 1 | grain | closed_form | 106 s | 3.7 GB |
| Tri-X / D-76，k = 1 | full | closed_form | 96 s | 3.5 GB |
| Tri-X / Rodinal，k = 1 | grain | closed_form（含成团协方差） | 168 s | 3.7 GB |
| Tri-X / D-76，k = 2 | grain | explicit（px/k = 2.97 µm < 3） | 1350 s（22.5 分钟，与其他任务同时运行） | 3.5 GB |
| Tri-X / D-76，k = 1，强制 explicit | — | explicit | 84 s / 百万像素 → 24 MP 约 34 分钟 | 0.4 GB / 百万像素块 |
| Tri-X / Rodinal，k = 1，强制 explicit | — | explicit | 131 s / 百万像素 → 24 MP 约 53 分钟 | |
| Portra 400 / C-41 | grain | 染料云（计数级） | 309 s | 5.7 GB |
| Portra 400 / C-41 | full | 染料云 | 308 s | 4.1 GB |

- 24 MP 用 `auto` 时，黑白在 2–3 分钟内完成（closed_form），彩色约 5 分钟。显式银粒要 0.5–1 小时，所以只在小图、细像素或 k 大时用。
- 测量用 `scratchpad/t55/work/perf24.py`、`perf_explicit.py`。

- 黑白 closed_form 的时间主要花在逐档曝光的协方差上（每档约 0.4 s，加整图的相关滤波）。
- 彩色染料云是计数级模拟，按 256 行分条，约 12 s / 百万像素。
- 峰值内存 3.5–5.7 GB。

## 8. 逐位核对与测试

- **逐位核对**：用 `scratchpad/t55/work/bitwise.py`，即 T53 的脚本原样；对照 HEAD = 1fd511a，用 `git archive` 导出到 scratchpad。
  - 组与 T53 相同：T51 的 140 组，加 t51_base / t51_bw 的 40 组，共 180 组与 HEAD 比较；
  - 新代码的显式设定（`grain_scale=1`、`+g1`、`film_format="35mm"`）另 58 组；
  - 结果 **238 组相同，0 组不同**（`bitcmp.py`）。
- **filmlab 测试**：580 项通过（568 + `tests/test_t55_silvergrain.py` 12 项）。新测试检查：
  - 薄接口重新导出的是同一对象；
  - materials 常数来自 silvergrain；
  - 通用常数与彩色曲线相同；
  - 黑白（5 组，含成团、XTOL、FX39）与彩色档案相等；
  - `apply_silver` 与 `silvergrain.render` 逐位相同，`apply` 与 `apply_dye_grain` 逐位相同；
  - `SILVER_WORKERS` 仍可设置。
- **silvergrain 测试**：27 项通过，另有 1 项 slow（T46 全网格留一，单独运行通过）。覆盖：
  - 包的独立性（AST 检查无 filmlab 导入）、数据库与来源；
  - 物理：显式平均 = 闭式，−log⟨T⟩，k 的标度，快速路径；
  - 检验：T46 / T47 / T47b / T53 逐位相同，T50 表 4.1；
  - apply_to_image：8 / 16 位、单色、两种模式、曝光与滤镜、扫描、确定性。

## 9. 整图（`/tmp/claude-0/-home-user/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/t55/view/`，不进仓库）

- **照片**：EPFL 人像 `people_61227` 与日景 `day_9195`，是 T48 的数码帧，scratchpad 里的线性 npy；第三张是合成测试图（`examples.apply_grain.synthetic`）。
- **每张照片 12 个变体**：
  - 数码原图；
  - Tri-X D-76 / Rodinal 与 Portra 400 的 grain 模式；
  - Tri-X D-76 / Rodinal 与 Portra 400 的 full 模式；
  - Tri-X D-76 的 k = 0.5 / 2 / 3；
  - Tri-X XTOL 与 Portra 400 加 Noritsu 扫描。
- 每个变体有原尺寸（`<照片>__<变体>.jpg`）与手机尺寸（`__phone.jpg`，Lanczos 长边 1200），另有每张照片一张 100% 局部对照（`<照片>__crops_100pct.jpg`）。
- 全部由 silvergrain 独立生成，不经 filmlab。用时记录在 `views.json`。

## 10. 局限

见 `silvergrain/README.md` 第 8 节，是 T45–T54 负责人结论的汇总：

- 没有 3200 片；
- 推冲不可用；
- HP5+ / FP4 楔常数；
- Pan F+ 0.70；
- XTOL 的顺序；
- FX39；
- 三维晶体 β 弱；
- EPFL 800T 2.7 倍未归因；
- k 的感光度假设；
- 渲染器边界。

T55 新增：

- apply_to_image 假设照片线性值与景物亮度成正比；
- full 模式是直线反转，不是冲印店影调（第 5 节）；
- 彩色 full 模式每层只在自己的通道里读，没有串色校正，也不模拟扫描仪通道；
- closed_form 只保证二阶统计，k 大、像素细时用 explicit。

## 11. 文件

- **新**：`silvergrain/`（全部），`tools/export_silvergrain_data.py`，`tests/test_t55_silvergrain.py`，`docs/tasks/T55_SILVERGRAIN_STANDALONE.md`。
- **改**：
  - `filmlab/grain.py`：1115 行 → 43 行薄接口；
  - `filmlab/materials.py`：常数与数据从 silvergrain 读；
  - `filmlab/recipes.py`：Wratten 数据路径；
  - `filmlab/engine.py`：`FILM_FORMATS` / `crop_to_format` 从 silvergrain.formats 导入，数值不变；
  - `tests/test_t48_bw.py`、`tests/test_t49_filters.py`：数据路径；
  - `tools/eval/t49_filter_factors.py`：说明里的路径。
- **移动**：`filmlab/data/{bw_t48, bw_t49_spectra, bw_t50_developers, wratten_filters}.json` → `silvergrain/data/`。
- scratchpad（不进仓库）`scratchpad/t55/`：
  - `work/`：逐位核对脚本与日志、性能、检验日志、`views.py`；
  - `view/`：整图。
  - HEAD 导出与逐位核对的 npz 用完已删。

## 负责人结论

审核范围：包结构（零 filmlab 依赖）、公开接口、颗粒自身数据的验证报告（T46 留一、T47 Schmidt 四孔径、T47b 官方值、T50 显影液、T53 粗细）、filmlab 逐位核对 238 组、580 + 27 项测试（通过）、三张照片 × 12 种设置的整图与 100% 对照、24 MP 用时与内存。

1. **银盐颗粒成为独立项目 `silvergrain/` 采用。** 物理与数值未变：所有验证数字与 T46–T53 文档相同；filmlab 改为调用它，现有代际逐位相同。以后颗粒的工作只在这个包里、只用颗粒自身的数据（官方颗粒度、Schmidt 实测、真实扫描的测量）检验，不再对照 800T 链路。
2. **任意照片加颗粒（apply_to_image）可用**：画幅定 µm/像素，胶片自身光谱感光与 EI 定曝光，"只加颗粒"模式保留原图影调和颜色；24 MP 快速路径约 2–3 分钟（彩色约 5 分钟），峰值内存 4–6 GB。
3. **审图发现、下一步（T55b）修正后再给用户看**：
   - "只加颗粒"模式把局部斜率取自"照片 = 场景线性的 sRGB 编码"，中间调比真实冲印店的影调平（每单位密度 124–231 码值 vs RPL 实测 275–302），颗粒比真实扫描弱 1.5–2.9 倍。改为用实测的冲印店影调（底片密度 ↔ 输出码值，T51b / T54 的配准，属冲印店属性，同 T44）。
   - "完整胶片"模式的直线反转扫描使黑白片整体发白，同样改用实测冲印店影调。
   - 黑白胶片颗粒加在彩色照片上时，高 k 出现带色的亮点；黑白银盐颗粒只改变亮度，应保持色度。
4. filmlab 中 Noritsu 光学 σ（0.35 像素）用离散高斯核近似为恒等，导致高频偏高（0.48 周/像素 2.36 vs 真实 1.42）；silvergrain 用连续传递函数。filmlab 的修正会改变 t51 代际输出，另行处理。
