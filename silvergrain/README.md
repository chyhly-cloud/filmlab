# silvergrain — 银盐颗粒（独立项目）

silvergrain 从第一性原理生成胶片的银盐颗粒：卤化银晶体 → 光子统计显影 → 显影银丝团 / 染料云 → 密度 → 透射率。它可以单独计算颗粒度，也可以给任意照片加颗粒。

- 只依赖 numpy 与 scipy。读写图片另需 Pillow，imageio 可选，用于 16 位 TIFF / PNG。
- **不依赖 filmlab**：包内没有任何 `import filmlab`，测试 `test_no_filmlab_imports` 会检查这一点。整个目录可以原样移到独立仓库。
- filmlab（仓库里的 CineStill 800T 模拟器）反过来依赖本包：`filmlab/grain.py` 只是一层薄接口。
- 本包的物理与数值来自 filmlab 的 T45–T54。T55 只是把它们搬出来，数值没有变，见"检验"一节。

## 1. 物理链

```
 胶片公开物理量 (感光度 EI、设计光源、晶体形态、特性曲线、官方颗粒度)        通用常数 (跨片种, 不按片调)
                         │                                                  η, β (T46 留一), 阈值 r = 4 光子,
                         ▼                                                  T 颗粒厚 0.12 µm, 5219 乳剂结构 λ_ck
 ① 卤化银晶体群: 3 个亚群, ECD 对数正态; a = (λ_k a0^β / (φ η H_s))^(1/(1+β)),  H_s = 0.8/EI
    每亚群涂布量 (彩色: 染料当量; 黑白: g Ag/m²) 由该片自己的特性曲线非负最小二乘定
                         │
 ② 曝光 H (lux·s) → 每晶体有效光子数 ~ Poisson(φ·a·(a/a0)^β·H); ≥ r 个 → 可显影 (P = gammainc(r, λ)); 灰雾 1 − e^(−a/a_f)
                         │                               显影液 (黑白, T50): r 与显影晶体成团 (Thomas 过程 μ, σ)
                         ▼
 ③ 显影 ─┬─ 黑白 (显式银粒, T47): 每颗显影晶体原位长成银丝团 (银质量守恒: 丝长 = 银体积 / 丝截面; 丝径 0.04 µm,
         │    丝团面积 = 4 × 晶体), 投影为不透明细丝的并集; 透射率 exp(−Σ τ), 漫射收集 g = 1/Q (Callier 1.4)
         └─ 彩色 (染料云, T46): 每颗显影晶体生成一团染料云 (积分密度 κ·v, 产量伽马随机 CV 0.3,
              云的高斯 σ = √((ECD/4)² + L²), L 由冲洗工艺定)
                         │
 ④ 像素 (边长 px = 画幅长边 / 图像长边): 透射率在像素面积上积分, D_px = −log10 ⟨T⟩_px
    闭式解: 平均 E[T] = exp(−Σ n p g A) (泊松布尔模型), 透射率协方差 Cov_T(r) (含成团的交叉项) → 任意孔径的 RMS 颗粒度
                         │
 ⑤ 颗粒粗细 k (T53): 所有颗粒线尺度 ×k, 投影光学截面守恒 → 特性曲线不变, 颗粒场 = k = 1 的场放大 k 倍
```

不在输出端加任何统计噪声。颗粒的位置、数目与强度只来自晶体。

## 2. 安装与运行

```bash
pip install -e silvergrain[image,test]     # 或直接把仓库根目录放进 PYTHONPATH (扁平布局: 本目录就是包)
python -m pytest silvergrain/tests -m "not slow"     # 27 项, 约 1 分钟 (另有 1 项 slow: T46 全网格留一)
python -m silvergrain.validation                     # 独立检验, 结果写入 silvergrain/validation/results/report.json
python -m silvergrain.examples.apply_grain synthetic out.jpg --film kodak_trix_400 --developer rodinal --phone out_phone.jpg
```

## 3. API

```python
import silvergrain as sg

sg.films(), sg.developers()                       # 数据库: {键: Film}, {键: Developer}
em = sg.derive("kodak_trix_400", "rodinal", grain_scale=1.0)   # Film + Developer + k → Emulsion (晶体群)
em.spec                                            # silver.SilverSpec (黑白) / dye.GrainSpec (彩色), 全部物理量
em.mean_density(H)                                 # 闭式平均密度 (不含片基)

r = sg.render(em, exposure=H, px_um=6.0, seed=1)   # 或 density=…, format="35mm"
r.density, r.mean, r.fluctuation, r.method        # 逐像素颗粒密度、闭式平均、两者之差、渲染法

sg.granularity(em, density=1.0, aperture_um=48)    # 闭式 RMS 颗粒度 (净密度 1.0, 48 µm 圆孔; 彩色返回每层)

out = sg.apply_to_image(image, film="kodak_trix_400", developer=None, grain_scale=1.0, format="35mm",
                        output_px=None, mode="grain", exposure_ev=0.0, metering="fixed",
                        filter=None, scanner=None, seed=0, method="auto")
out.linear, out.encoded, out.info, out.save("x.jpg")
```

- `Film`：`key`、`name`、`kind`（bw / negative / reversal）、`ei`、`thickness_um`（None 表示三维晶体）、`crystal`、`illuminant`、`process`、`official`（官方颗粒度、测量条件、来源 URL），以及方法 `curve()`、`spectral()`。
- `Developer`：`d76`、`xtol`、`rodinal`、`fx39`（T50），以及彩色的 `C-41`、`ECN-2`、`E-6`（T46）。属性有 `cluster_mu`、`cluster_um`，`photons_r(film)` 给出该片在这种显影液下的阈值。
- 底层的物理模块是 `silvergrain.silver`（显式银粒）、`silvergrain.dye`（染料云 / κ 银量）和 `silvergrain.constants`。函数名与原来的 `filmlab.grain` 相同。

### 3.1 render 的两种黑白渲染法

| method | 做法 | 用于 |
|---|---|---|
| `explicit` | 每颗晶体在连续坐标中撒下（泊松 / 成团），显影后按银丝团形状沉积，在透射率域积分到像素。与 filmlab 整帧用的是同一个渲染器。 | 小图、细像素、需要真实颗粒形状时。约 1–3 分钟 / 百万像素（6 µm 像素） |
| `closed_form` | 用闭式透射率协方差（含成团）在像素方盒上积分，得到像素间协方差。像素透射率取对数正态 `T = E[T]·exp(s z − s²/2)`，均值、方差、像素间协方差都与闭式解严格相等。按 log H 每 0.1 一档插值。 | 大图。`auto`：等效像素 px/k ≥ 3 µm 且图像大于 2 MP 时使用 |

两者的比较见 `validation/fastpath.py`，这里只列结论：像素 σ_D 之比在 k = 1 时为 0.95–1.08，k = 2、6 µm 时为 0.92–0.94；平均密度之差 ≤ 0.008 D（Rodinal 成团时最大）。

彩色的染料云路径本身就是计数级的显式模拟，不需要近似。为限制内存，按 256 行分条渲染。

## 4. apply_to_image 的物理定义

0. **读入**：可以是 sRGB 编码的 8 / 16 位 JPEG、PNG、TIFF，也可以是线性浮点数组；彩色或单色均可，尺寸和长宽比不限。整数输入按 sRGB 解码，浮点输入默认当作线性。
1. **几何**：画幅长边对图像长边，`px_um = 画幅长边 × 1000 / 图像长边像素数`。画幅默认 35 mm，可选 half、110、645、6x7、4x5…，也可以直接给长边的 mm 数。给了 `output_px` 时，先在线性光里按面积平均缩小，相当于以输出像素为扫描孔径。
2. **曝光**：
   - 照片的线性值当作景物反射率，中灰取 0.18。
   - 用 RGB → 光谱上采样得到反射光谱（Jakob–Hanika 2019 查找表，与 filmlab 同一方法），在胶片的设计光源下照明（日光片 D55，灯光片 3200 K）。
   - 可选镜头前加 Wratten 滤镜（T49 实测透过率），默认按中性灰补偿滤镜系数。
   - 按胶片每层自己的光谱感光度积分：黑白片用 T48 / T49b 的逐片全色光谱，彩色片用三层光谱。
   - 测光按 ISO 约定，18% 灰落在 `H = 10/EI`；`exposure_ev` 再乘 2^EV。`metering="average"` 时以对数平均亮度当中灰。
3. **底片**：晶体模型给出平均密度（闭式特性曲线）和逐像素的颗粒密度（`render`）。
4. **输出**，分两种模式：
   - **`mode="full"`（整卷胶片）**：底片密度 → 透射率 → 简单扫描成正像。
     - 负片：`P = 0.18 · 10^((D − D_g)/γ̄)`。D_g 是 18% 灰的密度，γ̄ 是该片在中灰 ±2 档的平均斜率。也就是按胶片自己的平均反差做直线反转，中灰对齐。不加冲印店的影调曲线，不做颜色校正，彩色片每层只在自己的通道里读。
     - 反转片：`P = 10^−(D − D_min)`，即透射率本身。
     - 输出的影调是该片的影调（趾部、肩部、反差）。
   - **`mode="grain"`（只加颗粒，默认）**：保留照片自己的影调与颜色，只加物理颗粒。这一模式里物理的部分如下：
     - 把每个像素看成"这种胶片在这一曝光下的底片，经扫描与影调映射得到了照片的这个值"。照片值与底片密度的映射，就是该片特性曲线的反函数（log v 与 log H 只差常数）。
     - 颗粒让该像素的密度偏离平均：`δ = D_px − D̄`。δ 来自显式银粒 / 染料云在这个密度上的起伏，带有完整的空间结构，并含物理的 −log⟨T⟩ 偏移。
     - δ 经反映射的局部斜率传到输出：`v' = v · 10^(δ/γ_eff − b(H))`。
       - γ = dD̄/dlog10 H 是该片在这一曝光处的斜率。
       - 在起伏范围内斜率取常数时，`v·10^(δ/γ)` 就是反映射的精确形式，与 full 模式的扫描同一形式，输出恒为正。一阶近似 `v·(1 + ln10·δ/γ)` 在粗颗粒（k ≥ 2）时会截断出黑点，所以不用。
       - `b(H) = log10 E[10^(δ/γ)] − E[δ/γ]` 是指数映射自己的二阶偏移。它在**同一曝光的全部像素**上算出（按 log H 每 0.05 分档，不是空间邻域，所以没有边缘光晕），扣掉后，输出的期望与一阶传递相同。
     - `γ_eff = max(γ, γ_s)`，γ_s 是速度点（净密度 0.1）的斜率。速度点以下，胶片按 ISO 感光度的定义不再记录可用的影像信息，扫描黑位也设在这里，所以斜率不再变小；否则 γ → 0 时颗粒会发散。这是这一模式唯一的约定。
     - 输出的平均等于照片，只差物理的 −log⟨T⟩：像素内的透射率先平均再取对数，E[δ] > 0，平均输出为 `v·10^(E[δ]/γ)`。颗粒越粗，画面越略亮（T53 2.6 节）。
       - 12 µm 像素、18% 灰时：Tri-X / D-76 k = 1 为 +1.1%，Rodinal +2.0%，k = 3 为 +8.8%；
       - Portra 400 约为 0，因为染料云在密度上是线性叠加的；
       - 数值记录在 `info["mean_rel_shift"]`。
     - 彩色片：每层染料颗粒进对应通道（R 层 → R 通道…），三层的颗粒相互独立。
     - 黑白片：只有亮度，三通道用同一个相对起伏，所以照片的色相不变。
     - 假设：照片解码后的线性值与景物亮度成正比。带 S 形影调曲线的相机 JPEG 会被当作景物本身的影调。
5. **冲印店扫描**（可选，`scanner="noritsu_rpl"` / `"frontier_epfl"`；参数是 T51 的实测）：
   - 扫描光学用连续高斯 MTF（频域）。T51 的参数是按连续传递函数定的，像素以下的 σ 不能用离散高斯核，见"局限 8"。
   - 冲印店像素孔径按画幅换算。
   - 亮度做 USM；彩色片另做色度降噪。
   - grain 模式下，这一步只作用于颗粒部分。
6. **输出**：线性 sRGB 与 sRGB 编码。不加噪声，不调色。

## 5. 胶片与显影液

| 胶片 | 键 | 类 | EI | 晶体 | 曲线 | 官方颗粒度 |
|---|---|---|---|---|---|---|
| Kodak Tri-X 400 | `kodak_trix_400` | 黑白 | 400 | 三维 | Schmidt D-76 原液 20 °C 9 min（T47） | RMS 17（HC-110） |
| Kodak T-Max 100 | `kodak_tmax_100` | 黑白 | 100 | T-GRAIN 0.12 µm | Kodak D-76 7.5 min | RMS 8（D-76） |
| Kodak T-Max 400 | `kodak_tmax_400` | 黑白 | 400 | T-GRAIN | Kodak D-76 8 min | RMS 10（D-76） |
| Kodak Plus-X 125 | `kodak_plusx_125` | 黑白 | 125 | 三维 | 借 HC-110 曲线 | RMS 10（HC-110） |
| Ilford HP5 Plus | `ilford_hp5_plus` | 黑白 | 400 | 三维 | 借 ILFOTEC HC 曲线 | 不公布 |
| Ilford FP4 Plus | `ilford_fp4_plus` | 黑白 | 125 | 三维 | 借 ILFOTEC HC 曲线 | 不公布 |
| Ilford Delta 100 / 400 | `ilford_delta_100` / `_400` | 黑白 | 100 / 400 | 平板（按 T 颗粒，假设） | ID-11 | 不公布 |
| Ilford Pan F Plus | `ilford_pan_f_plus` | 黑白 | 50 | 三维 | Schmidt D-76 | 不公布 |
| Fujifilm Acros 100 | `fuji_acros_100` | 黑白 | 100 | Σ（按 T 颗粒，假设） | Fuji D-76 7 min | RMS 7（Microfine） |
| Kodak Vision3 500T / 200T / 250D / 50D | `kodak_vision3_*` | 电影负片 | 500 / 200 / 250 / 50 | T 颗粒 | 数据册（ECN-2） | 每层 RMS 曲线（数字化） |
| Kodak Portra 160 / 400 / 800 | `kodak_portra_*` | 负片 | | T 颗粒 | 数据册（C-41） | PGI 28 / 37 / 48 |
| Kodak Ektar 100、Gold 200、UltraMax 400 | | 负片 | | T 颗粒 | 数据册 | PGI <25 / 44 / 46 |
| Fujifilm Pro 400H、Superia X-TRA 400、C200 | | 负片 | | 按平板（假设） | 数据册 | Fuji RMS 4 / 4 / 4 |
| Fujifilm Provia 100F、Velvia 100、Kodak Ektachrome 100 | | 反转 | 100 | 按 T 颗粒（假设） | 数据册（E-6） | RMS 8 / 8 / 8 |
| Kodak Kodachrome 64 | `kodak_kodachrome_64` | 反转 | 64 | 三维 | 数据册 | RMS 10。K-14 冲洗没有建模，这里按 E-6，没有进入检验 |

- 显影液（黑白，T50；μ 为每团平均显影晶体数）：
  - D-76：参照，泊松；
  - XTOL：不成团，r 按片；
  - Rodinal：μ 2、σ 1 µm；
  - FX-39：μ 0.5、σ 1.5 µm。
- 冲洗工艺（彩色，T46）：C-41 / ECN-2（产量 CV 0.3，L 1 µm）、E-6（κ ×5.41）。

数据文件都在 `data/`，每一条都有来源；`database.py` 列出了全部文件：
- `bw_t48.json`、`bw_t49_spectra.json`、`bw_t50_developers.json`、`wratten_filters.json`：从 filmlab/data 移来。
- `universal.json`：通用常数。
- `colour_films.json`：彩色片的曲线与光谱，由 filmlab 档案导出（spektrafilm，CC BY-SA 4.0）。
- `official_rms.tsv`：46 种胶片的官方颗粒度与测量条件，即 T46 的 films.tsv。
- `kodak_gran/`：Kodak 电影负片的颗粒度曲线数字化。
- `scanners.json`：T51 的冲印店参数。

## 6. 检验（`python -m silvergrain.validation`；只用颗粒数据，不用任何 800T 管线）

| 检验 | silvergrain 的结果 | 与原文档 |
|---|---|---|
| T46：38 种胶片官方 RMS 留一 | β 0.875、η 0.0248；留一 \|log\| 中位 0.170（×1.19）、RMS 0.208；电影负片 0.89–0.97 | 逐片比值与 `tools/eval/t46/loo.json` **逐位相同** |
| T47：Schmidt Tri-X / D-76 四孔径 | 模型 / 实测 −1：+0.4%、+0.2%、−2.2%、+2.2%（门槛 ±15%）；19 级密度最大 \|ΔD\| 0.089；Kodak 48 µm 14.6（官方 17，HC-110） | 逐位相同 |
| T47b：官方 RMS | T-Max 100 1.00、T-Max 400 1.11（D-76，判定）；Tri-X 0.86、Plus-X 1.06、Acros 1.14（只报告） | 逐位相同 |
| T47b：Schmidt 6 种片 | Tri-X 1.04（标定）；T-Max 100 1.14、T-Max 400 0.90、Delta 100 0.91、Delta 400 1.00、Pan F+ 0.70；留出 \|log\| 中位 0.10、RMS 0.18 | 逐位相同 |
| T50：D-76 参照 RMS_D | Tri-X 13.6 / 13.0、T-Max 100 9.0 / 7.9、T-Max 400 11.6 / 12.9、Delta 100 7.9 / 8.7、Delta 400 9.9 / 9.9、Pan F+ 6.2 / 8.8 | 与 T50 表 4.1 相同 |
| T50：显影液（数据库参数） | XTOL：RMS 倍数 ±20% 内 5/6、a_k 6/6。Rodinal（μ 2、σ 1）：Tri-X 1.42 / 1.58、Delta 100 1.50 / 1.57。T-Max 100 偏粗、Delta 400 偏细 | 与 T50 表 4.2 一致（全网格留一：`--full-t50`） |
| T53：颗粒粗细 k | σ√A(48 µm) ÷ k = 0.98–1.00；曲线最大差 0.006 D；涂布银 ∝ k（三维） | 与 `tools/eval/t53/validate.json` 逐位相同 |
| 快速路径 | closed_form / explicit 的像素 σ：0.95–1.08（k = 1） | 新 |
| 真实扫描（记录值） | 见下 | 新 |

**真实扫描的检验**（`validation/realscan.py`；只用记录的测量，不下载）。条件与 RPL 645 Noritsu（约 20 µm/像素）相同：均匀 18% 灰 × 2^EV，Tri-X 用 XTOL，HP5+ 用 D-76。

| | Tri-X（XTOL） | HP5+（D-76） |
|---|---|---|
| 胶片侧：(真实 σ_48 / 真实斜率) ÷ (silvergrain σ_48 / silvergrain 斜率)，码值 68–184 | 1.17–1.30（full）、1.08–1.25（grain）；扫描高斯按 filmlab 的离散核时 1.02–1.13 | 1.51–1.79（full）、1.44–1.57（grain）；离散核 1.31–1.54 |
| filmlab 模型的同一量（T51b） | 0.98–1.12 | 1.28–1.42（HP5+ 偏细是已知弱项） |
| 影调斜率 dCode/dD（码值 68 / 99 / 138 / 184） | RPL 275 / 302 / 302 / 242；silvergrain full 124 / 171 / 231 / 301 | RPL 249 / 267 / 261 / 197；full 123 / 172 / 226 / 298 |
| 同码值 σ_48 真实 / silvergrain（full） | 2.88 / 2.14 / 1.55 / 0.94 | 3.12 / 2.46 / 1.74 / 1.18 |
| 手机尺寸 σ_px 真实 / silvergrain（T52 锚） | 码值 148 / 184：full 1.32 / 0.94，grain 1.52 / 1.07 | 码值 102 / 147 / 181：full 1.91 / 1.49 / 1.13 |
| 谱形 P(f)/P(0.11)，f = 0.30 / 0.48 周/像素 | 真实 1.15 / 1.42；不经扫描 0.95 / 0.97（白）；经 Noritsu 1.32 / 0.99 | 真实 0.89 / 0.99；不经扫描 1.00 / 1.04；经 Noritsu 1.39 / 1.06 |

- **胶片物理与 filmlab 一致**。扫描按同一实现时，Tri-X 的胶片侧比值为 1.02–1.13，filmlab 为 0.98–1.12。
- **同码值上 silvergrain 的颗粒较弱，原因是影调，不是颗粒**。直线反转与照片自己的影调（场景线性 → sRGB），在暗部和中间调的斜率只有 RPL 冲印店影调的 0.45–0.77。码值 184 处斜率相当，比值为 0.94–1.10。没有按观感调任何量。
- **谱形**：
  - 不经扫描时，20 µm 像素上是白噪声，符合物理。
  - 经 Noritsu（连续 MTF）后中频上升，0.48 周/像素回落。真实谱还含扫描仪白噪声底与 JPEG，即 T51 的 N 项，不属于颗粒，没有建模。
  - 离散核实现会在 0.48 周/像素过冲到 2.36，见局限 8。
- **彩色（Portra 400，RPL）**：同码值的比值从暗到亮为 3.7 → 0.55，同样由影调主导。彩色物理与 filmlab 逐位相同；T54 的胶片侧亮部为 0.98。

## 7. 性能（24 MP：6064 × 4040，35 mm，5.9 µm/像素；单进程，4 核机器）

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

## 8. 局限

以下来自 T45–T54 的负责人结论，T55 没有改变。

1. **没有 3200 片**：T-Max P3200、Delta 3200 都没有档案。P3200 的曝光指数不明；Ilford 不公布 RMS。
2. **推冲不可用（T50 结论 3）**：只降低显影阈值，模型方向是错的——颗粒不变、反差不增、灰雾不增。推冲需要显影动力学，所以 `bw_t50_developers.json` 的 push 只作记录，`developers()` 里不列。
3. **HP5+ / FP4 的楔常数（T49b）**：Ilford 光谱图共用一个楔常数 k = 3.01，HP5 与 FP4 不能同时吻合。Delta、Pan F 的光谱是预测，HP5 的红端不可靠。
   - HP5+ 的晶体群由通用常数推出，胶片侧颗粒偏细约 1.35 倍（T51b）。
4. **Pan F+ 偏细**：预测 / 实测 0.70，按实情记录，没有调参。
5. **XTOL 的顺序（T50）**：真实 XTOL 比 D-76 细约 10%，模型给不出来，因为成团只会让颗粒变粗。
6. **FX39**：T47 的留出检验没通过；T50 的成团模型对 T-Max 100 偏粗、Delta 400 偏细。显影液对银丝形态的影响缺少独立数据。显式银粒只适用于 D-76 一类的溶解性显影。
7. **三维晶体的 β 弱**：T46 的 η、β 对 T 颗粒较好，三维晶体（Tri-X、Plus-X、HP5+）的尺寸关系不理想。T47 的 ×6.61 效率修正只由 Schmidt Tri-X 标定。
   - 其他已知偏差：较老或消费级彩负（Gold 200、Superia 100 / Reala、RTP II）偏细约 1.35 倍；Portra 800 偏细。
   - 染料产量 CV 0.3 是量级估计。
8. **冲印店扫描**：光学 σ 与 USM 半径以输出像素计（T51 的单位约定），同一底片两种尺寸的扫描未验证。
   - 谱形检验发现：filmlab 引擎用离散 `gaussian_filter` 实现 σ 0.35 像素的光学，几乎是恒等。silvergrain 改在频域按连续 MTF 实现（第 4 节第 5 步）。
9. **EPFL 800T 的 2.7 倍（T54）**：集中在胶片侧，只出现在 EPFL / CineStill 800T，未归因，没有修正。RPL 彩色暗部（码值 40–130）胶片侧为 1.3–2.2，同样未归因。
10. **感光度不随 k 变**：这是 T53 唯一的非物理假设。k ≥ 2 是假想乳剂（Tri-X k = 3 时涂银 11.6 g/m²）。
11. **照片的假设**：
    - 线性值与景物亮度成正比；
    - 景物在设计光源下照明；
    - grain 模式在速度点以下按速度点的斜率传递。
    - full 模式的扫描是直线反转（系统反差 1），不是冲印店的影调。RPL 的中间调斜率约为它的 1.5 倍，所以同码值比较时 silvergrain 的颗粒显得弱，而胶片侧是一致的（第 6 节）。
12. **渲染器**：
    - 视场边缘会少一圈团（T50 局限 5）；
    - 亚微米像素下，块边界丢失约 5% 的银团（T53 局限 7）；
    - 12 µm 及以上的整帧不受影响。

## 9. 与 filmlab 的关系

- `filmlab/grain.py` 重新导出 `silvergrain.constants / dye / silver` 的全部名字（含内部函数），只保留与 filmlab 胶片状态有关的 `apply` / `apply_silver`，以及进程级的 `SILVER_WORKERS`。
- `filmlab/materials.py` 的通用常数、Tri-X 曲线、黑白档案与显影液数据都从 `silvergrain/data` 读取。
- filmlab 全部已有输出与 T55 之前逐位相同：238 组（180 组与 HEAD 比较，另 58 组显式设定）。
- `tools/export_silvergrain_data.py`（在 filmlab 里）负责把彩色片档案导出成 `data/colour_films.json`。这是唯一的单向导出，本包不读 filmlab。
