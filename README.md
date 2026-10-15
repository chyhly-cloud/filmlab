# filmlab: 基于光化学过程的色彩科学系统

**颜色不是调出来的，是按照胶片的感光、显影、冲洗、印放、观看过程计算出来的。**

换一种胶片 = 换一组材料参数；换一种冲洗 = 换一串化学步骤；计算规则永远是同一套。
所以"交叉冲洗""漂白旁路""推冲""CineStill"在这里都不是特效，只是把某种材料放进了某个流程。

---

## 1. 七个物理阶段

| 阶段 | 物理量 | 代码 |
|---|---|---|
| **S0 场景** | 光谱 (380–780nm)。RAW → 场景线性光 → 基光谱重建。可扩展近红外估算 | `rawio.py` `spectral.py` |
| **S1 曝光** | 每层乳剂 log H = ∫ 场景 × 光源 × 镜头滤镜 × 光谱感光度。按包装 ISO 测光；乳剂散射、光晕（取决于防光晕层 / remjet）；过期胶片的灰雾曝光与感光度损失 | `film.py` `engine.py` |
| **S2 冲洗** | 化学步骤序列作用在 卤化银 / 银 / 染料 / 成色剂 上 | `chemistry.py` |
| **S3 底片** | 光谱透射率 = 染料光谱 + 银 (中性) + 残留卤化银 + 片基 / 橙色遮罩 (+ 未去除的 remjet) | `engine.py` |
| **S4 印放** | 负像：放大机 (卤素灯 + KG3 + Durst 分色滤镜) → 相纸三层感光，冲印店按灰卡定**印放光号**；正像：直接观看 | `engine.py` |
| **S5 相纸** | 相纸自己也走化学流程 (RA-4 / 黑白)，得到反射 (拷贝片为透射) 光谱 | `chemistry.py` |
| **S6 观看** | 观看光源 (D50 / 放映氙灯) × CIE 1931 观察者 → XYZ → 显示器 | `spectral.py` |

正像还是负像不由胶片类型决定，而是由化学结果决定 (`Engine.polarity`)。

## 2. 材料库 (`materials.py`)

统一单位：每层的卤化银、银、染料都用"染料当量密度"表示。
datasheet 的特性曲线 = "原生工艺标准冲洗后的染料密度"，由此反推乳剂在其**原生显影液**中的成像银曲线：

| 设计 | 原生工艺 | 成像银 S_nom | 卤化银总量 | 成色剂 |
|---|---|---|---|---|
| 彩色负片 | C-41 | = 染料曲线 | 1.4×Dmax ⚠估计 | 1.4×Dmax ⚠估计 |
| 电影负片 | ECN-2 | = 染料曲线, 有 remjet | 同上 | 同上 |
| 彩色反转片 | E-6 | = Dmax − 染料曲线 (第一显影的负像银) | = Dmax | = Dmax |
| 黑白 | D-76 | = 银密度曲线 | 1.4×Dmax | 无 |
| 相纸 / 拷贝片 | RA-4 / ECN-2 | = 染料曲线 | — | — |

材料变体：`without_remjet()` (CineStill)、`aged(years)` (过期)、`bw_film(pan/ortho)`、`bw_paper(grade)`。

## 3. 冲洗化学 (`chemistry.py`)

| 步骤 | 作用 |
|---|---|
| `Develop(显影液, 时间, 温度)` | 只显影带潜影的颗粒 (反转浴后则显影全部剩余卤化银)。黑白显影液只生成银；彩色显影液同时生成等量染料，受成色剂限制；DIR 抑制剂只在彩色显影时释放 |
| `Reversal()` | E-6 反转浴：化学灰化剩余卤化银 |
| `Bleach(效率)` | 银 → 卤化银。效率 0 = 漂白旁路 |
| `Fix()` | 溶去卤化银 |
| `RemjetRemoval()` | ECN-2 预浴 |

工艺：C-41、ECN-2、E-6、RA-4、D-76、Rodinal 1+50、HC-110 B、Xtol。`Process.push(n)`、`Process.with_bleach(e)`。

**显影动力学**（τ = 时间比 × 2^(温差/6°C) × 显影液活性比）：
- **彩色显影**：用 Kodak Portra 800 标准 / 推一档 / 推两档实测曲线拟合，密度绕中间调按 τ^1.10 旋转；拟合残差 0.040D（不建模为 0.11–0.17D）。暗部几乎不提速，与实测一致。
- **黑白显影 (含 E-6 第一显影)**：曲线沿曝光轴平移并略增反差。按 Kodak E-6 推冲定义标定（第一显影 +2 分钟 = 一档），在三种反转片上误差 < 0.06 档。

由化学过程自然得到的结果（测试里都有）：
- 正片走 C-41 → 无橙色遮罩的高反差负像 → 印放后青色高光、品红暗部
- 负片走 E-6 → 带橙色遮罩的正像，密且低反差
- 黑白胶片走 C-41 → **空白片**（无成色剂 → 无染料，银又被漂白）
- 漂白旁路 → 银留在片上 → 降饱和、高反差
- 灯光片在日光下拍 → 冲印店定光后基本校正，残留层间交叉偏色

## 4. 标准与校准

| 项目 | 规则 |
|---|---|
| 测光 | 按包装 ISO。负片：中灰在感光点 (灰雾+0.1D) 之上 0.9 logH；反转片：正像 D≈1.0 |
| 印放光号 | 1 光号 = 0.025 logE；标准底片 = 25-25-25。冲印店按**实际拍摄条件下**的灰卡定光号，并在 `describe()` 中报告 |
| 反转片 | 没有冲印店校正。只在标准 E-6、参考光源下做一次**材料校准**（抵消数据误差），之后的一切偏差（交叉冲洗、错误光源、推冲）都保留 |
| 密度计 | Status M (负片) / Status A (正片、相纸)，⚠ 按公开峰值波长的窄带近似（ISO 5-3 原表为付费标准） |
| 颗粒 | RMS 颗粒度 (48µm 孔径) 按 Selwyn 定律换算到像素，⚠ 按 ISO 估算；显影液决定颗粒倍数 |

## 5. 使用

```bash
pip install -r requirements.txt
sh samples/download.sh            # 下载测试 RAW
python -m filmlab --list
python -m filmlab 照片.NEF -p kodak_portra_400 -o out.jpg
python -m filmlab 照片.NEF -p fujifilm_velvia_100 --process C-41        # 交叉冲洗
python -m filmlab 照片.NEF -p kodak_portra_800 --push 2                  # 推两档 (自动按 EI 3200 曝光)
python -m filmlab 照片.NEF -p kodak_vision3_250d --bleach 0              # 漂白旁路
python -m filmlab 照片.NEF -p kodak_portra_400 --points 0 -2 1           # 手动改印放光号
python -m filmlab 照片.NEF -p bw_pan_red --process BW-Rodinal
python make_sheets.py 照片.NEF out/                                      # 所有配方对比
python -m pytest tests -q                                                 # 系统验证 (89 项)
```

每次渲染都会打印完整链条：

```
胶片: Kodak Portra 800  (negative, 原生工艺 C-41)
冲洗: C-41 push 2
  · C-41 彩色显影 (CD-4): 4.25 分钟 @ 37.8°C, 显影程度 τ=1.31
  · 漂白: 100% 的银变回卤化银
  · 定影: 溶去卤化银
成像: 负像
印放: Kodak Professional Portra Endura, 光号 R20 G21 B20 (标准底片 = 25-25-25)
```

`filmlab.densitometry.report(Params(...))` 会像冲洗厂的控制条一样，打印一条虚拟灰阶楔的分析密度与密度计读数。

## 6. 验证 (`tests/test_system.py`)

- 20 种胶片在原生工艺下的灰阶楔重现 datasheet 特性曲线（误差 < 0.02D）
- 推冲动力学 vs Kodak 实测；E-6 推冲 = 一档
- 极性由化学决定；黑白进 C-41 为空白；漂白旁路保留银
- 中灰中性、影调单调、负片宽容度、反转片曝光补偿、光晕偏红、密度计读到染料多余吸收

## 7. 估计值与已知局限 (⚠)

1. **负片的卤化银 / 成色剂储量** (1.4×Dmax)：只影响非原生工艺（如负片走 E-6）与推冲上限。负片走 E-6 的结果偏暗偏橙，可信度有限。
2. **显影液活性比**：C-41 / ECN-2 / E-6 第一显影之间的比值是估计值，只影响交叉冲洗。
3. **黑白胶片与相纸**：没有黑白 datasheet 数字化数据，由彩色数据推导。
4. **Status M/A**、颗粒强度、光晕强度、过期参数：近似值。
5. **光谱重建**：基光谱只覆盖 sRGB 色域；还没有使用相机光谱敏感度（ProRAW / Sony 输入层待做）。
6. **扫描路线**（Frontier / Noritsu）尚未实现；目前负片只能印放。
7. **实时引擎**：参考实现为逐像素光谱计算。与空间无关的部分可烘焙成 3D LUT，颗粒 / 光晕 / DIR 扩散用 GPU 实现。

## 数据与许可
- 胶片 / 相纸光谱数据：[spektrafilm](https://github.com/andreavolpato/spektrafilm)（Andrea Volpato，datasheet 数字化），CC BY-SA 4.0。KG3、Durst 滤镜光谱同源。
- 方法参考 spektrafilm（DIR 曲线反推、印放链路），按其要求本项目采用 GPLv3。
- 冲洗时间：Kodak 公开工艺资料（C-41 3:15 @ 37.8°C，推冲每档 +30 秒；E-6 第一显影 6:00）。
- colour-science；测试 RAW 来自 raw.pixls.us。
