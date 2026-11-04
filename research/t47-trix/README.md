# T47 · Tri‑X 银盐颗粒研究交接

这是 SilverGrainLab V48 的冻结快照，供 Claude 在 `handoff/recovered-research-2026-10-23` 接续。原型及测量脚本保存在 [src](src/)，本包独立于 filmlab 正式管线。原始和对比照片、第三方图表及 PDF 均未收录。表中数值来自既有实验，尚未通过片种验证。

## 光学事件

这里的「光学事件」是一个被激活并贡献投影光学深度足迹的有效位点。曝光用泊松捕获和潜影阈值决定激活概率，显影与灰雾进一步决定有效事件率。显影后的多个足迹在光学深度域相加，透射率用 exp(−Στ) 计算，随后在透射率域施加光学模糊和像素积分。事件的数密度乘以单事件积分光学深度面积，是模型的潜在遮光预算。它没有校准到实际卤化银晶体的数量、显影银质量或单位银质量覆盖力，也没有独立建模散射光的收集几何。16 倍试探将位点密度乘 16、单事件积分光学深度面积除以 16、足迹 σ 除以 4，保持总遮光预算，连续高斯足迹的理论峰值光学深度也保持不变。选择 16 时已经知道测量差距，应作为一次单参数标定试探记录。独立的圆盘闭式解检查使用有限 τ 或不透明圆盘，是验证数值采样的合成场。

## 实测数据与读图口径

- [Schmidt 特性曲线，19 级 CSV](data/schmidt-density-19.csv)。取自 [Details TriX](https://www.bernhard-w-schmidt.de/en/einzeln-trix) 中 *Density curve (left) and local contrast (right)* 图左侧，读紫色 Kodak D-76 标记，工艺为原液、20 °C、9 min、EI 400、γ=0.56。扣除采用的片基加灰雾总密度 0.318，再用线性插值将净 D=0.10 设为相对曝光零点。保留图上的原始相对 log10 曝光、总密度、净密度和对齐后的相对曝光。灰阶楔的 D 来自透射密度计。作者没有在 [密度方法页](https://www.bernhard-w-schmidt.de/en/swkurve) 或 Tri‑X 图注交代这组数据的密度计型号、照明和收集角，因此漫射或镜面口径未知。[原曲线图 URL](https://images.squarespace-cdn.com/content/v1/5f550d0c9dad03362307b29c/dd69fffd-0bc5-4ecb-9ede-d737ccfef62d/TriX_SK.png)
- [Schmidt 颗粒度，四档 CSV](data/schmidt-granularity.csv)。取自 [Granularity and grain structure](https://www.bernhard-w-schmidt.de/en/methodgrain) 图 4 从上往下第二条 TriX-D76 的蓝色测量点，横轴为软件方孔边长，纵轴为 RMS × 边长。用图上实读边长还原 RMS，名义 5/10/20/25 µm 对应约 0.07796/0.04877/0.02720/0.02099 D。图中已经采用相邻方孔密度差的 RMS ÷ √2。总 D≈1 的均匀区由密度计测量的灰阶楔选定，颗粒波动来自 250× 显微 RAW 的同一视场，作者给出约 1.5 µm 的分辨率。显微镜的照明角、收集 NA 及漫射或镜面口径未知。该分辨率也没有提供实测 PSF。CSV 的 σ²A 用实读边长计算，按四舍五入后的名义边长计算会略有出入。[原颗粒图 URL](https://images.squarespace-cdn.com/content/v1/5f550d0c9dad03362307b29c/8a19ca03-4616-4dbe-b3f4-503dbbc42d00/Grain_Abb2_1.png)
- [Kodak F-4017 第 8 页曲线点](data/kodak-f4017-p8-d76.csv)。左上 35 mm 400TX，D-76 大罐、20 °C、每分钟搅动，仪器口径标为 diffuse visual。既有数字化覆盖 8 和 10 分钟曲线各四个近似读点，保留绝对 log10(lux·s) 横轴与总 D。尚未数字化完整曲线、9 分钟插值或与 Schmidt 对齐后的残差。[Kodak 原资料](https://business.kodakmoments.com/sites/default/files/files/products/f4017_TriX.pdf#page=8)

## 既有验证

- [单半径闭式解 60 项](data/poisson-pointwise-single-60.csv) 和 [四个超阈值条件](data/poisson-pointwise-single-four-outliers.csv)。每个 τ、净 D 组合使用 16 个独立种子。每组比较 1 个平均透射率、4 个点协方差和 5 个方孔平均透射率方差。z = (样本均值 − 理论值) / 跨种子标准误，阈值为逐项 |z| ≤ 2。超阈值条件如下，协方差的尺度为滞后距离，方差的尺度为方孔边长。

  | τ | 净 D | 指标 | 尺度 µm | z |
  | --- | --- | --- | --- | --- |
  | 0.1 | 0.682 | 方孔平均透射率方差 | 5 | +3.01846 |
  | 0.1 | 1.576 | 透射率点协方差 | 1 | +2.00554 |
  | 1 | 0.682 | 透射率点协方差 | 1 | −2.08227 |
  | 1 | 0.682 | 方孔平均透射率方差 | 5 | −2.05392 |

  [全量 120 项](data/poisson-pointwise-all-120.csv) 还包括完成的混合半径组，总计 7 项超阈值。完整 [原始结果 JSON](results/silver-emulsion/field-check.json) 还含跨扫描尺度的配对检验。203.2 µm 的计算每个种子只有四个方框，适用于数值检查。
- [密度审计摘要](data/density-audit-summary.csv) 和 [19 级残差](data/density-audit-by-level.csv)。基线和 16 倍变体（`fine`）预算均为 4.05，净 D 上限约 1.0196。实测最高曝光档净 D=1.5763，25 µm 实测 RMS 为 0.02099 D。16 倍变体的线性化 RMS 从基线 0.10054 降到 0.02656 D，仍高约 26.6%。组合（`both`）的 25 µm 显式抽样值为 0.02748 D，高约 30.9%。组合加入慢响应群组后，预算变为 8.10，最高曝光档净 D=2.0351，残差 +0.4588，全曲线最大绝对误差 0.5856，超过此次密度审计的 0.10 门槛。把原预算的一半分给慢响应群组（`redistribute`），上限仍为 1.0196。原始因子实验的门槛和完整 RMS 另见 [measurements.json](results/silver-emulsion/measurements.json)，不要与后续密度审计门槛混用。

## 复现入口

在本目录运行，依赖 Python 3、NumPy、SciPy、Pillow 及 macOS 的 `/usr/bin/clang`。原型文件保持原样，圆盘脚本按原实现编译动态库。完整圆盘运行会生成可续跑的本地检查点，耗时约 20 分钟（原运行机器）。结果文件会被覆盖，复算时请保留交接快照。

```sh
python -m pip install numpy scipy pillow
python src/silver_field_check.py
python src/silver_jensen_check.py
```

下列入口用交接包中的数字读点重跑密度审计，无需取得或重新分发图表文件。它只替换读图输入，密度计算仍调用原型代码。

```sh
python - <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
import silver_density_audit as audit

reference = json.loads(Path("data/schmidt-reference.json").read_text())
points = np.array([[row["chart_relative_log10_exposure"], row["gross_D"]]
                   for row in reference["density"]])
audit.read_points = lambda: (points, reference["apertures"], reference["figure_sha256"])
audit.main()
PY
```

Schmidt 与 Kodak 的原始读图脚本仍在 `src/`。它们依赖本地 `sources/silver-validation/` 下相同的图表栅格，校验散列见数字化 JSON。原型照片渲染入口需要自行提供同名整图输入，位置见 `src/silver_process.py`。实物采样计划见 [validation-protocol.md](validation-protocol.md)。
