# 多半径逆实验的测量孔径隔离

本轮实际训练/留出划分在固定 Gaussian 与双线性测量算子下通过严格 native 支持隔离检查。
两灯均直接读取 `joint-radius-source-inverse/<name>_operator.npz`，与固定 H 重建逐权重比较，最大差为 **0**。
全 train（near+outer）与全 heldout 的非零 native 列交集均为 **0**。
这验证有限测量算子的空间隔离；不意味着同画幅扫描、真实镜头 PSF、噪声或物理传播统计独立，
也不会使已经多次检查过的同画幅留出区成为新的盲测数据。

## 检查与护带

25 µm Gaussian 对应 `sigma=3.7888888889` scan px；4σ截断取整半宽为15px。
实际有限足迹是 **31×31 方形**，每轴约99 µm，角点径向约140 µm，不是半径100 µm的圆孔。
再经固定双线性 H，native 采样中心相对目标像素的最大每轴位移约2.58 scan px。

因此使用纯几何规则：

```
axis_guard = round(4*sigma) + ceil(max projected bilinear-neighbour displacement) + 1
           = 15 + 3 + 1 = 19 output pixels
```

本工具遍历全部有效双线性行，得到近/中灯最大位移分别2.57948/2.58096px。
这一护带未读取 RGB 或拟合误差。它比原16px多3px（约19.8名义 µm）每轴。
实际 sparse P 去除显式零项后，**保留所有正权重，包括极小的截断尾部**，直接计算列交集，
不只检查 target 中心距离、角度间隙或 target 足迹。

| 实际19px划分 | near train / heldout | outer train / heldout | 全train native列 | 全heldout native列 | 交集 |
|---|---:|---:|---:|---:|---:|
| 近灯 | 160 / 160 | 160 / 160 | 6218 | 5219 | 0 |
| 中灯 | 143 / 140 | 160 / 160 | 4704 | 4663 | 0 |

near→near、outer→outer、near train→outer heldout、outer train→near heldout四种交集也全部为0。
中灯护带后的真实有效点数少于160，不补点或复用点以凑数。

另按相同固定规则、格点和确定性采样重建了本轮最初的16px**几何对照**，没有运行拟合：
近灯有30个共享native列，涉及6个train、18个heldout孔径，共同列上单行最大权重
分别 `4.54e−5` 与 `8.31e−5`；中灯原本就是0。
这是严格隔离声明的一个小缺口，不应描述为大量目标泄漏，也不应据此否定或重算无关历史实验。

## 半径范围是测量中心标签

原 `distance_um` 是到既有core掩膜外侧的名义距离，不是新的材料或真实光子路径测量。
下表通过固定H将该标签场插值到native采样中心，再按每行实际P非零权重统计跨300边界。
跨界计数包含极小正尾部；平均权重按该组全部测量行平均，不能把跨界行数当作跨界能量比例。

| 灯 / 分组 | 实际中心范围，名义µm | 跨300边界孔径 / 总数 | 跨界平均权重 | 单孔径最大跨界权重 |
|---|---:|---:|---:|---:|
| 近 near train | 52.8–298.4 | 108 / 160 | 6.28% | 51.64% |
| 近 near heldout | 112.2–299.2 | 147 / 160 | 9.50% | 51.60% |
| 近 outer train | 302.7–799.7 | 27 / 160 | 1.50% | 43.00% |
| 近 outer heldout | 312.9–798.0 | 30 / 160 | 0.69% | 28.00% |
| 中 near train | 126.6–299.5 | 137 / 143 | 11.11% | 55.14% |
| 中 near heldout | 136.2–298.8 | 134 / 140 | 10.52% | 52.37% |
| 中 outer train | 301.7–798.7 | 26 / 160 | 1.14% | 44.62% |
| 中 outer heldout | 306.0–799.5 | 26 / 160 | 0.81% | 37.72% |

near有限支持的最高名义距离约442–460µm，部分支持进入core距离0；
outer支持整体约150–951µm。因此应称“中心位于50–300 / 300–800µm的有限孔径测量”，
不能称全部足迹被限制在相应半径带，也不能声称有效near样本覆盖所有50–300范围。
特别是中灯没有50–126µm的实际near训练中心；近灯near train与heldout的半径分布也不完全相同。
JSON保留每组中心距离直方图、每行支持范围和跨界权重，避免只看名义zone名称。

near与outer**训练组之间**共享1165/838个native列。
二者作为共同目标用于联合拟合是允许的，但不是独立的径向实验，也不能将两组误差当独立证据计数。
本轮不强行清空300µm边界附近的样本，不因训练/留出误差重选护带。
若未来必须要求径向两组完整足迹独立，应直接按P在距离场上的支持最小/最大值作预声明筛选，
而不是套用每轴100µm的圆形近似；当前实验无需增加这项要求。

## 可复用工具与产物

[measurement_aperture_audit.py](../tools/eval/measurement_aperture_audit.py) 提供：

- `native_support_overlap(P, train_rows, heldout_rows)`：非零native列交集与共同权重。
- `geometric_axis_guard(shape, H, sigma)`：只用固定几何的保守轴向护带。
- `radial_footprint_summary(...)`：中心标签与完整有限支持的区别。
- `audit_sample_file(...)`：支持本轮groups编码与实际存储P的一致性复查。

独立结果在 `/workspace/filmlab-audit/measurement-aperture-audit/`：
`<name>.json`为本轮19px实际划分，`<name>_old16_geometry_only.json`及对应split仅为旧设计几何对照。
没有读取目标RGB、拟合任何参数、修改实际样本、材料或生产管线。
5个独立手算/坐标反例测试通过，包括“target足迹不重叠但native双线性列重叠”的反例。
