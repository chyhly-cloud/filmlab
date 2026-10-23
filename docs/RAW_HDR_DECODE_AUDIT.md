# RAW高光线性解码的独立数值审查

## 结论与候选边界

确认了传感器未饱和信息在默认解码时就可能丢失，早于显影或扫描：
LibRaw默认`HighlightMode.Clip`在白平衡后的内部范围截断，随后XYZ16还有独立的
上端和零端裁切。这不能仅靠最终调暗撤销，也不能全部归因于8位扫描量化。

独立数字输入验证通过的完整候选为：**neutral camera16输出，固定低区数字探针标定，
然后在float中完成颜色转换**。只在eval工具中提供显式helper，不改`rawio`、current、next。
候选不推断真正饱和的CFA值，没有更改显影、扫描或光晕参数。后续固定成像链复验单列如下。
这证明有可修正的数值损失，不证明修复了真实800T橙红光晕的全部差异。

## 已知输入与白电平混杂

使用G81 `P1480512.RW2`的相机元数据，只在内存中将可见CFA写入已知常数；不修改RAW文件。
四个低输入为：`(.0625,.0625,.0625)`，以及分别将R、G、B升到`.125`的三个输入。
按实际写入后的整数码计算camera输入，以四低probe求固定3×3矩阵和截距。
预声明验证为`.125`灰、`(.8,.2,.2)`红强、`(.2,.8,.2)`绿强、`.8`灰。
这些验证值不参与标定。每模式再检查解析XYZ最大角点：把1替换为低于白电平16码的
安全值，0替换为对应小正值。所有写入值均低于白电平；最高安全码4079也低于此文件
已独立识别的传感器堆积阈值4087.092。常量probe的人工堆积不能被当成真实饱和证据。

比较Clip/Ignore与`bright=1/.125`四模式，固定daylight WB、半尺寸、线性gamma、
无自动提亮、16位XYZ。第一轮保留默认`adjust_maximum_thr=.75`；第二轮显式设0，
固定元数据白电平以隔离场景最大值重估。没有逐probe重归一。

第一轮Ignore的`.8`输入出现约25%额外增益，而解析矩阵预测XYZ仍低于1；冻结白电平
后消失，证明这一常量probe条件下存在自适应最大值混杂。真实原帧最大码4097、
元数据white4095，与全帧`.8`常量的最大值不同，不能把该probe混杂直接说成实拍中的
同等曝光误差。本报告保留两轮结果，不把所有高区损失归于单一WB裁切。

## 冻结白电平后的判别结果

| 解码模式 | 四个高验证中最大XYZ误差（16位输出码） | 主要结果 |
| --- | ---: | --- |
| Clip，bright1 | 47970 | 红强发生内部截断，输出本身未到65535仍严重非线性 |
| Ignore，bright1 | 21.0 | 正XYZ验证区保留线性，但XYZ负端仍会归零 |
| Clip，bright.125 | 6015 | 仅降低最终输出范围不能撤销内部截断 |
| Ignore，bright.125 | 15.4 | 正XYZ区保留线性，但精度较低，仍不能保留负XYZ |

数值合同预声明容差为64个16位输出码，涵盖小步长四probe矩阵估计及整数处理误差；
这是保守数值检查门限，不是物理模型拟合常数。没有用最终照片挑选模式或gain。

低区矩阵按row-vector camera约定，XYZ立方体上限为各输出列正系数之和加截距。
Clip1的上限为`[2.0208,1.6929,1.9402]`，Ignore1为`[.7168,.6008,.6885]`。
后者上端安全并不等于整个camera立方体可逆：低矩阵有负系数，
Ignore1的XYZ下限约`[-.00465,-.14557,-.11551]`。解析最坏X角点包含负Z，
XYZ16会强制归零。它只能作为**预测XYZ非负子域内**的条件性候选。

Ignore1到原Clip低区数值范围的公共增益只用四低probe标定，为2.818755；
按三个camera输入行独立求得`[2.818769,2.818461,2.818867]`，差别符合整数精度，
与已记录WB比值2.819017相符，但没有用WB理论值替代实际数字标定。
极小矩阵交叉系数的单独比值会放大量化误差，不能把它当精确跨通道gain。
`bright=.125`的反增益同样只从低probe标定；无额外照片曝光补偿。

## 最终eval-only数值合同

```python
from tools.eval.raw_hdr_decode_audit import decode_hdr_float
decoded = decode_hdr_float(raw_path, digital_probe_calibration)
# decoded: camera, XYZ, linear_RGB；全为float，不在颜色转换后裁切
```

固定设置为`ColorSpace.raw`、`user_wb=[1,1,1,1]`、`HighlightMode.Ignore`、
`adjust_maximum_thr=0`、`bright=1`、线性gamma、无auto-bright、half-size、16位。
四个低probe标定camera16的固定矩阵C和截距o；先做
`camera = (camera16/65535 - o) @ inv(C)`，再乘此前固定低区daylight颜色矩阵。
这保留原低区的数值曝光尺度，并在float中保存大于1或负的颜色坐标。
负XYZ只表示该线性矩阵的数值结果，不被宣称为负光子辐射。

G81的C对角约1.000015、截距约−1/65535。四个验证及两个安全角点的camera反演
最大误差1.03e−5，输出误差不超过0.68个16位码。包括会使XYZ16损失负值的角点，
float仍保留例如`Z≈−.188`；红强输入保留`X≈1.388`。

Sony用户RAW只做最终方案的最小跨相机核查：重新用数字低probe独立标定该相机颜色矩阵，
另用四低probe标定其neutral camera16传输，只验证红强和灰高两个输入。
两项输出误差均1.68个16位码，camera反演误差2.56e−5；没有渲染或上传用户照片。
不把G81矩阵套到Sony，也不宣称所有相机都已通过。

helper核查black/white、CFA排列、相机矩阵和daylight WB与数字标定一致；不匹配即拒绝。
新增相机、WB或LibRaw条件需要自己的数字标定。16位量化和低于black的信号处理仍存在，
此次验证覆盖正的未饱和输入及实际低信号对照，不承诺恢复传感器噪声负值或真正饱和值。

## 实际两灯：仅完全可测区域

以独立显式body mask中所有RGB均未饱和的68/47个单元作数值验收，不选输出较好的子集。
参照为Bayer实测camera乘同一固定低矩阵，而不是胶片/扫描真值。

| 项目 | 近顶灯 | 中顶灯 |
| --- | ---: | ---: |
| 全RGB可测body单元 | 68 | 47 |
| 旧XYZ16到达上限单元 | 4 | 2 |
| 旧XYZ相对Bayer浮点预测最大绝对误差 | 1.0524 | 1.0531 |
| 新XYZ相对同一预测最大绝对误差 | 2.94e−5 | 2.34e−5 |
| 新camera相对Bayer测量最大绝对误差 | 1.93e−5 | 1.54e−5 |
| 旧线性RGB峰值 | 1.234 | 1.199 |
| 新浮点线性RGB峰值 | 3.719 | 3.783 |

旧路径逆低矩阵后的红camera代理最多约.373/.367，而实际可测红通道接近.997；
这里既有内部WB范围损失，也有后续XYZ16裁切，不能只数最终65535就判定全部信息保留。
同时，大多数body的旧新差异很小：两灯旧XYZ绝对误差中位数仅2.04e−5/1.24e−5。
这是局部严重损失，不是整个灯旁都被同等破坏。

同一裁剪中预声明低信号控制：RGB全可测、camera各通道>.005且最大<.1、预测XYZ为正。
两灯分别25047/25224单元，旧新XYZ对同一浮点预测最大误差均约3.1e−5以内。
新合同没有通过整体重调亮度来修正少量高信号样本。

本轮没有验收真正饱和核心，没有更改推断的源谱/幅度，没有选择更好的最终胶片颜色。
高光解码数值合同与最终800T光晕外观仍是不同验收问题。

## 最终helper经过固定胶片后端的复验

另运行`tools/eval/raw_hdr_film_replay.py`，直接读取上述最终helper的实际裁剪，
只替换同一115个完全可测body单元。沿用[既有body实验](MEASURED_BODY_DECODE_CANDIDATE.md)
捕获的原局部照明权重、实际成员谱和测光，核心逆解、部分边缘minY及显影/扫描全部固定。
没有用目标照片选择任何解码、色彩或输出参数。

| 灯/实际HDR解码候选 | 50–300µm留出中位ΔE00 | 300–800µm外缘中位ΔE00 |
|---|---:|---:|
| 近顶灯 | 4.84499 | 5.91546 |
| 中顶灯 | 4.08787 | 4.02187 |

与此前直接Bayer浮点body候选相比，最终RGB最大差约.00012/.00030，采样Lab分量最大差约.00044/.00669。
两种数值实现都支持同一个结论：本轮这些全测body的输入损失可独立修正，但对目前强化核心候选的
成片影响很小，没有恢复真实橙红过渡，不能宣称光晕主因已定位或修好。
真正饱和核心的新恢复数仍为0，也未将此小范围实验扩展成全幅HDR默认管线。

结果、曝光/染料/扫描密度数组和对比图在`/workspace/filmlab-audit/raw-hdr-film-replay/`。
复现：`python tools/eval/raw_hdr_film_replay.py --hdr-crops /workspace/filmlab-audit/raw-hdr-decode-audit/actual`。

## 复现及记录

```bash
PY=/workspace/filmlab-env/bin/python
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY tools/eval/raw_hdr_decode_audit.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY tools/eval/raw_hdr_decode_audit.py --freeze-white --out /workspace/filmlab-audit/raw-hdr-decode-fixed-white
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY tools/eval/raw_hdr_decode_audit.py --neutral-camera-only --real-lamp-controls --out /workspace/filmlab-audit/raw-hdr-decode-fixed-white
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY tools/eval/raw_hdr_decode_audit.py --neutral-camera-only --minimal-validation --raw /workspace/filmlab-data/user/DSC03906/DSC03906.ARW --out /workspace/filmlab-audit/raw-hdr-decode-sony
$PY -m pytest tests/test_raw_hdr_decode_audit.py -q
```

结果在仓库外`/workspace/filmlab-audit/`：
`raw-hdr-decode/results.json`保留默认最大值调整结果；
`raw-hdr-decode-fixed-white/results.json`保留四模式冻结白电平结果；
其`neutral-camera.json`和`actual-lamp-controls.json`保留最终候选及真实数值控制；
`raw-hdr-decode-sony/neutral-camera.json`保留跨相机核查。
两灯供固定后端验证的精确helper裁剪另保存于
`raw-hdr-decode-audit/actual/{ceiling_lamp,middle_ceiling}.npz`，key为`hdr_rgb/source_bounds`。
以后脚本的真实控制步骤也会在所选out下保存`actual/`裁剪。

两项helper测试通过，检查设置固定、HDR/负颜色值保留及错误WB标定拒绝。
脚本编译与差异检查通过；生产API与默认代际未修改。
