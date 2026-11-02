"""T33 A (第 3 节): 真实相机 (colour 自带实测光谱感光度) 看发光饱和色: 解码后的 sRGB 与 Z 和 CIE 观察者 (= 模型的假设) 比较。
若真实相机对 > 600 nm 的光给出明显的 B / Z, 那部分不是短波光, 胶片蓝层不会收到。"""
import sys, numpy as np, colour, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]
import camera_metamerism as cm
import t33_blue_map as a
from filmlab import lights
TO = a.TO_XYZ
G = np.full(3, 0.184)
em = dict(a.EMIT)
for n, (E, _) in a.red_like(np.array([1.0, 0.07, 0.22]) * 0.184).items():
    em["(RAW 一致) " + n] = E
print("CIE 观察者 (模型) | 各实测相机: 解码 sRGB (R=1) 与 Z/R; 真值 film B/R (R、B 都以 3200K 中灰为 1)")
L = lights._blackbody(3200); sc = 1 / (lights.camera_rgb(L) @ np.array([.2126, .7152, .0722])); Hg = (L * sc) @ a.S
cams = list(colour.MSDS_CAMERA_SENSITIVITIES)
fs = {c: cm.camera(c) for c in cams}
zg = ((lights.light_rgb(L)) @ TO)[2]; rg = lights.light_rgb(L)[0]
for n, E in em.items():
    x = lights.camera_rgb(E)
    tb = (E @ a.S[:, 2] / Hg[2]) / (E @ a.S[:, 0] / Hg[0])
    s = f"{n[:44]:44s} 真值 B/R {tb:.1e} | CIE sRGB {np.round(x / x[0], 3)} Z/R {((x @ TO)[2] / zg) / (x[0] / rg):.3f}"
    for c in cams:
        y = fs[c](E)
        s += f" | {c[:14]} {np.round(y / y[0], 3)} Z/R {((np.clip(y @ TO, 0, None))[2] / zg) / (y[0] / rg):.3f}"
    print(s)
print("\n相机原始通道 (D65 白平衡, 未经矩阵) R=1; 对照: EPFL G81 红灯箱 (R1-03-4) 原始通道 [1, 0.195, 0.103] (t33_raw_blue.py)")
from filmlab import spectral
for c in cams:
    S = np.clip(colour.MSDS_CAMERA_SENSITIVITIES[c].copy().align(cm.SH, extrapolator_kwargs={"method": "Constant", "left": 0, "right": 0}).values, 0, None)
    wb = spectral.illuminant("D65") @ S
    print(f"  {c}: " + " | ".join(f"{n} {np.round((a.EMIT[n] @ S / wb) / (a.EMIT[n] @ S / wb)[0], 3).tolist()}" for n in ("红 LED 630", "红霓虹 (Ne)", "琥珀 LED 590")))
