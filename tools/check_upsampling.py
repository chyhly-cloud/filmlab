"""检验 S0 光谱重建: 用 ColorChecker 的真实光谱反射率, 在日光与钨丝灯下
比较"真实光谱 → 乳剂曝光"与"相机 RGB → 重建光谱 → 乳剂曝光"的差异。"""
import sys
from pathlib import Path
import numpy as np
import colour

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filmlab import spectral  # noqa: E402
from filmlab.materials import emulsion  # noqa: E402

d = np.load(Path(__file__).resolve().parents[1] / "filmlab/data/upsampling_xyY.npz")
X, Y = np.meshgrid(d["x"], d["y"], indexing="ij")
VALID = d["valid"][:, :, int(np.argmin(np.abs(d["lum"] - 0.05)))]   # 亮度 0.05 (暗色) 时的可实现性


def inside_tri(x, y, tri):
    (x1, y1), (x2, y2), (x3, y3) = tri
    den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / den
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / den
    return (a >= 0) & (b >= 0) & (1 - a - b >= 0)


for name, tri in [("sRGB", [(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)]),
                  ("Display P3", [(0.68, 0.32), (0.265, 0.69), (0.15, 0.06)]),
                  ("Rec.2020", [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)])]:
    print(f"{name:10s} 色域内格点直接求解成功 (亮度 0.05): {VALID[inside_tri(X, Y, tri)].mean():.1%}")

shape = colour.SpectralShape(380, 780, 5)
cc = colour.SDS_COLOURCHECKERS["ISO 17321-1"]
names = list(cc.keys())
R = np.array([cc[n].copy().align(shape).values for n in names])
sens = 10 ** emulsion("kodak_vision3_500t").stock.log_sensitivity
D55, A, cmf = spectral.illuminant("D55"), spectral.illuminant("A"), spectral.cmfs()
M = colour.models.RGB_COLOURSPACE_sRGB.matrix_XYZ_to_RGB
d65 = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])


def cam_rgb(refl, light):   # 理想相机, 日光白平衡
    xyz, w = (refl * light) @ cmf, D55 @ cmf
    cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(w / w[1], d65, "Bradford")
    return (xyz / w[1]) @ (M @ cat).T


w = D55[:, None] * sens
new = spectral.spectral_response(w)
old = spectral.rgb_basis().T @ w
for light, ln in [(D55, "日光"), (A, "钨丝灯")]:
    rgb, Ht = cam_rgb(R, light), (R * light) @ sens
    for lab, H in [("旧 基光谱", np.clip(rgb @ old, 1e-9, None)), ("新 Jakob", np.clip(new(rgb), 1e-9, None))]:
        H = H * Ht[18, 1] / H[18, 1]
        rel = np.abs(H - Ht) / Ht[18]          # 以中灰曝光为单位的绝对误差
        worst = names[int(np.argmax(rel.max(1)))]
        print(f"{ln:4s} {lab:8s} 平均误差 {rel.mean():.3f}  最大 {rel.max():.3f} ({worst})  "
              f"红 {rel[names.index('red')].max():.3f}  橙 {rel[names.index('orange')].max():.3f}")
