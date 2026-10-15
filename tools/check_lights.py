"""检验人造光源光谱: ColorChecker 真实反射率 × 实测灯光光谱 → 胶片 (Vision3 500T) 三层曝光,
比较 (a) 把整个像素当作日光下的平滑光谱 与 (b) 反射率 × 真实灯光光谱 两种 S0 重建。"""
import sys
from pathlib import Path
import numpy as np
import colour

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filmlab import lights, spectral  # noqa: E402
from filmlab.materials import emulsion  # noqa: E402

shape = colour.SpectralShape(380, 780, 5)
cc = colour.SDS_COLOURCHECKERS["ISO 17321-1"]
names = list(cc.keys())
R = np.array([cc[n].copy().align(shape).values for n in names])
sens = 10 ** emulsion("kodak_vision3_500t").stock.log_sensitivity
D55 = spectral.illuminant("D55")
old_f = spectral.spectral_response(D55[:, None] * sens)
gi = names.index("neutral 5 (.70 D)") if "neutral 5 (.70 D)" in names else 21

print("指标 1: 中灰的三层曝光比例误差 (log10, R/G 与 B/G) —— 决定冲洗扫描后的偏色")
print("指标 2: 各色块相对误差 (以中灰曝光为单位, 平均/最大)")
print(f"{'光源':34s} {'旧: 平滑光谱':>26s} {'除光源+重建光谱':>28s} {'光源下的线性映射':>28s}")
for c in lights.classes():
    for name, E in c.members[:3]:
        rgb = lights.camera_rgb(R * E[None, :])          # 理想相机 (CIE 观察者), 日光白平衡
        scale = 0.184 / (rgb[gi] @ np.array([0.2126, 0.7152, 0.0722]))
        rgb = rgb * scale
        H_true = (R * E[None, :]) @ sens
        new_f = lights.response_under(E, sens)
        lin_f = lights.linear_response(E, sens)
        out = []
        for f in (old_f, new_f, lin_f):
            H = np.clip(f(rgb), 1e-9, None)
            H = H * (H_true[gi, 1] / H[gi, 1])           # 只对齐整体曝光 (绿层), 保留色偏
            cast = np.log10(H[gi] / H[gi, 1]) - np.log10(H_true[gi] / H_true[gi, 1])
            rel = np.abs(H - H_true) / H_true[gi]
            out.append(f"偏色 R{cast[0]:+.2f} B{cast[2]:+.2f} | {rel.mean():.3f}/{rel.max():.2f}")
        print(f"{c.name + ' · ' + name:34s} {out[0]:>30s} {out[1]:>30s} {out[2]:>30s}")
