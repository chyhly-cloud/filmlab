"""ASC (American Cinematographer, "Film Capture for LED Wall Shoots") 色块对比: Kodak 5219 vs Alexa 35 vs Ideal。

真实 5219: FotoKem ECN-2, 4K 扫描 → LogC4 → Rec709 LUT; Alexa 35 经同一管线 (对照)。
本系统: Ohta 色卡光谱 × 光源 → 色度学相机 RGB → 500T 模拟 (灰卡定光)。指标同 chart_500t.py, 参考为 Ideal 色块。
用法: python3 tools/eval/asc_5219.py 数据目录 (含 img/asc_*.jpg)
"""
import sys, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import chart_500t as c
from PIL import Image
from filmlab import spectral, lights
from filmlab.engine import Engine, Params
from filmlab.io import srgb_encode
D = sys.argv[1]
A = np.asarray(Image.open(f"{D}/img/asc_ledwall_macbeth_chips_5207_250D_and_5219_500T.jpg")).astype(float)
B = np.asarray(Image.open(f"{D}/img/asc_ledwall_macbeth_chips_alexa35_control_and_5203_50D.jpg")).astype(float)
def rel(name, V, ref):
    m = np.hypot(*V.T) / np.hypot(*ref.T)
    a = np.degrees(np.angle(np.exp(1j * (np.arctan2(V[:, 1], V[:, 0]) - np.arctan2(ref[:, 1], ref[:, 0])))))
    print(f"{name:34s} 分离比 中位 {np.median(m):.2f}  方向 中位 {np.median(a):+4.0f}° |Δ| {np.median(np.abs(a)):3.0f}°  "
          + "  ".join(f"{c.NAMES[i]} {m[i]:.2f}/{a[i]:+.0f}°" for i in (10, 15, 13, 16, 14, 2, 1)))
    return m, a
SC = 10875 / 1600
PANEL_X = {"left": 38, "right": 757, "alexa": 170}   # 1600 px 宽缩略图坐标
def cells(img, panel, row_sel):
    """row_sel: 'ideal' 或 物理色卡的列 0/1/2 (6500/5600/3200K) → (24, 3) 8 位 sRGB"""
    out = []
    for r in range(4):
        for c in range(6):
            x0 = (PANEL_X[panel] + 115 * c) * SC; y0 = (110 + 115 * r) * SC; w = 102 * SC
            if row_sel == "ideal":
                cx, cy = x0 + w * 0.5, y0 + w * 0.5
            elif row_sel.startswith("led"):
                cx, cy = x0 + w * (int(row_sel[3]) + 0.5) / 3, y0 + w / 6
            else:
                cx, cy = x0 + w * (int(row_sel) + 0.5) / 3, y0 + w * 5 / 6
            h = int(w / 12)
            out.append(np.median(img[int(cy) - h:int(cy) + h, int(cx) - h:int(cx) + h].reshape(-1, 3), 0))
    return np.array(out, float)
ideal = c.vec(cells(A, "right", "ideal"))
W = spectral.WAVELENGTHS
sds = colour.SDS_COLOURCHECKERS["ColorChecker N Ohta"]
R = np.array([sds[k].copy().align(colour.SpectralShape(W[0], W[-1], W[1] - W[0])).values for k in sds])
for col, cct, light in [(2, "3200K", "BB3200"), (1, "5600K", "D55")]:
    print(f"== {cct}")
    f5219 = c.vec(cells(A, "right", str(col))); alexa = c.vec(cells(B, "alexa", str(col)))
    rel("Alexa vs Ideal (管线本身)", alexa, ideal)
    rel("5219 vs Ideal", f5219, ideal)
    rel("★ 5219 vs Alexa (胶片本身)", f5219, alexa)
    rgb = lights.camera_rgb(R * spectral.illuminant(light))
    rgb = rgb / (rgb[21] @ [0.2126, 0.7152, 0.0722]) * 0.184
    img = np.repeat(np.repeat(rgb.reshape(4, 6, 3), 48, 0), 48, 1)
    pass
    for proc in ["ECN-2", "C-41"]:
      for scn in ["frontier", "noritsu"]:
            eng = Engine(Params(film="kodak_vision3_500t", process=proc, scanner=scn, timing="gray", grain=0, scene_lights=None,
                                scene_illuminant="D55"))
            out = np.round(srgb_encode(np.clip(eng.render(img), 0, 1)) * 255)
            P = out[24::48, 24::48].reshape(24, 3)
            rel(f"★ 模拟 500T {proc} {scn} vs Ideal", c.vec(P), ideal)
