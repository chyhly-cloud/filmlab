"""EPFL 透射色卡 (Esser TE226, R1-20-21 ↔ P1480487; 真实 800T C-41 Frontier 扫描) 上的消融对照, 只用 RAW 输入。
环境变量开关见 ablation_flags.py。输出: 36 色块 ΔE00, 相对自身灰轴的颜色分离比与方向误差, 灰阶 Δa/Δb, 以及逐色相区的方向误差。"""
import sys, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab"); sys.path.insert(0, "/home/user/filmlab/tools/eval")
import ablation_flags as af; af.install()
from PIL import Image
import pair_compare as pc
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.rawio import film_metering
from filmlab.io import srgb_encode
SP = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad"
G = {"film": (133/900, 153/602, 74.7/900, 74.3/602), "dig": (184/900, 233/675, 65.4/900, 64.2/675)}
def patches(img, g):
    H, W = img.shape[:2]; x0, y0, dx, dy = g; h = int(0.012 * W)
    return np.array([np.median(img[int((y0+dy*r)*H)-h:int((y0+dy*r)*H)+h, int((x0+dx*c)*W)-h:int((x0+dx*c)*W)+h].reshape(-1, 3), 0)
                     for r in range(5) for c in range(9)], float)
lab = lambda s: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(s, 0, 1)))
real = np.asarray(Image.open(SP + "/epfl/data/film/A002382raphael-R1-20-21.TIF").convert("RGB"), float) / 255
PR = patches(real, G["film"]); LR = lab(PR)
raw = pc.load_digital(SP + "/epfl/data/digital/P1480487.RW2", 1400)[0]
GRAY = list(range(36, 44))
Yn = np.log10(patches(raw, G["dig"])[GRAY] @ [0.2126, 0.7152, 0.0722])
def vec(P):
    s = np.empty((36, 3))
    for c in range(3):
        o = np.argsort(P[GRAY, c]); s[:, c] = np.interp(P[:36, c], P[GRAY, c][o], Yn[o])
    return np.stack([s[:, 0] - s[:, 1], s[:, 2] - s[:, 1]], 1)
VR = vec(PR); big = np.hypot(*VR.T) > 0.08
m, _ = film_metering(raw, rng=np.random.default_rng(1), bias_ev=0.53)
eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": 3}))
out = srgb_encode(np.clip(eng.render(m), 0, 1))
PS = patches(out, G["dig"]); LS = lab(PS); V = vec(PS)
ratio = np.hypot(*V.T) / np.maximum(np.hypot(*VR.T), 1e-3)
ang = np.degrees(np.angle(np.exp(1j * (np.arctan2(V[:, 1], V[:, 0]) - np.arctan2(VR[:, 1], VR[:, 0])))))
g = LS[GRAY] - LR[GRAY]
hue = np.degrees(np.arctan2(VR[:, 1], VR[:, 0])) % 360      # 真实色块在 (r−g, b−g) 平面上的方向
secs = [("偏红", 135, 225), ("偏黄", 225, 315), ("偏蓝/青", 315, 45), ("偏品/紫", 45, 135)]
row = []
for n, a, b in secs:
    s = big & (((hue >= a) & (hue < b)) if a < b else ((hue >= a) | (hue < b)))
    row.append(f"{n} 比 {np.median(ratio[s]):.2f} 角 {np.median(ang[s]):+3.0f}°")
print(f"{af.tag():30s} 色块 ΔE00 {np.median(colour.delta_E(LS[:36], LR[:36], method='CIE 2000')):5.2f} | 分离比 {np.median(ratio[big]):.2f} "
      f"|角| {np.median(np.abs(ang[big])):3.0f}° | 灰阶 Δa {g[:,1].mean():+5.1f} Δb {g[:,2].mean():+5.1f} | " + "  ".join(row))
