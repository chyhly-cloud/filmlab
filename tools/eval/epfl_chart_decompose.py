"""EPFL 透射色卡 (Esser TE226, R1-20-21 ↔ P1480487): 把误差拆为 输入 / 颜色还原 (相对自身灰轴) / 灰平衡与影调。"""
import sys, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")
from PIL import Image
import pair_compare as pc
from filmlab import scanner as scm
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
PR = patches(real, G["film"])
raw = pc.load_digital(SP + "/epfl/data/digital/P1480487.RW2", 1400)[0]
jpg = pc.load_digital(SP + "/inputfmt/jpg/P1480487.jpg", 1400)[0]
GRAY = list(range(36, 44))                    # 第 5 行前 8 级 (第 9 格为空框)
Yn = np.log10(patches(raw, G["dig"])[GRAY] @ [0.2126, 0.7152, 0.0722])   # 公共灰轴: RAW 的线性亮度
def vec(P):
    """每通道按本图灰阶反查到公共灰轴 → 36 个色块的 (r−g, b−g) 分离。只依赖色块相对灰阶的位置。"""
    s = np.empty((36, 3))
    for c in range(3):
        o = np.argsort(P[GRAY, c]); s[:, c] = np.interp(P[:36, c], P[GRAY, c][o], Yn[o])
    return np.stack([s[:, 0] - s[:, 1], s[:, 2] - s[:, 1]], 1)
def cmp(V, ref):
    m = np.hypot(*V.T) / np.maximum(np.hypot(*ref.T), 1e-3)
    a = np.degrees(np.angle(np.exp(1j * (np.arctan2(V[:, 1], V[:, 0]) - np.arctan2(ref[:, 1], ref[:, 0])))))
    big = np.hypot(*ref.T) > 0.08
    return np.median(m[big]), np.median(np.abs(a[big]))
VR = vec(PR); LR = lab(PR)
print(f"真实扫描 灰阶 a,b: " + " ".join(f"({a:+.0f},{b:+.0f})" for a, b in LR[GRAY, 1:]))
for name, P in [("输入 RAW (线性)", patches(raw, G["dig"])), ("输入 相机JPEG (解码)", patches(jpg, G["dig"]))]:
    r, d = cmp(vec(P), VR); print(f"{name:22s} 颜色分离 vs 真实: 比 {r:.2f} |Δ方向| {d:3.0f}°")
SM0 = scm.SATURATION_MATRIX.copy()
for fmt, rgb in [("RAW", raw), ("JPEG", jpg)]:
    for cond in ["cur", "s0", "s0_bare"]:
        m, _ = film_metering(rgb, rng=np.random.default_rng(1), bias_ev=0.53)
        scm.SATURATION_MATRIX = SM0 if cond != "s0_bare" else np.eye(3)
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": 3}))
        if cond != "cur": eng.scanner.strength = 0.0
        if cond == "s0_bare": eng.scanner.m = scm.SCANNERS["frontier_neutral"]
        out = srgb_encode(np.clip(eng.render(m), 0, 1))
        PS = patches(out, G["dig"]); LS = lab(PS)
        r, d = cmp(vec(PS), VR)
        de_all = np.median(colour.delta_E(LS[:36], LR[:36], method="CIE 2000"))
        g = LS[GRAY] - LR[GRAY]
        print(f"{fmt:4s} {cond:7s} 色块 ΔE00 {de_all:5.2f} | 颜色分离 比 {r:.2f} |Δ方向| {d:3.0f}° | 灰阶 ΔL {np.mean(g[:,0]):+5.1f} Δa {np.mean(g[:,1]):+5.1f} Δb {np.mean(g[:,2]):+5.1f}  模拟灰阶 a,b: "
              + " ".join(f"({a:+.0f},{b:+.0f})" for a, b in LS[GRAY, 1:]))
