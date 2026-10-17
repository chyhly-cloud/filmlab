"""TE226 色卡逐阶段方向误差: 每个阶段按自身灰阶反查 (单调), 与真实成片比较颜色分离的方向与大小。找出系统性色相旋转在哪一步出现。"""
import sys, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab"); sys.path.insert(0, "/home/user/filmlab/tools/eval")
import ablation_flags as af; af.install()
from PIL import Image
import pair_compare as pc
from filmlab import scanner as scm
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.rawio import film_metering
SP = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad"
G = {"film": (133/900, 153/602, 74.7/900, 74.3/602), "dig": (184/900, 233/675, 65.4/900, 64.2/675)}
def patches(img, g):
    H, W = img.shape[:2]; x0, y0, dx, dy = g; h = int(0.012 * W)
    return np.array([np.median(img[int((y0+dy*r)*H)-h:int((y0+dy*r)*H)+h, int((x0+dx*c)*W)-h:int((x0+dx*c)*W)+h].reshape(-1, img.shape[-1]), 0)
                     for r in range(5) for c in range(9)], float)
real = np.asarray(Image.open(SP + "/epfl/data/film/A002382raphael-R1-20-21.TIF").convert("RGB"), float) / 255
PR = patches(real, G["film"])
raw = pc.load_digital(SP + "/epfl/data/digital/P1480487.RW2", 1400)[0]
GRAY = list(range(36, 44)); Yn = np.log10(patches(raw, G["dig"])[GRAY] @ [0.2126, 0.7152, 0.0722])
def vec(P):
    s = np.empty((36, 3))
    for c in range(3):
        o = np.argsort(P[GRAY, c]); s[:, c] = np.interp(P[:36, c], P[GRAY, c][o], Yn[o])
    return np.stack([s[:, 0] - s[:, 1], s[:, 2] - s[:, 1]], 1)
VR = vec(PR); big = np.hypot(*VR.T) > 0.08
m, _ = film_metering(raw, rng=np.random.default_rng(1), bias_ev=0.53)
eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": 3}))
rec = {}
dev = eng._develop
def dv(x, **kw):
    if x.ndim == 3 and x.shape[:2] == m.shape[:2]: rec["logH"] = x.copy()
    return dev(x, **kw)
eng._develop = dv
sc = eng.scanner; tr, rp = sc.to_rgb, sc._reference_print
sc.to_rgb = lambda Dn: (rec.__setitem__("Dn", Dn.copy()), tr(Dn))[1]
sc._reference_print = lambda Dp, k, **kw: (rec.__setitem__("Dp", Dp.copy()), rp(Dp, k, **kw))[1]
out = eng.render(m); h, w = m.shape[:2]
stages = {"相机 RGB (输入)": m, "层曝光 logH": rec["logH"], "染料量": eng.state.dye,
          "扫描密度 Dn": rec["Dn"].reshape(h, w, 3), "相纸密度 Dp": rec["Dp"].reshape(h, w, 3), "输出 线性 RGB": out}
for n, a in stages.items():
    V = vec(patches(a, G["dig"]))
    r = np.hypot(*V.T) / np.maximum(np.hypot(*VR.T), 1e-3)
    ang = np.degrees(np.angle(np.exp(1j * (np.arctan2(V[:, 1], V[:, 0]) - np.arctan2(VR[:, 1], VR[:, 0])))))
    print(f"{af.tag():22s} {n:16s} 分离比 {np.median(r[big]):.2f}  方向误差 中位 {np.median(ang[big]):+4.0f}°  |中位| {np.median(np.abs(ang[big])):3.0f}°  (同号比例 {np.mean(np.sign(ang[big]) == np.sign(np.median(ang[big]))):.0%})")
