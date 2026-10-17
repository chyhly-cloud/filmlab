"""EPFL 灰阶楔 (R1-17-18 ↔ P1480492): 每级 L*, a*, b* —— 真实 vs 模拟 (染料谱表示 × C-41 τ)。"""
import sys, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab"); sys.path.insert(0, "/home/user/filmlab/tools/eval")
import mask_variant as variant; variant.install(sys.argv[1])
from PIL import Image
from filmlab import chemistry
from filmlab.rawio import load_raw, film_metering
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
D = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/epfl/data"; N = 14
def steps(img, x0, x1, y0, y1):
    h, w = img.shape[:2]; out = []
    for k in range(N):
        a = x0 + (x1 - x0) * (k + 0.3) / N; b = x0 + (x1 - x0) * (k + 0.7) / N
        out.append(np.median(img[int(y0*h):int(y1*h), int(a*w):int(b*w)].reshape(-1, 3), 0))
    return np.array(out)
Lab = lambda s: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(s, 1e-6, 1), apply_cctf_decoding=False))
film = np.asarray(Image.open(f"{D}/film/A002382raphael-R1-17-18.TIF").convert("RGB"), float) / 255
R = Lab(steps(colour.cctf_decoding(film), 125/1092, 765/1092, 0.35, 0.70))
raw = load_raw(f"{D}/digital/P1480492.RW2", max_size=1600); raw = raw[0] if isinstance(raw, tuple) else raw
if sys.argv[1] == "clip":
    print("真实 a*", " ".join(f"{v:+5.1f}" for v in R[:, 1])); print("真实 b*", " ".join(f"{v:+5.1f}" for v in R[:, 2]))
ok = (R[:, 0] > 15) & (R[:, 0] < 92)
old = chemistry.DEVELOPERS["ECN-2 CD"]
for tau in (1.0, 1 / 0.85, 1.3):
    chemistry.DEVELOPERS["ECN-2 CD"] = old.__class__(**{**old.__dict__, "activity": 1 / tau})
    rgb, _ = film_metering(raw, rng=np.random.default_rng(0), bias_ev=0.53)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "grain": 0, "roll": None}))
    S = Lab(steps(np.clip(eng.render(rgb), 0, 1), 232/1152, 822/1152, 0.40, 0.70))
    d = S - R
    # 漂移 = 模拟与真实 a/b 随亮度的斜率差 (去掉整体偏移, 只看交叉)
    sa = np.polyfit(R[ok, 0], d[ok, 1], 1)[0] * 50; sb = np.polyfit(R[ok, 0], d[ok, 2], 1)[0] * 50
    print(f"{sys.argv[1]:10s} τ={tau:.3f}  |ΔL| {np.mean(np.abs(d[ok,0])):4.1f}  Δa 均 {np.mean(d[ok,1]):+5.1f}  Δb 均 {np.mean(d[ok,2]):+5.1f}"
          f"  每 50 L* 的漂移差 a {sa:+5.1f} b {sb:+5.1f}  |Δab| {np.mean(np.hypot(d[ok,1], d[ok,2])):4.1f}")
chemistry.DEVELOPERS["ECN-2 CD"] = old
