"""图卡灰阶在每一级的通道反差 (相对绿): 底片 Status M → 扫描密度 Dn → 输出 (log 线性)。"""
import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")
import chart_500t as c
from PIL import Image
from filmlab.engine import Engine, Params
from filmlab.rawio import load_raw
import colour
D = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/charts"
def rel(X, ref=None):
    """X (6, 3) 随灰阶变化 → 各通道对绿通道的斜率。"""
    g = X[:, 1]
    return [np.polyfit(g, X[:, k], 1)[0] for k in (0, 2)]
for scene in c.SCENES:
    S = c.SCENES[scene]
    real = c.patches(np.asarray(Image.open(f"{D}/img/dlf_v3500T_500T_Noritsu_{S['scan']}.jpg")), S["grid_scan"])
    Lr = np.log10(np.clip(colour.models.eotf_sRGB(real / 255)[18:24], 1e-3, None))
    raw = load_raw(f"{D}/ref/dlf_x100v/{S['raf']}.RAF", max_size=3000); raw = raw[0] if isinstance(raw, tuple) else raw
    G = S["grid_raw"]; gray = c.patches(raw, G)[21] @ np.array([0.2126, 0.7152, 0.0722])
    print(f"== {scene}  真实输出 log  R/G {rel(Lr)[0]:.2f}  B/G {rel(Lr)[1]:.2f}")
    for proc in ("ECN-2", "C-41"):
        eng = Engine(Params(film="kodak_vision3_500t", process=proc, scanner="noritsu", timing="frame", output_conditioning=None))
        x = raw[::2, ::2] * 0.184 / gray
        neg = eng.develop_negative(x)
        tr = {}; out = eng.scan(neg, trace=tr)
        h, w = out.shape[:2]
        st = {"log H": neg.log_exposure, "Status M": neg.status_m(), "Dn": tr["Dn"],
              "输出 log": np.log10(np.clip(out, 1e-3, None))}
        line = []
        for k, v in st.items():
            P = c.patches(v, G, 0.5)[18:24]
            r = rel(P)
            line.append(f"{k} {r[0]:.2f}/{r[1]:.2f}")
        print(f"   {proc:6s} " + "   ".join(line))
