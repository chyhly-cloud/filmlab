"""中性灰阶 (3200K 参考光源) 上各级的通道反差: Status M (ECN-2 / C-41), Dn, 成片 (log10 输出 / logH)。
范围: 中灰 −0.3 … +0.9 logH (夜景亮物体所在段)。"""
import os
import numpy as np, sys
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2])]
from filmlab import generation, chemistry
from filmlab.engine import Engine, Params, GRAY
from filmlab.recipes import recipe
from filmlab.stocks import WAVELENGTHS as W
RAW = os.environ.get("KODAK5219_RAW", "raw5219")
def curve(n, zero=False):
    a = np.loadtxt(f"{RAW}/{n}.csv", delimiter=","); a = a[np.argsort(a[:, 0])]
    return np.interp(W, a[:, 0], a[:, 1], left=0 if zero else None, right=0 if zero else None)
F = np.stack([curve(f"statusM_{c}", True) for c in "rgb"], 1); F /= F.sum(0)
sm = lambda T: -np.log10(np.clip(T @ F, 1e-9, None))
hd = [np.loadtxt(f"{RAW}/density_curve_{c}.csv", delimiter=",") for c in "rgb"]
for gen in sys.argv[1:]:
    generation.use(gen)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))
    g = eng._log_exposure(eng._expose_lin(np.full((1, 3), GRAY), eng.M_ref))[0]
    steps = np.linspace(-0.3, 0.9, 13)
    lh = g[None, :] + steps[:, None]
    out = {}
    for proc in ("ECN-2", "C-41"):
        e2 = Engine(Params(**{**recipe("cinestill_800t"), "process": proc, "timing": "frame", "roll": None, "frame": 0, "grain": 0}))
        st = chemistry.run(e2.e, e2.proc, lh, 0.0, tau_factor=1.0)
        out[proc] = sm(e2._transmittance(st))
    st = eng._develop(lh)
    Dn = eng.scanner.normalized_density(eng._transmittance(st))
    # 成片: lab_measured 的固定定光 (中灰印成中性)
    o = eng.scan(st)
    sl = lambda Y: np.array([np.polyfit(steps, Y[:, c], 1)[0] for c in range(3)])
    kod = np.stack([np.interp(lh[:, 1] - g[1] + g[1], h[:, 0], h[:, 1]) for h in hd], 1)   # Kodak 同一 logH (绿层轴)
    print(f"== {gen}: 中灰 logH {g.round(3)}")
    for nm, Y in (("Kodak 5219 Status M (ECN-2)", np.stack([np.interp(lh[:, c], h[:, 0], h[:, 1]) for c, h in enumerate(hd)], 1)),
                  ("模型 Status M (ECN-2)", out["ECN-2"]), ("模型 Status M (C-41)", out["C-41"]), ("模型 Dn (S4b)", Dn),
                  ("模型 成片 log10 (S4d)", np.log10(np.clip(o, 1e-6, None)))):
        s = sl(Y); print(f"  {nm:28s} 斜率 R G B {s.round(3)}  R/G {s[0]/s[1]:.3f}  B/G {s[2]/s[1]:.3f}")
