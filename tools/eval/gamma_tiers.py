"""Kodak 原始曲线与 Status M 表: spektrafilm 提交 049f677^ 的 csv (CC BY-SA 4.0), 放在 scratchpad/mask/raw5219。
5219 Status M 中间调 gamma 与密度范围: Kodak 原始测量 / spektrafilm 重建 / FilmLab 模型 (ECN-2 与 C-41 各 τ)。"""
import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab"); sys.path.insert(0, "/home/user/filmlab/tools/eval")
import mask_variant; mask_variant.install(sys.argv[1] if len(sys.argv) > 1 else "clip")
M = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/mask/raw5219/"
W = np.arange(380, 781, 5.0)
def load(n): a = np.loadtxt(M + n + ".csv", delimiter=","); return a[np.argsort(a[:, 0])]
SM = np.stack([np.interp(W, *load(f"statusM_{c}").T, left=0, right=0) for c in "rgb"], 1); SM /= SM.sum(0)
def gamma(x, D, lo=0.3, hi=1.3):
    """中间调 gamma: 密度 (去最小值) 在 [lo, hi] 区间的线性斜率; 及总密度范围 (最大−最小)。"""
    out = []
    for c in range(3):
        d = D[:, c] - np.nanmin(D[:, c]); m = (d > lo) & (d < hi) & np.isfinite(d)
        out.append(np.polyfit(x[m], d[m], 1)[0])
    return np.array(out), np.nanmax(D, 0) - np.nanmin(D, 0)
if len(sys.argv) < 3:
    raw = [load(f"density_curve_{c}") for c in "rgb"]
    g = [gamma(r[:, 0], np.repeat(r[:, 1:2], 3, 1))[0][0] for r in raw]; rg = [np.ptp(r[:, 1]) for r in raw]
    print(f"Kodak 原始 H-D (ECN-2, Status M, 测量)           gamma R {g[0]:.3f} G {g[1]:.3f} B {g[2]:.3f}  密度范围 {np.round(rg, 2)}")
    d = json.load(open("/home/user/filmlab/filmlab/data/profiles/kodak_vision3_500t.json"))["data"]
    x = np.array(d["log_exposure"]); L = np.nan_to_num(np.array(d["density_curves"], float))
    from filmlab.stocks import _clean
    cd = _clean(d["channel_density"])
    for name, spec in (("带符号", cd), ("截断", np.clip(cd, 0, None))):
        S = -np.log10((10 ** -(L @ spec.T)) @ SM)
        g, r = gamma(x, S); print(f"spektrafilm 重建 × {name}谱 → Status M (重建)    gamma R {g[0]:.3f} G {g[1]:.3f} B {g[2]:.3f}  密度范围 {np.round(r, 2)}")
from filmlab import chemistry
from filmlab.engine import Engine, Params
from filmlab.materials import emulsion, without_remjet
old = chemistry.DEVELOPERS["ECN-2 CD"]
steps = np.arange(-4, 6.01, 0.1)
for label, proc, tau in [("ECN-2", "ECN-2", None)] + [(f"C-41 τ={t:.3f}", "C-41", t) for t in (1.0, 1.1, 1 / 0.85, 1.3, 1.4)]:
    if tau: chemistry.DEVELOPERS["ECN-2 CD"] = old.__class__(**{**old.__dict__, "activity": 1 / tau})
    eng = Engine(Params(film=without_remjet(emulsion("kodak_vision3_500t"), "800T"), process=proc, grain=0, roll=None, scene_lights=None, timing="gray"))
    rgb = (0.184 * 2.0 ** steps)[:, None] * np.ones(3)
    H = eng._expose_lin(rgb, eng.M_ref, compensation=False); st = eng._develop(eng._log_exposure(H))
    T = eng._transmittance(st); base = eng._transmittance_base()
    S = -np.log10((T @ SM) / (base @ SM))
    g, r = gamma(steps * np.log10(2), S)
    print(f"FilmLab {sys.argv[1] if len(sys.argv)>1 else 'clip':10s} {label:14s} (模型)  gamma R {g[0]:.3f} G {g[1]:.3f} B {g[2]:.3f}  密度范围 {np.round(r, 2)}")
    chemistry.DEVELOPERS["ECN-2 CD"] = old
