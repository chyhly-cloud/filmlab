"""模型链的跨通道作用分解: 对同一组像素 (存的 logH) 依次换成"单通道"版本:
  B 显影: DIR 只留同层 (内禀曲线按同一矩阵重解 → 中性灰阶不变)
  C 扫描: Dn_c = 中性灰阶上的 Dn_c(染料_c) (去掉染料光谱串扰与印放矩阵 P 的跨通道项)
  D 相纸: 输出_c = 中性灰在相纸密度 Dp_c 下的输出 (去掉相纸染料串扰)
定光固定为存下的 K/s/M, 只看链本身。"""
import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from filmlab import generation; generation.use("t31_base")
from filmlab import chemistry
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.negative import Negative
from filmlab.scanning.timing import PrintTiming
from filmlab.scanning.paper import OUTPUT_BLACK

eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))
sc = eng.scanner
def develop(logH, inter=True):
    if inter:
        return chemistry.run(eng.e, eng.proc, logH, 0.0, tau_factor=eng.v.tau_factor)
    A0 = chemistry._dir_matrix
    chemistry._dir_matrix = lambda a, neg=True, _f=A0: np.diag(np.diag(_f(a, neg)))
    try:
        return chemistry.run(eng.e, eng.proc, logH, 0.0, tau_factor=eng.v.tau_factor)
    finally:
        chemistry._dir_matrix = A0
def dn_of(st):
    return sc.normalized_density(eng._transmittance(st))
# 中性灰阶的 染料_c → Dn_c
xw = np.linspace(-3.5, 3.0, 400)
stw = develop(np.repeat(xw[:, None], 3, 1))
DYEW, DNW = stw.dye, dn_of(stw)
def dn_ci(dye):
    return np.stack([np.interp(dye[:, c], DYEW[:, c], DNW[:, c]) for c in range(3)], -1)
def out_of(Dn, timing, ci_paper=False):
    sc.timing = timing
    if not ci_paper:
        return np.clip(sc.to_rgb(Dn), 0, None)
    dp = sc.digital_paper
    Dp = dp.density(timing.paper_exposure(Dn))
    k = dp.k
    return OUTPUT_BLACK + (1 - OUTPUT_BLACK) * (10.0 ** -Dp - k) / (1 - k)
def chain(logH, timing):
    """返回各组合的输出 (n,3): full, noDIR, noScan, noPaper, all_ci"""
    res = {}
    st1, st0 = develop(logH, True), develop(logH, False)
    for dk, st in (("", st1), ("B", st0)):
        Dn = dn_of(st)
        Dc = dn_ci(st.dye)
        for sk, D in (("", Dn), ("C", Dc)):
            for pk, ci in (("", False), ("D", True)):
                res[(dk + sk + pk) or "full"] = out_of(D, timing, ci)
    return res, st1
if __name__ == "__main__":
    import glob
    from scipy import ndimage as ndi
    from t32_chart import patches, NAMES
    z = np.load("epfl/R1-20-21_t31_base_ev-1.npz")
    tim = PrintTiming(K=z["K"], s=z["s"], M=(z["M_off"], z["M_r"]))
    P = patches(z)
    keys = [(r, c) for r in range(4) for c in range(9)]
    lh = np.array([P[k]["logH"] for k in keys])
    res, st = chain(lh, tim)
    print("核对: full 与存档 out 之差", np.abs(res["full"] - np.array([P[k]["out"] for k in keys])).max().round(4))
    combos = ["full", "B", "C", "D", "BC", "BD", "CD", "BCD"]
    print("色块 | 真实 G/R | " + " | ".join(f"模型{c} G/R" for c in combos))
    for i, k in enumerate(keys):
        nm = NAMES[k[0]][k[1]]
        rr = P[k]["real"]
        print(f"{nm:5s} | {rr[1]/rr[0]:.3f} | " + " | ".join(f"{res[c][i,1]/res[c][i,0]:.3f}" for c in combos))
