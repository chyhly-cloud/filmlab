"""定光预言机 O1 (诊断): 每帧三通道印放光号偏移 ΔK_c, 使模型成片在全帧 24 px 格子 (中位) 上最接近真实 (log 线性 RGB)。
只用于在相同"印得多亮 / 多偏色"条件下比较饱和红; 不进入任何渲染。"""
import numpy as np
from scipy.optimize import least_squares
from scipy import ndimage as ndi
from t32_chain import sc, PrintTiming
def cells(z, excl=None, cell=24):
    real, Dn = z['real'], z['Dn'].astype(float)
    far = ndi.distance_transform_edt(~z['core']) * z['px_um'] > 500
    ok = np.isfinite(real).all(-1) & far & (real.min(-1) > 0.003) & (real.max(-1) < 0.95)
    if excl is not None: ok &= ~excl
    R, D = [], []
    h, w = ok.shape
    for y in range(0, h - cell + 1, cell):
        for x in range(0, w - cell + 1, cell):
            m = ok[y:y+cell, x:x+cell]
            if m.mean() < 0.9: continue
            R.append(np.median(real[y:y+cell, x:x+cell][m], 0)); D.append(np.median(Dn[y:y+cell, x:x+cell][m], 0))
    return np.array(R), np.array(D)
def fit(z, excl=None):
    R, D = cells(z, excl)
    t0 = PrintTiming(K=z["K"], s=z["s"], M=(z["M_off"], z["M_r"]))
    def make(dk):
        return PrintTiming(K=z["K"] + dk, s=z["s"], M=(z["M_off"], z["M_r"]))
    def resid(dk):
        sc.timing = make(dk)
        o = np.clip(sc.to_rgb(D), 1e-4, None)
        return (np.log10(o) - np.log10(R)).ravel()
    r = least_squares(resid, np.zeros(3), bounds=(-1.5, 1.5), loss="soft_l1", f_scale=0.1)
    return make(r.x), r.x, len(R)
