"""原始 Kodak 数字化曲线: spektrafilm 提交 049f677^ 的 src/spectral_film_lab/data/film/negative/kodak_vision3_500t/*.csv 与 densitometer/status_M (CC BY-SA 4.0)。
独立检验: Kodak 原始数字化曲线 (未经 spektrafilm 重建) 能否解释 中性中间调 − 最低密度。"""
import numpy as np, json, sys
from scipy.optimize import nnls
M = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/mask/raw5219/"
W = np.arange(380, 781, 5.0)
def load(n):
    a = np.loadtxt(M + n + ".csv", delimiter=","); o = np.argsort(a[:, 0]); a = a[o]
    v = np.interp(W, a[:, 0], a[:, 1], left=np.nan, right=np.nan); return v
c, m, y, dmin, dmid = (load("dye_density_" + k) for k in "c m y min mid".split())
K = np.stack([c, m, y], 1)
ok = np.all(np.isfinite(K), 1) & np.isfinite(dmin) & np.isfinite(dmid)
print("共同有效波段", W[ok].min(), "–", W[ok].max(), "nm, 点数", ok.sum())
t = (dmid - dmin)[ok]
prof = np.array(json.load(open("/home/user/filmlab/filmlab/data/profiles/kodak_vision3_500t.json"))["data"]["channel_density"], float)
def fit(A, name):
    A = A[ok]
    if np.isnan(A).any(): A = np.nan_to_num(A)
    co, _ = nnls(A, t); r = A @ co - t
    j = np.argmax(np.abs(r))
    print(f"{name:40s} RMSE {np.sqrt((r**2).mean()):.4f} D  最大 {abs(r[j]):.4f} @ {W[ok][j]:.0f} nm  浓度 {np.round(co, 3)}")
    return r
fit(K, "Kodak 原始曲线 (带原始符号)")
fit(np.clip(K, 0, None), "Kodak 原始曲线 截断")
fit(prof, "spektrafilm 重建 (带符号)")
fit(np.clip(prof, 0, None), "spektrafilm 重建 截断 (FilmLab 现行)")
print("Kodak 原始曲线负值:", {k: (round(float(np.nanmin(v)), 3), W[np.nanargmin(v)]) for k, v in zip("CMY", K.T)})
