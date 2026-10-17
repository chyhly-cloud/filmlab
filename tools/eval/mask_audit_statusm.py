"""原始 Kodak 数字化曲线: spektrafilm 提交 049f677^ 的 src/spectral_film_lab/data/film/negative/kodak_vision3_500t/*.csv 与 densitometer/status_M (CC BY-SA 4.0)。
Status M 自洽: 档案的分层浓度曲线 × 染料谱 → 混合 Status M 密度, 与 Kodak 原始 H-D (Status M) 比较。"""
import numpy as np, json
M = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/mask/raw5219/"
W = np.arange(380, 781, 5.0)
def load(n, left=0.0, right=0.0):
    a = np.loadtxt(M + n + ".csv", delimiter=","); a = a[np.argsort(a[:, 0])]
    return a
def onW(a): return np.interp(W, a[:, 0], a[:, 1], left=0.0, right=0.0)
SM = np.stack([onW(load(f"statusM_{c}")) for c in "rgb"], 1)
SM = 10 ** SM if SM.max() < 10 and SM.min() < 0 else SM          # 表若是 log 响应
SM /= SM.sum(0)
d = json.load(open("/home/user/filmlab/filmlab/data/profiles/kodak_vision3_500t.json"))["data"]
x = np.array(d["log_exposure"]); D = np.nan_to_num(np.array(d["density_curves"], float))
import sys; sys.path.insert(0, "/home/user/filmlab")
from filmlab.stocks import _clean
cd = _clean(d["channel_density"])
raw = [load(f"density_curve_{c}") for c in "rgb"]
def statusM(spec):
    T = 10 ** -(D @ spec.T)              # 净 (去 D-min) 光谱
    return -np.log10(T @ SM)
def compare(spec, name):
    S = statusM(spec)
    # Kodak 原始曲线 (含 D-min) → 去最小值; 横轴: 每条原始曲线按自身 logH; 用一个共同偏移对齐 (绿通道)
    best = None
    for off in np.linspace(-4, 4, 801):
        err = []
        for c in range(3):
            r = raw[c]; rx, ry = r[:, 0] + off, r[:, 1] - r[:, 1].min()
            m = (rx > x.min()) & (rx < x.max()) & (ry > 0.05) & (ry < 2.0)
            err.append(np.interp(rx[m], x, S[:, c]) - ry[m])
        e = np.sqrt(np.mean(np.concatenate(err) ** 2))
        if best is None or e < best[0]: best = (e, off, err)
    e, off, err = best
    print(f"{name:26s} Status M RMSE {e:.4f}  各通道 RMSE R {np.sqrt(np.mean(err[0]**2)):.4f} G {np.sqrt(np.mean(err[1]**2)):.4f} B {np.sqrt(np.mean(err[2]**2)):.4f}"
          f"  各通道平均偏差 R {np.mean(err[0]):+.4f} G {np.mean(err[1]):+.4f} B {np.mean(err[2]):+.4f}")
compare(cd, "带符号 (spektrafilm 原样)")
compare(np.clip(cd, 0, None), "截断 (FilmLab 现行)")
