"""T51: 真实冲印店扫描 (与模型输出, 同一代码) 的颗粒空间统计。

只读图像, 只存测量: 每帧的均匀块 → 输出密度 Do = −log10(线性 sRGB) 的
  · 径向功率谱 (每通道, 16 档, 0–0.5 周/像素; 块 64 × 64 像素, 汉宁窗, 方差归一: Σ 档均值 × 档内频点数 / T² = 方差),
  · 通道间互谱 (R–G、G–B、R–B) → 按频率的相干 (= 该频带的通道相关),
  · T46 第 7.2 节的 4 个高斯差分频带的通道相关 (同 T46 bandcorr: σ 0→1→2→4→8 像素),
  · 8 位量化方差 (每通道, Do 单位; 白噪声, 不扣除, 拟合时作为已知的白噪声底)。
均匀块判据同 T45 gstat: 去一次平面; 2b 盒平均方差 ≤ 1.6 × b 盒平均方差 / 4 (b ≈ 24 µm, 至少 2 像素); 码值 12–250。
用法: python3 -I t51_scan_measure.py 输出.json µm/像素 图1 [图2 ...]   (也可 import: measure_array)
"""
import json
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

NB = 16
TILE = 64
SIG = (0.0, 1.0, 2.0, 4.0, 8.0)


def srgb_lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def dlin_dcode(c):
    c = np.clip(c / 255.0, 1e-6, 1)
    return np.where(c <= 0.04045, 1 / 12.92, 2.4 / 1.055 * ((c + 0.055) / 1.055) ** 1.4) / 255.0


def boxmean(a, b):
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:h, :w].reshape(h // b, b, w // b, b, *a.shape[2:]).mean((1, 3))


def detrend(t):
    h, w = t.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    A = np.stack([np.ones(h * w), yy.ravel(), xx.ravel()], 1)
    coef, *_ = np.linalg.lstsq(A, t.reshape(h * w, -1), rcond=None)
    return t - (A @ coef).reshape(t.shape)


def flat_tiles(img8, um, T=TILE, min_code=12, max_code=250, flat=1.6):
    """均匀块: (y, x, 平均 Do, 去平面 Do, 量化方差)。img8 可为 float (模型输出已编码到 0–255)。"""
    b = max(2, int(round(24.0 / um)))
    H, W = img8.shape[:2]
    out = []
    for y in range(0, H - T + 1, T):
        for x in range(0, W - T + 1, T):
            t8 = np.asarray(img8[y:y + T, x:x + T], float)
            if t8.min() < min_code or t8.max() > max_code:
                continue
            lin = srgb_lin(t8)
            D = -np.log10(lin)
            r = detrend(D)
            v1 = boxmean(r, b).var((0, 1)); v2 = boxmean(r, 2 * b).var((0, 1))
            if flat is not None and np.any(v2 > flat * v1 / 4.0):
                continue
            dD = dlin_dcode(t8) / (lin * np.log(10))
            q = (dD ** 2 / 12.0).mean((0, 1))
            out.append((y, x, D.mean((0, 1)), r, q))
    return out


_RAD = {}


def _radial(T):
    if T not in _RAD:
        fy = np.fft.fftfreq(T)[:, None]; fx = np.fft.fftfreq(T)[None, :]
        fr = np.sqrt(fy ** 2 + fx ** 2)
        e = np.linspace(0, 0.5, NB + 1)
        _RAD[T] = ([(fr >= e[i]) & (fr < e[i + 1]) & (fr > 0) for i in range(NB)], 0.5 * (e[1:] + e[:-1]))
    return _RAD[T]


def tile_spectra(r):
    """r (T,T,3) → P (NB,3) 功率 (每频点, 方差单位 / T²·T²), X (NB,3) 互谱实部 [RG, GB, RB]。"""
    T = r.shape[0]
    w = np.hanning(T)[:, None] * np.hanning(T)[None, :]
    w = w / np.sqrt((w ** 2).mean())
    F = np.fft.fft2(r * w[..., None], axes=(0, 1)) / T
    m, _ = _radial(T)
    P = np.array([[np.mean(np.abs(F[mm, c]) ** 2) for c in range(3)] for mm in m])
    X = np.array([[np.mean((F[mm, a] * np.conj(F[mm, b])).real) for a, b in ((0, 1), (1, 2), (0, 2))] for mm in m])
    return P, X


def band_corr(r):
    """T46 7.2 的 4 个高斯差分频带的通道相关 (去掉 8 像素边框)。"""
    g = [r if s == 0 else np.stack([gaussian_filter(r[..., c], s, mode="reflect") for c in range(3)], -1) for s in SIG]
    out, var = [], []
    for i in range(len(SIG) - 1):
        z = (g[i] - g[i + 1])[8:-8, 8:-8].reshape(-1, 3)
        C = np.cov(z.T); d = np.sqrt(np.diag(C))
        out.append([C[0, 1] / d[0] / d[1], C[1, 2] / d[1] / d[2], C[0, 2] / d[0] / d[2]])
        var.append(np.diag(C))
    return np.array(out), np.array(var)


def measure_array(img8, um, T=TILE, max_tiles=600, dlim=None, synthetic=False):
    """img8: (H,W,3) 8 位或已编码的浮点 sRGB。dlim: (lo, hi) 只取平均 Do (G) 在范围内的块。
    synthetic: 模型的均匀灰场 (按构造均匀, 不做码值范围与平坦判据)。"""
    tiles = flat_tiles(img8, um, T, min_code=0.0, max_code=255.0, flat=None) if synthetic else flat_tiles(img8, um, T)
    if dlim is not None:
        tiles = [t for t in tiles if dlim[0] <= t[2][1] < dlim[1]]
    tiles = tiles[:max_tiles]
    if not tiles:
        return dict(n=0)
    Ps, Xs, Bc, Bv = [], [], [], []
    for (_, _, _, r, q) in tiles:
        P, X = tile_spectra(r); Ps.append(P); Xs.append(X)
        bc, bv = band_corr(r); Bc.append(bc); Bv.append(bv)
    _, fc = _radial(T)
    P, X = np.mean(Ps, 0), np.mean(Xs, 0)
    coh = np.stack([X[:, 0] / np.sqrt(P[:, 0] * P[:, 1]), X[:, 1] / np.sqrt(P[:, 1] * P[:, 2]),
                    X[:, 2] / np.sqrt(P[:, 0] * P[:, 2])], 1)
    D = np.array([t[2] for t in tiles]); q = np.array([t[4] for t in tiles])
    return dict(n=len(tiles), um=um, T=T, fc=fc.tolist(), D_med=np.median(D, 0).tolist(), q_mean=q.mean(0).tolist(),
                P=P.tolist(), X=X.tolist(), coh=coh.tolist(), band_corr=np.median(Bc, 0).tolist(),
                band_var=np.median(Bv, 0).tolist(), var=np.mean([t[3].var((0, 1)) for t in tiles], 0).tolist())


def main():
    from PIL import Image
    dst, um = sys.argv[1], float(sys.argv[2])
    res = {}
    for f in sys.argv[3:]:
        a = np.asarray(Image.open(f).convert("RGB"))
        k = f.split("/")[-2] + "/" + f.split("/")[-1]
        res[k] = measure_array(a, um)
        m = res[k]
        print(k, a.shape, m["n"], None if not m["n"] else np.round(np.array(m["band_corr"]), 2).tolist(), flush=True)
    json.dump(res, open(dst, "w"))


if __name__ == "__main__":
    main()
