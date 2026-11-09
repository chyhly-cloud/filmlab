"""输出图像上的颗粒量 (与 filmlab T51 / T51b / T52 的测量同一套估计器; 真实扫描的记录值就是用它们测的)。

  tiles(Y)        32 × 32 像素均匀块 (去一次平面; 块平均码值 12–250, 截断像素 ≤ 5%; T45 平坦判据): (y, x, 码值, σ_px, σ_hp, σ_48)
                  σ_px = 去平面后的标准差; σ_hp = 减去 σ = 1 像素高斯后的标准差; σ_48 = 2 × 2 盒平均的标准差 × (2·px/42.5)
                  (σ_48 只在 20 µm 像素上 = 40 µm 盒 ≈ 48 µm 圆孔); 8 位量化方差 1/12 已扣 (synthetic=True 时不扣)。
  phone(Y)        编码后的整张用 Lanczos 缩到长边 1200 (浮点), 取整到 8 位 (T52)。
  spectrum(img8)  64 × 64 像素块的输出密度 Do = −log10(线性) 的径向功率谱 (16 档, 0–0.5 周/像素; T51)。
"""
import numpy as np
from scipy.ndimage import gaussian_filter

T = 32
NB = 16
PHONE = 1200


def detrend(t):
    h, w = t.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    A = np.stack([np.ones(h * w), yy.ravel(), xx.ravel()], 1)
    c, *_ = np.linalg.lstsq(A, t.reshape(h * w, -1), rcond=None)
    return t - (A @ c).reshape(t.shape)


def boxmean(a, b):
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:h, :w].reshape(h // b, b, w // b, b, *a.shape[2:]).mean((1, 3))


def to_code(lin):
    """线性输出 → 8 位亮度码值 (sRGB 编码 × 255; 彩色取 BT.601 亮度, 黑白取均值)。"""
    from ..image import srgb_encode
    v = srgb_encode(np.clip(np.asarray(lin, float), 0, 1)) * 255.0          # 8 位输出: 截断到 0–255
    if v.ndim == 3:
        v = v @ np.array([0.299, 0.587, 0.114])
    return v


def tiles(Y, px_um=20.0, synthetic=False, lo=12, hi=250, flat=1.6, border=8):
    out = []
    H, W = Y.shape
    q = 0.0 if synthetic else 1.0 / 12.0
    for y in range(border, H - T + 1 - border, T):
        for x in range(border, W - T + 1 - border, T):
            t = Y[y:y + T, x:x + T]
            if not synthetic:
                if not lo <= t.mean() <= hi or ((t <= 0) | (t >= 255)).mean() > 0.05:
                    continue
            r = detrend(t)
            if not synthetic and boxmean(r, 4).var() > flat * boxmean(r, 2).var() / 4.0:
                continue
            s_px = np.sqrt(max(r.var() - q, 0.0))
            s_hp = (r - gaussian_filter(r, 1.0, mode="reflect"))[2:-2, 2:-2].std()
            s48 = np.sqrt(max(boxmean(r, 2).var() - q / 4, 0.0)) * (2 * px_um / 42.5)
            out.append((y, x, float(t.mean()), float(s_px), float(s_hp), float(s48)))
    return out


def median_stats(ts):
    a = np.array([t[2:] for t in ts])
    return dict(n=len(ts), code=float(np.median(a[:, 0])), s_px=float(np.median(a[:, 1])), s_hp=float(np.median(a[:, 2])),
                s48=float(np.median(a[:, 3])))


def phone(Y, factor=None, long=PHONE, quantize=True):
    """码值图 → Lanczos 缩小 (factor 给定时按倍数, 否则到长边 long), 取整到 8 位。"""
    from PIL import Image
    h, w = Y.shape
    k = 1.0 / factor if factor else long / max(h, w)
    im = Image.fromarray(np.asarray(Y, np.float32), "F").resize((round(w * k), round(h * k)), Image.LANCZOS)
    a = np.asarray(im, float)
    return np.clip(np.round(a), 0, 255) if quantize else a


def _radial(n):
    fy = np.fft.fftfreq(n)[:, None]; fx = np.fft.fftfreq(n)[None, :]
    fr = np.sqrt(fy ** 2 + fx ** 2)
    e = np.linspace(0, 0.5, NB + 1)
    return [(fr >= e[i]) & (fr < e[i + 1]) & (fr > 0) for i in range(NB)], 0.5 * (e[1:] + e[:-1])


def spectrum(Do, n=64, border=8):
    """输出密度 Do (h, w) 的径向功率谱 (块 n × n, 去平面, 汉宁窗, 方差归一); 返回 (fc, P)。"""
    m, fc = _radial(n)
    w = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    w = w / np.sqrt((w ** 2).mean())
    Ps = []
    H, W = Do.shape
    for y in range(border, H - n + 1 - border, n):
        for x in range(border, W - n + 1 - border, n):
            r = detrend(Do[y:y + n, x:x + n])
            F = np.fft.fft2(r * w) / n
            Ps.append([np.mean(np.abs(F[mm]) ** 2) for mm in m])
    return fc, np.mean(Ps, 0)
