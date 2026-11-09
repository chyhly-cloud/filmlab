"""颗粒场的渲染与颗粒度 (闭式)。

render(emulsion, exposure | density, px_um | format, seed) → RenderResult(density, mean, …):
  底片面的逐像素密度 (每像素 = 胶片上一个 px_um × px_um 方块的积分, 即理想方盒孔径的扫描采样)。
  黑白 (显式银粒路径, silvergrain.silver):
    method "explicit": 每颗卤化银晶体在连续坐标中撒下 (泊松 / T50 显影晶体成团), 按光子统计显影, 原位生成银丝团,
        exp(−Στ) 在透射率域积分到像素, D_px = −log10 (像素平均透射率) (T47 的渲染器, filmlab 整帧用的同一个);
    method "closed_form": 同一泊松布尔模型 (含成团) 的闭式透射率协方差 Cov_T(r) (silver.silver_covariance) 在像素方盒上积分,
        得像素间协方差 c(i, j); 像素透射率取对数正态 T_px = E[T]·exp(s z − s²/2), z 为单位方差的相关高斯场
        (功率谱合成), s 与 z 的相关由 c 精确换算, 所以像素平均透射率、像素方差与像素间协方差都与闭式解 (即显式渲染的
        期望) 严格相同; 高阶 (偏度) 是对数正态的 (显式渲染: 少量不透明银粒的平均, 同为右偏)。D_px = −log10 T_px。
        检验: validation.fastpath (closed_form 对 explicit)。曝光随像素变化时, 在 log H 上每 0.1 一档预先算好协方差,
        同一白噪声经相邻两档的核再按 log H 线性插值 (按两核的重叠归一, 方差不变)。
    method "auto": 等效像素 px/k ≥ 3 µm 且图像 > 2 百万像素时 "closed_form", 否则 "explicit" (慢, 约 1 分钟 / 百万像素)。
  彩色 (染料云路径, silvergrain.dye): 每像素晶体计数 → 尺寸档二项 → 显影二项 → 每晶体产量伽马 → 每亚群按云大小展开
    (T46; filmlab 整帧用的同一个)。本身就是计数级的显式模拟, 速度够快, 无需高斯近似; 按 256 行分条渲染以限制内存。
mean: 闭式平均密度 (同一晶体模型的特性曲线)。density − mean 的平均不为 0: 像素内透射率平均后再取对数,
  E[D_px] ≥ −log10 E[T] (Jensen; T53 2.6 节 "颗粒越粗越亮" 的物理效应), 不扣除。

granularity(emulsion, density, aperture_um): 闭式 RMS 颗粒度 (漫射, 圆孔取等面积方孔)。
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import dye, silver
from .constants import APERTURE_UM2
from .emulsion import silver_curve_table
from .formats import px_um_for

GAUSS_DLOGH = 0.1               # 高斯路径的曝光档间距 (log10 H)
AUTO_MIN_PX_UM = 3.0             # closed_form 的最小 "等效像素" px/k (µm); k = 1 时 6 µm 像素 σ 与显式差 ≤ 3–5%, 3 µm 约 −7%
AUTO_MIN_PIXELS = 2_000_000
DYE_BAND_ROWS = 256


@dataclass
class RenderResult:
    density: np.ndarray         # 逐像素密度 (不含片基): 黑白 (h, w) 银层; 彩色 (h, w, 3) 每层染料当量
    mean: np.ndarray            # 闭式平均密度 (同形状)
    exposure: np.ndarray        # 每像素曝光 H (lux·s): 黑白 (h, w); 彩色 (h, w, 3)
    px_um: float
    method: str
    base_density: float = 0.0

    @property
    def total_density(self):
        return self.density + self.base_density

    @property
    def fluctuation(self):
        return self.density - self.mean


# ------------------------------------------------------------ 曝光 / 密度 --
def exposure_from_density(emulsion, density):
    """平均密度 (不含片基) → 曝光 H (反求同一晶体模型的特性曲线)。"""
    d = np.asarray(density, float)
    if emulsion.path == "silver":
        lh, Dt = silver_curve_table(emulsion.spec)
        return 10.0 ** np.interp(np.clip(d, Dt[0], Dt[-1]), Dt, lh)
    Hx = emulsion.exposure_axis()
    D = emulsion.curve.density_curves
    out = np.empty(d.shape)
    for c in range(3):
        o = np.argsort(D[:, c], kind="stable")
        out[..., c] = 10.0 ** np.interp(d[..., c], D[o, c], np.log10(Hx[o, c]))
    return out


# --------------------------------------------------------- 高斯 (闭式) 路径 --
def _block(C, h, target=0.1):
    """居中协方差网格块平均到约 target µm (window_variance 的同一做法; 积分不变)。"""
    f = max(1, int(round(target / h)))
    if f == 1:
        return C, h
    n0 = C.shape[0]; c0 = n0 // 2
    lo = (c0 - f // 2) % f
    m = (n0 - lo) // f
    return C[lo:lo + m * f, lo:lo + m * f].reshape(m, f, m, f).mean((1, 3)), h * f


def pixel_covariance(C, h, px_um, psf_um=0.0):
    """闭式透射率协方差 (居中网格, 间距 h) → 像素方盒 (边长 px) 之间的协方差 c[i, j] (|i|, |j| ≤ R)。
    c(i, j) = ∫∫ box(u) box(v) Cov((i, j)·px + u − v) = Cov ⊛ 三角 (像素自相关), 在 (i, j)·px 处取值。"""
    from scipy.ndimage import gaussian_filter
    C, h = _block(C, h)
    half = C.shape[0] // 2 * h                     # 协方差的支撑半径
    R = int(np.ceil((half + 3 * psf_um) / px_um)) + 1
    pad = int(np.ceil(((R + 1) * px_um - half) / h)) + 2
    Cp = np.pad(C, pad)
    if psf_um > 0:
        Cp = gaussian_filter(Cp, np.sqrt(2) * psf_um / h, mode="constant", truncate=5.0)
    n = Cp.shape[0]
    x = (np.arange(n) - n // 2) * h
    Ls = [np.clip(1 - np.abs(x - i * px_um) / px_um, 0, None) / px_um * h for i in range(-R, R + 1)]
    L = np.stack(Ls)                               # (2R+1, n)
    return L @ Cp @ L.T, R


def _sqrt_kernel(c):
    """像素协方差 c (2R+1)² → 核 g (同大小 × 2 的周期格上), g ⋆ g = c (功率谱开方; 负的谱值截 0)。返回单位方差的核。"""
    R = c.shape[0] // 2
    N = 4 * R + 5
    P = np.zeros((N, N))
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            P[i % N, j % N] = c[i + R, j + R]
    S = np.fft.fft2(P).real
    g = np.fft.ifft2(np.sqrt(np.clip(S, 0, None))).real
    g = np.fft.fftshift(g)
    m = N // 2
    k = min(m, 2 * R + 1)
    g = g[m - k:m + k + 1, m - k:m + k + 1]
    return g / np.sqrt((g * g).sum())


@lru_cache(maxsize=256)
def _level(spec, logH, px_um, psf_um):
    """一档曝光的对数正态参数: (E[T], s, 单位方差核)。像素透射率取对数正态 T_px = E[T]·exp(s z − s²/2) (正值、右偏,
    与少量不透明银粒的像素平均一致); 前两阶矩与闭式解严格相等: s² = ln(1 + Var T_px / E[T]²), 像素间 z 的相关
    ρ(i, j) = ln(1 + c(i, j)/E[T]²) / s² (c = 像素间透射率协方差)。"""
    ET, C, h = silver.silver_covariance(spec, 10.0 ** logH)
    c, R = pixel_covariance(C, h, px_um, psf_um)
    ET = float(ET)
    q = np.log1p(np.clip(c / (ET * ET), -0.999999, None))
    s2 = float(q[R, R])
    if s2 <= 0:
        k = np.zeros_like(c); k[R, R] = 1.0
        return ET, 0.0, k
    return ET, np.sqrt(s2), _sqrt_kernel(q / s2)


def gaussian_levels(spec, H, px_um, psf_um=0.0):
    lh = np.log10(np.clip(H, 1e-12, None))
    lo = np.floor(np.nanmin(lh) / GAUSS_DLOGH) * GAUSS_DLOGH
    hi = np.ceil(np.nanmax(lh) / GAUSS_DLOGH) * GAUSS_DLOGH
    lv = np.round(np.arange(lo, hi + GAUSS_DLOGH / 2, GAUSS_DLOGH), 6)
    if len(lv) < 2:
        lv = np.array([lo, lo + GAUSS_DLOGH])
    return lv, [_level(spec, float(v), float(px_um), float(psf_um)) for v in lv]


def render_silver_gaussian(spec, H, px_um, seed, psf_um=0.0):
    """显式银粒的二阶统计等价场 (对数正态; 见模块说明)。返回每像素银层透射率 (与 silver.render_silver 同义)。"""
    from scipy.ndimage import correlate
    H = np.asarray(H, float)
    lv, L = gaussian_levels(spec, H, px_um, psf_um)
    lh = np.clip(np.log10(np.clip(H, 1e-12, None)), lv[0], lv[-1])
    t = (lh - lv[0]) / GAUSS_DLOGH
    i0 = np.minimum(np.floor(t).astype(np.int64), len(lv) - 2)
    w = (t - i0).astype(np.float64)
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 55])
    noise = rng.standard_normal(H.shape).astype(np.float32)
    acc = np.zeros(H.shape, np.float64)
    for j in np.unique(np.concatenate([i0.ravel(), i0.ravel() + 1])):
        sel0 = i0 == j; sel1 = (i0 + 1) == j
        field = correlate(noise, L[j][2].astype(np.float32), mode="reflect")
        acc[sel0] += (1.0 - w[sel0]) * field[sel0]
        acc[sel1] += w[sel1] * field[sel1]
        del field
    rho = np.array([float((_pad_to(L[j][2], L[j + 1][2]) * _pad_to(L[j + 1][2], L[j][2])).sum()) for j in range(len(lv) - 1)])
    z = acc / np.sqrt((1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * rho[i0])
    s_lv = np.array([x[1] for x in L])
    s = (1 - w) * s_lv[i0] + w * s_lv[i0 + 1]
    # E[T] 用闭式密度的细表 (与显式路径的平均相同); s 在档之间按 log H 线性插值
    lt, Dt = silver_curve_table(spec)
    ET = 10.0 ** -np.interp(np.log10(np.clip(H, 1e-30, None)), lt, Dt)
    return np.clip(ET * np.exp(s * z - 0.5 * s * s), 1e-12, 1.0)


def _pad_to(a, b):
    """把核 a 居中补零到 a、b 中较大的尺寸。"""
    n = max(a.shape[0], b.shape[0])
    p = (n - a.shape[0]) // 2
    return np.pad(a, p) if p else a


# ------------------------------------------------------------------ 渲染 --
def choose_method(emulsion, shape, px_um, method="auto"):
    if emulsion.path == "dye":
        return "explicit"
    if method != "auto":
        return method
    eq = px_um / float(getattr(emulsion, "grain_scale", 1.0))      # 颗粒粗细 k: 场是 k = 1 的场放大 k 倍 → 等效像素 px/k
    return "closed_form" if (eq >= AUTO_MIN_PX_UM and shape[0] * shape[1] > AUTO_MIN_PIXELS) else "explicit"


def render(emulsion, exposure=None, density=None, px_um=None, format=None, seed=0, method="auto", psf_um=0.0,
           workers=1, strength=1.0):
    """渲染颗粒场。exposure: 每像素曝光 H (lux·s; 黑白 (h, w), 彩色 (h, w, 3)); 或 density: 每像素平均密度 (不含片基)。
    px_um: 每像素的胶片边长 (µm); 或 format: 画幅 ("35mm" …, 长边对长边)。psf_um: 底片面高斯光学 PSF (透射率上; 仅黑白)。
    workers: 显式银粒的进程数。strength: 颗粒倍数 (1 = 物理; 只为诊断)。"""
    if exposure is None and density is None:
        raise ValueError("需要 exposure 或 density")
    ref = exposure if exposure is not None else density
    shape = np.shape(ref)[:2]
    if px_um is None:
        if format is None:
            raise ValueError("需要 px_um 或 format")
        px_um = px_um_for(shape, format)
    px_um = float(px_um)
    H = np.asarray(exposure, float) if exposure is not None else exposure_from_density(emulsion, density)
    mean = emulsion.mean_density(H) if exposure is not None else np.asarray(density, float)
    m = choose_method(emulsion, shape, px_um, method)
    sp = emulsion.spec
    if emulsion.path == "silver":
        if m == "explicit":
            if psf_um > 0:
                T = silver.render_silver(sp, H, px_um, seed, psf_um=psf_um)
            else:
                T = silver.render_silver(sp, H, px_um, seed, workers=workers)
        elif m == "closed_form":
            T = render_silver_gaussian(sp, H, px_um, seed, psf_um=psf_um)
        else:
            raise ValueError(f"未知 method {m!r}")
        D = -np.log10(np.clip(T, 1e-12, None))
        if strength != 1.0:
            D = mean + strength * (D - mean)
        return RenderResult(D, mean, H, px_um, m, emulsion.base_density)
    D = _render_dye(emulsion, mean, px_um, seed, strength)
    return RenderResult(D, mean, H, px_um, m, 0.0)


def _render_dye(emulsion, mean, px_um, seed, strength=1.0):
    """染料云路径, 按 DYE_BAND_ROWS 行分条 (每条独立随机数; 云的高斯展开 > 0.3 像素时, 条带上下各多算 4σ 再裁掉)。"""
    sp = emulsion.spec
    rev = emulsion.curve.design == "reversal"
    h = mean.shape[0]
    sig = max(float(np.max(dye.cloud_sigma_um(sp, c))) for c in range(3)) / px_um
    margin = int(np.ceil(4 * sig)) if sig > 0.3 else 0
    out = np.empty_like(mean)
    for bi, r0 in enumerate(range(0, h, DYE_BAND_ROWS)):
        r1 = min(h, r0 + DYE_BAND_ROWS)
        a0, a1 = max(0, r0 - margin), min(h, r1 + margin)
        tgt = np.array(mean[a0:a1], float)
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, bi, 7])
        dye.apply_dye_grain(sp, tgt, px_um, rng, reversal=rev, strength=strength)
        out[r0:r1] = tgt[r0 - a0:r0 - a0 + (r1 - r0)]
    return out


# ---------------------------------------------------------------- 颗粒度 --
def granularity(emulsion, density, aperture_um=48.0, shape="circle", net=True, psf_um=0.0):
    """闭式 RMS 颗粒度 (密度单位; ×1000 即数据册数值)。density: 净密度 (net=True: 高于片基 + 灰雾; 否则不含片基的总密度)。
    aperture_um: 圆孔直径 (shape "circle", 取等面积方孔) 或方孔边长 ("square")。
    黑白: 闭式透射率协方差在孔径上积分 (漫射收集, 不相邻; T47 / T47b 的 Kodak 48 µm 口径)。
    彩色: 每层 48 µm Selwyn 值 (dye.layer_curves), 按面积律换到其他孔径 (孔径 ≫ 染料云时成立); 返回 (3,)。"""
    side = float(aperture_um) * (np.sqrt(np.pi) / 2 if shape == "circle" else 1.0)
    if emulsion.path == "silver":
        sp = emulsion.spec
        fog = float(silver.silver_density(sp, 0.0)) if net else 0.0
        H = float(exposure_from_density(emulsion, float(density) + fog))
        return float(silver.silver_granularity(sp, H, [side], psf_um=psf_um, adjacent=False)[0])
    sp = emulsion.spec
    Hx = emulsion.exposure_axis()
    rev = emulsion.curve.design == "reversal"
    out = []
    for c in range(3):
        D, s = dye.layer_curves(sp, c, Hx[:, c], reversal=rev)
        D = D - D.min() if net else D
        i = np.argsort(D)
        out.append(float(np.interp(density, D[i], s[i])) * np.sqrt(APERTURE_UM2 / (side * side)))
    return np.array(out)
