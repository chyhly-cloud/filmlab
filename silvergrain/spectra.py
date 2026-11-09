"""照片的线性 RGB → 胶片每层的相对曝光 (光谱路径)。

物理定义: 把照片的线性 sRGB 值当作景物反射率 (中灰 0.18), 由 RGB 上采样出一条物理上可实现的平滑反射光谱
(Jakob & Hanika 2019 查找表, silvergrain/data/upsampling_xyY.npz, 与 filmlab.spectral.spectral_response 同一方法;
表不存在时退回 Mallett & Yuksel 2019 的线性基光谱, 只在 sRGB 色域内准确), 在胶片的设计光源下照明 (日光片 D55,
灯光片 3200 K 黑体), 可选镜头前 Wratten 滤镜 (T49 的实测透过率), 再按胶片每层自己的光谱感光度积分:
    E_k = Σ_λ ρ(λ) · S_illum(λ) · T_filter(λ) · 10^logS_k(λ)
黑白片: 一条全色光谱感光度 (T48 / T49b 的逐片数据, 来源见 bw_t48.json); 彩色片: 三层 (spektrafilm 档案)。
归一: 中性灰 (r = g = b = a) 在不带滤镜时得 E_k = a。带滤镜时默认按中性灰补偿 (compensate=True: 曝光乘以该片在该光源下
算出的滤镜系数, T49 检验这样算出的系数与 Kodak 公布值 RMS 0.23 档); compensate=False 时不补偿 (曝光不足, 同不加系数拍摄)。
照片本来已白平衡 (中性物体 r = g = b), 所以 "设计光源下的中性灰 = 中性" 正是照片的白平衡约定。
"""
from functools import lru_cache

import numpy as np

from . import database as db

_SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])
WAVELENGTHS = np.arange(380.0, 781.0, 5.0)


def illuminant(name):
    """(81,) 归一 (均值 1) 的光源光谱: D50 / D55 / D65 / BB3200 / BB2850 (由 filmlab 导出, colour-science)。"""
    return np.asarray(db.colour_films()["illuminants"][name], float)


def wratten(number):
    """Wratten 滤镜在 81 点上的透过率 (T49: Kodak Wratten Filters 1965 的表; 短于首个读数为 0, 长于 700 nm 保持 700 nm 的值)。"""
    aliases = {"k2": "8", "x1": "11", "g": "15", "a": "25"}
    k = str(number).lower().lstrip("w")
    k = aliases.get(k, k)
    f = db.wratten()["filters"][k]
    lam, t = np.asarray(f["lam"], float), np.asarray(f["T"], float)
    return np.where(WAVELENGTHS < lam[0], 0.0, np.interp(WAVELENGTHS, lam, t))


def sensitivity(film):
    """(81, k) 线性光谱感光度 (黑白 k = 1, 彩色 k = 3; 只有形状起作用)。"""
    lam, logS = film.spectral()
    lam = np.asarray(lam, float)
    logS = np.asarray(logS, float)
    if logS.ndim == 1:
        ls = np.where(WAVELENGTHS <= lam.max(), np.interp(WAVELENGTHS, lam, logS), -10.0)
        return 10.0 ** ls[:, None]
    return 10.0 ** logS


def _response_fn(weights):
    """线性 RGB → Σ_λ 光谱(λ)·weights(λ, k) (filmlab.spectral.spectral_response 的同一算法)。"""
    tab = db.upsampling_table()
    if tab is None:
        raise RuntimeError("缺少 silvergrain/data/upsampling_xyY.npz")
    xs, ys, lums, spectra = tab
    G = spectra @ weights
    from scipy.ndimage import map_coordinates
    w = _SRGB_TO_XYZ.sum(1)
    white = (w[0] / w.sum(), w[1] / w.sum())
    loglum = np.log(lums)

    def f(rgb):
        rgb = np.asarray(rgb, float)
        xyz = rgb @ _SRGB_TO_XYZ.T
        Y = np.clip(xyz[..., 1], 0, None)
        ssum = xyz.sum(-1)
        ok = ssum > 1e-9
        x = np.where(ok, xyz[..., 0] / np.where(ok, ssum, 1), white[0])
        y = np.where(ok, xyz[..., 1] / np.where(ok, ssum, 1), white[1])
        ci = np.clip(x / xs[-1] * (len(xs) - 1), 0, len(xs) - 1)
        cj = np.clip(y / ys[-1] * (len(ys) - 1), 0, len(ys) - 1)
        ck = np.interp(np.log(np.maximum(Y, 1e-6)), loglum, np.arange(len(lums)))
        coords = [ci.ravel(), cj.ravel(), ck.ravel()]
        out = np.stack([map_coordinates(G[..., k], coords, order=1, mode="nearest")
                        for k in range(G.shape[-1])], -1).reshape(rgb.shape[:-1] + (G.shape[-1],))
        return out * Y[..., None]
    return f


def layer_response(film, filter=None, compensate=True, light=None):
    """返回函数: 线性 RGB (…, 3) → 每层相对曝光 (…, k) (k = 1 黑白 / 3 彩色), 中性灰 a → a (不带滤镜或 compensate=True)。
    light: 拍摄光源 (默认胶片的设计光源)。返回 (函数, 信息 dict: 滤镜系数 (中性灰, 按档) 等)。"""
    light = light or film.illuminant
    S = sensitivity(film)
    E = illuminant(light)[:, None]
    T = wratten(filter)[:, None] if filter is not None else 1.0
    f0 = _response_fn(E * S)
    f1 = _response_fn(E * S * T) if filter is not None else f0
    ref = f0(np.ones(3) * 0.18) / 0.18                 # 中性灰在不带滤镜时的响应 (每层)
    refT = f1(np.ones(3) * 0.18) / 0.18
    factor = ref / refT                                # 中性灰的滤镜系数 (每层; 曝光倍数)
    norm = refT if (filter is not None and compensate) else ref
    info = dict(light=light, filter=filter, compensate=compensate,
                filter_factor=factor.tolist(), filter_stops=np.log2(factor).tolist())
    return (lambda rgb: f1(rgb) / norm), info


@lru_cache(maxsize=None)
def _cached(film_key, filter, compensate, light):
    from .films import film
    return layer_response(film(film_key), filter, compensate, light)
