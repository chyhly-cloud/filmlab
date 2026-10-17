"""光谱工具: 光源、CIE 观察者、RGB→光谱上采样、滤色片。所有光谱都在 380–780nm / 5nm 网格上。"""
from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np

warnings.filterwarnings("ignore", module="colour")
import colour  # noqa: E402

from .stocks import WAVELENGTHS

SHAPE = colour.SpectralShape(380, 780, 5)
FILTER_DIR = Path(__file__).parent / "data" / "filters"


def _sd(sd):
    return sd.copy().align(SHAPE).values


def _blackbody(T):
    return colour.colorimetry.blackbody_spectral_radiance(WAVELENGTHS * 1e-9, T)


def _csv_curve(name):
    d = np.loadtxt(FILTER_DIR / name, delimiter=",")
    v = np.interp(WAVELENGTHS, d[:, 0], d[:, 1])
    return np.clip(v / 100.0 if v.max() > 1.5 else v, 0, 1)


@lru_cache(maxsize=None)
def illuminant(name: str) -> np.ndarray:
    """返回归一化 (均值=1) 的光源光谱功率分布。"""
    if name.startswith("BB"):
        s = _blackbody(float(name[2:]))
    elif name == "T":                       # 钨丝灯, 灯光型胶片的参考光源
        s = _sd(colour.SDS_LIGHT_SOURCES["Incandescent"])
    elif name == "K75P":                    # 电影放映机氙灯
        s = _sd(colour.SDS_LIGHT_SOURCES["Kinoton 75P"])
    elif name == "TH-KG3":                  # 放大机: 卤素灯 + Schott KG3 隔热玻璃
        s = _blackbody(3400) * _csv_curve("schott_kg3.csv")
    else:                                   # D50 / D55 / D65 / A ...
        s = _sd(colour.SDS_ILLUMINANTS[name])
    return s / s.mean()


@lru_cache(maxsize=None)
def cmfs() -> np.ndarray:
    """CIE 1931 2° 标准观察者, (81, 3)。"""
    return _sd(colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"])


@lru_cache(maxsize=None)
def rgb_basis() -> np.ndarray:
    """Mallett & Yuksel (2019) sRGB 基光谱 (81, 3)。
    线性 sRGB (r,g,b) → 反射率光谱 r*B_r + g*B_g + b*B_b, 白 (1,1,1) → 平坦 1.0。
    因为是线性的, 胶片曝光 = 3×3 矩阵 × RGB, 速度极快。"""
    return _sd(colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019)


# ------------------------------------------------- S0 光谱重建 --
_SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])


@lru_cache(maxsize=None)
def upsampling_table():
    """Jakob-Hanika 2019 (x, y, 亮度) → 光谱表 (tools/build_upsampling.py 生成)。不存在时返回 None。"""
    path = FILTER_DIR.parent / "upsampling_xyY.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return d["x"], d["y"], d["lum"], d["spectra"].astype(np.float64)


def spectral_response(weights: np.ndarray):
    """返回函数: 场景线性 RGB → Σ_λ 光谱(λ) · weights(λ, k)。

    weights 已包含光源、滤镜和接收者 (乳剂感光度等)。
    有重建表时: 由像素的色度与亮度 (按反射率理解, 中灰 0.18) 查出一条物理上可实现的平滑光谱,
    色域外的颜色也正确; 超出反射物体颜色极限的高光取该色度下最亮的可实现光谱。
    没有表时: 退回 Mallett-Yuksel 基光谱 (线性, 只在 sRGB 色域内准确)。"""
    tab = upsampling_table()
    if tab is None:
        M = rgb_basis().T @ weights
        return lambda rgb: rgb @ M
    xs, ys, lums, spectra = tab
    G = spectra @ weights                      # (nx, ny, nY, k): 单位亮度的响应
    from scipy.ndimage import map_coordinates
    white = spectral_white_xy()
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


def spectral_white_xy():
    w = _SRGB_TO_XYZ.sum(1)
    return w[0] / w.sum(), w[1] / w.sum()


@lru_cache(maxsize=None)
def dichroic_filters() -> np.ndarray:
    """放大机分色滤镜 (Durst) 的透射率, (81, 3) 列为 C M Y。"""
    return np.stack([_csv_curve(f"durst_{c}.csv") for c in "cmy"], axis=1)


def enlarger_filtration(cmy) -> np.ndarray:
    """放大机 CMY 滤色: 每个滤镜按"强度"指数衰减 (相当于叠加多片)。"""
    T = dichroic_filters()
    return np.prod(np.power(np.clip(T, 1e-4, 1), np.asarray(cmy)[None, :]), axis=1)


def spectrum_to_xyz(spec: np.ndarray, illum: np.ndarray) -> np.ndarray:
    """透射/反射光谱 (..., 81) 在光源 illum 下 → XYZ, 以完美白 Y=1 归一化。"""
    w = cmfs() * illum[:, None]
    return spec @ w / w[:, 1].sum()


def xyz_to_srgb(xyz: np.ndarray, white_xy) -> np.ndarray:
    """XYZ (相对白点 white_xy) → 经 Bradford 适应到 D65 → 线性 sRGB。"""
    d65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    M = colour.adaptation.matrix_chromatic_adaptation_VonKries(
        colour.xy_to_XYZ(white_xy), colour.xy_to_XYZ(d65), transform="Bradford")
    M = colour.models.RGB_COLOURSPACE_sRGB.matrix_XYZ_to_RGB @ M
    return xyz @ M.T


def white_xy(illum: np.ndarray):
    return colour.XYZ_to_xy(spectrum_to_xyz(np.ones(81), illum))


def srgb_encode(lin):
    return colour.cctf_encoding(np.clip(lin, 0, 1), function="sRGB")
