"""S0 场景照明: 人造光源的真实光谱。

数码相机只记录三个数, 3000K 的荧光灯和 3000K 的白炽灯在它看来几乎一样 (同色异谱),
但胶片的三层乳剂看到的完全不同: 荧光灯的汞线与荧光粉尖峰、钠灯 589nm 附近的窄峰、
LED 的蓝光峰 —— 这正是 CineStill 800T 在荧光灯下偏青绿、在钠灯下呈特有橙色的原因。
用平滑光谱近似整个场景会丢掉这些。

做法 (都是物理量):
  1. 估计每个区域的照明颜色: 大范围、按亮度加权的局部平均色度。
  2. 与各类光源 (实测光谱, CIE / colour-science 数据) 的颜色比较, 得到软权重 ——
     RGB 区分不了的同色异谱光源, 就在物理上合理的几类之间混合。同一类里具体是哪款灯,
     按每张底片的种子在范围内随机。
  3. 像素 = 反射率 × 光源光谱: 先按光源颜色除去照明 (von Kries), 重建平滑的反射光谱,
     再乘真实光源光谱计算胶片曝光。
"""
from dataclasses import dataclass
from functools import lru_cache
import warnings

import numpy as np

warnings.filterwarnings("ignore", module="colour")
import colour  # noqa: E402

from . import spectral  # noqa: E402
from .stocks import WAVELENGTHS  # noqa: E402

SHAPE = colour.SpectralShape(380, 780, 5)


def _sd(name):
    src = colour.SDS_ILLUMINANTS.get(name) or colour.SDS_LIGHT_SOURCES[name]
    v = src.copy().align(SHAPE).values
    return np.clip(v, 0, None) / max(v.max(), 1e-9)


def _blackbody(T):
    v = colour.colorimetry.blackbody_spectral_radiance(WAVELENGTHS * 1e-9, T)
    return v / v.max()


# 光源类别 → 该类的实测光谱 (名称, 光谱)。先验权重反映城市夜景 / 室内的常见程度 (估计值)。
@dataclass(frozen=True)
class LightClass:
    key: str
    name: str
    members: tuple          # ((名称, 光谱), ...)
    prior: float


@lru_cache(maxsize=None)
def classes():
    daylight = tuple((f"D{t}", _sd(f"D{t}")) for t in (50, 55, 65, 75))
    # 白光钠灯 (Super HPS, 商店照明) 的颜色与 2700K 白炽灯几乎重合、光谱也是宽带暖光, 归入暖光一类;
    # "高压钠灯"一类只保留街道上常见的标准钠灯 (颜色明显更橙, 可与白炽灯区分)
    tungsten = tuple((f"{T}K 白炽 / 卤素", _blackbody(T)) for T in (2400, 2700, 3000, 3300)) \
        + (("SDW-T 100W/LV (Super HPS)", _sd("SDW-T 100W/LV (Super HPS)")),)
    hps = tuple((n, _sd(n)) for n in ("HPS", "C100S54 (HPS)", "HP1"))
    fluor = tuple((n, _sd(n)) for n in ("Cool White FL", "Triphosphor FL", "F32T8/TL830 (Triphosphor)",
                                        "F32T8/TL841 (Triphosphor)", "FL2", "FL11", "Philips TL-84",
                                        "F34T12WW/RS/EW (Warm White FL)"))
    led = tuple((n, _sd(n)) for n in ("LED-B1", "LED-B2", "LED-B3", "LED-B4", "LED-B5",
                                      "Phosphor LED YAG", "Luxeon WW 2880"))
    mh = tuple((n, _sd(n)) for n in ("Mercury", "Metal Halide", "MHC100/U/MP/4K", "HP3"))
    return (
        LightClass("daylight", "日光 / 天光", daylight, 1.0),
        LightClass("tungsten", "白炽灯 / 卤素灯 / 暖光", tungsten, 1.0),
        LightClass("hps", "高压钠灯", hps, 0.8),
        LightClass("fluorescent", "荧光灯", fluor, 0.8),
        LightClass("led", "白光 LED", led, 0.8),
        LightClass("metal_halide", "汞灯 / 金卤灯", mh, 0.4),
    )


# ------------------------------------------------------------ 颜色 --
_M_SRGB = colour.models.RGB_COLOURSPACE_sRGB.matrix_XYZ_to_RGB


# RAW 日光白平衡下视为中性的光源。dcraw / libraw 的 daylight_whitebalance 由相机矩阵与 sRGB (D65) 求得,
# 实测 X100V / G81 / ARW 三台相机均使 D65 中性 (tools/eval/CRITERIA.md "RAW → 曝光审计")。
# 默认仍为 "D55": 下游 SATURATION_MATRIX / LAB_TINT 是在 D55 假设下标定的, 改为 D65 后 RAW 配对 10/12 改善,
# 但 ref3 B 在 3 个随机种子下均变差 (+0.3…+1.2), 按验证规则暂不作为默认。
WB_REFERENCE = "D55"


def _daylight_wb():
    return _wb_for(WB_REFERENCE)


@lru_cache(maxsize=None)
def _wb_for(ref):
    """RAW 的日光白平衡: ref 白 → sRGB 白 (Bradford; ref = D65 时无需适应)。"""
    w = spectral.illuminant(ref) @ spectral.cmfs()
    if ref == "D65":
        return _M_SRGB, w[1]
    d65 = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
    cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(w / w[1], d65, "Bradford")
    return _M_SRGB @ cat, w[1]


def camera_rgb(spec):
    """光谱辐射 (..., 81) 在日光白平衡下的线性 sRGB (理想相机 = CIE 观察者, 中灰白 = WB_REFERENCE 的 Y)。"""
    M, wy = _daylight_wb()
    return (spec @ spectral.cmfs()) / wy @ M.T


def light_rgb(spec):
    """光源照在完美白上的颜色, 亮度归一 (Y=1)。"""
    rgb = camera_rgb(spec)
    return rgb / (rgb @ np.array([0.2126, 0.7152, 0.0722]))


def _logchroma(rgb):
    rgb = np.maximum(rgb, 1e-6)
    return np.stack([np.log(rgb[..., 0] / rgb[..., 1]), np.log(rgb[..., 2] / rgb[..., 1])], -1)


# ------------------------------------------------------ 照明估计 --
def illumination_map(rgb, px_scale):
    """每个像素所在区域的照明颜色 (对数色度)。
    在线性光里做按亮度加权的大范围平均 (局部的"灰色世界"), 再取色度: 照明是光的叠加,
    应在线性域平均; 若先取对数, 暗处接近零的通道会被放大到 −5 以下, 把估计拉偏。"""
    from scipy.ndimage import gaussian_filter
    lin = np.clip(rgb, 0, None)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    avg = np.stack([gaussian_filter(lin[..., c], px_scale) for c in range(3)], -1)
    return _logchroma(np.maximum(avg, 1e-9 + 1e-6 * gaussian_filter(Y, px_scale)[..., None]))


@lru_cache(maxsize=None)
def class_gamuts():
    """每类光源能照出的颜色范围 (色域): 几千种自然反射率在该类各款灯下的对数色度, 取凸包。
    色域约束的色彩恒常性 (Forsyth 1990): 画面里出现了某光源照不出来的颜色, 就排除这种光源 ——
    例如钠灯几乎是单色光, 在它下面不可能出现饱和的蓝色和绿色。"""
    from scipy.spatial import Delaunay
    R = training_reflectances()
    out = []
    for c in classes():
        pts = np.vstack([_logchroma(np.clip(camera_rgb(R * s[None, :]), 1e-6, None)) for _, s in c.members])
        out.append(Delaunay(pts))
    return out


def feasibility(rgb, px_scale, margin=0.15):
    """每个区域中, 有多少 (按亮度加权的) 像素颜色是该类光源照得出来的 → (h, w, 类别数), 0..1。"""
    from scipy.ndimage import gaussian_filter
    Y = np.clip(rgb @ np.array([0.2126, 0.7152, 0.0722]), 1e-6, None)
    w = np.sqrt(Y) * (Y > np.percentile(Y, 20))                      # 太暗的像素颜色不可靠
    lc = _logchroma(np.clip(rgb, 1e-6, None)).reshape(-1, 2)
    out = []
    for tri in class_gamuts():
        inside = tri.find_simplex(lc) >= 0
        if margin > 0:                                                 # 容差: 色域边缘附近也算可行
            for dx, dy in ((margin, 0), (-margin, 0), (0, margin), (0, -margin)):
                inside |= tri.find_simplex(lc + np.array([dx, dy])) >= 0
        f = inside.reshape(Y.shape).astype(float)
        out.append(gaussian_filter(f * w, px_scale) / np.maximum(gaussian_filter(w, px_scale), 1e-9))
    return np.stack(out, -1)


def class_weights(ill_lc, sigma=0.25, feas=None, sharpness=6.0):
    """照明色度 → 各类光源的软权重 (按到该类各成员颜色的最近距离, 高斯似然 × 先验)。"""
    cl = classes()
    logl = []
    for c in cl:
        pts = _logchroma(np.array([light_rgb(s) for _, s in c.members]))
        d2 = ((ill_lc[..., None, :] - pts) ** 2).sum(-1).min(-1)
        logl.append(-0.5 * d2 / sigma ** 2 + np.log(c.prior))
    L = np.stack(logl, -1)
    if feas is not None:                 # 区域里照不出来的颜色越多, 越不可能是这种光源
        L = L + sharpness * np.log(np.clip(feas, 1e-3, 1.0))
    L -= L.max(-1, keepdims=True)
    w = np.exp(L)
    return w / w.sum(-1, keepdims=True)


def nearest_members(ill_lc, W):
    """每类光源: 取与该类所照区域 (按类别权重加权) 的平均照明色度最接近的一款。
    与 class_weights 用同一判据 (到成员颜色的最近距离), 让光源识别与曝光计算用同一款灯。"""
    out = {}
    lc = ill_lc.reshape(-1, 2)
    for k, c in enumerate(classes()):
        w = W.reshape(-1, W.shape[-1])[:, k]
        m = (lc * w[:, None]).sum(0) / max(w.sum(), 1e-9)
        pts = _logchroma(np.array([light_rgb(sp) for _, sp in c.members]))
        out[c.key] = c.members[int(np.argmin(((pts - m) ** 2).sum(1)))]
    return out


def pick_members(seed):
    """每类光源里具体是哪款灯: 按每张底片的种子随机 (物理上都合理)。"""
    rng = np.random.default_rng(seed)
    return {c.key: c.members[rng.integers(len(c.members))] for c in classes()}


@lru_cache(maxsize=None)
def training_reflectances(n=3000, seed=7):
    """自然界典型的平滑反射光谱 (Jakob-Hanika sigmoid 族, 系数随机, 含中性灰阶), 用于拟合。"""
    rng = np.random.default_rng(seed)
    L = (WAVELENGTHS - 380.0) / 400.0
    c = np.column_stack([rng.normal(0, 25, n), rng.normal(0, 25, n), rng.normal(0, 3, n)])
    c[:, 1] -= c[:, 0] * rng.uniform(0.8, 1.2, n)            # 让峰 / 谷落在可见光范围内
    z = c @ np.stack([L ** 2, L, np.ones_like(L)])
    R = 0.5 + z / (2.0 * np.sqrt(1.0 + z * z))
    R *= rng.uniform(0.05, 0.95, (n, 1)) / np.maximum(R.max(1, keepdims=True), 1e-6)
    gray = np.linspace(0.02, 0.9, 40)[:, None] * np.ones((1, 81))
    return np.vstack([R, gray])


def linear_response(light_spec, receiver_weights, white_preserving=False):
    """返回函数: 场景线性 RGB (日光白平衡) → 该光源下接收者 (乳剂各层等) 的响应。

    用几千条自然界典型的平滑反射光谱作训练集, 拟合 "相机 RGB → 反射率 × 光源光谱 × 接收者"
    的 3×3 线性映射 (类似相机特性化)。比"先除光源再重建光谱"稳定: 钠灯这类强色光下
    蓝通道几乎没有信号, 除法会把噪声放大十几倍; 线性映射在不同光源之间也能平滑混合。

    系数约束为非负 (NNLS): 任何颜色的光多一些, 各层曝光只会增加, 不会减少。不加约束时,
    钠灯这类近单色光下训练样本的相机颜色几乎共线, 最小二乘会给出正负相消的大系数,
    遇到训练范围外的像素 (镜头眩光、重建的高光、噪声) 就会外推出离谱的曝光 (例如灯芯的
    红层曝光为 0、周围蓝层却被曝光)。"""
    R = training_reflectances()
    rad = R * light_spec[None, :]
    rgb = camera_rgb(rad)
    scale = 1.0 / (camera_rgb(light_spec) @ np.array([0.2126, 0.7152, 0.0722]))
    target = (rad * scale) @ receiver_weights
    rgb = rgb * scale
    from scipy.optimize import nnls
    w = 1.0 / np.maximum(target.mean(1, keepdims=True), 1e-6)     # 相对误差
    M = np.stack([nnls(rgb * w, target[:, k] * w[:, 0])[0] for k in range(target.shape[1])], 1)
    if white_preserving:
        # 保白点 (Finlayson & Drew 1996): 中性反射率在该光源下的响应必须精确 (可直接积分得到);
        # 无约束的拟合在 3200K 下会让中灰的蓝层曝光偏高 0.06 log。按列缩放, 保持非负。
        wt = (light_spec * scale) @ receiver_weights
        wp = (camera_rgb(light_spec) * scale) @ M
        M = M * (wt / np.maximum(wp, 1e-12))[None, :]
    return lambda x: np.asarray(x) @ M


def response_under(light_spec, receiver_weights):
    """返回函数: 场景线性 RGB (日光白平衡) → Σ_λ 反射率(λ)·光源(λ)·接收者(λ, k)。
    先按该光源的颜色除去照明 (von Kries) 得到反射率, 再重建平滑反射光谱。"""
    ill = light_rgb(light_spec)
    # 光源光谱的绝对标度: 照在完美白上时, 相机看到的亮度 = 1 (与 ill 的归一一致)
    scale = 1.0 / (camera_rgb(light_spec) @ np.array([0.2126, 0.7152, 0.0722]))
    f = spectral.spectral_response(light_spec[:, None] * scale * receiver_weights)
    # spectral_response 以 D65 下的反射率理解 RGB, 反射率白 (1,1,1) → 平坦光谱, 与此一致
    return lambda rgb: f(rgb / ill)
