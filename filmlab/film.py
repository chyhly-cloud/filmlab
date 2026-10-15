"""S1 曝光阶段: 光谱曝光积分与胶片内部的光学效应 (散射、光晕)。

约定:
- 图像数组为 (H, W, 3), 通道顺序为乳剂层 R(青染料) G(品红染料) B(黄染料)。
- 曝光量 H 为线性, logH = log10(H)。
显影与冲洗见 chemistry.py。
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from . import spectral
from .stocks import Stock


# ---------------------------------------------------------------- 曝光 ----
def exposure_matrix(stock: Stock, scene_illuminant: str, scene_filter=None) -> np.ndarray:
    """线性 sRGB → 三层乳剂曝光量的 3×3 矩阵。

    H_layer = Σ_λ  [r·B_r(λ) + g·B_g(λ) + b·B_b(λ)] · I(λ) · F(λ) · 10^logS_layer(λ)
    其中 B 为基光谱, I 为拍摄光源, F 为镜头前滤镜, S 为乳剂光谱感光度。
    """
    light = spectral.illuminant(scene_illuminant).copy()
    if scene_filter is not None:
        light = light * scene_filter
    sens = 10.0 ** stock.log_sensitivity                      # (81, 3)
    return (spectral.rgb_basis() * light[:, None]).T @ sens   # (3 rgb, 3 layers)


def expose(rgb_lin, M):
    return np.clip(rgb_lin @ M, 1e-9, None)


# ------------------------------------------------------------- 光学效应 ----
def halation(H, strength, px_um, radius_um=300.0):
    """光晕: 强光穿透乳剂与片基, 在片基背面反射回来再次曝光。
    回到乳剂的主要是红光 (蓝绿已被上层吸收), 因此主要作用于底部的红感层。"""
    if strength <= 0:
        return H
    s = radius_um / px_um
    # 宽 + 窄两段高斯: 近似片基内的多次反射
    glow = 0.6 * gaussian_filter(H[..., 0], s) + 0.4 * gaussian_filter(H[..., 0], s * 0.35)
    out = H.copy()
    out[..., 0] += strength * glow
    out[..., 1] += strength * 0.08 * glow
    return out


def emulsion_scatter(H, px_um, sigma_um=4.0):
    """乳剂内部的光散射 (决定胶片 MTF)。"""
    s = sigma_um / px_um
    if s < 0.3:
        return H
    return gaussian_filter(H, (s, s, 0))
