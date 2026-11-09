"""跨片种通用的物理常数与单位 (T45–T53)。数值与 filmlab/grain.py (T54 版) 逐位相同。"""
import numpy as np

SIGMA_LN = 0.30                 # 晶体 ECD 的对数正态宽度 (T 颗粒乳剂典型变异系数约 0.3)
QUAD = 5                        # 对数正态的 Gauss–Hermite 求积点数
PHOTONS_R = 4                   # 可显影潜影所需的有效光子数 (潜影中心约 4 个银原子)
ETA_REF = 0.05                  # Φ = η × 波段光子数 的参考 η (吸收率 × 潜影量子效率; 量级估计, 只影响绝对晶体尺寸)
APERTURE_UM2 = np.pi * 24.0 ** 2   # 48 µm 圆孔 (Kodak 漫射 RMS 颗粒度的测量孔径)
# 每 lux·s 每 µm² 的光子数 (400–500 / 500–600 / 600–700 nm; 普朗克谱, 按 CIE 明视觉换算到 lux), 通道顺序 R G B
BAND_PHOTONS = {"BB3200": (6408.0, 3576.0, 1336.0), "D55": (4090.0, 3790.0, 2914.0)}
SURFACE_RATIO_3D = 3.0          # 立方体表面积 / T 颗粒两面面积 (同一投影面积): 6a / 2a

AGBR_AG_G_UM3 = 3.72e-12        # 每 µm³ AgBr 含银 (g): 6.47 g/cm³ × 108/188
AG_G_UM3 = 10.49e-12            # 金属银密度 (g/µm³)
_FINE_UM = 0.01                 # 银丝团几何的参考栅格 (µm)
FINE_UM = _FINE_UM

# T53 颗粒粗细 k 的允许范围 (docs/tasks/T53_GRAIN_COARSENESS.md)
GRAIN_SCALE_RANGE = (0.5, 3.0)


def check_grain_scale(k):
    k = float(k)
    lo, hi = GRAIN_SCALE_RANGE
    if not (np.isfinite(k) and lo <= k <= hi):
        raise ValueError(f"颗粒粗细 grain_scale = {k:g} 超出范围 [{lo:g}, {hi:g}]")
    return k
