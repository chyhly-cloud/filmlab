"""S4f 冲印店扫描的空间传递 (T51, docs/tasks/T51_LAB_SCANNER_SPATIAL.md)。

扫描仪与冲印店软件对图像的空间作用, 按真实处理顺序:
  1. 扫描镜头 / 传感器的光学点扩散 (高斯 σ_o, 在透射率上; 由 engine._scan_psf 执行)。实测的是按输出像素频率的传递,
     所以 σ_o 也以输出像素计 (扫描仪的光学与它的采样匹配; 同一底片两种尺寸的配对扫描未验证, 见 T51 第 11 节)。
  2. 传感器孔径与采样: 每个输出像素对其面积积分 (方盒, 边长 = 冲印店输出像素间距)。渲染像素本身就是对像素面积的积分,
     所以渲染间距 ≥ 冲印店间距时不再加; 渲染更细时按冲印店间距加方盒 (不抽取, 输出尺寸不变)。
  3. 冲印店软件 (在编码后的输出上, BT.601 亮度 / 色度; 软件按输出像素工作, 半径以输出像素计):
     · 亮度锐化 (反锐化掩模, USM): Y' = Y + a·d, d = Y − G_σu * Y; 阈值 t_u > 0 时 d 乘 1 − exp(−(d/t_u)²) (小于 t_u 的细节不锐化);
     · 色度降噪: Cb、Cr 各做高斯低通 G_σc; 阈值 t_c > 0 时为保边形式 C' = G*C + d_c·(1 − exp(−(d_c/t_c)²)),
       d_c = C − G*C: 小幅度的色度起伏 (颗粒) 被平滑, 大的色度边 (|d_c| ≫ t_c) 保留。只作用于彩色; 黑白片的冲印店输出为灰度。
  参数是器件的物理量 (扫描光学 σ 与软件的半径以输出像素计; 量; 阈值以 8 位码值计), 由真实扫描的直边 (ESF)
  与颗粒功率谱联合实测, 每家冲印店一组; 不是输出端的自由滤波, 不加噪声, 不改颜色 (各项直流增益为 1: 均匀区与大面积颜色不变)。
  渲染尺寸应等于冲印店的输出尺寸 (35 mm: Frontier L 约 3089 × 2048, Noritsu 标准约 3000 × 2000); 渲染更细时按冲印店像素间距加
  传感器方盒孔径 (不抽取), 软件的半径仍按渲染像素计。
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

# BT.601 (JPEG / 冲印店软件常用的亮度-色度) 在编码 sRGB 上
_YCC = np.array([[0.299, 0.587, 0.114],
                 [-0.168736, -0.331264, 0.5],
                 [0.5, -0.418688, -0.081312]])
_YCC_INV = np.linalg.inv(_YCC)


@dataclass(frozen=True)
class LabSpatial:
    name: str
    pitch_um: float           # 冲印店 35 mm 输出的像素间距 (胶片面, µm) —— 传感器孔径
    optics_px: float          # 扫描光学点扩散高斯 σ (输出像素; 透射率上)
    usm_sigma_px: float       # USM 模糊高斯 σ (输出像素)
    usm_amount: float         # USM 量 a (0 = 不锐化)
    usm_threshold: float      # USM 阈值 (8 位码值; 0 = 线性)
    chroma_sigma_px: float    # 色度低通高斯 σ (输出像素; 0 = 无色度降噪)
    chroma_threshold: float   # 色度降噪阈值 (8 位码值; 0 = 线性高斯低通)
    source: str = ""


# 实测值见 docs/tasks/T51_LAB_SCANNER_SPATIAL.md 与 tools/eval/t51/fit.json (t51_fit.py)。
LABS = {
    # EPFL CineStill 800T 冲印店 (Fujifilm Frontier, 8 位未压缩 TIFF, 6.60 µm/像素)。
    #   直边 (色卡 45 条): USM a = 0.007 (≈ 0, 不锐化); 颗粒亮度谱: 高斯光学 σ_o = 6.0 µm = 0.91 像素, 无锐化;
    #   颗粒按频率的通道相干: 色度低通 σ_c = 1.75 像素 (4 帧测试卡; 3 帧夜景留出同值, 相干误差 0.13 对无降噪 0.42);
    #   强色度边 (Cb 台阶 46–114 码值) 不展宽
    #   → 保边形式, 阈值 8 码值 (由两侧界定, 非拟合: ≥ 6 才能把 2–3 码值的颗粒色度功率压到 0.15–0.3; ≤ 10 色卡边才不展宽 > 25%)。
    #   35 mm L 尺寸 (约 3089 × 2048) 的像素间距 11.65 µm 作孔径; 软件半径按输出像素 (在 6.6 µm/像素的文件上测得)。
    "frontier_epfl": LabSpatial("Fujifilm Frontier (EPFL 800T 冲印店, TIFF)", 11.65, 0.91, 1.0, 0.0, 0.0, 1.75, 8.0,
                                "T51: tools/eval/t51/fit.json frontier"),
    # Richard Photo Lab (Noritsu HS-1800, EZ Controller, JPEG; 645 中等尺寸 2048 × 2796, 约 20 µm/像素)。
    #   直边 (训练片 Pro 400H / Portra 400 / 160, 85 条): USM a = 1.03, σ_u = 0.68 像素; 留出片 (Portra 800、Ektar 100)
    #   的颗粒亮度谱由它预测, 与自由拟合同样好 (RMS 0.078 对 0.077; 不锐化 0.232); 颗粒给出的 USM 与直边一致 → 无阈值。
    #   光学 σ_o = 7 µm = 0.35 像素 (颗粒谱在 USM 固定后定; 最清晰直边的总模糊 11 µm 为上限)。色度: 4:2:0 JPEG 之外没有可测的色度降噪
    #   (相干剖面最优 0–0.5 像素, 与无降噪的误差相同), 取 0。
    #   35 mm 标准尺寸约 3000 × 2000 (12 µm/像素) 作孔径。
    "noritsu_rpl": LabSpatial("Noritsu HS-1800 (Richard Photo Lab, JPEG)", 12.0, 0.35, 0.68, 1.03, 0.0, 0.0, 0.0,
                              "T51: tools/eval/t51/fit.json noritsu"),
}


def encode(lin):
    """线性 → sRGB 编码 (0–1)。> 1 的值按同一幂律延伸 (不截断), < 0 截为 0。"""
    x = np.clip(lin, 0.0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def decode(v):
    v = np.clip(v, 0.0, None)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def _blur(a, s):
    return gaussian_filter(a, s, mode="reflect") if s >= 0.05 else a


def usm(Y, sigma_px, amount, threshold=0.0):
    """反锐化掩模 (码值 0–1 尺度上的 Y)。threshold: 8 位码值。"""
    if amount <= 0 or sigma_px < 0.05:
        return Y
    d = Y - _blur(Y, sigma_px)
    if threshold > 0:
        t = threshold / 255.0
        d = d * (1.0 - np.exp(-(d / t) ** 2))
    return Y + amount * d


def aperture(rgb, lab: LabSpatial, px_um):
    """冲印店传感器的像素孔径 (渲染比冲印店细时)。"""
    k = lab.pitch_um / px_um
    if k <= 1.2:
        return rgb
    return np.stack([uniform_filter(rgb[..., c], k, mode="reflect") for c in range(rgb.shape[-1])], -1)


def apply(rgb_lin, lab: LabSpatial, px_um, mono=False):
    """线性输出 RGB (H, W, 3) → 冲印店软件处理后的线性 RGB。"""
    rgb = aperture(np.asarray(rgb_lin, float), lab, px_um)
    v = encode(rgb)
    su, sc = lab.usm_sigma_px, lab.chroma_sigma_px
    if mono:
        Y = usm(v.mean(-1), su, lab.usm_amount, lab.usm_threshold)
        return decode(np.repeat(Y[..., None], 3, -1))
    ycc = v @ _YCC.T
    ycc[..., 0] = usm(ycc[..., 0], su, lab.usm_amount, lab.usm_threshold)
    if sc >= 0.05:
        for k in (1, 2):
            ycc[..., k] = chroma_nr(ycc[..., k], sc, lab.chroma_threshold)
    return decode(ycc @ _YCC_INV.T)


def chroma_nr(C, sigma_px, threshold=0.0):
    """色度低通 (码值 0–1 尺度上的 Cb 或 Cr); threshold > 0 时保边 (8 位码值)。"""
    low = _blur(C, sigma_px)
    if threshold <= 0:
        return low
    d = C - low
    return low + d * (1.0 - np.exp(-(d / (threshold / 255.0)) ** 2))
