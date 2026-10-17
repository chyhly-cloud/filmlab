"""S4a 扫描仪的光学测量 (硬件)。

窄带 R/G/B 光源透过底片, 传感器读透射率 → 三通道扫描密度。窄带测量"看"染料与橙色遮罩的方式
和相纸的宽带感光不同。光源峰值 / 带宽为估计值 (厂商未公开)。
"""
import numpy as np

from ..stocks import WAVELENGTHS


def led_response(m):
    """通道光谱响应 (81, 3), 每列归一 (开门 = 1)。m: 带 led_peaks / led_widths 的扫描仪型号。"""
    L = np.stack([np.exp(-0.5 * ((WAVELENGTHS - p) / w) ** 2) for p, w in zip(m.led_peaks, m.led_widths)], 1)
    return L / L.sum(0)


def measure(T, m):
    """光谱透射率 (..., 81) → 扫描密度 (..., 3)。"""
    return -np.log10(np.clip(T @ led_response(m), 1e-6, None))
