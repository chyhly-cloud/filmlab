"""真实相机 (实测光谱感光度) 与 CIE 理想观察者在各款灯下看到的照明色度之差。

FilmLab 假设相机 = CIE 观察者 (lights.camera_rgb)。线谱光源 (荧光灯、汞灯、部分 LED) 下真实相机的 RGB 会偏离,
偏离部分在光源识别 / 线性曝光映射里会被当成"表面颜色", 直接进入胶片曝光。
EPFL 的 Panasonic G81 没有公开的光谱感光度; 这里用 colour 自带的两台实测相机界定偏离的量级与方向。
相机处理: 日光白平衡 (D65 白中性) + 在 D65 下对自然反射率拟合的 3×3 (相机 → 线性 sRGB), 与 RAW 解码一致。

用法: python3 camera_metamerism.py
"""
import sys
import numpy as np
import colour
sys.path.insert(0, "/home/user/filmlab")
from filmlab import lights, spectral

W = spectral.WAVELENGTHS
SH = colour.SpectralShape(W[0], W[-1], W[1] - W[0])


def camera(name):
    m = colour.MSDS_CAMERA_SENSITIVITIES[name].copy().align(SH, extrapolator_kwargs={"method": "Constant", "left": 0, "right": 0})
    S = np.clip(m.values, 0, None)
    R = lights.training_reflectances()
    d65 = spectral.illuminant("D65")
    cam = (R * d65) @ S
    wb = (d65 @ S)
    ref = lights.camera_rgb(R * d65 * 1.0)          # 同样反射率在 CIE 观察者下的线性 sRGB (D65 中性)
    M = np.linalg.lstsq(cam / wb, ref, rcond=None)[0]      # 相机 (白平衡后) → sRGB, 不约束白点
    return lambda spec: ((spec @ S) / wb) @ M


def lc(rgb):
    return np.log(np.maximum(rgb[[0, 2]], 1e-6) / max(rgb[1], 1e-6))


if __name__ == "__main__":
    for cam_name in colour.MSDS_CAMERA_SENSITIVITIES:
        f = camera(cam_name)
        print(f"\n{cam_name}: 照明色度 (相机 − CIE 观察者), log(R/G)  log(B/G)   (+ = 相机看到更多红 / 蓝, 两者都负 = 相机看到更绿)")
        for c in lights.classes():
            for n, s in c.members:
                d = lc(f(s)) - lc(lights.camera_rgb(s))
                print(f"  {c.key:12s} {n:28s} {d[0]:+6.3f} {d[1]:+6.3f}")
