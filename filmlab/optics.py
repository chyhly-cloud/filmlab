"""拍摄端光学 (不属于胶片): 镜头 / 柔光镜的眩光扩散 (T20)。

到达胶片之前的场景光里, 一小部分 (η) 被镜头表面、镀膜缺陷、灰尘或柔光镜散射, 按长尾分布铺开:
    K(r) = (β − 1) / (π r0²) · (1 + r²/r0²)^(−β)        (归一化的 Moffat 分布; β = 1.5 时远处 ∝ r⁻³, 表面散射的常见规律)
    光 → (1 − η)·光 + η·(光 ⊗ K)      (能量守恒: 均匀画面不变, 所以灰卡定光不受影响)
这是相机的性质, 不是胶片物理; 数码 RAW 已含数码相机自身的眩光, 这里加的是胶片相机 (及柔光镜) 额外的部分。
η、r0 没有测量值, 由网上公开的 CineStill 800T 扫描统计确定 (外观设定), 见 docs/tasks/T20_CAMERA_GLOW.md。"""
import numpy as np


def glare_kernel(px_um, r0_um, beta=1.5, r_max_um=4000.0):
    n = int(np.ceil(r_max_um / px_um))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1] * px_um
    k = (1.0 + (yy ** 2 + xx ** 2) / r0_um ** 2) ** (-beta)
    return k / k.sum()


def lens_glare(rgb, px_um, eta, r0_um, beta=1.5, r_max_um=4000.0):
    if eta <= 0:
        return rgb
    from scipy.signal import fftconvolve
    k = glare_kernel(px_um, r0_um, beta, r_max_um)
    n = k.shape[0] // 2
    out = np.empty_like(rgb, dtype=float)
    for c in range(rgb.shape[-1]):
        x = np.pad(np.asarray(rgb[..., c], float), n, mode="reflect")
        out[..., c] = fftconvolve(x, k, mode="valid")
    return (1.0 - eta) * rgb + eta * out
