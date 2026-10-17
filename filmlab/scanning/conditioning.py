"""S4e 输出调整: 冲印店风格 (经验拟合, 不属于胶片物理)。

只作用于最终的输出 RGB。参数只能由冲印店扫描拟合, 不得用来补偿上游 (输入、胶片、扫描光学) 的误差;
物理验证在关闭本级 (Params.output_conditioning=None) 的物理参考下进行。
"""
import numpy as np


def output_tint(rgb, tint):
    """扫描仪输出的灰平衡目标: 在 Lab 中按亮度加上 tint 的 a/b 偏移。tint: (L, a, b) 锚点。"""
    import colour
    shape = rgb.shape
    lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb.reshape(-1, 3), 0, None), apply_cctf_decoding=False))
    Ls, As, Bs = (np.array(v) for v in zip(*tint))
    if not np.any(As) and not np.any(Bs):
        return rgb
    lab[:, 1] += np.interp(lab[:, 0], Ls, As)
    lab[:, 2] += np.interp(lab[:, 0], Ls, Bs)
    return colour.XYZ_to_sRGB(colour.Lab_to_XYZ(lab), apply_cctf_encoding=False).reshape(shape)
