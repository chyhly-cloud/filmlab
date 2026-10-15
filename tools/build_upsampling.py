"""预计算 S0 光谱重建表: CIE xy 色度 → 平滑反射光谱 (Jakob & Hanika 2019,
"A Low-Dimensional Function Space for Efficient Spectral Upsampling")。

每个光谱轨迹内的色度点求一条 sigmoid 形状的光谱 (物理上可实现, 值在 0..1),
再归一化为单位亮度 Y=1。运行时: 光谱 = Y × 表[x, y]。曝光对亮度线性, 所以二维表足够。
轨迹外的格点用最近的有效格点填充 (相当于色域映射)。

    python tools/build_upsampling.py   # 输出 filmlab/data/upsampling_xy64.npz
"""
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import colour  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

N = 64
XS = np.linspace(0.0, 0.75, N)
YS = np.linspace(0.0, 0.85, N)
SHAPE = colour.SpectralShape(380, 780, 5)
CMFS = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(SHAPE)
D65 = colour.SDS_ILLUMINANTS["D65"].copy().align(SHAPE)
YBAR = (CMFS.values[:, 1] * D65.values).sum()


def solve(xy):
    warnings.filterwarnings("ignore")
    for lum in (0.4, 0.15, 0.05, 0.015, 0.005):
        try:
            sd, err = colour.recovery.XYZ_to_sd_Jakob2019(
                colour.xyY_to_XYZ([xy[0], xy[1], lum]), CMFS, D65, additional_data=True)
        except Exception:
            continue
        if err < 0.05:
            r = sd.values
            return r / ((r * CMFS.values[:, 1] * D65.values).sum() / YBAR)
    return None


def inside_locus(px, py, locus):
    c = False
    for (x1, y1), (x2, y2) in zip(locus, np.roll(locus, -1, axis=0)):
        if (y1 > py) != (y2 > py) and px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1:
            c = not c
    return c


if __name__ == "__main__":
    locus = colour.XYZ_to_xy(np.clip(CMFS.values[2:66], 1e-9, None))
    pts = [(i, j) for i, x in enumerate(XS) for j, y in enumerate(YS) if y > 0.005 and inside_locus(x, y, locus)]
    with Pool(os.cpu_count()) as pool:
        res = pool.map(solve, [(XS[i], YS[j]) for i, j in pts], chunksize=8)
    table = np.zeros((N, N, 81), np.float32)
    valid = np.zeros((N, N), bool)
    for (i, j), r in zip(pts, res):
        if r is not None:
            table[i, j], valid[i, j] = r, True
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    table = table[idx[0], idx[1]]
    out = Path(__file__).resolve().parents[1] / "filmlab" / "data" / "upsampling_xy64.npz"
    np.savez_compressed(out, x=XS, y=YS, spectra=table, valid=valid)
    print(f"{valid.sum()} / {len(pts)} 个色度点求解成功 → {out}")
