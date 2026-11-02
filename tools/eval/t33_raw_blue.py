"""T33 A (第 3 节): G81 原始通道 (日光白平衡, 未经相机矩阵) 与解码 sRGB 在红像素上的比较。
用法: FILMLAB_DATA=… python3 t33_raw_blue.py R1-03-4 R1-00-1"""
import os, sys, numpy as np, rawpy, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]
import oracle_timing as ot
from filmlab.rawio import XYZ_TO_SRGB
from scipy import ndimage as ndi
for sid in sys.argv[1:]:
    rawp, _, _ = ot.sources(sid)
    with rawpy.imread(rawp) as raw:
        wb = list(raw.daylight_whitebalance); wb = wb[:3] + [wb[1] if wb[3] <= 0 else wb[3]]
        common = dict(gamma=(1, 1), no_auto_bright=True, output_bps=16, half_size=True, user_flip=None, user_wb=wb)
        xyz = raw.postprocess(output_color=rawpy.ColorSpace.XYZ, **common).astype(np.float64) / 65535
        cam = raw.postprocess(output_color=rawpy.ColorSpace.raw, **common).astype(np.float64) / 65535
        M = np.asarray(raw.rgb_xyz_matrix)[:3]
    srgb = xyz @ XYZ_TO_SRGB.T
    ok = (cam.max(-1) < 0.9) & (cam.min(-1) > 0.002)
    A = np.linalg.lstsq(cam[ok], srgb[ok], rcond=None)[0]           # 相机 (白平衡后) → sRGB, 本图拟合
    sm = lambda a: np.stack([ndi.uniform_filter(a[..., c], 5) for c in range(3)], -1)
    s, c = sm(srgb), sm(cam)
    g = np.median(s[ok & (np.ptp(np.log(np.clip(s, 1e-6, None)), -1) < 0.1)], 0)
    red = ok & (s[..., 1] / np.maximum(s[..., 0], 1e-9) < 0.15) & (s[..., 2] / np.maximum(s[..., 0], 1e-9) < 0.1) & (s.min(-1) > 0) & (s[..., 0] > 0.02)
    red = ndi.binary_erosion(red, iterations=2)
    print(sid, "红像素", red.sum(), " 相机→sRGB (本图拟合, 行 = 输出 R G B 的系数):\n", A.T.round(3))
    if red.sum():
        print("  sRGB  R=1:", np.median(s[red] / s[red][:, :1], 0).round(3), "  原始通道 R=1:", np.median(c[red] / c[red][:, :1], 0).round(3))
