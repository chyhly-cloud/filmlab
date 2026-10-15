"""预计算 S0 光谱重建表: (CIE x, y, 亮度 Y) → 平滑反射光谱 (Jakob & Hanika 2019,
"A Low-Dimensional Function Space for Efficient Spectral Upsampling")。

光谱形状: s(λ) = sigmoid(c0·l² + c1·l + c2), l 为归一化波长; 值域 (0,1), 平滑、物理上可实现。
同一色度在不同亮度下的光谱形状不同 (越亮的饱和色光谱带越宽), 所以和原论文一样用三维表:
色度 (x, y) × 反射率亮度 Y。太亮而物理上不可能的饱和色, 取该色度下能达到的最亮光谱。
存储时每条光谱归一化到 Y=1, 运行时: 光谱 = Y × 表[x, y, Y]。

求解: 先在系数空间铺一张候选网格 (约 13 万条光谱), 为每个目标挑最近的候选作初值 (避免局部极小),
再用向量化的 Levenberg-Marquardt (解析雅可比) 精修。

    python tools/build_upsampling.py     # 约 1 分钟, 输出 filmlab/data/upsampling_xyY.npz
    python tools/check_upsampling.py     # 用 ColorChecker 真实光谱检验
"""
from pathlib import Path
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import colour  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

N = 48
XS = np.linspace(0.0, 0.75, N)
YS = np.linspace(0.0, 0.85, N)
LUMS = np.geomspace(0.01, 0.95, 12)          # 反射率亮度 (中灰 0.18)
WL = np.arange(380.0, 781.0, 5.0)
L = (WL - 380.0) / 400.0
SHAPE = colour.SpectralShape(380, 780, 5)
CMFS = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(SHAPE).values
D65 = colour.SDS_ILLUMINANTS["D65"].copy().align(SHAPE).values
W = CMFS * D65[:, None]
W = W / W[:, 1].sum()                         # 完美白 Y = 1
BASIS = np.stack([L ** 2, L, np.ones_like(L)], 1)


def spectra(c):
    z = c @ BASIS.T
    return 0.5 + z / (2.0 * np.sqrt(1.0 + z * z))


def residual_and_jacobian(c, t):
    """残差 = (XYZ − 目标) / 目标亮度, 及解析雅可比。"""
    z = c @ BASIS.T
    sp = 0.5 + z / (2.0 * np.sqrt(1.0 + z * z))
    ds = 0.5 / (1.0 + z * z) ** 1.5
    xyz = sp @ W
    J = np.einsum("nl,lk,lj->njk", ds, BASIS, W)
    s = t[:, 1:2]
    return (xyz - t) / s, J / s[:, :, None]


def lm(c, t, iters=400):
    lam = np.full(len(c), 1e-3)
    r, J = residual_and_jacobian(c, t)
    for _ in range(iters):
        JT = np.transpose(J, (0, 2, 1))
        JTJ = JT @ J
        A = np.nan_to_num(JTJ + lam[:, None, None] * (np.eye(3) + np.einsum("nii->ni", JTJ)[:, :, None] * np.eye(3)))
        step = np.linalg.solve(A + 1e-12 * np.eye(3), -np.nan_to_num(JT @ r[..., None]))[..., 0]
        cn = np.clip(c + np.clip(np.nan_to_num(step), -50, 50), -1e4, 1e4)
        r2, J2 = residual_and_jacobian(cn, t)
        r2 = np.nan_to_num(r2, nan=1e6)
        better = (r2 ** 2).sum(1) < (r ** 2).sum(1)
        c = np.where(better[:, None], cn, c)
        r = np.where(better[:, None], r2, r)
        J = np.where(better[:, None, None], J2, J)
        lam = np.where(better, lam * 0.3, np.minimum(lam * 10, 1e8))
    return c


def inside_locus(px, py, locus):
    c = False
    for (x1, y1), (x2, y2) in zip(locus, np.roll(locus, -1, axis=0)):
        if (y1 > py) != (y2 > py) and px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1:
            c = not c
    return c


def build():
    locus = colour.XYZ_to_xy(np.clip(CMFS[2:66], 1e-9, None))
    pts = [(i, j) for i, x in enumerate(XS) for j, y in enumerate(YS) if y > 0.005 and inside_locus(x, y, locus)]
    xy = np.array([[XS[i], YS[j]] for i, j in pts])

    g = np.sinh(np.linspace(-6.0, 6.0, 51)) * 2.0
    cand = np.stack(np.meshgrid(g * 3.0, g * 3.0, g, indexing="ij"), -1).reshape(-1, 3)
    cxyz = np.concatenate([spectra(cand[i:i + 20000]) @ W for i in range(0, len(cand), 20000)])
    keep = cxyz[:, 1] > 1e-5
    cand, cxyz = cand[keep], cxyz[keep]
    tree = cKDTree(np.column_stack([cxyz[:, :2] / cxyz.sum(1, keepdims=True), np.log(cxyz[:, 1]) * 0.05]))

    table = np.zeros((N, N, len(LUMS), 81), np.float32)
    valid = np.zeros((N, N, len(LUMS)), bool)
    for k, lum in enumerate(LUMS):
        t = np.column_stack([xy[:, 0] / xy[:, 1] * lum, np.full(len(xy), lum),
                             (1 - xy[:, 0] - xy[:, 1]) / xy[:, 1] * lum])
        _, idx = tree.query(np.column_stack([xy, np.full(len(xy), np.log(lum) * 0.05)]))
        c = lm(cand[idx].copy(), t)
        sp = spectra(c)
        r, _ = residual_and_jacobian(c, t)
        ok = np.abs(r).max(1) < 0.01
        for (i, j), s, y, good in zip(pts, sp, (sp @ W)[:, 1], ok):
            if good:
                table[i, j, k] = s / y
                valid[i, j, k] = True
        print(f"  亮度 {lum:.3f}: {ok.sum()} / {len(pts)} 收敛", flush=True)

    # 太亮的饱和色: 取该色度下能达到的最亮光谱; 太暗未收敛: 取最近的有效亮度
    for i in range(N):
        for j in range(N):
            v = np.where(valid[i, j])[0]
            if len(v):
                for k in range(len(LUMS)):
                    if not valid[i, j, k]:
                        table[i, j, k] = table[i, j, v[-1] if k > v[-1] else v[np.argmin(np.abs(v - k))]]
    has = valid.any(-1)
    idx = distance_transform_edt(~has, return_distances=False, return_indices=True)
    table = table[idx[0], idx[1]]
    return table, valid, len(pts)


if __name__ == "__main__":
    table, valid, n = build()
    out = Path(__file__).resolve().parents[1] / "filmlab" / "data" / "upsampling_xyY.npz"
    np.savez_compressed(out, x=XS, y=YS, lum=LUMS, spectra=table.astype(np.float16), valid=valid)
    print(f"→ {out}  ({out.stat().st_size / 1e6:.1f} MB)")
