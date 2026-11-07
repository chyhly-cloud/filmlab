"""T51: 真实冲印店扫描里的直边 → 边缘扩散函数 (ESF), 用于测冲印店的空间传递 (锐化 / 色度降噪)。

只读图像, 不存图像: 输出每条边的位置、角度、反差与按 0.25 像素分档的 ESF (亮度与三通道, 8 位码值域)。
找边 (自动, 与图像内容无关的判据):
  1. 亮度 Y = (R + G + B)/3 (码值); 高斯 σ = 1 像素平滑后的梯度。
  2. 按 W × W 像素窗格扫描; 窗内梯度最大的像素须 ≥ GMIN 码值/像素; 结构张量相干度 ≥ 0.80 (单一方向)。
  3. 沿边方向逐行 / 逐列用梯度质心求亚像素边位置, 直线拟合; 残差 RMS ≤ 0.25 像素、角度离轴 2–40° (斜边: 投影后超采样)。
  4. 投影到法向距离 d, 取 |d| ≤ R 的像素按 0.25 像素分档 (斜边法, ISO 12233 的思路); 两侧平台 (|d| > R − 3) 平坦
     (平台内标准差 ≤ 0.25 × 台阶) 且台阶 ≥ STEP_MIN 码值、两侧都不触 0 / 255。
用法: python3 -I t51_edges.py 输出.json 图1 [图2 ...] [--W 48 --R 12]
"""
import argparse
import json
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

GMIN = 6.0
STEP_MIN = 30.0
BIN = 0.25


def find_edges(img, W=48, R=12, step_min=STEP_MIN, max_per_image=400, L=None):
    L = L or 2 * W
    img = img.astype(float)
    Y = img.mean(-1)
    gy = gaussian_filter(Y, 1.0, order=(1, 0))
    gx = gaussian_filter(Y, 1.0, order=(0, 1))
    g = np.hypot(gx, gy)
    Jxx, Jyy, Jxy = (gaussian_filter(a, 3.0) for a in (gx * gx, gy * gy, gx * gy))
    out = []
    H, Wd = Y.shape
    for y0 in range(R, H - W - R, W // 2):
        for x0 in range(R, Wd - W - R, W // 2):
            win = g[y0:y0 + W, x0:x0 + W]
            if win.max() < GMIN:
                continue
            iy, ix = np.unravel_index(np.argmax(win), win.shape)
            cy, cx = y0 + iy, x0 + ix
            a, b, c = Jxx[cy, cx], Jyy[cy, cx], Jxy[cy, cx]
            lam1 = 0.5 * (a + b + np.hypot(a - b, 2 * c)); lam2 = 0.5 * (a + b - np.hypot(a - b, 2 * c))
            if (lam1 - lam2) / (lam1 + lam2 + 1e-9) < 0.80:
                continue
            nx, ny = gx[cy, cx], gy[cy, cx]
            nrm = np.hypot(nx, ny); nx, ny = nx / nrm, ny / nrm           # 法向 (指向亮侧)
            ang = np.degrees(np.arctan2(abs(ny), abs(nx)))                 # 法向离 x 轴的角
            off = min(ang, 90 - ang)
            if off < 0.4 or off > 40.0:
                continue
            e = _profile(img, Y, gx, gy, cy, cx, nx, ny, L, R, step_min)
            if e is not None:
                e.update(y=int(cy), x=int(cx), angle_deg=float(np.degrees(np.arctan2(ny, nx))))
                out.append(e)
    # 去重 (相邻窗格可能找到同一条边)
    keep = []
    for e in sorted(out, key=lambda e: -e["step"]):
        if all(np.hypot(e["y"] - k["y"], e["x"] - k["x"]) > max(W * 0.6, L * 0.5) for k in keep):
            keep.append(e)
    return keep[:max_per_image]


def _profile(img, Y, gx, gy, cy, cx, nx, ny, W, R, step_min):
    h = W // 2
    vertical = abs(nx) > abs(ny)              # 边接近竖直 → 逐行求边位置
    pts = []
    for t in range(-h, h):
        if vertical:
            yy = cy + t
            xs = np.arange(cx - R, cx + R + 1)
            if yy < 0 or yy >= Y.shape[0] or xs[0] < 0 or xs[-1] >= Y.shape[1]:
                return None
            gg = np.abs(gx[yy, xs])
            if gg.sum() <= 0:
                return None
            pts.append((yy, (gg * xs).sum() / gg.sum()))
        else:
            xx = cx + t
            ys = np.arange(cy - R, cy + R + 1)
            if xx < 0 or xx >= Y.shape[1] or ys[0] < 0 or ys[-1] >= Y.shape[0]:
                return None
            gg = np.abs(gy[ys, xx])
            if gg.sum() <= 0:
                return None
            pts.append((xx, (gg * ys).sum() / gg.sum()))
    pts = np.array(pts, float)
    A = np.stack([np.ones(len(pts)), pts[:, 0]], 1)
    coef, *_ = np.linalg.lstsq(A, pts[:, 1], rcond=None)
    res = pts[:, 1] - A @ coef
    rms = float(np.sqrt((res ** 2).mean()))
    if rms > 0.25:
        return None
    # 法向距离: 直线 u = c0 + c1·t (vertical: x = c0 + c1·y)
    c0, c1 = coef
    ys, xs = np.mgrid[cy - h:cy + h, cx - h - R:cx + h + R + 1] if vertical else np.mgrid[cy - h - R:cy + h + R + 1, cx - h:cx + h]
    ok = (ys >= 0) & (ys < Y.shape[0]) & (xs >= 0) & (xs < Y.shape[1])
    ys, xs = ys[ok], xs[ok]
    if vertical:
        d = (xs - (c0 + c1 * ys)) / np.sqrt(1 + c1 ** 2)
    else:
        d = (ys - (c0 + c1 * xs)) / np.sqrt(1 + c1 ** 2)
    sgn = np.sign(nx if vertical else ny)
    d = d * sgn                                  # d > 0 = 亮侧
    along = (ys - cy) if vertical else (xs - cx)
    sel = (np.abs(d) <= R) & (np.abs(along) < h)
    d, vals = d[sel], img[ys[sel], xs[sel]]
    edges = np.arange(-R, R + BIN / 2, BIN)
    idx = np.digitize(d, edges) - 1
    nb = len(edges) - 1
    esf = np.full((nb, 4), np.nan)
    cnt = np.bincount(idx[(idx >= 0) & (idx < nb)], minlength=nb)
    for k in range(4):
        v = vals.mean(-1) if k == 0 else vals[:, k - 1]
        s = np.bincount(idx[(idx >= 0) & (idx < nb)], weights=v[(idx >= 0) & (idx < nb)], minlength=nb)
        esf[:, k] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    if np.isnan(esf[:, 0]).sum() > nb * 0.05:
        return None
    lo = vals[d < -(R - 3)]; hi = vals[d > (R - 3)]
    if len(lo) < 10 or len(hi) < 10:
        return None
    step = float(hi.mean(-1).mean() - lo.mean(-1).mean())
    if step < step_min:
        return None
    pl = esf[:, 0]
    nb4 = int(round(3 / BIN))
    if np.nanstd(pl[:nb4]) > 0.04 * step or np.nanstd(pl[-nb4:]) > 0.04 * step:
        return None
    if (lo.mean(-1) <= 1.5).mean() > 0.02 or (hi.mean(-1) >= 253.5).mean() > 0.02:
        return None
    return dict(step=step, lo=lo.mean(0).tolist(), hi=hi.mean(0).tolist(), rms_fit=rms, slope=float(c1),
                d=(0.5 * (edges[1:] + edges[:-1])).tolist(), esf=np.round(esf, 3).tolist(), count=cnt.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--W", type=int, default=48)
    ap.add_argument("--R", type=int, default=12)
    ap.add_argument("--L", type=int, default=0)
    a = ap.parse_args()
    res = {}
    for f in a.images:
        im = np.asarray(Image.open(f).convert("RGB"))
        e = find_edges(im, a.W, a.R, L=a.L or None)
        res[f.split("/")[-2] + "/" + f.split("/")[-1]] = dict(shape=list(im.shape), edges=e)
        print(f.split("/")[-1], im.shape, len(e), flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    sys.exit(main())
