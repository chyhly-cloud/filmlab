"""测量照片中点光源周围的光晕: 半径 (按 35mm 画幅换算成底片上的 µm) 与红绿比。
用于对照真实 CineStill 800T 成片与模拟结果。
    python tools/measure_halation.py 图片...
"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage


def srgb_to_lin(a):
    a = a / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def measure(path, frame_mm=36.0, max_sources=30):
    im = srgb_to_lin(np.asarray(Image.open(path).convert("RGB"), float))
    h, w, _ = im.shape
    um = frame_mm * 1000.0 / max(h, w)
    core = im.min(-1) > 0.85                           # 接近饱和的光源核心
    lab, n = ndimage.label(core)
    if n == 0:
        return None
    sizes = ndimage.sum(core, lab, range(1, n + 1))
    Y = im @ np.array([0.2126, 0.7152, 0.0722])
    profs = []
    radii = np.arange(60, 1260, 60)                    # µm, 从核心边缘算起
    for k in np.argsort(sizes)[::-1][:max_sources]:
        if not (2 <= sizes[k] <= 0.002 * h * w):
            continue
        blob = lab == k + 1
        sl = ndimage.find_objects(blob.astype(int))[0]
        pad = int(1300 / um) + 2
        y0, y1, x0, x1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, h), max(sl[1].start - pad, 0), min(sl[1].stop + pad, w)
        d = ndimage.distance_transform_edt(~blob[y0:y1, x0:x1]) * um
        sub = im[y0:y1, x0:x1]
        others = core[y0:y1, x0:x1] & ~blob[y0:y1, x0:x1]
        if others.any() and d[others].min() < 1300:    # 只要孤立的光源
            continue
        prof = []
        for r in radii:
            ring = (d > r - um) & (d <= r + um)
            prof.append(np.median(sub[ring], 0) if ring.sum() > 3 else np.full(3, np.nan))
        profs.append(np.array(prof))
    if not profs:
        return None
    P = np.nanmedian(np.array(profs), 0)
    bg = P[-3:].mean(0)
    ex = np.clip(P - bg, 0, None)
    return dict(n=len(profs), um=um, radii=radii, excess=ex)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        r = measure(f)
        if r is None:
            print(f"{f.split('/')[-1]:14s} 没有孤立的点光源")
            continue
        ex, radii = r["excess"], r["radii"]
        tot = ex[:, 0] + 1e-6
        half = radii[np.argmax(np.cumsum(ex[:, 0]) >= 0.5 * ex[:, 0].sum())]
        far = radii[np.argmax(ex[:, 0] < 0.1 * ex[0, 0])] if (ex[:, 0] < 0.1 * ex[0, 0]).any() else radii[-1]
        k = int(np.nanargmax(ex[:, 0] - ex[:, 1]))      # 红色最"多出来"的半径 (避开灯本身的白色光芒)
        rg = ex[k, 0] / (ex[k, 1] + 1e-4)
        print(f"{f.split('/')[-1]:14s} 光源 {r['n']:2d} 个 | 红色超出量减半半径 {half:4.0f}µm | 衰减到 10% {far:5.0f}µm | 红环处 ({radii[k]:.0f}µm) R/G {rg:4.1f}")
