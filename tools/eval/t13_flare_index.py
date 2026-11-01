"""T13 诊断: 截断光源周围数码镜头眩光的强度 (RAW 可观测, 与光源真实亮度成正比), 与逐场景最佳光晕强度 (t13_halo_scale.json) 对照。
每个截断块 (传感器饱和, 面积 ≥ 20 个下采样像素): 外圈 3–12 像素环带的平均亮度 / 截断亮度 (线性, 未重建的 RAW)。
场景的眩光指数 = 各块按面积加权的中位数。另给现有亮度先验分给该场景截断块的名义档数 (按面积分类, 无 EXIF 时)。

用法: FILMLAB_DATA=… python3 t13_flare_index.py
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]


def job(sid):
    import rawpy
    from scipy import ndimage
    import oracle_timing as ot
    from filmlab import rawio
    rawp, _, _ = ot.sources(sid)
    rgb = rawio.load_raw(rawp, max_size=1400, highlights="clip", denoise_raw=False)
    Y = rgb @ rawio.LUMA
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    s = ndimage.zoom(sat.astype(float), (Y.shape[0] / sat.shape[0], Y.shape[1] / sat.shape[1]), order=1)[:Y.shape[0], :Y.shape[1]] > 0.5
    clipY = np.median(Y[s]) if s.any() else 1.0
    lab, n = ndimage.label(s)
    idx, wts, prior = [], [], []
    total = s.size
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        blob = lab == k
        area = int(blob.sum())
        if area < 20:
            continue
        ring = ndimage.binary_dilation(blob, iterations=12) & ~ndimage.binary_dilation(blob, iterations=3) & ~s
        if ring.sum() < 10:
            continue
        idx.append(float(np.mean(Y[ring]) / clipY)); wts.append(area)
        prior.append(3.5 if area / total < 5e-3 else 0.75)          # HIGHLIGHT_PRIOR 档数范围的中点 (无 EXIF)
    if not idx:
        return sid, None, None
    o = np.argsort(idx); c = np.cumsum(np.array(wts)[o]) / np.sum(wts)
    med = float(np.array(idx)[o][np.searchsorted(c, 0.5)])
    return sid, med, float(np.average(prior, weights=wts))


if __name__ == "__main__":
    import multiprocessing as mp
    best = {s: v["best_scale"] for s, v in json.load(open(HERE / "t13_halo_scale.json")).items()}
    with mp.get_context("spawn").Pool(3) as p:
        res = p.map(job, list(best))
    print(f"{'场景':9s} 最佳光晕倍数 | 眩光指数 (环带/截断) | 先验名义档数 | 推算所需档数 (先验 + log2 倍数)")
    rows = []
    for sid, fi, pr in sorted(res, key=lambda r: best[r[0]]):
        need = pr + np.log2(best[sid]) if pr is not None else None
        rows.append((fi, need))
        print(f"{sid:9s} ×{best[sid]:<5g}      | {fi:.4f}               | {pr:.2f}          | {need:+.2f}")
    a = np.array([r for r in rows if r[0] is not None], float)
    print(f"\nlog 眩光指数 与 推算所需档数 的相关: r = {np.corrcoef(np.log(a[:, 0]), a[:, 1])[0, 1]:+.2f} (n={len(a)})")
