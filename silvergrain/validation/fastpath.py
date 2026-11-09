"""黑白快速路径 (render method="closed_form": 闭式协方差的对数正态场) 对显式银粒渲染 (method="explicit") 的检验。
均匀曝光 (18% 灰 × 2^EV), 160 × 160 像素, 同一种子; 比较像素平均密度、像素 σ_D、相邻像素相关、2 × 2 盒 σ。"""
import time

import numpy as np

from ..emulsion import derive
from ..render import render

CASES = (("kodak_trix_400", "d76", 1.0), ("kodak_trix_400", "rodinal", 1.0), ("kodak_tmax_400", "d76", 1.0),
         ("ilford_hp5_plus", "d76", 1.0), ("kodak_trix_400", "d76", 2.0))
PXS = (6.0, 12.0)
EVS = (-2, 0, 2)


def stats(D):
    D = D[8:-8, 8:-8]; x = D - D.mean()
    v = x.var()
    r1 = 0.5 * ((x[:, 1:] * x[:, :-1]).mean() + (x[1:] * x[:-1]).mean()) / v
    b2 = D[:D.shape[0] // 2 * 2, :D.shape[1] // 2 * 2].reshape(D.shape[0] // 2, 2, D.shape[1] // 2, 2).mean((1, 3)).std()
    return dict(mean=float(D.mean()), sd=float(np.sqrt(v)), r1=float(r1), box2=float(b2))


def run(n=160, log=print):
    rows = []
    for film, dev, k in CASES:
        em = derive(film, dev, k)
        for px in PXS:
            for ev in EVS:
                H = np.full((n, n), 10.0 / em.film.ei * 2.0 ** ev)
                t = time.time(); a = render(em, exposure=H, px_um=px, seed=3, method="explicit"); ta = time.time() - t
                t = time.time(); b = render(em, exposure=H, px_um=px, seed=3, method="closed_form"); tb = time.time() - t
                sa, sb = stats(a.density), stats(b.density)
                rows.append(dict(film=film, developer=dev, k=k, px_um=px, ev=ev, closed_mean=float(a.mean[0, 0]), explicit=sa,
                                 closed_form=sb, sd_ratio=sb["sd"] / sa["sd"], box2_ratio=sb["box2"] / sa["box2"],
                                 mean_diff=sb["mean"] - sa["mean"], seconds=dict(explicit=ta, closed_form=tb)))
        log(f"  fastpath {film}@{dev} k{k}")
    r = np.array([x["sd_ratio"] for x in rows]); k1 = np.array([x["sd_ratio"] for x in rows if x["k"] == 1.0])
    return dict(rows=rows, sd_ratio_range=[float(r.min()), float(r.max())], sd_ratio_range_k1=[float(k1.min()), float(k1.max())],
                max_abs_mean_diff=float(max(abs(x["mean_diff"]) for x in rows)))
