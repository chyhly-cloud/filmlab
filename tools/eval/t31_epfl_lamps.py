"""T31 第 2 步: EPFL 配对, 同一配准几何。每个传感器饱和灯芯 (连通块) 为一盏灯, 像素归属最近灯芯;
背景 = 该灯 1.5–2.5 mm 环的中位 (真实与模型各自); 距离档取中位 − 背景。只用背景较暗的灯 (真实背景 max < BGMAX)。
用法: BGMAX=0.1 python3 t31_epfl_lamps.py t31_epfl_render 的输出目录 代际   → e2_代际_BGMAX.npz (逐灯) 与汇总表"""
import sys, glob, os
import numpy as np
from scipy import ndimage as ndi
BINS = [25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500]
SHOW = [100, 150, 200, 300, 400, 500]
BGMAX = float(os.environ.get("BGMAX", "0.05"))
src = sys.argv[1] if len(sys.argv) > 1 else "epfl"
tag = sys.argv[2] if len(sys.argv) > 2 else "t30_base"


def per_lamp(z):
    core = z["core"]
    um = 36000.0 / max(core.shape)
    d, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
    d *= um
    lab, n = ndi.label(core)
    own = lab[iy, ix]
    idx = np.arange(1, n + 1)
    rad = np.sqrt(np.asarray(ndi.sum(core, lab, idx), float) / np.pi) * um
    ring = (d > 1500) & (d < 2500)
    res = {}
    valid = np.all(np.isfinite(z["out"]), -1) & np.all(np.isfinite(z["real"]), -1)
    for k in ("real", "out", "out_rj", "halo", "direct"):
        a = np.nan_to_num(z[k])
        bg = np.stack([np.asarray(ndi.median(a[..., c], own * ring * valid, index=idx), float) for c in range(3)], 1)
        prof = np.full((n, len(BINS), 3), np.nan)
        b = np.searchsorted(BINS, d)
        for bi in range(len(BINS)):
            m = (b == bi) & valid & (d > 0) & (d <= BINS[-1])
            if bi == 0:
                m &= d > 0
            cnt = np.asarray(ndi.sum(m, own, idx))
            for c in range(3):
                v = np.asarray(ndi.median(a[..., c], own * m, index=idx), float)
                v[cnt < 3] = np.nan
                prof[:, bi, c] = v
        res[k] = (prof - bg[:, None, :]) if k not in ("halo",) else prof   # 光晕增量本来就是超出量
        res[k + "_bg"] = bg
    return res, rad


rows = {k: [] for k in ("real", "out", "out_rj", "halo", "direct")}
rads, sids = [], []
for f in sorted(glob.glob(f"{src}/*_{tag}.npz")):
    z = np.load(f)
    r, rad = per_lamp(z)
    lab, n = ndi.label(z["core"])
    rc = np.asarray(ndi.median(np.nan_to_num(z["real"]).min(-1), lab, index=np.arange(1, n + 1)), float)
    ok = (np.nanmax(r["real_bg"], 1) < BGMAX) & (rad < 400)
    for k in ("real_bg", "out_bg", "out_rj_bg"):
        rows.setdefault(k, []).append(r[k][ok])
    rows.setdefault("real_core", []).append(rc[ok])
    for k in ("real", "out", "out_rj", "halo", "direct"):
        rows[k].append(r[k][ok])
    rads.append(rad[ok]); sids += [os.path.basename(f)[:7]] * int(ok.sum())
A = {k: np.concatenate(v) for k, v in rows.items()}
rad = np.concatenate(rads); sids = np.array(sids)
np.savez(f"e2_{tag}_{BGMAX}.npz", **A, rad=rad, sid=sids)
cols = [BINS.index(x) for x in SHOW]
print(f"灯数 {len(rad)}  (背景 < {BGMAX}); 每场景:", {s: int((sids == s).sum()) for s in np.unique(sids)})
f = lambda v, p=2: " ".join(f"{x:6.{p}f}" if np.isfinite(x) else "   nan" for x in v)
for lab, msk in (("全部", np.ones(len(rad), bool)), ("灯芯 <60", rad < 60), ("灯芯 60–400", rad >= 60)):
    print(f"\n## {lab}  n={msk.sum()}   列: {SHOW} µm")
    with np.errstate(all="ignore"):
        for k, nm in (("real", "真实冲扫"), ("out", "模型 t30_base"), ("out_rj", "模型 带碳"), ("halo", "模型 光晕曝光")):
            M = np.nanmedian(A[k][msk][:, cols], 0)
            gr = np.nanmedian(A[k][msk][:, cols, 1] / A[k][msk][:, cols, 0], 0)
            bgr = np.nanmedian(A[k][msk][:, cols, 2] / A[k][msk][:, cols, 1], 0)
            print(f"| {nm} | R {f(M[:,0],3)} | G/R(中位比) {f(M[:,1]/M[:,0])} | B/G(中位比) {f(M[:,2]/M[:,1])} | G/R(逐灯中位) {f(gr)} | B/G(逐灯) {f(bgr)} |")
        for nm, a, b in (("模型 光晕部分 (基准−带碳)", "out", "out_rj"), ("真实 − 模型带碳", "real", "out_rj")):
            M = np.nanmedian(A[a][msk][:, cols] - A[b][msk][:, cols], 0)
            print(f"| {nm} | R {f(M[:,0],3)} | G {f(M[:,1],3)} | B {f(M[:,2],3)} | G/R {f(M[:,1]/M[:,0])} | B/G {f(M[:,2]/M[:,1])} |")
