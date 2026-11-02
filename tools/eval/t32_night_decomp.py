import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from scipy import ndimage as ndi
from t32_chain import chain
from t32_o1 import fit
SC = ["R1-00-1","R1-01-2","R1-02-3","R1-03-4","R1-04-5","R1-05-6","R1-06-7","R1-07-8"]
rows = []
for sid in SC:
    z = np.load(f'epfl/{sid}_t31_base.npz')
    sm = lambda a: np.stack([ndi.uniform_filter(np.nan_to_num(a[..., c].astype(float), nan=-9), 5) for c in range(a.shape[-1])], -1)
    inp, real = sm(z['inp']), sm(z['real'])
    lh = z['logH'].astype(float)
    gr = inp[...,1]/np.maximum(inp[...,0],1e-9); br = inp[...,2]/np.maximum(inp[...,0],1e-9)
    far = ndi.distance_transform_edt(~z['core'])*z['px_um'] > 500
    ok = (inp.min(-1) > 0) & far & (real.min(-1) >= 0) & (z['halo_frac'].astype(float).max(-1) < 0.2)
    redish = (gr < 0.3) & (br < 0.5)
    sel = ndi.binary_erosion(redish & ok, iterations=2)
    tim, dk, nc = fit(z, excl=ndi.binary_dilation(redish, iterations=5))
    print(sid, "O1 ΔK", dk.round(3), "格子", nc, "红像素", sel.sum())
    if sel.sum() < 30: continue
    ys, xs = np.nonzero(sel)
    y0, y1, x0, x1 = ys.min()-3, ys.max()+4, xs.min()-3, xs.max()+4
    L = lh[y0:y1, x0:x1]; S = sel[y0:y1, x0:x1]
    res, _ = chain(L.reshape(-1, 3), tim)
    for k, v in res.items():
        v = v.reshape(L.shape); res[k] = np.stack([ndi.uniform_filter(v[..., c], 5) for c in range(3)], -1)[S]
    rows.append(dict(sid=np.full(S.sum(), sid), inp=inp[sel], real=real[sel], **{f"m_{k}": v for k, v in res.items()}))
T = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
ev = np.log2(T['inp'][:, 0] / 0.184)
combos = ["full", "B", "C", "D", "BC", "BCD"]
med = lambda a: np.median(a[:, 1] / np.maximum(a[:, 0], 1e-9))
print("\nO1 定光下: RAW R 档 | n (场景) | RAW G/R | 真实 G/R | " + " | ".join(f"模型{c}" for c in combos) + " | 真实 R | 模型 R")
for lo in range(-2, 5):
    m = (ev >= lo) & (ev < lo + 1)
    if m.sum() < 30: continue
    print(f"{lo:+d}…{lo+1:+d} | {m.sum()} ({len(np.unique(T['sid'][m]))}) | {med(T['inp'][m]):.3f} | {med(T['real'][m]):.2f} | " +
          " | ".join(f"{med(T['m_'+c][m]):.2f}" for c in combos) + f" | {np.median(T['real'][m,0]):.2f} | {np.median(T['m_full'][m,0]):.2f}")
for sid in np.unique(T['sid']):
    m = T['sid'] == sid
    print(f"  {sid} n={m.sum()} RAW G/R {med(T['inp'][m]):.3f} 真实 {med(T['real'][m]):.2f} " + " ".join(f"{c} {med(T['m_'+c][m]):.2f}" for c in combos))
np.savez("night_red_decomp_o1.npz", **T)
