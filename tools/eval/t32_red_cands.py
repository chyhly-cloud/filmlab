"""候选的红像素 G/R: 默认定光 (lab_measured) 与 O1 定光 (每帧三通道光号拟合到真实, 不含红像素)。"""
import numpy as np, sys, warnings, multiprocessing as mp
warnings.filterwarnings('ignore')
NIGHT = ["R1-00-1","R1-01-2","R1-02-3","R1-03-4","R1-04-5","R1-05-6","R1-06-7","R1-07-8"]
def job(args):
    cand, sid = args
    import t32_counterfactuals as cands, t32_rescan as rescan
    from scipy import ndimage as ndi
    from scipy.optimize import least_squares
    out, tr, z = rescan.render(getattr(cands, cand), sid)
    from filmlab.scanning.timing import PrintTiming
    sm = lambda a: np.stack([ndi.uniform_filter(np.nan_to_num(np.asarray(a[..., c], float), nan=-9), 5) for c in range(3)], -1)
    inp, real = sm(z['inp']), sm(z['real'])
    gr = inp[...,1]/np.maximum(inp[...,0],1e-9); br = inp[...,2]/np.maximum(inp[...,0],1e-9)
    far = ndi.distance_transform_edt(~z['core'])*z['px_um'] > 500
    ok = (inp.min(-1) > 0) & far & (real.min(-1) >= 0) & (z['halo_frac'].astype(float).max(-1) < 0.2)
    redish = (gr < 0.3) & (br < 0.5)
    sel = ndi.binary_erosion(redish & ok, iterations=2)
    # O1 on non-red cells
    import t32_o1 as o1
    Dn = tr['Dn']; t0 = tr['timing']
    zz = dict(real=z['real'], Dn=Dn, core=z['core'], px_um=z['px_um'], K=t0.K, s=t0.s, M_off=t0.M[0], M_r=t0.M[1])
    R, D = o1.cells(zz, excl=ndi.binary_dilation(redish, iterations=5))
    sc = o1.sc   # 注意: o1 用 decomp 的扫描仪 (t31_base 下构造); 改用本候选的扫描仪
    from filmlab.engine import Engine
    import zlib
    from filmlab.recipes import recipe
    from filmlab.engine import Params
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": zlib.crc32(sid.encode()) % 1000 % 36, "grain": 0}))
    sc = eng.scanner
    mk = lambda dk: PrintTiming(K=np.asarray(t0.K) + dk, s=t0.s, M=t0.M)
    def resid(dk):
        sc.timing = mk(dk); return (np.log10(np.clip(sc.to_rgb(D), 1e-4, None)) - np.log10(R)).ravel()
    dk = least_squares(resid, np.zeros(3), bounds=(-1.5, 1.5), loss="soft_l1", f_scale=0.1).x
    sc.timing = mk(dk)
    ys, xs = np.nonzero(sel)
    if len(ys) == 0:
        return cand, sid, None
    y0, y1, x0, x1 = ys.min()-3, ys.max()+4, xs.min()-3, xs.max()+4
    o1out = np.clip(sc.to_rgb(Dn[y0:y1, x0:x1].reshape(-1, 3)), 0, None).reshape(y1-y0, x1-x0, 3)
    o1out = sm(o1out)[sel[y0:y1, x0:x1]]
    return cand, sid, dict(inp=inp[sel], real=real[sel], out=sm(out)[sel], o1=o1out, dk=dk)
if __name__ == "__main__":
    cs = sys.argv[1:]
    with mp.get_context("spawn").Pool(4, maxtasksperchild=1) as p:
        res = p.map(job, [(c, s) for c in cs for s in NIGHT], chunksize=1)
    med = lambda a: np.median(a[:, 1] / np.maximum(a[:, 0], 1e-9))
    for c in cs:
        rows = [(s, r) for cc, s, r in res if cc == c and r is not None]
        T = {k: np.concatenate([r[k] for s, r in rows]) for k in ("inp", "real", "out", "o1")}
        sid = np.concatenate([np.full(len(r["inp"]), s) for s, r in rows])
        ev = np.log2(T["inp"][:, 0] / 0.184)
        print(f"== {c}")
        for lo in range(-2, 5):
            m = (ev >= lo) & (ev < lo + 1)
            if m.sum() < 30: continue
            bg = lambda a: np.median(a[m][:, 2] / np.maximum(a[m][:, 1], 1e-9))
            print(f"  {lo:+d}…{lo+1:+d} n={m.sum()} ({len(np.unique(sid[m]))}) RAW {med(T['inp'][m]):.3f} 真实 {med(T['real'][m]):.2f} | 默认 {med(T['out'][m]):.2f} | O1 {med(T['o1'][m]):.2f} || B/G 真实 {bg(T['real']):.2f} 默认 {bg(T['out']):.2f} O1 {bg(T['o1']):.2f}")
        for s, r in rows:
            print(f"    {s} n={len(r['inp'])} 真实 {med(r['real']):.2f} 默认 {med(r['out']):.2f} O1 {med(r['o1']):.2f} ΔK {np.round(r['dk'],3)}")
        np.savez(f"cand/red_{c}.npz", sid=sid, **T)
