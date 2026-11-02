"""从完整渲染的轨迹 (t32_epfl_trace) 评估一个代际: 红像素 G/R (默认 / O1) 与 ΔE00 全图 / 光源旁。"""
import numpy as np, sys, warnings, multiprocessing as mp, json
warnings.filterwarnings('ignore')
NIGHT = ["R1-00-1","R1-01-2","R1-02-3","R1-03-4","R1-04-5","R1-05-6","R1-06-7","R1-07-8"]
def job(a):
    gen, sid = a
    import t32_rescan as rescan, t32_counterfactuals as cands
    from scipy import ndimage as ndi
    from scipy.optimize import least_squares
    from filmlab import generation; generation.use(gen)
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    import t32_o1 as o1, zlib
    z = np.load(f"epfl/{sid}_{gen}.npz")
    full, near = rescan.de_scores(z["out"], z, sid)
    sm = lambda a: np.stack([ndi.uniform_filter(np.nan_to_num(np.asarray(a[..., c], float), nan=-9), 5) for c in range(3)], -1)
    inp, real, out = sm(z['inp']), sm(z['real']), sm(z['out'])
    gr = inp[...,1]/np.maximum(inp[...,0],1e-9); br = inp[...,2]/np.maximum(inp[...,0],1e-9)
    far = ndi.distance_transform_edt(~z['core'])*z['px_um'] > 500
    ok = (inp.min(-1) > 0) & far & (real.min(-1) >= 0) & (z['halo_frac'].astype(float).max(-1) < 0.2)
    redish = (gr < 0.3) & (br < 0.5)
    sel = ndi.binary_erosion(redish & ok, iterations=2)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": zlib.crc32(sid.encode()) % 1000 % 36, "grain": 0}))
    sc = eng.scanner
    R, D = o1.cells(z, excl=ndi.binary_dilation(redish, iterations=5))
    mk = lambda dk: PrintTiming(K=z["K"] + dk, s=z["s"], M=(z["M_off"], z["M_r"]))
    def resid(dk):
        sc.timing = mk(dk); return (np.log10(np.clip(sc.to_rgb(D), 1e-4, None)) - np.log10(R)).ravel()
    dk = least_squares(resid, np.zeros(3), bounds=(-1.5, 1.5), loss="soft_l1", f_scale=0.1).x
    sc.timing = mk(dk)
    r = dict(full=full, near=near, dk=dk.tolist())
    if sel.sum():
        ys, xs = np.nonzero(sel); y0, y1, x0, x1 = ys.min()-3, ys.max()+4, xs.min()-3, xs.max()+4
        o1o = np.clip(sc.to_rgb(z["Dn"][y0:y1, x0:x1].reshape(-1, 3).astype(float)), 0, None).reshape(y1-y0, x1-x0, 3)
        r.update(inp=inp[sel], real=real[sel], out=out[sel], o1=sm(o1o)[sel[y0:y1, x0:x1]],
                 lh=z["logH"][sel].astype(float), dye=z["dye"][sel].astype(float), Dn=z["Dn"][sel].astype(float))
    return gen, sid, r
if __name__ == "__main__":
    gens = sys.argv[1:]
    with mp.get_context("spawn").Pool(4, maxtasksperchild=1) as p:
        res = p.map(job, [(g, s) for g in gens for s in NIGHT], chunksize=1)
    med = lambda a: np.median(a[:, 1] / np.maximum(a[:, 0], 1e-9))
    summ = {}
    for g in gens:
        rows = [(s, r) for gg, s, r in res if gg == g]
        print(f"== {g}  全图 ΔE00 {np.mean([r['full'] for s, r in rows]):.3f}  光源旁 {np.nanmean([r['near'] for s, r in rows]):.3f}")
        print("   " + " ".join(f"{s}:{r['full']:.2f}/{r['near']:.2f}" for s, r in rows))
        rr = [(s, r) for s, r in rows if "inp" in r]
        T = {k: np.concatenate([r[k] for s, r in rr]) for k in ("inp", "real", "out", "o1", "lh", "dye", "Dn")}
        sid = np.concatenate([np.full(len(r["inp"]), s) for s, r in rr])
        ev = np.log2(T["inp"][:, 0] / 0.184)
        for lo in range(-2, 5):
            m = (ev >= lo) & (ev < lo + 1)
            if m.sum() < 30: continue
            print(f"  {lo:+d}…{lo+1:+d} n={m.sum()} ({len(np.unique(sid[m]))}) RAW {med(T['inp'][m]):.3f} 真实 {med(T['real'][m]):.2f} | 默认 {med(T['out'][m]):.2f} | O1 {med(T['o1'][m]):.2f} "
                  f"| 染料 {np.median(T['dye'][m],0).round(3)} Dn {np.median(T['Dn'][m],0).round(3)}")
        summ[g] = {s: dict(full=r["full"], near=r["near"], dk=r["dk"]) for s, r in rows}
    json.dump(summ, open("eval_trace_" + "_".join(gens) + ".json", "w"), indent=1)
