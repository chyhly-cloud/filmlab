"""T33 A (第 4 节): EPFL 夜景红像素 (RAW G/R < 0.3, B/R < 0.5; 未排除灯芯与光晕) 的光源类权重与各类映射给出的三层曝光
(相对 3200K 中灰), 以及输入的 Z。用法: FILMLAB_DATA=… python3 t33_red_classes.py  (写 a2.pkl 到当前目录)"""
import os, sys, zlib, warnings, numpy as np
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]
from scipy import ndimage as ndi
from scipy.ndimage import zoom
def job(sid):
    import oracle_timing as ot, pair_compare as pc
    from filmlab import generation, lights, film, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    generation.use("t31_base")
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = rawio.film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": seed % 36, "grain": 0}))
    e = eng.e
    h, w = m.shape[:2]
    f = max(1, int(max(h, w) / 400)); small = m[::f, ::f]; scale = 0.08 * max(small.shape[:2])
    ill = lights.illumination_map(small, scale)
    W = lights.class_weights(ill, feas=lights.feasibility(small, scale))
    W = zoom(W, (h / W.shape[0], w / W.shape[1], 1), order=1)[:h, :w]
    members = lights.pick_members(eng.v.frame_seed)
    sm = lambda a: np.stack([ndi.uniform_filter(a[..., c], 5) for c in range(3)], -1)
    inp = sm(m)
    gr = inp[..., 1] / np.maximum(inp[..., 0], 1e-9); br = inp[..., 2] / np.maximum(inp[..., 0], 1e-9)
    sel = ndi.binary_erosion((gr < 0.3) & (br < 0.5) & (inp.min(-1) > 0), iterations=2)
    ev = np.log2(inp[..., 0] / 0.184)
    S = 10 ** e.stock.log_sensitivity
    L = lights._blackbody(3200); sc = 1 / (lights.camera_rgb(L) @ np.array([.2126, .7152, .0722]))
    Hg = 0.184 * (L * sc) @ S                                 # 3200K 中灰 (Y=0.184) 的三层曝光
    TO = np.linalg.inv(lights._daylight_wb()[0]).T
    Zg = (0.184 * lights.light_rgb(L)) @ TO
    out = dict(sid=sid, ev=ev[sel], inp=inp[sel], W=W[sel], Zrel=(inp[sel] @ TO)[:, 2] / Zg[2])
    Hc = []
    for k, c in enumerate(lights.classes()):
        spec = members[c.key][1]
        meth = "spectral" if c.key == "daylight" else "linear_wp_bxyz"
        Hc.append(film.exposure_response(e.stock, spec, None, meth)(inp[sel]) / Hg)
    out["Hc"] = np.stack(Hc, 1)       # (n, 6, 3)
    out["members"] = [members[c.key][0] for c in lights.classes()]
    return out
if __name__ == "__main__":
    import multiprocessing as mp
    NIGHT = ["R1-00-1","R1-01-2","R1-02-3","R1-03-4","R1-04-5","R1-05-6","R1-06-7","R1-07-8"]
    with mp.get_context("spawn").Pool(4) as p:
        res = p.map(job, NIGHT)
    import pickle; pickle.dump(res, open("a2.pkl", "wb"))
    from filmlab import lights
    names = [c.key for c in lights.classes()]
    for lo in range(-1, 4):
        rows = [(r, (r["ev"] >= lo) & (r["ev"] < lo + 1)) for r in res]
        n = sum(m.sum() for _, m in rows)
        if n < 30: continue
        W = np.concatenate([r["W"][m] for r, m in rows]); Hc = np.concatenate([r["Hc"][m] for r, m in rows])
        Z = np.concatenate([r["Zrel"][m] for r, m in rows]); inp = np.concatenate([r["inp"][m] for r, m in rows])
        mix = (W[..., None] * Hc).sum(1)
        print(f"{lo:+d}…{lo+1:+d} n={n} RAW/0.184 {np.median(inp,0).round(3)/0.184} Z/Z(3200K 中灰) {np.median(Z):.2f}")
        print("   类权重 中位: " + " ".join(f"{nm} {np.median(W[:, k]):.2f}" for k, nm in enumerate(names)))
        print("   混合后 H/H(3200K 中灰) R G B:", np.median(mix, 0).round(3), " log10:", np.median(np.log10(mix), 0).round(2))
        for k, nm in enumerate(names):
            print(f"     {nm:12s} H/Hg {np.median(Hc[:, k], 0).round(3)}  B/Z {np.median(Hc[:, k, 2] / Z):.2f}")
    for r in res: print(r["sid"], r["members"])
