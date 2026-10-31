"""诊断: 各场景成片相对真实扫描的彩度比 (24px 格子中位数, 只取真实彩度 > 10、两者 L* 在 15–90 的格子)。
比 = median(C*_模型) / median(C*_真实); 另给彩度分档的比 (真实 C* 10–25 / 25–45 / > 45)。配置与验收入口相同, 可加变体。
用法: FILMLAB_DATA=… python3 chroma_ratio.py 变体或none 输出.json"""
import json, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[3])]


def job(args):
    variant, sid = args
    if variant != "none":
        import accept_variant
        accept_variant.install(variant)
    import colour
    import oracle_timing as ot, pair_compare as pc
    from skimage.transform import warp, ProjectiveTransform
    from filmlab import generation, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.io import srgb_encode
    generation.use("next")
    Hm, real = ot.geometry(sid)
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = rawio.film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": seed % 36, "grain": 0}))
    sim = warp(srgb_encode(np.clip(eng.render(m), 0, 1)), ProjectiveTransform(Hm).inverse, output_shape=real.shape[:2],
               order=1, cval=np.nan)
    ok = np.all(np.isfinite(sim), -1)
    S, _ = ot.cells(sim, ok)
    R, _ = ot.cells(real, ok)
    lab = lambda x: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(x, 0, 1)))
    ls, lr = lab(S), lab(R)
    cs, cr = np.hypot(ls[:, 1], ls[:, 2]), np.hypot(lr[:, 1], lr[:, 2])
    v = (cr > 10) & (ls[:, 0] > 15) & (ls[:, 0] < 90) & (lr[:, 0] > 15) & (lr[:, 0] < 90)
    out = {"n": int(v.sum())}
    if v.sum() >= 20:
        out["ratio"] = float(np.median(cs[v]) / np.median(cr[v]))
        for lo, hi in ((10, 25), (25, 45), (45, 999)):
            b = v & (cr >= lo) & (cr < hi)
            out[f"r{lo}"] = float(np.median(cs[b]) / np.median(cr[b])) if b.sum() >= 10 else None
    return sid, out


if __name__ == "__main__":
    import multiprocessing as mp
    from accept_calibration import TRAIN, HOLDOUT
    G = {**TRAIN, **HOLDOUT}
    variant, path = sys.argv[1], sys.argv[2]
    with mp.get_context("spawn").Pool(3, maxtasksperchild=1) as p:
        res = dict(p.map(job, [(variant, s) for s in G]))
    json.dump(res, open(path, "w"), indent=1)
    f = lambda x: "  -  " if x is None else f"{x:5.2f}"
    for s in sorted(res, key=lambda s: G[s]):
        r = res[s]
        print(f"{s:9s} {G[s]:5s} n {r['n']:4d}  彩度比 {f(r.get('ratio'))}  分档 {f(r.get('r10'))} {f(r.get('r25'))} {f(r.get('r45'))}")
    v = [r["ratio"] for r in res.values() if "ratio" in r]
    print(f"中位彩度比 {np.median(v):.2f}  (n 场景 = {len(v)})")
