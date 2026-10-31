"""诊断: 蓝层曝光 ×10^d (d = −0.1, 0, +0.1) 时, 区域内底片 B 偏色与成片 b* 的变化 (下游增益)。另测关闭逐帧定光 (固定档案中灰) 的成片。"""
import json, os, sys, zlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
from pathlib import Path; sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]


def job(args):
    sid, d, timing = args
    import colour, rawpy
    import oracle_timing as ot, pair_compare as pc
    from scipy.ndimage import zoom
    from skimage.transform import warp, ProjectiveTransform
    from filmlab import generation, rawio, film
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    from filmlab.io import srgb_encode
    generation.use("next")
    s0 = film.emulsion_scatter
    f = np.array([1.0, 1.0, 10.0 ** d])
    film.emulsion_scatter = lambda H, px: s0(H * f, px)
    Hm, real = ot.geometry(sid)
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": timing, "roll": None, "frame": seed % 36, "grain": 0}))
    ts = {}
    out = np.clip(eng.scan(eng.develop_negative(m), trace=ts), 0, None)
    gray = eng.scanner.normalized_density(eng._transmittance(eng._gray_state(actual_light=False)))[0]
    Y = m @ np.array([0.2126, 0.7152, 0.0722])
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    clip = np.zeros(Y.shape, bool) if sat is None else \
        zoom(sat.astype(float), (Y.shape[0] / sat.shape[0], Y.shape[1] / sat.shape[1]), order=1)[:Y.shape[0], :Y.shape[1]] > 0.05
    lo, hi = np.percentile(Y[~clip], [40, 90])
    mask = (Y >= lo) & (Y <= hi) & ~clip
    dn = np.median(ts["Dn"][mask], 0) - gray
    lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(srgb_encode(np.clip(out, 0, 1))[mask], 0, 1)))
    t = eng.scanner.timing
    off = t.M[0] if t.M is not None else np.zeros(3)
    return sid, d, timing, float(dn[2] - dn[1]), np.median(lab, 0).tolist(), [float(off[0] - off[1]), float(off[2] - off[1])]


if __name__ == "__main__":
    import multiprocessing as mp
    scenes = sys.argv[1:]
    jobs = [(s, d, "frame") for s in scenes for d in (-0.1, 0.0, 0.1)] + [(s, 0.0, "standard") for s in scenes]
    with mp.get_context("spawn").Pool(3, maxtasksperchild=1) as p:
        for sid, d, tm, nb, lab, off in p.imap(job, jobs):
            print(f"{sid:9s} {tm:8s} d={d:+.1f}  neg B−G {nb:+.3f}  成片 L {lab[0]:5.1f} a {lab[1]:+6.1f} b {lab[2]:+6.1f}  定光偏移 R {off[0]:+.3f} B {off[1]:+.3f}", flush=True)
