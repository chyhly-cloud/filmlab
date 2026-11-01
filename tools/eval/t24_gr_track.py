"""T24: 光晕的绿/红之比 —— 与灯的亮度无关的测量 (EPFL 真实冲扫 vs 模型, 同一配准几何)。

光晕主导的像素里 (模型红层 光晕/直接 > 3, 距灯芯 100–1500 µm), 光晕越弱颜色越从橙走向红。
灯的亮度只决定像素落在这条轨迹的哪个位置, 不改变轨迹本身; 乳剂散射 (film.TURBIDITY) 改变轨迹。
按成片线性 R 分档 (log2), 比较每档 G/R 中位数: 真实 vs 模型。

用法: FILMLAB_DATA=… python3 t24_gr_track.py 输出.json 代际 散射值,散射值,… [场景…]"""
import json, os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools/eval"), str(REPO)]
SCENES = ["R1-00-1", "R1-01-2", "R1-02-3", "R1-03-4", "R1-04-5", "R1-05-6", "R1-06-7", "R1-07-8"]
RB = np.arange(-4.0, 0.51, 0.5)          # log2 成片线性 R 的分档边界


def job(args):
    gen, turb, sid = args
    import colour, rawpy
    import oracle_timing as ot, pair_compare as pc
    from scipy.ndimage import distance_transform_edt, zoom
    from skimage.transform import warp, ProjectiveTransform
    from filmlab import generation, rawio, film
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    film.TURBIDITY = turb
    generation.use(gen)
    Hm, real = ot.geometry(sid)
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": seed % 36, "grain": 0}))
    tr = {}
    out_lin = np.clip(eng.scan(eng.develop_negative(m, trace=tr)), 0, None)
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    core = zoom(sat.astype(float), (m.shape[0] / sat.shape[0], m.shape[1] / sat.shape[1]), order=1) > 0.25
    tf = ProjectiveTransform(Hm)
    W = lambda a, o=1: warp(a, tf.inverse, output_shape=real.shape[:2], order=o, cval=np.nan)
    halo = tr["total_exposure"] - tr["scattered_exposure"]
    ratio = W(halo[..., 0] / np.maximum(tr["direct_exposure"][..., 0], 1e-9))
    mo, re = W(out_lin), colour.models.eotf_sRGB(real)
    cw = W(core.astype(float)) > 0.5
    um = 36000.0 / max(real.shape[:2])
    d = distance_transform_edt(~cw) * um
    mask = (d > 100) & (d < 1500) & (ratio > 3.0) & np.all(np.isfinite(mo), -1)
    res = {"n": int(mask.sum())}
    for k, a in (("real", re), ("model", mo)):
        R, G = a[mask][:, 0], a[mask][:, 1]
        lr = np.log2(np.maximum(R, 1e-6))
        rows = []
        for lo, hi in zip(RB[:-1], RB[1:]):
            s = (lr >= lo) & (lr < hi)
            rows.append([int(s.sum()), float(np.median(G[s] / np.maximum(R[s], 1e-6))) if s.sum() >= 15 else None])
        res[k] = rows
    return gen, turb, sid, res


if __name__ == "__main__":
    import multiprocessing as mp
    dst, gen, turbs = sys.argv[1], sys.argv[2], [float(t) for t in sys.argv[3].split(",")]
    scenes = sys.argv[4:] or SCENES
    jobs = [(gen, t, s) for t in turbs for s in scenes]
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as p:
        rows = p.map(job, jobs, chunksize=1)
    res = {}
    for g, t, s, r in rows:
        res.setdefault(str(t), {})[s] = r
    json.dump(res, open(dst, "w"), indent=1)
    centers = (RB[:-1] + RB[1:]) / 2
    print("成片线性 R (log2) 档中心:", " ".join(f"{c:5.1f}" for c in centers))
    for t in turbs:
        errs = []
        print(f"\n== 散射 {t}")
        for s in scenes:
            r = res[str(t)][s]
            line_r = " ".join(f"{v[1]:5.2f}" if v[1] is not None else "    -" for v in r["real"])
            line_m = " ".join(f"{v[1]:5.2f}" if v[1] is not None else "    -" for v in r["model"])
            e = [np.log(m[1] / q[1]) for q, m in zip(r["real"], r["model"]) if q[1] and m[1] and q[1] > 0 and m[1] > 0]
            if e:
                errs.append(np.mean(e))
            print(f"  {s} n={r['n']:6d} 真实 G/R {line_r}\n  {'':17s} 模型 G/R {line_m}   平均 ln(模型/真实) {np.mean(e) if e else np.nan:+.3f}")
        print(f"  场景等权 平均 ln(模型/真实) = {np.mean(errs):+.3f}   |偏差| 均值 = {np.mean(np.abs(errs)):.3f}")
