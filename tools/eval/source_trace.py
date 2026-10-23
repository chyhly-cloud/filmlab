"""光源周围的亮度在成像链的哪一级被扩散: 逐级的径向亮度剖面 (到传感器饱和区的距离, 胶片 µm)。

级: 传感器 (只解码、不降噪不重建) → 降噪 + 高光重建 (load_raw 默认) → 胶片曝光 (绿层 H, 关闭光晕) → 成片 (关闭光晕),
以及真实扫描 (线性化)。都对齐到真实扫描; 每级取距离分档的 log10 亮度中位数, 减去该级在 1000–1500 µm 的水平。
真实扫描是成片, 只与"成片"同级比较; 前几级用来定位扩散在哪里发生。

用法: python3 source_trace.py 输出.json 场景…
"""
import json, os, sys, zlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")

BINS = np.array([0, 25, 50, 100, 150, 200, 300, 400, 500, 650, 800, 1000, 1500])


def job(sid):
    import colour
    import oracle_timing as ot
    from scipy.ndimage import distance_transform_edt, zoom
    from skimage.transform import warp, ProjectiveTransform
    from filmlab import generation, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    generation.use("next")
    Hm, real = ot.geometry(sid)
    rawp, _, bias = ot.sources(sid)
    tf = ProjectiveTransform(Hm)
    wp = lambda a: warp(a, tf.inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
    import rawpy
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    sensor = rawio.load_raw(rawp, max_size=1400, highlights="clip", denoise_raw=False)
    default = rawio.load_raw(rawp, max_size=1400)
    step = sat.shape[0] / sensor.shape[0]
    satm = zoom(sat.astype(float), (sensor.shape[0] / sat.shape[0], sensor.shape[1] / sat.shape[1]), order=1) > 0.25
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(default, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": seed % 36}))
    eng.halo = None
    neg = eng.develop_negative(m)
    out = eng.scan(neg)
    Y = lambda a: a @ np.array([0.2126, 0.7152, 0.0722])
    stages = {"传感器": Y(sensor), "降噪+重建": Y(default), "胶片曝光 (G)": 10.0 ** neg.log_exposure[..., 1],
              "成片": Y(np.clip(out, 0, None)), }
    src = wp(satm.astype(float)) > 0.5
    um = 36000.0 / max(real.shape[:2])
    d = distance_transform_edt(~src) * um
    res = {"n_src": int(src.sum())}
    for k, v in list(stages.items()) + [("真实扫描", colour.models.eotf_sRGB(real) @ [0.2126, 0.7152, 0.0722])]:
        w = wp(v) if k != "真实扫描" else v
        lv = np.log10(np.clip(w, 1e-5, None))
        prof = []
        for a, b in zip(BINS[:-1], BINS[1:]):
            mk = np.isfinite(w) & (d >= a) & (d < b) & ~src
            prof.append(float(np.median(lv[mk])) if mk.sum() > 30 else np.nan)
        res[k] = (np.array(prof) - prof[-1]).tolist()
    return sid, res


if __name__ == "__main__":
    out = sys.argv[1]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3"))) as p:
        res = dict(p.imap(job, sys.argv[2:]))
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    keys = [k for k in next(iter(res.values())) if k != "n_src"]
    print("光源外 log10 亮度增量 (相对 1000–1500 µm); 列 = 距离档上沿 µm")
    print(f"{'':16s}" + "".join(f"{b:>6d}" for b in BINS[1:-1]))
    for k in keys:
        v = np.nanmedian([r[k] for r in res.values()], 0)[:-1]
        print(f"{k:14s}" + "".join(f"{x:6.2f}" for x in v))
