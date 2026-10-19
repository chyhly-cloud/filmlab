"""高光影调传递: 同一像素的输入曝光 (数码 RAW, 相对中灰的档数) → 输出亮度, 模拟与真实扫描对比。

光源轮廓是否被"冲白"取决于高光在多少档时到达纯白。按输入曝光分档 (0.5 档), 取输出亮度 (线性) 的中位数与
"已到白 (Y > 0.95)"的比例。重建的截断高光 (亮度来自先验) 单独排除: 只用传感器未饱和的像素, 其曝光是测量值。

用法: python3 tone_transfer.py 输出.json 场景…
"""
import json, os, sys, zlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")

EDGES = np.arange(-6.0, 4.01, 0.5)


def job(sid):
    import colour, rawpy
    import oracle_timing as ot
    from scipy.ndimage import zoom, binary_dilation
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
    rgb = rawio.load_raw(rawp, max_size=1400)
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": seed % 36}))
    sim = np.clip(eng.render(m), 0, None)
    Y = lambda a: a @ np.array([0.2126, 0.7152, 0.0722])
    ev = wp(np.log2(np.clip(Y(m), 1e-6, None) / 0.184))
    if sat is not None:
        s = zoom(sat.astype(float), (m.shape[0] / sat.shape[0], m.shape[1] / sat.shape[1]), order=1) > 0.01
        s = binary_dilation(s, iterations=2)
    else:
        s = np.zeros(m.shape[:2], bool)
    measured = wp((~s).astype(float)) > 0.99
    ys, yr = wp(Y(sim)), colour.models.eotf_sRGB(real) @ [0.2126, 0.7152, 0.0722]
    rows = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        k = measured & np.isfinite(ev) & np.isfinite(ys) & (ev >= a) & (ev < b)
        rows.append([int(k.sum())] + ([float(np.median(ys[k])), float(np.median(yr[k])), float((ys[k] > 0.95).mean()),
                                         float((yr[k] > 0.95).mean())] if k.sum() > 50 else [np.nan] * 4))
    return sid, rows


if __name__ == "__main__":
    out = sys.argv[1]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3"))) as p:
        res = dict(p.imap(job, sys.argv[2:]))
    json.dump(res, open(out, "w"), indent=1)
    A = np.array([r for r in res.values()], float)          # 场景 × 档 × 5
    print("输入曝光 (相对中灰的档, 只含传感器未饱和像素) → 输出亮度中位 (模拟 / 真实) 与到白比例 (模拟 / 真实)")
    for i, (a, b) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        n = np.nansum(A[:, i, 0])
        if n < 200:
            continue
        v = np.nanmedian(A[:, i, 1:], 0)
        print(f"{a:+5.1f}…{b:+5.1f} 档  n={int(n):7d}   Y {v[0]:5.3f} / {v[1]:5.3f}   到白 {v[2]:5.2f} / {v[3]:5.2f}")
