"""光源附近的色差: 传感器饱和区内部 (灯芯) 与外侧 0–300 µm (灯的周围), 真实扫描 vs 各 RAW 处理方式。

条件: next (旧截断判定 + 先验重建) / sensor (传感器饱和判定 + 先验重建) / sensor+逐通道 (保留未饱和通道的实测值)。
用法: python3 source_region_de.py 输出.json 场景…
"""
import json, os, sys, zlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")

CONDS = {"next": ("channel_max", "prior"), "sensor": ("sensor", "prior"), "sensor+逐通道": ("sensor", "per_channel")}


def job(sid):
    import colour, rawpy
    import oracle_timing as ot
    from scipy.ndimage import distance_transform_edt, zoom
    from skimage.transform import warp, ProjectiveTransform
    from filmlab import generation, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    from filmlab.io import srgb_encode
    generation.use("next")
    Hm, real = ot.geometry(sid)
    rawp, _, bias = ot.sources(sid)
    tf = ProjectiveTransform(Hm)
    wp = lambda a, o=1: warp(a, tf.inverse, output_shape=real.shape[:2], order=o, cval=np.nan)
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    seed = zlib.crc32(sid.encode()) % 1000
    lab = lambda x: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(x, 0, 1)))
    Lr = lab(real)
    res = {}
    for name, (det, rec) in CONDS.items():
        rawio.HIGHLIGHT_RECON = rec
        rgb = rawio.load_raw(rawp, max_size=1400, clip_detect=det)
        m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": seed % 36}))
        sim = wp(srgb_encode(np.clip(eng.render(m), 0, 1)))
        if "core" not in res:
            s = zoom(sat.astype(float), (rgb.shape[0] / sat.shape[0], rgb.shape[1] / sat.shape[1]), order=1) > 0.25
            core = wp(s.astype(float)) > 0.5
            um = 36000.0 / max(real.shape[:2])
            d = distance_transform_edt(~core) * um
            near = (d > 0) & (d <= 300)
            res["core"] = int(core.sum())
        ok = np.all(np.isfinite(sim), -1)
        Ls = lab(np.where(ok[..., None], sim, 0))
        out = {}
        for reg, mk in (("灯芯", core & ok), ("周围 0–300 µm", near & ok)):
            de = colour.delta_E(Ls[mk], Lr[mk], method="CIE 2000")
            dl = Ls[mk] - Lr[mk]
            out[reg] = [float(np.median(de)), *map(float, np.median(dl, 0))]
        res[name] = out
    rawio.HIGHLIGHT_RECON = "prior"
    return sid, res


if __name__ == "__main__":
    out = sys.argv[1]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3"))) as p:
        res = dict(p.imap(job, sys.argv[2:]))
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    for reg in ("灯芯", "周围 0–300 µm"):
        print(f"== {reg}: ΔE00 中位 / ΔL / Δa / Δb (模拟 − 真实)")
        for sid, r in res.items():
            print(f"  {sid:9s} " + " | ".join(f"{c} {r[c][reg][0]:5.1f} {r[c][reg][1]:+5.1f} {r[c][reg][2]:+5.1f} {r[c][reg][3]:+5.1f}" for c in CONDS))
        for c in CONDS:
            v = np.mean([r[c][reg] for r in res.values()], 0)
            print(f"  平均 {c:14s} ΔE {v[0]:5.2f}  ΔL {v[1]:+5.1f}  Δa {v[2]:+5.1f}  Δb {v[3]:+5.1f}")
