"""诊断: 逐帧定光 (frame) 的哪一部分压缩了光源偏色。对 balance_frame 得到的 PrintTiming 逐项替换, 测区域 L a b。
变体: frame 原样 / 斜率 r=1 / 去掉高光肩部 / 亮度 K 加上 O1 实测的 x_K / 三者都改 / standard 定光。"""
import json, sys, zlib, warnings
import numpy as np
from dataclasses import replace
warnings.filterwarnings("ignore")
from pathlib import Path; sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]
XK = {s: v["x"][0] for s, v in json.load(open(str(Path(__file__).resolve().parents[1] / "t04_result.json")))["frames"].items()}


def job(args):
    sid, var = args
    import colour, rawpy
    import oracle_timing as ot, pair_compare as pc
    from scipy.ndimage import zoom
    from filmlab import generation, rawio, scanner as scmod
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    from filmlab.io import srgb_encode
    generation.use("next")
    b0 = scmod.Scanner.balance_frame

    def bal(self, *a, **k):
        b0(self, *a, **k)
        t = self.timing
        if var in ("r1", "all"):
            t = replace(t, M=(t.M[0], np.ones(3)))
        if var in ("noknee", "all"):
            t = replace(t, x_knee=None, x_white=None)
        if var in ("K", "all"):
            t = replace(t, K=t.K + XK[sid])
        self.timing = t
    scmod.Scanner.balance_frame = bal
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "standard" if var == "standard" else "frame",
                           "roll": None, "frame": seed % 36, "grain": 0}))
    out = np.clip(eng.render(m), 0, None)
    Y = m @ np.array([0.2126, 0.7152, 0.0722])
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    clip = np.zeros(Y.shape, bool) if sat is None else \
        zoom(sat.astype(float), (Y.shape[0] / sat.shape[0], Y.shape[1] / sat.shape[1]), order=1)[:Y.shape[0], :Y.shape[1]] > 0.05
    lo, hi = np.percentile(Y[~clip], [40, 90])
    mask = (Y >= lo) & (Y <= hi) & ~clip
    lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(srgb_encode(np.clip(out, 0, 1))[mask], 0, 1)))
    return sid, var, np.median(lab, 0).tolist()


if __name__ == "__main__":
    import multiprocessing as mp
    tr = json.load(open(sys.argv[1]))
    scenes = sys.argv[2:]
    jobs = [(s, v) for s in scenes for v in ("frame", "r1", "noknee", "K", "all", "standard")]
    with mp.get_context("spawn").Pool(3, maxtasksperchild=1) as p:
        for sid, var, lab in p.imap(job, jobs):
            rl = tr[sid]["real_Lab"]
            print(f"{sid:9s} {var:8s} L {lab[0]:5.1f} a {lab[1]:+6.1f} b {lab[2]:+6.1f}   (真实 L {rl[0]:5.1f} a {rl[1]:+6.1f} b {rl[2]:+6.1f})", flush=True)
