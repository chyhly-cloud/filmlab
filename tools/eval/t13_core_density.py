"""诊断: R1-03-4 灯芯 (传感器饱和) 处各层直接曝光、底片归一化密度与成片; 对照该胶片各层在极大曝光下的密度 (饱和上限)。"""
import sys, zlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
from pathlib import Path; sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]


def run(sid, stops):
    import rawpy
    from scipy.ndimage import zoom
    import oracle_timing as ot, pair_compare as pc
    from filmlab import generation, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    generation.use("next_lab_halo")
    rawio.SOURCE_STOPS = stops
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = rawio.film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": seed % 36, "grain": 0}))
    tn, ts = {}, {}
    out = np.clip(eng.scan(eng.develop_negative(m, trace=tn), trace=ts), 0, None)
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    core = zoom(sat.astype(float), (m.shape[0] / sat.shape[0], m.shape[1] / sat.shape[1]), order=1)[:m.shape[0], :m.shape[1]] > 0.75
    H, Dn = tn["total_exposure"][core], ts["Dn"][core]
    gray = eng.scanner.normalized_density(eng._transmittance(eng._gray_state(actual_light=False)))[0]
    # 饱和上限: 中性极大曝光 (中灰 × 2^12) 下的底片密度
    big = eng._expose_lin(np.full((1, 1, 3), 0.18 * 2 ** 12), eng.M_ref, compensation=False)
    st = eng._develop(eng._log_exposure(big), blur_px=0.0)
    dmax = eng.scanner.normalized_density(eng._transmittance(st))[0]
    q = lambda a: np.round(np.percentile(a, [50, 90], axis=0), 2)
    print(f"{sid} 光源 +{stops} 档 | 灯芯像素 {core.sum()}")
    print("  总曝光 (相对中灰=1 的 log10) 中位/90%:", np.round(np.log10(np.percentile(H, [50, 90], axis=0) / 0.18), 2).tolist())
    print("  底片 Dn − 档案中灰 中位/90%:", (q(Dn) - gray).tolist())
    print("  中性极大曝光下 Dn − 档案中灰 (饱和上限):", np.round(dmax - gray, 2).tolist())
    print("  成片线性 RGB 中位:", np.round(np.median(out[core], 0), 2).tolist())


if __name__ == "__main__":
    for st in (0.0, 3.0, 6.0):
        run("R1-03-4", st)
