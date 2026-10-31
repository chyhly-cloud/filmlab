"""T06 光源颜色逐级追踪: 同一区域 (画面大面积亮区, 灯箱里即白墙) 的颜色在每一级是多少, 光源之间的差在哪一级塌缩。

配置与 T03 / T04 / T05 验收相同 (next, roll=None, grain=0, 实际自动定光)。区域: 输入亮度 40–90 百分位、非截断的像素
(在模拟几何里定义, 再按固定单应性变换到真实扫描几何)。各级 (全部为区域中位数):
  cam   相机线性 RGB (RAW 日光白平衡, 测光后): log10(R/G), log10(B/G)。日光白 = 0; 钨丝灯 R/G > 0、B/G < 0。
  expo  三层乳剂直接曝光 H (散射与光晕之前): log10(H/H_G) − 同一胶片在设计光源 (3200K) 下中灰的 log10(H/H_G)。
        0 = 对胶片而言与 3200K 中性无异; 冷光 → B 层多。
  neg   归一化底片密度 Dn − 扫描档案中灰 (自动定光以它为中性): 底片相对档案的偏色 (密度)。
  sim / real  成片与真实扫描的 CIELAB a*, b*。
另记录模型对该帧估计的光源类别份额 (light_mix)。

用法: FILMLAB_DATA=… python3 light_colour_trace.py 输出.json [场景…]   (默认 21 个验收场景)
"""
import json, os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(Path(__file__).resolve().parent), str(REPO)]


def job(sid):
    import colour, rawpy
    import oracle_timing as ot, pair_compare as pc
    from scipy.ndimage import zoom
    from skimage.transform import warp, ProjectiveTransform
    from filmlab import generation, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    from filmlab.io import srgb_encode
    generation.use("next")
    Hm, real = ot.geometry(sid)
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": seed % 36, "grain": 0}))
    tn, ts = {}, {}
    neg = eng.develop_negative(m, trace=tn)
    out = np.clip(eng.scan(neg, trace=ts), 0, None)
    Dn = ts["Dn"]
    gray = eng.scanner.normalized_density(eng._transmittance(eng._gray_state(actual_light=False)))[0]
    # 3200K 中灰对胶片的层间比例
    g_ref = eng._expose_lin(np.full((1, 1, 3), 0.18), eng.M_ref, compensation=False)[0, 0]
    # 区域
    Y = m @ np.array([0.2126, 0.7152, 0.0722])
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    clip = np.zeros(Y.shape, bool) if sat is None else \
        zoom(sat.astype(float), (Y.shape[0] / sat.shape[0], Y.shape[1] / sat.shape[1]), order=1)[:Y.shape[0], :Y.shape[1]] > 0.05
    lo, hi = np.percentile(Y[~clip], [40, 90])
    mask = (Y >= lo) & (Y <= hi) & ~clip
    med = lambda a: np.median(a[mask], 0)
    lg = lambda v: [float(np.log10(v[0] / v[1])), float(np.log10(v[2] / v[1]))]
    cam = lg(med(m))
    H = tn["direct_exposure"]
    e = [float(np.log10(med(H)[c] / med(H)[1]) - np.log10(g_ref[c] / g_ref[1])) for c in (0, 2)]
    dn = med(Dn) - gray
    negc = [float(dn[0] - dn[1]), float(dn[2] - dn[1])]
    tf = ProjectiveTransform(Hm)
    W = lambda a, o=1: warp(a, tf.inverse, output_shape=real.shape[:2], order=o, cval=np.nan)
    mk = W(mask.astype(float)) > 0.5
    sim = W(srgb_encode(np.clip(out, 0, 1)))
    ok = mk & np.all(np.isfinite(sim), -1)
    lab = lambda x: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(x, 0, 1)))
    ls, lr = np.median(lab(sim[ok]), 0), np.median(lab(real[ok]), 0)
    mix = {k: [v[0], round(float(v[1]), 3)] for k, v in (eng.light_mix or {}).items() if v[1] >= 0.02}
    return sid, {"n": int(ok.sum()), "cam": cam, "expo": e, "neg": negc,
                 "sim_Lab": ls.tolist(), "real_Lab": lr.tolist(), "light_mix": mix}


if __name__ == "__main__":
    from accept_calibration import TRAIN, HOLDOUT
    G = {**TRAIN, **HOLDOUT}
    scenes = sys.argv[2:] or list(G)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3")), maxtasksperchild=1) as p:
        res = dict(p.map(job, scenes))
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    print(f"{'场景':9s} {'组':5s} | {'cam R/G':>7s} {'B/G':>6s} | {'expo R':>6s} {'B':>6s} | {'neg R':>6s} {'B':>6s} | "
          f"{'sim a':>6s} {'b':>6s} | {'real a':>6s} {'b':>6s} | 光源")
    for s in sorted(res, key=lambda s: G[s]):
        r = res[s]
        print(f"{s:9s} {G[s]:5s} | {r['cam'][0]:+7.3f} {r['cam'][1]:+6.3f} | {r['expo'][0]:+6.3f} {r['expo'][1]:+6.3f} | "
              f"{r['neg'][0]:+6.3f} {r['neg'][1]:+6.3f} | {r['sim_Lab'][1]:+6.1f} {r['sim_Lab'][2]:+6.1f} | "
              f"{r['real_Lab'][1]:+6.1f} {r['real_Lab'][2]:+6.1f} | {r['light_mix']}")
