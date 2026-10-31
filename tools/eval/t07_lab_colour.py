"""T07 冲印店颜色处理的实测: 以 "固定光号" (standard: 800T 设计光源中灰印成中性) 为参照, O1 拟合的颜色偏移 x_R、x_B 就是
冲印店对这一帧额外做的颜色调整; 横轴是模型底片自身的偏色。回归 x = a − k·cast: k = 校色力度 (1 = 完全中和, 0 = 不校正)。

基础配置 (预先写定): accept_variant 的 fixed_exposure,plain (胶片与数码同一份曝光; 不套他店输出调整), timing="standard",
next, roll=None, grain=0。偏色估计量 (相对档案中灰, R−G / B−G, 密度):
  mean   整帧算术平均密度 (主估计量; Fuji US6668096B1 的平均密度 / 密度直方图)
  latd   透射加权平均密度 (印放机积分中性)
  bright 输入亮度 40–90 百分位区域的中位密度
拟合与边界同 oracle_timing 的 O1 (lab_timing.oracle_fit, 原样执行)。只用 EPFL 20 帧; 咖啡店不参与。

用法: FILMLAB_DATA=… python3 t07_lab_colour.py 输出.json
"""
import json, os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from accept_calibration import TRAIN, HOLDOUT
BASE = "fixed_exposure,plain"


def job(sid):
    import accept_variant
    accept_variant.install(BASE)
    import colour
    from skimage.transform import warp, ProjectiveTransform
    import oracle_timing as ot, pair_compare as pc, lab_timing as lt
    from filmlab import generation, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    generation.use("next")
    Hm, real = ot.geometry(sid)
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = rawio.film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "standard", "roll": None, "frame": seed % 36, "grain": 0}))
    ts = {}
    eng.scan(eng.develop_negative(m), trace=ts)
    Dn, t0, sc = ts["Dn"], ts["timing"], eng.scanner
    gray = sc.normalized_density(eng._transmittance(eng._gray_state(actual_light=False)))[0]
    ok_d = np.all(np.isfinite(Dn), -1)
    rel = lambda v: [float((v[0] - gray[0]) - (v[1] - gray[1])), float((v[2] - gray[2]) - (v[1] - gray[1]))]
    Y = m @ np.array([0.2126, 0.7152, 0.0722])
    lo, hi = np.percentile(Y, [40, 90])
    br = (Y >= lo) & (Y <= hi) & ok_d
    cast = {"mean": rel(Dn[ok_d].mean(0)), "latd": rel(-np.log10(np.mean(10.0 ** -Dn[ok_d], 0))),
            "bright": rel(np.median(Dn[br], 0))}
    # 亮度水平 (绿通道相对档案中灰, 密度): 冲印店逐帧亮度调整的横轴 (T08)
    level = {"mean": float(Dn[ok_d].mean(0)[1] - gray[1]),
             "latd": float(-np.log10(np.mean(10.0 ** -Dn[ok_d][:, 1])) - gray[1]),
             "bright": float(np.median(Dn[br][:, 1]) - gray[1])}
    tf = ProjectiveTransform(Hm)
    dw = warp(Dn, tf.inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
    ok = np.all(np.isfinite(dw), -1)
    dc, _ = ot.cells(dw, ok)
    R = ot.lab_lin(colour.models.eotf_sRGB(ot.cells(real, ok)[0]))
    keep = R[:, 0] > 5

    def render(t):
        sc.timing = t
        return sc.to_rgb(dc)
    _, x = lt.oracle_fit(render, t0, dc, R, keep, 1)
    sc.timing = t0
    return sid, {"x": [float(v) for v in x], "cast": cast, "level": level}


def regress(res, S, est):
    out = {}
    for j, ch in ((0, "R"), (1, "B")):
        c = np.array([res[s]["cast"][est][j] for s in S]); x = np.array([res[s]["x"][j + 1] for s in S])
        b, a = np.polyfit(c, x, 1)
        out[ch] = {"a": float(a), "k": float(-b), "r": float(np.corrcoef(c, x)[0, 1])}
    return out


if __name__ == "__main__":
    import multiprocessing as mp
    E = [s for s in {**TRAIN, **HOLDOUT} if s != "cafe"]
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3")), maxtasksperchild=1) as p:
        res = dict(p.map(job, E))
    tr = [s for s in E if s in TRAIN]; ho = [s for s in E if s in HOLDOUT]
    summary = {}
    for est in ("mean", "latd", "bright"):
        fit = regress(res, tr, est)
        pred = {ch: [fit[ch]["a"] - fit[ch]["k"] * res[s]["cast"][est][j] for s in ho] for j, ch in ((0, "R"), (1, "B"))}
        act = {ch: [res[s]["x"][j + 1] for s in ho] for j, ch in ((0, "R"), (1, "B"))}
        hold = {ch: {"r": float(np.corrcoef(pred[ch], act[ch])[0, 1]),
                     "rmse_model": float(np.sqrt(np.mean((np.array(pred[ch]) - act[ch]) ** 2))),
                     "rmse_const": float(np.sqrt(np.mean((np.mean([res[s]["x"][j + 1] for s in tr]) - np.array(act[ch])) ** 2)))}
                for j, ch in ((0, "R"), (1, "B"))}
        summary[est] = {"train_fit": fit, "holdout": hold}
        print(f"\n[{est}]  训练: R k={fit['R']['k']:+.3f} a={fit['R']['a']:+.3f} r={fit['R']['r']:+.2f} | "
              f"B k={fit['B']['k']:+.3f} a={fit['B']['a']:+.3f} r={fit['B']['r']:+.2f}")
        for ch in ("R", "B"):
            h = hold[ch]
            print(f"   留出 {ch}: 预测与实测相关 {h['r']:+.2f}  RMSE 模型 {h['rmse_model']:.3f} / 常数 {h['rmse_const']:.3f}")
    G = {**TRAIN, **HOLDOUT}
    print()
    for s in sorted(E, key=lambda s: G[s]):
        r = res[s]
        print(f"{s:9s} {G[s]:5s} x K {r['x'][0]:+.3f} R {r['x'][1]:+.3f} B {r['x'][2]:+.3f} | mean 偏色 R {r['cast']['mean'][0]:+.3f} B {r['cast']['mean'][1]:+.3f}")
    json.dump({"frames": res, "summary": summary}, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
