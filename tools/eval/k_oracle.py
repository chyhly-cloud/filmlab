"""亮度预言机评估 (T06 起): 每帧只放开一个亮度自由度 K (= 数据中缺失的胶片实际曝光 + 冲印店的亮度光号, 二者都是
真实存在、但本数据无法观测的量), 对准真实扫描后再比较颜色。颜色光号 (R、B 偏移) 不放开。

做法: 与 T04 相同的入口与捕获 (lab_timing.capture_auto: next 标称配置, roll=None, grain=0, 实际自动定光), 在 24 px 格子上
用 oracle_timing 的同一损失 (Lab 残差, soft_l1, f_scale 5) 只拟合 K 的平移 (界 ±1.5), 再把 K 套回整幅图, 算全图 ΔE00 中位。
同时报告原指标 (auto, 不拟合)。变体见 accept_variant.py (在工作进程内施加)。

判定 (对比同一评估下的 next, 只看留出集; 条件与 T05 / accept_variant 相同): EPFL 留出平均改善 ≥ 0.5;
任何留出分组变差 ≤ 0.5; 任何留出场景变差 ≤ 1.5。咖啡店与训练集只报告。

用法: FILMLAB_DATA=… python3 k_oracle.py 变体或none 输出.json   (先用 none 生成 k_oracle_next.json 作为基准)
"""
import json, os, sys, warnings
from dataclasses import replace
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from accept_calibration import TRAIN, HOLDOUT


def job(args):
    variant, sid = args
    if variant != "none":
        import accept_variant
        accept_variant.install(variant)
    import colour
    from scipy.optimize import least_squares
    from skimage.transform import warp, ProjectiveTransform
    import oracle_timing as ot, lab_timing as lt
    from filmlab.io import srgb_encode
    c = lt.capture_auto(sid)
    real_lab = lt.encoded_lab(c["real"])
    full = lambda enc: lt.ledger_stage(enc, real_lab, {"全图": c["ok"]})["全图"]["dE"]
    de_auto = full(c["aligned"])
    tf = ProjectiveTransform(c["H"])
    wp = lambda a: warp(a, tf.inverse, output_shape=c["real"].shape[:2], order=1, cval=np.nan)
    dw = wp(c["Dn"])
    ok = c["ok"] & np.all(np.isfinite(dw), -1)
    dc, _ = ot.cells(dw, ok)
    R = ot.lab_lin(colour.models.eotf_sRGB(ot.cells(c["real"], ok)[0]))
    keep = R[:, 0] > 5
    sc, t0 = c["scanner"], c["timing"]

    def resid(x):
        sc.timing = replace(t0, K=t0.K + x[0])
        return np.nan_to_num((ot.lab_lin(sc.to_rgb(dc)) - R)[keep].ravel(), nan=50.0)
    x = least_squares(resid, [0.0], bounds=([-1.5], [1.5]), loss="soft_l1", f_scale=5.0).x
    sc.timing = replace(t0, K=t0.K + x[0])
    lin = sc.to_rgb(c["Dn"].reshape(-1, 3)).reshape(c["Dn"].shape)
    de_k = full(wp(srgb_encode(np.clip(lin, 0, 1))))
    sc.timing = t0
    return sid, {"auto": float(de_auto), "k": float(de_k), "dK": float(x[0])}


def verdict(res, base):
    H = {s: g for s, g in HOLDOUT.items() if s != "cafe"}
    d = {s: res[s]["k"] - base[s]["k"] for s in res}
    ok, msg = True, []
    m = float(np.mean([d[s] for s in H]))
    if m > -0.5:
        ok = False; msg.append(f"EPFL 留出平均变化 {m:+.3f} (需 ≤ −0.5)")
    groups = {g: float(np.mean([d[s] for s in H if H[s] == g])) for g in set(H.values())}
    for g, v in groups.items():
        if v > 0.5:
            ok = False; msg.append(f"留出分组 {g} 变差 {v:+.2f}")
    for s in H:
        if d[s] > 1.5:
            ok = False; msg.append(f"留出场景 {s} 变差 {d[s]:+.2f}")
    rep = {"holdout_epfl_change": m, "holdout_groups": groups, "train_change_in_sample": float(np.mean([d[s] for s in TRAIN])),
           "cafe_change_report_only": d.get("cafe")}
    return ok, msg, rep


if __name__ == "__main__":
    variant, out = sys.argv[1], sys.argv[2]
    import multiprocessing as mp
    S = {**TRAIN, **HOLDOUT}
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3")), maxtasksperchild=1) as p:
        res = dict(p.map(job, [(variant, s) for s in S]))
    json.dump(res, open(out, "w"), indent=1)
    for s in sorted(res, key=lambda s: S[s]):
        print(f"{s:9s} {S[s]:5s} auto {res[s]['auto']:6.2f}  K 预言机 {res[s]['k']:6.2f}  dK {res[s]['dK']:+.3f}")
    print("平均 auto", round(np.mean([v["auto"] for v in res.values()]), 3), " K 预言机", round(np.mean([v["k"] for v in res.values()]), 3))
    bf = HERE / "k_oracle_next.json"
    if variant != "none" and bf.exists():
        ok, msg, rep = verdict(res, json.load(open(bf)))
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        print("通过" if ok else "不通过: " + "; ".join(msg))
