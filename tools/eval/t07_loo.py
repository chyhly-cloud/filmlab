"""T07 候选的逐帧留一交叉验证: 对 EPFL 20 帧中的每一帧, 只用其余 19 帧重新测候选的 5 个参数, 再渲染被留出的那一帧。
  曝光常数 = 其余 19 帧 film_metering 倍数 (t06/meter_result.json) 的 log10 中位数;
  冲印店颜色 (R、B 各 a、k) = 其余 19 帧 t07_result.json 的 x 对 mean 偏色的回归 (与 t07_lab_colour.py 同一定义)。
近似: 偏色取自 T07 已测的每帧值 (测量时曝光常数为训练集中位数), 偏色对曝光常数不敏感。
对比: next (t03_baseline.json) 与全参数候选 (t07_accept_full.json)。入口同验收 (accept_t03.scene_job)。

用法: FILMLAB_DATA=… python3 t07_loo.py 输出.json
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from accept_calibration import TRAIN, HOLDOUT
CAND = "fixed_exposure,plain,lab_colour"


def params(leave):
    meter = json.load(open(HERE / "t06" / "meter_result.json"))
    fr = json.load(open(HERE / "t07_result.json"))["frames"]
    S = [s for s in fr if s != leave]
    ls = float(np.median([meter[s]["log_scale"] for s in S]))
    lc = {}
    for j, ch in ((0, "R"), (1, "B")):
        c = np.array([fr[s]["cast"]["mean"][j] for s in S]); x = np.array([fr[s]["x"][j + 1] for s in S])
        b, a = np.polyfit(c, x, 1)
        lc[ch] = (float(a), float(-b))
    return ls, lc


def job(sid):
    import accept_variant
    ls, lc = params(sid)
    accept_variant.FIXED_LOG_SCALE = ls
    accept_variant.LAB_COLOUR = lc
    accept_variant.install(CAND)
    import accept_t03
    return sid, {"full": accept_t03.scene_job(("next", sid))[1]["full"], "log_scale": ls, "lab_colour": lc}


if __name__ == "__main__":
    import multiprocessing as mp
    G = {**TRAIN, **HOLDOUT}
    E = [s for s in G if s != "cafe"]
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3")), maxtasksperchild=1) as p:
        res = dict(p.map(job, E))
    base = json.load(open(HERE / "t03_baseline.json"))["scenes"]
    full = json.load(open(HERE / "t07_accept_full.json"))["scenes"]
    rows = []
    for s in sorted(E, key=lambda s: G[s]):
        r = res[s]
        d_loo, d_full = r["full"] - base[s]["full"], full[s]["full"] - base[s]["full"]
        rows.append((s, d_loo, d_full))
        print(f"{s:9s} {G[s]:5s} next {base[s]['full']:6.2f} | 留一 {r['full']:6.2f} ({d_loo:+.2f}) | 全参数 {full[s]['full']:6.2f} ({d_full:+.2f}) | "
              f"log 倍数 {r['log_scale']:.3f}  kR {r['lab_colour']['R'][1]:.3f} kB {r['lab_colour']['B'][1]:.3f}")
    dl = np.array([r[1] for r in rows]); df = np.array([r[2] for r in rows])
    print(f"\n20 帧平均变化: 留一 {dl.mean():+.3f} / 全参数 {df.mean():+.3f};  变差的帧: 留一 {int((dl > 0).sum())} / 全参数 {int((df > 0).sum())}")
    print(f"留一与全参数的差: 平均 {np.mean(dl - df):+.3f}, 最大 {np.max(np.abs(dl - df)):.3f}")
    ks = np.array([[res[s]['lab_colour']['R'][1], res[s]['lab_colour']['B'][1], res[s]['log_scale']] for s in E])
    print(f"参数范围: kR {ks[:,0].min():.3f}–{ks[:,0].max():.3f}  kB {ks[:,1].min():.3f}–{ks[:,1].max():.3f}  log 倍数 {ks[:,2].min():.3f}–{ks[:,2].max():.3f}")
    json.dump(res, open(sys.argv[1], "w"), indent=1)
