"""T09: next_lab 上叠加光晕斜程衰减 (halo_oblique) 是否仍有益。对比 next_lab 本身, 21 个场景, 入口同验收。
判定 (运行前写定, 沿用 T03): EPFL 留出全图平均不变差 (≤ +0.05); 留出光源旁平均改善 (< 0); 任何 EPFL 场景变差 ≤ 1.0。训练集同时报告。

用法: FILMLAB_DATA=… python3 t09_oblique_on_lab.py 输出.json
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from accept_calibration import TRAIN, HOLDOUT


def job(args):
    gen, sid = args
    from filmlab import generation
    if gen == "next_lab_oblique":
        w, corr, rev = generation.GENERATIONS["next_lab"]
        generation.GENERATIONS[gen] = (w, corr + ("halo_oblique",), rev)
        generation.METERING[gen] = generation.METERING["next_lab"]
    import accept_t03
    return gen, sid, accept_t03.scene_job((gen, sid))[1]


if __name__ == "__main__":
    import multiprocessing as mp
    G = {**TRAIN, **HOLDOUT}
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3")), maxtasksperchild=1) as p:
        out = p.map(job, [(g, s) for s in G for g in ("next_lab", "next_lab_oblique")])
    res = {s: {} for s in G}
    for g, s, r in out:
        res[s][g] = r
    d = {s: res[s]["next_lab_oblique"]["full"] - res[s]["next_lab"]["full"] for s in G}
    dn = {s: res[s]["next_lab_oblique"]["near"] - res[s]["next_lab"]["near"] for s in G}
    H = [s for s in HOLDOUT if s != "cafe"]; E = [s for s in G if s != "cafe"]
    mf = float(np.mean([d[s] for s in H])); mn = float(np.nanmean([dn[s] for s in H]))
    worst = max(E, key=d.get)
    ok = mf <= 0.05 and mn < 0 and d[worst] <= 1.0
    for s in sorted(G, key=lambda s: G[s]):
        print(f"{s:9s} {G[s]:5s} 全图 {res[s]['next_lab']['full']:6.2f} → {res[s]['next_lab_oblique']['full']:6.2f} ({d[s]:+.2f})  光源旁 {dn[s]:+.2f}")
    print(f"\n留出: 全图 {mf:+.3f}, 光源旁 {mn:+.3f}; 训练: 全图 {np.mean([d[s] for s in TRAIN]):+.3f}, 光源旁 {np.nanmean([dn[s] for s in TRAIN]):+.3f}; "
          f"最大变差 {worst} {d[worst]:+.2f} → {'通过' if ok else '不通过'}")
    json.dump({"scenes": res, "pass": ok}, open(sys.argv[1], "w"), indent=1)
