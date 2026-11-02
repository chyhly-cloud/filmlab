"""控制帧 (图卡 + 灯箱) 上, 各候选的"实际 / 单通道预测"(模型自身的中性传递) 按色相。真实列不随候选变。"""
import numpy as np, sys, warnings, multiprocessing as mp, os
warnings.filterwarnings('ignore')
FR = [("R1-20-21", "_t31_base_ev-1"), ("R1-30-31", "_t31_base"), ("R1-31-32", "_t31_base"), ("R1-25-26", "_t31_base"), ("R1-27-28", "_t31_base"),
      ("R1-23-24", "_t31_base"), ("R1-35-36", "_t31_base"), ("R1-26-27", "_t31_base"), ("R1-32-33", "_t31_base"), ("R1-33-34", "_t31_base"),
      ("R1-28-29", "_t31_base"), ("R1-21-22", "_t31_base")]
def job(a):
    cand, sid, tag = a
    import t32_counterfactuals as cands, t32_rescan as rescan
    out, tr, z = rescan.render(getattr(cands, cand), sid, tag)
    f = f"cand/tmp_{cand}_{sid}.npz"
    d = dict(z); d["out"] = out.astype(np.float32); np.savez(f, **d)
    import t32_hue_ci as hue_ci
    r, nn = hue_ci.run(f, show=False)
    os.remove(f)
    return cand, sid, r
if __name__ == "__main__":
    cs = sys.argv[1:]
    with mp.get_context("spawn").Pool(4, maxtasksperchild=1) as p:
        res = p.map(job, [(c, s, t) for c in cs for s, t in FR], chunksize=1)
    tab = {}
    for c, s, rows in res:
        for r in rows:
            tab.setdefault((r['hue'], r['c']), {}).setdefault(c, []).append((s, r))
    for (h, cb), d in sorted(tab.items(), key=lambda kv: (["红","黄","绿","青","蓝","品红"].index(kv[0][0]), kv[0][1])):
        c0 = cs[0]
        for s, r in d[c0]:
            line = f"{h:3s} {cb[0]:.2f}–{cb[1]:.2f} {s} n={r['n']:6d} | 真实 实际/预测 {r['r_real'].round(2)} G/R {r['gr_real']:.2f}"
            for c in cs:
                rr = [x for ss, x in d.get(c, []) if ss == s]
                if rr: line += f" | {c} {rr[0]['r_out'].round(2)} G/R {rr[0]['gr_out']:.2f}"
            print(line)
