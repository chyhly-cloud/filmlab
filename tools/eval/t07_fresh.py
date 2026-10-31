"""T07 候选的新数据独立验证: 只用此前从未参与任何评估的 EPFL 帧 (判定条件在运行前写定, 见 docs/tasks/T07_LAB_COLOUR.md)。
  判定帧 (已确认配对): R1-17-18 灰阶楔, R1-29-30 空灯箱·钨丝, R1-34-35 空灯箱·白光灯管
  只报告 (配对待确认): R1-22-23, R1-24-25 品红灯管
判定: 判定帧上候选相对 next 的全图 ΔE00 中位平均改善 ≥ 0.5, 且任何一帧变差 ≤ 1.5。只看颜色的评估 (K 预言机) 一并报告, 不参与判定。
入口与验收相同 (accept_t03.scene_job: next, roll=None, grain=0; 候选经 accept_variant 施加)。配准缓存按需建立 (固定种子)。

用法: FILMLAB_DATA=… python3 t07_fresh.py 输出.json
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
JUDGED = ["R1-17-18", "R1-29-30", "R1-34-35"]
REPORT = ["R1-22-23", "R1-24-25"]
CAND = "fixed_exposure,plain,lab_colour"


def full_job(args):
    variant, sid = args
    if variant != "none":
        import accept_variant
        accept_variant.install(variant)
    import accept_t03
    return variant, sid, accept_t03.scene_job(("next", sid))[1]


def k_job(args):
    import k_oracle
    variant, sid = args
    return variant, sid, k_oracle.job((variant, sid))[1]


if __name__ == "__main__":
    import multiprocessing as mp, oracle_timing as ot
    unaligned = []
    for s in JUDGED + REPORT:         # 先在主进程里建好配准缓存, 避免并行写同一文件; 失败的帧记为无法评估 (不替换)
        try:
            ot.geometry(s)
        except RuntimeError as e:
            print("无法评估 (配准失败, 发生在任何 ΔE 计算之前):", e)
            unaligned.append(s)
    JUDGED[:] = [s for s in JUDGED if s not in unaligned]
    REPORT[:] = [s for s in REPORT if s not in unaligned]
    S = JUDGED + REPORT
    jobs = [(v, s) for s in S for v in ("none", CAND)]
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3")), maxtasksperchild=1) as p:
        full = p.map(full_job, jobs)
        kor = p.map(k_job, jobs)
    res = {s: {} for s in S}
    for v, s, r in full:
        res[s][v] = {"full": r["full"], "near": r["near"]}
    for v, s, r in kor:
        res[s][v]["k"] = r["k"]
    for s in S:
        d = res[s][CAND]["full"] - res[s]["none"]["full"]; dk = res[s][CAND]["k"] - res[s]["none"]["k"]
        tag = "判定" if s in JUDGED else "报告"
        print(f"{s:9s} [{tag}] 全图 next {res[s]['none']['full']:6.2f} → 候选 {res[s][CAND]['full']:6.2f} ({d:+.2f})   "
              f"只看颜色 {res[s]['none']['k']:6.2f} → {res[s][CAND]['k']:6.2f} ({dk:+.2f})")
    dj = [res[s][CAND]["full"] - res[s]["none"]["full"] for s in JUDGED]
    ok = np.mean(dj) <= -0.5 and max(dj) <= 1.5
    print(f"\n判定帧平均变化 {np.mean(dj):+.3f}, 最大变差 {max(dj):+.2f} → {'通过' if ok else '不通过'}")
    print(f"只看颜色 (报告) 判定帧平均变化 {np.mean([res[s][CAND]['k'] - res[s]['none']['k'] for s in JUDGED]):+.3f}")
    json.dump({"results": res, "pass": bool(ok), "unaligned": unaligned}, open(sys.argv[1], "w"), indent=1)
