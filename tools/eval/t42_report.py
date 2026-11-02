"""T42 汇总: t42_sat.py 的逐帧结果 → 按色相、彩度分组的彩度比表。
每组先在每帧内取中位 (≥ MIN 个像素 / 格子), 再取各帧中位的中位; 括号内为帧数。
用法: python3 t42_report.py 结果目录 代际 [代际2]"""
import sys, glob, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t42_chain as tc
HUES = [("红", 0), ("黄", 60), ("绿", 120), ("青", 180), ("蓝", 240), ("品红", 300)]
LAB_HUES = [("红", 30), ("黄", 90), ("绿", 150), ("青", 210), ("蓝", 270), ("品红", 330)]   # CIELAB h_ab 扇区中心
CB = [(0.25, 0.5), (0.5, 0.8), (0.8, 3.0)]
LB = [(10, 20), (20, 35), (35, 200)]
STAGES = ["dye", "sm", "scan", "dn", "dp", "out", "out_emp", "real"]
NIGHT = {f"R1-0{i}-{i + 1}" for i in range(8)}
MIN = 100
MODE = os.environ.get("MODE", "_m")      # "_m": 中性 = 本帧白 v0 (T32 的做法); "": 中性 = 3200K 中灰


def frames(d, gen):
    return sorted(glob.glob(f"{d}/*_{gen}.npz"))


def stage_table(d, gen, subset=None):
    rows = {}
    for f in frames(d, gen):
        sid = os.path.basename(f).rsplit("_" + gen, 1)[0]
        if subset and sid not in subset: continue
        z = np.load(f)
        lh = z["logH"].astype(float) - tc.G0 - (z["v0"] if MODE == "_m" else 0.0)
        c0, h = np.ptp(lh, -1), tc.hue(lh)
        for name, h0 in HUES + [("全部", None)]:
            mh = np.ones(len(h), bool) if h0 is None else ((h - h0 + 180) % 360 - 180) ** 2 < 30 ** 2
            for lo, hi in CB + [(0.25, 3.0)]:
                m0 = mh & (c0 >= lo) & (c0 < hi)
                med = {}
                for k in STAGES:
                    r = np.ptp(z["eq_" + k + MODE].astype(float), -1) / c0
                    m = m0 & np.isfinite(r)
                    if m.sum() >= MIN:
                        med[k] = np.median(r[m])
                        rows.setdefault((name, lo, hi, k), []).append(med[k])
                if "real" in med and "out_emp" in med:
                    rows.setdefault((name, lo, hi, "ratio"), []).append(med["out_emp"] / med["real"])
    return rows


def lab_table(d, gen, subset=None):
    rows = {}
    for f in frames(d, gen):
        sid = os.path.basename(f).rsplit("_" + gen, 1)[0]
        if subset and sid not in subset: continue
        z = np.load(f)
        R, M = z["lab_real"].astype(float), z["lab_model"].astype(float)
        cr, cm = np.hypot(R[:, 1], R[:, 2]), np.hypot(M[:, 1], M[:, 2])
        h = np.degrees(np.arctan2(R[:, 2], R[:, 1])) % 360
        for name, h0 in LAB_HUES + [("全部", None)]:
            mh = np.ones(len(h), bool) if h0 is None else ((h - h0 + 180) % 360 - 180) ** 2 < 30 ** 2
            for lo, hi in LB + [(10, 200)]:
                m = mh & (cr >= lo) & (cr < hi)
                if m.sum() >= 10:
                    rows.setdefault((name, lo, hi), []).append(np.median(cm[m] / cr[m]))
    return rows


def fmt(v):
    return f"{np.median(v):.2f} ({len(v)})" if v else "—"


if __name__ == "__main__":
    d, gens = sys.argv[1], sys.argv[2:]
    for sub_name, sub in (("全部 18 帧", None), ("夜景 8 帧", NIGHT), ("室内 10 帧", "indoor")):
        subset = sub if sub != "indoor" else {os.path.basename(f).rsplit("_" + gens[0], 1)[0] for f in frames(d, gens[0])} - NIGHT
        print(f"\n## {sub_name}")
        print("\n### A. Lab C*ab 彩度比 模型/真实 (O1 定光), 按真实 h_ab 色相与真实 C*")
        print("| 色相 | " + " | ".join(f"{g} C* {lo}–{hi}" for g in gens for lo, hi in LB + [(10, 200)]) + " |")
        print("|---|" + "---|" * (len(gens) * 4))
        L = {g: lab_table(d, g, subset) for g in gens}
        for name, _ in LAB_HUES + [("全部", None)]:
            print(f"| {name} | " + " | ".join(fmt(L[g].get((name, lo, hi), [])) for g in gens for lo, hi in LB + [(10, 200)]) + " |")
        for g in gens:
            T = stage_table(d, g, subset)
            print(f"\n### B/C. 等效曝光彩度比 (ptp(e)/ptp(logH−G0)), {g}; 按胶片曝光色相, 彩度 ≥ 0.25 (log10)")
            print("| 色相 | " + " | ".join(STAGES) + " | 模型/真实 (out_emp/real) |")
            print("|---|" + "---|" * (len(STAGES) + 1))
            for name, _ in HUES + [("全部", None)]:
                v = {k: T.get((name, 0.25, 3.0, k), []) for k in STAGES}
                ratio = T.get((name, 0.25, 3.0, "ratio"), [])
                print(f"| {name} | " + " | ".join(fmt(v[k]) for k in STAGES) + f" | {fmt(ratio)} |")
            print(f"\n按彩度分档 (out_emp / real / 模型逐级 dye→out), {g}")
            for name, _ in HUES:
                for lo, hi in CB:
                    v = {k: T.get((name, lo, hi, k), []) for k in STAGES}
                    if not v["real"]: continue
                    print(f"  {name} {lo:.2f}–{hi:.2f}: 真实 {fmt(v['real'])}  模型 {fmt(v['out_emp'])}  模型/真实 {fmt(T.get((name, lo, hi, 'ratio'), []))} | 逐级 " +
                          " ".join(f"{k} {np.median(v[k]):.2f}" if v[k] else f"{k} —" for k in STAGES[:6]))
