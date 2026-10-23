"""输出调整 (SATURATION_MATRIX / LAB_TINT) 的拆分: 汇总定光预言机结果 (oracle_timing.py 的 json), 按光源组。

O1 = 只给冲印店印放光号 (公平定光); 彩度比 = 模拟 / 真实 (彩度 > 10 的格子, 1 = 正确); ΔL / Δa / Δb = 模拟 − 真实 中位。
用法: python3 cond_summary.py 标签=o_x.json ...
"""
import json, sys
import numpy as np

GROUPS = {
    "夜景": ["R1-00-1", "R1-01-2", "R1-02-3", "R1-04-5", "R1-05-6", "R1-06-7"],
    "黄昏": ["R1-07-8", "R1-03-4"],
    "钨丝灯箱": ["R1-26-27", "R1-27-28", "R1-28-29", "R1-30-31"],
    "白光灯管": ["R1-31-32", "R1-32-33", "R1-33-34", "R1-35-36"],
    "品红灯管": ["R1-21-22", "R1-23-24", "R1-25-26"],
    "TE226 色卡": ["R1-20-21"],
    "咖啡店 (他店)": ["cafe"],
}

if __name__ == "__main__":
    runs = {a.split("=")[0]: json.load(open(a.split("=")[1])) for a in sys.argv[1:]}
    print(f"{'组':12s} " + " | ".join(f"{k:^34s}" for k in runs))
    print(f"{'':12s} " + " | ".join(f"{'自动':>5s} {'O1':>5s} {'O2':>5s} {'彩度比':>5s} {'Δa':>5s} {'Δb':>5s}" for _ in runs))
    for g, ids in list(GROUPS.items()) + [("全部", sum(GROUPS.values(), []))]:
        cells = []
        for r in runs.values():
            s = [i for i in ids if i in r]
            if not s:
                cells.append(" " * 34); continue
            m = lambda f: np.mean([f(r[i]) for i in s])
            cells.append(f"{m(lambda x: x['auto']):5.2f} {m(lambda x: x['O1']['dE']):5.2f} {m(lambda x: x['O2']['dE']):5.2f} "
                         f"{m(lambda x: x['O1']['chroma']):5.2f} {m(lambda x: x['O1']['dLab'][1]):+5.1f} {m(lambda x: x['O1']['dLab'][2]):+5.1f}")
        print(f"{g:12s} " + " | ".join(cells))
