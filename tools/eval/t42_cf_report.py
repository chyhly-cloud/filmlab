"""T42: t42_cf.py 的结果汇总。每组取各帧中位的中位 [帧数]。用法: python3 t42_cf_report.py cf.json [变体…]"""
import json, sys
import numpy as np
HUES = ["红", "黄", "绿", "青", "蓝", "品红", "全部"]
NIGHT = {f"R1-0{i}-{i + 1}" for i in range(8)}
d = json.load(open(sys.argv[1]))
names = sys.argv[2:] or list(d)
med = lambda v: f"{np.median(v):.2f}" if v else "—"
for sub, keep in (("全部帧", lambda s: True), ("夜景", lambda s: s in NIGHT), ("室内", lambda s: s not in NIGHT)):
    print(f"\n## {sub}")
    print("\n### 逐级等效曝光彩度比 (彩度 ≥ 0.25), 色相=全部; 以及 模型/真实 (out_emp/real) 按色相")
    print("| 变体 | dye | sm | scan | dn | out | out_emp | 模型/真实: " + " ".join(HUES) + " |")
    print("|---|---|---|---|---|---|---|---|")
    for n in names:
        fr = {s: r for s, r in d[n].items() if keep(s)}
        st = lambda k, h="全部": [r["stage"][f"{h}|0.25|3.0"][k] for r in fr.values() if f"{h}|0.25|3.0" in r["stage"] and k in r["stage"][f"{h}|0.25|3.0"]]
        rat = lambda h: [r["stage"][f"{h}|0.25|3.0"]["out_emp"] / r["stage"][f"{h}|0.25|3.0"]["real"] for r in fr.values()
                         if f"{h}|0.25|3.0" in r["stage"] and {"out_emp", "real"} <= set(r["stage"][f"{h}|0.25|3.0"])]
        print(f"| {n} | " + " | ".join(med(st(k)) for k in ("dye", "sm", "scan", "dn", "out", "out_emp")) + " | " +
              " ".join(f"{med(rat(h))}[{len(rat(h))}]" for h in HUES) + " |")
    print("\n### Lab C* 模型/真实 (O1, 胶片曝光彩度 ≥ 0.1, 按胶片曝光色相) 与格子 ΔE00 中位")
    print("| 变体 | " + " | ".join(HUES) + " | ΔE00 |")
    print("|---|" + "---|" * (len(HUES) + 1))
    for n in names:
        fr = {s: r for s, r in d[n].items() if keep(s)}
        lab = lambda h: [r["lab"][h][0] for r in fr.values() if h in r["lab"]]
        print(f"| {n} | " + " | ".join(f"{med(lab(h))} [{len(lab(h))}]" for h in HUES) + f" | {med([r['lab_dE'] for r in fr.values()])} |")
    print("\n### 输出中性传递斜率 (log 输出 / logH, 本帧中性像素) R G B: 模型 / 真实")
    for n in names:
        fr = {s: r for s, r in d[n].items() if keep(s)}
        sm = np.array([r["slope_model"] for r in fr.values()]); sr = np.array([r["slope_real"] for r in fr.values()])
        q = sm / sr
        print(f"  {n}: 模型 {np.round(np.nanmedian(sm, 0), 3)}  真实 {np.round(np.nanmedian(sr, 0), 3)}  模型/真实 {np.round(np.nanmedian(q, 0), 3)}")
