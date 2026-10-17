"""消融汇总: RAW 配对按场景类型 (日光 / 夜景 / 钨丝灯箱) 的 ΔE00、彩度比、Δb, 以及 ref3 A/B 与 C。"""
import sys, glob, os, re, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
D = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/"
RUNS = {"base (现行)": ("expo/taxo_cur.npz", None), "WB65": ("expo/taxo_wb65.npz", ("expo/score_wb65.txt", "expo/C_wb65.txt"))}
for f in sorted(glob.glob(D + "abl/taxo_*.npz")):
    t = os.path.basename(f)[5:-4]
    RUNS[t] = (f"abl/taxo_{t}.npz", (f"abl/score_{t}.txt", f"abl/C_{t}.txt"))
GROUPS = {"日光(咖啡店)": ["cafe"], "夜景(8)": ["R1-07-8", "R1-03-4", "R1-01-2", "R1-00-1", "R1-02-3", "R1-04-5", "R1-05-6", "R1-06-7"],
          "钨丝灯箱(3)": ["R1-27-28", "R1-28-29", "R1-30-31"]}
def stats(p):
    d = np.load(D + p); S, R, sc = d["S"], d["R"], d["scene"]
    ok = (R[:, 0] > 5) & (S[:, 0] > 2)
    out = []
    for g, ids in GROUPS.items():
        # 每个场景等权: 场景内中位, 再取场景平均
        per = []
        for s in ids:
            m = ok & (sc == s)
            if m.sum() < 50: continue
            cr, cs = np.hypot(*R[m, 1:].T), np.hypot(*S[m, 1:].T); ch = (cr + cs) / 2 > 10
            per.append((np.median(colour.delta_E(S[m], R[m], method="CIE 2000")), np.median(cs[ch]) / np.median(cr[ch]), np.median(S[m, 2] - R[m, 2])))
        per = np.array(per); out.append(f"{g} ΔE {per[:,0].mean():5.2f} 彩度 {per[:,1].mean():4.2f} Δb {per[:,2].mean():+5.1f}")
    allde = np.mean([np.median(colour.delta_E(S[ok & (sc == s)], R[ok & (sc == s)], method="CIE 2000")) for s in np.unique(sc)])
    return f"RAW 12 场景平均 ΔE {allde:5.2f} | " + " | ".join(out)
def ref3(q):
    if q is None: return "A 20.82 B 20.78 C 18.50"
    try:
        s = open(D + q[0]).read(); c = open(D + q[1]).read()
        a = re.search(r"A\(调参\)\s+([\d.]+)", s).group(1); b = re.search(r"B\(验证\)\s+([\d.]+)", s).group(1); cc = re.search(r"模拟 ([\d.]+)", c).group(1)
        return f"A {a} B {b} C {cc}"
    except Exception:
        return "(未完成)"
for k, (p, q) in RUNS.items():
    if os.path.exists(D + p): print(f"{k:22s} {stats(p)} || {ref3(q)}")
