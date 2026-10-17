"""pair_stage_trace.py 输出的逐阶段统计。用法: pair_stage_stats.py cells.npz"""
import numpy as np, sys
d = np.load(sys.argv[1]); n = d["name"]
LR, LS = d["labR"], d["labS"]
cr, cs = np.hypot(*LR[:, 1:].T), np.hypot(*LS[:, 1:].T)
hm = np.degrees(np.arctan2(LR[:, 2] + LS[:, 2], LR[:, 1] + LS[:, 1])) % 360
ok = (LR[:, 0] > 8) & (LS[:, 0] > 3)
neutral = ok & (cr < 5)
def sep(x): return np.stack([x[:, 0] - x[:, 1], x[:, 2] - x[:, 1]], 1)
V = {}
for k in ["scene", "logH", "dye", "Dn", "DnM", "Dp", "pre", "post", "out", "real"]:
    v = sep(d[k]).copy()
    for i in range(9):
        v[n == i] -= np.median(v[(n == i) & neutral], 0)
    V[k] = v
s = V["scene"]
secs = [("红", 330, 30), ("橙/肤", 30, 75), ("黄", 75, 110), ("绿", 110, 180), ("青", 180, 240), ("蓝", 240, 300)]
print("色相    n    " + " ".join(f"{k:>6s}" for k in V if k != "scene") + "   (k = 该阶段分离 / 场景分离, 投影中位数)")
for nm, a, b in secs:
    m = ok & ((cr + cs) / 2 > 12) & (((hm >= a) & (hm < b)) if a < b else ((hm >= a) | (hm < b)))
    m &= np.hypot(*s.T) > 0.05
    row = [np.median((V[k][m] * s[m]).sum(1) / (s[m] ** 2).sum(1)) for k in V if k != "scene"]
    print(f"{nm:6s}{m.sum():5d}  " + " ".join(f"{r:6.2f}" for r in row) + f"   场景|分离| {np.median(np.hypot(*s[m].T)):.2f}")
print("\nLab: 色相区  ΔL  Δa  Δb  彩度 模拟/真实   真实L中位")
for nm, a, b in secs + [("中性", 0, 360)]:
    if nm == "中性": m = ok & neutral
    else: m = ok & ((cr + cs) / 2 > 12) & (((hm >= a) & (hm < b)) if a < b else ((hm >= a) | (hm < b)))
    dd = np.median(LS[m] - LR[m], 0)
    print(f"{nm:6s}{m.sum():5d} {dd[0]:+5.1f} {dd[1]:+5.1f} {dd[2]:+5.1f}   {np.median(cs[m]):5.1f}/{np.median(cr[m]):5.1f}   L {np.median(LR[m,0]):.0f}")
print("\n按场景 中性格子 Δa Δb, 蓝格子 Δa Δb")
for i in range(9):
    mn = ok & neutral & (n == i); mb = ok & (n == i) & ((cr + cs) / 2 > 12) & (hm >= 240) & (hm < 300)
    a = np.median(LS[mn] - LR[mn], 0); b = np.median(LS[mb] - LR[mb], 0) if mb.sum() > 10 else [np.nan]*3
    print(i, mn.sum(), np.round(a, 1), mb.sum(), np.round(b, 1), "蓝彩度", np.round([np.median(cs[mb]), np.median(cr[mb])], 1) if mb.sum() > 10 else "")
