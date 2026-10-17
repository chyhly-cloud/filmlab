"""误差分类: 按场景光源类型 / 引擎判定的主光源 / 曝光 (真实 L*) / 色相区, 统计 Δ 与跨场景一致性。"""
import sys, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
d = np.load(sys.argv[1]); S, R, Wt, sc, cat = d["S"], d["R"], d["W"], d["scene"], d["cat"]
CL = ["日光", "钨丝", "钠灯", "荧光", "LED", "汞/金卤"]
ok = (R[:, 0] > 5) & (S[:, 0] > 2) & np.all(np.isfinite(Wt), 1)
S, R, Wt, sc, cat = S[ok], R[ok], Wt[ok], sc[ok], cat[ok]
D = S - R; de = colour.delta_E(S, R, method="CIE 2000")
cr, cs = np.hypot(R[:, 1], R[:, 2]), np.hypot(S[:, 1], S[:, 2])
hm = np.degrees(np.arctan2(R[:, 2] + S[:, 2], R[:, 1] + S[:, 1])) % 360
chrom = (cr + cs) / 2 > 12
def row(name, m, scenes_min=3):
    if m.sum() < 30: return
    sc_ids = [s for s in np.unique(sc[m]) if (m & (sc == s)).sum() >= 20]
    per = np.array([np.median(D[m & (sc == s)], 0) for s in sc_ids]) if sc_ids else np.zeros((0, 3))
    agree = lambda k: (np.mean(np.sign(per[:, k]) == np.sign(np.median(D[m, k]))) if len(per) else np.nan)
    ratio = np.median(cs[m]) / max(np.median(cr[m]), 1e-3)
    flag = " ←系统性" if len(per) >= scenes_min and any(agree(k) >= 0.8 and abs(np.median(D[m, k])) > 3 for k in range(3)) else ""
    print(f"  {name:22s} n={m.sum():5d} 场景 {len(sc_ids):2d}  ΔE00 {np.median(de[m]):5.1f}  ΔL {np.median(D[m,0]):+5.1f} ({agree(0):.0%})  Δa {np.median(D[m,1]):+5.1f} ({agree(1):.0%})  Δb {np.median(D[m,2]):+5.1f} ({agree(2):.0%})  彩度比 {ratio:4.2f}{flag}")
print("== 全部"); row("全部", np.ones(len(S), bool))
print("== 按场景光源类型 (清单)")
for c in np.unique(cat): row(c, cat == c, 2)
print("== 按引擎判定的主光源 (格子内权重最大的一类)")
dom = np.argmax(Wt, 1)
for k, n in enumerate(CL): row(f"{n} (权重>0.4)", (dom == k) & (Wt[:, k] > 0.4))
print("== 按曝光 (真实 L*)")
for lo, hi in [(5, 20), (20, 40), (40, 60), (60, 80), (80, 101)]: row(f"L* {lo}–{hi}", (R[:, 0] >= lo) & (R[:, 0] < hi))
print("== 按色相区 (彩度 > 12)")
for n, a, b in [("红/品红", 330, 30), ("橙/肤", 30, 75), ("黄", 75, 110), ("绿/黄绿", 110, 180), ("青", 180, 240), ("蓝", 240, 300), ("紫", 300, 330)]:
    row(n, chrom & (((hm >= a) & (hm < b)) if a < b else ((hm >= a) | (hm < b))))
print("== 中性 (真实彩度 < 6) 按亮度")
for lo, hi in [(5, 30), (30, 60), (60, 101)]: row(f"中性 L* {lo}–{hi}", (cr < 6) & (R[:, 0] >= lo) & (R[:, 0] < hi))
