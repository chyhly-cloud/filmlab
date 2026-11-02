"""T28: 形状 × 亮度组合的汇总 —— 光晕部分 (除碳 − 带碳, 灯芯 60–150 µm, 按灯, 全部 / 去掉 city_61360) 的绝对值与形状对真实之比,
小灯检查 (< 40 / 40–60 / 0–60 µm, 去掉 city_61360), 环的可见度 (每盏暗背景灯 150–300 µm 环 R ÷ 700–1000 µm 尾 R)。
数据: t27_size_halo.py 的逐灯 JSON (环境变量 JSON=…)。
用法: python3 t28_summary.py 真实800T.json 真实对照.json 标签=模型.json …   (模型带碳 JSON 取同名 _remjet.json)"""
import json, sys, numpy as np
BINS = [25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500]
I = list(range(2, 10))            # 100..1000
EX = "city_61360"
def load(fn):
    return [(f, r, np.array(v)) for f, r, v in json.load(open(fn))["lamps"]]
def sel(L, lo, hi, ex=False):
    return [v for f, r, v in L if lo <= r < hi and not (ex and EX in f)]
def medR(V):
    with np.errstate(all="ignore"):
        return np.nanmedian(np.array([v[:, 0] for v in V]), 0) if V else np.full(len(BINS), np.nan)
REF, CTL = load(sys.argv[1]), load(sys.argv[2])
variants = [a.split("=", 1) for a in sys.argv[3:]]          # 标签=模型 JSON
M = {lab: (load(p), load(p[:-5] + "_remjet.json")) for lab, p in variants}
f3 = lambda a: " ".join(f"{x:6.3f}" for x in a)
f2 = lambda a: " ".join(f"{x:6.2f}" for x in a)
out = {}
print("距灯芯边缘 µm          " + " ".join(f"{BINS[i]:6d}" for i in I))
for ex in (False, True):
    print(f"\n### 光晕部分 (800T − 带碳) R, 灯芯 60–150 µm, 按灯{'，去掉 city_61360' if ex else '（全部）'}")
    real = (medR(sel(REF, 60, 150)) - medR(sel(CTL, 60, 150)))[I]
    rn = real / np.nanmax(real)
    print(f"真实                  {f3(real)}   700/峰 {real[6]/real.max():.2f}")
    print(f"真实 归一             {f2(rn)}")
    for lab, (m8, mc) in M.items():
        mod = (medR(sel(m8, 60, 150, ex)) - medR(sel(mc, 60, 150, ex)))[I]
        mn = mod / np.nanmax(mod)
        rms = np.sqrt(np.mean((mn - rn) ** 2)); rmsa = np.sqrt(np.mean((mod - real) ** 2))
        print(f"{lab:22s}{f3(mod)}   700/峰 {mod[6]/mod.max():.2f}  (灯 {len(sel(m8,60,150,ex))}/{len(sel(mc,60,150,ex))})")
        print(f"{'  绝对/真实':20s}{f2(mod/real)}   强度 RMS {rmsa:.3f}")
        print(f"{'  形状/真实':20s}{f2(mn/rn)}   形状 RMS {rms:.3f}")
        out.setdefault(lab, {})[f"halo60_{'ex' if ex else 'all'}"] = dict(abs=mod.tolist(), ratio=(mod/real).tolist(), shape_ratio=(mn/rn).tolist(), shape_rms=float(rms), abs_rms=float(rmsa))
print("\n### 小灯检查 (去掉 city_61360): 800T 成片超出背景的 R @150 / 200 / 300 µm;  光晕部分 @150/200/300")
J = [3, 4, 5]
for (lo, hi) in ((0, 40), (40, 60), (0, 60)):
    r8 = medR(sel(REF, lo, hi)); rc = medR(sel(CTL, lo, hi))
    print(f"[{lo}-{hi} µm] 真实 800T (灯 {len(sel(REF,lo,hi))})  R {f3(r8[J])}   光晕部分 {f3((r8-rc)[J])}")
    for lab, (m8, mc) in M.items():
        a8, ac = medR(sel(m8, lo, hi, True)), medR(sel(mc, lo, hi, True))
        print(f"   {lab:20s} (灯 {len(sel(m8,lo,hi,True)):4d})  R {f3(a8[J])}  /真实 {f2(a8[J]/r8[J])}   光晕部分 {f3((a8-ac)[J])}  /真实 {f2((a8-ac)[J]/(r8-rc)[J])}")
        out[lab][f"small_{lo}_{hi}"] = dict(R=a8[J].tolist(), halo=(a8 - ac)[J].tolist(), n=len(sel(m8, lo, hi, True)))
    out.setdefault("real", {})[f"small_{lo}_{hi}"] = dict(R=r8[J].tolist(), halo=(r8 - rc)[J].tolist())
print("\n### 环的可见度: 每盏暗背景灯 环(150–300 µm, 两档中位的平均) R ÷ 尾(700–1000 µm) R; 尾不足 ε=0.003 时按 ε 计")
print("    列: 灯数, 比值中位 [四分位], 中位环 R, 中位尾 R, 尾≤ε 的比例, 环 G/R 中位")
def vis(V):
    A = np.array([v for v in V])
    ring = np.nanmean(A[:, 4:6, 0], 1); tail = A[:, 9, 0]
    ringG = np.nanmean(A[:, 4:6, 1], 1)
    ok = np.isfinite(ring) & np.isfinite(tail)
    ring, tail, ringG = ring[ok], tail[ok], ringG[ok]
    rat = ring / np.maximum(tail, 0.003)
    return len(ring), np.median(rat), np.percentile(rat, 25), np.percentile(rat, 75), np.median(ring), np.median(tail), np.mean(tail <= 0.003), np.median(ringG / ring)
for (lo, hi) in ((0, 60), (60, 150), (0, 150)):
    n, m, q1, q3, rg, tl, fz, gr = vis(sel(REF, lo, hi))
    print(f"[{lo}-{hi} µm] 真实 800T        n={n:5d}  {m:6.1f} [{q1:5.1f}–{q3:6.1f}]  环 {rg:.3f}  尾 {tl:.4f}  中位比 {rg/tl:5.1f}  尾≤ε {fz:.0%}  G/R {gr:.2f}")
    out["real"][f"vis_{lo}_{hi}"] = [float(m), float(q1), float(q3), float(rg), float(tl), float(gr)]
    for lab, (m8, mc) in M.items():
        n, m, q1, q3, rg, tl, fz, gr = vis(sel(m8, lo, hi, True))
        print(f"   {lab:22s} n={n:5d}  {m:6.1f} [{q1:5.1f}–{q3:6.1f}]  环 {rg:.3f}  尾 {tl:.4f}  中位比 {rg/tl:5.1f}  尾≤ε {fz:.0%}  G/R {gr:.2f}")
        out[lab][f"vis_{lo}_{hi}"] = [float(m), float(q1), float(q3), float(rg), float(tl), float(gr)]
import os
if os.environ.get("OUT"):
    json.dump(out, open(os.environ["OUT"], "w"))
