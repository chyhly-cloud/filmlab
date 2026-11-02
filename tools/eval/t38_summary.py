"""T38 汇总: HDRI 假想 RAW 的观测量 (t38_hdri_obs.py) 与 flux 的相关、条件分位表、留出集检验, 以及仓库测量表。
用法: python3 -I t38_summary.py 测量目录 [仓库表输出路径] [条件变量 (默认自动选 bg)]"""
import csv, glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr, kstest

MEAS = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
Q = np.linspace(0, 1, 21)
SZB = (60.0, 150.0)
SZN = ("0–60", "60–150", "≥150")
CLS = ("lamp", "line", "large")
HOLD = ("street_lamp", "hansaplatz", "shanghai_bund", "zwinger_night")
OBS = ("bg", "skirt", "slope", "fmean", "compact", "r_um", "base_rel", "nsat3", "col_rg", "col_bg")
NBIN = 5
MIN_N = 60

rows = []
for p in sorted(glob.glob(os.path.join(MEAS, "*.tsv"))):
    if "_16k" in p:
        continue
    rows += list(csv.DictReader(open(p), delimiter="\t"))
for x in rows:
    for k in ("flux", "peak") + OBS:
        x[k] = float(x[k])
    x["k"] = int(np.searchsorted(SZB, x["r_um"], side="right"))
print(f"HDRI {len(set(x['hdri'] for x in rows))} 张, 块 {len(rows)} (35 / 50 mm 合并)")

# 1. 相关
def rho(sel, o):
    a = np.array([x[o] for x in sel]); f = np.array([x["flux"] for x in sel]); m = np.isfinite(a)
    return (spearmanr(a[m], f[m])[0], int(m.sum())) if m.sum() > 20 else (np.nan, int(m.sum()))
print("\n#### Spearman ρ(观测量, flux)")
groups = [("全部", rows)] + [(c, [x for x in rows if x["cls"] == c]) for c in CLS] + \
         [(f"lamp {SZN[k]}", [x for x in rows if x["cls"] == "lamp" and x["k"] == k]) for k in range(3)]
print("| 组 | 块数 | " + " | ".join(OBS) + " |"); print("|---|---|" + "---|" * len(OBS))
for g, sel in groups:
    print(f"| {g} | {len(sel)} | " + " | ".join(f"{rho(sel, o)[0]:+.2f}" for o in OBS) + " |")
print("\n每张 HDRI 内 (灯具类) 的 ρ(bg, flux) / ρ(skirt, flux):")
for h in sorted(set(x["hdri"] for x in rows)):
    sel = [x for x in rows if x["hdri"] == h and x["cls"] == "lamp"]
    print(f"  {h:28s} n={len(sel):5d}  bg {rho(sel, 'bg')[0]:+.2f}  skirt {rho(sel, 'skirt')[0]:+.2f}  slope {rho(sel, 'slope')[0]:+.2f}")

VAR = sys.argv[3] if len(sys.argv) > 3 else "bg"


def build(train, var):
    """条件分位表: 观测量按训练集全部块的分位等分为 NBIN 档 (边界), 每 类 × 尺寸档 × 观测档 21 分位; 样本 < MIN_N 并入相邻观测档。"""
    v = np.array([x[var] for x in train]); v = v[np.isfinite(v)]
    edges = [float(e) for e in np.percentile(v, np.linspace(0, 100, NBIN + 1)[1:-1])]
    tab, allt = {}, {}
    for c in CLS:
        tab[c], allt[c] = [], []
        for k in range(3):
            sel = [x for x in train if x["cls"] == c and (x["k"] == k)]
            kk = k
            while len(sel) < 100 and kk > 0:
                kk -= 1; sel = [x for x in train if x["cls"] == c and x["k"] >= kk]
            if len(sel) < 100:
                sel = [x for x in train if x["cls"] == c]
            allt[c].append([round(float(q), 3) for q in np.percentile([x["flux"] for x in sel], Q * 100)])
            row = []
            for b in range(NBIN):
                lo, hi = b, b
                pick = lambda lo, hi: [x["flux"] for x in sel if np.isfinite(x[var]) and lo <= int(np.searchsorted(edges, x[var], side="right")) <= hi]
                s = pick(lo, hi)
                while len(s) < MIN_N and (lo > 0 or hi < NBIN - 1):
                    lo, hi = max(lo - 1, 0), min(hi + 1, NBIN - 1); s = pick(lo, hi)
                row.append([round(float(q), 3) for q in np.percentile(s if s else [x["flux"] for x in sel], Q * 100)])
            tab[c].append(row)
    return {"var": var, "edges": [round(e, 3) for e in edges], "flux": tab, "flux_all": allt}


def pred_q(T, x, cond=True):
    if cond and np.isfinite(x[T["var"]]):
        return np.array(T["flux"][x["cls"]][x["k"]][int(np.searchsorted(T["edges"], x[T["var"]], side="right"))])
    return np.array(T["flux_all"][x["cls"]][x["k"]])


def evaluate(T, test, label):
    out = {}
    for cond in (False, True):
        pit = np.array([np.interp(x["flux"], pred_q(T, x, cond), Q) for x in test])
        med = np.array([pred_q(T, x, cond)[10] for x in test]); f = np.array([x["flux"] for x in test])
        out[cond] = dict(ks=kstest(pit, "uniform").statistic, mae=float(np.median(np.abs(med - f))), rho=spearmanr(med, f)[0],
                         bias=float(np.median(f) - np.median(med)))
    print(f"| {label} | {len(test)} | " + " | ".join(f"{out[c]['mae']:.2f} / {out[c]['rho']:+.2f} / {out[c]['ks']:.3f}" for c in (False, True)) + " |")
    return out


print("\n#### 留出集检验 (训练 20 张, 留出 " + ", ".join(HOLD) + ")")
print("| 条件变量 | 块 | 无条件: 中位绝对误差 / ρ(预测中位, flux) / PIT 的 KS | 有条件: 同左 |")
print("|---|---|---|---|")
train = [x for x in rows if x["hdri"] not in HOLD]; test = [x for x in rows if x["hdri"] in HOLD]
for var in ("bg", "skirt", "slope", "fmean", "compact"):
    evaluate(build(train, var), test, var)
T = build(train, VAR)
print(f"\n条件 {VAR}: 留出集分组 (每张 / 类):")
print("| 组 | 块 | 无条件 | 有条件 |"); print("|---|---|---|---|")
for h in HOLD:
    evaluate(T, [x for x in test if x["hdri"] == h], h)
for c in CLS:
    s = [x for x in test if x["cls"] == c]
    if len(s) > 20:
        evaluate(T, s, c)
# 留出集: 每观测档的实测中位 vs 预测中位
print(f"\n留出集 灯具类: {VAR} 档 × 尺寸档 的 实测 flux 中位 [四分位] / 预测中位 (块)")
for k in range(3):
    cells = []
    for b in range(NBIN):
        s = [x for x in test if x["cls"] == "lamp" and x["k"] == k and np.isfinite(x[VAR]) and int(np.searchsorted(T["edges"], x[VAR], side="right")) == b]
        if s:
            f = [x["flux"] for x in s]
            cells.append(f"{np.median(f):.2f} [{np.percentile(f, 25):.2f}, {np.percentile(f, 75):.2f}] / {T['flux']['lamp'][k][b][10]:.2f} ({len(s)})")
        else:
            cells.append("—")
    print(f"  {SZN[k]}: " + " | ".join(cells))

# 2. 最终表 (24 张全部)
F = build(rows, VAR)
print(f"\n#### 全部 24 张: 条件 {VAR} 的档边界 {F['edges']}")
for c in CLS:
    for k in range(3):
        print(f"  {c} {SZN[k]}: " + " | ".join(f"中位 {t[10]:.2f} [{t[5]:.2f}, {t[15]:.2f}]" for t in F["flux"][c][k]) + f"   无条件中位 {F['flux_all'][c][k][10]:.2f}")
json.dump(F, open(os.path.join(MEAS, f"cond_{VAR}.json"), "w"))
print("MEASURED_COND =", json.dumps(F))

if OUT:
    keep = ["hdri", "focal", "yaw", "pitch", "S_over_P90", "blob", "area", "cls", "r_um", "base_rel", "flux", "peak",
            "bg", "skirt", "slope", "fmean", "compact", "nsat3", "col_rg", "col_bg", "edge"]
    with open(OUT, "w") as fo:
        fo.write("# T38 夜景 HDRI 假想 RAW 的截断块 (Poly Haven CC0 官方 8k; 只存测量)。1600×1067 像素 (22.5 µm), 35/50 mm, 偏航/俯仰 (度);"
                 " 截断 S = 帧内 P90 × 2^3.71 (按灯具块 base 中位修正, S_over_P90 = S / P90); flux/peak/bg/skirt/slope 为档;"
                 " 观测量由 filmlab.rawio._block_obs 计算。URL = https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/8k/{hdri}_8k.hdr。"
                 "生成: tools/eval/t38_hdri_obs.py + t38_summary.py\n")
        fo.write("\t".join(keep) + "\n")
        for x in rows:
            fo.write("\t".join(f"{x[k]:.3f}" if isinstance(x[k], float) else str(x[k]) for k in keep) + "\n")
    print("wrote", OUT)

# 3. 帧内相关: 条件分布下每块的 PIT u = F(flux), 同一假想照片 (HDRI × 焦距 × 方向) 内是否相关 (组内相关 ICC)
print(f"\n#### 条件 {VAR} 下 PIT 的帧内相关 (同一假想照片、同类同尺寸档的块; ICC = 帧间方差 / 总方差)")
from collections import defaultdict
for c, k in (("lamp", 0), ("lamp", 1), ("lamp", 2), ("line", 0)):
    g = defaultdict(list)
    for x in rows:
        if x["cls"] == c and x["k"] == k:
            g[(x["hdri"], x["focal"], x["yaw"], x["pitch"])].append(float(np.interp(x["flux"], pred_q(F, x), Q)))
    g = {a: v for a, v in g.items() if len(v) >= 3}
    allv = np.concatenate([v for v in g.values()]); tot = allv.var()
    means = np.array([np.mean(v) for v in g.values()]); ns = np.array([len(v) for v in g.values()])
    within = np.sum([np.var(v) * len(v) for v in g.values()]) / ns.sum()
    icc = max(0.0, 1 - within / tot)
    # 同一张 HDRI 内 (跨方向)
    gh = defaultdict(list)
    for (h, *_), v in g.items():
        gh[h] += v
    wh = np.sum([np.var(v) * len(v) for v in gh.values()]) / ns.sum()
    print(f"  {c} {SZN[k]}: 帧 {len(g)}, 块 {ns.sum()}, PIT 方差 {tot:.3f} (均匀 = 0.083); 帧内 ICC {icc:.2f}; HDRI 内 ICC {max(0, 1 - wh / tot):.2f}")
