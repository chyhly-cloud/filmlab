"""T39 汇总: HDRI 假想 RAW (相机通道截断, t39_hdri_obs.py) 的观测量与 flux 的相关、nsat3 的相机矩阵敏感性、
二维条件分位表 (bg × nsat3 × 类 × 尺寸档)、留出集检验 (对照无条件 / 只 bg / 只 nsat3)、帧内相关、相近配对, 以及仓库测量表。
用法: python3 -I t39_summary.py 测量目录 [仓库表输出路径 [条件表 json 输出路径]]"""
import csv, glob, json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, kstest

MEAS = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
OUTJ = sys.argv[3] if len(sys.argv) > 3 else None
Q = np.linspace(0, 1, 21)
SZB = (60.0, 150.0)
SZN = ("0–60", "60–150", "≥150")
CLS = ("lamp", "line", "large")
HOLD = ("street_lamp", "hansaplatz", "shanghai_bund", "zwinger_night")
ALT = ("sony_n33920", "fuji_n30464")
OBS = ("bg", "skirt", "slope", "fmean", "compact", "nsat3", "nch", "base_rel", "col_rg", "col_bg")
NBG = 5
NS_EDGES = [float(v) for v in os.environ.get("NS", "0.05,0.5").split(",")]   # nsat3 档边界 (默认 < 0.05 / 0.05–0.5 / ≥ 0.5; 环境变量 NS 可改)
MIN_N = 60

rows = []
for p in sorted(glob.glob(os.path.join(MEAS, "*.tsv"))):
    rows += list(csv.DictReader(open(p), delimiter="\t"))
for x in rows:
    for k in list(x):
        if k not in ("hdri", "cls"):
            x[k] = float(x[k])
    x["k"] = int(np.searchsorted(SZB, x["r_um"], side="right"))
print(f"HDRI {len(set(x['hdri'] for x in rows))} 张, 块 {len(rows)} (35 / 50 mm 合并)")

def rho(sel, o):
    a = np.array([x[o] for x in sel]); f = np.array([x["flux"] for x in sel]); m = np.isfinite(a)
    return spearmanr(a[m], f[m])[0] if m.sum() > 20 else np.nan

print("\n#### Spearman ρ(观测量, flux)")
groups = [("全部", rows)] + [(c, [x for x in rows if x["cls"] == c]) for c in CLS] + \
         [(f"灯具 {SZN[k]}", [x for x in rows if x["cls"] == "lamp" and x["k"] == k]) for k in range(3)]
print("| 组 | 块数 | " + " | ".join(OBS) + " |"); print("|---|---|" + "---|" * len(OBS))
for g, sel in groups:
    print(f"| {g} | {len(sel)} | " + " | ".join(f"{rho(sel, o):+.2f}" for o in OBS) + " |")
print("\n每张 HDRI 内 (灯具类) ρ(bg) / ρ(nsat3) / ρ(nch):")
for h in sorted(set(x["hdri"] for x in rows)):
    sel = [x for x in rows if x["hdri"] == h and x["cls"] == "lamp"]
    print(f"  {h:28s} n={len(sel):5d}  bg {rho(sel, 'bg'):+.2f}  nsat3 {rho(sel, 'nsat3'):+.2f}  nch {rho(sel, 'nch'):+.2f}")

# nsat3 分布与相机矩阵敏感性
nsb = lambda v: int(np.searchsorted(NS_EDGES, v, side="right"))
NNS = len(NS_EDGES) + 1
print(f"\n#### nsat3 分档 (边界 {NS_EDGES}) 的块占比与相机矩阵敏感性 (同一批块)")
for c, k in (("lamp", 0), ("lamp", 1), ("lamp", 2), ("line", None), ("large", None)):
    sel = [x for x in rows if x["cls"] == c and (k is None or x["k"] == k)]
    occ = lambda key: np.bincount([nsb(x[key]) for x in sel], minlength=NNS) / len(sel)
    line = f"  {c} {'' if k is None else SZN[k]}: 主 (Nikon) {np.round(occ('nsat3'), 2)}"
    for a in ALT:
        mig = np.mean([nsb(x["nsat3"]) != nsb(x[f"nsat3_{a}"]) for x in sel])
        line += f" | {a} {np.round(occ(f'nsat3_{a}'), 2)} 换档 {mig:.1%} ρ(nsat3) {rho(sel, f'nsat3_{a}'):+.2f}"
    print(line)
print("各 nsat3 档的灯具 flux 中位 [四分位] (尺寸档 0–60 / 60–150 / ≥150):")
for j in range(NNS):
    print(f"  nsat3 档 {j}: " + " | ".join(
        (lambda f: f"{np.median(f):.2f} [{np.percentile(f, 25):.2f}, {np.percentile(f, 75):.2f}] ({len(f)})")(
            [x["flux"] for x in rows if x["cls"] == "lamp" and x["k"] == k and nsb(x["nsat3"]) == j] or [np.nan]) for k in range(3)))


def size_sel(train, c, k):
    sel = [x for x in train if x["cls"] == c and x["k"] == k]
    kk = k
    while len(sel) < 100 and kk > 0:
        kk -= 1; sel = [x for x in train if x["cls"] == c and x["k"] >= kk]
    if len(sel) < 100:
        sel = [x for x in train if x["cls"] == c]
    return sel


def qt(v):
    return [round(float(q), 3) for q in np.percentile(v, Q * 100)]


def build(train, use_bg=True, use_ns=True):
    """条件表: bg 按训练集分位 NBG 档 (use_bg), nsat3 按 NS_EDGES 3 档 (use_ns)。每格 < MIN_N 块时先合并相邻 bg 档, 再合并相邻 nsat3 档。"""
    v = np.array([x["bg"] for x in train]); v = v[np.isfinite(v)]
    eb = [float(e) for e in np.percentile(v, np.linspace(0, 100, NBG + 1)[1:-1])] if use_bg else []
    en = NS_EDGES if use_ns else []
    nb, nn = len(eb) + 1, len(en) + 1
    T = {"vars": ["bg", "nsat3"], "edges": {"bg": [round(e, 3) for e in eb], "nsat3": en}, "flux": {}, "flux_ns": {}, "flux_all": {}, "merged": []}
    for c in CLS:
        T["flux"][c], T["flux_ns"][c], T["flux_all"][c] = [], [], []
        for k in range(3):
            sel = size_sel(train, c, k)
            T["flux_all"][c].append(qt([x["flux"] for x in sel]))
            bi = lambda x: int(np.searchsorted(eb, x["bg"], side="right")) if np.isfinite(x["bg"]) else -1
            ni = lambda x: int(np.searchsorted(en, x["nsat3"], side="right"))
            T["flux_ns"][c].append([qt([x["flux"] for x in sel if ni(x) == j] or [x["flux"] for x in sel]) for j in range(nn)])
            grid = []
            for b in range(nb):
                rowb = []
                for j in range(nn):
                    blo, bhi, jlo, jhi = b, b, j, j
                    pick = lambda: [x["flux"] for x in sel if blo <= bi(x) <= bhi and jlo <= ni(x) <= jhi]
                    s = pick()
                    while len(s) < MIN_N and (blo > 0 or bhi < nb - 1):
                        blo, bhi = max(blo - 1, 0), min(bhi + 1, nb - 1); s = pick()
                    while len(s) < MIN_N and (jlo > 0 or jhi < nn - 1):
                        jlo, jhi = max(jlo - 1, 0), min(jhi + 1, nn - 1); s = pick()
                    if (blo, bhi, jlo, jhi) != (b, b, j, j):
                        T["merged"].append([c, k, b, j, blo, bhi, jlo, jhi, len(s)])
                    rowb.append(qt(s if s else [x["flux"] for x in sel]))
                grid.append(rowb)
            T["flux"][c].append(grid)
    return T


def pred_q(T, x):
    eb, en = T["edges"]["bg"], T["edges"]["nsat3"]
    j = int(np.searchsorted(en, x["nsat3"], side="right"))
    if np.isfinite(x["bg"]):
        return np.array(T["flux"][x["cls"]][x["k"]][int(np.searchsorted(eb, x["bg"], side="right"))][j])
    return np.array(T["flux_ns"][x["cls"]][x["k"]][j])


def evaluate(T, test):
    pit = np.array([np.interp(x["flux"], pred_q(T, x), Q) for x in test])
    med = np.array([pred_q(T, x)[10] for x in test]); f = np.array([x["flux"] for x in test])
    return dict(mae=float(np.median(np.abs(med - f))), rho=float(spearmanr(med, f)[0]), ks=float(kstest(pit, "uniform").statistic))


train = [x for x in rows if x["hdri"] not in HOLD]; test = [x for x in rows if x["hdri"] in HOLD]
VARIANTS = {"无条件": (False, False), "只 bg (T38 口径)": (True, False), "只 nsat3": (False, True), "bg × nsat3": (True, True)}
TT = {k: build(train, *v) for k, v in VARIANTS.items()}
print(f"\n#### 留出集检验 (训练 20 张, 留出 {', '.join(HOLD)})   中位绝对误差 / ρ(预测中位, flux) / PIT 的 KS")
print("| 组 | 块 | " + " | ".join(VARIANTS) + " |"); print("|---|---|" + "---|" * len(VARIANTS))
for lab, sel in [("全部留出", test)] + [(h, [x for x in test if x["hdri"] == h]) for h in HOLD] + \
        [(f"灯具 {SZN[k]}", [x for x in test if x["cls"] == "lamp" and x["k"] == k]) for k in range(3)] + [("细线", [x for x in test if x["cls"] == "line"])]:
    print(f"| {lab} | {len(sel)} | " + " | ".join((lambda e: f"{e['mae']:.2f} / {e['rho']:+.2f} / {e['ks']:.3f}")(evaluate(TT[v], sel)) for v in VARIANTS) + " |")
print("\n留出集 灯具类 各格: 实测中位 / 预测中位 (块数); 行 = bg 档 (暗→亮), 列 = nsat3 档")
T2 = TT["bg × nsat3"]
for k in range(3):
    print(f"  尺寸 {SZN[k]}")
    for b in range(NBG):
        cells = []
        for j in range(NNS):
            s = [x["flux"] for x in test if x["cls"] == "lamp" and x["k"] == k and np.isfinite(x["bg"])
                 and int(np.searchsorted(T2["edges"]["bg"], x["bg"], side="right")) == b and nsb(x["nsat3"]) == j]
            cells.append(f"{np.median(s):5.2f} / {T2['flux']['lamp'][k][b][j][10]:5.2f} ({len(s)})" if s else "—")
        print("    bg" + str(b) + ": " + " | ".join(cells))

F = build(rows)
print(f"\n#### 全部 24 张: bg 档边界 {F['edges']['bg']}, nsat3 档边界 {F['edges']['nsat3']}; 合并的格 {len(F['merged'])}")
for m in F["merged"]:
    print("  合并:", m)
for c in CLS:
    for k in range(3):
        print(f"  {c} {SZN[k]} 中位 (行 bg 暗→亮; 列 nsat3 低→高): " + " ; ".join(
            "/".join(f"{F['flux'][c][k][b][j][10]:.2f}" for j in range(NNS)) for b in range(NBG)))
if OUTJ:
    G = {k: v for k, v in F.items() if k != "merged"}
    G["note"] = ("T39 二维条件分位表 (docs/tasks/T39_NSAT_CONDITIONAL_PRIOR.md): 24 张 Poly Haven 夜景 HDRI 假想 RAW (相机通道截断), "
                 "生成 tools/eval/t39_summary.py; flux[类][尺寸档 0-60/60-150/>=150 µm][bg 档][nsat3 档] = 21 分位 (u = 0, 0.05, ..., 1), 档")
    json.dump(G, open(OUTJ, "w"), ensure_ascii=False)
    print("wrote", OUTJ)

# 帧内相关 (二维条件下的 PIT)
print("\n#### 二维条件下 PIT 的帧内相关 (同一假想照片、同类同尺寸档; ICC)")
for c, k in (("lamp", 0), ("lamp", 1), ("lamp", 2), ("line", 0)):
    g = defaultdict(list)
    for x in rows:
        if x["cls"] == c and x["k"] == k:
            g[(x["hdri"], x["focal"], x["yaw"], x["pitch"])].append(float(np.interp(x["flux"], pred_q(F, x), Q)))
    g = {a: v for a, v in g.items() if len(v) >= 3}
    allv = np.concatenate(list(g.values())); ns = sum(len(v) for v in g.values())
    within = sum(np.var(v) * len(v) for v in g.values()) / ns
    print(f"  {c} {SZN[k]}: 帧 {len(g)}, 块 {ns}, 帧内 ICC {max(0, 1 - within / allv.var()):.2f}")

# 相近配对
print("\n#### 同一假想照片内的相近配对 |Δflux| (档)")
rng = np.random.default_rng(0)
gp = defaultdict(list)
for x in rows:
    if x["cls"] == "lamp" and all(np.isfinite([x["bg"], x["skirt"], x["col_rg"], x["col_bg"]])):
        gp[(x["hdri"], x["focal"], x["yaw"], x["pitch"])].append(x)
real, sim1, sim2 = [], [], []
for v in gp.values():
    if len(v) > 300:
        v = [v[i] for i in rng.choice(len(v), 300, replace=False)]
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            a, b = v[i], v[j]
            if (abs(a["r_um"] - b["r_um"]) < 0.15 * max(a["r_um"], b["r_um"]) and abs(a["bg"] - b["bg"]) < 0.3 and abs(a["skirt"] - b["skirt"]) < 0.3
                    and abs(a["col_rg"] - b["col_rg"]) < 0.15 and abs(a["col_bg"] - b["col_bg"]) < 0.15 and abs(a["nsat3"] - b["nsat3"]) < 0.2):
                real.append(abs(a["flux"] - b["flux"]))
                qa, qb = pred_q(F, a), pred_q(F, b)
                sim1.append(abs(np.interp(rng.uniform(), Q, qa) - np.interp(rng.uniform(), Q, qb)))
                u = rng.uniform(); sim2.append(abs(np.interp(u, Q, qa) - np.interp(u, Q, qb)))
for lab, v in (("实测 (r、bg、skirt、颜色、nsat3 都相近)", real), ("模型 bg × nsat3, 各自独立抽样", sim1), ("模型 bg × nsat3, 共享随机数", sim2)):
    v = np.array(v)
    print(f"| {lab} | {len(v)} | {np.median(v):.2f} | {np.percentile(v, 25):.2f}–{np.percentile(v, 75):.2f} | {np.percentile(v, 90):.2f} |")

if OUT:
    keep = ["hdri", "focal", "yaw", "pitch", "S_over_P90", "blob", "area", "cls", "r_um", "base_rel", "flux", "peak", "bg", "skirt", "slope",
            "fmean", "compact", "nsat3", "nch"] + [f"{q}_{a}" for a in ALT for q in ("nsat3", "nch")] + ["col_rg", "col_bg", "edge"]
    with open(OUT, "w") as fo:
        fo.write("# T39 夜景 HDRI 假想 RAW 的截断块 (Poly Haven CC0 官方 8k; 只存测量)。截断在相机原始通道 (主矩阵 Nikon n29240, 对照 Sony n33920 / Fuji n30464);"
                 " 1600×1067 像素 (22.5 µm), 35/50 mm; S_over_P90 = 白饱和亮度 / P90; flux/peak/bg/skirt/slope 为档; nsat3 = 三通道全饱和比例, nch = 平均饱和通道数;"
                 " 观测量由 filmlab.rawio._block_obs 计算。URL = https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/8k/{hdri}_8k.hdr。生成: tools/eval/t39_hdri_obs.py + t39_summary.py\n")
        fo.write("\t".join(keep) + "\n")
        for x in rows:
            fo.write("\t".join(f"{x[k]:.3f}" if isinstance(x[k], float) else str(x[k]) for k in keep) + "\n")
    print("wrote", OUT)

# 帧内、同一条件格内的相关 (t39_corr 用): 组 = (假想照片, 类, 尺寸档, bg 档, nsat3 档)
print("\n#### 同一假想照片、同一条件格内 PIT 的组内相关 (ICC), 以及相近配对的 PIT 相关")
from scipy.stats import norm
eb, en = F["edges"]["bg"], F["edges"]["nsat3"]
cell = lambda x: (int(np.searchsorted(eb, x["bg"], side="right")) if np.isfinite(x["bg"]) else -1, int(np.searchsorted(en, x["nsat3"], side="right")))
for c in CLS:
    g = defaultdict(list)
    for x in rows:
        if x["cls"] == c:
            g[(x["hdri"], x["focal"], x["yaw"], x["pitch"], x["k"]) + cell(x)].append(float(np.interp(x["flux"], pred_q(F, x), Q)))
    g = {a: v for a, v in g.items() if len(v) >= 2}
    if not g:
        continue
    allv = np.concatenate(list(g.values())); ns = sum(len(v) for v in g.values())
    within = sum(np.var(v) * len(v) for v in g.values()) / ns
    # 高斯秩 (z) 尺度上的组内相关: 供高斯 copula 使用
    z = {a: norm.ppf(np.clip(v, 0.005, 0.995)) for a, v in g.items()}
    allz = np.concatenate(list(z.values())); wz = sum(np.var(v) * len(v) for v in z.values()) / ns
    print(f"  {c}: 组 {len(g)}, 块 {ns}, ICC(u) {max(0, 1 - within / allv.var()):.2f}, ICC(z) {max(0, 1 - wz / allz.var()):.2f}")
pa, pb = [], []
for v in gp.values():
    if len(v) > 300:
        v = [v[i] for i in np.random.default_rng(1).choice(len(v), 300, replace=False)]
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            a, b = v[i], v[j]
            if (abs(a["r_um"] - b["r_um"]) < 0.15 * max(a["r_um"], b["r_um"]) and abs(a["bg"] - b["bg"]) < 0.3 and abs(a["skirt"] - b["skirt"]) < 0.3
                    and abs(a["col_rg"] - b["col_rg"]) < 0.15 and abs(a["col_bg"] - b["col_bg"]) < 0.15 and abs(a["nsat3"] - b["nsat3"]) < 0.2):
                pa.append(np.interp(a["flux"], pred_q(F, a), Q)); pb.append(np.interp(b["flux"], pred_q(F, b), Q))
print(f"  相近配对 ({len(pa)} 对) PIT 的 Spearman 相关: {spearmanr(pa, pb)[0]:.2f}")
# 相近配对在 t39_corr (同格共享 z_组, ρ = 0.42) 下的 |Δflux|
sim3 = []
rng3 = np.random.default_rng(5)
for v in gp.values():
    if len(v) > 300:
        v = [v[i] for i in np.random.default_rng(1).choice(len(v), 300, replace=False)]
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            a, b = v[i], v[j]
            if (abs(a["r_um"] - b["r_um"]) < 0.15 * max(a["r_um"], b["r_um"]) and abs(a["bg"] - b["bg"]) < 0.3 and abs(a["skirt"] - b["skirt"]) < 0.3
                    and abs(a["col_rg"] - b["col_rg"]) < 0.15 and abs(a["col_bg"] - b["col_bg"]) < 0.15 and abs(a["nsat3"] - b["nsat3"]) < 0.2):
                zg = rng3.standard_normal()
                same = cell(a) == cell(b) and a["k"] == b["k"]
                ua = norm.cdf(np.sqrt(0.42) * zg + np.sqrt(0.58) * rng3.standard_normal())
                ub = norm.cdf(np.sqrt(0.42) * (zg if same else rng3.standard_normal()) + np.sqrt(0.58) * rng3.standard_normal())
                sim3.append(abs(np.interp(ua, Q, pred_q(F, a)) - np.interp(ub, Q, pred_q(F, b))))
sim3 = np.array(sim3)
print(f"| 模型 bg × nsat3, t39_corr (同格 ρ = 0.42) | {len(sim3)} | {np.median(sim3):.2f} | {np.percentile(sim3, 25):.2f}–{np.percentile(sim3, 75):.2f} | {np.percentile(sim3, 90):.2f} |")
