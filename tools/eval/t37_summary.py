"""T37 汇总: HDRI 块测量 (t37_hdri_blobs.py 的 *_blobs.tsv) → 各类 / 尺寸档的 flux 分布、锚的敏感性、分辨率对照、穹顶指数、
rawio.MEASURED_FLUX 分位表, 以及仓库测量表 tools/eval/t37/hdri_blobs.tsv (主锚 p90、δ = 0、官方 8k)。
用法: python3 -I t37_summary.py 测量目录 [仓库表输出路径]"""
import csv, glob, os, sys
import numpy as np

MEAS = sys.argv[1]
OUT_TSV = sys.argv[2] if len(sys.argv) > 2 else None
Q = np.linspace(0, 1, 21)
BINS = (60.0, 150.0)
SZN = ("0–60", "60–150", "≥150")
CLS = ("lamp", "line", "large")
HI16 = ("street_lamp", "hansaplatz", "shanghai_bund", "golden_bay")


def load(pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(MEAS, pattern))):
        rows += list(csv.DictReader(open(p), delimiter="\t"))
    return rows


def samples(rows, anchor="p90", delta="0.0", lens=("35", "50")):
    """每块 × 每个焦距一条: (hdri, 类, 尺寸档, flux, peak, p_fit, s07)。"""
    out = []
    for x in rows:
        if x["anchor"] != anchor or x["delta"] != delta:
            continue
        for L in lens:
            r = float(x[f"r{L}_um"]); k = int(np.searchsorted(BINS, r, side="right"))
            out.append((x["hdri"], x[f"cls{L}"], k, float(x["flux"]), float(x["peak"]), float(x["p_fit"]), float(x["s07"]), float(x["dmax"])))
    return out


def stats(v):
    v = np.asarray(v)
    if len(v) == 0:
        return "—"
    p = np.percentile(v, [5, 25, 50, 75, 95])
    return f"{p[2]:.2f} [{p[1]:.2f}, {p[3]:.2f}]（{p[0]:.2f}–{p[4]:.2f}）"


def table(S, title):
    print(f"\n#### {title}")
    print("| 类 | 尺寸档 µm | 块数 | HDRI 数 | flux 中位 [四分位]（5–95%） | peak 中位 [四分位] |")
    print("|---|---|---|---|---|---|")
    for c in CLS:
        for k in (0, 1, 2, None):
            sel = [s for s in S if s[1] == c and (k is None or s[2] == k)]
            if not sel:
                continue
            f = [s[3] for s in sel]; pk = [s[4] for s in sel]
            pp = np.percentile(pk, [25, 50, 75])
            print(f"| {c} | {'全部' if k is None else SZN[k]} | {len(sel)} | {len(set(s[0] for s in sel))} | {stats(f)} | {pp[1]:.2f} [{pp[0]:.2f}, {pp[2]:.2f}] |")


rows8 = [x for x in load("*_8k_blobs.tsv")]
print(f"HDRI 8k: {len(set(x['hdri'] for x in rows8))} 张")
fr = load("*_8k_frames.tsv")
print("HDRI 自身裁切检查 (Ymax 处 ≥ 0.98·max 的像素数):", {x["hdri"]: x["n_ge98max"] for x in fr if x["yaw"] == "0" and x["pitch"] == "0.0" and x["anchor"] == "p90" and x["delta"] == "0.0"})

S = samples(rows8)
table(S, "主结果: 锚 p90 (C = 帧内 P90 × 2^3.71), 官方 8k, 35 / 50 mm 合并")
table(samples(rows8, lens=("35",)), "只 35 mm")
table(samples(rows8, lens=("50",)), "只 50 mm")
table(samples(rows8, anchor="p50"), "锚 p50 (C = 帧内中值 × 2^5.63)")

print("\n#### 锚敏感性 (灯具类各尺寸档 flux 中位 [四分位]; 块数)")
print("| 锚 | δ | 0–60 | 60–150 | ≥150 | 细线 全部 | 大面积 全部 |")
print("|---|---|---|---|---|---|---|")
for an in ("p90", "p50"):
    for dl in ("-1.0", "-0.5", "0.0", "0.5", "1.0"):
        Sd = samples(rows8, an, dl)
        cell = lambda sel: (lambda f: f"{np.median(f):.2f} [{np.percentile(f, 25):.2f}, {np.percentile(f, 75):.2f}] ({len(f)})")([s[3] for s in sel]) if sel else "—"
        print(f"| {an} | {dl} | " + " | ".join(cell([s for s in Sd if s[1] == "lamp" and s[2] == k]) for k in (0, 1, 2))
              + f" | {cell([s for s in Sd if s[1] == 'line'])} | {cell([s for s in Sd if s[1] == 'large'])} |")

print("\n#### 每张 HDRI 的灯具类 flux 中位 (主锚; 块数)")
for h in sorted(set(s[0] for s in S)):
    sel = [s[3] for s in S if s[0] == h and s[1] == "lamp"]
    big = [s[3] for s in S if s[0] == h and s[1] == "lamp" and s[2] >= 1]
    print(f"  {h:28s} 全部 {np.median(sel) if sel else np.nan:5.2f} ({len(sel)})   ≥60 µm {np.median(big) if big else np.nan:5.2f} ({len(big)})  最大 {max(sel) if sel else np.nan:5.2f}")
# 每张 HDRI 等权 (敏感性)
print("\n每张 HDRI 等权的灯具类分位 (各 HDRI 的块权重和相等):")
for k in (0, 1, 2):
    sel = [s for s in S if s[1] == "lamp" and s[2] == k]
    hs = sorted(set(s[0] for s in sel)); cnt = {h: sum(1 for s in sel if s[0] == h) for h in hs}
    v = np.array([s[3] for s in sel]); w = np.array([1.0 / cnt[s[0]] for s in sel]); o = np.argsort(v); cw = np.cumsum(w[o]) / w.sum()
    print(f"  {SZN[k]}: q25/50/75 = " + " ".join(f"{v[o][np.searchsorted(cw, q)]:.2f}" for q in (0.25, 0.5, 0.75)))

# 分辨率: 16k 原生 vs 16k 2×2 平均 vs 官方 8k (同 4 张)
print("\n#### 分辨率对照 (4 张; 主锚; 35 / 50 mm 合并)")
print("| 数据 | 类 / 尺寸档 | 块数 | flux 中位 [四分位]（5–95%） | peak 中位 |")
print("|---|---|---|---|---|")
sets = {"官方 8k": [x for x in rows8 if x["hdri"] in HI16], "16k 2×2 平均": load("*_16k_bin2_blobs.tsv"), "16k 原生": load("*_16k_blobs.tsv")}
for lab, rr in sets.items():
    Sr = samples(rr)
    for c, k in (("lamp", 0), ("lamp", 1), ("lamp", 2), ("line", None), ("large", None)):
        sel = [s for s in Sr if s[1] == c and (k is None or s[2] == k)]
        if sel:
            print(f"| {lab} | {c} {'' if k is None else SZN[k]} | {len(sel)} | {stats([s[3] for s in sel])} | {np.median([s[4] for s in sel]):.2f} |")

# 穹顶指数与 s07 对 peak
print("\n#### 穹顶指数 (d_max ≥ 3 像素的块; 拟合 log2(Y/C) = s·ρ^p)")
for lab, rr in (("官方 8k", rows8), ("16k 原生", load("*_16k_blobs.tsv"))):
    Sr = [s for s in samples(rr, lens=("35",)) if np.isfinite(s[5])]
    if not Sr:
        continue
    for c in ("lamp", "large"):
        sel = [s for s in Sr if s[1] == c]
        if sel:
            p = np.percentile([s[5] for s in sel], [25, 50, 75])
            d07 = np.median([s[4] - s[6] for s in sel])
            print(f"  {lab} {c}: n={len(sel)} p 中位 {p[1]:.2f} [{p[0]:.2f}, {p[2]:.2f}];  peak − s(0.7 反解) 中位 {d07:+.2f} 档")

# rawio 分位表 (主结果)
print("\n#### rawio.MEASURED_FLUX (u = 0, 0.05, …, 1 的分位; 档)")
# 某类某档样本 < MIN_N 时并入相邻较小档 (细线 ≥150 → 细线 ≥60); 大面积块按定义 ≥ 1.17 mm, 三档同用全部大面积样本
MIN_N = 100
tab = {}
for c in CLS:
    tab[c] = []
    for k in (0, 1, 2):
        sel = [s[3] for s in S if s[1] == c and s[2] == k]
        kk = k
        while len(sel) < MIN_N and kk > 0:
            kk -= 1; sel = [s[3] for s in S if s[1] == c and s[2] >= kk]
        if len(sel) < MIN_N:
            sel = [s[3] for s in S if s[1] == c]
        tab[c].append([round(float(x), 3) for x in np.percentile(sel, Q * 100)])
        print(f"  {c} {SZN[k]}: 用 n={len(sel)}" + ("" if kk == k else f" (并入 ≥{(0, 60, 150)[kk]} µm)"))
print("MEASURED_FLUX = {")
for c in CLS:
    print(f'    "{c}": (')
    for k in (0, 1, 2):
        print(f"        {tuple(tab[c][k])},")
    print("    ),")
print("}")

if OUT_TSV:
    keep = ["hdri", "url", "res", "yaw", "pitch", "blob", "lat", "lon", "n_clip", "edge", "frac35", "cls35", "cls50", "r35_um", "r50_um",
            "flux", "peak", "dmax", "p_fit", "s_fit", "s07", "rho0", "rho1", "rho2", "rho3", "rho4"]
    with open(OUT_TSV, "w") as fo:
        fo.write("# T37 夜景 HDRI 截断块 (Poly Haven, CC0, 官方 8k .hdr; 只存测量, 不存图)。锚 p90: C = 帧内亮度 P90 × 2^3.71; "
                 "帧 = 36×24 mm、35 mm 焦距的假想照片 (yaw 度, pitch 度)。flux / peak 为高出 C 的档数; r35/r50 为截断面积等效半径 (胶片 µm); "
                 "rho0..4 = 归一化距边界距离五档的平均 log2(Y/C)。生成: tools/eval/t37_hdri_blobs.py + t37_summary.py\n")
        fo.write("\t".join(keep) + "\n")
        for x in rows8:
            if x["anchor"] == "p90" and x["delta"] == "0.0":
                fo.write("\t".join(x[k] for k in keep) + "\n")
    print("wrote", OUT_TSV)
