"""T38: HDRI 假想照片里"看起来相同"的灯是否同样亮: 同一假想照片内两两配对的灯具块,
按观测量是否相近 (r 相差 < 15%、bg / skirt 相差 < 0.3 档、边缘颜色 log2(R/G)、log2(B/G) 相差 < 0.15) 分组, 比较 |Δflux|。
用法: python3 -I t38_pairs.py 测量目录"""
import csv, glob, os, sys
from collections import defaultdict
import numpy as np

rows = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.tsv"))):
    rows += list(csv.DictReader(open(p), delimiter="\t"))
g = defaultdict(list)
for x in rows:
    if x["cls"] != "lamp":
        continue
    v = [float(x[k]) for k in ("r_um", "bg", "skirt", "col_rg", "col_bg", "flux")]
    if np.all(np.isfinite(v)):
        g[(x["hdri"], x["focal"], x["yaw"], x["pitch"])].append(v)
rng = np.random.default_rng(0)
out = {"相近 (r, bg, skirt, 颜色)": [], "只 bg 相近 (同尺寸档)": [], "同一照片任意两块 (同尺寸档)": []}
for v in g.values():
    a = np.array(v)
    if len(a) < 2:
        continue
    if len(a) > 400:
        a = a[rng.choice(len(a), 400, replace=False)]
    i, j = np.triu_indices(len(a), 1)
    r1, r2 = a[i, 0], a[j, 0]
    same_sz = np.searchsorted([60, 150], r1) == np.searchsorted([60, 150], r2)
    df = np.abs(a[i, 5] - a[j, 5])
    close = (np.abs(r1 - r2) < 0.15 * np.maximum(r1, r2)) & (np.abs(a[i, 1] - a[j, 1]) < 0.3) & (np.abs(a[i, 2] - a[j, 2]) < 0.3) \
        & (np.abs(a[i, 3] - a[j, 3]) < 0.15) & (np.abs(a[i, 4] - a[j, 4]) < 0.15)
    out["相近 (r, bg, skirt, 颜色)"] += df[close].tolist()
    out["只 bg 相近 (同尺寸档)"] += df[same_sz & (np.abs(a[i, 1] - a[j, 1]) < 0.3)].tolist()
    out["同一照片任意两块 (同尺寸档)"] += df[same_sz].tolist()
print("| 配对 | 对数 | |Δflux| 中位 | 四分位 | 90% |")
print("|---|---|---|---|---|")
for k, v in out.items():
    v = np.array(v)
    print(f"| {k} | {len(v)} | {np.median(v):.2f} | {np.percentile(v, 25):.2f}–{np.percentile(v, 75):.2f} | {np.percentile(v, 90):.2f} |")

# 对照: 同样的相近配对, 若两块各自独立从 rawio 条件分布 (MEASURED_COND) 抽样, |Δflux| 会是多少
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__)) + "/../.."))
from filmlab import rawio
T = rawio.MEASURED_COND
sim = []
for v in g.values():
    a = np.array(v)
    if len(a) < 2:
        continue
    if len(a) > 400:
        a = a[rng.choice(len(a), 400, replace=False)]
    i, j = np.triu_indices(len(a), 1)
    r1, r2 = a[i, 0], a[j, 0]
    close = (np.abs(r1 - r2) < 0.15 * np.maximum(r1, r2)) & (np.abs(a[i, 1] - a[j, 1]) < 0.3) & (np.abs(a[i, 2] - a[j, 2]) < 0.3) \
        & (np.abs(a[i, 3] - a[j, 3]) < 0.15) & (np.abs(a[i, 4] - a[j, 4]) < 0.15)
    for ii, jj in zip(i[close], j[close]):
        f = [rawio._cond_flux("lamp", a[t, 0], {"bg": a[t, 1]}, rng.uniform()) for t in (ii, jj)]
        sim.append(abs(f[0] - f[1]))
sim = np.array(sim)
print(f"| 相近配对, 模型各自独立抽样 (measured_cond) | {len(sim)} | {np.median(sim):.2f} | {np.percentile(sim, 25):.2f}–{np.percentile(sim, 75):.2f} | {np.percentile(sim, 90):.2f} |")
