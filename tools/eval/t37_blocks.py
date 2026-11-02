"""T37: 8 个夜景 RAW 在 load_raw 层面每块的档数与 flux: t31_base ("stops", U[6, 8]) 对 t37_base ("measured")。
同一 RAW、同一随机数 (每块一次)。长边 1600 像素 (同 t19_model)。用法: python3 t37_blocks.py 输出.json RAW…"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab import generation, rawio

res = {}
for f in sys.argv[2:]:
    res[Path(f).stem] = {}
    for g in ("t31_base", "t37_base"):
        generation.use(g)
        rawio.STOPS_LOG = []
        rgb = rawio.load_raw(f, max_size=1600)
        um = 36000.0 / max(rgb.shape[:2])
        res[Path(f).stem][g] = [dict(b, r_um=b["r_eq"] * um) for b in rawio.STOPS_LOG]
        rawio.STOPS_LOG = None
json.dump(res, open(sys.argv[1], "w"))
B = (60.0, 150.0)
q = lambda v: f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}, {np.percentile(v, 75):.2f}]" if len(v) else "—"
print("| 场景 | 灯具块 (0–60 / 60–150 / ≥150 µm) | stops 旧 → 新 (中位 [四分位]) | flux 旧 → 新 | 灯具 ≥60 µm flux 旧 → 新 | 细线 flux 旧 → 新 (块) | 大面积 flux 旧 → 新 |")
print("|---|---|---|---|---|---|---|")
for s, d in res.items():
    o, n = d["t31_base"], d["t37_base"]
    L = lambda D, c, lo=0, hi=1e9, k="flux": [b[k] for b in D if b["cls"] == c and lo <= b["r_um"] < hi]
    cnt = [len(L(o, "lamp", lo, hi)) for lo, hi in ((0, 60), (60, 150), (150, 1e9))]
    print(f"| {s} | {cnt[0]} / {cnt[1]} / {cnt[2]} | {q(L(o, 'lamp', k='stops'))} → {q(L(n, 'lamp', k='stops'))} | {q(L(o, 'lamp'))} → {q(L(n, 'lamp'))} | "
          f"{q(L(o, 'lamp', 60))} → {q(L(n, 'lamp', 60))} | {q(L(o, 'line'))} → {q(L(n, 'line'))} ({len(L(o, 'line'))}) | {q(L(o, 'large'))} → {q(L(n, 'large'))} |")
allo = {k: [b for d in res.values() for b in d[k]] for k in ("t31_base", "t37_base")}
for lo, hi in ((0, 60), (60, 150), (150, 1e9)):
    vo = [b["flux"] for b in allo["t31_base"] if b["cls"] == "lamp" and lo <= b["r_um"] < hi]
    vn = [b["flux"] for b in allo["t37_base"] if b["cls"] == "lamp" and lo <= b["r_um"] < hi]
    print(f"全部场景 灯具 {lo}–{hi} µm: n={len(vo)} flux 旧 {q(vo)} → 新 {q(vn)}")
