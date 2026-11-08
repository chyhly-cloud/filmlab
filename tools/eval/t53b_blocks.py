"""T53b 块边界检验 (docs/tasks/T53b_SILVER_BLOCK_BOUNDARY.md): 640×640, 0.5 µm 像素, sub=4, count_sub=4 (同 T53 4.2),
净 D 0.682; render_silver 按 rows_per_task=16 分块与 640 (整块一次) 对比: 逐位是否相同、视场平均密度、块内行位置
(行号 mod 16) 的平均密度剖面。去 40 像素边框, 2 个种子。
用法: python3 -I t53b_blocks.py 仓库根 输出.json   (仓库根 = 本仓库或 HEAD 的 worktree)"""
import json, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from filmlab import generation, grain, materials
generation.use("t51_bw"); grain.SILVER_WORKERS = 1
N, PX, TRIM, NET_D = 640, 0.5, 40, 0.682
CASES = [("kodak_trix_400", 1.0), ("kodak_trix_400", 2.0), ("kodak_trix_400@rodinal", 1.0), ("kodak_tmax_400", 1.0)]
SEEDS = (1, 2)
out = {}
for key, k in CASES:
    sp1 = materials.silver_spec(materials.bw_silver_film(key))
    lh = np.linspace(-4.0, 0.0, 2001)
    H0 = 10.0 ** np.interp(NET_D, grain.silver_curve(sp1, lh) - grain.silver_density(sp1, 0.0), lh)
    sp = grain.scale_silver(sp1, k)
    res = dict(closed=float(grain.silver_density(sp, H0)), seeds={})
    for sd in SEEDS:
        t = time.time()
        T = {rpt: grain.render_silver(sp, np.full((N, N), H0), PX, seed=1000 * sd + int(10 * k), sub=4, count_sub=4,
                                      workers=1, rows_per_task=rpt) for rpt in (16, N)}
        r = {}
        for rpt, Tf in T.items():
            Tc = Tf[TRIM:-TRIM, TRIM:-TRIM]
            D = -np.log10(Tc)
            rows = np.arange(TRIM, N - TRIM) % 16
            prof = [float(-np.log10(Tc[rows == i].mean())) for i in range(16)]
            r[str(rpt)] = dict(D_mean=float(-np.log10(Tc.mean())), row_in_block_D=prof,
                               row_sd=float(D.mean(1).std()))
        r["bitwise_equal_16_vs_640"] = bool(np.array_equal(T[16], T[N]))
        r["max_abs_dT_16_vs_640"] = float(np.abs(T[16] - T[N]).max())
        r["frac_px_diff_16_vs_640"] = float((T[16] != T[N]).mean())
        r["sec"] = time.time() - t
        res["seeds"][str(sd)] = r
        print(key, k, sd, r["bitwise_equal_16_vs_640"], r["16"]["D_mean"], r[str(N)]["D_mean"], f"{r['sec']:.0f}s", flush=True)
    out[f"{key}/k{k:g}"] = res
    json.dump(out, open(sys.argv[2], "w"), indent=1)
