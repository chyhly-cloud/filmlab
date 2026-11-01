"""T19 验收量: 参考集统计量的自助法 90% 区间 (按图像重抽样), 按摄影师拆分的一致性, 各代际的偏差。
用法: python3 t19_criteria.py 参考目录 t19_model.json"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t19_halo_stats as hs

ref, model = Path(sys.argv[1]), json.load(open(sys.argv[2]))
per = {s: [] for s in hs.SIZES}; who = {s: [] for s in hs.SIZES}
for line in open(ref / "curated.tsv"):
    f, url, page, fmt = line.rstrip("\n").split("\t")
    im = np.asarray(Image.open(ref / f).convert("RGB")) / 255.0
    um = hs.px_um(im.shape, fmt)
    if um is None:
        continue
    pr = hs.profiles(hs.srgb_to_lin(im), um)
    author = "Schroeder" if "Schroeder" in url else "其他"
    for s, v in pr.items():
        if v:
            per[s].append(np.nanmedian(np.array(v), 0)); who[s].append(author)


def stats(P):
    P = np.array(P); S = np.nansum(P, -1); r = P[..., 0] / S
    Y = P @ np.array([0.2126, 0.7152, 0.0722])
    # Δr; 亮度: 减去局部背景 (1.5–2.5 mm 环) 后相对 25 µm 环, 去掉整体曝光、扫描黑位与暗部提亮的差异, 只看光晕本身的衰减
    return r - r[:, -1:], (Y - Y[:, -1:]) / np.maximum(Y[:, :1] - Y[:, -1:], 1e-6)


rng = np.random.default_rng(0)
B = hs.BINS[1:-1]
crit = {}
for s in hs.SIZES:
    k = f"{s[0]}-{s[1]}"
    dr, yr = stats(per[s])
    boot = np.array([[np.nanmedian(dr[i], 0), np.nanmedian(yr[i], 0)] for i in
                     (rng.integers(0, len(dr), len(dr)) for _ in range(2000))])
    lo, hi = np.nanpercentile(boot, 5, 0), np.nanpercentile(boot, 95, 0)
    med = np.array([np.nanmedian(dr, 0), np.nanmedian(yr, 0)])
    a = np.array(who[s]); sch = a == "Schroeder"
    crit[k] = {"dr": med[0][:-1], "dr_lo": lo[0][:-1], "dr_hi": hi[0][:-1], "y": med[1][:-1], "y_lo": lo[1][:-1], "y_hi": hi[1][:-1]}
    print(f"\n灯芯半径 {k} µm: 参考 {len(dr)} 张 (Schroeder {sch.sum()}, 其他 {(~sch).sum()})")
    print("  环 µm          " + " ".join(f"{b:6d}" for b in B))
    print("  参考 Δr 中位   " + " ".join(f"{x:6.3f}" for x in med[0][:-1]))
    print("  90% 区间       " + " ".join(f"{a_:.2f}-{b_:.2f}"[1:] if True else "" for a_, b_ in zip(lo[0][:-1], hi[0][:-1])))
    if sch.sum() >= 3 and (~sch).sum() >= 3:
        print("  Schroeder Δr  " + " ".join(f"{x:6.3f}" for x in np.nanmedian(dr[sch], 0)[:-1]))
        print("  其他      Δr  " + " ".join(f"{x:6.3f}" for x in np.nanmedian(dr[~sch], 0)[:-1]))
    print("  参考 Y/Y25     " + " ".join(f"{x:6.3f}" for x in med[1][:-1]))
    for g, v in model.items():
        if k not in v["summary"]:
            continue
        P = [np.array(p[k]) for p in v["per_image"].values() if p.get(k) is not None]
        mdr, myr = (np.nanmedian(x, 0)[:-1] for x in stats(P))
        out_dr = ((mdr < crit[k]["dr_lo"]) | (mdr > crit[k]["dr_hi"])).sum()
        out_y = ((myr < crit[k]["y_lo"]) | (myr > crit[k]["y_hi"])).sum()
        print(f"  {g:20s} Δr " + " ".join(f"{x:6.3f}" for x in mdr) + f"   超出区间 Δr {out_dr}/{len(B)}  Y {out_y}/{len(B)}"
              f"  RMS(Δr) {np.sqrt(np.nanmean((mdr - crit[k]['dr']) ** 2)):.3f}")
        print(f"  {'':20s} Y  " + " ".join(f"{x:6.3f}" for x in myr))
json.dump({k: {a: np.round(b, 4).tolist() for a, b in v.items()} for k, v in crit.items()},
          open(Path(sys.argv[2]).with_name("t19_criteria.json"), "w"), indent=1)
