"""T25: 按灯型分开的光晕颜色轨迹 (完整色度)。
暗背景灯 (灯芯外 1.5–2.5 mm 背景线性 R、G、B < 0.02; 灯芯等效半径 60–400 µm)。
灯型按灯芯边缘 25–75 µm 环 (片基全反射环 ≥ 234 µm 之内, 光晕很少) 的颜色 (减背景) 分:
  暖 (钠灯 / 暖光): B/G < 0.6;  白: B/G ≥ 0.8;  其间不计。
光晕像素: 距灯芯 100–1500 µm; 按线性 R 分档, 每档 G/R 与 B/G 中位。
用法: python3 t25_type_track.py 目录 [文件通配]"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t19_halo_stats as hs
RB = np.arange(-6.0, 0.51, 1.0)
ref = Path(sys.argv[1])
lines = [l.rstrip("\n").split("\t") for l in open(ref / "curated.tsv")] if len(sys.argv) < 3 else \
    [[f, "", "", "135"] for f in sorted(p.name for p in ref.glob(sys.argv[2]))]
rows = []
for f, url, page, fmt in lines:
    im = hs.srgb_to_lin(np.asarray(Image.open(ref / f).convert("RGB")) / 255.0)
    um = hs.px_um(im.shape, fmt)
    if um is None:
        continue
    core = im.min(-1) >= 0.93
    if core.sum() < 3:
        continue
    d, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
    d = d * um
    lab, n = ndi.label(core)
    own = lab[iy, ix]
    idx = np.arange(1, n + 1)
    rad = np.sqrt(np.asarray(ndi.sum(core, lab, idx), float) / np.pi) * um
    ring = (d > 1500) & (d < 2500)
    bg = np.stack([np.asarray(ndi.median(im[..., c], own * ring, index=idx), float) for c in range(3)], 1)
    edge = (d > 25) & (d <= 75)
    ec = np.stack([np.asarray(ndi.median(im[..., c], own * edge, index=idx), float) for c in range(3)], 1) - bg
    ok = np.all(np.nan_to_num(bg, nan=1) < 0.02, 1) & (rad >= 60) & (rad < 400) & np.all(np.isfinite(ec), 1)
    with np.errstate(all="ignore"):
        bgr = ec[:, 2] / ec[:, 1]
    typ = np.where(bgr < 0.6, 1, np.where(bgr >= 0.8, 2, 0)) * ok
    T = np.concatenate([[0], typ])
    m = (d > 100) & (d < 1500) & (T[own] > 0) & (im.max(-1) < 0.999)
    if not m.any():
        continue
    rows.append(pd.DataFrame({"img": f, "lamp": own[m], "type": T[own][m], "R": im[m][:, 0], "G": im[m][:, 1], "B": im[m][:, 2]}))
df = pd.concat(rows)
df["bin"] = np.digitize(np.log2(np.maximum(df.R, 1e-6)), RB)
c = (RB[:-1] + RB[1:]) / 2
for t, nm in ((1, "暖灯 (钠灯/暖光)"), (2, "白灯")):
    s = df[df.type == t]
    print(f"[{nm}] 图 {s.img.nunique()} 灯 {s.groupby(['img', 'lamp']).ngroups} 像素 {len(s)}")
    print("   R(log2) " + " ".join(f"{x:6.1f}" for x in c))
    for q, lab_ in ((s.G / s.R, "G/R"), (s.B / s.G.clip(lower=1e-6), "B/G")):
        g = q.groupby(s.bin).median()
        print(f"   {lab_}     " + " ".join(f"{g.get(i, np.nan):6.2f}" for i in range(1, len(RB))))
