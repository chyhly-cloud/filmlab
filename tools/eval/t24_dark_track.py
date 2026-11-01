"""T24: 暗背景前的灯 (灯芯外 1.5–2.5 mm 背景线性 R、G < 0.02; 灯芯等效半径 RMIN–400 µm) 的光晕 G/R 轨迹; 参考扫描与模型成片用同一规则。
光晕像素: 距灯芯 100–1500 µm, 且属于这类灯 (最近灯芯)。按线性 R 分档。
用法: python3 t24_dark_track.py 目录 [文件通配]"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t19_halo_stats as hs
RB = np.arange(-6.0, 0.51, 0.5)
import os
RMIN = float(os.environ.get("RMIN", "60"))      # 灯芯等效半径下限 (µm): 排除小白物体
ref = Path(sys.argv[1])
lines = [l.rstrip("\n").split("\t") for l in open(ref / "curated.tsv")] if len(sys.argv) < 3 else \
    [[f, "", "", "135"] for f in sorted(p.name for p in ref.glob(sys.argv[2]))]
allR, allG, nl, ni = [], [], 0, 0
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
    ring = (d > 1500) & (d < 2500)
    bR = np.asarray(ndi.median(im[..., 0], own * ring, index=np.arange(1, n + 1)), float)
    bG = np.asarray(ndi.median(im[..., 1], own * ring, index=np.arange(1, n + 1)), float)
    rad = np.sqrt(np.asarray(ndi.sum(core, lab, np.arange(1, n + 1)), float) / np.pi) * um
    ok = np.concatenate([[False], (np.nan_to_num(bR, nan=1) < 0.02) & (np.nan_to_num(bG, nan=1) < 0.02) & (rad >= RMIN) & (rad < 400)])
    m = (d > 100) & (d < 1500) & ok[own] & (im[..., 0] < 0.999)
    if m.sum() == 0:
        continue
    nl += int(ok.sum()); ni += 1
    allR.append(im[m][:, 0]); allG.append(im[m][:, 1])
R, G = np.concatenate(allR), np.concatenate(allG)
lr = np.log2(np.maximum(R, 1e-6))
c = (RB[:-1] + RB[1:]) / 2
print(f"图 {ni}  暗背景灯 {nl}  像素 {len(R)}")
print("R(log2)  " + " ".join(f"{x:5.1f}" for x in c))
for q, nm in ((50, "中位 G/R"), (25, "25%    "), (75, "75%    ")):
    print(nm + " " + " ".join(f"{np.percentile(G[s] / R[s], q):5.2f}" if (s := (lr >= a) & (lr < b)).sum() >= 30 else "    -"
                             for a, b in zip(RB[:-1], RB[1:])))
print("像素数   " + " ".join(f"{((lr >= a) & (lr < b)).sum():5d}" for a, b in zip(RB[:-1], RB[1:])))
