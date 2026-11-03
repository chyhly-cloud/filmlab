"""T24: 暗背景前的灯, 光晕颜色随距离的变化 (减去 1.5–2.5 mm 外圈背景后的线性 R G B, 中位)。
用法: python3 t24_dist_colour.py 目录 [文件通配]"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t19_halo_stats as hs
BINS = [25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500]
ref = Path(sys.argv[1])
lines = [l.rstrip("\n").split("\t") for l in open(ref / "curated.tsv")] if len(sys.argv) < 3 else \
    [[f, "", "", "135"] for f in sorted(p.name for p in ref.glob(sys.argv[2]))]
per = {k: [] for k in ("all", "0-60", "60-150", "150-400")}
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
    rad = np.sqrt(ndi.sum(core, lab, np.arange(1, n + 1)) / np.pi) * um
    ring = (d > 1500) & (d < 2500)
    bg = np.stack([np.asarray(ndi.median(im[..., c], own * ring, index=np.arange(1, n + 1)), float) for c in range(3)], 1)
    okl = np.where(np.all(np.nan_to_num(bg, nan=1) < hs.dark_bg_max(f), 1))[0] + 1
    m = np.isin(own, okl) & (d <= BINS[-1]) & (im.max(-1) < 0.999)
    if not m.any():
        continue
    import pandas as pd
    b = np.searchsorted(BINS, d[m])
    df = pd.DataFrame({"l": own[m], "b": b, "R": im[m][:, 0], "G": im[m][:, 1], "B": im[m][:, 2]})
    g = df.groupby(["l", "b"])
    med, cnt = g.median(), g.size()
    med = med[cnt >= 3]
    for l, sub in med.groupby(level=0):
        rows = np.full((len(BINS), 3), np.nan)
        for (_, bi), v in sub.iterrows():
            if bi < len(BINS):
                rows[bi] = v[["R", "G", "B"]].values - bg[l - 1]
        r_ = rad[l - 1]
        cls = "0-60" if r_ < 60 else "60-150" if r_ < 150 else "150-400" if r_ < 400 else None
        per["all"].append(rows)
        if cls:
            per[cls].append(rows)
print("距灯芯边缘 µm   " + " ".join(f"{b:6d}" for b in BINS))
for k, v in per.items():
    if not v:
        continue
    A = np.array(v)
    with np.errstate(all="ignore"):
        gr = np.nanmedian(A[..., 1] / A[..., 0], 0); bg_ = np.nanmedian(A[..., 2] / A[..., 1], 0)
        rr = np.nanmedian(A[..., 0], 0)
    print(f"[{k}] 灯 {len(A)}")
    print("  G/R (超出背景) " + " ".join(f"{x:6.2f}" for x in gr))
    print("  B/G (超出背景) " + " ".join(f"{x:6.2f}" for x in bg_))
    print("  R   (超出背景) " + " ".join(f"{x:6.3f}" for x in rr))
