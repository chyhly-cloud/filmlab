"""T24: 网上真实 800T 扫描里光晕的绿/红轨迹 (与 t24_gr_track.py 同一分档)。
灯芯: 三通道线性 ≥ 0.93; 光晕主导: 距灯芯 100–1500 µm 且线性 R ≥ 4 × 该灯 1.5–2.5 mm 外圈的背景 R (只按 R 选, 不偏向 G/R)。
用法: python3 t24_ref_track.py 参考目录"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t19_halo_stats as hs
RB = np.arange(-4.0, 0.51, 0.5)
ref = Path(sys.argv[1])
tracks = []
for line in open(ref / "curated.tsv"):
    f, url, page, fmt = line.rstrip("\n").split("\t")
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
    own = lab[iy, ix]                                  # 最近的灯芯
    bgR = ndi.median(im[..., 0], own * ((d > 1500) & (d < 2500)), index=np.arange(1, n + 1))
    bgR = np.concatenate([[np.nan], np.nan_to_num(np.asarray(bgR, float), nan=np.inf)])
    m = (d > 100) & (d < 1500) & (im[..., 0] >= 4 * bgR[own]) & (im[..., 0] < 0.999)
    R, G = im[m][:, 0], im[m][:, 1]
    lr = np.log2(np.maximum(R, 1e-6))
    row = [float(np.median(G[(lr >= a) & (lr < b)] / R[(lr >= a) & (lr < b)])) if ((lr >= a) & (lr < b)).sum() >= 15 else np.nan
           for a, b in zip(RB[:-1], RB[1:])]
    tracks.append(row)
T = np.array(tracks)
c = (RB[:-1] + RB[1:]) / 2
print("图数", len(T))
print("R(log2)   " + " ".join(f"{x:5.1f}" for x in c))
print("中位 G/R  " + " ".join(f"{np.nanmedian(T[:, i]):5.2f}" for i in range(T.shape[1])))
print("图数/档   " + " ".join(f"{np.isfinite(T[:, i]).sum():5d}" for i in range(T.shape[1])))
print("25%       " + " ".join(f"{np.nanpercentile(T[:, i], 25):5.2f}" for i in range(T.shape[1])))
print("75%       " + " ".join(f"{np.nanpercentile(T[:, i], 75):5.2f}" for i in range(T.shape[1])))
