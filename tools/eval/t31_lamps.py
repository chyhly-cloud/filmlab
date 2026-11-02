"""T31 第 1 步 (docs/tasks/T31_HALO_COLOUR.md): 暗背景灯超出背景的 R G B (逐灯, t29_measure 同一算法) + 灯型 (灯芯边缘 25–75 µm 环 B/G, 同 t25_type_track)。
保存逐灯数组到 npz: 用法 python3 t31_lamps.py 输出.npz 目录 [通配]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t19_halo_stats as hs, t29_measure as tm
BINS = tm.BINS


def lamps(im, fmt="135"):
    um = hs.px_um(im.shape, fmt)
    out = []
    if um is None:
        return out
    core = im.min(-1) >= 0.93
    if core.sum() < 3:
        return out
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
    okl = np.where(np.all(np.nan_to_num(bg, nan=1) < 0.02, 1))[0] + 1
    m = np.isin(own, okl) & (d <= BINS[-1]) & (im.max(-1) < 0.999)
    if not m.any():
        return out
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
        with np.errstate(all="ignore"):
            bgr = ec[l - 1, 2] / ec[l - 1, 1]
        out.append((rows, rad[l - 1], bgr))
    return out


if __name__ == "__main__":
    dst, d = sys.argv[1], Path(sys.argv[2])
    if len(sys.argv) > 3:
        files = [(p, "135") for p in sorted(d.glob(sys.argv[3]))]
    else:
        files = [(d / l.split("\t")[0], l.rstrip("\n").split("\t")[3]) for l in open(d / "curated.tsv")]
    A, rad, bgr, img = [], [], [], []
    for p, fmt in files:
        for rows, r_, t in lamps(tm.load_lin(p), fmt):
            A.append(rows); rad.append(r_); bgr.append(t); img.append(p.name)
    np.savez(dst, A=np.array(A), rad=np.array(rad), bgr=np.array(bgr), img=np.array(img))
    print(dst, len(A))
