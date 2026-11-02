"""T29: 暗背景灯按距灯芯边缘距离的超出背景线性 RGB —— 与 t24_dist_colour.py 同一算法, 但保留逐灯数据, 可直接对内存中的图测量。
(灯芯 = 三通道 ≥ 0.93; 每像素归属最近灯芯; 背景 = 1.5–2.5 mm 环中位, 三通道都 < 0.02 才算暗背景灯; 各距离档取中位后减背景。)
列 BINS[i] 表示距离 (BINS[i-1], BINS[i]] µm, 与 t24 打印的列一致。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t19_halo_stats as hs

BINS = [25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500]
CLASSES = ("0-60", "60-150", "150-400")


def load_lin(path):
    return hs.srgb_to_lin(np.asarray(Image.open(path).convert("RGB")) / 255.0)


def lamps(im, fmt="135"):
    """im: 线性 RGB (已按 8 位量化)。→ {类: [(len(BINS),3) 超出背景的 RGB]}"""
    per = {k: [] for k in CLASSES}
    um = hs.px_um(im.shape, fmt)
    if um is None:
        return per
    core = im.min(-1) >= 0.93
    if core.sum() < 3:
        return per
    d, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
    d = d * um
    lab, n = ndi.label(core)
    own = lab[iy, ix]
    rad = np.sqrt(ndi.sum(core, lab, np.arange(1, n + 1)) / np.pi) * um
    ring = (d > 1500) & (d < 2500)
    bg = np.stack([np.asarray(ndi.median(im[..., c], own * ring, index=np.arange(1, n + 1)), float) for c in range(3)], 1)
    okl = np.where(np.all(np.nan_to_num(bg, nan=1) < 0.02, 1))[0] + 1
    m = np.isin(own, okl) & (d <= BINS[-1]) & (im.max(-1) < 0.999)
    if not m.any():
        return per
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
        if cls:
            per[cls].append(rows)
    return per


def merge(list_of_per):
    out = {k: [] for k in CLASSES}
    for p in list_of_per:
        for k in CLASSES:
            out[k] += p[k]
    return {k: np.array(v) if v else np.zeros((0, len(BINS), 3)) for k, v in out.items()}


def summary(A):
    """A: (n, bins, 3) → 中位 R, G/R, B/G (与 t24 相同: 先逐灯求比, 再取中位)。"""
    with np.errstate(all="ignore"):
        return {"n": len(A), "R": np.nanmedian(A[..., 0], 0), "G": np.nanmedian(A[..., 1], 0),
                "B": np.nanmedian(A[..., 2], 0),
                "GR": np.nanmedian(A[..., 1] / A[..., 0], 0), "BG": np.nanmedian(A[..., 2] / A[..., 1], 0)}


def ring_visibility(A, floor=0.002):
    """逐灯: 150–300 µm 环 (列 200、300 的均值) 的超出背景 R ÷ 700–1000 µm (列 1000) 的 R。
    分母 ≤ floor (约 8 位编码的 1 个码值以下) 的灯另计。→ (中位比, 中位数之比, 有效灯数, 分母过小灯数)"""
    if len(A) == 0:
        return np.nan, np.nan, 0, 0
    i200, i300, i1000 = BINS.index(200), BINS.index(300), BINS.index(1000)
    num = np.nanmean(A[:, [i200, i300], 0], 1)
    den = A[:, i1000, 0]
    ok = np.isfinite(num) & np.isfinite(den)
    big = ok & (den > floor)
    with np.errstate(all="ignore"):
        rr = np.nanmedian(num[big] / den[big]) if big.any() else np.nan
        rm = np.nanmedian(num[ok]) / np.nanmedian(den[ok]) if ok.any() else np.nan
    return rr, rm, int(big.sum()), int((ok & ~big).sum())


def measure_dir(d, glob=None):
    d = Path(d)
    if glob is None:
        lines = [l.rstrip("\n").split("\t") for l in open(d / "curated.tsv")]
        files = [(d / l[0], l[3]) for l in lines]
    else:
        files = [(p, "135") for p in sorted(d.glob(glob))]
    return merge([lamps(load_lin(p), fmt) for p, fmt in files])


if __name__ == "__main__":
    A = measure_dir(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print("距灯芯边缘 µm   " + " ".join(f"{b:6d}" for b in BINS))
    for k in CLASSES:
        s = summary(A[k])
        if not s["n"]:
            continue
        print(f"[{k}] 灯 {s['n']}")
        print("  G/R (超出背景) " + " ".join(f"{x:6.2f}" for x in s["GR"]))
        print("  B/G (超出背景) " + " ".join(f"{x:6.2f}" for x in s["BG"]))
        print("  R   (超出背景) " + " ".join(f"{x:6.3f}" for x in s["R"]))
        print("  环可见度 (150–300)/(700–1000): 逐灯中位 %.2f  中位数之比 %.2f  (有效 %d, 分母过小 %d)" % ring_visibility(A[k]))
