"""T27: 暗背景灯的"超出背景的线性 R"剖面, 按灯芯等效半径细分档, 同时给按灯计 (所有灯的中位) 与按图计 (每图中位再取中位)。
与 t24_dist_colour.py 同一测量 (灯芯 = 线性 RGB 三通道 ≥ 0.93; 背景 = 1.5–2.5 mm 环的中位 < 0.02; 每个像素归最近的灯芯)。
可选 UM=22.5: 先按线性光盒式平均把每张图降到 22.5 µm/像素 (= 模型 1600 像素输出), 使不同分辨率的参考扫描与模型的灯芯判定可比。
用法: [UM=22.5] python3 t27_size_halo.py 目录 [文件通配]     (无通配时读目录下 curated.tsv)
      输出 JSON 到 stdout 末行之前的表格; 环境变量 JSON=路径 另存全部结果。"""
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t19_halo_stats as hs

BINS = [25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500]
CLASSES = [(0, 25), (25, 40), (40, 60), (60, 100), (100, 150), (150, 400)]


def lamp_profiles(im, um, fname=""):
    """→ [(r_eq µm, (len(BINS),3) 超出背景的中位 RGB)], 只含暗背景灯。"""
    core = im.min(-1) >= 0.93
    if core.sum() < 3:
        return []
    d, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
    d = d * um
    lab, n = ndi.label(core)
    own = lab[iy, ix]
    rad = np.sqrt(ndi.sum(core, lab, np.arange(1, n + 1)) / np.pi) * um
    ring = (d > 1500) & (d < 2500)
    bg = np.stack([np.asarray(ndi.median(im[..., c], own * ring, index=np.arange(1, n + 1)), float) for c in range(3)], 1)
    okl = np.where(np.all(np.nan_to_num(bg, nan=1) < hs.dark_bg_max(fname), 1))[0] + 1
    m = np.isin(own, okl) & (d <= BINS[-1]) & (im.max(-1) < 0.999)
    if not m.any():
        return []
    b = np.searchsorted(BINS, d[m])
    L, px = own[m], im[m]
    key = L.astype(np.int64) * 16 + b
    order = np.argsort(key, kind="stable")
    key, px = key[order], px[order]
    uk, start, cnt = np.unique(key, return_index=True, return_counts=True)
    prof = {}
    for kk, s, c in zip(uk, start, cnt):
        if c < 3:
            continue
        l, bi = divmod(int(kk), 16)
        if bi >= len(BINS):
            continue
        prof.setdefault(l, np.full((len(BINS), 3), np.nan))[bi] = np.median(px[s:s + c], 0) - bg[l - 1]
    return [(float(rad[l - 1]), v) for l, v in prof.items()]


def load(path, fmt, target_um=None):
    im = hs.srgb_to_lin(np.asarray(Image.open(path).convert("RGB")) / 255.0)
    um = hs.px_um(im.shape, fmt)
    if um is None:
        return None, None
    if target_um and um < target_um * 0.95:
        f = target_um / um
        h, w = int(im.shape[0] / f), int(im.shape[1] / f)
        im = np.stack([np.asarray(Image.fromarray(im[..., c].astype(np.float32)).resize((w, h), Image.BOX)) for c in range(3)], -1)
        um = um * im.shape[1] / w if False else hs.px_um(im.shape, fmt)
    return im, um


def measure(directory, glob=None, target_um=None):
    ref = Path(directory)
    lines = [l.rstrip("\n").split("\t") for l in open(ref / "curated.tsv")] if glob is None else \
        [[f, "", "", "135"] for f in sorted(p.name for p in ref.glob(glob))]
    out = []
    for f, url, page, fmt in lines:
        im, um = load(ref / f, fmt, target_um)
        if im is None:
            continue
        for r, v in lamp_profiles(im, um, f):
            out.append((f, r, v))
    return out


def summarize(lamps, classes=CLASSES):
    res = {}
    for lo, hi in classes:
        sel = [(f, v) for f, r, v in lamps if lo <= r < hi]
        if not sel:
            continue
        A = np.array([v[:, 0] for f, v in sel])
        G = np.array([v[:, 1] for f, v in sel])
        per = {}
        for f, v in sel:
            per.setdefault(f, []).append(v[:, 0])
        img = np.array([np.nanmedian(np.array(x), 0) for x in per.values() if len(x) >= 3])
        with np.errstate(all="ignore"):
            res[f"{lo}-{hi}"] = {"n": len(sel), "n_img": len(per), "n_img3": len(img),
                                 "R": np.nanmedian(A, 0).tolist(), "GR": np.nanmedian(G / A, 0).tolist(),
                                 "R_img": (np.nanmedian(img, 0).tolist() if len(img) else None),
                                 "top_img_share": max(len(x) for x in per.values()) / len(sel)}
    return res


if __name__ == "__main__":
    tu = float(os.environ["UM"]) if os.environ.get("UM") else None
    lamps = measure(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, tu)
    res = summarize(lamps)
    print("距灯芯边缘 µm              " + " ".join(f"{b:6d}" for b in BINS))
    for k, v in res.items():
        print(f"[{k} µm] 灯 {v['n']}  图 {v['n_img']} (≥3 灯 {v['n_img3']}), 最多一图占 {v['top_img_share']:.0%}")
        print("  R 按灯                  " + " ".join(f"{x:6.3f}" for x in v["R"]))
        if v["R_img"]:
            print("  R 按图                  " + " ".join(f"{x:6.3f}" for x in v["R_img"]))
    if os.environ.get("JSON"):
        json.dump({"res": res, "lamps": [(f, r, v.tolist()) for f, r, v in lamps]}, open(os.environ["JSON"], "w"))
