"""T39: 按 load_raw 层面的块类型分组的光晕部分 (成片测量)。
块类型 (来自 t39_base 的 STOPS_LOG 观测量, t38_blocks.py 的 JSON): 裸灯 = 灯具类、nsat3 ≥ 0.5、bg 在条件表最暗的 3 档 (≤ 第 3 条边界);
亮面 = 灯具类、nsat3 < 0.05。成片里的灯 (同 m1.py: min 通道 ≥ 0.93 的灯芯、暗背景) 按中心距离匹配到块。
每盏灯: 超出背景 (1500–2500 µm 环中位) 的线性 R, 按距灯芯边缘 BINS 分档; 光晕部分 = 除碳成片 − 同一 RAW 带碳成片 (同一灯芯像素, 配对)。
真实: 800T 参考 − 带碳对照 (t27 逐灯 JSON, 同灯芯尺寸档, 不同灯的总体)。
用法: python3 t39_bare.py 块JSON 缓存目录 真实800T.json 真实对照.json 代际后缀…   (如 t31_base_0_f4 t39_base_0_f4)"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]
import t19_halo_stats as hs

BINS = [25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500]
COLS = [2, 3, 4, 5, 6]                 # 100 150 200 300 400 µm
SZ = ((0, 60), (60, 150), (150, 400))


def lamps_R(p8, prj):
    im = hs.srgb_to_lin(np.asarray(Image.open(p8).convert("RGB")) / 255.0)
    rj = hs.srgb_to_lin(np.asarray(Image.open(prj).convert("RGB")) / 255.0)
    um = 36000.0 / max(im.shape[:2])
    core = im.min(-1) >= 0.93
    lab, n = ndi.label(core)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    rad = np.sqrt(np.asarray(ndi.sum(core, lab, idx), float) / np.pi) * um
    com = ndi.center_of_mass(core, lab, idx)
    d, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True); d *= um
    own = lab[iy, ix]
    ring = (d > 1500) & (d < 2500)
    out = []
    for name, a in (("m", im), ("r", rj)):
        bg = np.asarray(ndi.median(a[..., 0], own * ring, index=idx), float)
        prof = np.full((n, len(BINS)), np.nan)
        b = np.searchsorted(BINS, d)
        for j in range(len(BINS)):
            m = (b == j) & (d > 0)
            prof[:, j] = np.asarray(ndi.median(a[..., 0], own * m, index=idx), float) - bg
        out.append((prof, bg))
    (pm, bgm), (pr, _) = out
    res = []
    for i in range(n):
        if np.isfinite(bgm[i]) and bgm[i] < 0.02:
            res.append(dict(rad=rad[i], cy=com[i][0], cx=com[i][1], R=pm[i], H=pm[i] - pr[i]))
    return res


def real_halo(ref, ctl, lo, hi):
    L = lambda fn: np.array([np.array(v)[:, 0] for f, r, v in json.load(open(fn))["lamps"] if lo <= r < hi])
    with np.errstate(all="ignore"):
        return np.nanmedian(L(ref), 0) - np.nanmedian(L(ctl), 0), np.nanmedian(L(ref), 0)


def block_type(o, eb):
    if o["nsat3"] >= 0.5 and np.isfinite(o["bg"]) and o["bg"] <= eb[2]:
        return "裸灯"
    if o["nsat3"] < 0.05:
        return "亮面"
    return "其他"


if __name__ == "__main__":
    from filmlab import rawio
    B = json.load(open(sys.argv[1])); cache = Path(sys.argv[2]); ref, ctl = sys.argv[3], sys.argv[4]; sufs = sys.argv[5:]
    eb = rawio.MEASURED_COND2["edges"]["bg"]
    acc = {}          # (类型, 尺寸档, 后缀) → 光晕部分剖面列表
    flux = {}         # (类型, 代际) → 块 flux
    for s, d in B.items():
        if "t39_base" not in d:
            continue
        blk = d["t39_base"]
        types = [block_type(b["obs"], eb) if b["cls"] == "lamp" else "非灯具" for b in blk]
        for g in d:
            if g == "shape":
                continue
            for t, b in zip(types, d[g]):
                flux.setdefault((t, g), []).append(b["flux"])
        cy = np.array([b["cy"] for b in blk]); cx = np.array([b["cx"] for b in blk]); rr = np.array([b["r_eq"] for b in blk])
        for suf in sufs:
            p8, prj = cache / f"{s}_{suf}.png", cache / f"{s}_{suf}_remjet.png"
            if not (p8.exists() and prj.exists()):
                continue
            for L in lamps_R(p8, prj):
                dist = np.hypot(cy - L["cy"], cx - L["cx"])
                i = int(np.argmin(dist))
                if dist[i] > max(4.0, 2.0 * rr[i] + 2.0):
                    continue
                k = next((j for j, (lo, hi) in enumerate(SZ) if lo <= L["rad"] < hi), None)
                if k is None:
                    continue
                acc.setdefault((types[i], k, suf), []).append(L["H"])
    print("#### 块 flux 中位 [四分位] (load_raw 层面, 同一块)")
    gens = sorted({g for (_, g) in flux})
    for t in ("裸灯", "其他", "亮面"):
        print(f"  {t}: " + " | ".join(f"{g} {np.median(flux[(t, g)]):.2f} [{np.percentile(flux[(t, g)], 25):.2f}, {np.percentile(flux[(t, g)], 75):.2f}] (n={len(flux[(t, g)])})"
                                      for g in gens if (t, g) in flux))
    print("\n#### 光晕部分 (除碳 − 带碳, 配对) 对真实之比, 100 / 150 / 200 / 300 / 400 µm (灯数)")
    for k, (lo, hi) in enumerate(SZ):
        real, _ = real_halo(ref, ctl, lo, hi)
        print(f"  灯芯 {lo}–{hi} µm, 真实光晕部分 " + " ".join(f"{real[c]:.3f}" for c in COLS))
        for t in ("裸灯", "其他", "亮面"):
            for suf in sufs:
                v = acc.get((t, k, suf), [])
                if not v:
                    continue
                with np.errstate(all="ignore"):
                    m = np.nanmedian(np.array(v), 0)
                print(f"    {t} {suf:16s} (n={len(v):4d}) 绝对 " + " ".join(f"{m[c]:.3f}" for c in COLS) + "  / 真实 " + " ".join(f"{m[c] / real[c]:.2f}" for c in COLS))
