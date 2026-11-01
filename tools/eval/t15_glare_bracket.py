"""T15 输入端: 用 RAW 中截断光源外圈的镜头眩光估计截断光源的真实亮度, 以 EPFL 包围曝光的短曝光为真值检验。

物理: 成像 = 场景 ⊗ 镜头 PSF。截断区外的眩光 (未截断, 实测) = 光源总光通量 × PSF 翼部, 与光源亮度成正比。
同一帧内 PSF 相同, 用未截断的紧凑亮源 (光通量全部实测) 标定 "眩光 / 光通量" 比例 c, 再用 c 推算截断光源的光通量。
真值: 同机位短曝光 (EXIF 曝光比 k 换算, 不拟合增益) 中该光源全部通道未饱和时的光通量。

度量 (每个截断块, 半尺寸 Bayer 像素):
  F_obs  = 长曝光中块 (膨胀 2 像素) 内的信号和 (截断后, 偏低);  F_true = 短曝光同区域 × k
  G      = 长曝光外圈 3–8 像素环带均值 − 外圈 14–22 像素环带均值 (局部背景)
  需要的 "缺失档数" = log2(F_true / F_obs);  预测 = log2(G / c / F_obs)
用法: python3 t15_glare_bracket.py
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import os
D = os.environ.get("FILMLAB_DATA", "") + "/epfl/data/digital/"
PAIRS = [("P1480502", "P1480503"), ("P1480506", "P1480505"), ("P1480509", "P1480508"), ("P1480509", "P1480507"),
         ("P1480511", "P1480510"), ("P1480514", "P1480515"), ("P1480513", "P1480515"), ("P1480504", "P1480507")]


def load(name):
    import rawpy
    from filmlab import rawio
    p = D + name + ".RW2"
    with rawpy.imread(p) as r:
        sat = rawio.sensor_saturation(r, per_channel=True)          # 必须在 postprocess 之前
        cam = r.postprocess(output_color=rawpy.ColorSpace.raw, user_wb=[1, 1, 1, 1], gamma=(1, 1), no_auto_bright=True,
                            output_bps=16, half_size=True, user_flip=None).astype(np.float64) / 65535.0
    if sat is None:
        sat = np.zeros(cam.shape, bool)
    h, w = min(cam.shape[0], sat.shape[0]), min(cam.shape[1], sat.shape[1])
    t = rawio.exif_exposure(p, complete_exif=True)
    return cam[:h, :w], sat[:h, :w], t


def ring(mask, a, b):
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, iterations=b) & ~binary_dilation(mask, iterations=a)


def pair(lo_name, sh_name):
    from scipy import ndimage
    from skimage.registration import phase_cross_correlation
    L, Ls, tl = load(lo_name)
    S, Ss, ts = load(sh_name)
    k = (tl[0] / tl[1] ** 2 * tl[2]) / (ts[0] / ts[1] ** 2 * ts[2])
    gl, gs = np.log1p(1e3 * L.mean(-1)), np.log1p(1e3 * k * S.mean(-1))
    sh, err, _ = phase_cross_correlation(gl[::2, ::2], gs[::2, ::2], upsample_factor=4)
    S = ndimage.shift(S, (2 * sh[0], 2 * sh[1], 0), order=1, mode="nearest")
    Ss = ndimage.shift(Ss.astype(float), (2 * sh[0], 2 * sh[1], 0), order=0, mode="nearest") > 0.5
    Y = lambda a: a.sum(-1)
    yl, ys = Y(L), Y(S) * k
    # 配准检验: 长短曝光都未饱和的中等亮度像素的比值
    ok = ~Ls.any(-1) & ~Ss.any(-1) & (yl > 0.05) & (yl < 1.5)
    ratio = float(np.median(yl[ok] / np.maximum(ys[ok], 1e-9))) if ok.sum() > 100 else np.nan
    lab, n = ndimage.label(ndimage.binary_dilation(Ls.any(-1), iterations=1))
    rows = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        y0, y1 = max(sl[0].start - 30, 0), min(sl[0].stop + 30, L.shape[0])
        x0, x1 = max(sl[1].start - 30, 0), min(sl[1].stop + 30, L.shape[1])
        m = lab[y0:y1, x0:x1] == i
        area = int(m.sum())
        if area < 3 or area > 400:
            continue
        core = ndimage.binary_dilation(m, iterations=2)
        r1, r2 = ring(m, 3, 8), ring(m, 14, 22)
        anysat_l = Ls[y0:y1, x0:x1].any(-1)
        if (r1 & anysat_l).any() or (r2 & anysat_l).any():
            continue                      # 环带里有别的截断源
        if Ss[y0:y1, x0:x1][core].any():
            truth = None                  # 短曝光也截断: 只给下限
        else:
            truth = float(ys[y0:y1, x0:x1][core].sum() - ys[y0:y1, x0:x1][r2].mean() * core.sum())
        f_obs = float(yl[y0:y1, x0:x1][core].sum() - yl[y0:y1, x0:x1][r2].mean() * core.sum())
        g = float(yl[y0:y1, x0:x1][r1].mean() - yl[y0:y1, x0:x1][r2].mean())
        rows.append({"area": area, "f_obs": f_obs, "f_true": truth, "glare": g, "y": int(y0), "x": int(x0)})
    # 标定源: 长曝光未截断的紧凑亮源 (峰值 0.3–0.95 截断)
    cal = []
    peak = (L / np.maximum(L.reshape(-1, 3).max(0), 1e-9)).max(-1)
    blab, bn = ndimage.label((peak > 0.3) & ~Ls.any(-1))
    for i, sl in enumerate(ndimage.find_objects(blab), 1):
        y0, y1 = max(sl[0].start - 30, 0), min(sl[0].stop + 30, L.shape[0])
        x0, x1 = max(sl[1].start - 30, 0), min(sl[1].stop + 30, L.shape[1])
        m = blab[y0:y1, x0:x1] == i
        if m.sum() < 3 or m.sum() > 400 or Ls[y0:y1, x0:x1].any():
            continue
        core = ndimage.binary_dilation(m, iterations=2)
        r1, r2 = ring(m, 3, 8), ring(m, 14, 22)
        f = yl[y0:y1, x0:x1][core].sum() - yl[y0:y1, x0:x1][r2].mean() * core.sum()
        g = yl[y0:y1, x0:x1][r1].mean() - yl[y0:y1, x0:x1][r2].mean()
        if f > 0 and g > 0:
            cal.append(g / f)
    return {"pair": [lo_name, sh_name], "k": k, "reg_ratio": ratio, "shift": [float(s) for s in sh], "blobs": rows,
            "cal": cal}


if __name__ == "__main__":
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "2"))) as p:
        res = p.starmap(pair, PAIRS)
    json.dump(res, open(HERE / "t15_glare_bracket.json", "w"), indent=1, default=float)
    allp = []
    for r in res:
        c = np.median(r["cal"]) if r["cal"] else np.nan
        tr = [b for b in r["blobs"] if b["f_true"] and b["f_true"] > 0 and b["f_obs"] > 0 and b["glare"] > 0]
        need = np.array([np.log2(b["f_true"] / b["f_obs"]) for b in tr])
        pred = np.array([np.log2(b["glare"] / c / b["f_obs"]) for b in tr]) if np.isfinite(c) else np.full(len(tr), np.nan)
        allp += list(zip(need, pred, [r["pair"][0]] * len(tr)))
        print(f"{r['pair']} k={r['k']:.2f} 配准比 {r['reg_ratio']:.3f} 截断块 {len(r['blobs'])} 有真值 {len(tr)} 标定源 {len(r['cal'])} c={c:.4g}")
        if len(tr) >= 3:
            print(f"   缺失档数 中位 {np.median(need):+.2f}  预测−真值 中位 {np.nanmedian(pred - need):+.2f}  |误差|中位 {np.nanmedian(abs(pred - need)):.2f}"
                  f"  相关 r={np.corrcoef(need, pred)[0, 1] if np.isfinite(pred).all() else np.nan:+.2f}")
