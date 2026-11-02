"""T27 A: RAW 截断核外的"裙边"剖面 (半尺寸相机原始 RGB, 未白平衡, 线性; 不经降噪与重建)。

截断判定用 rawio.sensor_saturation (传感器饱和堆积, 逐通道)。每个截断核 (8 连通, 离画面边缘 ≥ 32 px):
  裙边(Δ) = 距截断边界 Δ = 1..10 px 的环中位 / 该通道饱和电平 − 背景 (20–30 px 环中位), 取本核饱和的通道中的最大值;
另给未截断的紧凑亮点 (峰值 0.2–0.9 饱和电平, 30 px 内无截断) 的径向剖面 / 峰值, 作为同一镜头 PSF 近场的参考。
用法: python3 t27_raw_skirt.py 输出.json RAW…   (打印按等效半径 ≤1.5 / 1.5–3 / 3–8 / >8 px 分档的中位裙边)"""
import sys, json
from pathlib import Path
import numpy as np, rawpy
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab import rawio


def run(f):
    with rawpy.imread(f) as r:
        sat = rawio.sensor_saturation(r, per_channel=True)
        cam = r.postprocess(output_color=rawpy.ColorSpace.raw, user_wb=[1, 1, 1, 1], gamma=(1, 1), no_auto_bright=True,
                            output_bps=16, half_size=True, user_flip=None).astype(np.float32) / 65535.0
    if sat is None:
        return [], [], None
    h, w = min(sat.shape[0], cam.shape[0]), min(sat.shape[1], cam.shape[1])
    sat, cam = sat[:h, :w], cam[:h, :w]
    lv = np.array([np.median(cam[..., c][sat[..., c]]) if sat[..., c].any() else np.inf for c in range(3)])
    q = cam / np.where(np.isfinite(lv), lv, cam.reshape(-1, 3).max(0))[None, None, :]
    core = sat.any(-1)
    lab, n = ndi.label(core, structure=np.ones((3, 3)))
    d, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
    own = lab[iy, ix]
    um = 36000.0 / max(h, w)
    rows, R = [], 32
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl[0].start < R or sl[1].start < R or sl[0].stop > h - R or sl[1].stop > w - R:
            continue
        y0, y1, x0, x1 = sl[0].start - R, sl[0].stop + R, sl[1].start - R, sl[1].stop + R
        blob = lab[y0:y1, x0:x1] == k
        mine, dd = own[y0:y1, x0:x1] == k, d[y0:y1, x0:x1]
        satk = sat[y0:y1, x0:x1][blob]
        chans = satk.any(0)
        qs = q[y0:y1, x0:x1][..., chans].max(-1)
        ring = mine & (dd > 20) & (dd <= 30)
        if ring.sum() < 10:
            continue
        bg = float(np.median(qs[ring]))
        sk = []
        for D in range(1, 11):
            m = mine & (dd > D - 0.5) & (dd <= D + 0.5)
            sk.append(float(np.median(qs[m]) - bg) if m.sum() >= 3 else np.nan)
        area = int(blob.sum())
        rows.append(dict(scene=Path(f).name, area=area, req=float(np.sqrt(area / np.pi)), um=um, bg=bg, sk=sk,
                         chans="".join("RGB"[c] for c in range(3) if chans[c]), allsat=float(satk.all(1).mean())))
    qm = q.max(-1)
    pk = (qm == ndi.maximum_filter(qm, 9)) & (qm > 0.2) & (qm < 0.9) & (d > 30)
    yy, xx = np.mgrid[-25:26, -25:26]; rr = np.hypot(yy, xx)
    psf = []
    for y, x in zip(*np.nonzero(pk)):
        if y < R or x < R or y > h - R or x > w - R:
            continue
        win = qm[y - 25:y + 26, x - 25:x + 26]
        bg = np.median(win[(rr > 18) & (rr <= 25)])
        if bg > 0.02 * qm[y, x]:
            continue
        psf.append(dict(scene=Path(f).name, peak=float(qm[y, x] - bg),
                        prof=[float(np.median(win[(rr > D - 0.5) & (rr <= D + 0.5)]) - bg) for D in range(0, 11)]))
    return rows, psf, lv.tolist()


if __name__ == "__main__":
    import multiprocessing as mp
    fs = sys.argv[2:]
    with mp.get_context("spawn").Pool(4) as p:
        res = p.map(run, fs)
    cores = sum((r[0] for r in res), [])
    json.dump({"cores": cores, "psf": sum((r[1] for r in res), []), "lv": {Path(f).name: r[2] for f, r in zip(fs, res)}},
              open(sys.argv[1], "w"))
    print("裙边 = 距截断边界 Δ px 处 / 饱和电平 − 背景 (中位); 只计背景 < 3% 饱和电平的核")
    print(f"{'场景':16s}{'r_eq px':>9s}{'核数':>6s}  Δ=1      2      3      4      6      8")
    for sc in sorted({c["scene"] for c in cores}):
        for lo, hi in ((0, 1.5), (1.5, 3), (3, 8), (8, 1e9)):
            S = [c for c in cores if c["scene"] == sc and lo < c["req"] <= hi and c["bg"] < 0.03]
            if len(S) < 3:
                continue
            A = np.nanmedian(np.array([c["sk"] for c in S]), 0)
            print(f"{sc:16s}{f'{lo}-{hi:g}':>9s}{len(S):6d}  " + " ".join(f"{A[i]:.3f}" for i in (0, 1, 2, 3, 5, 7)))
