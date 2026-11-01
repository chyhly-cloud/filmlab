"""T13 诊断: next_lab_halo 上逐场景扫描光晕源强度 (halation_response × s, 画面与定光中灰一致), 看每个场景的光源旁 (0–300 µm)
ΔE00 在哪个 s 最小; 同时记录 RAW 可观测量 (截断面积比例、截断区个数与大小、EXIF 截断亮度), 供后续判断最佳强度能否由 RAW 预测。
这是诊断, 不是候选。

用法: FILMLAB_DATA=… [MODE=stops] python3 t13_halo_scale.py 输出.json
  MODE=stops: 改为平移截断光源亮度先验 (−2…+3 档), 同时影响灯芯与光晕 (物理上的未知量)
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from accept_calibration import TRAIN, HOLDOUT
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# 光源亮度旋钮 (诊断): 截断光源先验的档数整体平移 (STOPS 模式, 同时影响灯芯与光晕)
OFFSETS = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
SCENES = ["R1-00-1", "R1-02-3", "R1-04-5", "R1-06-7", "R1-07-8", "R1-01-2", "R1-05-6", "R1-03-4"]


def source_stops_patch(offset):
    """截断光源亮度先验整体平移 offset 档 (HIGHLIGHT_PRIOR / LINE_PRIOR 的档数范围与 MAX_STOPS), 总是基于原值。"""
    from filmlab import rawio
    hp = rawio.__dict__.setdefault("_HIGHLIGHT_PRIOR_orig", rawio.HIGHLIGHT_PRIOR)
    lp = rawio.__dict__.setdefault("_LINE_PRIOR_orig", rawio.LINE_PRIOR)
    ms = rawio.__dict__.setdefault("_MAX_STOPS_orig", rawio.MAX_STOPS)
    rawio.HIGHLIGHT_PRIOR = [(f, lum, (a + offset, b + offset)) for f, lum, (a, b) in hp]
    rawio.LINE_PRIOR = (lp[0], (lp[1][0] + offset, lp[1][1] + offset))
    rawio.MAX_STOPS = ms + max(offset, 0.0)


def job(args):
    sid, s = args
    if os.environ.get("MODE") == "stops":
        source_stops_patch(s)
    else:
        import red_band
        red_band.halo_diag_patch(scale=s)
    import accept_t03
    return sid, s, accept_t03.scene_job(("next_lab_halo", sid))[1]


def features(sid):
    import rawpy
    import oracle_timing as ot
    from scipy import ndimage
    from filmlab import rawio
    rawp, _, _ = ot.sources(sid)
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    lab, n = ndimage.label(sat)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    return {"clip_frac": float(sat.mean()), "n_blobs": int(n), "blob_max": int(sizes.max()),
            "blob_p90": float(np.percentile(sizes, 90)), "l_clip": rawio.clip_luminance(rawp)}


if __name__ == "__main__":
    import multiprocessing as mp
    if os.environ.get("MODE") == "stops":
        SCALES = OFFSETS
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3")), maxtasksperchild=1) as p:
        out = p.map(job, [(sid, s) for sid in SCENES for s in SCALES], chunksize=1)
        feats = dict(zip(SCENES, p.map(features, SCENES, chunksize=1)))
    res = {sid: {"features": feats[sid], "by_scale": {}} for sid in SCENES}
    for sid, s, r in out:
        res[sid]["by_scale"][str(s)] = r
    print(f"{'场景':9s} {'集':3s} | " + " ".join(f"×{s:<5g}" for s in SCALES) + " | 最佳 | 截断比例  个数  最大块  EXIF截断亮度")
    for sid in SCENES:
        near = [res[sid]["by_scale"][str(s)]["near"] for s in SCALES]
        best = SCALES[int(np.nanargmin(near))]
        res[sid]["best_scale"] = best
        f = feats[sid]
        print(f"{sid:9s} {'训练' if sid in TRAIN else '留出'} | " + " ".join(f"{v:6.2f}" for v in near) +
              f" | ×{best:<4g} | {f['clip_frac']:.5f} {f['n_blobs']:5d} {f['blob_max']:6d}  {f['l_clip']}")
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=float)
