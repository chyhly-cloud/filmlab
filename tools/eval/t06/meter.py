"""诊断: 每帧测光倍数 log10(scale) 与 T04 实测亮度偏移 x_K 的关系; 另记数码 RAW 的绝对曝光 (EXIF)。"""
import csv, json, sys, zlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
from pathlib import Path; sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[3])]


def job(sid):
    import oracle_timing as ot, pair_compare as pc
    from filmlab.rawio import film_metering
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, note = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    i = np.argmax(rgb.reshape(-1, 3).sum(1) > 0)
    scale = float(np.median(m[..., 1][rgb[..., 1] > 1e-4] / rgb[..., 1][rgb[..., 1] > 1e-4]))
    return sid, scale, float(np.median(rgb @ np.array([0.2126, 0.7152, 0.0722])))


if __name__ == "__main__":
    import multiprocessing as mp
    from accept_calibration import TRAIN, HOLDOUT
    G = {**TRAIN, **HOLDOUT}
    t4 = json.load(open(str(Path(__file__).resolve().parents[1] / "t04_result.json")))["frames"]
    man = {r["scene_id"]: r for r in csv.DictReader(open(str(Path(__file__).resolve().parents[1] / "epfl_800t_manifest.csv")))}
    S = [s for s in G if s != "cafe"]
    with mp.get_context("spawn").Pool(3) as p:
        res = p.map(job, S)
    out = {}
    for sid, scale, ymed in res:
        out[sid] = {"log_scale": float(np.log10(scale)), "raw_median_Y": ymed, "xK": t4[sid]["x"][0], "exif": man[sid]["raw_exposure"]}
        print(f"{sid:9s} {G[sid]:5s} log10 测光倍数 {np.log10(scale):+.2f}  RAW 中位亮度 {ymed:.4f}  x_K {t4[sid]['x'][0]:+.3f}  EXIF {man[sid]['raw_exposure']}")
    a = np.array([out[s]["log_scale"] for s in S]); b = np.array([out[s]["xK"] for s in S])
    c = np.polyfit(a, b, 1)
    print(f"\nx_K = {c[1]:+.3f} + {c[0]:+.3f}·log10(倍数)   r = {np.corrcoef(a, b)[0, 1]:+.2f}  n = {len(S)}")
    for tg, ss in (("训练", [s for s in S if s in TRAIN]), ("留出", [s for s in S if s in HOLDOUT])):
        a2 = np.array([out[s]["log_scale"] for s in ss]); b2 = np.array([out[s]["xK"] for s in ss])
        print(f"  {tg}: 斜率 {np.polyfit(a2, b2, 1)[0]:+.3f}  r {np.corrcoef(a2, b2)[0, 1]:+.2f}")
    json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
