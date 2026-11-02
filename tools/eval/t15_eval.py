"""T15 评估: 光晕链第一性原理补项 (返回斜光程吸收 / 相机压片板反射) 的研究代际, 对照 next_lab_halo。
代际写法: 代际名[@压片板反射率], 如 next_lab_halo_plate@0.2。
输出: 红带 RMS (accept_t03 的 4 个场景, 150–1500 µm 红色度) + T13 的 8 个场景的全图 / 光源旁 (0–300 µm) ΔE00。

用法: FILMLAB_DATA=… [FWD=m] [SEED=n] python3 t15_eval.py 输出.json 代际…   (FWD: engine.HALO_FORWARD, T34; SEED: rawio.SEED_OFFSET, T39)
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
SCENES = ["R1-00-1", "R1-02-3", "R1-04-5", "R1-06-7", "R1-07-8", "R1-01-2", "R1-05-6", "R1-03-4"]
BAND = ["R1-00-1", "R1-03-4", "R1-04-5", "R1-07-8"]


def job(args):
    spec, kind, sid, out = args
    gen, _, pl = spec.partition("@")
    from filmlab import engine
    if pl:                             # 镜面 (T15) 与朗伯 (T36, halo_plate_lambert) 两种处理都用这个值
        engine.PLATE_REFLECTANCE = engine.PLATE_REFLECTANCE_LAMBERT = float(pl)
    if os.environ.get("FWD"):          # T34: 光晕核角分布指数 cos^m (engine.HALO_FORWARD; 仅含 "halo_forward" 的代际生效)
        engine.HALO_FORWARD = float(os.environ["FWD"])
    if os.environ.get("SEED"):         # T39: 截断重建随机数种子偏移
        from filmlab import rawio
        rawio.SEED_OFFSET = int(os.environ["SEED"])
    import accept_t03
    if kind == "band":
        os.environ["TAG"] = "_" + spec.replace("@", "_")
        return spec, kind, sid, accept_t03.band_job((gen, out, sid))[1]
    return spec, kind, sid, accept_t03.scene_job((gen, sid))[1]


if __name__ == "__main__":
    import multiprocessing as mp
    dst, specs = sys.argv[1], sys.argv[2:]
    out = str(Path(dst).with_suffix("")) + "_band"
    os.makedirs(out, exist_ok=True)
    jobs = [(g, "band", s, out) for g in specs for s in BAND] + [(g, "scene", s, out) for g in specs for s in SCENES]
    res = {g: {"band": {}, "scene": {}} for g in specs}
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as p:
        for g, k, s, r in p.imap_unordered(job, jobs, chunksize=1):
            res[g][k][s] = r
            print(g, k, s, r if k == "band" else {a: round(b, 2) for a, b in r.items() if isinstance(b, float)}, flush=True)
    json.dump(res, open(dst, "w"), indent=1, default=float)
    print(f"\n{'代际':28s} 红带RMS(均)  " + " ".join(f"{s:>8s}" for s in BAND) + " | 光源旁ΔE(均) 全图ΔE(均)")
    for g in specs:
        b = res[g]["band"]; sc = res[g]["scene"]
        print(f"{g:28s} {np.mean(list(b.values())):8.3f}    " + " ".join(f"{b[s]:8.3f}" for s in BAND) +
              f" | {np.nanmean([sc[s]['near'] for s in SCENES]):8.2f}    {np.nanmean([sc[s]['full'] for s in SCENES]):8.2f}")
