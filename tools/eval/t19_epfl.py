"""T19 对照: EPFL 同场景配对 (真实冲印店扫描 vs 模型) 上的同一光晕统计量; 区分胶片/相机与网上参考图的后期、镜头因素。
真实扫描与模型各自按 36 mm 画幅换算 µm, 用各自画面里的白色灯芯 (同一定义)。
用法: FILMLAB_DATA=… python3 t19_epfl.py 输出.json 代际[,代际…]"""
import json, os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
SCENES = ["R1-00-1", "R1-01-2", "R1-02-3", "R1-03-4", "R1-04-5", "R1-05-6", "R1-06-7", "R1-07-8"]


def job(a):
    spec, sid = a
    gen, _, gl = spec.partition("~")            # 代际~η:r0: 拍摄端眩光 (T20)
    if gl:
        from filmlab import engine
        eta, r0 = (float(x) for x in gl.split(":"))
        engine.CAMERA_GLARE = {"eta": eta, "r0_um": r0, "beta": 1.5}
    gen, _, pl = gen.partition("@")             # 代际@压片板反射率
    if pl:
        from filmlab import engine
        engine.PLATE_REFLECTANCE = float(pl)
    import t19_halo_stats as hs
    import oracle_timing as ot
    if gen == "real":
        _, real = ot.geometry(sid)
        img = np.clip(real, 0, 1)
    else:
        import t03_sheet
        t03_sheet.GENS = (gen,)
        img = t03_sheet.render((gen, sid))[0]
        img = np.round(np.clip(img, 0, 1) * 255) / 255.0
    um = 36000.0 / max(img.shape[:2])
    pr = hs.profiles(hs.srgb_to_lin(img), um)
    return spec, sid, {f"{s[0]}-{s[1]}": (np.nanmedian(np.array(v), 0).tolist() if v else None) for s, v in pr.items()}, um


if __name__ == "__main__":
    import multiprocessing as mp
    dst, gens = sys.argv[1], ["real"] + sys.argv[2].split(",")
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as p:
        rows = p.map(job, [(g, s) for g in gens for s in SCENES], chunksize=1)
    res = {g: {"per_image": {s: v for gg, s, v, _ in rows if gg == g}, "um": {s: u for gg, s, _, u in rows if gg == g}} for g in gens}
    json.dump(res, open(dst, "w"), indent=1)
    import t19_halo_stats as hs
    def st(P):
        P = np.array(P); S = np.nansum(P, -1); r = P[..., 0] / S; b = P[..., 2] / S
        Y = P @ np.array([0.2126, 0.7152, 0.0722])
        return np.nanmedian(r - r[:, -1:], 0), np.nanmedian((Y - Y[:, -1:]) / np.maximum(Y[:, 1:2] - Y[:, -1:], 1e-6), 0), np.nanmedian(b, 0)
    print("环 µm", hs.BINS[1:-1].tolist(), "(亮度形状以 50 µm 环归一; RMS 取 50–1000 µm)")
    score = {}
    for s in hs.SIZES:
        k = f"{s[0]}-{s[1]}"
        print("== 灯芯", k)
        for g in gens:
            P = [np.array(v[k]) for v in res[g]["per_image"].values() if v.get(k) is not None]
            if not P:
                continue
            dr, y, b = st(P)
            if g == "real":
                ref = (dr, y)
            sel = slice(1, 10)
            e_dr = float(np.sqrt(np.nanmean((dr[sel] - ref[0][sel]) ** 2)))
            e_y = float(np.sqrt(np.nanmean((np.log10(np.maximum(y[sel], 1e-3)) - np.log10(np.maximum(ref[1][sel], 1e-3))) ** 2)))
            score.setdefault(g, {})[k] = (e_dr, e_y)
            print(f"  {g:18s} n={len(P)} RMS Δr {e_dr:.3f}  RMS log亮度 {e_y:.3f}")
            print(f"  {'':18s}      Δr " + " ".join(f"{x:6.3f}" for x in dr[:-1]))
            print(f"  {'':18s}      Y  " + " ".join(f"{x:6.3f}" for x in y[:-1]))
            print(f"  {'':18s}      b  " + " ".join(f"{x:6.3f}" for x in b[:-1]))
    print("\n汇总 (三组均值)  RMS Δr / RMS log亮度")
    for g, v in score.items():
        print(f"  {g:24s} {np.mean([a for a, _ in v.values()]):.3f} / {np.mean([b_ for _, b_ in v.values()]):.3f}")
