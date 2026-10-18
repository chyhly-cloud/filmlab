"""冷暖跟随检验: 真实扫描与模拟的整体偏色是否随拍摄光色同步变化 ("遇冷则冷, 遇暖则暖")。只用 RAW 配对, 不拟合任何参数。

光色坐标 (与渲染无关): 相机记录的拍摄白平衡相对日光白平衡的增益差,
  暖 = log2(B_拍摄/B_日光) − log2(R_拍摄/R_日光)  (档; >0 = 相机为了拉正而加蓝 = 光偏暖)
  绿 = −(log2(R_拍摄/R_日光) + log2(B_拍摄/B_日光)) / 2  (>0 = 光偏品红)
注意: 相机白平衡本身是估计 (有的帧是固定预设), 只当坐标用, 不当真值。

偏色: 整幅 15 < L* < 90 像素的 a*、b* 中位数 (真实扫描与模拟同法)。
条件 (均为 next 代际): 正式定光 ± 输出调整 (矩阵 + tint, 由以 JPEG 为主的数据拟合), 锁定颜色 ± 输出调整。

用法: python3 wb_cast.py 输出.json [场景…]
"""
import json, os, sys, zlib, csv, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")

CONDS = {
    "逐帧定光 + 输出调整": {},
    "逐帧定光, 无输出调整": {"output_conditioning": None},
    "锁定颜色 + 输出调整": {"scan_neutral": "film", "scan_color_strength": 0.0},
    "锁定颜色, 无输出调整": {"scan_neutral": "film", "scan_color_strength": 0.0, "output_conditioning": None},
}


def scenes():
    man = csv.DictReader(open("/home/user/filmlab/tools/eval/epfl_800t_manifest.csv"))
    return [r["scene_id"] for r in man if "灯泡" not in r["description"]] + ["cafe"]


def light_coord(rawp):
    import rawpy
    with rawpy.imread(rawp) as R:
        c, d = np.array(R.camera_whitebalance[:3], float), np.array(R.daylight_whitebalance[:3], float)
    g = np.log2((c / c[1]) / (d / d[1]))
    return float(g[2] - g[0]), float(-(g[0] + g[2]) / 2)


def cast(srgb):
    import colour
    lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(srgb, 0, 1)))
    m = (lab[..., 0] > 15) & (lab[..., 0] < 90)
    return [float(np.median(lab[..., k][m])) for k in (1, 2)]


def job(sid):
    import oracle_timing as ot
    import pair_compare as pc
    from PIL import Image
    from filmlab import generation
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    from filmlab.io import srgb_encode
    rawp, scan, bias = ot.sources(sid)
    real = Image.open(scan).convert("RGB"); real.thumbnail((1600, 1600))
    out = {"light": light_coord(rawp), "real": cast(np.asarray(real, float) / 255)}
    rgb = pc.load_digital(rawp, 1400)[0]
    out["input"] = cast(srgb_encode(np.clip(rgb / max(np.percentile(rgb, 99), 1e-6), 0, 1)))
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    generation.use("next")
    for name, extra in CONDS.items():
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": seed % 36, **extra}))
        out[name] = cast(srgb_encode(np.clip(eng.render(m), 0, 1)))
    return sid, out


def report(res):
    ids = list(res)
    L = np.array([res[s]["light"] for s in ids])
    print(f"{'场景':10s} 暖(档) 绿(档) | 真实 a* b* | " + " | ".join(CONDS))
    for s, l in zip(ids, L):
        print(f"{s:10s} {l[0]:5.2f} {l[1]:5.2f} | {res[s]['real'][0]:5.1f} {res[s]['real'][1]:5.1f} | "
              + " | ".join(f"{res[s][c][0]:5.1f} {res[s][c][1]:5.1f}" for c in CONDS))
    print("\n偏色随光色的斜率 (b* 对 暖, a* 对 绿; 每档 Lab 单位), 与真实扫描的偏色差 (中位 |Δb*|, |Δa*|)")
    for name in ["real", "input", *CONDS]:
        C = np.array([res[s][name] for s in ids])
        kb = np.polyfit(L[:, 0], C[:, 1], 1)[0]; ka = np.polyfit(L[:, 1], C[:, 0], 1)[0]
        R = np.array([res[s]["real"] for s in ids])
        d = np.median(np.abs(C - R), 0)
        print(f"  {name:16s} b*/暖 {kb:6.2f}  a*/绿 {ka:6.2f}   |Δa*| {d[0]:5.2f}  |Δb*| {d[1]:5.2f}  "
              f"相关 r(b*, 真实 b*) {np.corrcoef(C[:, 1], R[:, 1])[0, 1]:5.2f}")


if __name__ == "__main__":
    out = sys.argv[1]
    if os.path.exists(out) and len(sys.argv) == 2:
        report(json.load(open(out))); sys.exit()
    run = sys.argv[2:] or scenes()
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4"))) as p:
        res = dict(p.imap(job, run))
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    report(res)
