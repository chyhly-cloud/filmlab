"""T20: 拍摄端眩光 (η, r0) 网格, 在夜景 RAW 上与网上 CineStill 800T 参考集比较光晕统计量 (T19 同一测量)。
得分: 三组灯芯平均的 RMS Δr 与 RMS log(亮度形状), 100–1000 µm; 亮度形状 = (Y − Y背景)/(Y50 − Y背景)。
用法: python3 t20_glare_fit.py 参考目录 输出.json RAW…"""
import json, os, sys, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import t19_halo_stats as hs

ETAS = [0.005, 0.01, 0.02, 0.04, 0.08]
R0S = [50.0, 150.0, 400.0]
SEL = slice(2, 10)                                     # 100 … 1000 µm


def shape(P):
    P = np.array(P); S = np.nansum(P, -1); r = P[..., 0] / S
    Y = P @ np.array([0.2126, 0.7152, 0.0722])
    return np.nanmedian(r - r[:, -1:], 0), np.nanmedian((Y - Y[:, -1:]) / np.maximum(Y[:, 1:2] - Y[:, -1:], 1e-6), 0)


def ref_profiles(ref):
    per = {s: [] for s in hs.SIZES}
    for line in open(ref / "curated.tsv"):
        f, url, page, fmt = line.rstrip("\n").split("\t")
        im = np.asarray(Image.open(ref / f).convert("RGB")) / 255.0
        um = hs.px_um(im.shape, fmt)
        if um is None:
            continue
        for s, v in hs.profiles(hs.srgb_to_lin(im), um).items():
            if v:
                per[s].append(np.nanmedian(np.array(v), 0))
    return {f"{s[0]}-{s[1]}": shape(v) for s, v in per.items()}


def score(model_summary, R):
    e = []
    for k, (rdr, ry) in R.items():
        P = [np.array(v[k]) for v in model_summary["per_image"].values() if v.get(k) is not None]
        mdr, my = shape(P)
        e.append((np.sqrt(np.nanmean((mdr[SEL] - rdr[SEL]) ** 2)),
                  np.sqrt(np.nanmean((np.log10(np.maximum(my[SEL], 1e-3)) - np.log10(np.maximum(ry[SEL], 1e-3))) ** 2))))
    return float(np.mean([a for a, _ in e])), float(np.mean([b for _, b in e]))


if __name__ == "__main__":
    ref, dst, files = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    R = ref_profiles(ref)
    res = {}
    configs = [None] + [(e, r) for e in ETAS for r in R0S]
    for c in configs:
        tag = "none" if c is None else f"{c[0]},{c[1]:g}"
        out = Path(dst).with_name(f"t20_model_{tag.replace(',', '_')}.json")
        env = dict(os.environ)
        gen = "next_lab_halo_e2" if c is None else "next_lab_halo_e2g"
        if c is not None:
            env["GLARE"] = f"{c[0]},{c[1]:g}"
        if not out.exists():
            subprocess.run([sys.executable, str(HERE / "t19_model.py"), str(out), gen, *files], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        m = json.load(open(out))[gen]
        res[tag] = score(m, R)
        print(f"眩光 {tag:12s}  RMS Δr {res[tag][0]:.3f}  RMS log亮度 {res[tag][1]:.3f}", flush=True)
    json.dump(res, open(dst, "w"), indent=1)
