"""T45: 颗粒模块 (filmlab.grain) 与 Kodak 官方 48 µm 漫射 RMS 颗粒度的对照 (解析, 不经扫描)。
每层: 模型轴上的曝光 → 名义净密度 D 与 σ(D); 官方曲线按净密度 (减去 Dmin) 插值比较。
官方数据: tools/eval/t45/k52xx_gran.json (数字化读数, 只存数值与来源 URL)。Tri-X 400: F-4017 净密度 1.0 处 17。
用法: python3 tools/eval/t45_grain_check.py [输出.json]"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from filmlab import grain, materials
from filmlab.materials import emulsion, bw_film


def kodak(f):
    K = json.load(open(HERE / "t45" / f)); out = {}
    for i, c in enumerate("RGB"):
        D = np.array(K["density_" + c]); net = np.maximum.accumulate(np.clip(D - D[:3].mean(), 0, None))
        if "sigma_" + c in K:
            from scipy.ndimage import median_filter
            s = np.exp(median_filter(np.log(np.array(K["sigma_" + c])), 5, mode="nearest"))
            out[i] = (net, s)
    return out, K["source"]


def model_curve(e, c):
    sp = materials.grain_spec(e)
    x = e.log_exposure
    H = grain.absolute_exposure(e, sp, np.repeat(x[:, None], 3, 1))[:, c]
    cc = 1 if sp.single_layer else c
    return grain.layer_curves(sp, cc, H)


def main(dst=None):
    res = {}
    for key, f, name in (("kodak_vision3_500t", "k5219_gran.json", "5219 (500T)"), ("kodak_vision3_200t", "k5213_gran.json", "5213 (200T)"),
                         ("kodak_vision3_50d", "k5203_gran.json", "5203 (50D)")):
        e = emulsion(key); K, src = kodak(f)
        for c, (net, s) in K.items():
            D, sg = model_curve(e, c)
            i = np.argsort(D); sel = (net >= 0) & (net <= min(D.max(), net.max()) * 0.97)
            m = np.interp(net[sel], D[i], sg[i])
            err = np.log(m / s[sel])
            pts = {f"{d:.1f}": [float(np.interp(d, net, s)), float(np.interp(d, D[i], sg[i]))] for d in (0.0, 0.2, 0.5, 1.0, 1.5)}
            res[f"{name} {'RGB'[c]}"] = dict(rms_log=float(np.sqrt(np.mean(err ** 2))), median_ratio=float(np.exp(np.median(err))), points=pts)
            key = f"{name} {'RGB'[c]}"
            print(f"{key}: rms log {res[key]['rms_log']:.3f}  中位比 {np.exp(np.median(err)):.2f}  "
                  + "  ".join(f"D{k}: {v[0]*1e3:.1f}/{v[1]*1e3:.1f}" for k, v in pts.items()))
    e = bw_film()
    D, sg = model_curve(e, 1)
    s1 = float(np.interp(1.0, D, sg))
    res["Tri-X 400 (bw)"] = dict(sigma_D1=s1, kodak=0.017)
    print(f"黑白 (Tri-X 400 参数): σ(D=1.0) = {s1*1e3:.1f}  (Kodak F-4017: 17)")
    if dst:
        json.dump(res, open(dst, "w"), indent=1)
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
