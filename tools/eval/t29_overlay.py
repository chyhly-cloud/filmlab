"""T29: 成片阶段的光晕叠加 (对照用, 不是胶片物理, 不进主线)。
底图 = 带 remjet 的 Vision3 500T 成片 (*_t22_bxyz_0_remjet.png, 没有片基光晕)。全部运算在线性 sRGB 中, 最后编码回 8 位。

  源   s = max(Y − T, 0) / (1 − T)          Y: 线性亮度 (Rec.709 系数); 截断白 = 1
  扩散 ② 高斯 σ (µm)                        —— 纯 DaVinci 式
       ③ film.layer_halation_kernel(px_um, Variation(forward=5), rho=bounce_retention 红层), 归一到总和 1
  叠加 out = base + k · (s ⊛ 核) · (1, g, b)  (相加, 线性)

k 按"暗背景灯 (灯芯 60–150 µm) 光晕部分 (叠加成片 − 底图) 在距灯芯边缘 150 / 200 µm 档的 R 均值 = 真实 0.263"标定
(真实 = 800T 参考集 − 带 remjet 对照集, T25/T26 的 0.270 / 0.256)。
③ 的 g 按"叠加成片的 G/R (200 / 300 µm 档逐灯中位) = 真实 800T 0.58"标定, b = 0.3·g (真实光晕部分 B/G≈0.3)。

用法:
  python3 t29_overlay.py calibrate 底图目录 输出.json      # 标定 k (与 ③ 的 g)
  python3 t29_overlay.py render 底图目录 参数.json 输出目录  # 写 {场景}_t29_{版本}.png
"""
import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t29_measure as M  # noqa: E402

SCENES = ["n29240", "n40825", "n27416", "n40481", "city_61360", "city_40095", "n20595", "n33920"]
LUMA = np.array([0.2126, 0.7152, 0.0722])
T_DEFAULT = 0.8
TARGET_R = 0.263          # 真实光晕部分 R, 150/200 µm 档均值 (0.270, 0.256)
TARGET_GR = 0.58          # 真实 800T 叠加后 G/R, 200/300 µm 档 (0.57, 0.59)
HALO_BG = 0.3             # 真实光晕部分 B/G (0–60 µm 灯 200–300 µm: 0.26–0.40; 60–150 µm 灯 300 µm: 0.28)
I150, I200, I300 = M.BINS.index(150), M.BINS.index(200), M.BINS.index(300)


def px_um(shape):
    return 36000.0 / max(shape[:2])            # 与 engine 相同 (frame_width_mm = 36)


def encode8(lin):
    x = np.clip(lin, 0.0, 1.0)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return np.round(s * 255.0).astype(np.uint8)


def source(lin, T=T_DEFAULT):
    return np.clip((lin @ LUMA - T) / (1.0 - T), 0.0, None)


_KCACHE = {}


def phys_kernel(um):
    """片基全反射环核 (红层, 多次反射), 只读取包代码。→ (归一核, 原总量)"""
    if um not in _KCACHE:
        from filmlab import film, materials, variation
        v = dataclasses.replace(variation.Variation(), forward=5.0)
        rho = film.bounce_retention(materials.emulsion("kodak_vision3_500t").stock, v)[0]
        k = film.layer_halation_kernel(um, v, rho)
        _KCACHE[um] = (k / k.sum(), float(k.sum()), rho)
    return _KCACHE[um]


def spread(src, um, kind, sigma_um=None):
    if kind == "gauss":
        return gaussian_filter(src, sigma_um / um, mode="constant", truncate=4.0)
    k, _, _ = phys_kernel(um)
    return np.clip(fftconvolve(src, k, mode="same"), 0.0, None)


def compose(base, sp, k, colour):
    return base + k * sp[..., None] * np.asarray(colour, float)[None, None, :]


VERSIONS = {   # 名称: (扩散, σ µm, 颜色)   颜色 None = 标定
    "dv150": ("gauss", 150.0, (1.0, 0.35, 0.05)),
    "dv250": ("gauss", 250.0, (1.0, 0.35, 0.05)),
    "hyb": ("phys", None, None),
    "hyb_pc": ("phys", None, (1.0, 0.15, 0.0)),   # 附: 用模型自身管线 (t22_bxyz − remjet) 的光晕部分颜色
}

# ---- 标定 (各进程缓存底图与扩散结果)
_W = {}


def _init(base_dir, T):
    _W["T"] = T
    _W["dir"] = Path(base_dir)


def _prep(s, kind, sigma):
    key = (s, kind, sigma)
    if key not in _W:
        base = M.load_lin(_W["dir"] / f"{s}_t22_bxyz_0_remjet.png")
        um = px_um(base.shape)
        _W[key] = (base, spread(source(base, _W["T"]), um, kind, sigma))
    return _W[key]


def _measure_one(args):
    s, kind, sigma, k, colour = args
    base, sp = _prep(s, kind, sigma)
    out = M.hs.srgb_to_lin(encode8(compose(base, sp, k, colour)) / 255.0)
    return M.lamps(out)


def measure_version(pool, kind, sigma, k, colour):
    return M.merge(pool.map(_measure_one, [(s, kind, sigma, k, colour) for s in SCENES]))


def halo_R(A, base_sum):
    return M.summary(A["60-150"])["R"] - base_sum["R"]


def calibrate_k(pool, kind, sigma, colour, base_sum, lo=0.01, hi=20.0, it=14):
    f = lambda k: np.mean(halo_R(measure_version(pool, kind, sigma, k, colour), base_sum)[[I150, I200]]) - TARGET_R
    for _ in range(it):            # 对数二分
        mid = np.sqrt(lo * hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


def main_calibrate(base_dir, out_json, T=T_DEFAULT):
    base_sum = M.summary(M.measure_dir(base_dir, "*_t22_bxyz_0_remjet.png")["60-150"])
    res = {"T": T, "target_R": TARGET_R, "target_GR": TARGET_GR, "halo_BG": HALO_BG, "versions": {}}
    with Pool(4, initializer=_init, initargs=(base_dir, T)) as pool:
        for name, (kind, sigma, colour) in VERSIONS.items():
            if colour is not None:
                k = calibrate_k(pool, kind, sigma, colour, base_sum)
            else:                       # ③: g 外层二分 (叠加成片 G/R), k 内层
                glo, ghi = 0.0, 1.0
                for _ in range(7):
                    g = 0.5 * (glo + ghi)
                    col = (1.0, g, HALO_BG * g)
                    k = calibrate_k(pool, kind, sigma, col, base_sum, it=10)
                    gr = np.nanmean(M.summary(measure_version(pool, kind, sigma, k, col)["60-150"])["GR"][[I200, I300]])
                    print(f"   g={g:.3f} k={k:.3f} G/R={gr:.3f}", flush=True)
                    if gr > TARGET_GR:
                        ghi = g
                    else:
                        glo = g
                g = round(0.5 * (glo + ghi), 3)
                colour = (1.0, g, round(HALO_BG * g, 3))
                k = calibrate_k(pool, kind, sigma, colour, base_sum)
            res["versions"][name] = {"kind": kind, "sigma_um": sigma, "colour": list(colour), "k": round(k, 4)}
            print(name, res["versions"][name], flush=True)
    if any(v["kind"] == "phys" for v in res["versions"].values()):
        _, tot, rho = phys_kernel(22.5)
        res["phys_kernel"] = {"forward": 5.0, "rho_red": rho, "total_at_22.5um": tot}
    json.dump(res, open(out_json, "w"), indent=1, ensure_ascii=False)


def main_render(base_dir, par_json, out_dir):
    par = json.load(open(par_json))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in SCENES:
        base = M.load_lin(Path(base_dir) / f"{s}_t22_bxyz_0_remjet.png")
        um = px_um(base.shape)
        src = source(base, par["T"])
        for name, v in par["versions"].items():
            sp = spread(src, um, v["kind"], v["sigma_um"])
            Image.fromarray(encode8(compose(base, sp, v["k"], v["colour"]))).save(out_dir / f"{s}_t29_{name}.png")
        print(s, f"{um:.2f} µm/px", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "calibrate":
        main_calibrate(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else T_DEFAULT)
    elif sys.argv[1] == "render":
        main_render(sys.argv[2], sys.argv[3], sys.argv[4])
