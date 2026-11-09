"""T56: Noritsu (RPL) 黑白颗粒亮度谱在引擎域重新检验扫描光学 σ_o —— 连续传递 (scan_mtf) 对取样核 (t51)。

T51 的拟合 (t51_fit.fit_grain_luma) 在 5 µm 细网格上加 σ_o, 并带每帧自由的白噪声底 N (扫描仪传感器噪声, 在 USM 之前):
  真实(f) = |U(f)|² · (K · 参照(f; σ_o) + N)。
引擎没有 N。这里直接用引擎 (t56_bw: Noritsu 空间传递 + 连续传递) 在 20 µm/像素 (RPL 645) 渲染均匀灰场, 用 T51 同一套测量
(t51_scan_measure.measure_array) 得到输出密度的颗粒谱, 与仓库里的真实谱 (tools/eval/t51/spec_rpl_bw.json.gz) 比较:
  · 引擎域 (N = 0): 真实(f) = K · 引擎(f; σ_o), 每帧 K 自由 —— 引擎实际能复现的谱形;
  · 带 N (与 T51 同形): 真实(f) = K · 引擎(f; σ_o) + |U(f)|² · N, 每帧 K、N 自由 —— 与 T51 拟合可比。
σ_o 网格 (µm); 另算 t51_bw 的取样核 (σ_o = 7 µm = 0.35 像素) 作对照。同一张底片 (develop_negative 一次) 换不同的扫描传递。
用法: python3 -I t56_engine_fit.py 输出.json [边长像素 (默认 512)]
"""
import gzip
import json
import os
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

import t51_scan_measure as SM                                   # noqa: E402
from t51_fit import FIT_BINS, cov_spec, luma_psd, usm_tf       # noqa: E402

PX_UM = 20.0
SO_GRID = (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0, 13.0)
FILMS = {"6_KODAK400TX": "kodak_trix_400@xtol", "7_ILFORDHP5PLUS": "ilford_hp5_plus"}
EVS = tuple(float(v) for v in os.environ.get("T56_EVS", "-1 0 1 2 3").split())     # 输出 Do 约 0.3–0.7, 覆盖真实块
F_SHAPE = (0.11, 0.48)


def render_variants(rec, n):
    """每档曝光: 底片显影一次, 再按各扫描传递扫描 → {变体: {档: measure_array}}。"""
    from filmlab import generation, grain
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.scanning import spatial
    generation.use("t56_bw")
    grain.SILVER_WORKERS = 1
    base = spatial.LABS["noritsu_rpl"]
    variants = {"t51_sampled_7um": (False, base.optics_px)}
    variants.update({f"cont_{so:g}um": (True, so / PX_UM) for so in SO_GRID})
    out = {k: {} for k in variants}
    for ev in EVS:
        t = time.time()
        p = Params(**{**recipe(rec), "timing": "gray", "roll": None, "frame": 0, "grain": 1.0, "scene_lights": None,
                      "frame_width_mm": n * PX_UM / 1000.0})
        eng = Engine(p)
        neg = eng.develop_negative(np.full((n, n, 3), 0.18 * 2.0 ** ev))
        corr_c = tuple(p.corrections)
        corr_d = tuple(c for c in corr_c if c != "scan_mtf")
        for name, (cont, sp) in variants.items():
            spatial.LABS["_t56_fit"] = replace(base, optics_px=sp, name=name)
            eng.p.scan_spatial = "_t56_fit"
            eng.p.corrections = corr_c if cont else corr_d
            o = eng.scan(neg)
            v = spatial.encode(np.clip(o, 0, None)) * 255.0
            out[name][f"ev{ev:+.1f}"] = SM.measure_array(v, PX_UM, synthetic=True)
        spatial.LABS.pop("_t56_fit", None)
        print(rec, ev, f"{time.time() - t:.0f} s", flush=True)
    return out


def _pick(ref, D):
    return min((m for m in ref.values() if m["n"]), key=lambda m: abs(m["D_med"][1] - D))


def fit_shape(frames, ref, with_noise, u):
    """frames: 真实 measure_array 列表; ref: {档: 引擎 measure_array}。每帧 K (与 N) 自由。
    返回 (log 残差 RMS, 各频档的帧平均 log 残差 = 真实 / 模型 的谱形差, 频率)。"""
    ys, ms, fs = [], [], []
    for fr in frames:
        Lr = luma_psd(cov_spec(fr), None, mono=True)
        m = _pick(ref, fr["D_med"][1])
        Lm = luma_psd(cov_spec(m, subtract_q=False), None, mono=True)
        f = np.array(fr["fc"])
        ys.append(np.log(np.clip(Lr[FIT_BINS], 1e-12, None))); ms.append(Lm[FIT_BINS]); fs.append(f[FIT_BINS])
    k = 2 if with_noise else 1

    def res(p):
        r = []
        for i in range(len(ys)):
            mod = np.exp(p[k * i]) * ms[i] + (np.exp(p[k * i + 1]) * u(fs[i]) ** 2 if with_noise else 0.0)
            r.append(ys[i] - np.log(mod))
        return np.concatenate(r)
    p0 = []
    for i in range(len(ys)):
        p0.append(float(np.log(np.median(np.exp(ys[i]) / ms[i]))))
        if with_noise:
            p0.append(float(np.log(np.exp(ys[i]).min() * 0.1)))
    sol = least_squares(res, p0)
    curve = sol.fun.reshape(len(ys), -1).mean(0)
    return float(np.sqrt(np.mean(sol.fun ** 2))), curve.tolist(), fs[0].tolist()


def shape_ratio(m, real=False):
    L = luma_psd(cov_spec(m, subtract_q=real), None, mono=True)
    f = np.array(m["fc"])
    i0, i1 = (int(np.argmin(np.abs(f - x))) for x in F_SHAPE)
    return float(L[i1] / L[i0])


def main():
    dst = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    from filmlab.scanning import spatial
    lab = spatial.LABS["noritsu_rpl"]
    u = lambda f: usm_tf(f, lab.usm_amount, lab.usm_sigma_px)
    real = json.load(gzip.open(HERE / "t51" / "spec_rpl_bw.json.gz", "rt"))
    res = {}
    for film, rec in FILMS.items():
        frames = [v for k, v in real.items() if k.startswith(film) and v["n"] >= 40]
        sims = render_variants(rec, n)
        rows = {}
        for name, ref in sims.items():
            r0, c0, f = fit_shape(frames, ref, False, u)
            r1, c1, _ = fit_shape(frames, ref, True, u)
            rows[name] = dict(rms_engine=r0, rms_with_N=r1, resid_engine=c0, resid_with_N=c1, f=f,
                              P048_P011=float(np.median([shape_ratio(m) for m in ref.values() if m["n"]])))
        res[film] = dict(recipe=rec, n_frames=len(frames),
                         real_P048_P011=float(np.median([shape_ratio(fr, real=True) for fr in frames])),
                         real_P048_P011_frames=[shape_ratio(fr, real=True) for fr in frames], variants=rows)
        print(film, "real P(.48)/P(.11)", round(res[film]["real_P048_P011"], 3))
        for k, r in rows.items():
            print(f"  {k:18s} engine-domain RMS {r['rms_engine']:.3f}  with N {r['rms_with_N']:.3f}  "
                  f"P(.48)/P(.11) {r['P048_P011']:.3f}  resid(engine) {np.round(r['resid_engine'], 2).tolist()}", flush=True)
    json.dump(res, open(dst, "w"), indent=1)


if __name__ == "__main__":
    main()
