"""T53: 颗粒粗细 k 的标度律 (闭式解; filmlab tools/eval/t53_validate.py 的闭式部分, 同一算法)。

k = 同一胶片、所有颗粒线尺度 ×k 的假想乳剂; 守恒每亚群显影后的投影光学截面 → 特性曲线不变, 颗粒场 = k = 1 的场放大 k 倍:
  · 曲线: 19 个曝光的闭式密度与 k = 1 之差 (门槛: 0.01 D; T53 实测 ≤ 0.006);
  · Selwyn σ√A ÷ k: 方孔 12 / 24 / 48 / 96 µm 的孔径密度 RMS (直接方差, 无 PSF), 以 k = 1 归一后除以 k (应为 1);
  · σ_k(d) = σ_1(d/k) (严格的空间放大);
  · 结构长度: 透射率协方差降到 1/e 的半径 r_e ÷ k;
  · 银量: 涂布银 ∝ k (三维) / ≈ 常数 (T 颗粒)。
彩色 (Portra 400 / C-41): 每层 48 µm σ ÷ k、云 σ ÷ k、曲线不变。
曝光 H0 = k = 1 的净漫射 0.682 (黑白) / 绿层净密度为其最大值一半 (彩色)。"""
import numpy as np

from .. import dye, silver
from ..emulsion import derive
from .refdata import load

KS = (0.5, 1.0, 1.5, 2.0, 3.0)
SIDES = (12.0, 24.0, 48.0, 96.0)
NET_D = 0.682
CASES = (("kodak_trix_400", "d76"), ("kodak_trix_400", "rodinal"), ("kodak_tmax_400", "d76"))
REF_KEY = {("kodak_trix_400", "d76"): "kodak_trix_400", ("kodak_trix_400", "rodinal"): "kodak_trix_400@rodinal",
           ("kodak_tmax_400", "d76"): "kodak_tmax_400"}


def r_e(C, h):
    c0 = C.shape[0] // 2
    prof = 0.5 * (C[c0, c0:] + C[c0:, c0]) / C[c0, c0]
    i = int(np.argmax(prof < np.exp(-1)))
    if i == 0:
        return float("nan")
    a, b = prof[i - 1], prof[i]
    return float(h * (i - 1 + (a - np.exp(-1)) / (a - b)))


def silver_case(film, dev):
    sp1 = derive(film, dev).spec
    lh = np.linspace(-4.0, 0.0, 2001)
    D1 = silver.silver_curve(sp1, lh) - silver.silver_density(sp1, 0.0)
    H0 = 10.0 ** np.interp(NET_D, D1, lh)
    Hc = 0.8 / sp1.ei * 10.0 ** np.linspace(-0.3, 2.3, 19)
    Dc1 = silver.silver_density(sp1, Hc)
    out = dict(film=film, developer=dev, H0=float(H0), ks={})
    for k in KS:
        sp = derive(film, dev, k).spec
        b = silver.silver_budget(sp)
        ET, C, h = silver.silver_covariance(sp, H0)
        sig = silver.silver_granularity(sp, H0, SIDES, psf_um=0.0, adjacent=False)
        sig_k1_dk = silver.silver_granularity(sp1, H0, tuple(d / k for d in SIDES), psf_um=0.0, adjacent=False)
        out["ks"][str(k)] = dict(coat_g_m2=b["coat_g_m2"], covering_power=b["covering_power"],
                                 curve_max_abs_dD=float(np.abs(silver.silver_density(sp, Hc) - Dc1).max()),
                                 D_H0=float(-np.log10(ET)), sigma=dict(zip(map(str, SIDES), map(float, sig))),
                                 sigma_eq_k1_at_d_over_k=float(np.max(np.abs(sig / sig_k1_dk - 1))), r_e_closed=r_e(C, h))
    b1 = out["ks"]["1.0"]
    for k in KS:
        r = out["ks"][str(k)]
        r["coat_ratio"] = r["coat_g_m2"] / b1["coat_g_m2"]
        r["selwyn_ratio_over_k"] = {d: r["sigma"][d] / b1["sigma"][d] / k for d in r["sigma"]}
        r["r_e_ratio_over_k"] = r["r_e_closed"] / b1["r_e_closed"] / k
    return out


def dye_case(film="kodak_portra_400", process="C-41"):
    sp1 = derive(film, process).spec
    lh = np.linspace(-5.0, 1.0, 1201)
    D1, _ = dye.layer_curves(sp1, 1, 10.0 ** lh)
    H0 = 10.0 ** np.interp(0.5 * D1.max(), D1, lh)
    out = dict(film=film, process=process, H0=float(H0), ks={})
    for k in KS:
        sp = derive(film, process, k).spec
        out["ks"][str(k)] = dict(
            D=[float(dye.layer_curves(sp, c, H0)[0]) for c in range(3)],
            sigma48_closed=[float(dye.layer_curves(sp, c, H0)[1]) for c in range(3)],
            curve_max_abs_dD=max(float(np.abs(dye.layer_curves(sp, c, 10.0 ** lh)[0] - dye.layer_curves(sp1, c, 10.0 ** lh)[0]).max())
                                 for c in range(3)),
            cloud_sigma_um=[dye.cloud_sigma_um(sp, c).tolist() for c in range(3)])
    b1 = out["ks"]["1.0"]
    for k in KS:
        r = out["ks"][str(k)]
        r["selwyn48_ratio_over_k"] = [a / b / k for a, b in zip(r["sigma48_closed"], b1["sigma48_closed"])]
        r["cloud_ratio_over_k"] = [float(np.mean(np.asarray(a) / np.asarray(b) / k)) for a, b in zip(r["cloud_sigma_um"], b1["cloud_sigma_um"])]
    return out


def run():
    ref = load("t53_reference.json")
    res = {}
    diffs = []
    for film, dev in CASES:
        o = silver_case(film, dev)
        rk = ref[REF_KEY[(film, dev)]]
        for k in KS:
            a, b = o["ks"][str(k)], rk["ks"][str(k)]
            diffs += [abs(a["curve_max_abs_dD"] - b["curve_max_abs_dD"]), abs(a["D_H0"] - b["D_H0"]),
                      max(abs(a["sigma"][d] - b["sigma"][d]) for d in a["sigma"]), abs(a["coat_g_m2"] - b["coat_g_m2"])]
        res[f"{film}@{dev}"] = o
    o = dye_case()
    rk = ref["kodak_portra_400"]
    for k in KS:
        a, b = o["ks"][str(k)], rk["ks"][str(k)]
        diffs += [max(abs(x - y) for x, y in zip(a["sigma48_closed"], b["sigma48_closed"])), abs(a["curve_max_abs_dD"] - b["curve_max_abs_dD"])]
    res["kodak_portra_400@C-41"] = o
    res["max_abs_diff_vs_reference"] = float(max(diffs))
    return res
