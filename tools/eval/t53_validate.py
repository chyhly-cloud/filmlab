"""T53 颗粒粗细 k: 闭式解与显式模拟 (底片面, 均匀曝光) 的检验 (docs/tasks/T53_GRAIN_COARSENESS.md)。

对 k ∈ {0.5, 1, 1.5, 2, 3}, 同一曝光 H (k = 1 的净漫射密度 0.682, Schmidt 的视场密度; 感光度按 T53 不随 k):
  黑白 (显式银粒, t51_bw 档案): Tri-X 400 / D-76 (三维, 不成团)、Tri-X 400 / Rodinal (显影晶体成团)、T-Max 400 / D-76 (T 颗粒)
    · 平均密度: 闭式解 D(H) (19 个曝光) 与 k = 1 之差; 显式渲染 −log10 E[T] 与闭式解之差;
    · Selwyn σ√A: 方孔 d = 12 / 24 / 48 / 96 µm 的孔径密度 RMS (闭式解 window_variance, 无 PSF, 直接方差;
      显式: 不重叠方孔的平均透射率 → 密度的标准差), 以 k = 1 归一后除以 k;
    · 自相关长度: 透射率协方差归一后降到 1/e 的半径 r_e (闭式解含 0.5 µm 像素方盒; 显式 = 0.5 µm 像素场的自相关)。
  彩色 (染料云, t46 档案): Portra 400 / C-41, 三层
    · 平均密度 (每层): 闭式解 layer_curves 与 k = 1; 显式 grain.apply 后的场平均与目标密度;
    · σ√A: 闭式解 48 µm (白噪声面积律); 显式方孔 12 / 24 / 48 µm;
    · 自相关长度: 每亚群云的高斯 σ (闭式, cloud_sigma_um) 与显式场的 r_e。
显式视场: 0.5 µm 像素, 银粒 640 × 640 (320 µm)、染料 1600 × 1600 (800 µm), 去掉 40 像素边框 (银粒渲染 / 成团的视场边界, T50 局限 5), 2 个种子。
用法: python3 -I t53_validate.py 输出.json [--quick]
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

KS = (0.5, 1.0, 1.5, 2.0, 3.0)
SIDES = (12.0, 24.0, 48.0, 96.0)
AK_SIDES = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0)   # 结构尺度 a_k 的拟合孔径 (µm, 0.5 µm 像素的整数倍)


def fit_ak(sides, sig):
    """Schmidt 两参数式 σ²A = c²·a/(a + a_k) (a = d²) 对各孔径拟合 → (a_k µm², c²)。结构尺度 √a_k (T47 / T50 的颗粒结构尺度)。"""
    from scipy.optimize import curve_fit
    a = np.asarray(sides, float) ** 2
    y = np.asarray(sig, float) ** 2 * a
    ok = np.isfinite(y)
    (c2, ak), _ = curve_fit(lambda a, c2, ak: c2 * a / (a + ak), a[ok], y[ok], p0=(y[ok].max(), 4.0), maxfev=20000)
    return float(ak), float(c2)
PX = 0.5
N, TRIM = 640, 40               # 显式银粒视场 (320 µm)
N_DYE = 1600                    # 染料云视场 (800 µm; 计算便宜, 用大视场让 48 µm 方孔有 16 × 16 个)
NET_D = 0.682


def r_e(C, h):
    """归一协方差 (居中网格, 间距 h) 沿 x、y 轴平均的剖面降到 1/e 的半径 (线性插值)。"""
    c0 = C.shape[0] // 2
    prof = 0.5 * (C[c0, c0:] + C[c0:, c0]) / C[c0, c0]
    i = int(np.argmax(prof < np.exp(-1)))
    if i == 0:
        return float("nan")
    a, b = prof[i - 1], prof[i]
    return float(h * (i - 1 + (a - np.exp(-1)) / (a - b)))


def field_autocorr(T):
    """场的自相关 (FFT, 去均值, 补零), 居中, 与 r_e 同格式 (间距 = 像素)。"""
    x = T - T.mean()
    n0, n1 = x.shape
    F = np.fft.rfft2(x, (2 * n0, 2 * n1))
    ac = np.fft.irfft2(np.abs(F) ** 2, (2 * n0, 2 * n1))
    ac = np.fft.fftshift(ac)
    m = min(n0, n1) // 4
    c0, c1 = n0, n1
    return ac[c0 - m:c0 + m + 1, c1 - m:c1 + m + 1]


def box_cells(F, s):
    h, w = (F.shape[0] // s) * s, (F.shape[1] // s) * s
    return F[:h, :w].reshape(h // s, s, w // s, s).mean((1, 3))


def pixel_box_cov(C, h, px):
    """闭式协方差经 px 像素方盒平均 (两次方盒卷积 = 像素的自相关三角)。"""
    from scipy.ndimage import uniform_filter
    s = max(1, int(round(px / h)))
    return uniform_filter(uniform_filter(C, s, mode="constant"), s, mode="constant")


def silver_case(key, quick):
    from filmlab import grain, materials
    e = materials.bw_silver_film(key)
    sp1 = materials.silver_spec(e)
    lh = np.linspace(-4.0, 0.0, 2001)
    D1 = grain.silver_curve(sp1, lh) - grain.silver_density(sp1, 0.0)
    H0 = 10.0 ** np.interp(NET_D, D1, lh)
    Hc = 0.8 / sp1.ei * 10.0 ** np.linspace(-0.3, 2.3, 19)            # 曲线 19 点 (速度点以上 −0.3 … +2.3 logH)
    Dc1 = grain.silver_density(sp1, Hc)
    out = dict(film=key, H0=float(H0), net_D=NET_D, ks={})
    base = None
    for k in KS:
        t = time.time()
        sp = grain.scale_silver(sp1, k)
        b = grain.silver_budget(sp)
        Dc = grain.silver_density(sp, Hc)
        ET, C, h = grain.silver_covariance(sp, H0)
        sig = grain.silver_granularity(sp, H0, SIDES, psf_um=0.0, adjacent=False)
        sig_k1_dk = grain.silver_granularity(sp1, H0, tuple(d / k for d in SIDES), psf_um=0.0, adjacent=False)
        sig_ak = grain.silver_granularity(sp, H0, AK_SIDES, psf_um=0.0, adjacent=False)
        ak_cf = fit_ak(AK_SIDES, sig_ak)[0]
        re_cf = r_e(C, h)
        re_cf_px = r_e(pixel_box_cov(C, h, PX), h)
        row = dict(coat_g_m2=b["coat_g_m2"], coat_ratio=None, covering_power=b["covering_power"],
                   curve_max_abs_dD=float(np.abs(Dc - Dc1).max()), D_H0=float(-np.log10(ET)),
                   sigma=dict(zip(map(str, SIDES), map(float, sig))),
                   sigma_eq_k1_at_d_over_k=float(np.max(np.abs(sig / sig_k1_dk - 1))),
                   r_e_closed=re_cf, r_e_closed_px=re_cf_px, ecd_fast_um=float(sp.ecd_um[0]),
                   ak_closed_um2=ak_cf, sigma_ak_closed=sig_ak.tolist())
        # 显式银粒 (0.5 µm 像素, 无 PSF)
        seeds = (1,) if quick else (1, 2)
        Ds, sx, rex, sak, sdk = [], [], [], [], []
        for sd in seeds:
            # 整个视场一块渲染 (rows_per_task = N)。T53 时按 16 行分块会在块边界行丢掉块上方晶体伸进来的部分
            # (_stamp_entries 的缓冲区纵向留白只有 M); T53b 留白改为 2M 后分块与整块逐位相同, 这里保持整块以复现 validate.json
            T = grain.render_silver(sp, np.full((N, N), H0), PX, seed=1000 * sd + int(10 * k), sub=4, count_sub=4,
                                    workers=1, rows_per_task=N)
            T = T[TRIM:-TRIM, TRIM:-TRIM]
            Ds.append(-np.log10(T.mean()))
            sx.append([float((-np.log10(box_cells(T, int(d / PX)))).std()) if (T.shape[0] // int(d / PX)) >= 4
                       else float("nan") for d in SIDES])
            ac = field_autocorr(T)
            rex.append(r_e(ac, PX))
            sak.append([float((-np.log10(box_cells(T, int(round(d / PX))))).std()) for d in AK_SIDES])
            # 空间放大的直接检验: 显式 σ_k(d) 与 k = 1 的显式 σ_1(d/k) (d/k 取 0.5 µm 的整数倍)
            sdk.append([float((-np.log10(box_cells(T, int(round(d / PX))))).std()) for d in (12.0, 24.0)])
        sig_ak_x = np.mean(sak, 0)
        row.update(D_explicit=float(np.mean(Ds)), sigma_explicit=dict(zip(map(str, SIDES), np.nanmean(sx, 0).tolist())),
                   r_e_explicit=float(np.mean(rex)), sigma_ak_explicit=sig_ak_x.tolist(),
                   ak_explicit_um2=fit_ak(AK_SIDES, sig_ak_x)[0], sigma_12_24_explicit=np.mean(sdk, 0).tolist(),
                   seconds=round(time.time() - t, 1))
        if base is None and k == 1.0:
            base = row
        out["ks"][str(k)] = row
        print(key, k, json.dumps({a: row[a] for a in ("coat_g_m2", "curve_max_abs_dD", "D_H0", "D_explicit", "r_e_closed_px",
                                                    "r_e_explicit", "seconds")}), flush=True)
    b1 = out["ks"]["1.0"]
    for k in KS:
        r = out["ks"][str(k)]
        r["coat_ratio"] = r["coat_g_m2"] / b1["coat_g_m2"]
        r["selwyn_ratio_over_k"] = {d: r["sigma"][d] / b1["sigma"][d] / k for d in r["sigma"]}
        r["selwyn_ratio_over_k_explicit"] = {d: r["sigma_explicit"][d] / b1["sigma_explicit"][d] / k
                                             for d in r["sigma_explicit"]}
        r["structure_len_over_k"] = dict(closed=np.sqrt(r["ak_closed_um2"] / b1["ak_closed_um2"]) / k,
                                         explicit=np.sqrt(r["ak_explicit_um2"] / b1["ak_explicit_um2"]) / k)
        # σ_k(12, 24 µm) / σ_1(12/k, 24/k) (显式; k = 1 的 σ_1 在 AK_SIDES 上插值到 d/k)
        s1 = np.interp(np.log(np.array([12.0, 24.0]) / k), np.log(AK_SIDES), b1["sigma_ak_explicit"])
        r["explicit_sigma_k_d_over_sigma1_d_over_k"] = (np.asarray(r["sigma_12_24_explicit"]) / s1).tolist()
        r["r_e_ratio_over_k"] = dict(closed=r["r_e_closed"] / b1["r_e_closed"] / k,
                                     closed_px=r["r_e_closed_px"] / b1["r_e_closed_px"] / k,
                                     explicit=r["r_e_explicit"] / b1["r_e_explicit"] / k)
        r["explicit_vs_closed"] = dict(dD=r["D_explicit"] - r["D_H0"],
                                       sigma={d: r["sigma_explicit"][d] / r["sigma"][d] for d in r["sigma"]
                                              if np.isfinite(r["sigma_explicit"][d])})
    return out


def dye_case(stock, process, quick):
    from types import SimpleNamespace
    from filmlab import grain, materials
    e = materials.emulsion(stock)
    sp1 = materials.grain_spec_t46(e, process)
    lh = np.linspace(-5.0, 1.0, 1201)
    D1, s1 = grain.layer_curves(sp1, 1, 10.0 ** lh)
    H0 = 10.0 ** np.interp(0.5 * D1.max(), D1, lh)                          # 绿层净密度为其最大值的一半
    out = dict(film=stock, process=process, H0=float(H0), ks={})
    for k in KS:
        t = time.time()
        sp = grain.scale_grain(sp1, k)
        D = [float(grain.layer_curves(sp, c, H0)[0]) for c in range(3)]
        s48 = [float(grain.layer_curves(sp, c, H0)[1]) for c in range(3)]
        Dk_curve = max(float(np.abs(grain.layer_curves(sp, c, 10.0 ** lh)[0] - grain.layer_curves(sp1, c, 10.0 ** lh)[0]).max())
                       for c in range(3))
        clouds = [grain.cloud_sigma_um(sp, c).tolist() for c in range(3)]
        seeds = (1,) if quick else (1, 2)
        mx, sx, rex = [], [], []
        for sd in seeds:
            st = SimpleNamespace(dye=np.repeat(np.repeat(np.asarray(D)[None, None], N_DYE, 0), N_DYE, 1).copy(),
                                 silver=np.zeros((N_DYE, N_DYE, 3)))
            grain.apply(e, sp, st, PX, np.random.default_rng(2000 * sd + int(10 * k)))
            F = st.dye[TRIM:-TRIM, TRIM:-TRIM]
            mx.append(F.mean((0, 1)).tolist())
            sx.append([[float(box_cells(F[..., c], int(d / PX)).std()) for d in SIDES[:3]] for c in range(3)])
            rex.append([r_e(field_autocorr(F[..., c]), PX) for c in range(3)])
        row = dict(D=D, sigma48_closed=s48, curve_max_abs_dD=Dk_curve, cloud_sigma_um=clouds,
                   mean_explicit=np.mean(mx, 0).tolist(), sigma_explicit=np.mean(sx, 0).tolist(),
                   r_e_explicit=np.mean(rex, 0).tolist(), seconds=round(time.time() - t, 1))
        out["ks"][str(k)] = row
        print(stock, k, json.dumps({a: row[a] for a in ("D", "sigma48_closed", "mean_explicit", "r_e_explicit", "seconds")}),
              flush=True)
    b1 = out["ks"]["1.0"]
    for k in KS:
        r = out["ks"][str(k)]
        r["selwyn48_ratio_over_k"] = [a / b / k for a, b in zip(r["sigma48_closed"], b1["sigma48_closed"])]
        r["selwyn_ratio_over_k_explicit"] = [[a / b / k for a, b in zip(ra, rb)]
                                             for ra, rb in zip(r["sigma_explicit"], b1["sigma_explicit"])]
        r["r_e_ratio_over_k_explicit"] = [a / b / k for a, b in zip(r["r_e_explicit"], b1["r_e_explicit"])]
        r["cloud_ratio_over_k"] = [float(np.mean(np.asarray(a) / np.asarray(b) / k))
                                   for a, b in zip(r["cloud_sigma_um"], b1["cloud_sigma_um"])]
        r["explicit_mean_minus_target"] = [a - b for a, b in zip(r["mean_explicit"], r["D"])]
        # 显式 48 µm 方孔 (42.5 µm 等面积圆的近似: 用 48 µm 方孔按面积换算到 48 µm 圆孔) 与闭式解 (白噪声面积律)
        r["explicit48_vs_closed"] = [r["sigma_explicit"][c][2] * (48.0 / 42.53) / r["sigma48_closed"][c] for c in range(3)]
    return out


def engine_px_checks():
    """整帧像素 (35 mm, 36000/3032 µm) 上的两项: 染料云显式场的 σ_px ∝ k (每层); 显式银粒的平均密度 = 闭式解,
    且没有按 16 行分块的块边界行偏差 (0.5 µm 像素上有, 见 silver_case 的说明)。"""
    from types import SimpleNamespace
    from filmlab import grain, materials
    px = 36000 / 3032
    out = {}
    e = materials.emulsion("kodak_portra_400")
    sp1 = materials.grain_spec_t46(e, "C-41")
    lh = np.linspace(-5.0, 1.0, 1201)
    D1, _ = grain.layer_curves(sp1, 1, 10.0 ** lh)
    H0 = 10.0 ** np.interp(0.5 * D1.max(), D1, lh)
    base = None
    for k in KS:
        sp = grain.scale_grain(sp1, k)
        D = [float(grain.layer_curves(sp, c, H0)[0]) for c in range(3)]
        st = SimpleNamespace(dye=np.repeat(np.repeat(np.asarray(D)[None, None], 800, 0), 800, 1).copy(),
                             silver=np.zeros((800, 800, 3)))
        grain.apply(e, sp, st, px, np.random.default_rng(5))
        sd = st.dye[20:-20, 20:-20].std((0, 1))
        base = sd if k == 1.0 else base
        out.setdefault("dye_portra400_sigma_px", {})[str(k)] = sd.tolist()
    out["dye_portra400_sigma_px_over_k"] = {k: (np.asarray(v) / np.asarray(out["dye_portra400_sigma_px"]["1.0"]) / float(k)).tolist()
                                            for k, v in out["dye_portra400_sigma_px"].items()}
    sp = materials.silver_spec(materials.bw_silver_film("kodak_trix_400"))
    lhs = np.linspace(-4.0, 0.0, 2001)
    Hs = 10.0 ** np.interp(NET_D, grain.silver_curve(sp, lhs) - grain.silver_density(sp, 0.0), lhs)
    for k in (1.0, 2.0, 3.0):
        s_ = grain.scale_silver(sp, k)
        T = grain.render_silver(s_, np.full((1024, 256), Hs), px, seed=3, workers=1)[16:-16, 16:-16]
        Dp = -np.log10(T)
        prof = [float(Dp[(np.arange(Dp.shape[0]) + 16) % 16 == i].mean()) for i in range(16)]
        out.setdefault("silver_trix_12um", {})[str(k)] = dict(closed=float(grain.silver_density(s_, Hs)),
                                                             explicit=float(-np.log10(T.mean())), row_in_block_mean_D=prof)
    return out


def main():
    dst = sys.argv[1]
    quick = "--quick" in sys.argv
    from filmlab import generation, grain
    generation.use("t51_bw")
    grain.SILVER_WORKERS = 1
    res = json.load(open(dst)) if Path(dst).exists() else {}
    jobs = [("silver", "kodak_trix_400"), ("silver", "kodak_trix_400@rodinal"), ("silver", "kodak_tmax_400"),
            ("dye", "kodak_portra_400")]
    if "engine_px" not in res:
        res["engine_px"] = engine_px_checks()
        json.dump(res, open(dst, "w"), indent=1, ensure_ascii=False)
    for kind, key in jobs:
        if key in res:
            continue
        res[key] = silver_case(key, quick) if kind == "silver" else dye_case(key, "C-41", quick)
        json.dump(res, open(dst, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
