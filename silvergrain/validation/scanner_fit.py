"""T55c: 按 silvergrain 自己实现的扫描传递重新拟合 Noritsu (RPL) 的光学 σ_o (T51 第 3–5 节的同一方法)。

T51 的拟合 (filmlab tools/eval/t51_fit.py) 用的模型参照是: 细像素 (5 µm) 渲染 → 连续高斯 σ_o (在细像素上) → 冲印店像素方盒积分;
filmlab 引擎实际执行的是在 20 µm 输出像素上的离散高斯核 (σ 0.35 像素, 几乎是恒等)。silvergrain 执行的是:
  底片 (输出像素积分) → 扫描光学 = 输出像素网格上的连续高斯 MTF exp(−2π² σ² f²) (image._gauss, 频域) → 冲印店孔径 → USM。
这里的参照就用 silvergrain 的这条链 (同一像素、同一实现), 所以拟合出的 σ_o 可以原样写进 data/scanners.json。
真实: RPL 颗粒谱 (validation/data/spec_rpl_{c,bw}.json.gz: T51 t51_scan_measure 的 64 × 64 块、16 档径向谱, 只存测量)。
模型: 真实亮度谱 = |U(f)|² · (K · 参照谱(f; σ_o) + N), K、N 每帧自由 (影调斜率 × 量级、扫描仪白噪声底)。
  · USM (a, σ_u) 取 T51 直边 (训练片 85 条) 的结果 a = 1.032, σ_u = 0.683 像素 —— 直边拟合里每条边的模糊 σ_e 自由, 与光学的实现无关。
  · 训练片 (Pro 400H、Portra 400 / 160) 自由拟合; 直边 → 颗粒: 固定 USM, 在留出片 (Portra 800、Ektar 100) 上定 σ_o;
    跨片种留出: 同一 (σ_o, USM) 预测 Tri-X、HP5+ 的谱形 (只放开 K、N)。
"""
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import measure as M

DATA = Path(__file__).resolve().parent / "data"
USM_EDGES = (1.0322610822583158, 0.6826243370961412)      # T51 tools/eval/t51/fit.json noritsu.edges_train
SIGMAS_UM = (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0)
PX = 20.0
NB = 16
TILE = 64
FIT_BINS = slice(2, 15)
W601 = np.array([0.299, 0.587, 0.114])
TRAIN = ("1_FUJIPRO400H", "2_KODAKPORTRA400", "4_KODAKPORTRA160")
HOLD = ("3_KODAKPORTRA800", "5_KODAKEKTAR100")
BW = {"6_KODAK400TX": ("kodak_trix_400", "xtol"), "7_ILFORDHP5PLUS": ("ilford_hp5_plus", "d76")}


def usm_tf(f, a, su):
    return 1.0 + a * (1.0 - np.exp(-2 * np.pi ** 2 * su ** 2 * f ** 2))


# --------------------------------------------------------------- 谱 (t51_scan_measure 的同一算法)
def _radial(T):
    fy = np.fft.fftfreq(T)[:, None]; fx = np.fft.fftfreq(T)[None, :]
    fr = np.sqrt(fy ** 2 + fx ** 2)
    e = np.linspace(0, 0.5, NB + 1)
    return [(fr >= e[i]) & (fr < e[i + 1]) & (fr > 0) for i in range(NB)], 0.5 * (e[1:] + e[:-1])


def measure_array(img8, T=TILE, border=8):
    """均匀合成场 (H, W, 3) 编码码值 → 输出密度 Do 的每档谱 P、互谱 X、D_med (同 t51_scan_measure, synthetic)。"""
    m, fc = _radial(T)
    w = np.hanning(T)[:, None] * np.hanning(T)[None, :]
    w = w / np.sqrt((w ** 2).mean())
    H, W = img8.shape[:2]
    Ps, Xs, Ds = [], [], []
    for y in range(border, H - T + 1 - border, T):
        for x in range(border, W - T + 1 - border, T):
            t8 = np.asarray(img8[y:y + T, x:x + T], float)
            c = t8 / 255.0
            lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
            D = -np.log10(np.clip(lin, 1e-6, None))
            r = M.detrend(D)
            F = np.fft.fft2(r * w[..., None], axes=(0, 1)) / T
            Ps.append([[np.mean(np.abs(F[mm, k]) ** 2) for k in range(3)] for mm in m])
            Xs.append([[np.mean((F[mm, a] * np.conj(F[mm, b])).real) for a, b in ((0, 1), (1, 2), (0, 2))] for mm in m])
            Ds.append(D.mean((0, 1)))
    return dict(fc=fc.tolist(), D_med=np.median(Ds, 0).tolist(), q_mean=[0.0, 0.0, 0.0],
                P=np.mean(Ps, 0).tolist(), X=np.mean(Xs, 0).tolist())


def cov_spec(m, subtract_q=True):
    P = np.array(m["P"]); X = np.array(m["X"])
    if subtract_q:
        P = P - np.array(m["q_mean"])[None, :]
    C = np.zeros((len(P), 3, 3))
    for i in range(3):
        C[:, i, i] = P[:, i]
    for k, (i, j) in enumerate(((0, 1), (1, 2), (0, 2))):
        C[:, i, j] = C[:, j, i] = X[:, k]
    return C


def code_slopes(D):
    from ..image import srgb_encode
    lin = 10.0 ** -np.asarray(D)
    dv = np.where(lin <= 0.0031308, 12.92, 1.055 / 2.4 * np.power(np.maximum(lin, 1e-9), 1 / 2.4 - 1))
    return -dv * lin * np.log(10)


def luma_psd(C, D, mono=False):
    if mono:
        return C[:, 1, 1]
    w = W601 * code_slopes(D)
    return np.einsum("i,fij,j->f", w, C, w)


def pick_level(ref, D):
    return min(ref, key=lambda k: abs(ref[k]["D_med"][1] - D[1]))


def fit_grain_luma(frames, refs_by_so, mono=False, fixed_usm=None, bins=FIT_BINS):
    """t51_fit.fit_grain_luma 的同一算法。"""
    best, table = None, {}
    for so in sorted(refs_by_so):
        ys, fs, ms = [], [], []
        for fr in frames:
            D = np.array(fr["D_med"])
            Lr = luma_psd(cov_spec(fr), D, mono)
            m = refs_by_so[so][pick_level(refs_by_so[so], D)]
            Lm = luma_psd(cov_spec(m, subtract_q=False), np.array(m["D_med"]), mono)
            ys.append(np.log(np.clip(Lr[bins], 1e-12, None))); fs.append(np.array(fr["fc"])[bins]); ms.append(Lm[bins])
        nfr = len(frames)

        def model(p, i):
            a, su = fixed_usm if fixed_usm else (p[0], np.exp(p[1]))
            o = 0 if fixed_usm else 2
            K, N = np.exp(p[o + 2 * i]), np.exp(p[o + 2 * i + 1])
            return np.log(usm_tf(fs[i], a, su) ** 2 * (K * ms[i] + N))

        def res(p):
            return np.concatenate([ys[i] - model(p, i) for i in range(nfr)])
        p0 = [] if fixed_usm else [0.5, np.log(1.0)]
        for i in range(nfr):
            p0 += [np.log(float(np.median(np.exp(ys[i]) / ms[i]))), np.log(max(np.exp(ys[i]).min() * 0.1, 1e-12))]
        lb = ([] if fixed_usm else [0.0, np.log(0.2)]) + [-np.inf] * (2 * nfr)
        ub = ([] if fixed_usm else [8.0, np.log(8.0)]) + [np.inf] * (2 * nfr)
        sol = least_squares(res, p0, bounds=(lb, ub))
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        a, su = fixed_usm if fixed_usm else (float(sol.x[0]), float(np.exp(sol.x[1])))
        o = 0 if fixed_usm else 2
        # 白噪声底 N 占 0.11–0.5 周/像素的真实谱的比例 (扫描仪 / JPEG 的非颗粒部分)
        nshare = [float(np.mean(np.exp(sol.x[o + 2 * i + 1]) / (np.exp(sol.x[o + 2 * i]) * ms[i] + np.exp(sol.x[o + 2 * i + 1]))))
                  for i in range(nfr)]
        # 每帧在 2 × 2 盒 (40 µm, σ_48 的口径) 与单像素上, 白噪声底 N 占真实方差的比例 (模型谱在全部频档上外推)
        m_all = [luma_psd(cov_spec(refs_by_so[so][pick_level(refs_by_so[so], np.array(fr["D_med"]))], subtract_q=False),
                          np.array(refs_by_so[so][pick_level(refs_by_so[so], np.array(fr["D_med"]))]["D_med"]), mono) for fr in frames]
        mk, fc_all = _radial(TILE)
        fy = np.fft.fftfreq(TILE)[:, None]; fx = np.fft.fftfreq(TILE)[None, :]
        box2 = (np.cos(np.pi * fx) ** 2) * (np.cos(np.pi * fy) ** 2)
        w_box = np.array([(box2[mm] ** 2).sum() for mm in mk]); w_px = np.array([mm.sum() for mm in mk], float)
        sh_box, sh_px = [], []
        for i in range(nfr):
            K, Nn = np.exp(sol.x[o + 2 * i]), np.exp(sol.x[o + 2 * i + 1])
            U2 = usm_tf(fc_all, a, su) ** 2
            g, n = U2 * K * m_all[i], U2 * Nn
            sh_box.append(float((w_box * n).sum() / (w_box * (g + n)).sum())); sh_px.append(float((w_px * n).sum() / (w_px * (g + n)).sum()))
        r = dict(sigma_o_um=so, a=float(a), sigma_u_px=float(su), rms=rms, noise_floor_share=float(np.median(nshare)),
                 noise_var_share_box40=float(np.median(sh_box)), noise_var_share_pixel=float(np.median(sh_px)))
        table[so] = rms
        if best is None or rms < best["rms"]:
            best = r
    best["rms_by_sigma_o"] = table
    return best


# --------------------------------------------------------------- silvergrain 的参照
def reference(film, dev, evs=(-3, -2, -1, 0, 1, 2, 3), n=384, seed=5):
    """silvergrain full 模式 (固定定光, 不加冲印店扫描) 的均匀照片 → 每档输出; 之后按每个 σ_o 加 silvergrain 的光学。"""
    import silvergrain as sg
    out = {}
    for ev in evs:
        r = sg.apply_to_image(np.full((n, n, 3), 0.18), film, dev, format=n * PX / 1000.0, mode="full", timing="fixed",
                              exposure_ev=ev, scanner=None, seed=seed, method="explicit")
        out[ev] = r.linear if r.linear.ndim == 3 else np.repeat(r.linear[..., None], 3, -1)
    return out


def refs_by_sigma(levels):
    from ..image import _gauss, srgb_encode
    res = {}
    for so in SIGMAS_UM:
        res[so] = {}
        for ev, o in levels.items():
            oo = np.stack([_gauss(o[..., c], so / PX) for c in range(3)], -1) if so > 0 else o
            res[so][ev] = measure_array(srgb_encode(np.clip(oo, 0, None)) * 255.0)
    return res


def run(log=print):
    rc = json.load(gzip.open(DATA / "spec_rpl_c.json.gz", "rt"))
    rb = json.load(gzip.open(DATA / "spec_rpl_bw.json.gz", "rt"))
    fam = lambda d, films: [v for k, v in d.items() if k.split("/")[0] in films and v["n"] >= 40]
    ref_c = refs_by_sigma(reference("kodak_portra_400", None)); log("  scanner_fit ref Portra 400")
    out = {}
    gt = fit_grain_luma(fam(rc, TRAIN), ref_c)
    gh = fit_grain_luma(fam(rc, HOLD), ref_c)
    eg = fit_grain_luma(fam(rc, HOLD), ref_c, fixed_usm=USM_EDGES)
    a0 = fit_grain_luma(fam(rc, HOLD), ref_c, fixed_usm=(0.0, 1.0))
    out["colour_train_free"] = gt
    out["colour_hold_free"] = gh
    out["edge(train)→grain(hold)"] = eg
    out["colour_hold_no_usm"] = a0
    so = eg["sigma_o_um"]
    for film, (key, dev) in BW.items():
        ref = refs_by_sigma(reference(key, dev)); log(f"  scanner_fit ref {key}")
        frs = fam(rb, (film,))
        out[f"bw_holdout_{film}"] = dict(
            at_edge_usm=fit_grain_luma(frs, ref, mono=True, fixed_usm=USM_EDGES),
            at_colour_sigma_o=fit_grain_luma(frs, {so: ref[so]}, mono=True, fixed_usm=USM_EDGES),
            free=fit_grain_luma(frs, ref, mono=True), no_usm=fit_grain_luma(frs, ref, mono=True, fixed_usm=(0.0, 1.0)))
    out["sigma_o_um"] = so
    out["optics_px_at_20um"] = so / PX
    out["usm"] = dict(a=USM_EDGES[0], sigma_u_px=USM_EDGES[1], source="T51 直边 (训练片 85 条), tools/eval/t51/fit.json")
    return out
