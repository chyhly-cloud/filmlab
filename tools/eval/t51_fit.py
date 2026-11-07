"""T51: 冲印店扫描空间传递的实测 (每家一组器件参数) 与交叉检验。

输入 (只含测量, 不含图像):
  · 真实扫描的颗粒谱 (t51_scan_measure.py) 与直边 ESF (t51_edges.py), 在 scratchpad;
  · 模型的胶片侧颗粒参照 (t51_model_ref.py: 均匀灰场, 细像素, 不含扫描光学)。
器件模型 (filmlab.scanning.spatial):
  亮度传递 H_L(f) = O(f; σ_o) · A(f; 像素) · U(f; a, σ_u)，U = 1 + a·(1 − e^{−2π²σ_u²f²})
  色度传递 H_C(f) = O · A · e^{−2π²σ_c²f²} · J(f)            (J: JPEG 4:2:0 色度二次采样, 只用于 RPL 彩色)
两种独立的量:
  (a) 直边 ESF 的过冲 / 下冲 (只由 U 决定) 与色度边相对亮度边的展宽 (只由色度低通决定);
  (b) 颗粒功率谱: 真实 / 模型胶片侧 (同像素间距、同孔径与混叠) 的亮度谱比 = K·|O·U|², 与按频率的通道相干 (只由 H_C / H_L 决定)。
用法: python3 -I t51_fit.py 工作目录 输出.json
  工作目录: 测量 (spec_epfl / spec_rpl_c / spec_rpl_bw / edges_epfl_chart / edges_rpl, .json 或 .json.gz; 仓库 tools/eval/t51/ 有)
  与模型参照 (ref_800t.npz、ref_p400.npz、ref_trix.npz、ref_hp5.npz, 由 t51_model_ref.py 生成, 不进仓库)。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import t51_esf  # noqa: E402
import t51_scan_measure as SM  # noqa: E402
from filmlab.scanning import spatial  # noqa: E402

YCC = spatial._YCC
W601 = YCC[0]


# ------------------------------------------------------------------ 频率域的器件传递
def usm_tf(f, a, su):
    return 1.0 + a * (1.0 - np.exp(-2 * np.pi ** 2 * su ** 2 * f ** 2))


def gauss_tf(f, s):
    return np.exp(-2 * np.pi ** 2 * s ** 2 * f ** 2)


def jpeg420_tf(fc, T=SM.TILE):
    """JPEG 4:2:0: 2×2 平均下采样 + 线性上采样, 每轴约 cos²(πf); 按径向档平均。"""
    m, _ = SM._radial(T)
    fy = np.fft.fftfreq(T)[:, None]; fx = np.fft.fftfreq(T)[None, :]
    h = (np.cos(np.pi * fx) ** 2) * (np.cos(np.pi * fy) ** 2)
    return np.array([h[mm].mean() for mm in m])


# ------------------------------------------------------------------ 谱的整理
def cov_spec(m, subtract_q=True):
    """measure_array 结果 → 每档 3×3 协方差 (Do), 扣去 8 位量化白噪声。"""
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
    """平均 Do → d(码值 0–1)/dDo (每通道), 把 Do 域的小噪声换到编码域。"""
    lin = 10.0 ** -np.asarray(D)
    v = spatial.encode(lin)
    dv = np.where(lin <= 0.0031308, 12.92, 1.055 / 2.4 * np.power(np.maximum(lin, 1e-9), 1 / 2.4 - 1))
    return -dv * lin * np.log(10)          # dv/dDo


def luma_psd(C, D, mono=False):
    if mono:
        return C[:, 1, 1]
    s = code_slopes(D)
    w = W601 * s
    return np.einsum("i,fij,j->f", w, C, w)


def process_cov(C, D, u, c):
    """Do 域协方差经编码域 BT.601 亮度 (×u) / 色度 (×c) 处理后的协方差 (小信号, 局部线性)。"""
    S = np.diag(code_slopes(D))
    Mi = np.linalg.inv(YCC)
    out = np.empty_like(C)
    for k in range(len(C)):
        A = np.linalg.inv(S) @ Mi @ np.diag([u[k], c[k], c[k]]) @ YCC @ S
        out[k] = A @ C[k] @ A.T
    return out


def coherence(C):
    d = np.sqrt(np.clip(np.einsum("fii->fi", C), 1e-30, None))
    return np.stack([C[:, 0, 1] / d[:, 0] / d[:, 1], C[:, 1, 2] / d[:, 1] / d[:, 2], C[:, 0, 2] / d[:, 0] / d[:, 2]], 1)


# ------------------------------------------------------------------ 模型的胶片侧参照
def model_reference(npz, fine_um, pitch_um, sigma_o_um, tile=SM.TILE):
    """模型输出 (细像素) → 扫描光学 (高斯 σ_o, 小信号近似在线性输出上) → 冲印店像素方盒积分 → 每档谱。"""
    z = np.load(npz)
    k = int(round(pitch_um / fine_um))
    res = {}
    for key in z.files:
        o = z[key].astype(float)
        if sigma_o_um > 0:
            s = sigma_o_um / fine_um
            o = np.stack([gaussian_filter(o[..., c], s, mode="wrap") for c in range(3)], -1)
        h = (o.shape[0] // k) * k
        o = o[:h, :h].reshape(h // k, k, h // k, k, 3).mean((1, 3))
        v = spatial.encode(o) * 255.0
        res[key] = SM.measure_array(v, pitch_um, tile, synthetic=True)
    return res


def pick_level(ref, D):
    return min(ref, key=lambda k: abs(ref[k]["D_med"][1] - D[1]))


# ------------------------------------------------------------------ 颗粒: 亮度谱比
FIT_BINS = slice(2, 15)


def grain_ratio(real, refs_by_so, mono=False):
    """每个 σ_o 的 (频率, 真实/模型 亮度谱比)。real: measure_array 结果 (单帧或合并)。"""
    out = {}
    D = np.array(real["D_med"])
    Lr = luma_psd(cov_spec(real), D, mono)
    for so, ref in refs_by_so.items():
        m = ref[pick_level(ref, D)]
        Lm = luma_psd(cov_spec(m, subtract_q=False), np.array(m["D_med"]), mono)
        out[so] = Lr / Lm
    return np.array(real["fc"]), out


def fit_grain_luma(frames, refs_by_so, mono=False, fixed_usm=None, bins=FIT_BINS):
    """frames: [measure_array 结果]; 真实亮度谱 = |U|²·(K·模型胶片侧亮度谱 + N)。
    N: 扫描仪传感器白噪声 (在软件处理之前; 每帧自由, ≥ 0); K: 每帧的影调斜率 × 颗粒量级 (自由)。
    σ_o 在参照网格上逐个试, (a, σ_u) 连续拟合 (或 fixed_usm 固定)。返回最优与各 σ_o 的残差。"""
    best, table = None, {}
    for so in sorted(refs_by_so):
        ys, fs, ms = [], [], []
        for fr in frames:
            D = np.array(fr["D_med"])
            Lr = luma_psd(cov_spec(fr), D, mono)
            m = refs_by_so[so][pick_level(refs_by_so[so], D)]
            Lm = luma_psd(cov_spec(m, subtract_q=False), np.array(m["D_med"]), mono)
            fc = np.array(fr["fc"])
            ys.append(np.log(np.clip(Lr[bins], 1e-12, None))); fs.append(fc[bins]); ms.append(Lm[bins])
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
            k0 = float(np.median(np.exp(ys[i]) / ms[i]))
            p0 += [np.log(k0), np.log(max(np.exp(ys[i]).min() * 0.1, 1e-12))]
        lb = ([] if fixed_usm else [0.0, np.log(0.2)]) + [-np.inf] * (2 * nfr)
        ub = ([] if fixed_usm else [8.0, np.log(8.0)]) + [np.inf] * (2 * nfr)
        sol = least_squares(res, p0, bounds=(lb, ub))
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        a, su = (fixed_usm if fixed_usm else (float(sol.x[0]), float(np.exp(sol.x[1]))))
        o = 0 if fixed_usm else 2
        r = dict(sigma_o_um=so, a=float(a), sigma_u_px=float(su), rms=rms,
                 logK=sol.x[o::2].tolist(), logN=sol.x[o + 1::2].tolist(), resid=sol.fun.tolist())
        table[so] = rms
        if best is None or rms < best["rms"]:
            best = r
    best["rms_by_sigma_o"] = table
    return best


# ------------------------------------------------------------------ 颗粒: 按频率的通道相干 → 色度 / 亮度之比
def fit_chroma_from_coherence(frames, u_of_f, jpeg=False, bins=slice(1, 15), fixed_sigma=None, sigma0=2.0, shapes=None):
    """真实协方差 C_real(f) = g(f)·A(f) C0 A(f)ᵀ + N, A = 编码域 diag(U, c, c) (亮度 USM、色度低通),
    N = 每通道独立的白噪声底 (扫描仪传感器 / 输出, 对角, 每帧 3 个)。C0 (该冲印店低频的颜色协方差, 每帧 6 个数)
    g(f): 公共谱形, 取模型胶片侧颗粒经扫描光学与孔径的亮度谱 (shapes; 物理上单调下降, 不能自由取形状, 否则 g 的升降
    与白噪声底可以凑出任意相干); c(f) = e^{−2π²σ_c²f²}·J(f)。拟合 σ_c (像素)。"""
    fc = np.array(frames[0]["fc"])
    J = jpeg420_tf(fc) if jpeg else np.ones_like(fc)
    data = [(cov_spec(fr)[bins], np.array(fr["D_med"])) for fr in frames]
    f = fc[bins]
    nb = len(f)

    def unpack(p, i):
        q = p[1 + 13 * i: 1 + 13 * (i + 1)]
        L = np.zeros((3, 3)); L[np.tril_indices(3)] = q[:6]
        return L @ L.T, np.exp(q[6:9]), q[9:13]

    def model(p, i, D):
        C0, N, gc = unpack(p, i)
        c = gauss_tf(f, np.exp(p[0]) if fixed_sigma is None else fixed_sigma) * J[bins]
        if shapes is not None:                             # 公共谱形 = 模型胶片侧颗粒经扫描光学与孔径 (单调下降, 已知)
            g = np.asarray(shapes[i])[bins]
        else:                                              # 无参照时: 三次多项式 (log), 只作诊断
            g = np.exp(np.polyval(gc, f))
        Cm = process_cov(g[:, None, None] * C0[None], D, u_of_f(f), c)
        Cm[:, [0, 1, 2], [0, 1, 2]] += N[None, :]
        return Cm

    def res(p):
        out = []
        for i, (C, D) in enumerate(data):
            Cm = model(p, i, D)
            # 相干 (与各通道量级无关) + 各通道谱形 (对数, 去掉每通道的整体量级)
            out.append((coherence(C) - coherence(Cm)).ravel())
            dr = np.log(np.clip(np.einsum("fii->fi", C), 1e-30, None)); dm = np.log(np.einsum("fii->fi", Cm))
            out.append(0.25 * ((dr - dr.mean(0)) - (dm - dm.mean(0))).ravel())
        return np.concatenate(out)
    p0 = [np.log(sigma0)]
    for i, (C, D) in enumerate(data):
        g0 = float(np.asarray(shapes[i])[bins][0]) if shapes is not None else 1.0
        L = np.linalg.cholesky(C[0] / g0 + 1e-9 * np.trace(C[0]) * np.eye(3))
        p0 += L[np.tril_indices(3)].tolist() + np.log(np.einsum("ii->i", C[-1]) * 0.5 + 1e-12).tolist() + [0, 0, 0, 0]
    lb = np.full(len(p0), -np.inf); ub = np.full(len(p0), np.inf)
    lb[0], ub[0] = np.log(0.05), np.log(20.0)
    for i in range(len(data)):
        lb[1 + 13 * i + 6: 1 + 13 * i + 9] = -60.0; ub[1 + 13 * i + 6: 1 + 13 * i + 9] = 0.0
    p0 = np.clip(p0, lb + 1e-6, ub - 1e-6)
    sol = least_squares(res, p0, x_scale="jac", max_nfev=3000, bounds=(lb, ub))
    out = dict(sigma_c_px=float(np.exp(sol.x[0])) if fixed_sigma is None else float(fixed_sigma), rms=float(np.sqrt(np.mean(sol.fun ** 2))), frames=[])
    for i, (C, D) in enumerate(data):
        C0, N, gc = unpack(sol.x, i)
        out["frames"].append(dict(C0=C0.tolist(), N=N.tolist(), coh_fit=coherence(model(sol.x, i, D)).tolist()))
    return out


def coherence_prediction(fr, sigma_c_px, u_of_f, C0, jpeg=False):
    fc = np.array(fr["fc"])
    J = jpeg420_tf(fc) if jpeg else np.ones_like(fc)
    r = gauss_tf(fc, sigma_c_px) * J / u_of_f(fc)
    Cm = process_cov(np.repeat(np.asarray(C0)[None], len(fc), 0), np.array(fr["D_med"]), np.ones_like(fc), r)
    return coherence(Cm)


# ------------------------------------------------------------------ 直边
def edge_list(edges_json, keys, channel=0, min_step=40.0):
    E = []
    for k in keys:
        for e in edges_json[k]["edges"]:
            if e["step"] < min_step:
                continue
            y = np.array(e["esf"], float)
            if channel == "Y601":
                v = y[:, 1:4] @ W601
            else:
                v = y[:, channel]
            E.append((np.array(e["d"]), v, k))
    return E


def fit_usm_edges(E, halfwidth, fixed=None, prune=0.2, se_max=3.0):
    """联合拟合 (a, σ_u)。第一遍后去掉不像台阶的 (σ_e ≥ se_max 像素: 渐变而非边) 与残差最大的 prune 比例, 再拟合。"""
    pairs = [(d, v) for d, v, _ in E]
    f = t51_esf.fit_edges(pairs, shared_fixed=fixed, fit_halfwidth=halfwidth)
    if prune is not None:
        rms = np.array([e["rms"] for e in f["edges"]]); se = np.array([e["sigma_e"] for e in f["edges"]])
        keep = (se < se_max) & (rms <= np.quantile(rms, 1 - prune))
        pairs = [p for p, k in zip(pairs, keep) if k]
        E = [e for e, k in zip(E, keep) if k]
        f = t51_esf.fit_edges(pairs, shared_fixed=fixed, fit_halfwidth=halfwidth)
    f["n_used"] = len(pairs)
    f["E"] = E
    return f


def edge_overshoots(E, fit):
    """每条边的实测过冲 / 下冲 (相对台阶) 与拟合 σ_e, 供"由颗粒预测直边"比较。"""
    out = []
    for (d, v, _), p in zip(E, fit["edges"]):
        o_hi, o_lo = t51_esf.overshoot_metrics(d, v, p["lo"], p["step"], p["d0"])
        out.append((p["sigma_e"], o_hi, o_lo))
    return np.array(out)


def predicted_overshoot(sigma_e, a, su, halfwidth=5.0):
    d = np.arange(-8, 8, 0.25)
    y = t51_esf.esf_shape(d, 0.0, sigma_e, 0.0, 0.0, a, su)
    hi = (d > 0) & (d <= halfwidth); lo = (d < 0) & (d >= -halfwidth)
    return float(y[hi].max() - 1.0), float(-y[lo].min())


# ------------------------------------------------------------------ 主流程
def _load(work, name):
    """测量文件: 工作目录里的 name 或 name.gz (仓库 tools/eval/t51/ 存 .gz)。"""
    import gzip
    p = Path(work) / name
    return json.load(open(p)) if p.exists() else json.load(gzip.open(str(p) + ".gz", "rt"))

FRONTIER_SO = (2.0, 3.0, 4.0, 4.9, 6.0, 7.0, 8.0)
NORITSU_SO = (0.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0)
F_REPORT = (0.125, 0.25, 0.5)


def _refs(work, tag, npz, fine, pitch, sos):
    import pickle
    p = Path(work) / f"refspec_{tag}.pkl"
    have = pickle.load(open(p, "rb")) if p.exists() else {}
    for so in sos:
        if so not in have:
            have[so] = model_reference(Path(work) / npz, fine, pitch, so)
    pickle.dump(have, open(p, "wb"))
    return {so: have[so] for so in sos}


def _usm_summary(a, su):
    return {f"U({f})": float(usm_tf(np.array(f), a, su)) for f in F_REPORT}


def _edge_block(E, halfwidth):
    f = fit_usm_edges(E, halfwidth)
    f0 = fit_usm_edges(f["E"], halfwidth, fixed=(0.0, 1.0), prune=None)
    ov = edge_overshoots(f["E"], f)
    return f, f0, ov


def _edge_report(f, f0, ov):
    se = np.array([e["sigma_e"] for e in f["edges"]])
    return dict(n=f["n_used"], a=f["a"], sigma_u_px=f["sigma_u_px"], cost=f["cost"], cost_a0=f0["cost"],
                sigma_e_px_q=np.quantile(se, [0.1, 0.5, 0.9]).tolist(), **_usm_summary(f["a"], f["sigma_u_px"]),
                overshoot_hi_med=float(np.median(ov[:, 1])), overshoot_lo_med=float(np.median(ov[:, 2])))


def _pred_edges(E_fit, fixed, halfwidth):
    """固定 (a, σ_u) 拟合这些边 (每条边的 σ_e、平台自由), 比较代价与过冲。"""
    E = E_fit["E"]
    f = fit_usm_edges(E, halfwidth, fixed=fixed, prune=None)
    pred = np.array([predicted_overshoot(e["sigma_e"], fixed[0], fixed[1]) for e in f["edges"]])
    ov = edge_overshoots(E, E_fit)
    return dict(cost=f["cost"], cost_free=E_fit["cost"], overshoot_hi_pred_med=float(np.median(pred[:, 0])),
                overshoot_lo_pred_med=float(np.median(pred[:, 1])), overshoot_hi_meas_med=float(np.median(ov[:, 1])),
                overshoot_lo_meas_med=float(np.median(ov[:, 2])))


def model_shapes(frames, refs, so, mono=False):
    out = []
    for fr in frames:
        m = refs[so][pick_level(refs[so], np.array(fr["D_med"]))]
        L = luma_psd(cov_spec(m, subtract_q=False), np.array(m["D_med"]), mono)
        out.append(L / L[1])
    return out


def chroma_profile(frames, u_of_f, jpeg=False, grid=(0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0), shapes=None):
    rms = []
    for s in grid:
        r = fit_chroma_from_coherence(frames, u_of_f, jpeg, fixed_sigma=s, sigma0=s, shapes=shapes)
        rms.append(r["rms"])
    rms = np.array(rms); i = int(np.argmin(rms))
    ok = rms <= rms.min() * 1.05
    return dict(grid=list(grid), rms=rms.tolist(), best=float(grid[i]),
                range=[float(np.array(grid)[ok].min()), float(np.array(grid)[ok].max())])


def chroma_edges(edges_json, halfwidth=5.0, min_chroma=30.0):
    """强色度边: 亮度 (Y601) 与色度 (Cb / Cr 中台阶大者) 的 ESF 宽度 σ_Y、σ_C (a = 0)。"""
    rows = []
    for k, v in edges_json.items():
        for e in v["edges"]:
            y = np.array(e["esf"], float)[:, 1:4]; d = np.array(e["d"])
            ycc = y @ YCC.T
            st = np.nanmean(ycc[-10:], 0) - np.nanmean(ycc[:10], 0)
            j = 1 + int(np.argmax(np.abs(st[1:])))
            if abs(st[j]) < min_chroma:
                continue
            fy = t51_esf.fit_edges([(d, ycc[:, 0])], shared_fixed=(0.0, 1.0), fit_halfwidth=halfwidth)["edges"][0]
            fc = t51_esf.fit_edges([(d, np.sign(st[j]) * ycc[:, j])], shared_fixed=(0.0, 1.0), fit_halfwidth=halfwidth)["edges"][0]
            rows.append(dict(frame=k, dY=float(st[0]), dC=float(st[j]), sigma_Y=fy["sigma_e"], sigma_C=fc["sigma_e"]))
    return rows


def main():
    work, dst = Path(sys.argv[1]), Path(sys.argv[2])
    out = {}
    # ================= Frontier (EPFL) =================
    ep = _load(work, "spec_epfl.json")
    ep = {k: v for k, v in ep.items() if v["n"] >= 40}
    refs_f = _refs(work, "800t", "ref_800t.npz", 2.2, 6.6, FRONTIER_SO)
    ee = _load(work, "edges_epfl_chart.json")
    E = edge_list(ee, list(ee), channel="Y601")
    fE, fE0, ovE = _edge_block(E, 5.0)
    g_all = fit_grain_luma(list(ep.values()), refs_f)
    g_a0 = fit_grain_luma(list(ep.values()), refs_f, fixed_usm=(fE["a"], fE["sigma_u_px"]) if fE["a"] > 0 else (0.0, 1.0))
    per = {k: {kk: vv for kk, vv in fit_grain_luma([v], refs_f).items() if kk in ("sigma_o_um", "a", "sigma_u_px", "rms")}
           for k, v in ep.items()}
    charts = [k for k in ep if any(s in k for s in ("R1-17-18", "R1-18-19", "R1-19-20", "R1-20-21"))]
    nights = [k for k in ep if k not in charts]
    one = lambda f: np.ones_like(f)
    so_f = g_a0["sigma_o_um"]
    shp = lambda ks: model_shapes([ep[k] for k in ks], refs_f, so_f)
    chp = {k: chroma_sigma([ep[k]], shp([k]), one) for k in ep}
    ch_train = chroma_sigma([ep[k] for k in charts], shp(charts), one)
    ch_hold_free = chroma_sigma([ep[k] for k in nights], shp(nights), one)
    i_tr = ch_train["grid"].index(ch_train["best"])
    ch_hold_fixed = dict(rms=ch_hold_free["rms"][i_tr], sigma_c=ch_train["best"])
    pred_coh = {k: dict(real=np.array(ep[k]["coh"]).tolist(),
                        pred=coherence(chroma_forward(ep[k], shp([k])[0], ch_train["best"], one)).tolist(),
                        none=coherence(chroma_forward(ep[k], shp([k])[0], 0.0, one)).tolist()) for k in nights}
    cedges = chroma_edges(ee)
    out["frontier"] = dict(
        pitch_um=6.6,
        edges=_edge_report(fE, fE0, ovE),
        grain_luma=dict(free={k: g_all[k] for k in ("sigma_o_um", "a", "sigma_u_px", "rms", "rms_by_sigma_o")},
                        edge_usm={k: g_a0[k] for k in ("sigma_o_um", "a", "sigma_u_px", "rms", "rms_by_sigma_o")},
                        per_frame=per, **{"grain→edge": _pred_edges(fE, (g_all["a"], g_all["sigma_u_px"]), 5.0)}),
        chroma=dict(per_frame=chp, train_charts=ch_train, holdout_nights_at_train_sigma=ch_hold_fixed,
                    holdout_nights_free=ch_hold_free, holdout_coherence=pred_coh, edges=cedges))
    print("frontier", json.dumps(out["frontier"]["edges"]), out["frontier"]["grain_luma"]["free"], flush=True)
    # ================= Noritsu (RPL) =================
    rc = _load(work, "spec_rpl_c.json")
    rb = _load(work, "spec_rpl_bw.json")
    er = _load(work, "edges_rpl.json")
    refs_n = _refs(work, "p400", "ref_p400.npz", 5.0, 20.0, NORITSU_SO)
    TRAIN = ("1_FUJIPRO400H", "2_KODAKPORTRA400", "4_KODAKPORTRA160")
    HOLD = ("3_KODAKPORTRA800", "5_KODAKEKTAR100")
    BW = ("6_KODAK400TX", "7_ILFORDHP5PLUS")
    fam = lambda d, films: [v for k, v in d.items() if k.split("/")[0] in films and v["n"] >= 40]
    ekeys = lambda films: [k for k in er if k.split("/")[0] in films]
    res = {}
    edges = {}
    for name, films in (("train", TRAIN), ("hold", HOLD), ("bw", BW)):
        f, f0, ov = _edge_block(edge_list(er, ekeys(films), channel="Y601"), 4.0)
        edges[name] = f
        res[f"edges_{name}"] = _edge_report(f, f0, ov)
    gt = fit_grain_luma(fam(rc, TRAIN), refs_n)
    gh = fit_grain_luma(fam(rc, HOLD), refs_n)
    keep = ("sigma_o_um", "a", "sigma_u_px", "rms", "rms_by_sigma_o")
    res["grain_train"] = {k: gt[k] for k in keep} | _usm_summary(gt["a"], gt["sigma_u_px"])
    res["grain_hold_free"] = {k: gh[k] for k in keep} | _usm_summary(gh["a"], gh["sigma_u_px"])
    ft = edges["train"]
    # 直边 (训练片) → 颗粒 (留出片): 固定 USM, 只放开 σ_o、K、N
    eg = fit_grain_luma(fam(rc, HOLD), refs_n, fixed_usm=(ft["a"], ft["sigma_u_px"]))
    a0 = fit_grain_luma(fam(rc, HOLD), refs_n, fixed_usm=(0.0, 1.0))
    res["edge(train)→grain(hold)"] = dict(rms=eg["rms"], sigma_o_um=eg["sigma_o_um"], rms_free=gh["rms"], rms_no_usm=a0["rms"])
    # 颗粒 (训练片) → 直边 (留出片)
    res["grain(train)→edge(hold)"] = _pred_edges(edges["hold"], (gt["a"], gt["sigma_u_px"]), 4.0)
    res["grain(train)→edge(hold)"]["cost_no_usm"] = fit_usm_edges(edges["hold"]["E"], 4.0, fixed=(0.0, 1.0), prune=None)["cost"]
    # 黑白留出 (跨片种): 彩色训练片定的 (σ_o, a, σ_u) 预测 Tri-X / HP5 的颗粒谱形
    for film, npz in (("6_KODAK400TX", "ref_trix.npz"), ("7_ILFORDHP5PLUS", "ref_hp5.npz")):
        if not (work / npz).exists():
            continue
        rbw = _refs(work, npz[4:-4], npz, 5.0, 20.0, NORITSU_SO)
        frs = fam(rb, (film,))
        free = fit_grain_luma(frs, rbw, mono=True)
        pred = fit_grain_luma(frs, {gt["sigma_o_um"]: rbw[gt["sigma_o_um"]]}, mono=True, fixed_usm=(gt["a"], gt["sigma_u_px"]))
        pe = fit_grain_luma(frs, rbw, mono=True, fixed_usm=(ft["a"], ft["sigma_u_px"]))
        none = fit_grain_luma(frs, rbw, mono=True, fixed_usm=(0.0, 1.0))
        res[f"bw_holdout_{film}"] = dict(free={k: free[k] for k in keep} | _usm_summary(free["a"], free["sigma_u_px"]),
                                         colour_grain_pred_rms=pred["rms"], colour_edge_usm_rms=pe["rms"],
                                         colour_edge_usm_sigma_o=pe["sigma_o_um"], no_usm_rms=none["rms"])
    res["bw_edges_at_colour_grain_usm"] = _pred_edges(edges["bw"], (gt["a"], gt["sigma_u_px"]), 4.0)
    # 色度 (JPEG 4:2:0 已计入)
    u_n = lambda f: usm_tf(f, ft["a"], ft["sigma_u_px"])
    so_n = eg["sigma_o_um"]
    for name, films in (("train", TRAIN), ("hold", HOLD)):
        frs = fam(rc, films)
        res[f"chroma_{name}"] = chroma_sigma(frs, model_shapes(frs, refs_n, so_n), u_n, jpeg=True, bins=slice(2, 9))
    # 颗粒 (留出片) → 直边 (训练片: 清晰的边多) —— 反方向
    res["grain(hold)→edge(train)"] = _pred_edges(edges["train"], (gh["a"], gh["sigma_u_px"]), 4.0)
    res["grain(hold)→edge(train)"]["cost_no_usm"] = fit_usm_edges(edges["train"]["E"], 4.0, fixed=(0.0, 1.0), prune=None)["cost"]
    res["chroma_edges"] = chroma_edges({k: er[k] for k in ekeys(TRAIN + HOLD)}, halfwidth=4.0, min_chroma=25.0)
    out["noritsu"] = dict(pitch_um=20.0, **res)
    json.dump(out, open(dst, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if "chroma" not in k}, indent=1, default=float))



def _A(D, u, c):
    S = np.diag(code_slopes(D))
    return np.linalg.inv(S) @ np.linalg.inv(YCC) @ np.diag([u, c, c]) @ YCC @ S


def chroma_forward(fr, shape, sigma_c, u_of_f, jpeg=False, ref_bin=1, n_iter=3):
    """确定性前向模型 (只有 σ_c 一个参数): 处理前的颜色协方差 C0 由第 ref_bin 档 (最低的可靠频带) 反解 (A⁻¹ (C − N) A⁻ᵀ / g),
    N (处理后的每通道白噪声底, 对角) 由最高 3 档的对角残差定; 返回各档预测协方差。"""
    fc = np.array(fr["fc"]); D = np.array(fr["D_med"])
    C = cov_spec(fr)
    J = jpeg420_tf(fc) if jpeg else np.ones_like(fc)
    u, c = u_of_f(fc), gauss_tf(fc, sigma_c) * J
    A = [_A(D, u[k], c[k]) for k in range(len(fc))]
    N = np.zeros(3)
    for _ in range(n_iter):
        Ai = np.linalg.inv(A[ref_bin])
        C0 = Ai @ (C[ref_bin] - np.diag(N)) @ Ai.T / shape[ref_bin]
        Cm = np.array([shape[k] * A[k] @ C0 @ A[k].T for k in range(len(fc))])
        N = np.clip(np.mean(np.einsum("fii->fi", C[-3:] - Cm[-3:]), 0), 0, None)
    return Cm + np.diag(N)[None]


def chroma_sigma(frames, shapes, u_of_f, jpeg=False, grid=np.arange(0.0, 6.01, 0.25), bins=slice(2, 12), ref_bin=1):
    """σ_c 剖面: 各帧按频率的通道相干 (bins) 的 RMS 误差; 最小处与 +10% 区间。"""
    rms = []
    for s in grid:
        e = []
        for fr, sh in zip(frames, shapes):
            Cm = chroma_forward(fr, sh, s, u_of_f, jpeg, ref_bin)
            e.append((coherence(Cm) - np.array(fr["coh"]))[bins])
        rms.append(float(np.sqrt(np.mean(np.square(e)))))
    rms = np.array(rms); i = int(np.argmin(rms)); ok = rms <= rms.min() * 1.1
    return dict(grid=grid.tolist(), rms=rms.tolist(), best=float(grid[i]), range=[float(grid[ok].min()), float(grid[ok].max())],
                rms_best=float(rms[i]), rms_none=float(rms[0]))


if __name__ == "__main__":
    main()
