"""T47: Tri-X 400 银盐颗粒 —— 数据、几何检验、密度审计、标定、颗粒验证、团聚判定、FX39 留出 (docs/tasks/T47_TRIX_SILVER_GRAIN.md)。

用法: python3 tools/eval/t47_trix.py [--sim N]    (N = 每个配置的显微视场模拟数, 默认 4; 0 = 只算闭式解)
输出: tools/eval/t47/results.json, tools/eval/t47/calib.json
门槛 (事先写定): 19 级 |ΔD| ≤ 0.10; Schmidt 四孔径 RMS 都在 ±15% 以内。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab import grain, materials  # noqa: E402

D = Path(__file__).resolve().parent / "t47"
GATE_D, GATE_RMS = 0.10, 0.15
SCHMIDT_PX = 0.25                       # Schmidt 显微 RAW: 4x4 像素 ≈ 1 µm
FOG = 0.318                             # 片基 + 灰雾 (Codex 读图口径)


def csv(name):
    rows = [l for l in (D / name).read_text().splitlines() if l and not l.startswith("#")]
    head = rows[0].split(",")
    return {h: np.array([float(r.split(",")[i]) if r.split(",")[i] else np.nan for r in rows[1:]]) for i, h in enumerate(head)}


def schmidt_ak(sides, s2a):
    """Schmidt 的两参数式 σ²A(a) = c² a / (a + a_k) (a = 边长²) → (c², a_k)。"""
    f = lambda a, c2, ak: c2 * a / (a + ak)
    (c2, ak), _ = curve_fit(f, np.asarray(sides) ** 2, np.asarray(s2a), p0=(0.3, 20.0))
    return float(c2), float(ak)


# ---------------------------------------------------------------- A. 几何检验 --
def geometry_check(sd):
    k = csv("kodak_f4017_p8_d76.csv")
    x = k["log_H_lux_s"]
    curves = {t: k[f"gross_D_{t}min"] for t in (6, 8, 10, 12)}
    def kod(t):
        ts = np.array([6, 8, 10, 12.0]); i = int(np.clip(np.searchsorted(ts, t) - 1, 0, 2))
        w = (t - ts[i]) / (ts[i + 1] - ts[i])
        D = (1 - w) * curves[int(ts[i])] + w * curves[int(ts[i + 1])]
        ok = np.isfinite(D)
        return x[ok], D[ok] - np.nanmin(D[ok])
    def x01(xx, net):
        m = np.maximum.accumulate(net) + 1e-9 * np.arange(len(net)); return float(np.interp(0.10, m, xx))
    def gbar(xx, net, span=1.5):
        a = x01(xx, net); return float((np.interp(a + span, xx, net) - 0.10) / span)
    Es, net_s = sd["chart_relative_log10_exposure"], sd["net_D"]
    g_s = gbar(Es, net_s)
    ts = np.linspace(6, 12, 61); gk = np.array([gbar(*kod(t)) for t in ts])
    t_g = float(np.interp(g_s, gk, ts))
    out = dict(schmidt_gbar=g_s, kodak_gbar_9min=gbar(*kod(9.0)), t_gbar_match=t_g, cases={})
    for label, t in (("9min", 9.0), ("gbar", t_g)):
        xk, nk = kod(t)
        pred = np.interp(x01(xk, nk) + (Es - x01(Es, net_s)), xk, nk, right=np.nan)
        out["cases"][label] = dict(t_min=t, kodak_net=pred.tolist(), resid=(net_s - pred).tolist(),
                                   max_abs=float(np.nanmax(np.abs(net_s - pred))), log_H_net01=x01(xk, nk))
    c = csv("kodak_f4017_p8_d76_codex.csv")
    out["codex_points_vs_T47"] = {f"{t}min": (np.interp(c["log_H_lux_s"], x, curves[t]) - c[f"gross_D_{t}min"]).tolist()
                                  for t in (8, 10)}
    return out


# ---------------------------------------------------- B. 晶体模型的特性曲线 --
def H_levels(sd, ei=400.0):
    """Schmidt 对齐后的相对曝光 → 绝对曝光: 净 D 0.10 处 H = 0.8/EI (ISO 6; Schmidt 实测 EI 400)。"""
    return 0.8 / ei * 10.0 ** sd["aligned_log10_exposure"]


def spec_for(sd, eta_scale, morph=None, **kw):
    U = materials.grain_universal()
    return grain.silver_derive(400.0, U, eta_scale, H_levels(sd), sd["net_D"], FOG,
                               morph or materials.GRAIN_T47_MORPH, process="D-76 原液 20 °C 9 min", **kw)


def H_at_net(spec, net, geometry="diffuse"):
    lh = np.linspace(-6, 2, 4001)
    Dn = grain.silver_curve(spec, lh, geometry) - grain.silver_curve(spec, [-12.0], geometry)[0]
    return float(10 ** np.interp(net, np.maximum.accumulate(Dn + 1e-12 * np.arange(len(Dn))), lh))


def density_audit(spec, sd):
    H = H_levels(sd)
    dif = grain.silver_density(spec, H, "diffuse") - grain.silver_density(spec, 0.0, "diffuse")
    spe = grain.silver_density(spec, H, "specular") - grain.silver_density(spec, 0.0, "specular")
    ag = grain.silver_developed_mass(spec, H)
    b = grain.silver_bins(spec)
    rows = []
    for i in range(len(H)):
        p = grain.silver_develop(spec, H[i], b["a"])
        frac = [(b["n"][k] * p[k] * b["m"][k]).sum() / (b["n"][k] * b["m"][k]).sum() for k in range(3)]
        rows.append(dict(level=i + 1, log_H=float(np.log10(H[i])), net_meas=float(sd["net_D"][i]), net_diffuse=float(dif[i]),
                         resid=float(sd["net_D"][i] - dif[i]), net_specular=float(spe[i]), developed_ag_g_m2=float(ag[i]),
                         developed_fraction_by_subpop=[float(f) for f in frac]))
    bd, bs = grain.silver_budget(spec, "diffuse"), grain.silver_budget(spec, "specular")
    fog_d = float(grain.silver_density(spec, 0.0, "diffuse"))
    return dict(levels=rows, max_abs_resid=float(np.max(np.abs(sd["net_D"] - dif))), passes=bool(np.max(np.abs(sd["net_D"] - dif)) <= GATE_D),
                budget_diffuse=bd, budget_specular=bs, net_ceiling_diffuse=bd["D_max"] - fog_d,
                net_ceiling_specular=bs["D_max"] - float(grain.silver_density(spec, 0.0, "specular")),
                fog_diffuse=fog_d, base_density=spec.base_density, coat_by_subpop=list(spec.coat_g_m2),
                ecd_um=list(spec.ecd_um), crystals_per_um2=[float(v) for v in grain.silver_bins(spec)["n"].sum(1)])


# ------------------------------------------------------------- C. 标定 --
def model_estimator(spec, sides, psf, geometry):
    H = H_at_net(spec, 0.682, "diffuse")             # 颗粒视场: 密度计 (漫射) 总 D ≈ 1 → 净 0.682
    return grain.silver_granularity(spec, H, sides, psf_um=psf, geometry=geometry, adjacent=True)


def calibrate(sd, gr, geometry):
    """标定量 1: η_3D (三维晶体的有效量子效率 → 晶体尺寸) 使 20/25 µm 的 σ²A 均值 = 实测均值;
    标定量 2: 显微镜 PSF σ 使 Schmidt 两参数式的 a_k = 实测 a_k。两者交替求解。"""
    sides = gr["digitized_side_um"]; s2 = gr["sigma2_A_um2_D2"]
    target_pl = float(np.mean(s2[2:])); _, ak_meas = schmidt_ak(sides, s2)
    psf, eta = 1.0, 8.0
    for _ in range(6):
        def f_eta(le):
            sp = spec_for(sd, np.exp(le))
            r = model_estimator(sp, sides[2:], psf, geometry)
            return float(np.mean(r ** 2 * sides[2:] ** 2)) - target_pl
        eta = float(np.exp(brentq(f_eta, np.log(0.5), np.log(200.0), xtol=1e-3)))
        sp = spec_for(sd, eta)
        def f_psf(s):
            r = model_estimator(sp, sides, s, geometry)
            return schmidt_ak(sides, r ** 2 * sides ** 2)[1] - ak_meas
        try:
            psf_new = float(brentq(f_psf, 0.05, 3.0, xtol=1e-3))
        except ValueError:
            psf_new = psf
        if abs(psf_new - psf) < 2e-3:
            psf = psf_new
            break
        psf = psf_new
    return dict(eta_scale=eta, psf_um=psf, target_plateau=target_pl, ak_meas=ak_meas, geometry=geometry)


# ------------------------------------------------------ D. 显微视场模拟 --
def micro_field(spec, H, psf, seed, geometry, field_um=375.0, extra_um=None):
    extra = extra_um if extra_um is not None else 4 * psf + 4 + 4 * spec.cluster_um
    n = int(round((field_um + 2 * extra) / SCHMIDT_PX))
    T = grain.render_silver(spec, np.full((n, n), H), SCHMIDT_PX, seed, geometry=geometry, sub=2, count_sub=1, psf_um=psf)
    m = int(round(extra / SCHMIDT_PX)); k = int(round(field_um / SCHMIDT_PX))
    return T[m:m + k, m:m + k]


def schmidt_estimator(T, cell_px):
    """Schmidt: 视场切成边长 cell_px 的方孔, 单元密度 = −log10(单元平均透射率), 相邻单元 (水平 + 垂直) 差的 RMS / √2。"""
    n = T.shape[0] // cell_px * cell_px
    c = T[:n, :n].reshape(n // cell_px, cell_px, n // cell_px, cell_px).mean((1, 3))
    Dc = -np.log10(c)
    d = np.concatenate([np.diff(Dc, axis=0).ravel(), np.diff(Dc, axis=1).ravel()])
    return float(np.sqrt(np.mean(d ** 2) / 2)), Dc


def skew(x):
    x = np.asarray(x, float).ravel(); x = x - x.mean()
    return float(np.mean(x ** 3) / np.mean(x ** 2) ** 1.5)


def simulate(spec, psf, geometry, nfield, sides, base_seed=4700):
    H = H_at_net(spec, 0.682, "diffuse")
    cells = [int(round(s / SCHMIDT_PX)) for s in sides]
    rms, sk5, skpx, skT, dmean = [], [], [], [], []
    for i in range(nfield):
        T = micro_field(spec, H, psf, base_seed + i, geometry)
        r = []
        for c in cells:
            v, Dc = schmidt_estimator(T, c)
            r.append(v)
            if c == cells[0]:
                sk5.append(skew(Dc))
        rms.append(r)
        skpx.append(skew(-np.log10(T))); skT.append(skew(T)); dmean.append(float(-np.log10(T.mean())))
    rms = np.array(rms)
    return dict(rms_mean=rms.mean(0).tolist(), rms_sd=rms.std(0, ddof=1).tolist() if nfield > 1 else None,
                skew_5um_D=float(np.mean(sk5)), skew_pixel_D=float(np.mean(skpx)), skew_pixel_T=float(np.mean(skT)),
                field_D_total=float(np.mean(dmean)), H=H, cells_px=cells)


def disc_control(net_spec_D, psf, seed, tau, radius, sides, field_um=375.0):
    """对照 (颗粒外形之外的两种极端): 同一平均密度下 "少而不透明" (τ 大) 与 "多而淡" (τ 小) 的泊松圆盘场, 同一估计器。"""
    rng = np.random.default_rng(seed)
    extra = 4 * psf + 2; L = field_um + 2 * extra
    a = np.pi * radius ** 2
    lam = net_spec_D * np.log(10) / (a * (1 - np.exp(-tau)))
    N = rng.poisson(lam * L * L)
    d = SCHMIDT_PX / 2; n = int(round(L / d))
    cnt = np.zeros((n, n))
    xy = rng.random((N, 2)) * L
    r = int(np.ceil(radius / d))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            yy = np.floor(xy[:, 0] / d).astype(int) + dy; xx = np.floor(xy[:, 1] / d).astype(int) + dx
            inside = ((yy + 0.5) * d - xy[:, 0]) ** 2 + ((xx + 0.5) * d - xy[:, 1]) ** 2 <= radius ** 2
            ok = inside & (yy >= 0) & (yy < n) & (xx >= 0) & (xx < n)
            cnt += np.bincount(yy[ok] * n + xx[ok], minlength=n * n).reshape(n, n)
    from scipy.ndimage import gaussian_filter
    T = gaussian_filter(np.exp(-tau * cnt), psf / d, mode="nearest").reshape(n // 2, 2, n // 2, 2).mean((1, 3))
    m = int(round(extra / SCHMIDT_PX)); k = int(round(field_um / SCHMIDT_PX))
    T = T[m:m + k, m:m + k]
    rms = [schmidt_estimator(T, int(round(s / SCHMIDT_PX)))[0] for s in sides]
    return dict(tau=tau, radius_um=radius, rate_per_um2=float(lam), rms=rms,
                skew_5um_D=skew(schmidt_estimator(T, int(round(sides[0] / SCHMIDT_PX)))[1]), skew_pixel_D=skew(-np.log10(T)))


def skew_controls(spec, psf, G, sides):
    """同一平均密度、同一平台 σ²A 的两种圆盘场: "少而不透明" (τ = 30, r = 0.5 µm) 与 "多而淡" (τ = 0.05, 面积 ×20 使平台相同);
    只看 RMS 分不开, 看偏度。"""
    H = H_at_net(spec, 0.682)
    net = float(grain.silver_density(spec, H, G))
    return [disc_control(net, psf, 91, tau, rad, sides) for tau, rad in ((30.0, 0.5), (0.05, 0.5 * np.sqrt(20.0)))]


# --------------------------------------------------------------- main --
def main(nsim=4):
    t0 = time.time()
    sd = csv("schmidt_density_19.csv"); gr = csv("schmidt_granularity.csv")
    sides = gr["digitized_side_um"]; rms_meas = gr["corrected_RMS_D"]
    R = dict(gates=dict(density_abs_D=GATE_D, rms_rel=GATE_RMS))
    R["geometry"] = geometry_check(sd)
    c2, ak = schmidt_ak(sides, gr["sigma2_A_um2_D2"])
    R["schmidt_fit"] = dict(c2=c2, a_k=ak, side_k=float(np.sqrt(ak)), rmsD_48=1000 * np.sqrt(c2) * 2 / (48 * np.sqrt(np.pi)))
    print("geometry", {k: round(v["max_abs"], 3) for k, v in R["geometry"]["cases"].items()}, "Schmidt a_k", round(ak, 2), flush=True)

    # T46 推导 (η 未标定) 的密度审计: 晶体尺寸按 T46 通用常数
    R["density_t46_sizes"] = density_audit(spec_for(sd, 1.0), sd)
    cal = {}
    for geom in ("diffuse", "specular"):
        cal[geom] = calibrate(sd, gr, geom)
        print("calib", geom, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cal[geom].items()}, flush=True)
    R["calibration"] = cal
    G = materials.GRAIN_T47_MICRO_GEOMETRY
    spec = spec_for(sd, cal[G]["eta_scale"])
    psf = cal[G]["psf_um"]
    R["density"] = density_audit(spec, sd)
    R["density_specular_calibration"] = density_audit(spec_for(sd, cal["specular"]["eta_scale"]), sd)
    print("density max resid", round(R["density"]["max_abs_resid"], 4), "budget", R["density"]["budget_diffuse"], flush=True)

    # 四孔径: 闭式解 (主口径 + 镜面口径 + 两种 PSF 读法)
    est = lambda sp, s, g: model_estimator(sp, sides, s, g)
    R["granularity"] = dict(sides=sides.tolist(), measured=rms_meas.tolist(),
                            closed=est(spec, psf, G).tolist(),
                            closed_specular_cal=est(spec_for(sd, cal["specular"]["eta_scale"]), cal["specular"]["psf_um"], "specular").tolist(),
                            psf_readings={str(s): est(spec, s, G).tolist() for s in (0.52, 0.64, 1.0, 1.5)})
    r = np.array(R["granularity"]["closed"])
    R["granularity"]["closed_rel"] = (r / rms_meas - 1).tolist()
    R["granularity"]["closed_passes"] = bool(np.all(np.abs(r / rms_meas - 1) <= GATE_RMS))
    for s, v in R["granularity"]["psf_readings"].items():
        R["granularity"]["psf_readings"][s] = dict(rms=v, a_k=schmidt_ak(sides, np.array(v) ** 2 * sides ** 2)[1])
    if nsim:
        R["granularity"]["sim"] = simulate(spec, psf, G, nsim, sides)
        rs = np.array(R["granularity"]["sim"]["rms_mean"])
        R["granularity"]["sim_rel"] = (rs / rms_meas - 1).tolist()
        R["granularity"]["sim_passes"] = bool(np.all(np.abs(rs / rms_meas - 1) <= GATE_RMS))
        print("sim", np.round(rs, 4), "meas", np.round(rms_meas, 4), flush=True)
        H = H_at_net(spec, 0.682)
        spec_net = float(grain.silver_density(spec, H, G) - 0.0)
        R["skew_controls"] = skew_controls(spec, psf, G, sides)
        print("controls", [(c["tau"], np.round(c["rms"], 4), round(c["skew_5um_D"], 3)) for c in R["skew_controls"]], flush=True)

    # 团聚判定: 泊松 vs Thomas (闭式近似 + 模拟)
    R["clustering"] = clustering_study(sd, gr, spec, psf, G, nsim)
    # Kodak 48 µm (漫射, 净 D 1.0, 42.5 µm 等面积方孔, 直接方差) —— 独立检查
    Hk = H_at_net(spec, 1.0)
    k48 = grain.silver_granularity(spec, Hk, [42.5], psf_um=0.0, geometry="diffuse", adjacent=False)[0]
    spc = spec_for(sd, cal["specular"]["eta_scale"])
    k48s = grain.silver_granularity(spc, H_at_net(spc, 1.0), [42.5], psf_um=0.0, geometry="diffuse", adjacent=False)[0]
    R["kodak48_specular_calibration"] = dict(model_x1000=1000 * float(k48s), ratio=float(k48s * 1000 / 17.0))
    R["kodak48"] = dict(model_x1000=1000 * float(k48), official=17.0, ratio=float(k48 * 1000 / 17.0),
                        schmidt_rmsD=13.0, note="Schmidt RMS_D 13 为净 0.682、显微估计器; Kodak 17 为净漫射 1.0、48 µm 微密度计")
    # FX39 留出
    R["fx39"] = fx39_holdout(sd, gr, spec, psf, G, nsim)
    # 颗粒随密度 (未验证)
    nets = [0.05, 0.1, 0.2, 0.4, 0.682, 1.0, 1.3, 1.576]
    R["grain_vs_density"] = dict(net=nets,
        schmidt25_micro=[float(grain.silver_granularity(spec, H_at_net(spec, n), [25.0], psf, G)[0]) for n in nets],
        kodak48_diffuse=[float(grain.silver_granularity(spec, H_at_net(spec, n), [42.5], 0.0, "diffuse", adjacent=False)[0]) for n in nets],
        verified=False)
    calib = dict(eta_scale=cal[G]["eta_scale"], micro_psf_um=psf, coat_g_m2=list(spec.coat_g_m2), ecd_um=list(spec.ecd_um),
                 base_density=spec.base_density, geometry=G, morph=materials.GRAIN_T47_MORPH)
    (D / "calib.json").write_text(json.dumps(calib, indent=1, ensure_ascii=False))
    (D / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float))
    print("done", f"{time.time() - t0:.0f} s")
    return R


def clustering_study(sd, gr, spec, psf, G, nsim):
    """团聚由 a_k 决定: (1) 泊松 + 各 PSF 读法的 a_k; (2) PSF 取下限读法 (FWHM 1.5 µm → σ 0.64) 时, Thomas 团簇
    (σ_c 2 / 3 / 5 µm) 需要多强才能把 a_k 拉到实测, 以及 PSF 取 Schmidt 自己的 σ 1.5 µm 时团簇的方向。
    团簇项 (线性近似): Cov_T += E[T]² (μ / n_tot) (ln10 · D_spec,g)² G_{√2 σ_c}(r)。"""
    sides = gr["digitized_side_um"]; s2m = gr["sigma2_A_um2_D2"]
    _, ak_meas = schmidt_ak(sides, s2m)
    b = grain.silver_bins(spec); ntot = float(b["n"].sum())
    H = H_at_net(spec, 0.682)
    ET, C0, h = grain.silver_covariance(spec, H, G)
    Dg = float(grain.silver_density(spec, H, G))
    n = C0.shape[0]; x = (np.arange(n) - n // 2) * h
    out = dict(ak_meas=ak_meas, n_total_per_um2=ntot, poisson={}, thomas=[])
    for s in (0.52, 0.64, 1.0, 1.5, psf):
        v = grain.window_variance(C0, h, sides, s) / (ET * np.log(10)) ** 2 * sides ** 2
        out["poisson"][f"{s:.3f}"] = dict(a_k=schmidt_ak(sides, v)[1], plateau=float(np.mean(v[2:])))
    from scipy.ndimage import gaussian_filter
    for s in (0.64, 1.5):
        for sc in (2.0, 3.0, 5.0):
            def stats(mu):
                pad = int(4 * sc * 1.5 / h) + 4
                C = np.pad(C0, pad)
                xx = (np.arange(C.shape[0]) - C.shape[0] // 2) * h
                Gk = np.exp(-(xx[:, None] ** 2 + xx[None] ** 2) / (4 * sc * sc)) / (4 * np.pi * sc * sc)
                C = C + ET * ET * (mu / ntot) * (np.log(10) * Dg) ** 2 * Gk
                v = grain.window_variance(C, h, sides, s) / (ET * np.log(10)) ** 2 * sides ** 2
                return schmidt_ak(sides, v)[1], float(np.mean(v[2:])), v
            ak0 = stats(0.0)[0]
            row = dict(psf=s, sigma_c=sc, ak_poisson=ak0)
            if (ak0 - ak_meas) * (stats(ntot * 0.5)[0] - ak_meas) < 0:
                mu = brentq(lambda m: stats(m)[0] - ak_meas, 0.0, ntot * 0.5, xtol=1e-3)
                akf, pl, v = stats(mu)
                row.update(mu=float(mu), cluster_area_um2=float(mu / ntot), plateau=pl,
                           plateau_rel=pl / float(np.mean(s2m[2:])) - 1, rms=np.sqrt(v / sides ** 2).tolist())
            else:
                row.update(mu=None, note="团簇只会增大 a_k" if ak0 > ak_meas else "团簇强度到上限仍不够")
            out["thomas"].append(row)
    # 模拟验证一组: PSF 0.64, σ_c 3 µm, 由闭式近似给出的 μ (η 不重标, 只看 a_k 与平台的方向)
    if nsim:
        cand = [t for t in out["thomas"] if t["psf"] == 0.64 and t["sigma_c"] == 3.0 and t.get("mu")]
        if cand:
            mu = cand[0]["mu"]
            sp = spec.with_(cluster_mu=float(mu), cluster_um=3.0)
            sim = simulate(sp, 0.64, G, max(2, nsim // 2), sides, base_seed=4800)
            v = np.array(sim["rms_mean"]) ** 2 * sides ** 2
            out["thomas_sim"] = dict(mu=mu, sigma_c=3.0, psf=0.64, rms=sim["rms_mean"], a_k=schmidt_ak(sides, v)[1],
                                     plateau=float(np.mean(v[2:])), skew_5um_D=sim["skew_5um_D"])
            print("thomas sim", out["thomas_sim"], flush=True)
    return out


def fx39_holdout(sd, gr, spec, psf, G, nsim):
    """D-76 标定后, 只改显影工艺参数 (显影阈值光子数 r: 显影越强, 越小的潜影可显影) 拟合 FX39 自己的特性曲线,
    然后预测同一总密度 (≈1, 净 0.682) 的颗粒 (25 µm 与平台) 相对 D-76 的倍数。实测: 平台 17.4/13.0 = 1.34 (Schmidt 表),
    25 µm 约 1.39 (Codex 读图)。FX39: 1+12, 24 °C, 10 min, γ 0.59; D-76: 原液, 20 °C, 9 min, γ 0.56。"""
    fx = csv("schmidt_fx39_density.csv")
    E01 = float(np.interp(0.10, sd["net_D"], sd["chart_relative_log10_exposure"]))
    Hf = 0.8 / 400 * 10 ** (fx["chart_relative_log10_exposure"] - E01)
    def resid(r):
        sp = spec.with_(photons_r=float(r))
        return fx["net_D"] - (grain.silver_density(sp, Hf) - grain.silver_density(sp, 0.0))
    rs = np.linspace(2.0, 5.0, 61)
    err = [float(np.sqrt(np.mean(resid(r) ** 2))) for r in rs]
    r_best = float(rs[int(np.argmin(err))])
    sp = spec.with_(photons_r=r_best)
    sides = gr["digitized_side_um"]
    base = model_estimator(spec, sides, psf, G)
    fxr = model_estimator(sp, sides, psf, G)
    out = dict(r_fit=r_best, curve_rms=min(err), curve_max_abs=float(np.max(np.abs(resid(r_best)))),
               rms_ratio_by_aperture=(fxr / base).tolist(), measured_ratio_plateau=17.4 / 13.0, measured_ratio_25um=1.39)
    # 温度 / γ: r 的变化同时代表 24 °C 与 1+12 稀释的净显影强度 (曲线 γ 0.56 → 0.59); 对同一总密度的颗粒的影响
    out["gamma_model_d76"] = float(np.polyfit(np.log10(H_levels(sd))[6:16], (grain.silver_density(spec, H_levels(sd)))[6:16], 1)[0])
    out["gamma_model_fx39"] = float(np.polyfit(np.log10(H_levels(sd))[6:16], (grain.silver_density(sp, H_levels(sd)))[6:16], 1)[0])
    for alt in (dict(spread=materials.GRAIN_T47_MORPH["spread"] * 2), dict(filament_um=materials.GRAIN_T47_MORPH["filament_um"] * 2)):
        s2 = sp.with_(**alt)
        out.setdefault("morphology_sensitivity", []).append(dict(change=alt, ratio=(model_estimator(s2, sides, psf, G) / base).tolist()))
    return out


if __name__ == "__main__":
    if "--controls" in sys.argv:                 # 只重算偏度对照 (其余结果不变)
        R = json.loads((D / "results.json").read_text()); cal = json.loads((D / "calib.json").read_text())
        sd = csv("schmidt_density_19.csv"); gr = csv("schmidt_granularity.csv")
        R["skew_controls"] = skew_controls(spec_for(sd, cal["eta_scale"]), cal["micro_psf_um"], cal["geometry"], gr["digitized_side_um"])
        (D / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float))
        print([(c["tau"], round(c["radius_um"], 2), np.round(c["rms"], 4).tolist(), round(c["skew_5um_D"], 3), round(c["skew_pixel_D"], 3)) for c in R["skew_controls"]])
        sys.exit(0)
    n = 4
    if "--sim" in sys.argv:
        n = int(sys.argv[sys.argv.index("--sim") + 1])
    main(n)
