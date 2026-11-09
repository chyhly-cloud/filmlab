"""T47 / T47b: 显式银粒对 Schmidt 显微颗粒与 Kodak / Fuji 官方 RMS 的检验 (filmlab tools/eval/t47_trix.py、t47b_bw_loo.py 的
同一算法; 数据在 validation/data/)。

T47 (Tri-X 400 / D-76 原液 20 °C 9 min, Schmidt https://www.bernhard-w-schmidt.de/en/methodgrain):
  晶体 = T46 通用结构 + 三维有效量子效率 ×6.61 (标定量 1: 由 20 / 25 µm 平台定) + 显微镜 PSF (标定量 2: 由结构尺度 a_k 定);
  涂布银量由 Schmidt 的 19 级 D-76 曲线 (漫射) 定。检验: 19 级密度 |ΔD| ≤ 0.10, 四孔径 (5 / 10 / 20 / 25 µm 方孔, 相邻差估计器)
  RMS 都在 ±15% 内 (门槛事先写定); 独立的只有 5、10 µm 两点 (T47 负责人结论 3)。
T47b (同一常数用于全部黑白片):
  A 组 = 官方 48 µm 漫射 RMS × 1000 (净漫射 1.0): 只有 D-76 测的 T-Max 100 / 400 判定; Tri-X / Plus-X (HC-110)、Acros (Microfine) 只报告。
  B 组 = Schmidt 同一显微镜、同一 D-76 的 6 种片 RMS_D (平台换算到 48 µm 圆孔, 总 D ≈ 1); Tri-X 标定, 其余 5 种留出。
"""
import numpy as np

from .. import database as db
from .. import dye, silver
from ..emulsion import process_params, universal
from ..films import Curve
from .refdata import csv_cols, csv_rows, load

GATE_D, GATE_RMS = 0.10, 0.15
FOG = 0.318
T_UM = 0.12


def _morph():
    return db.t47_morph()


def _eta3d():
    return db.universal()["t47"]["eta3d_scale"]


# ------------------------------------------------------------------ T47 --
def trix_spec(eta_scale=None):
    sd = csv_cols("schmidt_density_19.csv")
    H = 0.8 / 400.0 * 10.0 ** sd["aligned_log10_exposure"]
    return silver.silver_derive(400.0, universal(), _eta3d() if eta_scale is None else eta_scale, H, sd["net_D"], FOG, _morph(),
                                process="D-76 原液 20 °C 9 min"), sd


def H_at_net(spec, net, geometry="diffuse"):
    lh = np.linspace(-6, 2, 4001)
    Dn = silver.silver_curve(spec, lh, geometry) - silver.silver_curve(spec, [-12.0], geometry)[0]
    return float(10 ** np.interp(net, np.maximum.accumulate(Dn + 1e-12 * np.arange(len(Dn))), lh))


def schmidt_four_apertures():
    """T47 表 5.2 的闭式解 (主口径: 漫射, 标定的 η 与 PSF)。"""
    cal = load("t47_calib.json")
    spec, sd = trix_spec(cal["eta_scale"])
    gr = csv_cols("schmidt_granularity.csv")
    sides, meas = gr["digitized_side_um"], gr["corrected_RMS_D"]
    H = H_at_net(spec, 0.682)
    r = silver.silver_granularity(spec, H, sides, psf_um=cal["micro_psf_um"], geometry="diffuse", adjacent=True)
    Hl = 0.8 / 400.0 * 10.0 ** sd["aligned_log10_exposure"]
    dif = silver.silver_density(spec, Hl) - silver.silver_density(spec, 0.0)
    k48 = silver.silver_granularity(spec, H_at_net(spec, 1.0), [42.5], psf_um=0.0, geometry="diffuse", adjacent=False)[0]
    ref = load("t47_reference.json")
    return dict(sides=sides.tolist(), measured=meas.tolist(), closed=r.tolist(), closed_rel=(r / meas - 1).tolist(),
                passes=bool(np.all(np.abs(r / meas - 1) <= GATE_RMS)),
                density_max_abs_resid=float(np.max(np.abs(sd["net_D"] - dif))),
                density_passes=bool(np.max(np.abs(sd["net_D"] - dif)) <= GATE_D),
                budget=silver.silver_budget(spec), kodak48_x1000=1000 * float(k48), kodak48_ratio=1000 * float(k48) / 17.0,
                reference=dict(closed=ref["granularity"]["closed"], density_max_abs_resid=ref["density_max_abs_resid"],
                               kodak48_x1000=ref["kodak48"]["model_x1000"]),
                identical=bool(np.array_equal(r, np.asarray(ref["granularity"]["closed"]))))


# ----------------------------------------------------------------- T47b --
OFFICIAL_DEV = {"trix400": "HC-110 (B)", "trix320": "HC-110 (B)", "plusx125": "HC-110 (B)", "tmax100": "D-76",
                "tmax400": "D-76", "acros100": "Microfine"}
CURVE_OF = {"trix400": "trix", "trix320": "trix", "plusx125": "trix", "tmax100": "tmx", "tmax400": "tmy", "acros100": "tmx"}
SCHMIDT = {"trix": ("Tri-X 400", None), "tmx": ("T-Max 100 (TMX)", 0.12), "tmy": ("T-Max 400 (TMY)", 0.12),
           "d100": ("Delta 100", 0.12), "d400": ("Delta 400", 0.12), "panf": ("Pan F+", None)}


def schmidt_curves():
    """{key: (E, net, fog, EI_eff, RMS_D)} (t47b_bw_loo.schmidt_curves 的同一读法)。"""
    t = csv_cols("schmidt_density_19.csv")
    out = {"trix": (t["chart_relative_log10_exposure"], t["net_D"], FOG, 400.0, 13.0)}
    names = {v[0]: k for k, v in SCHMIDT.items()}
    rows = csv_rows("schmidt_bw_d76.csv")
    for name in dict.fromkeys(r["film"] for r in rows):
        rr = [r for r in rows if r["film"] == name]
        E = np.array([float(r["E"]) for r in rr]); g = np.array([float(r["gross_D"]) for r in rr])
        net = np.array([float(r["net_D"]) for r in rr])
        out[names[name]] = (E, net, float(g[0] - net[0]), float(rr[0]["EI_eff"]), float(rr[0]["RMS_D"]))
    return out


def H_of(E, net, ei):
    m = np.maximum.accumulate(net) + 1e-9 * np.arange(len(net))
    E01 = float(np.interp(0.10, m, E))
    return 0.8 / ei * 10.0 ** (E - E01)


def t47_spec(E, net, fog, ei, thickness, eta, beta, scale_tgrain=True):
    U = dict(universal(eta=eta, beta=beta))
    return silver.silver_derive(ei, U, _eta3d(), H_of(E, net, ei), net, fog, _morph(), thickness_um=thickness,
                                scale_tgrain=scale_tgrain)


def H_at(spec, net):
    lh = np.linspace(-7, 3, 5001)
    Dn = silver.silver_curve(spec, lh) - silver.silver_curve(spec, [-14.0])[0]
    return float(10 ** np.interp(net, np.maximum.accumulate(Dn + 1e-12 * np.arange(len(Dn))), lh))


def kodak48(spec):
    return 1000 * float(silver.silver_granularity(spec, H_at(spec, 1.0), [42.5], 0.0, "diffuse", adjacent=False)[0])


def schmidt_rmsd(spec, net):
    """Schmidt 的 RMS_D: 大孔径平台 σ²A 换算到 48 µm 圆孔 (1000 √σ²A · 2 / (48 √π)); 漫射。"""
    s = float(silver.silver_granularity(spec, H_at(spec, net), [200.0], 0.0, "diffuse", adjacent=False)[0]) * 200.0
    return 1000 * s * 2 / (48 * np.sqrt(np.pi))


def _bw_curve_from(E, net, ei):
    """T46 对照用: 以这条曲线 (净 D 对绝对 log H) 为特性曲线的黑白乳剂。"""
    x = np.linspace(-5.0, 2.0, 256)
    lh = np.log10(H_of(E, net, ei))
    slope = (net[-1] - net[-3]) / (lh[-1] - lh[-3])
    d = np.where(x <= lh[-1], np.interp(x, lh, net, left=0.0), net[-1] + slope * (x - lh[-1]))
    return Curve(x, np.repeat((np.clip(d, 0, None) + 0.0)[:, None], 3, 1), "bw")


def t46_pred(E, net, fog, ei, thickness, eta, beta, target="kodak", schmidt_net=None):
    e = _bw_curve_from(E, net, ei)
    U = universal(eta=eta, beta=beta)
    sp = dye.derive_spec(e, ei, "D55", U, kind="silver", thickness_um=thickness, process=process_params("BW"))
    Hx = dye.exposure_axis(e, ei)[:, 1]
    Dm, sg = dye.layer_curves(sp, 1, Hx)
    Dm = Dm - Dm.min(); i = np.argsort(Dm)
    return 1000 * float(np.interp(1.0 if target == "kodak" else schmidt_net, Dm[i], sg[i]))


def t47b():
    S = schmidt_curves()
    loo46 = load("t46_loo_reference.json")
    rows = [r for r in db.official_rms() if r["class"] == "bw" and r["metric"] == "rms_net1" and r["key"] != "tmaxp3200"]
    A = []
    for r in rows:
        k = r["key"]
        E, net, fog, _, _ = S[CURVE_OF[k]]
        ei = float(r["iso"])
        th = None if r["model_morph"].startswith("3D") else T_UM
        lo = loo46["loo"][k]
        sp = t47_spec(E, net, fog, ei, th, lo["eta"], lo["beta"])
        p47 = kodak48(sp)
        p46 = t46_pred(E, net, fog, ei, th, lo["eta"], lo["beta"])
        A.append(dict(key=k, film=r["film"], developer=OFFICIAL_DEV[k], judged=OFFICIAL_DEV[k] == "D-76", official=float(r["value"]),
                      t47=p47, t47_ratio=p47 / float(r["value"]), t46_same_curve_ratio=p46 / float(r["value"]), t46_loo_ratio=lo["ratio"]))
    B = []
    t46 = db.universal()["t46"]
    for k, (name, th) in SCHMIDT.items():
        E, net, fog, ei, rms = S[k]
        sp = t47_spec(E, net, fog, ei, th, t46["eta"], t46["beta"])
        nt = 1.0 - fog
        p47 = schmidt_rmsd(sp, nt)
        dmax = float(np.max(np.abs(silver.silver_density(sp, H_of(E, net, ei)) - silver.silver_density(sp, 0.0) - net)))
        B.append(dict(key=k, film=name, ei_eff=ei, measured=rms, t47=p47, t47_ratio=p47 / rms, role="标定" if k == "trix" else "留出",
                      curve_max_abs=dmax))
    ref = load("t47b_reference.json")
    dA = max(abs(a["t47"] - b["t47"]) for a, b in zip(A, ref["official"]))
    dB = max(abs(a["t47"] - b["t47"]) for a, b in zip(B, ref["schmidt"]))
    lj = [abs(np.log(a["t47_ratio"])) for a in A if a["judged"]]
    lb = [abs(np.log(b["t47_ratio"])) for b in B if b["role"] == "留出"]
    return dict(official=A, schmidt=B, max_abs_diff_vs_reference=dict(official=dA, schmidt=dB),
                summary=dict(judged_median_abs_log=float(np.median(lj)), judged_rms_log=float(np.sqrt(np.mean(np.square(lj)))),
                             holdout_median_abs_log=float(np.median(lb)), holdout_rms_log=float(np.sqrt(np.mean(np.square(lb))))))
