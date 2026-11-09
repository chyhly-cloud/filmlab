"""T50: 显影液 = 显影晶体成团 (Thomas 过程, cluster_mode="developed") + 显影阈值 r (filmlab tools/eval/t50_developers.py 的同一算法)。

数据 (validation/data/): Schmidt 同一显微镜、同一估计器的 6 种片 × 显影液 (methodgrain 结果表 RMS_D、a_k; 各片种页 SK 图曲线)。
  - 片 (晶体、涂布银、片基) 由 Schmidt 该片的 D-76 曲线定 (同 T47b); 不随显影液改。
  - 显影液: r (每组合由该组合自己的曲线定) + 团簇 (μ, σ) 对同一显影液的所有片共用; D-76 固定泊松。
  - 颗粒在总密度 1 处: RMS_D = 200 µm 方孔平台 → 48 µm 圆孔; a_k: 4 方孔 + 显微 PSF (T47) → Schmidt 两参数式 (表的单位由 Tri-X / D-76 定)。
检验: 每种显影液留出一种片, 用其余 5 种定 (μ, σ), 预测留出片 "相对同片 D-76 的倍数"; 门槛 ±20%。
quick: 只用数据库里的显影液参数 (silvergrain/data/bw_t50_developers.json = 全部片共同拟合的 (μ, σ) 与逐组合 r) 预测, 不重拟合。
"""
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar

from .. import database as db
from .. import silver
from ..emulsion import universal
from .refdata import csv_rows, load
from .t47_schmidt import H_of, SCHMIDT, schmidt_curves

FILMS = ["trix", "tmx", "tmy", "d100", "d400", "panf"]
MG = {"trix": "TriX", "tmx": "Tmax100", "tmy": "Tmax400", "d100": "Delta100", "d400": "Delta400", "panf": "PanF"}
DB_KEY = {"trix": "kodak_trix_400", "tmx": "kodak_tmax_100", "tmy": "kodak_tmax_400", "d100": "ilford_delta_100",
          "d400": "ilford_delta_400", "panf": "ilford_pan_f_plus"}
DEVS = ["XTOL", "Rodinal", "FX39"]
REPORT = {"Alpha": "Wehner/Alpha", "SD2525": "SD2525"}
SIDES = [4.92, 9.87, 19.95, 25.19]
MU = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
SIG = [1.0, 1.5, 2.0, 3.0, 4.0]
GATE = 0.20


def psf():
    return load("t47_calib.json")["micro_psf_um"]


def measured():
    return {(r["film"], r["developer"]): (float(r["RMS_D"]), float(r["a_k_um"])) for r in csv_rows("schmidt_methodgrain.csv")}


def sk(film, dev):
    pts = [(float(r["E"]), float(r["gross_D"])) for r in csv_rows("schmidt_sk_curves.csv") if r["film"] == film and r["developer"] == dev]
    return np.array(pts).reshape(-1, 2)


def film_base(film):
    S = schmidt_curves()
    E, net, fog, ei, rms = S[film]
    th = SCHMIDT[film][1]
    sp = silver.silver_derive(ei, dict(universal()), db.universal()["t47"]["eta3d_scale"], H_of(E, net, ei), net, fog,
                              db.t47_morph(), thickness_um=th)
    m = np.maximum.accumulate(net) + 1e-9 * np.arange(len(net))
    E01 = float(np.interp(0.10, m, E))
    return sp, (lambda Ex: 0.8 / ei * 10.0 ** (np.asarray(Ex, float) - E01))


def gross(sp, H):
    return sp.base_density + silver.silver_density(sp, H)


def fit_r(sp0, Hof, pts, mu, sig):
    if len(pts) < 3:
        return 4.0, np.nan
    H = Hof(pts[:, 0])
    def err(r):
        sp = sp0.with_(photons_r=float(r), cluster_mu=mu, cluster_um=sig, cluster_mode="developed")
        return float(np.mean((gross(sp, H) - pts[:, 1]) ** 2))
    res = minimize_scalar(err, bounds=(1.5, 8.0), method="bounded", options=dict(xatol=1e-3))
    sp = sp0.with_(photons_r=float(res.x), cluster_mu=mu, cluster_um=sig, cluster_mode="developed")
    return float(res.x), float(np.max(np.abs(gross(sp, H) - pts[:, 1])))


def H_at_gross(sp, Dg):
    lh = np.linspace(-7, 3, 4001)
    g = np.maximum.accumulate(gross(sp, 10 ** lh) + 1e-12 * np.arange(len(lh)))
    return float(10 ** np.interp(Dg, g, lh))


def ak_fit(sides, s2a):
    f = lambda a, c2, ak: c2 * a / (a + ak)
    (c2, ak), _ = curve_fit(f, np.asarray(sides) ** 2, np.asarray(s2a), p0=(0.3, 20.0), maxfev=20000)
    return float(ak)


def grain_at(sp):
    H = H_at_gross(sp, 1.0)
    s = float(silver.silver_granularity(sp, H, [200.0], 0.0, "diffuse", adjacent=False)[0]) * 200.0
    rms = 1000 * s * 2 / (48 * np.sqrt(np.pi))
    r = silver.silver_granularity(sp, H, SIDES, psf(), "diffuse", adjacent=True)
    return rms, ak_fit(SIDES, r ** 2 * np.asarray(SIDES) ** 2)


def quick():
    """数据库参数 (全部片共同拟合) 的预测: 每片 × 显影液的 RMS 与 a_k 相对同片 D-76 的倍数, 对实测。"""
    M = measured()
    devdb = db.bw_t50_developers()
    base = {}
    for f in FILMS:
        sp0, _ = film_base(f)
        base[f] = (sp0, grain_at(sp0.with_(photons_r=4.0)))
    ak_unit = M[("TriX", "D76")][1] / np.sqrt(base["trix"][1][1])
    out = dict(d76={}, developers={})
    for f in FILMS:
        rms0, ak0 = base[f][1]; m = M[(MG[f], "D76")]
        out["d76"][f] = dict(rms_pred=rms0, rms_meas=m[0], ak_pred=ak_unit * np.sqrt(ak0), ak_meas=m[1])
    for dev in DEVS:
        c = devdb["developers"][dev.lower()]
        rows, npass = {}, [0, 0, 0]
        for f in FILMS:
            mm, m0 = M.get((MG[f], dev)), M[(MG[f], "D76")]
            if mm is None:
                continue
            r = float(devdb["photons_r"].get(DB_KEY[f], {}).get(dev.lower(), 4.0))
            mod = dict(photons_r=r)
            if c["cluster_mu"] > 0:
                mod.update(cluster_mu=float(c["cluster_mu"]), cluster_um=float(c["cluster_um"]), cluster_mode="developed")
            rms, ak = grain_at(base[f][0].with_(**mod))
            rms0, ak0 = base[f][1]
            p = dict(rms=rms, ak=ak_unit * np.sqrt(ak), rms_ratio=rms / rms0, ak_ratio=np.sqrt(ak / ak0), r=r)
            ms = dict(rms=mm[0], ak=mm[1], rms_ratio=mm[0] / m0[0], ak_ratio=mm[1] / m0[1])
            er, ea = p["rms_ratio"] / ms["rms_ratio"] - 1, p["ak_ratio"] / ms["ak_ratio"] - 1
            npass[0] += abs(er) <= GATE; npass[1] += abs(ea) <= GATE; npass[2] += 1
            rows[f] = dict(pred=p, meas=ms, err_rms_ratio=er, err_ak_ratio=ea)
        out["developers"][dev] = dict(params=dict(mu=c["cluster_mu"], sigma_um=c["cluster_um"]), films=rows,
                                      pass_rms=f"{npass[0]}/{npass[2]}", pass_ak=f"{npass[1]}/{npass[2]}")
    return out


def full(log=print):
    """全网格 (μ, σ) × 每组合拟合 r 与留一 (T50 第 4.2 节; 约 10–30 分钟)。"""
    M = measured()
    G = {}
    for f in FILMS:
        sp0, Hof = film_base(f)
        rms0, ak0 = grain_at(sp0.with_(photons_r=4.0))
        G[f"{f}|D76"] = dict(base=dict(rms=rms0, ak=ak0))
        for dev in DEVS + list(REPORT):
            pts = sk(f, dev)
            cell = {}
            for mu in MU:
                for sig in (SIG if mu > 0 else [1.0]):
                    r, res = fit_r(sp0, Hof, pts, mu, sig)
                    rms, ak = grain_at(sp0.with_(photons_r=r, cluster_mu=mu, cluster_um=sig, cluster_mode="developed"))
                    cell[f"{mu}|{sig}"] = dict(r=r, curve_maxres=res, rms=rms, ak=ak)
            G[f"{f}|{dev}"] = cell
            log(f"  T50 网格 {f} {dev}")
    ak_unit = M[("TriX", "D76")][1] / np.sqrt(G["trix|D76"]["base"]["ak"])
    keys = list(G["trix|XTOL"])
    def pred(f, dev, key):
        c = G[f"{f}|{dev}"][key]; b = G[f"{f}|D76"]["base"]
        return dict(rms=c["rms"], ak=ak_unit * np.sqrt(c["ak"]), rms_ratio=c["rms"] / b["rms"], ak_ratio=np.sqrt(c["ak"] / b["ak"]), r=c["r"])
    def meas(f, dev):
        m, m0 = M.get((MG[f], REPORT.get(dev, dev))), M[(MG[f], "D76")]
        return None if m is None else dict(rms=m[0], ak=m[1], rms_ratio=m[0] / m0[0], ak_ratio=m[1] / m0[1])
    def loss(dev, films, key):
        e = []
        for f in films:
            m = meas(f, dev)
            if m is None:
                continue
            p = pred(f, dev, key)
            e += [np.log(p["rms_ratio"] / m["rms_ratio"]), np.log(p["ak_ratio"] / m["ak_ratio"])]
        return float(np.mean(np.square(e)))
    res = dict(developers={})
    for dev in DEVS + list(REPORT):
        allf = [f for f in FILMS if meas(f, dev)]
        best_all = min(keys, key=lambda k: loss(dev, allf, k))
        loo = {}
        for f in allf:
            kb = min(keys, key=lambda k: loss(dev, [g for g in allf if g != f], k))
            p, m = pred(f, dev, kb), meas(f, dev)
            loo[f] = dict(params=kb, pred=p, meas=m, err_rms_ratio=p["rms_ratio"] / m["rms_ratio"] - 1,
                          err_ak_ratio=p["ak_ratio"] / m["ak_ratio"] - 1)
        res["developers"][dev] = dict(fit_all=best_all, loo=loo, report_only=dev in REPORT,
                                      pass_rms=sum(abs(v["err_rms_ratio"]) <= GATE for v in loo.values()),
                                      pass_ak=sum(abs(v["err_ak_ratio"]) <= GATE for v in loo.values()), n=len(loo))
    return res
