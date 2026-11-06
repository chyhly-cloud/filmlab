"""T50: 显影液对银颗粒的影响 —— 显影晶体成团 (感染显影 / 银聚集, Thomas 过程, grain.SilverSpec cluster_mode="developed")。

数据 (tools/eval/t50/): Schmidt 同一显微镜、同一估计器的 6 种片 × 显影液 (methodgrain 结果表 RMS_D、a_k; 各片种页 SK 图曲线)。
只用配方公开 (或有安全数据表) 的显影液建模: D-76 (参照, 泊松, T47 结论)、XTOL、Rodinal、FX39; SD2525 / Wehner-Alpha / Shadowmax 只报告。

模型 (每片 × 显影液):
  - 片 (晶体、涂布银、片基) 由 Schmidt 该片的 D-76 曲线定 (同 T47b, tools/eval/t47b_bw_loo.py); 不随显影液改。
  - 同一片的 SK 图各显影液用同一曝光楔 → 共用 D-76 的 E → H (净 D 0.1 处 H = 0.8/EI, EI = 片种页 ISO 6 口径)。
  - 显影液: 显影阈值光子数 r (每组合由该组合自己的曲线定, 同 T47 的 FX39) + 团簇 (μ = 每团平均显影晶体数, σ = 团的高斯半径)。
    (μ, σ) 对同一显影液的所有片共用; D-76 固定 μ = 0 (泊松)。
  - 颗粒在总密度 D = 1 处 (Schmidt 口径): RMS_D = 200 µm 方孔平台换算到 48 µm 圆孔 (同 T47); a_k: 4 种方孔 (相邻差) + 显微 PSF
    (T47 标定 1.096 µm) → Schmidt 两参数式, 换算到表的单位 (√a_k × 比例, 比例由 Tri-X / D-76 定)。
检验: 每种显影液每次留出一种片, 用其余 5 种片定 (μ, σ), 预测留出片的 RMS 与 a_k; 门槛 ±20%。主判定用"相对同片 D-76 的倍数"
(去掉 T47b 的片基准误差), 同时报告绝对值。
用法: python3 tools/eval/t50_developers.py [网格缓存.json] [输出.json]"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from filmlab import grain, materials  # noqa: E402
import t47b_bw_loo as L  # noqa: E402

D = HERE / "t50"
FILMS = ["trix", "tmx", "tmy", "d100", "d400", "panf"]
MG = {"trix": "TriX", "tmx": "Tmax100", "tmy": "Tmax400", "d100": "Delta100", "d400": "Delta400", "panf": "PanF"}
DEVS = ["XTOL", "Rodinal", "FX39"]                         # 建模 (D76 为参照)
REPORT = {"Alpha": "Wehner/Alpha", "SD2525": "SD2525"}       # 只报告
SIDES = [4.92, 9.87, 19.95, 25.19]                          # Schmidt 图 4 的方孔边长 (T47 实读)
PSF = json.loads((HERE / "t47" / "calib.json").read_text())["micro_psf_um"]
MU = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
SIG = [1.0, 1.5, 2.0, 3.0, 4.0]


def rows(name):
    return list(csv.DictReader(l for l in (D / name).read_text().splitlines() if not l.startswith("#")))


def measured():
    out = {}
    for r in rows("schmidt_methodgrain.csv"):
        out[(r["film"], r["developer"])] = (float(r["RMS_D"]), float(r["a_k_um"]))
    return out


def sk(film, dev):
    pts = [(float(r["E"]), float(r["gross_D"])) for r in rows("schmidt_sk_curves.csv") if r["film"] == film and r["developer"] == dev]
    return np.array(pts).reshape(-1, 2)


def film_base(film):
    S = L.schmidt_curves()
    E, net, fog, ei, rms = S[film]
    th = L.SCHMIDT[film][1]
    sp = grain.silver_derive(ei, dict(materials.grain_universal()), materials.GRAIN_T47_ETA3D_SCALE, L.H_of(E, net, ei), net, fog,
                             materials.GRAIN_T47_MORPH, thickness_um=th)
    m = np.maximum.accumulate(net) + 1e-9 * np.arange(len(net))
    E01 = float(np.interp(0.10, m, E))
    # SK 图横轴与 D-76 曲线 (T47b 读数) 的横轴: 同一 Schmidt 图, 读数定标相同 (E 原样)
    return sp, (lambda Ex: 0.8 / ei * 10.0 ** (np.asarray(Ex, float) - E01))


def gross(sp, H):
    return sp.base_density + grain.silver_density(sp, H)


def fit_r(sp0, Hof, pts, mu, sig):
    """该组合的显影阈值 r: 模型总密度与该组合 SK 读数的最小二乘。"""
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
    s = float(grain.silver_granularity(sp, H, [200.0], 0.0, "diffuse", adjacent=False)[0]) * 200.0
    rms = 1000 * s * 2 / (48 * np.sqrt(np.pi))
    r = grain.silver_granularity(sp, H, SIDES, PSF, "diffuse", adjacent=True)
    ak = ak_fit(SIDES, r ** 2 * np.asarray(SIDES) ** 2)
    return rms, ak


def build_grid(cache):
    """每片 × 显影液 × (μ, σ): 拟合 r, 算 RMS 与 a_k (面积, µm²)。"""
    if Path(cache).exists():
        return json.loads(Path(cache).read_text())
    G = {}
    for f in FILMS:
        sp0, Hof = film_base(f)
        r0, res0 = fit_r(sp0, Hof, sk(f, "D76"), 0.0, 1.0)
        rms0, ak0 = grain_at(sp0.with_(photons_r=4.0))
        G[f"{f}|D76"] = dict(base=dict(rms=rms0, ak=ak0, r_fit=r0, curve_maxres_r_fit=res0))
        for dev in DEVS + list(REPORT):
            pts = sk(f, dev)
            cell = {}
            for mu in MU:
                for sig in (SIG if mu > 0 else [1.0]):
                    r, res = fit_r(sp0, Hof, pts, mu, sig)
                    sp = sp0.with_(photons_r=r, cluster_mu=mu, cluster_um=sig, cluster_mode="developed")
                    rms, ak = grain_at(sp)
                    cell[f"{mu}|{sig}"] = dict(r=r, curve_maxres=res, rms=rms, ak=ak, npts=int(len(pts)))
            G[f"{f}|{dev}"] = cell
            print(f, dev, "done", flush=True)
        Path(cache).write_text(json.dumps(G, indent=1))
    return G


def main(cache, out):
    G = build_grid(cache)
    M = measured()
    trix_ak_model = G["trix|D76"]["base"]["ak"]
    ak_unit = M[("TriX", "D76")][1] / np.sqrt(trix_ak_model)      # 表的 a_k (µm) = 比例 × √a_k (µm²); Tri-X / D-76 定
    keys = [k for k in G["trix|XTOL"]]
    def pred(f, dev, key):
        c = G[f"{f}|{dev}"][key]; b = G[f"{f}|D76"]["base"]
        return dict(rms=c["rms"], ak=ak_unit * np.sqrt(c["ak"]), rms_ratio=c["rms"] / b["rms"],
                    ak_ratio=np.sqrt(c["ak"] / b["ak"]), r=c["r"], curve_maxres=c["curve_maxres"], npts=c["npts"])
    def meas(f, dev):
        m, m0 = M.get((MG[f], REPORT.get(dev, dev))), M[(MG[f], "D76")]
        return None if m is None else dict(rms=m[0], ak=m[1], rms_ratio=m[0] / m0[0], ak_ratio=m[1] / m0[1])
    def loss(dev, films, key, mode="ratio"):
        e = []
        for f in films:
            m = meas(f, dev)
            if m is None: continue
            p = pred(f, dev, key)
            e += [np.log(p[f"rms_{mode}" if mode == "ratio" else "rms"] / m[f"rms_{mode}" if mode == "ratio" else "rms"]),
                  np.log(p[f"ak_{mode}" if mode == "ratio" else "ak"] / m[f"ak_{mode}" if mode == "ratio" else "ak"])]
        return float(np.mean(np.square(e)))
    res = dict(ak_unit=ak_unit, d76={}, developers={})
    for f in FILMS:
        b = G[f"{f}|D76"]["base"]; m = M[(MG[f], "D76")]
        res["d76"][f] = dict(rms_pred=b["rms"], rms_meas=m[0], ak_pred=ak_unit * np.sqrt(b["ak"]), ak_meas=m[1],
                             r_fit=b["r_fit"], curve_maxres=b["curve_maxres_r_fit"])
    for dev in DEVS + list(REPORT):
        allf = [f for f in FILMS if meas(f, dev)]
        best_all = min(keys, key=lambda k: loss(dev, allf, k))
        loo = {}
        for f in allf:
            kb = min(keys, key=lambda k: loss(dev, [g for g in allf if g != f], k))
            p, m = pred(f, dev, kb), meas(f, dev)
            loo[f] = dict(params=kb, pred=p, meas=m,
                          err_rms_ratio=p["rms_ratio"] / m["rms_ratio"] - 1, err_ak_ratio=p["ak_ratio"] / m["ak_ratio"] - 1,
                          err_rms_abs=p["rms"] / m["rms"] - 1, err_ak_abs=p["ak"] / m["ak"] - 1)
        res["developers"][dev] = dict(fit_all=best_all, fit_all_pred={f: pred(f, dev, best_all) for f in allf}, loo=loo,
                                      report_only=dev in REPORT)
        print(f"\n{dev}{' (只报告)' if dev in REPORT else ''}: 全部片拟合 μ|σ = {best_all}")
        for f, v in loo.items():
            print(f"  留出 {f:5s} μ|σ {v['params']:8s} RMS 倍数 预测 {v['pred']['rms_ratio']:.2f} 实测 {v['meas']['rms_ratio']:.2f} ({v['err_rms_ratio']:+.0%})"
                  f"  a_k 倍数 {v['pred']['ak_ratio']:.2f}/{v['meas']['ak_ratio']:.2f} ({v['err_ak_ratio']:+.0%})"
                  f"  绝对 RMS {v['pred']['rms']:.1f}/{v['meas']['rms']:.1f} ({v['err_rms_abs']:+.0%}) a_k {v['pred']['ak']:.1f}/{v['meas']['ak']:.1f} ({v['err_ak_abs']:+.0%})"
                  f"  r {v['pred']['r']:.2f} 曲线 {v['pred']['curve_maxres']:.3f} ({v['pred']['npts']} 点)")
    json.dump(res, open(out, "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/t50_grid.json", sys.argv[2] if len(sys.argv) > 2 else "/tmp/t50_res.json")
