"""T55g: 黑白片颗粒随密度的形状 —— 漫射收集 (Callier) 的作用方式, 用独立数据定, RPL 只作留出。

两种作用方式 (silver.SilverSpec.callier):
  "grain" (T47): 每颗银丝团按 g = 1/Q 部分透明, 透射域重叠按部分透明相乘;
  "macro": 银丝团不透明 (透射域 = 镜面统计), 漫射密度 = 镜面密度 / Q。平均密度相同; 起伏的形状不同,
           且镜面 / 漫射颗粒度之比 = Q (Kodak 研究所: 仪器测得的 RMS 颗粒度随有效密度 / 单次漫射密度之比 q 线性增长)。
独立数据 (data/bw_granularity_vs_density.json, 只有读数与 URL):
  · Bahler et al. 1968 (J. SMPTE 77, p. 1196) 图 2: Eastman Tri-X Negative 5233 与 4-X 5224, D-96 类显影, γ 0.70, 毛密度 0.3–1.7;
  · Bahler et al. 1968 (J. SMPTE 77, p. 1140) 图 3: Plus-X 4231 与 XT 5220 (细颗粒, 高密度段饱和);
  · ISPRS (Kodak 2405 Double-X Aerographic, D-76 / DK-50, 25 / 50 µm 圆孔, 0.6–1.7): 直线关系。
标定: η_3D 按 T47 的同一方法 (tools/eval/t47_trix.calibrate: Schmidt Tri-X / D-76 显微颗粒 20 / 25 µm 平台 σ²A + 显微镜 PSF
使 a_k 吻合) 在 "macro" 下重标; 不用任何形状数据。σ 对密度的形状是预测, 与独立数据比较; RPL 的胶片侧 σ_D(D)
(realscan_rpl_filmside.json, validation.filmside) 只作留出。
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, curve_fit

from .. import database as db
from .. import silver
from ..emulsion import universal
from .refdata import csv_cols

DATA = Path(__file__).resolve().parent / "data"
FOG = 0.318
SIDE48 = 48.0 * np.sqrt(np.pi) / 2


def schmidt_ak(sides, s2a):
    f = lambda a, c2, ak: c2 * a / (a + ak)
    (c2, ak), _ = curve_fit(f, np.asarray(sides) ** 2, np.asarray(s2a), p0=(0.3, 20.0))
    return float(c2), float(ak)


def trix_spec(eta_scale, callier):
    sd = csv_cols("schmidt_density_19.csv")
    morph = dict(db.t47_morph(), callier=callier)
    H = 0.8 / 400.0 * 10.0 ** sd["aligned_log10_exposure"]
    return silver.silver_derive(400.0, universal(), eta_scale, H, sd["net_D"], FOG, morph, process="D-76 原液 20 °C 9 min"), sd


def H_at_net(spec, net):
    lh = np.linspace(-6, 2, 4001)
    Dn = silver.silver_curve(spec, lh) - silver.silver_curve(spec, [-12.0])[0]
    return float(10 ** np.interp(net, np.maximum.accumulate(Dn + 1e-12 * np.arange(len(Dn))), lh))


def _est(spec, sides, psf):
    return silver.silver_granularity(spec, H_at_net(spec, 0.682), sides, psf_um=psf, geometry="diffuse", adjacent=True)


@lru_cache(maxsize=4)
def calibrate(callier):
    """T47 的标定 (tools/eval/t47_trix.calibrate 的同一算法), 在给定的 Callier 作用方式下。"""
    gr = csv_cols("schmidt_granularity.csv")
    sides = gr["digitized_side_um"]; s2 = gr["sigma2_A_um2_D2"]
    target = float(np.mean(s2[2:])); _, ak_meas = schmidt_ak(sides, s2)
    psf, eta = 1.0, 8.0
    for _ in range(6):
        f_eta = lambda le: float(np.mean(_est(trix_spec(np.exp(le), callier)[0], sides[2:], psf) ** 2 * sides[2:] ** 2)) - target
        eta = float(np.exp(brentq(f_eta, np.log(0.5), np.log(200.0), xtol=1e-3)))
        sp = trix_spec(eta, callier)[0]
        try:
            psf_new = float(brentq(lambda s: schmidt_ak(sides, _est(sp, sides, s) ** 2 * sides ** 2)[1] - ak_meas, 0.05, 3.0, xtol=1e-3))
        except ValueError:
            psf_new = psf
        done = abs(psf_new - psf) < 2e-3
        psf = psf_new
        if done:
            break
    sp, sd = trix_spec(eta, callier)
    r = _est(sp, sides, psf)
    Hl = 0.8 / 400.0 * 10.0 ** sd["aligned_log10_exposure"]
    dif = silver.silver_density(sp, Hl) - silver.silver_density(sp, 0.0)
    k48 = 1000 * float(silver.silver_granularity(sp, H_at_net(sp, 1.0), [42.5], geometry="diffuse", adjacent=False)[0])
    return dict(callier=callier, eta_scale=eta, psf_um=psf, sides=sides.tolist(), measured=gr["corrected_RMS_D"].tolist(),
                closed=r.tolist(), rel=(r / gr["corrected_RMS_D"] - 1).tolist(), curve_max_resid=float(np.max(np.abs(sd["net_D"] - dif))),
                kodak48_x1000=k48, coat=list(sp.coat_g_m2), ecd=list(sp.ecd_um))


def film_spec(film, callier):
    """silvergrain 产品路径 (emulsion._silver_spec_d76) 的档案, 在给定方式与该方式的 η 下。"""
    from ..emulsion import _film
    from ..films import _bw_def
    f = _film(film); c = f.curve()
    if callier == "macro":
        eta, morph = db.universal()["t55g"]["eta3d_scale"], dict(db.t47_morph(), callier="macro")
    else:
        eta, morph = db.universal()["t47"]["eta3d_scale"], db.t47_morph()
    return silver.silver_derive(f.ei, universal(), eta, 0.8 / f.ei * 10.0 ** c["x"], c["net"], c["gross_fog"], morph,
                                process="BW-D76", source=_bw_def(film)["spec_source"], thickness_um=f.thickness_um)


def sigma_curve(spec, nets, side=SIDE48):
    lh = np.linspace(-6, 3, 1801)
    D = silver.silver_curve(spec, lh); f = float(silver.silver_density(spec, 0.0))
    Dm = np.maximum.accumulate(D - f + 1e-12 * np.arange(len(D)))
    return np.array([float(silver.silver_granularity(spec, 10 ** np.interp(n, Dm, lh), [side], adjacent=False)[0]) for n in nets])


def shape_vs_independent(callier):
    """独立数据的 σ(D)/σ(D_ref) 对模型 (只用净密度 ≥ min_net 的段); 另给 log σ 对 log D 的斜率 (指数)。"""
    ref = json.loads((DATA / "bw_granularity_vs_density.json").read_text(encoding="utf-8"))
    out = {}
    for key, d in ref["series"].items():
        if d["model_film"] is None:
            continue
        sp = film_spec(d["model_film"], callier)
        Dg = np.asarray(d["density"], float); s = np.asarray(d["sigma"], float)
        rows = []
        for fog in d["fog_band"]:
            net = Dg - fog
            ok = net >= d["min_net"]
            if ok.sum() < 3:
                continue
            m = sigma_curve(sp, net[ok])
            dr = s[ok] / np.interp(d["ref_net"], net[ok], s[ok]); mr = m / np.interp(d["ref_net"], net[ok], m)
            rows.append(dict(fog=fog, net=net[ok].tolist(), data_ratio=dr.tolist(), model_ratio=mr.tolist(),
                             rms_log=float(np.sqrt(np.mean(np.log(mr / dr) ** 2))),
                             exponent_data=float(np.polyfit(np.log(net[ok]), np.log(s[ok]), 1)[0]),
                             exponent_model=float(np.polyfit(np.log(net[ok]), np.log(m), 1)[0]),
                             top_data=float(dr[-1]), top_model=float(mr[-1])))
        out[key] = rows
    return out


def isprs_lines(callier):
    """ISPRS 的直线 (Kodak 2405 等): σ(1.4)/σ(0.7) 的分布, 对模型 Tri-X (D-76) 同一比值 (按净密度读; 毛 / 净未注明)。"""
    ref = json.loads((DATA / "bw_granularity_vs_density.json").read_text(encoding="utf-8"))["lines_isprs"]
    r = [row["s14"] / row["s07"] for row in ref["rows"]]
    sp = film_spec("kodak_trix_400", callier)
    out = {}
    for ap in (25.0, 50.0):
        m = sigma_curve(sp, [0.7, 1.4], side=ap * np.sqrt(np.pi) / 2)
        out[f"model_{ap:g}um"] = float(m[1] / m[0])
    return dict(data=r, data_median=float(np.median(r)), data_range=[float(min(r)), float(max(r))], **out)


class _macro_db:
    """把 T47b / T50 的检验临时换成 "macro" 方式与重标的 η (这些模块读 db.t47_morph / db.universal)。"""
    def __enter__(self):
        self.m, self.u = db.t47_morph, db.universal
        u = json.loads(json.dumps(self.u()))
        u["t47"]["eta3d_scale"] = u["t55g"]["eta3d_scale"]
        db.t47_morph = lambda: dict(self.m(), callier="macro")
        db.universal = lambda: u
        return self

    def __exit__(self, *a):
        db.t47_morph, db.universal = self.m, self.u


def legacy_checks():
    """T47b (官方 RMS、Schmidt 6 片) 与 T50 (显影液) 在两种方式下 (T47b / T50 本身的逐位检验仍是 "grain")。"""
    from . import t47_schmidt, t50_developers
    out = {}
    for c in ("grain", "macro"):
        if c == "macro":
            with _macro_db():
                b, q = t47_schmidt.t47b(), t50_developers.quick()
        else:
            b, q = t47_schmidt.t47b(), t50_developers.quick()
        dev = {d: {f: dict(pred=x["pred"]["rms_ratio"], meas=x["meas"]["rms_ratio"]) for f, x in v["films"].items()}
               for d, v in q["developers"].items()}
        out[c] = dict(t47b_official={a["key"]: a["t47_ratio"] for a in b["official"]},
                      t47b_schmidt={a["key"]: a["t47_ratio"] for a in b["schmidt"]}, t47b_summary=b["summary"],
                      t50_d76={k: v["rms_pred"] / v["rms_meas"] for k, v in q["d76"].items()}, t50_developers=dev)
    return out


def run(log=print):
    out = dict(calibration={c: calibrate(c) for c in ("grain", "macro")})
    log("  t55g calibration")
    out["shape"] = {c: shape_vs_independent(c) for c in ("grain", "macro")}
    out["isprs"] = {c: isprs_lines(c) for c in ("grain", "macro")}
    log("  t55g shape")
    out["legacy"] = legacy_checks()
    log("  t55g T47b / T50")
    return out
