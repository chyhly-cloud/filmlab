"""T55h: Ilford 三维晶体片 (HP5+、FP4+) —— 独立证据能定什么, 不能定什么。RPL 胶片侧读数只作留出。

独立证据 (data/ilford_3d_evidence.json):
  · Ilford 电影负片 fact sheet: HP5 Plus 漫射 RMS 16, FP4 Plus 10 (Kodak D-96, γ 0.65–0.70) —— D ≈ 1 的量级;
  · SMPTE 1968 的三维晶体片 σ–D 形状 (Tri-X 5233、4-X 5224; data/bw_granularity_vs_density.json) —— 形状的先验;
  · RPL 底片翻拍的特性曲线 γ(D) (曲线量, 不是颗粒读数) —— 冲印店 (XTOL) 的显影反差。
检验:
  1. 量级: 模型 (D-76 档案, 宏观 Callier) 在净 D 1.0、48 µm 的 RMS 对 Ilford 公布值, 与 Kodak / Schmidt 的同一比值并列 (模型的已知散布)。
  2. 形状: HP5+ 模型 σ ∝ D^p 的指数, 对三维晶体先验 (SMPTE 5233 / 4-X) 与 RPL 留出。
  3. 显影反差: 用 RPL 翻拍量出的特性曲线代替数据库曲线 (ILFOTEC HC 借用口径) 重新推导 HP5+ 的档案, 看留出的形状能解释多少。
"""
import json
from pathlib import Path

import numpy as np

from .. import database as db
from .. import silver
from ..emulsion import Emulsion, _film, derive, universal
from ..films import developer as _developer
from . import scanner_fit as S
from .t55g_callier import sigma_curve

DATA = Path(__file__).resolve().parent / "data"
NETS = (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5)


def _exponent(nets, s, lo=0.5, hi=1.5):
    n = np.asarray(nets); m = (n >= lo - 1e-9) & (n <= hi + 1e-9)
    return float(np.polyfit(np.log(n[m]), np.log(np.asarray(s)[m]), 1)[0])


def curve_from_gamma(gd, ei):
    """γ(D) (按净密度分档) → 特性曲线: 净 0.1 处 H = 0.8/EI, dD/dlogH = γ(D) 积分到 2.1; 趾部按 10^(1.2 Δlog H) 衰减。"""
    Ds = np.array(sorted(float(k) for k in gd)); gs = np.array([gd[k] for k in sorted(gd, key=float)], float)
    gs = np.convolve(np.pad(gs, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    lh0 = np.log10(0.8 / ei); D = [0.1]; lh = [lh0]
    while D[-1] < 2.1:
        lh.append(lh[-1] + 0.02); D.append(D[-1] + max(np.interp(D[-1], Ds, gs), 0.2) * 0.02)
    tl = np.arange(lh0 - 1.0, lh0, 0.02)
    return np.concatenate([tl, lh]), np.concatenate([0.1 * 10 ** ((tl - lh0) * 1.2), D])


def spec_with_curve(film, logH, net):
    f = _film(film); c = f.curve()
    morph = dict(db.t47_morph(), callier="macro")
    return silver.silver_derive(f.ei, universal(), db.universal()["t55g"]["eta3d_scale"], 10.0 ** np.asarray(logH), np.asarray(net),
                                c["gross_fog"], morph, process="RPL 翻拍曲线 (T55h 检验)", thickness_um=f.thickness_um)


def film_side_chain(em, evs=(-3, -2, -1, 0, 1, 2, 3, 4)):
    """validation.filmside 的同一读法 (120 原生网格 closed_form → 缩小 → 锐化, 密度域 σ_48), 对给定的 Emulsion。"""
    from ..image import GREY, expose, get_scanner, scan_grid, _to_output_grid, _sharpen3
    from ..render import render
    from .filmside import block_stats
    sc = get_scanner("noritsu_rpl")
    rows = []
    for ev in evs:
        ex = expose(np.full((S.N_OUT, S.N_OUT, 3), GREY * 2.0 ** ev), "ilford_hp5_plus", "d76", format=S.N_OUT * S.PX / 1000.0)
        g = scan_grid(ex.rgb.shape, ex.px_um, ex.format, sc, gauge="120")
        shp = tuple(g["shape_native"])
        rr = render(em, exposure=np.full(shp, float(ex.H.ravel()[0])), px_um=g["px_native"], seed=11 + ev, method="closed_form")
        Dm = float(np.asarray(rr.mean).mean())
        D = np.asarray(_to_output_grid(rr, shp, (S.N_OUT, S.N_OUT), 1).density, float)
        st = block_stats(_sharpen3(D, sc.sharpen_kx, sc.sharpen_ky) * 1000.0)
        rows.append(dict(ev=ev, D=Dm, s48=st["s48"]))
    return rows


def run(log=print):
    ev = json.loads((DATA / "ilford_3d_evidence.json").read_text(encoding="utf-8"))
    out = dict(level={}, shape={}, development={})
    from ..render import granularity
    for f, d in ev["ilford_mp_rms"].items():
        m = 1000 * float(granularity(derive(f, "d76"), 1.0))
        out["level"][f] = dict(official=d["rms"], model=m, ratio_model_over_official=m / d["rms"])
    log("  t55h level")
    fs = json.loads((DATA / "realscan_rpl_filmside.json").read_text(encoding="utf-8"))["film_side"]
    prior = json.loads((DATA / "bw_granularity_vs_density.json").read_text(encoding="utf-8"))["series"]
    for key in ("trix_5233_smpte1968", "4x_5224_smpte1968"):
        d = prior[key]
        out["shape"][f"prior_{key}"] = [_exponent(np.asarray(d["density"]) - fog, d["sigma"]) for fog in d["fog_band"]]
    for film, rk in (("kodak_trix_400", "6_KODAK400TX"), ("ilford_hp5_plus", "7_ILFORDHP5PLUS"), (None, "8_ILFORDDELTA3200PRO")):
        real = [r["s48"] for r in fs[rk]]; D = [r["D"] for r in fs[rk]]
        row = dict(real_exponent=_exponent(D, real))
        if film:
            row["model_exponent_48um"] = _exponent(NETS, sigma_curve(derive(film, "d76").spec, NETS))
        out["shape"][rk] = row
    log("  t55h shape")
    # 显影反差: RPL 翻拍的曲线 → HP5+ 档案 → 同一胶片侧读法 (留出比较)
    g = ev["rpl_negative_gamma"]
    hp5 = _film("ilford_hp5_plus")
    real = {r["D"]: r["s48"] for r in fs["7_ILFORDHP5PLUS"]}
    for lab, sp in (("database_curve", derive("ilford_hp5_plus", "d76").spec),
                    ("rpl_negative_curve", spec_with_curve("ilford_hp5_plus", *curve_from_gamma(g["7_ILFORDHP5PLUS"]["gamma_by_D"], hp5.ei)))):
        em = Emulsion(hp5, _developer("d76", hp5), 1.0, sp)
        rows = film_side_chain(em)
        Dm = [r["D"] for r in rows]; s = [r["s48"] for r in rows]
        ratio = {Dk: real[Dk] / float(np.interp(Dk, Dm, s)) for Dk in sorted(real)}
        m07 = float(np.interp(0.7, Dm, s)); m15 = float(np.interp(1.5, Dm, s))
        out["development"][lab] = dict(model=rows, real_over_model=ratio, model_shape_15_07=m15 / m07,
                                       real_shape_15_07=real[1.5] / real[0.7], s48_at_1=float(np.interp(1.0, Dm, s)))
        log(f"  t55h development {lab}")
    out["gamma_mid"] = {k: float(np.median([v for d, v in g[k]["gamma_by_D"].items() if 0.5 <= float(d) <= 1.3]))
                        for k in g}
    return out
