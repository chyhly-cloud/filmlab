"""真实扫描的量级检验 (只用记录的测量值, 不下载; data/realscan_reference.json)。物理是唯一的依据, 这里不拟合任何量。

条件 (与记录值一致): RPL 645 的 Noritsu 扫描约 20 µm/像素; Tri-X 用 XTOL (RPL 的黑白显影液), HP5+ 用 D-76 (没有 XTOL 档案,
T51b 同样处理); 均匀 18% 灰 × 2^EV 的照片 (EV −3 … +3), silvergrain apply_to_image 的两种模式, 冲印店扫描 Noritsu 开 / 关。
测量 (measure.py, 与真实同一套估计器): 原尺寸 σ_px / σ_hp / σ_48, 手机尺寸 (缩 2.33 倍, = RPL 2796 → 1200) σ_px / σ_hp,
输出密度的径向功率谱形状。
比较:
  · 同码值 (T51b / T52 的方法): 真实 / silvergrain 的码值颗粒。它含影调斜率之差: silvergrain 的 "full" 是按该片平均反差的
    直线反转 (不含冲印店影调曲线), RPL 的影调更陡; 所以另给
  · 胶片侧 (与影调无关): (真实 σ_48 / 真实斜率 dCode/dD) / (silvergrain σ_48 / silvergrain 斜率), 斜率 = 输出码值对底片密度的导数。
  · 谱形: 真实 RPL 黑白的亮度谱从 0.1 到 0.5 周/像素上升 (Tri-X 约 ×1.5, 来自 USM); 不经扫描的 silvergrain 场在 20 µm 上是白的。
"""
import numpy as np

from ..image import GREY, LN10, expose, finish, srgb_encode, _curve_tables, _slope_at, _avg_gradient
from ..render import render
from . import measure as M
from .refdata import load

PX = 20.0
N = 512
EVS = (-3, -2, -1, 0, 1, 2, 3)
PHONE_FACTOR = 2796 / 1200
CASES = (("kodak_trix_400", "xtol", "Tri-X 400 (RPL: XTOL)", "6_KODAK400TX"),
         ("ilford_hp5_plus", "d76", "HP5 Plus (filmlab 模型: D-76)", "7_ILFORDHP5PLUS"))


def _enc_slope(v):
    """d(255·sRGB 编码)/d ln(线性值) 在 v 处。"""
    v = np.maximum(v, 1e-9)
    e = srgb_encode(v * 1.001) - srgb_encode(v / 1.001)
    return 255.0 * e / (2 * np.log(1.001))


def patch(film, dev, ev, seed=1, k=1.0):
    img = np.full((N, N, 3), GREY * 2.0 ** ev)
    ex = expose(img, film, dev, k, format=N * PX / 1000.0)
    rr = render(ex.emulsion, exposure=ex.H, px_um=ex.px_um, seed=seed, method="explicit")
    return ex, rr


def outputs(ex, rr):
    """四种输出 (full / grain × 扫描开 / 关) 与每种的影调斜率 dCode/dD (亮度); 另加 "full+noritsu_discrete":
    扫描光学与 USM 的高斯用离散核 (filmlab 引擎的实现), 只为对照 T51b 的数。"""
    from .. import image as _im
    em = ex.emulsion
    tabs = _curve_tables(em)
    lg = np.log10(ex.info["H_grey_lux_s"])
    res = {}
    for mode, sc, impl in (("full", None, "continuous"), ("full", "noritsu_rpl", "continuous"), ("grain", None, "continuous"),
                           ("grain", "noritsu_rpl", "continuous"), ("full", "noritsu_rpl", "discrete")):
        if True:
            _im.GAUSS_IMPL = impl
            try:
                o, info = finish(ex, rr, mode, sc)
            finally:
                _im.GAUSS_IMPL = "continuous"
            Y = M.to_code(o)
            if mode == "full":
                lay = []
                for c, s in enumerate(info["scan"]):
                    Dm = float(np.mean(rr.mean.reshape(N, N, -1)[..., c]))
                    P = GREY * 10.0 ** ((Dm - s["D_grey"]) / s["gamma_bar"])
                    lay.append(_enc_slope(P) * LN10 / s["gamma_bar"])
                slope = float(np.mean(lay))
            else:
                v = float(ex.rgb[0, 0, 1])
                lH = np.log10(np.asarray(rr.exposure).reshape(N, N, -1)[0, 0])
                g = [float(_slope_at(tabs[c], lH[c])[0]) for c in range(len(tabs))]
                slope = float(np.mean([_enc_slope(v) * LN10 / gg for gg in g]))
            res[(mode, sc if impl == "continuous" else sc + "_discrete")] = (Y, slope, o)
    return res


def measure_case(film, dev, seed=1):
    rows = []
    for ev in EVS:
        ex, rr = patch(film, dev, ev, seed)
        outs = outputs(ex, rr)
        r = dict(ev=ev, D_mean=float(np.mean(rr.mean)), D_px_mean=float(np.mean(rr.density)))
        for (mode, sc), (Y, slope, o) in outs.items():
            nat = M.median_stats(M.tiles(Y, PX, synthetic=True))
            ph = M.median_stats(M.tiles(M.phone(Y, PHONE_FACTOR), PX * PHONE_FACTOR, synthetic=True, border=4))
            key = f"{mode}{'+' + sc.replace('noritsu_rpl', 'noritsu') if sc else ''}"
            r[key] = dict(native=nat, phone=ph, slope=slope)
            if mode == "full" and ev == 0 and key in ("full", "full+noritsu", "full+noritsu_discrete"):
                Do = -np.log10(np.clip(np.asarray(o if o.ndim == 2 else o.mean(-1)), 1e-6, None))
                fc, P = M.spectrum(Do)
                r[key]["spectrum"] = dict(fc=fc.tolist(), P=P.tolist())
        rows.append(r)
    return rows


def _interp(rows, key, scale, q, code):
    c = np.array([r[key][scale]["code"] for r in rows]); v = np.array([r[key][scale][q] for r in rows])
    s = np.array([r[key]["slope"] for r in rows])
    o = np.argsort(c)
    if not (c.min() <= code <= c.max()):
        return None, None
    return float(np.interp(code, c[o], v[o])), float(np.interp(code, c[o], s[o]))


def compare(rows, film_label, rpl_key):
    ref = load("realscan_reference.json")
    out = dict(t51b=[], t52_phone=[], t52_native=[])
    for b in ref["t51b_same_code"]["films"][film_label]:
        for key in ("full+noritsu", "grain+noritsu", "full+noritsu_discrete"):
            s48, sl = _interp(rows, key, "native", "s48", b["code"])
            spx, _ = _interp(rows, key, "native", "s_px", b["code"])
            if s48 is None:
                continue
            out["t51b"].append(dict(mode=key, code=b["code"], real_s48=b["real"]["s48"], sg_s48=s48, ratio_code=b["real"]["s48"] / s48,
                                    real_s_px=b["real"]["s_px"], sg_s_px=spx, ratio_px=b["real"]["s_px"] / spx,
                                    real_slope=b["real_slope"], sg_slope=sl,
                                    film_side=(b["real"]["s48"] / b["real_slope"]) / (s48 / sl),
                                    filmlab_film_side=(b["real"]["s48"] / b["real_slope"]) / (b["filmlab_model"]["s48"] / b["filmlab_slope"])))
    for a in ref["t52_anchor"]["rows"]:
        if a["film"] != rpl_key:
            continue
        scale = a["scale"]
        for key in ("full+noritsu", "grain+noritsu"):
            spx, sl = _interp(rows, key, scale, "s_px", a["code"])
            shp, _ = _interp(rows, key, scale, "s_hp", a["code"])
            if spx is None:
                continue
            out[f"t52_{scale}"].append(dict(mode=key, code=a["code"], real_px=a["real_px"], sg_px=spx, ratio_px=a["real_px"] / spx,
                                            real_hp=a["real_hp"], sg_hp=shp, ratio_hp=a["real_hp"] / shp,
                                            filmlab_ratio_px=a["ratio_px"]))
    # 谱形: P(f) / P(0.11 周/像素)
    sp = ref["t51_spectrum_rpl_bw"]
    real = np.asarray(sp["tri_x" if "trix" in rpl_key.lower() or "400TX" in rpl_key else "hp5"])
    r0 = next(r for r in rows if r["ev"] == 0)
    shapes = {}
    for key in ("full", "full+noritsu", "full+noritsu_discrete"):
        P = np.asarray(r0[key]["spectrum"]["P"])
        shapes[key] = (P / P[3]).tolist()
    out["spectrum_shape"] = dict(fc=sp["fc"], real=(real / real[3]).tolist(), **shapes)
    return out


def run(log=print):
    res = {}
    for film, dev, label, rkey in CASES:
        rows = measure_case(film, dev)
        res[f"{film}@{dev}"] = dict(rows=rows, compare=compare(rows, label, rkey))
        log(f"  realscan {film}@{dev}")
    res["portra400"] = colour_case()
    return res


def colour_case():
    """Portra 400 / C-41 (RPL Noritsu): σ_48 亮度对码值, 与 T54 的真实记录 (同码值) 比较。"""
    ref = load("realscan_reference.json")["t54_rpl_portra400"]
    rows = []
    for ev in (-2, -1, 0, 1, 2, 3):
        ex, rr = patch("kodak_portra_400", "C-41", ev)
        for mode in ("full", "grain"):
            o, info = finish(ex, rr, mode, "noritsu_rpl")
            st = M.median_stats(M.tiles(M.to_code(o), PX, synthetic=True))
            rows.append(dict(ev=ev, mode=mode, **st))
    cmp = []
    for mode in ("full", "grain"):
        rr_ = [r for r in rows if r["mode"] == mode]
        c = np.array([r["code"] for r in rr_]); v = np.array([r["s48"] for r in rr_]); o = np.argsort(c)
        for b in ref["same_code"]:
            if c.min() <= b["code"] <= c.max():
                sg = float(np.interp(b["code"], c[o], v[o]))
                cmp.append(dict(mode=mode, code=b["code"], real=b["real"], sg=sg, ratio=b["real"] / sg, filmlab_ratio=b["ratio"]))
    return dict(rows=rows, compare=cmp)


# ------------------------------------------------------------------ T55b: 冲印店实测影调 --
LAB_EVS = (-3, -2, -1, 0, 1, 2)
COLOUR_RPL = (("kodak_portra_400", "2_KODAKPORTRA400"), ("kodak_portra_160", "4_KODAKPORTRA160"),
              ("kodak_portra_800", "3_KODAKPORTRA800"), ("kodak_ektar_100", "5_KODAKEKTAR100"),
              ("fujifilm_pro_400h", "1_FUJIPRO400H"))


def lab_rows(film, dev, mode="grain", n=N, seed=1):
    """grain 模式 (T55b, 冲印店影调 rpl_noritsu, Noritsu 空间传递) 的均匀照片 18% × 2^EV: 原尺寸 / 手机尺寸的颗粒量。
    full 模式用 timing="fixed" (均匀照片在自动定光下都落到同一码值)。"""
    import silvergrain as sg
    rows = []
    for ev in LAB_EVS:
        img = np.full((n, n, 3), GREY * 2.0 ** ev)
        kw = dict(timing="fixed", exposure_ev=ev) if mode == "full" else {}
        if mode == "full":
            img = np.full((n, n, 3), GREY)
        r = sg.apply_to_image(img, film, dev, format=n * PX / 1000.0, mode=mode, scanner="noritsu_rpl", seed=seed,
                              method="explicit", **kw)
        Y = M.to_code(r.linear)
        rows.append(dict(ev=ev, native=M.median_stats(M.tiles(Y, PX, synthetic=True)),
                         phone=M.median_stats(M.tiles(M.phone(Y, PHONE_FACTOR), PX * PHONE_FACTOR, synthetic=True, border=4)),
                         slope=r.info.get("slope_code_per_D")))
    return rows


def _at(rows, scale, q, code):
    c = np.array([r[scale]["code"] for r in rows]); v = np.array([r[scale][q] for r in rows]); o = np.argsort(c)
    return float(np.interp(code, c[o], v[o])) if c.min() <= code <= c.max() else None


def lab_compare(log=print):
    ref = load("realscan_reference.json")
    out = {}
    for (film, dev, label, rkey) in CASES:
        res = {}
        for mode in ("grain", "full"):
            rows = lab_rows(film, dev, mode)
            cmp = dict(t51b=[], t52=[])
            for b in ref["t51b_same_code"]["films"][label]:
                s48 = _at(rows, "native", "s48", b["code"]); spx = _at(rows, "native", "s_px", b["code"])
                if s48 is not None:
                    cmp["t51b"].append(dict(code=b["code"], real_s48=b["real"]["s48"], sg_s48=s48, ratio_s48=b["real"]["s48"] / s48,
                                            real_s_px=b["real"]["s_px"], sg_s_px=spx, ratio_px=b["real"]["s_px"] / spx,
                                            filmlab_ratio_s48=b["real"]["s48"] / b["filmlab_model"]["s48"]))
            for a in ref["t52_anchor"]["rows"]:
                if a["film"] != rkey:
                    continue
                spx = _at(rows, a["scale"], "s_px", a["code"])
                if spx is not None:
                    cmp["t52"].append(dict(scale=a["scale"], code=a["code"], real_px=a["real_px"], sg_px=spx,
                                           ratio_px=a["real_px"] / spx, filmlab_ratio_px=a["ratio_px"]))
            res[mode] = dict(rows=rows, compare=cmp)
        out[f"{film}@{dev}"] = res
        log(f"  realscan lab {film}@{dev}")
    rpl = load("realscan_reference.json")
    import json
    from pathlib import Path
    col = json.loads((Path(__file__).resolve().parent / "data" / "realscan_colour_reference.json").read_text(encoding="utf-8"))
    for film, rkey in COLOUR_RPL:
        rows = lab_rows(film, None, "grain")
        cmp = []
        for b in col["films"][rkey]["same_code"]:
            s48 = _at(rows, "native", "s48", b["code"])
            if s48 is not None:
                cmp.append(dict(code=b["code"], real=b["real"], sg=s48, ratio=b["real"] / s48, filmlab_ratio=b["ratio"]))
        out[f"{film}@C-41"] = dict(grain=dict(rows=rows, compare=cmp))
        log(f"  realscan lab {film}")
    return out
