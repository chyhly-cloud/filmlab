"""T54: 彩色颗粒量级的模型侧分解 (docs/tasks/T54_COLOUR_GRAIN_LEVEL.md)。

子命令 (模型与真实同一套量; 真实见 t54_real.py):
  spectral 输出.json
      扫描仪通道与 Status M 看染料密度起伏的比例 (第 5 节)。每种胶片在模型自己的中灰底片上 (显影后的染料量 d0, 片基 + 遮罩),
      仪器 j 对第 c 层染料量的局部导数 J_jc = ∂D_j/∂C_c = Σ_λ S_j T E_c / Σ_λ S_j T (T 为中灰的光谱透射率)。
      仪器: Status M (ISO 5-3, Giorgianni & Madden 2009 的数字化表, tools/eval/t54/status_m.json)、仓库的 Status M 近似
      (densitometry.py 的高斯)、仓库的 Frontier / Noritsu 扫描仪 LED (scanner.SCANNERS)。
  jac 输出.json
      整条链 (代际 t51_base; 800T + Frontier, Portra 400 + Noritsu) 在均匀灰上的局部雅可比: 8 位输出码值对每层染料量
      (固定定光 timing="gray", 均匀扰动不触发定光), 以及 dCode/dlogE、dC/dlogE、每层 48 µm 胶片颗粒 (闭式, layer_curves)。
      由此得到每通道与亮度的预测颗粒、相关、"颜色换算项" (第 3 节)。另算几种已有开关的变体 (paper_channel、无输出调整、
      Kodak 原始 5219 染料谱)。
  uniform 配方 冲印店 像素µm 边长 输出.json [EV ...]
      颗粒开的均匀灰 (t51_base, 指定冲印店的空间传递), 与真实同一套颗粒量 (t54_real.grain_stats)。
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

LUMA = np.array([0.299, 0.587, 0.114])          # BT.601, 编码域 (冲印店软件与 T51 的亮度)


def status_m_table():
    from filmlab.stocks import WAVELENGTHS
    d = json.load(open(HERE / "t54" / "status_m.json"))
    S = []
    for c in "rgb":
        a = np.array(d[c]); a = a[np.argsort(a[:, 0])]
        S.append(np.interp(WAVELENGTHS, a[:, 0], a[:, 1], left=0.0, right=0.0))
    return np.stack(S, 1)


def instruments():
    from filmlab import densitometry
    from filmlab.scanner import SCANNERS
    from filmlab.scanning.optics import led_response
    return {"status_m_iso": status_m_table(), "status_m_repo": densitometry.status_response("M"),
            "frontier_led": led_response(SCANNERS["frontier"]), "noritsu_led": led_response(SCANNERS["noritsu"])}


def inst_jac(S, T, E):
    """仪器 S (81, 3) 在光谱透射率 T (81,) 处对染料 E (81, 3) 单位浓度的局部密度导数 (3 通道 × 3 层)。"""
    w = S * T[:, None]
    return (w.T @ E) / w.sum(0)[:, None]


def bandwidth(S):
    from filmlab.stocks import WAVELENGTHS
    out = []
    for j in range(3):
        y = S[:, j] / S[:, j].max()
        x = np.linspace(380, 780, 4001); yi = np.interp(x, WAVELENGTHS, y)
        h = x[yi >= 0.5]
        out.append([float(x[np.argmax(yi)]), float(h.max() - h.min())])
    return out


def engine_for(rec, gen="t51_base", lab=None, extra=(), conditioning="lab", revision=None, n=4):
    from filmlab import engine, generation, stocks, materials
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    generation.use(gen)
    if revision:
        stocks.REVISIONS.update(revision); stocks.load_stock.cache_clear(); materials.emulsion.cache_clear()
    kw = {**recipe(rec), "timing": "gray", "roll": None, "frame": 0, "scene_lights": None,
          "corrections": tuple(engine.DEFAULT_CORRECTIONS) + tuple(extra), "output_conditioning": conditioning}
    if lab:
        kw["scan_spatial"] = lab
    return lambda px_um, grain=0.0: Engine(Params(**kw, grain=grain, frame_width_mm=n * px_um / 1000.0))


def code_of(eng, neg):
    from filmlab.scanning.spatial import encode
    tr = {}
    o = eng.scan(neg, trace=tr)
    return encode(np.clip(o, 0, None)).reshape(-1, 3).mean(0) * 255.0, tr["Dn"].reshape(-1, 3).mean(0)


def perturbed(neg, dc):
    from copy import copy
    from filmlab.chemistry import FilmState
    st = neg.state
    s2 = FilmState(latent=None, halide=st.halide, silver=st.silver, dye=st.dye + dc, coupler=None, remjet=st.remjet,
                   dye_agent=st.dye_agent)
    n2 = copy(neg); n2.state = s2
    return n2


def film_sigma48(eng, d):
    """每层 48 µm 漫射 RMS (染料量单位, 闭式) 在染料量 d (3,) 处: 按 grain.apply 的反求 (H_eff 使名义净密度 = d)。"""
    from filmlab import grain
    from filmlab.materials import grain_spec_t46
    spec = grain_spec_t46(eng.e, eng.proc.name)
    out = []
    for c in range(3):
        lh, Dt = grain._state_table(spec, c)
        o = np.argsort(Dt, kind="stable")
        H = 10.0 ** np.interp(min(max(d[c], 0.0), Dt.max()), Dt[o], lh[o])
        out.append(float(grain.layer_curves(spec, c, np.array([H]))[1][0]) * max(d[c] / Dt.max(), 1.0))
    return np.array(out)


def chain_point(mk, px_um, ev, delta=0.01):
    """均匀灰 18% × 2^ev: 码值、J_code (码值 / 单位染料)、dCode/dlogE、dC/dlogE、每层胶片 σ48。"""
    eng = mk(px_um)
    neg = eng.develop_negative(np.full((4, 4, 3), 0.18 * 2.0 ** ev))
    c0, dn0 = code_of(eng, neg)
    d0 = neg.state.dye.reshape(-1, 3).mean(0)
    J = np.zeros((3, 3)); Jdn = np.zeros((3, 3))
    for c in range(3):
        e = np.zeros(3); e[c] = delta
        cp, dp = code_of(eng, perturbed(neg, e))
        cm, dm = code_of(eng, perturbed(neg, -e))
        J[:, c] = (cp - cm) / (2 * delta); Jdn[:, c] = (dp - dm) / (2 * delta)
    h = 0.1 * np.log10(2.0)
    res = []
    for s in (-1, 1):
        e2 = mk(px_um)
        n2 = e2.develop_negative(np.full((4, 4, 3), 0.18 * 2.0 ** (ev + s * 0.1)))
        res.append((code_of(e2, n2)[0], n2.state.dye.reshape(-1, 3).mean(0)))
    slope_E = (res[1][0] - res[0][0]) / (2 * h)
    dC_dE = (res[1][1] - res[0][1]) / (2 * h)
    sig = film_sigma48(eng, d0)
    T = eng._transmittance(neg.state).reshape(-1, 81)[0]
    E = eng._dye_spectra(neg.state.dye_agent)
    return dict(ev=ev, code=c0.tolist(), luma=float(LUMA @ c0), dye=d0.tolist(), Dn=dn0.tolist(), J_code=J.tolist(),
                J_Dn=Jdn.tolist(), slope_E=slope_E.tolist(), slope_E_luma=float(LUMA @ slope_E), dC_dlogE=dC_dE.tolist(),
                sigma48_film=sig.tolist(), T=T.tolist(), E=E.tolist())


def predict(p, sig=None):
    """由 J 与独立的层颗粒预测每通道 / 亮度的 48 µm 码值颗粒与相关; 颜色换算项 = 实际 / 只看本层 (对角)。"""
    J = np.array(p["J_code"]); s = np.array(p["sigma48_film"] if sig is None else sig)
    C = J @ np.diag(s ** 2) @ J.T
    sd = np.sqrt(np.diag(C))
    yl = LUMA @ J
    luma = float(np.sqrt((yl ** 2 * s ** 2).sum()))
    diag = np.abs(np.diag(J)) * s                      # 只有本层 (对角) 时
    nee = sd / np.abs(np.array(p["slope_E"]))          # 噪声等效曝光 (logE)
    nee_layer = s / np.abs(np.array(p["dC_dlogE"]))    # 本层自己的噪声等效曝光
    corr = C / np.outer(sd, sd)
    return dict(sigma48_code=sd.tolist(), sigma48_luma=luma, mix=(sd / diag).tolist(),
                corr_rg_gb_rb=[float(corr[0, 1]), float(corr[1, 2]), float(corr[0, 2])],
                nee=nee.tolist(), nee_luma=luma / abs(p["slope_E_luma"]), nee_layer=nee_layer.tolist(),
                luma_layer_share=((yl ** 2 * s ** 2) / (yl ** 2 * s ** 2).sum()).tolist())


SETUPS = {
    # 名字: (配方, 代际, 冲印店, 附加修正, 输出调整, 修订, 像素 µm)
    "800T_frontier": ("cinestill_800t", "t51_base", None, (), "lab", None, 6.6),
    "800T_frontier_paperch": ("cinestill_800t", "t51_base", None, ("paper_channel",), "lab", None, 6.6),
    "800T_frontier_nocond": ("cinestill_800t", "t51_base", None, (), None, None, 6.6),
    "800T_frontier_paperch_nocond": ("cinestill_800t", "t51_base", None, ("paper_channel",), None, None, 6.6),
    "800T_frontier_k5219raw": ("cinestill_800t", "t51_base", None, (), "lab", {"kodak_vision3_500t": "kodak_raw"}, 6.6),
    "P400_noritsu": ("kodak_portra_400", "t51_base", "noritsu_rpl", (), "lab", None, 20.0),
    "P160_noritsu": ("kodak_portra_160", "t51_base", "noritsu_rpl", (), "lab", None, 20.0),
    "P800_noritsu": ("kodak_portra_800", "t51_base", "noritsu_rpl", (), "lab", None, 20.0),
    "E100_noritsu": ("kodak_ektar_100", "t51_base", "noritsu_rpl", (), "lab", None, 20.0),
    "F400H_noritsu": ("fujifilm_pro_400h", "t51_base", "noritsu_rpl", (), "lab", None, 20.0),
    "P400_noritsu_paperch": ("kodak_portra_400", "t51_base", "noritsu_rpl", ("paper_channel",), "lab", None, 20.0),
    "P400_noritsu_nocond": ("kodak_portra_400", "t51_base", "noritsu_rpl", (), None, None, 20.0),
}


def cmd_jac(dst, names=None, evs=(-3, -2, -1, 0, 1, 2)):
    out = json.load(open(dst)) if Path(dst).exists() else {}
    for name in (names or SETUPS):
        if name in out:
            continue
        rec, gen, lab, extra, cond, rev, px = SETUPS[name]
        mk = engine_for(rec, gen, lab, extra, cond, rev)
        pts = []
        for ev in evs:
            p = chain_point(mk, px, ev)
            p["pred"] = predict(p)
            pts.append(p)
            print(name, ev, np.round(p["code"], 1), "J diag", np.round(np.diag(p["J_code"]), 1),
                  "σ48 film", np.round(p["sigma48_film"], 4), "pred luma", round(p["pred"]["sigma48_luma"], 2),
                  "mix", np.round(p["pred"]["mix"], 2), "corr", np.round(p["pred"]["corr_rg_gb_rb"], 2), flush=True)
        out[name] = pts
        json.dump(out, open(dst, "w"))


def cmd_spectral(dst):
    from filmlab import generation, stocks, materials
    from filmlab.stocks import WAVELENGTHS
    S = instruments()
    out = {"bandwidth_peak_fwhm_nm": {k: bandwidth(v) for k, v in S.items()}, "films": {}}
    films = [("800T", "cinestill_800t", None), ("800T_kodak_raw", "cinestill_800t", {"kodak_vision3_500t": "kodak_raw"}),
             ("Portra 400", "kodak_portra_400", None), ("Portra 160", "kodak_portra_160", None),
             ("Portra 800", "kodak_portra_800", None), ("Ektar 100", "kodak_ektar_100", None),
             ("Pro 400H", "fujifilm_pro_400h", None)]
    for label, rec, rev in films:
        mk = engine_for(rec, "t51_base", None, (), "lab", rev)
        rows = []
        for ev in (-2, 0, 2):
            eng = mk(20.0)
            neg = eng.develop_negative(np.full((4, 4, 3), 0.18 * 2.0 ** ev))
            T = eng._transmittance(neg.state).reshape(-1, 81)[0]
            E = eng._dye_spectra(neg.state.dye_agent)
            Js = {k: inst_jac(v, T, E) for k, v in S.items()}
            r = {"ev": ev, "dye": neg.state.dye.reshape(-1, 3).mean(0).tolist(), "J": {k: v.tolist() for k, v in Js.items()}}
            M = Js["status_m_iso"]
            for k in ("frontier_led", "noritsu_led", "status_m_repo"):
                r[f"diag_ratio_{k}_over_statusM"] = (np.diag(Js[k]) / np.diag(M)).tolist()
            rows.append(r)
            print(label, ev, "J_M diag", np.round(np.diag(M), 3), "Frontier/M", np.round(r["diag_ratio_frontier_led_over_statusM"], 3),
                  "Noritsu/M", np.round(r["diag_ratio_noritsu_led_over_statusM"], 3), flush=True)
        out["films"][label] = rows
        generation.use("t51_base"); stocks.REVISIONS.clear(); stocks.load_stock.cache_clear(); materials.emulsion.cache_clear()
    json.dump(out, open(dst, "w"), indent=1)


def cmd_uniform(name, dst, n=512, evs=(-3, -2, -1, 0, 1, 2, 3)):
    """颗粒开的均匀灰 (SETUPS[name] 的链), 同一套颗粒量 (t54_real.grain_stats, synthetic); 另记颗粒关的斜率 (chain_point)。"""
    from t54_real import grain_stats
    from filmlab.scanning.spatial import encode
    rec, gen, lab, extra, cond, rev, px = SETUPS[name]
    out = json.load(open(dst)) if Path(dst).exists() else {}
    for ev in evs:
        key = f"{name}/{ev:+d}"
        if key in out:
            continue
        mk = engine_for(rec, gen, lab, extra, cond, rev, n=n)
        eng = mk(px, grain=1.0)
        neg = eng.develop_negative(np.full((n, n, 3), 0.18 * 2.0 ** ev))
        tr = {}
        o = eng.scan(neg, trace=tr)
        img = encode(np.clip(o, 0, None)) * 255.0
        rows = grain_stats(img[8:-8, 8:-8], px, synthetic=True)
        Dn = tr["Dn"][8:-8, 8:-8]
        p = chain_point(engine_for(rec, gen, lab, extra, cond, rev), px, ev)
        p["pred"] = predict(p)
        for k in ("T", "E"):
            p.pop(k)
        out[key] = dict(rows=np.round(rows, 4).tolist(), Dn_mean=Dn.reshape(-1, 3).mean(0).tolist(), point=p)
        med = np.median(rows, 0)
        print(key, "code", np.round(med[2:6], 1), "σ48", np.round(med[18:22], 2), "hp25", np.round(med[14:18], 2),
              "ρ48", np.round(med[22:25], 2), "slopeE", np.round(p["slope_E"], 1), flush=True)
        json.dump(out, open(dst, "w"))


if __name__ == "__main__":
    if sys.argv[1] == "spectral":
        cmd_spectral(sys.argv[2])
    elif sys.argv[1] == "uniform":
        cmd_uniform(sys.argv[2], sys.argv[3], evs=tuple(int(v) for v in sys.argv[4:]) or (-3, -2, -1, 0, 1, 2, 3))
    elif sys.argv[1] == "jac":
        cmd_jac(sys.argv[2], sys.argv[3:] or None)
