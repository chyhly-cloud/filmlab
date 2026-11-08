"""T53 汇总: 闭式解 / 显式模拟检验、整条链的均匀灰、颗粒粗细 k 的"真实"对应 (锚), 整帧 → tools/eval/t53/summary.json。
用法: python3 -I t53_summary.py   (读 tools/eval/t53/validate.json、uniform.json、scenes.json 与 tools/eval/t52/uniform_model.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
D = HERE / "t53"


def interp_k(ks, ys, target):
    """σ(k) 单调: 在 log–log 上线性插值求 σ(k) = target 的 k (超出范围时外推一段)。"""
    lk, ly = np.log(ks), np.log(ys)
    o = np.argsort(ly)
    return float(np.exp(np.interp(np.log(target), ly[o], lk[o], left=np.nan, right=np.nan)))


def pixel_jensen(key="kodak_trix_400", ks=(0.5, 1.0, 1.5, 2.0, 3.0), px=36000 / 3032):
    """逐像素取对数的平均密度偏差 (Jensen): E[−log10 T_px] − (−log10 E[T]) ≈ ln10/2 · σ_D,px² (闭式, 12 µm 方孔)。"""
    from filmlab import generation, grain, materials
    generation.use("t51_bw")
    sp = materials.silver_spec(materials.bw_silver_film(key))
    lh = np.linspace(-4.0, 0.0, 2001)
    H0 = 10.0 ** np.interp(0.682, grain.silver_curve(sp, lh) - grain.silver_density(sp, 0.0), lh)
    out = {}
    for k in ks:
        s = float(grain.silver_granularity(grain.scale_silver(sp, k), H0, (px,), adjacent=False)[0])
        out[str(k)] = dict(sigma_D_px=s, dD_pixel_log=float(np.log(10) / 2 * s * s))
    return out


def d_half(sides, sig, plateau):
    """Selwyn 起始长度: σ²A(d) 升到平台 (96 µm 闭式) 一半处的方孔边长 d½ (log 插值); 颗粒结构 ×k 时 d½ ∝ k。"""
    y = np.asarray(sig, float) ** 2 * np.asarray(sides, float) ** 2 / plateau
    i = int(np.argmax(y >= 0.5))
    if i == 0:
        return float("nan")
    return float(np.exp(np.interp(0.5, [y[i - 1], y[i]], np.log([sides[i - 1], sides[i]]))))


def main():
    res = {}
    v = json.load(open(D / "validate.json"))
    tab = {}
    res["engine_px"] = v.get("engine_px")
    for key, r in v.items():
        if "ks" not in r:
            continue
        rows = {}
        for k, x in r["ks"].items():
            if "coat_g_m2" in x:                        # 银
                rows[k] = dict(
                    coat_ratio=x["coat_ratio"], covering_power=x["covering_power"],
                    curve_max_abs_dD=x["curve_max_abs_dD"], dD_explicit_minus_closed=x["explicit_vs_closed"]["dD"],
                    selwyn_over_k_closed={d: round(y, 4) for d, y in x["selwyn_ratio_over_k"].items()},
                    selwyn_over_k_explicit={d: round(y, 3) for d, y in x["selwyn_ratio_over_k_explicit"].items()
                                            if np.isfinite(y)},
                    sigma_k_eq_sigma1_d_over_k=x["sigma_eq_k1_at_d_over_k"],
                    explicit_over_closed_sigma={d: round(y, 3) for d, y in x["explicit_vs_closed"]["sigma"].items()},
                    r_e_um=dict(closed=x["r_e_closed"], closed_px=x["r_e_closed_px"], explicit=x["r_e_explicit"]),
                    r_e_over_k=x["r_e_ratio_over_k"],
                    explicit_sigma_k_d_over_sigma1_d_over_k=x["explicit_sigma_k_d_over_sigma1_d_over_k"])
                from t53_validate import AK_SIDES
                pl = x["sigma"]["96.0"] ** 2 * 96.0 ** 2
                rows[k]["d_half_um"] = dict(closed=d_half(AK_SIDES, x["sigma_ak_closed"], pl),
                                            explicit=d_half(AK_SIDES, x["sigma_ak_explicit"], pl))
            else:                                       # 染料
                rows[k] = dict(curve_max_abs_dD=x["curve_max_abs_dD"],
                               mean_explicit_minus_target=x["explicit_mean_minus_target"],
                               selwyn48_over_k_closed=x["selwyn48_ratio_over_k"],
                               selwyn_over_k_explicit_12_24_48=x["selwyn_ratio_over_k_explicit"],
                               explicit48_over_closed=x["explicit48_vs_closed"],
                               r_e_um_explicit=x["r_e_explicit"], r_e_over_k_explicit=x["r_e_ratio_over_k_explicit"],
                               cloud_over_k=x["cloud_ratio_over_k"])
        if "1.0" in rows and "d_half_um" in rows["1.0"]:
            for k, x in rows.items():
                x["d_half_over_k"] = {m: x["d_half_um"][m] / rows["1.0"]["d_half_um"][m] / float(k) for m in ("closed", "explicit")}
        tab[key] = rows
    res["validate"] = tab
    res["pixel_jensen_trix"] = pixel_jensen()

    u = json.load(open(D / "uniform.json"))
    u52 = json.load(open(HERE / "t52" / "uniform_model.json"))
    ks = sorted({float(n.split("_k")[1].split("/")[0]) for n in u if n.startswith("35_trix_d76_k")})
    curve = {}
    for k in ks:
        r = u[f"35_trix_d76_k{k:g}/+0"]
        curve[str(k)] = dict(native_s_px=r["native"]["s_px"], phone_s_px=r["phone"]["s_px"], mean_code=r["mean_code"],
                             mean_code_phone=r["mean_code_phone"], mean_linear=float(np.mean(r["mean_linear"])),
                             rho_x=r["field"]["rho_x"], R_phone=r.get("phone_visibility", {}).get("R_max"),
                             R_100=r.get("native_visibility_100pct", {}).get("R_max"))
        for ev in (-1, 1):
            q = u.get(f"35_trix_d76_k{k:g}/{ev:+d}")
            if q:
                curve[str(k)][f"ev{ev:+d}"] = dict(native_s_px=q["native"].get("s_px"), phone_s_px=q["phone"].get("s_px"),
                                                  mean_code=q["mean_code"])
    res["uniform_trix_d76"] = curve
    kk = np.array(ks)
    nat = np.array([curve[str(k)]["native_s_px"] for k in ks])
    pho = np.array([curve[str(k)]["phone_s_px"] for k in ks])
    res["fit"] = dict(native_loglog_slope=float(np.polyfit(np.log(kk), np.log(nat), 1)[0]),
                      phone_loglog_slope=float(np.polyfit(np.log(kk), np.log(pho), 1)[0]))
    # 锚 (只报告): Rodinal 与半格各相当于多大的 k
    anc = {}
    for src, tab_ in (("t53", u), ("t52", u52)):
        rk = "35_trix_rodinal_k1/+0" if src == "t53" else "35_trix_rodinal/+0"
        hk = "half_trix_rodinal_k1/+0" if src == "t53" else "half_trix_rodinal/+0"
        if rk in tab_:
            ro = tab_[rk]
            anc[f"rodinal_{src}"] = dict(native_s_px=ro["native"]["s_px"], phone_s_px=ro["phone"]["s_px"],
                                         k_native=interp_k(kk, nat, ro["native"]["s_px"]),
                                         k_phone=interp_k(kk, pho, ro["phone"]["s_px"]))
        if rk in tab_ and hk in tab_:
            ratio = tab_[hk]["phone"]["s_px"] / tab_[rk]["phone"]["s_px"]
            anc[f"half_vs_35_rodinal_{src}"] = dict(ratio_phone=ratio, k_phone=interp_k(kk, pho / pho[ks.index(1.0)], ratio))
    if "half_trix_d76_k1/+0" in u:
        h = u["half_trix_d76_k1/+0"]
        anc["half_d76_t53"] = dict(phone_s_px=h["phone"]["s_px"], ratio_phone=h["phone"]["s_px"] / curve["1.0"]["phone_s_px"],
                                   k_phone=interp_k(kk, pho, h["phone"]["s_px"]),
                                   mean_code=h["mean_code"])
    res["anchors"] = anc
    for name in ("35_portra400_k1/+0", "35_portra400_k2/+0"):
        if name in u:
            r = u[name]
            res.setdefault("uniform_portra400", {})[name] = dict(native_s_px=r["native"].get("s_px"),
                                                                 phone_s_px=r["phone"].get("s_px"),
                                                                 mean_code=r["mean_code"],
                                                                 mean_linear=r["mean_linear"])
    if (D / "scenes.json").exists():
        sc = json.load(open(D / "scenes.json"))
        res["scenes"] = {n: dict(px_um=r["meta"]["px_um"], native=r["native"]["mid"], phone={a: b for a, b in r["phone"]["mid"].items()
                                                                                          if a != "band_rms"},
                                 R_phone=r["phone"].get("visibility", {}).get("R_max"),
                                 R_100=r["native"].get("visibility_100pct", {}).get("R_max")) for n, r in sc.items()}
    json.dump(res, open(D / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({a: res[a] for a in ("fit", "anchors")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
