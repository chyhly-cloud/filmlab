"""T54: 汇总 (docs/tasks/T54_COLOUR_GRAIN_LEVEL.md 的表) → tools/eval/t54/summary.json。

输入 (测量, 无图像): tools/eval/t54/ 下的 spectral.json、jac.json、uniform_model.json.gz、epfl_real.json.gz、
epfl_tone.json、epfl_tiles.json.gz、rpl_real.json.gz。
分解 (每档; 亮度 Y 与 R、G、B):
  总比 (同码值) = 斜率 (dCode/dlogE, 真实 / 模型同码值) × 位置 (模型自己的 NEE: 真实块所在曝光 / 同码值的曝光)
                × 胶片侧 (同曝光的 NEE 真实 / 模型)。
  通道相关项 (亮度): f = (σ_Y / σ_Y,独立)_真实 / (σ_Y / σ_Y,独立)_模型, σ_Y,独立 = √Σ w_c² σ_c² ——
  同样的每通道颗粒下, 通道间相关 (颜色换算的交叉项、色度降噪) 让亮度颗粒多出的倍数。
用法: python3 -I t54_summary.py
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t54_compare as tc  # noqa: E402
from t54_real import COLS, LUMA  # noqa: E402

D = HERE / "t54"


def load(name):
    p = D / name
    return json.load(gzip.open(p, "rt") if name.endswith(".gz") else open(p))


def med(rows, key):
    a = np.array([r[key] for r in rows if np.isfinite(r[key])])
    return [round(float(np.median(a)), 3), round(float(np.percentile(a, 25)), 3), round(float(np.percentile(a, 75)), 3),
            int(len(a))] if len(a) else None


def corr_factor(rows_real, model, setup, key, lo, hi):
    """亮度的通道相关项 (码值 lo–hi 的块)。"""
    R = np.asarray(rows_real)
    s = R[(R[:, COLS["mean"] + 3] >= lo) & (R[:, COLS["mean"] + 3] < hi)]
    if len(s) < 8:
        return None
    sig = np.median(s[:, COLS[key]:COLS[key] + 4], 0)
    fr = sig[3] / np.sqrt((LUMA ** 2 * sig[:3] ** 2).sum())
    code = float(np.median(s[:, COLS["mean"] + 3]))
    m = [tc.interp(tc.model_curve(model, setup, key, c), code, 1) for c in range(4)]
    fm = m[3] / np.sqrt((LUMA ** 2 * np.array(m[:3]) ** 2).sum())
    rho = np.median(s[:, COLS["rho"]:COLS["rho"] + 3], 0)
    rm = [tc.interp(tc.model_curve(model, setup, key, 3), code, k) for k in (3, 4, 5)]
    return dict(code=round(code, 1), f_real=round(float(fr), 3), f_model=round(float(fm), 3), factor=round(float(fr / fm), 3),
                rho48_real=np.round(rho, 2).tolist(), rho48_model=np.round(rm, 2).tolist())


def main():
    out = {}
    sp = load("spectral.json")
    out["spectral"] = {"bandwidth_peak_fwhm_nm": sp["bandwidth_peak_fwhm_nm"],
                       "films": {f: {k: np.round(v, 3).tolist() for k, v in [x for x in rows if x["ev"] == 0][0].items()
                                     if k.startswith("diag_ratio")} | {"J_statusM_diag": np.round(np.diag(np.array(
                                         [x for x in rows if x["ev"] == 0][0]["J"]["status_m_iso"])), 3).tolist()}
                                 for f, rows in sp["films"].items()}}
    model = load("uniform_model.json.gz")
    # ---- EPFL
    er, et, ti = load("epfl_real.json.gz"), load("epfl_tone.json"), load("epfl_tiles.json.gz")
    ep = {}
    for key in ("hp25", "s48"):
        same = tc.compare_epfl(er, et, model, "800T_frontier", key)
        ev = tc.compare_epfl_ev(er, et, ti, model, "800T_frontier", key)
        for ch in ("Y", "R", "G", "B"):
            for frame in er:
                rs = [r for r in same[ch] if r["frame"] == frame and 50 <= r["code"] < 200]
                re = [r for r in ev[ch] if r["frame"] == frame and 50 <= r["code_real"] < 200
                      and np.isfinite(r["film_side"]) and 0 < r["slope_ratio_same_code"]]
                re = [r for r in re if 0.3 < r["slope_ratio_same_code"] < 3]      # 配准差的帧会给出无意义的斜率
                ep.setdefault(key, {}).setdefault(ch, {})[frame] = dict(
                    ratio_same_code=med(rs, "ratio"), n_bins=len(rs),
                    slope=med(re, "slope_ratio_same_code"), position=med(re, "position"), film_side=med(re, "film_side"),
                    bins=[dict(ev=round(r["ev"], 2), code=round(r["code_real"], 1), total=round(r["ratio_same_code"], 2),
                               slope=round(r["slope_ratio_same_code"], 2), position=round(r["position"], 2),
                               film_side=round(r["film_side"], 2)) for r in re])
        ep.setdefault("corr_" + key, {})
        for frame, v in er.items():
            ep["corr_" + key][frame] = [corr_factor(v["rows"], model, "800T_frontier", key, lo, lo + 40)
                                        for lo in (40, 80, 120, 160)]
    out["epfl"] = ep
    # ---- RPL
    rr = load("rpl_real.json.gz")
    setups = {"2_KODAKPORTRA400": "P400_noritsu", "4_KODAKPORTRA160": "P160_noritsu", "3_KODAKPORTRA800": "P800_noritsu",
              "5_KODAKEKTAR100": "E100_noritsu", "1_FUJIPRO400H": "F400H_noritsu"}
    rp = {}
    for film, setup in setups.items():
        if film not in rr or not any(k.startswith(setup + "/") for k in model):
            continue
        same = tc.compare_rpl(rr, model, setup, film, "s48")
        dens = tc.compare_rpl_density(rr, model, setup, film, "s48")
        f = {"u": np.round(dens["u"], 3).tolist(), "pairs_used": len(tc.good_pairs(rr[film]["gamma"])),
             "pairs": len(rr[film]["gamma"])}
        for ch in ("Y", "R", "G", "B"):
            f[ch] = dict(same_code=[dict(code=round(r["code"], 1), frames=r["frames"], tiles=r["tiles"], real=round(r["real"], 2),
                                         model=round(r["model"], 2), ratio=round(r["ratio"], 2)) for r in same[ch]])
            for name, lo, hi in (("dark", 40, 130), ("bright", 130, 215)):
                sel = [r for r in dens[ch] if lo <= r["code"] < hi and 0.3 < r["slope_ratio_same_code"] < 3]
                f[ch][name] = dict(total=med(sel, "ratio_same_code"), slope=med(sel, "slope_ratio_same_code"),
                                   position=med(sel, "position"), film_side=med(sel, "film_side"))
            f[ch]["per_frame"] = {}
            for r in dens[ch]:
                f[ch]["per_frame"].setdefault(r["frame"], []).append(
                    dict(code=round(r["code"], 1), D=round(r["D_photo"], 2), ev_model=round(r["ev_model"], 2),
                         total=round(r["ratio_same_code"], 2), slope=round(r["slope_ratio_same_code"], 2),
                         position=round(r["position"], 2), film_side=round(r["film_side"], 2)))
        f["corr_s48"] = {tag: [corr_factor(v["rows"], model, setup, "s48", lo, lo + 40) for lo in (40, 80, 120, 160)]
                         for tag, v in rr[film]["frames"].items()}
        rp[film] = f
    out["rpl"] = rp
    json.dump(out, open(D / "summary.json", "w"), indent=1, ensure_ascii=False)
    # 打印主表
    for key in ("hp25", "s48"):
        for ch in ("Y", "G"):
            for frame, v in ep[key][ch].items():
                print("EPFL", key, ch, frame, "同码值比", v["ratio_same_code"], "斜率", v["slope"], "位置", v["position"],
                      "胶片侧", v["film_side"])
    for film, f in rp.items():
        for ch in ("Y",):
            print("RPL", film, ch, "u", f["u"], "pairs", f["pairs_used"], "/", f["pairs"])
            print("   同码值:", [(r["code"], r["ratio"]) for r in f[ch]["same_code"]])
            for nm in ("dark", "bright"):
                print("  ", nm, {k: f[ch][nm][k] for k in ("total", "slope", "position", "film_side")})


if __name__ == "__main__":
    main()
