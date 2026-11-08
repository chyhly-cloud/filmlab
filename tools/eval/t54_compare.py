"""T54: 真实 / 模型在同一输出码值上的颗粒比与分解 (docs/tasks/T54_COLOUR_GRAIN_LEVEL.md 第 3、4 节)。

每通道 (R、G、B) 与亮度 Y, 每个码值档 (真实各帧的中位):
  总比 = σ_真实 / σ_模型 (同码值)
       = 影调斜率比 (dCode/dEV 真实 / 模型)  ×  噪声等效曝光比 NEE = σ / (dCode/dEV) (真实 / 模型)
  RPL: 真实 dCode_c/dEV = dCode_c/dD_c (扫描 ↔ 翻拍配准) × dD_c/dEV (相邻包围档翻拍), 翻拍密度单位抵消。
  模型: dCode/dEV 由颗粒关的 EV ± 0.1 有限差分 (t54_model chain_point)。亮度的斜率 = Σ BT.601 × 通道斜率。
用法: python3 -I t54_compare.py rpl 真实.json 模型.json 模型SETUP 胶片目录名 [输出.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t54_real import COLS, LUMA  # noqa: E402

EDGES = np.arange(20, 241, 20)
LOG2 = np.log10(2.0)
NAMES = ("R", "G", "B", "Y")


def model_curve(model, setup, key, ch):
    pts = []
    for k, v in model.items():
        if not k.startswith(setup + "/"):
            continue
        rows = np.array(v["rows"])
        code = float(np.median(rows[:, COLS["mean"] + ch]))
        s = float(np.median(rows[:, COLS[key] + ch]))
        sl = np.array(v["point"]["slope_E"]) * LOG2
        slope = float(LUMA @ sl) if ch == 3 else float(sl[ch])
        rho = np.median(rows[:, COLS["rho"]:COLS["rho"] + 3], 0)
        pts.append((code, s, slope, *rho))
    return np.array(sorted(pts))


def interp(curve, code, col):
    return float(np.interp(code, curve[:, 0], curve[:, col]))


def good_pairs(gam, min_corr=0.45, max_shift=30):
    """相邻包围档翻拍的配准: 相关 ≥ min_corr 且平移 ≤ max_shift 像素的才用 (失败的配准给出无意义的 γ)。"""
    return [g for g in gam if g[2][0] >= min_corr and abs(g[2][2]) <= max_shift and abs(g[2][3]) <= max_shift]


def real_slope_ev(frame, gam, ch, code, wD=0.08):
    """该帧通道 ch 在码值 code 处的 dCode/dEV = dCode/dD × dD/dEV, 两者都用局部回归 (不用全程多项式):
    dCode/dD: 配准的 (D, 码值) 分档点中 D 在 D(code) ± wD 内的加权线性回归; dD/dEV: 各对包围档在 D ± wD/2 内的中位。
    亮度 (ch 3) 由三通道组合 (近中性块, 用同一码值近似)。"""
    if ch == 3:
        return float(sum(LUMA[c] * real_slope_ev(frame, gam, c, code, wD) for c in range(3)))
    t = frame["tone"][ch]
    D = np.array(t["D"]); C = np.array(t["code"]); n = np.array(t["n"], float)
    o = np.argsort(C)
    if code < C.min() or code > C.max():
        return np.nan
    d = float(np.interp(code, C[o], D[o]))
    m = np.abs(D - d) <= wD
    if m.sum() < 3:
        return np.nan
    A = np.stack([D[m], np.ones(m.sum())], 1) * np.sqrt(n[m])[:, None]
    sD = float(np.linalg.lstsq(A, C[m] * np.sqrt(n[m]), rcond=None)[0][0])
    g = np.array([r for pair in good_pairs(gam) for r in pair[3][ch]])       # (D, dD/dEV, n)
    sel = np.abs(g[:, 0] - d) <= wD / 2
    if sel.sum() < 3:
        sel = np.argsort(np.abs(g[:, 0] - d))[:5]
    gam_d = float(np.median(g[sel, 1]))
    return sD * gam_d


def local_slope(ch, code, win=20.0):
    """EPFL: 该帧通道的 (logE 档, 码值) 上, 码值在 code ± win 内的点做线性回归 → dCode/dlogE。"""
    L = np.array(ch["logE"]); C = np.array(ch["code"]); n = np.array(ch["n"], float)
    m = np.abs(C - code) <= win
    if m.sum() < 3 or C[m].min() > code or C[m].max() < code:
        return np.nan
    A = np.stack([L[m], np.ones(m.sum())], 1) * np.sqrt(n[m])[:, None]
    return float(np.linalg.lstsq(A, C[m] * np.sqrt(n[m]), rcond=None)[0][0])


def compare_epfl(real, tone, model, setup, key="hp25"):
    """EPFL: 同码值的真实 / 模型颗粒, 真实 dCode/dlogE 由同帧 RAW 配准 (t54_epfl_tone)。每帧单独一行。"""
    out = {}
    for ch in range(4):
        mc = model_curve(model, setup, key, ch)
        rows_out = []
        for frame, v in real.items():
            R = np.array(v["rows"])
            tn = tone.get(frame, {}).get("channels")
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                s = R[(R[:, COLS["mean"] + ch] >= lo) & (R[:, COLS["mean"] + ch] < hi)]
                if len(s) < 5:
                    continue
                code = float(np.median(s[:, COLS["mean"] + ch])); sr = float(np.median(s[:, COLS[key] + ch]))
                if tn:
                    cc = np.median(s[:, COLS["mean"]:COLS["mean"] + 3], 0)
                    sl3 = [local_slope(tn[c], cc[c]) for c in range(3)]
                    slr = float(LUMA @ np.array(sl3)) if ch == 3 else sl3[ch]
                else:
                    slr = np.nan
                sm = interp(mc, code, 1); slm = interp(mc, code, 2) / LOG2
                rows_out.append(dict(frame=frame, code=code, tiles=int(len(s)), real=sr, model=sm, ratio=sr / sm,
                                     slope_real=slr, slope_model=slm, slope_ratio=slr / slm, nee_ratio=(sr / slr) / (sm / slm),
                                     rho_real=np.median(s[:, COLS["rho"]:COLS["rho"] + 3], 0).tolist(),
                                     rho_model=[interp(mc, code, 3), interp(mc, code, 4), interp(mc, code, 5)]))
        out[NAMES[ch]] = rows_out
    return out


def compare_rpl(real, model, setup, film, key="s48"):
    fr = real[film]["frames"]; gam = real[film]["gamma"]
    out = {}
    for ch in range(4):
        mc = model_curve(model, setup, key, ch)
        rows_out = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            per = []
            for tag, f in fr.items():
                R = np.array(f["rows"])
                lev = R[:, COLS["mean"] + ch]
                s = R[(lev >= lo) & (lev < hi)]
                if len(s) < 5:
                    continue
                code = float(np.median(s[:, COLS["mean"] + ch]))
                sl = real_slope_ev(f, gam, ch, code) if "tone" in f else np.nan
                rho = np.median(s[:, COLS["rho"]:COLS["rho"] + 3], 0)
                per.append((code, float(np.median(s[:, COLS[key] + ch])), sl, len(s), *rho))
            if len(per) < 2:
                continue
            per = np.array(per)
            code = float(np.median(per[:, 0])); sr = float(np.median(per[:, 1]))
            slr = float(np.nanmedian(per[:, 2])) if np.isfinite(per[:, 2]).any() else np.nan
            sm = interp(mc, code, 1); slm = interp(mc, code, 2)
            rows_out.append(dict(code=code, frames=int(len(per)), tiles=int(per[:, 3].sum()), real=sr, model=sm,
                                 ratio=sr / sm, slope_real=slr, slope_model=slm, slope_ratio=slr / slm,
                                 nee_ratio=(sr / slr) / (sm / slm),
                                 slope_q=np.nanpercentile(per[:, 2], [25, 75]).tolist(),
                                 rho_real=np.median(per[:, 4:7], 0).tolist(),
                                 rho_model=[interp(mc, code, 3), interp(mc, code, 4), interp(mc, code, 5)]))
        out[NAMES[ch]] = rows_out
    return out


def show(res, title):
    print(title)
    for ch, rows in res.items():
        for r in rows:
            print(f" {ch} 码值 {r['code']:5.1f} 帧 {r['frames']} 块 {r['tiles']:5d} 真实 {r['real']:5.2f} 模型 {r['model']:5.2f} "
                  f"比 {r['ratio']:4.2f} | 斜率 真实 {r['slope_real']:6.1f} ({r['slope_q'][0]:.0f}–{r['slope_q'][1]:.0f}) "
                  f"模型 {r['slope_model']:6.1f} 比 {r['slope_ratio']:4.2f} | NEE 比 {r['nee_ratio']:4.2f} "
                  f"| ρ48 真实 {np.round(r['rho_real'], 2)} 模型 {np.round(r['rho_model'], 2)}")


if __name__ == "__main__":
    if sys.argv[1] == "epfl":
        real = json.load(open(sys.argv[2])); tone = json.load(open(sys.argv[3])); model = json.load(open(sys.argv[4]))
        res = {}
        for key in ("hp25", "s48", "px"):
            res[key] = compare_epfl(real, tone, model, sys.argv[5], key)
            print("==", key)
            for ch, rows in res[key].items():
                for r in rows:
                    print(f" {ch} {r['frame']} 码值 {r['code']:5.1f} 块 {r['tiles']:5d} 真实 {r['real']:5.2f} 模型 {r['model']:5.2f} "
                          f"比 {r['ratio']:4.2f} | 斜率/logE 真实 {r['slope_real']:6.1f} 模型 {r['slope_model']:6.1f} "
                          f"比 {r['slope_ratio']:4.2f} | NEE 比 {r['nee_ratio']:4.2f} | ρ48 {np.round(r['rho_real'], 2)} 模型 {np.round(r['rho_model'], 2)}")
        if len(sys.argv) > 6:
            json.dump(res, open(sys.argv[6], "w"), indent=1)
    elif sys.argv[1] == "rpl":
        real = json.load(open(sys.argv[2])); model = json.load(open(sys.argv[3]))
        res = {}
        for key in ("s48", "px", "hp25"):
            res[key] = compare_rpl(real, model, sys.argv[4], sys.argv[5], key)
            show(res[key], f"== {sys.argv[5]} {key}")
        if len(sys.argv) > 6:
            json.dump(res, open(sys.argv[6], "w"), indent=1)


def model_vs_ev(model, setup, key, ch):
    """模型均匀灰: 每个 EV 的 (EV, 码值, σ, dCode/dlogE)。"""
    pts = []
    for k, v in model.items():
        if not k.startswith(setup + "/"):
            continue
        rows = np.array(v["rows"])
        sl = np.array(v["point"]["slope_E"])
        pts.append((float(k.split("/")[1]), float(np.median(rows[:, COLS["mean"] + ch])), float(np.median(rows[:, COLS[key] + ch])),
                    float(LUMA @ sl) if ch == 3 else float(sl[ch])))
    return np.array(sorted(pts))


def compare_epfl_ev(real, tone, tiles, model, setup, key="hp25", width=0.5):
    """EPFL: 按块的胶片曝光档 (RAW 会话测光) 分档 → 同曝光的真实 / 模型: 码值、σ、斜率、NEE。"""
    out = {}
    for ch in range(4):
        mv = model_vs_ev(model, setup, key, ch)
        rows_out = []
        for frame, v in real.items():
            if frame not in tiles or "ev" not in tiles[frame] or "channels" not in tone.get(frame, {}):
                continue
            R = np.array(v["rows"]); ev = np.array(tiles[frame]["ev"], float)
            tn = tone[frame]["channels"]
            for e0 in np.arange(-3.0, 3.01, width):
                s = R[np.abs(ev - e0) <= width / 2]
                if len(s) < 8:
                    continue
                code = float(np.median(s[:, COLS["mean"] + ch])); sr = float(np.median(s[:, COLS[key] + ch]))
                cc = np.median(s[:, COLS["mean"]:COLS["mean"] + 3], 0)
                sl3 = [local_slope(tn[c], cc[c]) for c in range(3)]
                slr = float(LUMA @ np.array(sl3)) if ch == 3 else sl3[ch]
                em = float(np.median(ev[np.abs(ev - e0) <= width / 2]))
                cm = float(np.interp(em, mv[:, 0], mv[:, 1])); sm = float(np.interp(em, mv[:, 0], mv[:, 2]))
                slm = float(np.interp(em, mv[:, 0], mv[:, 3]))
                # 同码值的模型 (固定定光)
                o = np.argsort(mv[:, 1])
                sm_c = float(np.interp(code, mv[o, 1], mv[o, 2])); slm_c = float(np.interp(code, mv[o, 1], mv[o, 3]))
                rows_out.append(dict(frame=frame, ev=em, tiles=int(len(s)), code_real=code, code_model=cm, real=sr,
                                     model_same_ev=sm, model_same_code=sm_c, slope_real=slr, slope_model_same_ev=slm,
                                     slope_model_same_code=slm_c, nee_real=sr / slr, nee_model=sm / slm,
                                     ratio_same_code=sr / sm_c, slope_ratio_same_code=slr / slm_c,
                                     position=(sm / slm) / (sm_c / slm_c), film_side=(sr / slr) / (sm / slm)))
        out[NAMES[ch]] = rows_out
    return out


def show_ev(res):
    for ch, rows in res.items():
        for r in rows:
            print(f" {ch} {r['frame']} EV {r['ev']:+5.2f} 块 {r['tiles']:4d} 码值 真实 {r['code_real']:5.1f} 模型 {r['code_model']:5.1f} | "
                  f"σ 真实 {r['real']:5.2f} 模型同EV {r['model_same_ev']:5.2f} 同码值 {r['model_same_code']:5.2f} | "
                  f"总比(同码值) {r['ratio_same_code']:4.2f} = 斜率 {r['slope_ratio_same_code']:4.2f} × 位置 {r['position']:4.2f} "
                  f"× 胶片侧 {r['film_side']:4.2f}")


def compare_rpl_density(real, model, setup, film, key="s48", ch_d=1):
    """RPL: 同底片密度 (翻拍) 的比较。块的翻拍密度 D_photo,c (该帧影调曲线的反函数) 按 u_c = γ_翻拍 / γ_模型
    (每 EV 的密度增量, 直线段中位) 换到模型染料量单位 → 模型中同染料量的 EV → 同曝光的模型 σ、斜率、NEE。
    亮度 (ch 3) 的位置按 G 通道定。返回每帧每码值档: 总比 (同码值) = 斜率 × 位置 × 胶片侧。"""
    fr = real[film]["frames"]; gam = real[film]["gamma"]
    pts = sorted((float(k.split("/")[1]), v["point"]) for k, v in model.items() if k.startswith(setup + "/"))
    evs = np.array([p[0] for p in pts]); dyes = np.array([p[1]["dye"] for p in pts])
    gm_model = np.array([p[1]["dC_dlogE"] for p in pts]) * LOG2          # 每 EV
    g = [np.array([r for pair in good_pairs(gam) for r in pair[3][c]]) for c in range(3)]
    u = []
    for c in range(3):
        sel = (g[c][:, 0] > 0.3) & (g[c][:, 0] < 1.0)
        u.append(float(np.median(g[c][sel, 1]) / np.median(gm_model[(evs >= -1) & (evs <= 2), c])))
    out = {"u": u}
    for ch in range(4):
        mv = model_vs_ev(model, setup, key, ch)
        cd = ch_d if ch == 3 else ch
        rows_out = []
        for tag, f in sorted(fr.items(), key=lambda kv: kv[1]["ev"]):
            if "tone" not in f:
                continue
            R = np.array(f["rows"])
            t = f["tone"][cd]; p = np.array(t["poly"]); Dg = np.linspace(min(t["D"]), max(t["D"]), 400)
            Cg = np.polyval(p, Dg); o = np.argsort(Cg)
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                s = R[(R[:, COLS["mean"] + ch] >= lo) & (R[:, COLS["mean"] + ch] < hi)]
                if len(s) < 8:
                    continue
                code = float(np.median(s[:, COLS["mean"] + ch])); sr = float(np.median(s[:, COLS[key] + ch]))
                slr = real_slope_ev(f, gam, ch, code) / LOG2
                cdc = float(np.median(s[:, COLS["mean"] + cd]))
                if cdc < Cg.min() or cdc > Cg.max() or not np.isfinite(slr):
                    continue
                dph = float(np.interp(cdc, Cg[o], Dg[o]))
                cmod = dph / u[cd]
                # 模型的净染料量 (未曝光处为 0, 与翻拍的"减片基"同一零点) → EV
                em = float(np.interp(cmod, dyes[:, cd], evs))
                sm = float(np.interp(em, mv[:, 0], mv[:, 2])); slm = float(np.interp(em, mv[:, 0], mv[:, 3]))
                oo = np.argsort(mv[:, 1])
                sm_c = float(np.interp(code, mv[oo, 1], mv[oo, 2])); slm_c = float(np.interp(code, mv[oo, 1], mv[oo, 3]))
                rows_out.append(dict(frame=tag, frame_ev=f["ev"], code=code, D_photo=dph, ev_model=em, tiles=int(len(s)),
                                     ratio_same_code=sr / sm_c, slope_ratio_same_code=slr / slm_c,
                                     position=(sm / slm) / (sm_c / slm_c), film_side=(sr / slr) / (sm / slm)))
        out[NAMES[ch]] = rows_out
    return out
