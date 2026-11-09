"""检验报告的文字摘要 (中文)。"""
import numpy as np


def _f(x, n=2):
    return "—" if x is None else f"{x:.{n}f}"


def summarize(rep):
    L = ["silvergrain 检验摘要", "=" * 20]
    if "t46" in rep:
        r = rep["t46"]
        if "loo" in r:
            c = r.get("compare", {})
            L.append(f"T46 官方 RMS 留一 (38 种): 全体拟合 β {r['beta']}, η {r['eta']:.4f}; 留一 |log| 中位 {r['loo_median_abs_log']:.3f} "
                     f"(×{np.exp(r['loo_median_abs_log']):.2f}), RMS {r['loo_rms_log']:.3f}; 与 filmlab 参考逐片比值最大差 "
                     f"{c.get('max_abs_ratio_diff', float('nan')):.2e} ({'逐位相同' if c.get('identical') else '不同'})")
        else:
            L.append("T46: 快速模式 (只在参考的每片 (β, η) 上重算)")
    if "t47" in rep:
        r = rep["t47"]
        L.append("T47 Schmidt Tri-X / D-76 四孔径 (5 / 10 / 20 / 25 µm), 模型 / 实测 − 1: "
                 + ", ".join(f"{v:+.1%}" for v in r["closed_rel"]) + f" (门槛 ±15%: {'通过' if r['passes'] else '未过'});"
                 f" 19 级密度最大 |ΔD| {r['density_max_abs_resid']:.3f}; Kodak 48 µm {r['kodak48_x1000']:.1f} (官方 17, HC-110);"
                 f" 与参考{'逐位相同' if r['identical'] else '不同'}")
    if "t47b" in rep:
        r = rep["t47b"]
        a = {x["key"]: x for x in r["official"]}; b = {x["key"]: x for x in r["schmidt"]}
        L.append("T47b 官方 RMS (预测 / 官方): " + ", ".join(f"{x['film'].split(' (')[0]} {x['t47_ratio']:.2f}{'' if x['judged'] else '*'}"
                                                        for x in r["official"]) + "  (* 只报告: 官方值用 HC-110 / Microfine)")
        L.append("T47b Schmidt 6 片 (预测 / 实测): " + ", ".join(f"{x['film']} {x['t47_ratio']:.2f}" for x in r["schmidt"])
                 + f"; 判定 (T-Max) |log| 中位 / RMS {r['summary']['judged_median_abs_log']:.2f} / {r['summary']['judged_rms_log']:.2f},"
                 f" 留出 5 片 {r['summary']['holdout_median_abs_log']:.2f} / {r['summary']['holdout_rms_log']:.2f};"
                 f" 与参考最大差 {max(r['max_abs_diff_vs_reference'].values()):.1e}")
    if "t50" in rep:
        q = rep["t50"]["quick"]
        L.append("T50 D-76 参照 RMS_D 模型 / 实测: " + ", ".join(f"{k} {v['rms_pred']:.1f}/{v['rms_meas']:.1f}" for k, v in q["d76"].items()))
        for d, v in q["developers"].items():
            L.append(f"T50 {d} (数据库 μ {v['params']['mu']}, σ {v['params']['sigma_um']} µm): RMS 倍数 ±20% 内 {v['pass_rms']}, a_k {v['pass_ak']}; "
                     + ", ".join(f"{f} {x['pred']['rms_ratio']:.2f}/{x['meas']['rms_ratio']:.2f}" for f, x in v["films"].items()))
        if "full" in rep["t50"]:
            for d, v in rep["t50"]["full"]["developers"].items():
                L.append(f"T50 留一 {d}{' (只报告)' if v['report_only'] else ''}: 全部片拟合 {v['fit_all']}; RMS 倍数 {v['pass_rms']}/{v['n']}, a_k {v['pass_ak']}/{v['n']}")
    if "t53" in rep:
        r = rep["t53"]
        t = r["kodak_trix_400@d76"]["ks"]
        L.append("T53 Tri-X / D-76 σ√A(48 µm) ÷ k: " + ", ".join(f"k {k} {t[k]['selwyn_ratio_over_k']['48.0']:.3f}" for k in t)
                 + "; 曲线最大差 " + f"{max(max(v['curve_max_abs_dD'] for v in r[c]['ks'].values()) for c in r if c.endswith(('d76', 'rodinal'))):.4f} D"
                 + f"; 与参考最大差 {r['max_abs_diff_vs_reference']:.1e}")
    if "fastpath" in rep:
        r = rep["fastpath"]
        L.append(f"快速路径 closed_form / explicit 的像素 σ_D: k = 1 {r['sd_ratio_range_k1'][0]:.2f}–{r['sd_ratio_range_k1'][1]:.2f}, "
                 f"含 k = 2 {r['sd_ratio_range'][0]:.2f}–{r['sd_ratio_range'][1]:.2f}; 平均密度最大差 {r['max_abs_mean_diff']:.4f} D")
    if "realscan" in rep:
        r = rep["realscan"]
        for key in ("kodak_trix_400@xtol", "ilford_hp5_plus@d76"):
            if key not in r:
                continue
            c = r[key]["compare"]
            fs = [x["film_side"] for x in c["t51b"] if x["mode"] == "full+noritsu"]
            fl = [x["filmlab_film_side"] for x in c["t51b"] if x["mode"] == "full+noritsu"]
            rc = [x["ratio_code"] for x in c["t51b"] if x["mode"] == "full+noritsu"]
            ph = [x["ratio_px"] for x in c["t52_phone"] if x["mode"] == "full+noritsu"]
            gs = [x["film_side"] for x in c["t51b"] if x["mode"] == "grain+noritsu"]
            L.append(f"真实 RPL {key}: 胶片侧 (真实/silvergrain) {_f(min(fs, default=None))}–{_f(max(fs, default=None))} "
                     f"(filmlab 模型 {_f(min(fl, default=None))}–{_f(max(fl, default=None))}); 同码值 σ_48 {_f(min(rc, default=None))}–{_f(max(rc, default=None))}; "
                     f"手机尺寸 σ_px {_f(min(ph, default=None))}–{_f(max(ph, default=None))}; 胶片侧 (grain 模式) {_f(min(gs, default=None))}–{_f(max(gs, default=None))}"
                     " (lab=None 的旧影调)")
            s = c["spectrum_shape"]
            L.append(f"  谱形 P(f)/P(0.11), f = 0.30 / 0.48 周/像素: 真实 {s['real'][9]:.2f} / {s['real'][-1]:.2f}; silvergrain 无扫描 {s['full'][9]:.2f} / {s['full'][-1]:.2f};"
                     f" 经 Noritsu (T55e 采样链) {s['full+noritsu'][9]:.2f} / {s['full+noritsu'][-1]:.2f}")
        if "portra400" in r:
            c = [x for x in r["portra400"]["compare"] if x["mode"] == "full"]
            L.append("真实 RPL Portra 400 同码值 σ_48 (真实/silvergrain full): " + ", ".join(f"{x['code']:.0f}: {x['ratio']:.2f}" for x in c))
    if "scanner_fit" in rep:
        r = rep["scanner_fit"]
        t = r["train"]
        L.append(f"T55e Noritsu HS-1800 采样链 (原生 11.6 µm → 三角核缩小 → 3 抽头锐化): 训练 (彩色 {len(t['frames'])} 帧) kx {t['kx']:.3f} ky {t['ky']:.3f}, "
                 f"2D 谱形 RMS {t['rms']:.3f} (各向同性 {t['isotropic']['rms']:.3f}, 不锐化 {t['rms_no_sharpen']:.3f}, 加光学 σ 3 µm {t['optics_3um']['rms']:.3f}, "
                 f"T55c 的链 {t['t55c_chain_rms']:.3f})")
        for k in ("hold_colour", "bw_cross_type"):
            h = r[k]
            L.append(f"  留出 {k}: RMS {h['rms_at_train']:.3f} (自由 {h['free']['rms']:.3f}: kx {h['free']['kx']:.3f} ky {h['free']['ky']:.3f}; 不锐化 {h['rms_no_sharpen']:.3f})")
        e = r["edges"]["train"]
        L.append(f"  直边 (训练片 {e['n']} 条, 输送方向): 本链 + kx {e['chain_sep3_kx']['cost'] * 1e4:.3f}e-4, T51 USM {e['t51_usm']['cost'] * 1e4:.3f}e-4, "
                 f"不锐化 {e['none_t51_box']['cost'] * 1e4:.3f}e-4")
    if "filmside" in rep:
        r = rep["filmside"]
        for f, v in r["film_side"].items():
            L.append(f"T55f 胶片侧 σ_48,D 真实 / 模型 ({f}, 配准, 本帧斜率): "
                     + ", ".join(f"D {x['D']:.1f}: {x['ratio_s48']:.2f}" for x in v["rows"])
                     + "; 超额方差 (×1e-6) " + ", ".join(f"{x['excess_var_s48']:.0f}" for x in v["rows"]))
        L.append("T55f 尾巴 (每像素密度偏度 / 峰度, 显式 | closed_form): "
                 + "; ".join(f"{t['gauge']} {t['px_um']:g} µm EV{t['ev']:+d}: {t['explicit']['D_skew']:+.2f}/{t['explicit']['D_kurt']:+.2f} | "
                             f"{t['closed_form']['D_skew']:+.2f}/{t['closed_form']['D_kurt']:+.2f}" for t in r["tails"]))
    if "labtone" in rep:
        r = rep["labtone"]
        for key in ("kodak_trix_400@xtol", "ilford_hp5_plus@d76"):
            for mode in ("grain", "full"):
                c = r[key][mode]["compare"]
                L.append(f"T55b 冲印店影调 {key} {mode}: 同码值 σ_48 真实/silvergrain "
                         + ", ".join(f"{x['code']:.0f}: {x['ratio_s48']:.2f}" for x in c["t51b"])
                         + " (filmlab " + ", ".join(f"{x['filmlab_ratio_s48']:.2f}" for x in c["t51b"]) + "); σ_px 原尺寸 / 手机 "
                         + ", ".join(f"{x['scale'][0]}{x['code']:.0f}: {x['ratio_px']:.2f}" for x in c["t52"]))
        for key, v in r.items():
            if key.endswith("@C-41"):
                L.append(f"T55b {key} grain: 同码值 σ_48 真实/silvergrain " + ", ".join(f"{x['code']:.0f}: {x['ratio']:.2f}" for x in v["grain"]["compare"]))
    return "\n".join(L)
