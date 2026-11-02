"""T42 逐级反事实 (只诊断): 从 t42_sat.py 存下的像素 / 格子 logH (S1 不变) 重算模型链, 每个变体给出
  · 逐级等效曝光彩度比 (中性 = 本帧白 v0, 同 t42_sat 的 "_m");
  · 输出的单通道传递彩度比 (模型自己的中性像素建传递, 同 t42_sat 的 out_emp), 与真实 (存档) 之比;
  · Lab C* (O1: 每帧三通道光号重新拟合到真实, s = 1), 按胶片曝光色相;
  · 输出中性传递的斜率 (log 输出 / logH, 本帧中性像素, 每通道), 真实与模型。
变体见 VARIANTS。用法: python3 t42_cf.py 结果目录(t42_sat) 输出.json 变体… (场景从结果目录取)"""
import json, os, sys, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]
HUES = [("红", 0), ("黄", 60), ("绿", 120), ("青", 180), ("蓝", 240), ("品红", 300)]


def _diag_only(f):
    return lambda a, neg=True: np.diag(np.diag(f(a, neg)))


def variant(name, sid):
    """→ (Chain 的构造参数, 代际)。"""
    import t42_chain as tc
    from filmlab import chemistry, stocks
    gen, kw = "t31_base", {}
    if name == "dir0":                       # S2: 只留同层 (无层间抑制)
        kw["dir_fn"] = _diag_only(chemistry._dir_matrix)
    elif name.startswith("dirx"):            # S2: 层间项整体 × a (同层不变)
        a = float(name[4:])
        f0 = chemistry._dir_matrix

        def f(amount, neg=True, a=a):
            A = f0(amount, neg); D = np.diag(np.diag(A))
            return D + a * (A - D)
        kw["dir_fn"] = f
    elif name == "kdye":                     # S3: 只换染料光谱为 Kodak 数据手册带符号谱 (曲线不变)
        k = np.load(stocks.PROFILE_DIR / "kodak_vision3_500t_kodak_raw.npz")
        kw["dye_spectra"] = np.asarray(k["E"], float)
    elif name == "k5219":                    # S2+S3: Kodak 染料 + 按 Kodak Status M 重新解混的曲线 (t32_k5219)
        gen = "t32_k5219"
    elif name == "paper_indep":              # S4d 诊断: 数字相纸按通道独立 (每通道 = 中性灰在该相纸密度下的输出; 同 T32 S4d)
        from filmlab.scanning import paper

        def rp(self, Dp, output_black=paper.OUTPUT_BLACK, chunk=1 << 16):
            return (output_black + (1 - output_black) * (10.0 ** -Dp - self.k) / (1 - self.k)).reshape(-1, 3)
        paper.DigitalPaper.reference_print = rp
    elif name.startswith("gen:"):
        gen = name[4:]
    return gen, kw


def emp(lh, y, nv, tol=0.05):
    import t42_sat as ts
    neu = np.ptp(lh - nv, -1) < tol
    e = np.full(lh.shape, np.nan); slope = np.full(3, np.nan)
    if neu.sum() < 200: return e, slope
    for c in range(3):
        tr = ts.emp_transfer(lh[neu][:, c], y[neu][:, c])
        if tr is None: continue
        e[:, c] = ts.emp_equiv(tr, y[:, c]) - nv[c]
        x = lh[neu][:, c]; ly = np.log10(np.clip(y[neu][:, c], 1e-9, None))
        lo, hi = np.percentile(x, [10, 90]); m = (x >= lo) & (x <= hi) & (y[neu][:, c] > 0.01) & (y[neu][:, c] < 0.9)
        if m.sum() > 50: slope[c] = np.polyfit(x[m], ly[m], 1)[0]
    return e, slope


def job(args):
    f, name = args
    import colour
    import t42_chain as tc, t42_sat as ts
    from scipy.optimize import least_squares
    from filmlab.scanning.timing import PrintTiming
    sid = os.path.basename(f).split("_t31_base")[0]
    z = np.load(f)
    gen, kw = variant(name, sid)
    eng = tc.engine(gen, sid)
    timing = (z["K"], z["s"], z["M_off"], z["M_r"])
    v0 = z["v0"]
    ch = tc.Chain(eng, timing, neutral=v0, **kw)
    logH = z["logH"].astype(float); lh = logH - tc.G0
    st = ch.stages(logH)
    eq = {k: ch.equiv(st[k], k) for k in ("dye", "sm", "scan", "dn", "dp", "out")}
    eq["out_emp"], slope_m = emp(lh, st["out"], v0)
    eq["real"], slope_r = emp(lh, z["real"].astype(float), v0)
    c0 = np.ptp(lh - v0, -1); h = tc.hue(lh - v0)
    res = {"stage": {}, "slope_model": slope_m.tolist(), "slope_real": slope_r.tolist()}
    for hn, h0 in HUES + [("全部", None)]:
        mh = np.ones(len(h), bool) if h0 is None else ((h - h0 + 180) % 360 - 180) ** 2 < 30 ** 2
        for lo, hi in ((0.25, 0.5), (0.5, 3.0), (0.25, 3.0)):
            m0 = mh & (c0 >= lo) & (c0 < hi)
            d = {}
            for k, e in eq.items():
                r = np.ptp(e, -1) / c0; m = m0 & np.isfinite(r)
                if m.sum() >= 100: d[k] = float(np.median(r[m]))
            if d: res["stage"][f"{hn}|{lo}|{hi}"] = d
            if (lo, hi) == (0.25, 3.0):
                for k in ("out_emp", "real", "dye", "sm", "dn", "out"):
                    e = eq[k] - (lh - v0); m = m0 & np.all(np.isfinite(e), -1)
                    if m.sum() >= 100: res.setdefault("dev", {}).setdefault(hn, {})[k] = np.median(e[m], 0).tolist()
    # Lab (O1) on cells
    lab = lambda rgb: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb, 0, 1), apply_cctf_decoding=False))
    LHc = z["lab_logH"].astype(float); R = z["lab_real"].astype(float)
    Dc = ch.stages(LHc, timing=None)["dn"]
    sc = eng.scanner; mo, mr = np.asarray(z["M_off"], float), np.asarray(z["M_r"], float)

    def render(K):
        sc.timing = PrintTiming(K=np.asarray(K, float), s=np.ones(3), M=(mo, mr))
        return lab(sc.to_rgb(Dc))
    rs = lambda S: np.nan_to_num((S - R).ravel(), nan=50.0)
    fit = least_squares(lambda k: rs(render(k)), z["K"].astype(float), loss="soft_l1", f_scale=5.0)
    M = render(fit.x)
    lc = LHc - tc.G0 - v0; cl, hl = np.ptp(lc, -1), tc.hue(lc)
    cr, cm = np.hypot(R[:, 1], R[:, 2]), np.hypot(M[:, 1], M[:, 2])
    res["lab"] = {}
    for hn, h0 in HUES + [("全部", None)]:
        mh = np.ones(len(hl), bool) if h0 is None else ((hl - h0 + 180) % 360 - 180) ** 2 < 30 ** 2
        m = mh & (cl >= 0.1) & (R[:, 0] > 10) & (M[:, 0] > 10)
        if m.sum() >= 8:
            res["lab"][hn] = [float(np.median(cm[m]) / max(np.median(cr[m]), 1e-6)), float(np.median(cr[m])), float(np.median(cm[m]))]
    res["lab_dE"] = float(np.median(colour.delta_E(M, R, method="CIE 2000")))
    return sid, name, res


if __name__ == "__main__":
    import multiprocessing as mp
    d, dst, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    out = json.load(open(dst)) if os.path.exists(dst) else {}
    fs = sorted(glob.glob(f"{d}/*_t31_base.npz"))
    jobs = [(f, n) for n in names for f in fs if os.path.basename(f).split("_t31_base")[0] not in out.get(n, {})]
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as p:
        for sid, name, r in p.imap_unordered(job, jobs):
            out.setdefault(name, {})[sid] = r
            json.dump(out, open(dst, "w"))
            print(name, sid, flush=True)
