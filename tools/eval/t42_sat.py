"""T42 第 1–2 步: EPFL 配对帧的彩度比 (模型 / 真实) 与模型链逐级彩度。

每帧 (t32_epfl_trace.py 的存档, 数码几何):
  A. Lab 彩度 (O1): 24 px 格子中位; 每帧三通道光号拟合到真实 (s = 1, Lab soft_l1, 同 T40 A3), 然后只比较 C*ab。
  B. 单通道传递 (T32 第 6 节): 胶片中性像素 (logH − G0 的通道极差 < TOL) → 每通道 logH_c → 输出_c 的单调传递,
     真实、模型各一条; 输出经各自的传递反查成等效曝光 e, 彩度比 = ptp(e) / ptp(logH − G0)。
     (定光是逐通道单调的, 已被传递吸收, 所以与定光无关; 亮度不进入。)
  C. 模型逐级 (t42_chain): 同一批像素从存档 logH 重算 dye / Status M / 扫描密度 / Dn / 相纸密度 / 输出,
     每级按模型自己的中性灰阶反查成等效曝光, 彩度比同上。
像素: 5×5 平滑后, 距传感器饱和灯芯 > 0.5 mm、光晕份额 < 0.2、真实 / 模型输出 0.002–0.97, 腐蚀 3 像素 (同 T32)。
用法: GEN=t31_base python3 t42_sat.py 存档目录 输出目录 场景…"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]
TOL = 0.05
STRIDE = 2
C = 24


def load(f):
    from scipy import ndimage as ndi
    z = np.load(f)
    sm = lambda a: np.stack([ndi.uniform_filter(np.nan_to_num(a[..., c].astype(float), nan=-9), 5) for c in range(a.shape[-1])], -1)
    d = {k: sm(z[k]) for k in ("inp", "real", "out", "logH")}
    far = ndi.distance_transform_edt(~z["core"]) * z["px_um"] > 500
    d["ok"] = ndi.binary_erosion((d["inp"].min(-1) > 0) & far & (d["real"].min(-1) > 0.002) & (d["out"].min(-1) > 0.002)
                                 & (d["real"].max(-1) < 0.97) & (d["out"].max(-1) < 0.97)
                                 & (z["halo_frac"].astype(float).max(-1) < 0.2), iterations=3)
    return z, d


def emp_transfer(x, y):
    """中性像素的 (logH_c, 输出_c) → 单调传递 (binned 中位), 返回 (xs, log ys)。"""
    edges = np.arange(-3.0, 3.0, 0.05); xc, yc = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= 20: xc.append(np.median(x[m])); yc.append(np.median(np.log(y[m])))
    if len(xc) < 4: return None
    return np.array(xc), np.maximum.accumulate(np.array(yc))


def emp_equiv(tr, y):
    xs, ys = tr
    ly = np.log(np.clip(y, 1e-9, None))
    # 只在严格上升段反查
    ok = np.r_[True, np.diff(ys) > 1e-4]
    xs, ys = xs[ok], ys[ok]
    v = np.interp(ly, ys, xs)
    v[(ly < ys[0]) | (ly > ys[-1])] = np.nan
    return v


def cells(a, ok):
    h, w = a.shape[:2]; out = []
    for y in range(0, h - C + 1, C):
        for x in range(0, w - C + 1, C):
            if ok[y:y + C, x:x + C].mean() >= 0.95: out.append(np.nanmedian(a[y:y + C, x:x + C].reshape(-1, a.shape[-1]), 0))
    return np.array(out)


def lab_o1(z, eng):
    """A: 24 px 格子, O1 (s=1) 光号拟合到真实, 返回 (真实 Lab, 模型 Lab, 光号差, logH 格子)。"""
    import colour
    from scipy.optimize import least_squares
    from filmlab.scanning.timing import PrintTiming
    sc = eng.scanner
    lab = lambda rgb: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb, 0, 1), apply_cctf_decoding=False))
    ok = np.all(np.isfinite(z["real"]), -1) & np.all(np.isfinite(z["Dn"]), -1) & ~z["core"] \
        & (z["halo_frac"].astype(float).max(-1) < 0.2)
    A = cells(np.concatenate([z["Dn"], z["real"], z["logH"]], -1).astype(float), ok)
    Dc, R, LH = A[:, :3], lab(A[:, 3:6]), A[:, 6:9]
    keep = R[:, 0] > 5
    mo, mr = np.asarray(z["M_off"], float), np.asarray(z["M_r"], float)

    def render(K):
        sc.timing = PrintTiming(K=np.asarray(K, float), s=np.ones(3), M=(mo, mr))
        return lab(sc.to_rgb(Dc))
    rs = lambda S: np.nan_to_num((S - R)[keep].ravel(), nan=50.0)
    f = least_squares(lambda k: rs(render(k)), z["K"].astype(float), loss="soft_l1", f_scale=5.0)
    return R[keep], render(f.x)[keep], (f.x - z["K"]).tolist(), LH[keep]


def job(args):
    src, dst, sid, gen = args
    import t42_chain as tc
    z, d = load(f"{src}/{sid}_{gen}.npz")
    eng = tc.engine(gen, sid)
    timing = (z["K"], z["s"], z["M_off"], z["M_r"])
    ok = d["ok"].copy()
    sel = np.zeros_like(ok); sel[::STRIDE, ::STRIDE] = True
    ok &= sel
    logH = d["logH"][ok]; lh = logH - tc.G0
    real, out = d["real"][ok], d["out"][ok]
    # 本帧的"白" v0 (T32 mode): 亮像素胶片曝光色度的众数; 中性与彩度都相对它
    v = lh - lh.mean(-1, keepdims=True)
    bright = lh.mean(-1) > -1.0
    a, b = v[bright][:, 0] - v[bright][:, 1], v[bright][:, 2] - v[bright][:, 1]
    hst, xe, ye = np.histogram2d(a, b, bins=60, range=[[-1.5, 1.5], [-1.5, 1.5]])
    i, j = np.unravel_index(np.argmax(hst), hst.shape)
    k = (np.abs(a - (xe[i] + xe[i + 1]) / 2) < 0.05) & (np.abs(b - (ye[j] + ye[j + 1]) / 2) < 0.05)
    v0 = np.median(v[bright][k], 0)
    res = {"n": int(ok.sum()), "v0": np.round(v0, 3).tolist()}
    eq = {}
    st = None
    for tag, nv in (("", np.zeros(3)), ("_m", v0)):
        # B: 单通道传递 (真实 / 模型各自), 中性 = logH − G0 − nv 的通道极差 < TOL
        neu = np.ptp(lh - nv, -1) < TOL
        res["n_neutral" + tag] = int(neu.sum())
        for w, y in (("real", real), ("out_emp", out)):
            e = np.full(lh.shape, np.nan)
            if neu.sum() >= 200:
                for c in range(3):
                    tr = emp_transfer(lh[neu][:, c], y[neu][:, c])
                    if tr is not None:
                        e[:, c] = emp_equiv(tr, y[:, c]) - nv[c]
            eq[w + tag] = e
        # C: 模型逐级 (从平滑后的 logH 重算), 中性灰阶沿 G0 + nv
        ch = tc.Chain(eng, timing, neutral=nv)
        if st is None:
            st = ch.stages(logH)
        for kk in ("dye", "sm", "scan", "dn", "dp", "out"):
            eq[kk + tag] = ch.equiv(st[kk], kk)
    # A: Lab (O1)
    R, M, dK, LHc = lab_o1(z, eng)
    f16 = lambda a: np.asarray(a, np.float32)
    np.savez_compressed(f"{dst}/{sid}_{gen}.npz", logH=f16(logH), v0=v0, real=f16(real), out=f16(out),
                        **{f"eq_{k}": f16(v) for k, v in eq.items()},
                        lab_real=f16(R), lab_model=f16(M), lab_logH=f16(LHc), dK=np.asarray(dK),
                        K=z["K"], s=z["s"], M_off=z["M_off"], M_r=z["M_r"])
    return sid, res


if __name__ == "__main__":
    import multiprocessing as mp
    src, dst = sys.argv[1], sys.argv[2]
    gen = os.environ.get("GEN", "t31_base")
    os.makedirs(dst, exist_ok=True)
    jobs = [(src, dst, s, gen) for s in sys.argv[3:] if not os.path.exists(f"{dst}/{s}_{gen}.npz")]
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3")), maxtasksperchild=1) as p:
        for sid, r in p.imap_unordered(job, jobs):
            print(sid, json.dumps(r), flush=True)
