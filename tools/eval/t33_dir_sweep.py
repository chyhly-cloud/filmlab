"""T33 B: DIR 层间抑制量 (Emulsion.dir_amount) 扫描。只改 S2, 从存档的 S1 logH 重新显影 + 扫描 (t32_rescan; 逐位等价于完整重渲)。

在含 epfl/ (t32_epfl_trace.py 轨迹, GENERATION=$BASE_GEN, 默认 t31_base) 的工作目录运行:
  FILMLAB_DATA=… python3 t33_dir_sweep.py 输出.json 1.0 0.75 0.5 0.25 0
每个取值输出:
  控制帧 (背光色卡 R1-20-21 −1 档、灯箱、灯管帧; 与 T32 同一组) 的单通道传递检验 (t32_hue_ci): 各色相 / 彩度档的 实际/预测, 真实与模型;
  灰阶楔 R1-17-18 的三通道 log 输出 / logH 斜率 (中间 5–11 条, 同 t32_wedge);
  8 夜景的 ΔE00 全图 / 光源旁 (t32_rescan.de_scores, 同 accept_t02) 与红像素 (同 T32 选择) 的 G/R、B/G, 默认定光与 O1 定光。"""
import json, os, sys, warnings, zlib
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
NIGHT = ["R1-00-1", "R1-01-2", "R1-02-3", "R1-03-4", "R1-04-5", "R1-05-6", "R1-06-7", "R1-07-8"]
CONTROL = [("R1-20-21", "_ev-1"), ("R1-30-31", ""), ("R1-31-32", ""), ("R1-25-26", ""), ("R1-27-28", ""), ("R1-23-24", ""),
           ("R1-35-36", ""), ("R1-26-27", ""), ("R1-32-33", ""), ("R1-33-34", ""), ("R1-28-29", ""), ("R1-21-22", "")]
BASE = os.environ.get("BASE_GEN", "t31_base")
STOCK = "kodak_vision3_500t"


def setup(a):
    """代际 BASE + 500T 的 dir_amount = a ("base" = 不覆盖)。"""
    def f():
        from filmlab import generation, materials
        generation.use(BASE)
        if a != "base":
            materials.DIR_AMOUNT[STOCK] = float(a)
            materials.emulsion.cache_clear()
    return f


def wedge_slopes(z, out):
    g = json.load(open(HERE / "epfl_wedge_geometry.json"))
    C = np.array(g["source_corners"]) * z["inp"].shape[1] / g["source_size"][0]
    rows = []
    for i in range(14):
        u = (i + 0.5) / 14
        pts = [(1-u)*(1-v)*C[0] + u*(1-v)*C[1] + u*v*C[2] + (1-u)*v*C[3] for v in (0.35, 0.5, 0.65)]
        vals = {}
        for k, a in (("logH", z["logH"]), ("real", z["real"]), ("out", out)):
            a = np.asarray(a, float)
            vals[k] = np.nanmedian(np.concatenate([a[int(p[1])-6:int(p[1])+7, int(p[0])-6:int(p[0])+7].reshape(-1, 3) for p in pts]), 0)
        rows.append(vals)
    L = np.array([r["logH"] for r in rows]); s = slice(4, 11)
    sl = lambda A: [float(np.polyfit(L[s, c], np.log10(A[s, c]), 1)[0]) for c in range(3)]
    O = np.array([r["out"] for r in rows]); R = np.array([r["real"] for r in rows])
    return dict(real=sl(R), out=sl(O), gr_out=(O[:, 1] / O[:, 0]).tolist(), gr_real=(R[:, 1] / R[:, 0]).tolist())


def red_stats(out, tr, z, sid):
    """t32_red_cands 的红像素与 O1 定光。"""
    from scipy import ndimage as ndi
    from scipy.optimize import least_squares
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    import t32_o1 as o1
    sm = lambda a: np.stack([ndi.uniform_filter(np.nan_to_num(np.asarray(a[..., c], float), nan=-9), 5) for c in range(3)], -1)
    inp, real = sm(z['inp']), sm(z['real'])
    gr = inp[..., 1] / np.maximum(inp[..., 0], 1e-9); br = inp[..., 2] / np.maximum(inp[..., 0], 1e-9)
    far = ndi.distance_transform_edt(~z['core']) * z['px_um'] > 500
    ok = (inp.min(-1) > 0) & far & (real.min(-1) >= 0) & (z['halo_frac'].astype(float).max(-1) < 0.2)
    redish = (gr < 0.3) & (br < 0.5)
    sel = ndi.binary_erosion(redish & ok, iterations=2)
    Dn = tr['Dn']; t0 = tr['timing']
    zz = dict(real=z['real'], Dn=Dn, core=z['core'], px_um=z['px_um'])
    R, D = o1.cells(zz, excl=ndi.binary_dilation(redish, iterations=5))
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None,
                           "frame": zlib.crc32(sid.encode()) % 1000 % 36, "grain": 0}))
    sc = eng.scanner
    mk = lambda dk: PrintTiming(K=np.asarray(t0.K) + dk, s=t0.s, M=t0.M)

    def resid(dk):
        sc.timing = mk(dk)
        return (np.log10(np.clip(sc.to_rgb(D), 1e-4, None)) - np.log10(R)).ravel()
    dk = least_squares(resid, np.zeros(3), bounds=(-1.5, 1.5), loss="soft_l1", f_scale=0.1).x
    sc.timing = mk(dk)
    ys, xs = np.nonzero(sel)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min() - 3, ys.max() + 4, xs.min() - 3, xs.max() + 4
    o1o = np.clip(sc.to_rgb(Dn[y0:y1, x0:x1].reshape(-1, 3)), 0, None).reshape(y1 - y0, x1 - x0, 3)
    return dict(inp=inp[sel], real=real[sel], out=sm(out)[sel], o1=sm(o1o)[sel[y0:y1, x0:x1]], dk=dk)


def job(args):
    a, sid, tag, kind = args
    import t32_rescan as rescan
    out, tr, z = rescan.render(setup(a), sid, "_" + BASE + tag)
    if kind == "control":
        import t32_hue_ci as hue_ci
        f = f"cand/tmp_dir{a}_{sid}.npz"
        d = dict(z); d["out"] = out.astype(np.float32); np.savez(f, **d)
        try:
            rows, nn = hue_ci.run(f, show=False)
        finally:
            os.remove(f)
        return a, sid, kind, [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()} for r in rows]
    if kind == "wedge":
        return a, sid, kind, wedge_slopes(z, out)
    full, near = rescan.de_scores(out, z, sid)
    r = red_stats(out, tr, z, sid)
    return a, sid, kind, dict(full=full, near=near, red=r)


if __name__ == "__main__":
    import multiprocessing as mp
    os.makedirs("cand", exist_ok=True)
    dst, vals = sys.argv[1], sys.argv[2:]
    jobs = [(a, s, t, "control") for a in vals for s, t in CONTROL] + [(a, "R1-17-18", "", "wedge") for a in vals] \
        + [(a, s, "", "night") for a in vals for s in NIGHT]
    res = {}
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as p:
        for a, sid, kind, r in p.imap_unordered(job, jobs, chunksize=1):
            res.setdefault(a, {}).setdefault(kind, {})[sid] = r
            print("done", a, kind, sid, flush=True)
    # 红像素按档汇总 (不存逐像素)
    for a in vals:
        rr = [(s, r["red"]) for s, r in res[a]["night"].items() if r["red"] is not None]
        T = {k: np.concatenate([r[k] for s, r in rr]) for k in ("inp", "real", "out", "o1")}
        ev = np.log2(T["inp"][:, 0] / 0.184)
        bins = {}
        for lo in range(-2, 5):
            m = (ev >= lo) & (ev < lo + 1)
            if m.sum() < 30:
                continue
            q = lambda A, i, j: float(np.median(A[m][:, i] / np.maximum(A[m][:, j], 1e-9)))
            bins[f"{lo:+d}"] = dict(n=int(m.sum()), gr={k: q(T[k], 1, 0) for k in ("inp", "real", "out", "o1")},
                                    bg={k: q(T[k], 2, 1) for k in ("inp", "real", "out", "o1")})
        for s, r in res[a]["night"].items():
            if r["red"] is not None:
                r["red"] = dict(n=len(r["red"]["inp"]), dk=r["red"]["dk"].tolist())
        res[a]["red_bins"] = bins
    json.dump(res, open(dst, "w"), indent=1, ensure_ascii=False, default=float)
