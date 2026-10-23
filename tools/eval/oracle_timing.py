"""定光预言机: 把冲印店的逐帧校色 (操作员变量) 从 RAW 配对误差里剥离 (docs/ARCHITECTURE.md §4)。

每个 RAW 配对: 渲染到 S4b 的 Dn (负片密度域), 用固定的单应性 (数码输入 → 真实扫描, 与被测条件无关)
把 Dn 对齐到真实扫描, 取 24px 格子中位数; 然后只用冲印店真实拥有的自由度拟合定光, 使模拟最接近真实扫描:
  O1  印放光号: 三通道曝光偏移 (3 个自由度)
  O2  O1 + 红 / 蓝通道斜率 (±SLOPE_LIMIT) + 整体反差 (±20%)   (= Frontier 灰平衡 LUT 的能力)
剩余误差与"逐帧平衡力度"无关, 归于胶片物理 + 扫描硬件 + 数字相纸 (+ 打开时的输出调整)。
拟合定光与自动定光之差 = 自动校色模型的误差。

用法: python3 oracle_timing.py 输出.json   (物理条件用 ablation_flags 的环境变量; COND_NONE=1 关闭输出调整)
"""
import json, os, sys, zlib, csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/eval")); sys.path.insert(0, str(REPO))
SP = os.environ.get("FILMLAB_DATA", str(REPO / "data_cache"))
EP = SP + "/epfl/data"
GEO = SP + "/oracle"
MAN = {r["scene_id"]: r for r in csv.DictReader(open(REPO / "tools/eval/epfl_800t_manifest.csv"))}
IDS = ["R1-07-8", "R1-03-4", "R1-01-2", "R1-00-1", "R1-02-3", "R1-04-5", "R1-05-6", "R1-06-7",
       "R1-27-28", "R1-28-29", "R1-30-31", "cafe"]
GROUPS = {"day": ["cafe"], "night": IDS[:8], "tungsten": IDS[8:11]}
CELL = 24


SILLY_JPEG = ["film1", "film2", "film3", "film4", "film5", "castle", "chess", "swisstech", "single"]


def sources(sid):
    if sid.startswith("ss:"):        # SillyStill 其他配对: 数码端是相机 JPEG / PNG (已做白平衡), 结论需打折扣
        n = sid[3:]
        d, f = (f"digital{n[-1]}.jpeg", f"{n}.jpeg") if n.startswith("film") else (f"digital-{n}.png", f"film-{n}.png")
        return f"{SP}/pairs/sillystill/{d}", f"{SP}/pairs/sillystill/{f}", 0.5
    if sid == "cafe":
        return SP + "/pairs/sillystill/table_cafe_digital.ARW", SP + "/pairs/sillystill/table_cafe_film.JPG", 1.5
    r = MAN[sid]
    return f"{EP}/{r['raw_file']}", f"{EP}/{r['film_file']}", 0.53


def geometry(sid):
    """数码输入 → 真实扫描的单应性 (缓存; 固定随机种子)。与被测的物理条件无关。"""
    os.makedirs(GEO, exist_ok=True)
    f = f"{GEO}/geo_{sid}.npz"
    if os.path.exists(f):
        d = np.load(f)
        return d["H"], d["real"]
    import pair_compare as pc
    from PIL import Image
    from skimage.feature import ORB, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import ProjectiveTransform
    from skimage.exposure import equalize_adapthist
    rawp, scan, _ = sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    real = Image.open(scan).convert("RGB"); real.thumbnail((1600, 1600)); real = np.asarray(real, float) / 255
    prev = np.clip(rgb / np.percentile(rgb, 99), 0, 1) ** (1 / 2.2)        # 只用于找特征点
    kp, de = [], []
    for g in (equalize_adapthist(pc._gray(prev)), equalize_adapthist(pc._gray(real))):
        o = ORB(n_keypoints=4000, fast_threshold=0.03); o.detect_and_extract(g); kp.append(o.keypoints); de.append(o.descriptors)
    mt = match_descriptors(de[0], de[1], cross_check=True, max_ratio=0.85)
    mod, inl = ransac((kp[0][mt[:, 0]][:, ::-1], kp[1][mt[:, 1]][:, ::-1]), ProjectiveTransform, min_samples=4,
                      residual_threshold=3, max_trials=20000, rng=0)
    if mod is None or inl.sum() < 25:
        mod, inl = _sift(prev, real)           # 大块均匀色块 (灯箱) 上 ORB 特征点太少, 改用 SIFT
    if mod is None or inl.sum() < 20:
        raise RuntimeError(f"{sid}: 对齐失败")
    np.savez(f, H=mod.params, real=real, inliers=int(inl.sum()))
    return mod.params, real


def _sift(prev, real, s=0.5):
    import pair_compare as pc
    from skimage.feature import SIFT, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import ProjectiveTransform, resize
    from skimage.exposure import equalize_adapthist
    kp, de = [], []
    for im in (prev, real):
        g = equalize_adapthist(resize(pc._gray(im), (int(im.shape[0] * s), int(im.shape[1] * s))))
        d = SIFT(); d.detect_and_extract(g); kp.append(d.keypoints / s); de.append(d.descriptors)
    mt = match_descriptors(de[0], de[1], max_ratio=0.8, cross_check=True)
    if len(mt) < 8:
        return None, None
    return ransac((kp[0][mt[:, 0]][:, ::-1], kp[1][mt[:, 1]][:, ::-1]), ProjectiveTransform, min_samples=4,
                  residual_threshold=4, max_trials=20000, rng=0)


def cells(img, ok):
    h, w = img.shape[:2]
    out, idx = [], []
    for y in range(0, h - CELL + 1, CELL):
        for x in range(0, w - CELL + 1, CELL):
            if ok[y:y + CELL, x:x + CELL].mean() < 0.95:
                continue
            out.append(np.nanmedian(img[y:y + CELL, x:x + CELL].reshape(-1, img.shape[-1]), 0)); idx.append((y, x))
    return np.array(out), idx


def lab_lin(rgb):
    import colour
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb, 0, 1), apply_cctf_decoding=False))


def job(sid):
    import ablation_flags as af; af.install()
    import colour
    from dataclasses import replace
    from scipy.optimize import least_squares
    from skimage.transform import warp, ProjectiveTransform
    from scipy.ndimage import zoom
    import pair_compare as pc
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    Hm, real = geometry(sid)
    rawp, _, bias = sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    extra = {"output_conditioning": None} if os.environ.get("COND_NONE") else {}
    if os.environ.get("CORRECTIONS"):
        extra["corrections"] = tuple(os.environ["CORRECTIONS"].split(","))
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": seed % 36, **extra}))
    neg = eng.develop_negative(m)
    tr = {}
    auto = eng.scan(neg, trace=tr)
    sc, Dn = eng.scanner, tr["Dn"]
    h, w = Dn.shape[:2]
    W = neg.light_weights
    W = zoom(W, (h / W.shape[0], w / W.shape[1], 1), order=1)[:h, :w] if W is not None else np.zeros((h, w, 1))
    tf = ProjectiveTransform(Hm)
    wp = lambda a: warp(a, tf.inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
    stack = np.concatenate([Dn, auto, W], -1)
    sw = wp(stack)
    ok = np.all(np.isfinite(sw), -1)
    C, _ = cells(sw, ok)
    R = lab_lin(colour.models.eotf_sRGB(cells(real, ok)[0]))
    Dc, Ac, Wc = C[:, :3], C[:, 3:6], C[:, 6:]
    keep = (R[:, 0] > 5)
    t0 = sc.timing

    def render(t):
        sc.timing = t
        return sc.to_rgb(Dc)

    def de(S):
        v = keep & (S[:, 0] > 2)
        return float(np.median(colour.delta_E(S[v], R[v], method="CIE 2000"))), v

    def fit(level):
        off0, r0 = (t0.M if t0.M is not None else (np.zeros(3), np.ones(3)))

        def make(x):
            off = off0 + np.array([x[1], 0.0, x[2]])
            K = t0.K + x[0]
            if level == 1:
                return replace(t0, K=K, M=(off, r0))
            r = r0 * np.array([x[3], 1.0, x[4]])
            return replace(t0, K=K, M=(off, r), s=t0.s * x[5])

        def resid(x):
            S = lab_lin(render(make(x)))
            d = (S - R)[keep]
            return np.nan_to_num(d.ravel(), nan=50.0)
        x0 = [0, 0, 0] if level == 1 else [0, 0, 0, 1, 1, 1]
        lb = [-1.5, -0.8, -0.8] + ([1 - 0.15, 1 - 0.15, 0.8] if level == 2 else [])
        ub = [1.5, 0.8, 0.8] + ([1 + 0.15, 1 + 0.15, 1.25] if level == 2 else [])
        x = least_squares(resid, x0, bounds=(lb, ub), loss="soft_l1", f_scale=5.0).x
        return make(x), x

    res = {"n": int(keep.sum())}
    S_auto = lab_lin(render(t0))
    res["auto"] = de(S_auto)[0]
    for lv in (1, 2):
        t, x = fit(lv)
        S = lab_lin(render(t))
        e, v = de(S)
        cr, cs = np.hypot(*R[v, 1:].T), np.hypot(*S[v, 1:].T); ch = (cr + cs) / 2 > 10
        cls = np.argmax(Wc[v], 1) if Wc.shape[1] > 1 else np.zeros(v.sum(), int)
        per_cls = {}
        for k in np.unique(cls):
            mk = cls == k
            if mk.sum() >= 30:
                dd = S[v][mk] - R[v][mk]
                per_cls[int(k)] = [int(mk.sum()), float(np.median(colour.delta_E(S[v][mk], R[v][mk], method="CIE 2000"))),
                                   *map(float, np.median(dd, 0))]
        res[f"O{lv}"] = {"dE": e, "x": list(map(float, x)), "chroma": float(np.median(cs[ch]) / np.median(cr[ch])),
                         "dLab": list(map(float, np.median(S[v] - R[v], 0))), "by_light": per_cls}
    sc.timing = t0
    return sid, res


if __name__ == "__main__":
    from multiprocessing import Pool
    run = os.environ["SCENES"].split(",") if os.environ.get("SCENES") else IDS
    for sid in run:            # 单应性先串行求好 (缓存)
        geometry(sid)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4"))) as p:
        out = dict(p.map(job, run))
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for g, ids in GROUPS.items():
        ids = [i for i in ids if i in out]
        if not ids:
            continue
        a = np.mean([out[s]["auto"] for s in ids]); o1 = np.mean([out[s]["O1"]["dE"] for s in ids]); o2 = np.mean([out[s]["O2"]["dE"] for s in ids])
        print(f"{g:9s} 自动 {a:5.2f}  O1 {o1:5.2f}  O2 {o2:5.2f}")
    print(f"{'all':9s} 自动 {np.mean([out[s]['auto'] for s in out]):5.2f}  O1 {np.mean([out[s]['O1']['dE'] for s in out]):5.2f}"
          f"  O2 {np.mean([out[s]['O2']['dE'] for s in out]):5.2f}")
