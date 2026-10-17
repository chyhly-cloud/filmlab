"""RAW 配对误差分类: 每个格子记录 场景 / 引擎光源类别 / 真实 L* / 色相, 及模拟与真实的 Lab。用法: taxo_collect.py 输出.npz [条件]"""
import sys, os, csv, zlib, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")
from multiprocessing import Pool
from PIL import Image
SP = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad"; EP = SP + "/epfl/data"
MAN = {r["scene_id"]: r for r in csv.DictReader(open("/home/user/filmlab/tools/eval/epfl_800t_manifest.csv"))}
IDS = ["R1-07-8", "R1-03-4", "R1-01-2", "R1-00-1", "R1-02-3", "R1-04-5", "R1-05-6", "R1-06-7", "R1-27-28", "R1-28-29", "R1-30-31", "cafe"]
COND = sys.argv[2] if len(sys.argv) > 2 else "cur"
def job(sid):
    import pair_compare as pc
    if os.environ.get('WB65'):
        import wb65_variant; wb65_variant.install()
    from filmlab import lights, scanner as scm
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    from filmlab.io import srgb_encode
    from scipy.ndimage import zoom
    from skimage.transform import warp
    if sid == "cafe":
        rawp, scan, bias, cat = SP + "/pairs/sillystill/table_cafe_digital.ARW", SP + "/pairs/sillystill/table_cafe_film.JPG", 1.5, "daylight_cafe"
    else:
        r = MAN[sid]; rawp, scan, bias, cat = f"{EP}/{r['raw_file']}", f"{EP}/{r['film_file']}", 0.53, r["scene_category"]
    rgb = pc.load_digital(rawp, 1400)[0]
    real = Image.open(scan).convert("RGB"); real.thumbnail((1600, 1600)); real = np.asarray(real, float) / 255
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    rec = {}
    cw = lights.class_weights
    def cw2(*a, **k):
        w = cw(*a, **k); rec["W"] = w; return w
    lights.class_weights = cw2
    if COND in ("s0bare", "ref"): scm.SATURATION_MATRIX = np.eye(3)
    # ref = 固定物理参考输出: 按胶片标准 3200K 中灰一次性定标 (不逐帧), 无饱和度矩阵, 输出严格中性
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "standard" if COND == "ref" else "frame", "roll": 1, "frame": seed % 36,
                           **({"scanner": "frontier_neutral"} if COND == "ref" else {})}))
    if COND == "s0bare": eng.scanner.strength = 0.0; eng.scanner.m = scm.SCANNERS["frontier_neutral"]
    sim = srgb_encode(np.clip(eng.render(m), 0, 1))
    h, w = sim.shape[:2]; Wc = zoom(rec["W"], (h / rec["W"].shape[0], w / rec["W"].shape[1], 1), order=1)[:h, :w]
    try:
        warped, mask, n = pc.align(sim, real)
    except Exception as e:
        return sid, None
    # 用同一单应性 warp 光源权重: 重新求单应性 (与 align 相同步骤)
    from skimage.feature import ORB, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import ProjectiveTransform
    from skimage.exposure import equalize_adapthist
    kp, de = [], []
    for g in (equalize_adapthist(pc._gray(sim)), equalize_adapthist(pc._gray(real))):
        o = ORB(n_keypoints=3000, fast_threshold=0.03); o.detect_and_extract(g); kp.append(o.keypoints); de.append(o.descriptors)
    mt = match_descriptors(de[0], de[1], cross_check=True, max_ratio=0.85)
    mod, inl = ransac((kp[0][mt[:, 0]][:, ::-1], kp[1][mt[:, 1]][:, ::-1]), ProjectiveTransform, min_samples=4, residual_threshold=3, max_trials=5000)
    Ww = warp(Wc, mod.inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
    sim_w = warp(sim, mod.inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
    ok = np.all(np.isfinite(sim_w), -1)
    lab = lambda x: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(x, 0, 1)))
    Ls, Lr = lab(np.where(ok[..., None], sim_w, 0)), lab(real)
    c = 24; S, R, Wl = [], [], []
    for y in range(0, real.shape[0] - c + 1, c):
        for x in range(0, real.shape[1] - c + 1, c):
            if ok[y:y + c, x:x + c].mean() < 0.95: continue
            S.append(np.median(Ls[y:y + c, x:x + c].reshape(-1, 3), 0)); R.append(np.median(Lr[y:y + c, x:x + c].reshape(-1, 3), 0))
            Wl.append(np.nanmean(Ww[y:y + c, x:x + c].reshape(-1, Ww.shape[-1]), 0))
    return sid, dict(S=np.array(S), R=np.array(R), W=np.array(Wl), cat=cat)
if __name__ == "__main__":
    with Pool(6) as p:
        res = p.map(job, IDS)
    out = {"S": [], "R": [], "W": [], "scene": [], "cat": []}
    for sid, d in res:
        if d is None: print("对齐失败", sid); continue
        for k in ("S", "R", "W"): out[k].append(d[k])
        out["scene"] += [sid] * len(d["S"]); out["cat"] += [d["cat"]] * len(d["S"])
        print(sid, d["cat"], len(d["S"]), flush=True)
    np.savez(sys.argv[1], S=np.concatenate(out["S"]), R=np.concatenate(out["R"]), W=np.concatenate(out["W"]), scene=np.array(out["scene"]), cat=np.array(out["cat"]))
