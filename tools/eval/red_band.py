"""红带诊断: 光源周围的颜色与空间分布, 真实扫描 vs 模拟各级 (输入 → 直接曝光 → 光晕增量 → 成片)。

在真实扫描几何里, 以传感器饱和区 (灯芯) 为原点, 按距离 (胶片 µm) 分档, 只取局部窗口 (光源最密集处 ±1.8 mm) 内的像素:
  真实扫描 / 模拟成片: 线性 RGB 的中位与色度 r = R/(R+G+B), b = B/(R+G+B), 亮度 Y;
  模拟胶片: 三层直接曝光 (乳剂散射前), 光晕增量 (总曝光 − 散射后), 光晕 / 直接 的比例;
  模拟输入: 数码 RAW (测光后) 的线性 RGB 与色度 —— 判断泛光的颜色是不是输入带来的。
另输出局部图: 真实 | 模拟成片 | 模拟输入 (按灯芯亮度归一) | 光晕增量 / 直接曝光 (红层, 伪彩)。

用法: FILMLAB_DATA=… python3 red_band.py 输出目录 场景…
"""
import json, os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools/eval"), str(REPO)]
BINS = np.array([0, 25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500])
HALF = 80          # 局部窗口半宽 (真实扫描 1600 像素缩略图, 约 22.5 µm / 像素)


def oblique_patch(D):
    """实验: 光晕核加入下行斜程衰减 10^(−D·(1/cosθ − 1)) (D = 斜穿的乳剂密度; 垂直方向的衰减已在 halation_response 里)。"""
    from filmlab import film
    k0 = film.halation_kernel

    def kernel(px_um, base_um=130.0, n_base=1.5, forward=1.5, r_max_um=1500.0):
        th = np.linspace(1e-4, np.radians(89.5), 4000)
        r = 2.0 * base_um * np.tan(th)
        ang = np.sin(th) * np.cos(th) ** forward
        ang /= (ang * np.gradient(th)).sum()
        w = film._fresnel_unpolarized(th, n_base) * ang * np.gradient(th) * 10.0 ** (-D * (1.0 / np.cos(th) - 1.0))
        n = int(np.ceil(r_max_um / px_um))
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1] * px_um
        rr = np.hypot(yy, xx)
        edges = np.append(r, r[-1] * 1.01)
        ring = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        k = np.interp(rr, r, w / ring, right=0.0) * px_um ** 2
        return k * (w[r <= r_max_um].sum() / max(k.sum(), 1e-12))
    film.halation_kernel = kernel


def job(args):
    out, sid = args
    if os.environ.get("HALO_OBLIQUE_D"):
        oblique_patch(float(os.environ["HALO_OBLIQUE_D"]))
    import colour, rawpy
    import oracle_timing as ot, pair_compare as pc
    from scipy.ndimage import distance_transform_edt, zoom, uniform_filter
    from skimage.transform import warp, ProjectiveTransform
    from filmlab import generation, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    generation.use("next")
    Hm, real = ot.geometry(sid)
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": seed % 36, "grain": 0}))
    tr = {}
    neg = eng.develop_negative(m, trace=tr)
    out_lin = np.clip(eng.scan(neg), 0, None)
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    core = zoom(sat.astype(float), (m.shape[0] / sat.shape[0], m.shape[1] / sat.shape[1]), order=1) > 0.25
    tf = ProjectiveTransform(Hm)
    W = lambda a, o=1: warp(a, tf.inverse, output_shape=real.shape[:2], order=o, cval=np.nan)
    halo = tr["total_exposure"] - tr["scattered_exposure"]
    st = {"input": W(m), "direct": W(tr["direct_exposure"]), "halo": W(halo), "output": W(out_lin),
          "real": colour.models.eotf_sRGB(real)}
    cw = W(core.astype(float)) > 0.5
    um = 36000.0 / max(real.shape[:2])
    # 窗口: 灯芯最密集处
    dens = uniform_filter(cw.astype(float), 2 * HALF)
    dens[:HALF] = dens[-HALF:] = 0; dens[:, :HALF] = dens[:, -HALF:] = 0
    y, x = np.unravel_index(np.argmax(dens), dens.shape)
    win = (slice(y - HALF, y + HALF), slice(x - HALF, x + HALF))
    d = (distance_transform_edt(~cw) * um)[win]
    res = {"um_px": um, "center": [int(y), int(x)]}
    for k, a in st.items():
        a = a[win]
        rows = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            mk = (d > lo) & (d <= hi) & np.all(np.isfinite(a), -1)
            if mk.sum() < 20:
                rows.append(None); continue
            v = np.median(a[mk], 0)
            rows.append([float(x) for x in v])
        res[k] = rows
    # 局部图
    from PIL import Image, ImageDraw, ImageFont
    from filmlab.io import srgb_encode
    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 16)
    inp = st["input"][win]; inp = inp / max(np.nanpercentile(inp, 99), 1e-6)
    ratio = np.nan_to_num(st["halo"][win][..., 0] / np.maximum(st["direct"][win][..., 0], 1e-9))
    rmap = np.stack([np.clip(np.log10(np.maximum(ratio, 1e-3)) / 2 + 0.5, 0, 1)] * 3, -1) * np.array([1.0, 0.35, 0.2])
    tiles = [(real[win], "真实扫描"), (srgb_encode(np.clip(np.nan_to_num(st["output"][win]), 0, 1)), "模拟成片 (next)"),
             (srgb_encode(np.clip(np.nan_to_num(inp), 0, 1)), "模拟输入 (RAW, 归一)"),
             (rmap, "红层 光晕/直接 (0.1 黑 … 10 亮)")]
    ims = []
    for t, l in tiles:
        im = Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8)).resize((3 * 2 * HALF, 3 * 2 * HALF), Image.LANCZOS)
        c = Image.new("RGB", (im.width, im.height + 26), (20, 20, 20)); c.paste(im, (0, 26))
        ImageDraw.Draw(c).text((6, 4), l, fill=(235, 235, 235), font=font)
        ims.append(c)
    full = Image.new("RGB", (sum(i.width for i in ims), ims[0].height), (20, 20, 20))
    xx = 0
    for i in ims:
        full.paste(i, (xx, 0)); xx += i.width
    full.save(f"{out}/redband_{sid}{os.environ.get('TAG', '')}.jpg", quality=90)
    return sid, res


def report(res):
    def chrom(v):
        s = sum(v)
        return (v[0] / s, v[2] / s) if s > 0 else (np.nan, np.nan)
    for sid, r in res.items():
        print(f"\n== {sid}  (色度 r=R/Σ, b=B/Σ; 胶片层顺序 R G B)")
        print(f"{'µm':>6s} | {'真实 r':>6s} {'b':>5s} {'Y':>6s} | {'成片 r':>6s} {'b':>5s} {'Y':>6s} | {'输入 r':>6s} {'b':>5s} | "
              f"{'光晕/直接 R':>9s} {'G':>6s} {'B':>6s}")
        for i, hi in enumerate(BINS[1:]):
            R, O, I, D, H = (r[k][i] for k in ("real", "output", "input", "direct", "halo"))
            if None in (R, O, I, D, H):
                continue
            yr = 0.2126 * R[0] + 0.7152 * R[1] + 0.0722 * R[2]
            yo = 0.2126 * O[0] + 0.7152 * O[1] + 0.0722 * O[2]
            hr = [H[c] / max(D[c], 1e-9) for c in range(3)]
            print(f"{hi:6d} | {chrom(R)[0]:6.3f} {chrom(R)[1]:5.3f} {yr:6.3f} | {chrom(O)[0]:6.3f} {chrom(O)[1]:5.3f} {yo:6.3f} | "
                  f"{chrom(I)[0]:6.3f} {chrom(I)[1]:5.3f} | {hr[0]:9.3f} {hr[1]:6.3f} {hr[2]:6.3f}")


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3")), maxtasksperchild=1) as p:
        res = dict(p.map(job, [(out, s) for s in sys.argv[2:]]))
    json.dump(res, open(f"{out}/redband{os.environ.get('TAG', '')}.json", "w"), indent=1)
    report(res)
