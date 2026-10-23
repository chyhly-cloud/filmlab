"""光晕局部对照: 真实扫描 | 模拟 (光晕 ×1) | 模拟 (光晕 ×k) | 关闭光晕, 对齐到真实扫描, 裁取亮光源最密集的区域并放大。

用法: python3 halo_sheet.py 输出目录 k 场景…
"""
import os, sys, zlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")


def render(sid, scale):
    from filmlab import film, generation
    import oracle_timing as ot, pair_compare as pc
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    from filmlab.io import srgb_encode
    generation.use("next")
    hr0 = film.halation_response
    film.halation_response = lambda st, li, sf, v, br, method="spectral": hr0(st, li, sf, v, br * (scale or 0.0), method)
    try:
        rawp, _, bias = ot.sources(sid)
        rgb = pc.load_digital(rawp, 1400)[0]
        seed = zlib.crc32(sid.encode()) % 1000
        m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": seed % 36}))
        if not scale:
            eng.halo = None
        return srgb_encode(np.clip(eng.render(m), 0, 1)), m
    finally:
        film.halation_response = hr0


def sheet(args):
    out, k, sid = args
    import oracle_timing as ot
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import uniform_filter
    from skimage.transform import warp, ProjectiveTransform
    Hm, real = ot.geometry(sid)
    tf = ProjectiveTransform(Hm)
    wp = lambda a: warp(a, tf.inverse, output_shape=real.shape[:2], order=1, cval=0.0)
    tiles, labels = [real], ["真实扫描"]
    for name, s in ((f"模拟 光晕 ×1 (现行)", 1.0), (f"模拟 光晕 ×{k:g}", k), ("模拟 关闭光晕", 0.0)):
        img, m = render(sid, s)
        tiles.append(wp(img)); labels.append(name)
    Y = wp(m @ np.array([0.2126, 0.7152, 0.0722]))
    h, w = Y.shape
    c = 180
    dens = uniform_filter((Y > 8.0).astype(float), c)
    dens[:c // 2] = dens[-c // 2:] = 0; dens[:, :c // 2] = dens[:, -c // 2:] = 0
    y, x = np.unravel_index(np.argmax(dens), dens.shape)
    ims = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 18)
    for t, l in zip(tiles, labels):
        crop = np.clip(t[y - c // 2:y + c // 2, x - c // 2:x + c // 2], 0, 1)
        im = Image.fromarray((crop * 255).astype(np.uint8)).resize((3 * c, 3 * c), Image.LANCZOS)
        canvas = Image.new("RGB", (im.width, im.height + 30), (20, 20, 20)); canvas.paste(im, (0, 30))
        ImageDraw.Draw(canvas).text((6, 5), l, fill=(235, 235, 235), font=font)
        ims.append(canvas)
    full = Image.new("RGB", (sum(i.width for i in ims), ims[0].height), (20, 20, 20))
    xx = 0
    for i in ims:
        full.paste(i, (xx, 0)); xx += i.width
    f = os.path.join(out, f"halo_{sid}.jpg")
    full.save(f, quality=90)
    return f


if __name__ == "__main__":
    out, k = sys.argv[1], float(sys.argv[2]); os.makedirs(out, exist_ok=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3"))) as p:
        for f in p.imap_unordered(sheet, [(out, k, s) for s in sys.argv[3:]]):
            print(f, flush=True)
