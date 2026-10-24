"""T02 目视对照: 完整画面 (真实 | next | next + 档案) 与光源最密集处的 3 倍局部, 对齐到真实扫描并标全图 ΔE00 中位。

档案通过 accept_t02.apply() 注入 (与验收一致)。用法: FILMLAB_DATA=… python3 t02_sheet.py 参数.json 输出目录 场景…
"""
import json, os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools/eval"), str(REPO)]


def render(args):
    params, sid = args
    import accept_t02 as acc
    extra = acc.apply(params)
    import oracle_timing as ot, pair_compare as pc
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    from filmlab.io import srgb_encode
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": seed % 36,
                           "grain": 0, **extra}))
    return srgb_encode(np.clip(eng.render(m), 0, 1)), m @ np.array([0.2126, 0.7152, 0.0722])


def strip(tiles, labels, h):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 18)
    ims = []
    for t, l in zip(tiles, labels):
        im = Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8)).resize((int(t.shape[1] * h / t.shape[0]), h))
        c = Image.new("RGB", (im.width, h + 30), (20, 20, 20)); c.paste(im, (0, 30))
        ImageDraw.Draw(c).text((6, 5), l, fill=(235, 235, 235), font=font)
        ims.append(c)
    full = Image.new("RGB", (sum(i.width for i in ims), h + 30), (20, 20, 20))
    x = 0
    for i in ims:
        full.paste(i, (x, 0)); x += i.width
    return full


def sheet(out, params, sid, pool):
    import colour
    import oracle_timing as ot
    from scipy.ndimage import uniform_filter
    from skimage.transform import warp, ProjectiveTransform
    Hm, real = ot.geometry(sid)
    tf = ProjectiveTransform(Hm)
    outs = pool.map(render, [({}, sid), (params, sid)])
    lab = lambda x: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(x, 0, 1)))
    Lr = lab(real)
    tiles, labels = [real], ["真实扫描"]
    for name, (sim, _) in zip(("next (现值)", "next + T02 档案"), outs):
        w = warp(sim, tf.inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
        ok = np.all(np.isfinite(w), -1)
        de = np.median(colour.delta_E(lab(np.where(ok[..., None], w, 0))[ok], Lr[ok], method="CIE 2000"))
        tiles.append(np.where(ok[..., None], w, 0)); labels.append(f"{name}  ΔE00 {de:.2f}")
    f1 = os.path.join(out, f"t02_full_{sid}.jpg")
    strip(tiles, labels, 420).save(f1, quality=90)
    Y = warp(outs[0][1], tf.inverse, output_shape=real.shape[:2], order=1, cval=0)
    c = 160
    dens = uniform_filter((Y > 8.0).astype(float), c)
    dens[:c // 2] = dens[-c // 2:] = 0; dens[:, :c // 2] = dens[:, -c // 2:] = 0
    if dens.max() <= 0:
        return [f1]
    y, x = np.unravel_index(np.argmax(dens), dens.shape)
    crops = [t[y - c // 2:y + c // 2, x - c // 2:x + c // 2] for t in tiles]
    f2 = os.path.join(out, f"t02_lamp_{sid}.jpg")
    strip(crops, [l.split("  ")[0] for l in labels], 3 * c).save(f2, quality=90)
    return [f1, f2]


if __name__ == "__main__":
    params, out = json.load(open(sys.argv[1])), sys.argv[2]
    os.makedirs(out, exist_ok=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(2, maxtasksperchild=1) as p:
        for sid in sys.argv[3:]:
            print(*sheet(out, params, sid, p), flush=True)
