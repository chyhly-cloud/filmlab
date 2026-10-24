"""T01 目视对照: 真实扫描 | next | next + 校准候选 (自动定光, 与用户实际看到的成片一致), 对齐到真实扫描并标 ΔE00 中位。

用法: FILMLAB_DATA=… python3 t01_sheet.py 参数.json 输出目录 场景…
"""
import json, os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools/eval"), str(REPO)]


def render(args):
    params, sid = args
    import accept_calibration as acc
    acc.apply(params)
    import oracle_timing as ot, pair_compare as pc
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    from filmlab.io import srgb_encode
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": seed % 36, "grain": 0}))
    return srgb_encode(np.clip(eng.render(m), 0, 1))


def sheet(out, params, sid, pool):
    import colour
    import oracle_timing as ot
    from PIL import Image, ImageDraw, ImageFont
    from skimage.transform import warp, ProjectiveTransform
    Hm, real = ot.geometry(sid)
    sims = pool.map(render, [({}, sid), (params, sid)])
    lab = lambda x: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(x, 0, 1)))
    Lr = lab(real)
    tiles, labels = [real], ["真实扫描"]
    for name, sim in zip(("next (现值)", "next + T01 候选"), sims):
        w = warp(sim, ProjectiveTransform(Hm).inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
        ok = np.all(np.isfinite(w), -1)
        de = np.median(colour.delta_E(lab(np.where(ok[..., None], w, 0))[ok], Lr[ok], method="CIE 2000"))
        tiles.append(np.where(ok[..., None], w, 0)); labels.append(f"{name}  ΔE00 {de:.2f}")
    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 18)
    h, ims = 420, []
    for t, l in zip(tiles, labels):
        im = Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8)).resize((int(t.shape[1] * h / t.shape[0]), h))
        c = Image.new("RGB", (im.width, h + 30), (20, 20, 20)); c.paste(im, (0, 30))
        ImageDraw.Draw(c).text((6, 5), l, fill=(235, 235, 235), font=font)
        ims.append(c)
    full = Image.new("RGB", (sum(i.width for i in ims), h + 30), (20, 20, 20))
    x = 0
    for i in ims:
        full.paste(i, (x, 0)); x += i.width
    f = os.path.join(out, f"t01_{sid}.jpg")
    full.save(f, quality=90)
    return f


if __name__ == "__main__":
    params, out = json.load(open(sys.argv[1])), sys.argv[2]
    os.makedirs(out, exist_ok=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(2, maxtasksperchild=1) as p:
        for sid in sys.argv[3:]:
            print(sheet(out, params, sid, p), flush=True)
