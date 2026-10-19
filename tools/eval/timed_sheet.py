"""目视验证: 真实扫描 | 模拟 (自动定光) | 模拟 (定光预言机 O1 的印放光号: 只改曝光与两个色光号, 不改物理)。
若光源周围的泛光主要来自成片亮度, 第三格的光源应恢复轮廓。印放光号取自 oracle_timing.py 的结果 (评估用, 不进入预设)。

用法: python3 timed_sheet.py oracle.json 输出目录 场景…
"""
import json, os, sys, zlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")


def job(args):
    oj, out, sid = args
    from dataclasses import replace
    import oracle_timing as ot, pair_compare as pc
    from PIL import Image, ImageDraw, ImageFont
    from skimage.transform import warp, ProjectiveTransform
    from filmlab import generation
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    from filmlab.io import srgb_encode
    generation.use("next")
    x = json.load(open(oj))[sid]["O1"]["x"]
    Hm, real = ot.geometry(sid)
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": seed % 36}))
    neg = eng.develop_negative(m)
    tr = {}
    auto = eng.scan(neg, trace=tr)
    sc, t0 = eng.scanner, tr["timing"]
    off0, r0 = t0.M if t0.M is not None else (np.zeros(3), np.ones(3))
    sc.timing = replace(t0, K=t0.K + x[0], M=(off0 + np.array([x[1], 0.0, x[2]]), r0))
    Dn = tr["Dn"]
    timed = sc.to_rgb(Dn.reshape(-1, 3)).reshape(Dn.shape)
    wp = lambda a: warp(srgb_encode(np.clip(a, 0, 1)), ProjectiveTransform(Hm).inverse, output_shape=real.shape[:2], order=1, cval=0)
    tiles = [real, wp(auto), wp(timed)]
    labels = ["真实扫描", "模拟 next (自动定光)", f"模拟 next (印放光号与真实一致: 曝光 {x[0]:+.2f} D)"]
    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 18)
    h, ims = 420, []
    for t, l in zip(tiles, labels):
        im = Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8))
        im = im.resize((int(im.width * h / im.height), h))
        c = Image.new("RGB", (im.width, h + 30), (20, 20, 20)); c.paste(im, (0, 30))
        ImageDraw.Draw(c).text((6, 5), l, fill=(235, 235, 235), font=font)
        ims.append(c)
    full = Image.new("RGB", (sum(i.width for i in ims), h + 30), (20, 20, 20))
    xx = 0
    for i in ims:
        full.paste(i, (xx, 0)); xx += i.width
    f = os.path.join(out, f"timed_{sid}.jpg")
    full.save(f, quality=90)
    return f


if __name__ == "__main__":
    oj, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3"))) as p:
        for f in p.imap_unordered(job, [(oj, out, s) for s in sys.argv[3:]]):
            print(f, flush=True)
