"""光晕检验的距离基准: 亮区在胶片上的实际范围 (真实扫描 / 模拟 ×1 / ×0.1 / 关闭光晕) 与数码光源掩膜的差距,
并输出完整画面的四联对照图。

halo_edge.py 以"到数码输入亮光源的距离"为横轴; 若胶片上的过曝区本身比数码掩膜大, 近处 (≤ 300 µm) 的读数
就混入了光源边缘, 不能当作光晕的空间分布。这里比较各版本 Y > 0.8 的近白区在掩膜外的范围。

用法: python3 halo_footprint.py 输出目录 场景…
"""
import os, sys, zlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")

VARIANTS = (("模拟 光晕 ×1 (现行)", 1.0), ("模拟 光晕 ×0.1 (next_halo)", 0.1), ("模拟 关闭光晕", 0.0))


def job(args):
    out, sid = args
    import colour
    import oracle_timing as ot
    import halo_edge as he
    import halo_sheet as hs
    from scipy.ndimage import distance_transform_edt
    from skimage.transform import warp, ProjectiveTransform
    from PIL import Image, ImageDraw, ImageFont
    Hm, real = ot.geometry(sid)
    tf = ProjectiveTransform(Hm)
    wp = lambda a: warp(a, tf.inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
    tiles, labels, Y = [real], ["真实扫描"], {}
    m = None
    for name, k in VARIANTS:
        img, m = hs.render(sid, k)
        w = wp(img)
        tiles.append(np.nan_to_num(w)); labels.append(name)
        Y[name] = colour.models.eotf_sRGB(np.clip(w, 0, 1)) @ [0.2126, 0.7152, 0.0722]
    Y["真实扫描"] = colour.models.eotf_sRGB(real) @ [0.2126, 0.7152, 0.0722]
    Yin = wp(m @ np.array([0.2126, 0.7152, 0.0722]))
    ok = np.isfinite(Yin)
    src = ok & (Yin > he.SRC)
    um = he.FRAME_MM * 1000.0 / max(real.shape[:2])
    d = distance_transform_edt(~src) * um
    stats = {}
    for name, y in Y.items():
        near = ok & np.isfinite(y) & (y > 0.8) & ~src & (d < 1000)
        stats[name] = (float(np.median(d[near])) if near.any() else np.nan, float(near.sum() / max(src.sum(), 1)))
    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 18)
    h, ims = 420, []
    for t, l in zip(tiles, labels):
        im = Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8))
        im = im.resize((int(im.width * h / im.height), h))
        canvas = Image.new("RGB", (im.width, h + 30), (20, 20, 20)); canvas.paste(im, (0, 30))
        ImageDraw.Draw(canvas).text((6, 5), l, fill=(235, 235, 235), font=font)
        ims.append(canvas)
    rows = [ims[:2], ims[2:]]
    full = Image.new("RGB", (max(sum(i.width for i in r) for r in rows), 2 * (h + 30)), (20, 20, 20))
    for yy, r in enumerate(rows):
        xx = 0
        for i in r:
            full.paste(i, (xx, yy * (h + 30))); xx += i.width
    f = os.path.join(out, f"full_{sid}.jpg")
    full.save(f, quality=90)
    return sid, int(src.sum()), stats, f


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    import multiprocessing as mp
    names = ["真实扫描"] + [n for n, _ in VARIANTS]
    print("近白区 (Y > 0.8) 在数码光源掩膜外: 距离中位 µm / 面积 ÷ 掩膜面积")
    print(f"{'场景':9s} " + " | ".join(names))
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3"))) as p:
        for sid, n, st, f in p.imap_unordered(job, [(out, s) for s in sys.argv[2:]]):
            print(f"{sid:9s} " + " | ".join(f"{st[k][0]:5.0f} / {st[k][1]:4.2f}" for k in names), flush=True)
