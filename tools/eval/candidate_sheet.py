"""候选成片对照: 真实扫描 | 现行基线 | 各候选, 用与条件无关的固定单应性对齐到真实扫描, 每格标注中位 ΔE00。

用法: candidate_sheet.py 输出目录 场景… (场景 id 同 oracle_timing.sources)
"""
import os, sys, zlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")

SETS = {
    "timing": {
        "现行基线": {},
        "候选 C: US6160634 灰平衡": {"scan_gray_balance": "us6160634"},
        "冲印店预设: 锁定颜色 (胶片 3200K 中灰)": {"scan_neutral": "film", "scan_color_strength": 0.0},
    },
    "generation": {
        "旧基线 (legacy)": {"_generation": "legacy"},
        "正式基线 (current)": {},
        "下一代候选 (next)": {"_generation": "next"},
    },
    "all": {
        "旧基线 legacy": {"_generation": "legacy"},
        "正式默认 current": {},
        "研究基准 next": {"_generation": "next"},
        "实验 next_k5219 (Kodak 原始 5219)": {"_generation": "next_k5219"},
        "next + 候选 C (US6160634)": {"_generation": "next", "scan_gray_balance": "us6160634"},
        "next + US6160634 相对灰轴": {"_generation": "next", "scan_gray_balance": "us6160634_axis"},
        "next + 锁定颜色预设": {"_generation": "next", "scan_neutral": "film", "scan_color_strength": 0.0},
    },
    "cast": {      # 冷暖跟随 (wb_cast.py); 配合 SHEET_METRIC=cast: 不对齐, 标注整体偏色, 空灯箱也能用
        "正式默认 (逐帧定光 + 输出调整)": {"_generation": "next"},
        "逐帧定光, 无输出调整": {"_generation": "next", "output_conditioning": None},
        "锁定颜色 (不校色) + 输出调整": {"_generation": "next", "scan_neutral": "film", "scan_color_strength": 0.0},
        "锁定颜色, 无输出调整": {"_generation": "next", "scan_neutral": "film", "scan_color_strength": 0.0,
                                "output_conditioning": None},
    },
}
CANDIDATES = SETS[os.environ.get("SHEET_SET", "timing")]


def render(sid, extra):
    import oracle_timing as ot
    import pair_compare as pc
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    from filmlab.io import srgb_encode
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    from filmlab import generation
    extra = dict(extra); generation.use(extra.pop("_generation", "current"))
    try:
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": seed % 36, **extra}))
        return srgb_encode(np.clip(eng.render(m), 0, 1))
    finally:
        generation.use("current")


def sheet(args):
    out, sid = args
    import colour
    from PIL import Image, ImageDraw, ImageFont
    from skimage.transform import warp, ProjectiveTransform
    import oracle_timing as ot
    lab = lambda x: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(x, 0, 1)))
    if os.environ.get("SHEET_METRIC") == "cast":
        from wb_cast import cast
        real = Image.open(ot.sources(sid)[1]).convert("RGB"); real.thumbnail((1600, 1600)); real = np.asarray(real, float) / 255
        tag = lambda x: "a* %+.1f  b* %+.1f" % tuple(cast(x))
        tiles, labels = [real], ["真实扫描  " + tag(real)]
        for name, extra in CANDIDATES.items():
            sim = render(sid, extra)
            tiles.append(sim); labels.append(f"{name}  {tag(sim)}")
        return _save(out, sid, tiles, labels)
    Hm, real = ot.geometry(sid)
    tiles, labels = [real], ["真实扫描"]
    Lr = lab(real)
    for name, extra in CANDIDATES.items():
        sim = render(sid, extra)
        w = warp(sim, ProjectiveTransform(Hm).inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
        ok = np.all(np.isfinite(w), -1)
        de = colour.delta_E(lab(np.where(ok[..., None], w, 0))[ok], Lr[ok], method="CIE 2000")
        tiles.append(np.where(ok[..., None], w, 0)); labels.append(f"{name}  ΔE00 中位 {np.median(de):.1f}")
    return _save(out, sid, tiles, labels)


def _save(out, sid, tiles, labels):
    from PIL import Image, ImageDraw, ImageFont
    h = 420; ims = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 18)
    except OSError:
        font = ImageFont.load_default()
    for t, l in zip(tiles, labels):
        im = Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8))
        im = im.resize((int(im.width * h / im.height), h))
        canvas = Image.new("RGB", (im.width, h + 30), (20, 20, 20)); canvas.paste(im, (0, 30))
        ImageDraw.Draw(canvas).text((6, 5), l, fill=(235, 235, 235), font=font)
        if ImageDraw.Draw(canvas).textlength(l, font=font) > im.width - 12:      # 窄格: 标题与数值分两行
            canvas = Image.new("RGB", (im.width, h + 30), (20, 20, 20)); canvas.paste(im.crop((0, 22, im.width, h)), (0, 52))
            for k, part in enumerate(l.split("  ", 1)):
                ImageDraw.Draw(canvas).text((6, 4 + 24 * k), part, fill=(235, 235, 235), font=font)
        ims.append(canvas)
    per = int(os.environ.get("SHEET_COLS", "2" if len(ims) <= 4 else "4"))
    rows = [ims[i:i + per] for i in range(0, len(ims), per)]
    full = Image.new("RGB", (max(sum(i.width for i in r) for r in rows), len(rows) * (h + 30)), (20, 20, 20))
    for y, r in enumerate(rows):
        x = 0
        for i in r:
            full.paste(i, (x, y * (h + 30))); x += i.width
    f = os.path.join(out, f"cand_{sid.replace(':', '_')}.jpg")
    full.save(f, quality=88)
    return f, labels


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3"))) as p:
        for f, labels in p.imap_unordered(sheet, [(out, s) for s in sys.argv[2:]]):
            print(f, " | ".join(labels[1:]), flush=True)
