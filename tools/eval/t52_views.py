"""T52 出图: 整张原尺寸 JPEG (质量 95)、手机尺寸 (长边 1200, Lanczos, 质量 92)、手机尺寸并排 (显影液 / 片种; 画幅)、
手机尺寸局部的 3 倍放大 (只作诊断, 缩略图上红框标位置)。不加任何输出端处理 (与 T51 的出图相同: 线性 → sRGB 编码 → 8 位)。
用法: python3 -I t52_views.py 帧目录 出图目录
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from filmlab.scanning.spatial import encode  # noqa: E402

FONT = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 28)
PHONE = 1200
SCENES = {"people_61227": "人像", "day_9195": "日景"}
# (帧名后缀, 文件名里的说明, 并排图的标签)
CASES = [
    ("trix_d76_35", "TriX400_D76_35mm", "Tri-X 400 / D-76 / 35 mm"),
    ("trix_rodinal_35", "TriX400_Rodinal_35mm", "Tri-X 400 / Rodinal / 35 mm"),
    ("hp5_d76_35", "HP5+_D76_35mm", "HP5+ / D-76 / 35 mm"),
    ("hp5_rodinal_35", "HP5+_Rodinal_35mm", "HP5+ / Rodinal / 35 mm"),
    ("trix_rodinal_half", "TriX400_Rodinal_半格24x18", "Tri-X 400 / Rodinal / 半格 24×18"),
    ("trix_rodinal_110", "TriX400_Rodinal_110画幅17x13", "Tri-X 400 / Rodinal / 110 17×13"),
]
DEV_SET = ["trix_d76_35", "trix_rodinal_35", "hp5_rodinal_35"]
FMT_SET = ["trix_rodinal_35", "trix_rodinal_half", "trix_rodinal_110"]
# 诊断框 (35 mm 原尺寸坐标 x0, y0, x1, y1; 平坦中间调: 人像的背景墙, 日景的天空)
DIAG = {"people_61227": (1680, 220, 2000, 540), "day_9195": (1300, 250, 1700, 650)}


def load(p):
    a = np.load(p).astype(np.float32)
    return Image.fromarray((np.clip(encode(np.clip(a, 0, 1)), 0, 1) * 255 + 0.5).astype(np.uint8))


def phone(im, L=PHONE):
    k = L / max(im.size)
    return im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)


def titled(im, text):
    s = Image.new("RGB", (im.width, im.height + 44), (20, 20, 20))
    ImageDraw.Draw(s).text((10, 6), text, fill=(240, 240, 240), font=FONT)
    s.paste(im, (0, 44))
    return s


def row(ims, vertical=False, gap=12):
    if vertical:
        s = Image.new("RGB", (max(i.width for i in ims), sum(i.height for i in ims) + gap * (len(ims) - 1)), (20, 20, 20))
        y = 0
        for i in ims:
            s.paste(i, (0, y)); y += i.height + gap
        return s
    s = Image.new("RGB", (sum(i.width for i in ims) + gap * (len(ims) - 1), max(i.height for i in ims)), (20, 20, 20))
    x = 0
    for i in ims:
        s.paste(i, (x, 0)); x += i.width + gap
    return s


def scene_box(meta, box):
    """35 mm 原尺寸坐标的框 → 该画幅帧的坐标 (裁切 + 重采样)。"""
    y0, y1, x0, x1 = meta["crop_box_in_35mm"]
    k = max(meta["shape"]) / max(y1 - y0, x1 - x0)
    return tuple(round(v) for v in ((box[0] - x0) * k, (box[1] - y0) * k, (box[2] - x0) * k, (box[3] - y0) * k))


def diag(phones, metas, labels, box, dst, scale=3, size=110):
    """手机尺寸图上同一景物位置的 size × size 像素, 最近邻放大 scale 倍; 左侧缩略图标位置。"""
    tiles = []
    for im, m, lab in zip(phones, metas, labels):
        b = scene_box(m, box)
        k = im.width / m["shape"][1]
        cx, cy = (b[0] + b[2]) / 2 * k, (b[1] + b[3]) / 2 * k
        c = (round(cx - size / 2), round(cy - size / 2), round(cx + size / 2), round(cy + size / 2))
        th = im.resize((im.width // 3, im.height // 3), Image.LANCZOS)
        ImageDraw.Draw(th).rectangle(tuple(v / 3 for v in c), outline=(255, 40, 40), width=3)
        z = im.crop(c).resize((size * scale, size * scale), Image.NEAREST)
        tiles.append(titled(row([th, z], gap=8), lab))
    row(tiles, vertical=True).save(dst, quality=92)


def main(fdir, vdir):
    fdir, vdir = Path(fdir), Path(vdir)
    vdir.mkdir(parents=True, exist_ok=True)
    for scene, zh in SCENES.items():
        ims, metas = {}, {}
        for key, fname, lab in CASES:
            p = fdir / f"{scene}_{key}.npy"
            if not p.exists():
                print("缺", p); continue
            metas[key] = json.load(open(str(p).replace(".npy", "_meta.json")))
            im = load(p)
            ims[key] = phone(im)
            im.save(vdir / f"{zh}_{fname}_整张原尺寸{im.width}x{im.height}.jpg", quality=95)
            ims[key].save(vdir / f"{zh}_{fname}_手机尺寸1200.jpg", quality=92)
            print(scene, key, im.size, flush=True)
        portrait = all(i.height > i.width for i in ims.values())
        lab = {k: l for k, _, l in CASES}
        for tag, keys in (("显影液与片种_TriX-D76_vs_TriX-Rodinal_vs_HP5-Rodinal", DEV_SET),
                          ("画幅_35mm_vs_半格_vs_110", FMT_SET)):
            if all(k in ims for k in keys):
                row([titled(ims[k], lab[k]) for k in keys], vertical=not portrait).save(
                    vdir / f"{zh}_对照_手机尺寸_{tag}.jpg", quality=92)
                diag([ims[k] for k in keys], [metas[k] for k in keys], [lab[k] for k in keys], DIAG[scene],
                     vdir / f"{zh}_诊断_手机尺寸局部3倍_{tag.split('_')[0]}.jpg")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
