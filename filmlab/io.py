import numpy as np
from PIL import Image

from .spectral import srgb_encode


def save_srgb(lin, path, quality=92):
    img = (srgb_encode(lin) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(path, quality=quality)
    return img


def contact_sheet(tiles, labels, path, cols=3, width=600):
    """把多张结果拼成带标签的对比图。"""
    from PIL import ImageDraw, ImageFont
    ims = [Image.fromarray(t) if isinstance(t, np.ndarray) else t for t in tiles]
    h = int(width * ims[0].height / ims[0].width)
    rows = (len(ims) + cols - 1) // cols
    pad = 34
    sheet = Image.new("RGB", (cols * width, rows * (h + pad)), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 20)
    except OSError:
        font = ImageFont.load_default(20)
    for i, (im, lab) in enumerate(zip(ims, labels)):
        x, y = (i % cols) * width, (i // cols) * (h + pad)
        sheet.paste(im.resize((width, h), Image.LANCZOS), (x, y + pad))
        d.text((x + 8, y + 5), lab, fill=(235, 235, 235), font=font)
    sheet.save(path, quality=90)
