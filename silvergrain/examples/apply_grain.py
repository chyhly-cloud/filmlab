"""示例: 给一张照片加银盐颗粒 (silvergrain.apply_to_image)。不需要第三方图片: input 写 "synthetic" 时用合成的测试图。

用法:
  python -m silvergrain.examples.apply_grain 输入 (路径 | synthetic) 输出.jpg [--film kodak_trix_400] [--developer rodinal]
         [--k 1.0] [--mode grain|full] [--format 35mm] [--ev 0] [--filter 25] [--scanner noritsu_rpl]
         [--output-px 3000] [--phone 输出_手机.jpg] [--method auto|explicit|closed_form] [--seed 0] [--bits 8|16]
例:
  python -m silvergrain.examples.apply_grain synthetic /tmp/trix.jpg --film kodak_trix_400 --phone /tmp/trix_phone.jpg
  python -m silvergrain.examples.apply_grain photo.jpg out.jpg --film kodak_portra_400 --mode full
"""
import argparse
import json

import numpy as np


def synthetic(h=2000, w=3000):
    """合成测试图 (线性 sRGB): 天空渐变、灰阶楔、彩色块、柔和的圆盘 (类似离焦背景前的皮肤), 没有任何第三方内容。"""
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    sky = np.stack([0.25 + 0.2 * y, 0.38 + 0.18 * y, 0.65 - 0.1 * y], -1) * (0.6 + 0.5 * (1 - y))[..., None]
    img = sky.copy()
    lower = y > 0.55
    ground = np.stack([0.10 + 0.05 * x, 0.12 + 0.03 * x, 0.06 + 0.02 * x], -1)
    img[lower] = ground[lower]
    # 灰阶楔: 18% 灰 × 2^(−5 … +3) 档
    n = 9
    for i in range(n):
        x0, x1 = int(w * (0.05 + 0.9 * i / n)), int(w * (0.05 + 0.9 * (i + 1) / n))
        img[int(h * 0.82):int(h * 0.95), x0:x1] = 0.18 * 2.0 ** (i - 5)
    # 彩色块 (Macbeth 量级的反射率)
    cols = [(0.40, 0.08, 0.06), (0.08, 0.25, 0.08), (0.07, 0.10, 0.35), (0.55, 0.45, 0.06), (0.45, 0.30, 0.22), (0.05, 0.25, 0.30)]
    for i, c in enumerate(cols):
        x0 = int(w * (0.06 + 0.13 * i))
        img[int(h * 0.62):int(h * 0.76), x0:x0 + int(w * 0.1)] = c
    # 柔和的圆盘
    r = np.hypot((x - 0.75) * w / h, y - 0.3)
    m = np.clip((0.16 - r) / 0.03, 0, 1)[..., None]
    img = img * (1 - m) + m * np.array([0.42, 0.28, 0.20]) * (1.1 - 0.6 * np.clip(r / 0.16, 0, 1))[..., None]
    return img


def main(argv=None):
    import silvergrain as sg
    p = argparse.ArgumentParser(description="给照片加银盐颗粒 (silvergrain)")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--film", default="kodak_trix_400"); p.add_argument("--developer", default=None)
    p.add_argument("--k", type=float, default=1.0); p.add_argument("--mode", default="grain", choices=("grain", "full"))
    p.add_argument("--format", default="35mm"); p.add_argument("--ev", type=float, default=0.0)
    p.add_argument("--filter", default=None); p.add_argument("--scanner", default=None)
    p.add_argument("--output-px", type=int, default=None); p.add_argument("--phone", default=None)
    p.add_argument("--method", default="auto"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bits", type=int, default=8)
    a = p.parse_args(argv)
    src = synthetic() if a.input == "synthetic" else a.input
    fmt = a.format if a.format in sg.FILM_FORMATS else float(a.format)
    r = sg.apply_to_image(src, a.film, a.developer, a.k, fmt, a.output_px, a.mode, a.ev, filter=a.filter, scanner=a.scanner,
                          seed=a.seed, method=a.method)
    r.save(a.output, a.bits)
    if a.phone:
        from PIL import Image
        v = sg.phone_size(r.encoded * 255.0)
        Image.fromarray(np.clip(np.round(v), 0, 255).astype(np.uint8)).save(a.phone, quality=95)
    info = {k: r.info[k] for k in ("film_name", "developer", "grain_scale", "mode", "px_um", "method", "emulsion", "seconds")}
    print(json.dumps(info, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
