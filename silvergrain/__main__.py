"""给任意照片加银盐颗粒的命令行 (T55k)。

    python -m silvergrain 输入 输出 [--film kodak_trix_400] [--developer d76] [--grain-scale 1.0] [--format 35mm]
                          [--mode grain|full] [--filter 25] [--ev 0] [--seed 0] [--bits 8|16] [--workers auto]
                          [--output-px N] [--timing auto|fixed] [--raw-p99 0.85] [--list]

输入: JPEG / PNG / TIFF (8 / 16 位, sRGB 编码); DNG 与相机 RAW (.dng .cr2 .cr3 .nef .arw .raf .orf .rw2 .pef .srw) 经 rawpy
(可选依赖) 解成线性 sRGB, 再乘常数使亮度 (BT.709) 的第 99 百分位 = --raw-p99 (默认 0.85, 相当于相机的曝光提升, 不改影调形状;
同 examples/scenes_fivek.json 的转换)。输出: .jpg / .png (8 位) 或 .png / .tif (16 位, 需要 imageio 或 tifffile)。
结果与 --workers 无关 (逐位相同)。--list 列出可用的胶片、显影液与画幅。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

RAW_EXT = (".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw", ".raw", ".rwl", ".3fr", ".iiq")


def read_raw(path, p99=0.85):
    """相机 RAW / DNG → 线性 sRGB (float64, (h, w, 3))。需要 rawpy。"""
    try:
        import rawpy
    except ImportError as e:
        raise SystemExit("读取 RAW / DNG 需要 rawpy: pip install rawpy") from e
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True,
                              output_color=rawpy.ColorSpace.sRGB, highlight_mode=rawpy.HighlightMode.Blend)
    lin = rgb.astype(np.float64) / 65535.0
    if p99:
        Y = lin @ np.array([0.2126, 0.7152, 0.0722])
        q = float(np.percentile(Y, 99))
        if q > 0:
            lin *= p99 / q
    return lin


def read_input(path, p99=0.85):
    """路径 → apply_to_image 的输入 (RAW 为线性浮点; 其他为文件路径, 由 load_image 按 sRGB 解码)。"""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"找不到输入文件: {path}")
    if p.suffix.lower() in RAW_EXT:
        return read_raw(p, p99)
    return str(p)


def write_output(path, lin, bits=8):
    from .image import save_image, srgb_encode
    ext = Path(path).suffix.lower()
    if bits == 16:
        if ext not in (".png", ".tif", ".tiff"):
            raise SystemExit("16 位输出请用 .png 或 .tif")
        v = np.round(srgb_encode(np.clip(lin, 0, 1)) * 65535).astype(np.uint16)
        try:
            import imageio.v3 as iio
            iio.imwrite(path, v)
            return
        except ImportError:
            pass
        if ext in (".tif", ".tiff"):
            try:
                import tifffile
                tifffile.imwrite(path, v)
                return
            except ImportError:
                pass
        raise SystemExit("16 位输出需要 imageio 或 tifffile")
    if ext not in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"):
        raise SystemExit(f"不认识的输出格式: {ext}")
    save_image(path, lin, 8)


def _list():
    import silvergrain as sg
    print("胶片 (--film):")
    for k, f in sg.films().items():
        print(f"  {k:28s} {f.name}")
    print("显影液 (--developer; 黑白片, 默认 d76; 彩色片按片的原生冲洗):")
    for k, d in sg.developers().items():
        print(f"  {k:28s} {d.name}")
    print("画幅 (--format, 或长边毫米数):", ", ".join(sg.FILM_FORMATS))


def parser():
    p = argparse.ArgumentParser(prog="python -m silvergrain", description="给照片加银盐颗粒 (silvergrain)")
    p.add_argument("input", nargs="?"); p.add_argument("output", nargs="?")
    p.add_argument("--film", default="kodak_trix_400", help="胶片 (--list 列出)")
    p.add_argument("--developer", default=None, help="显影液 (黑白; 默认 d76)")
    p.add_argument("--grain-scale", type=float, default=1.0, help="颗粒粗细 k (0.5–3)")
    p.add_argument("--format", default="35mm", help='画幅 ("35mm", "645", "6x7" … 或长边毫米数)')
    p.add_argument("--mode", default="grain", choices=("grain", "full"), help="grain = 只加颗粒; full = 整卷胶片的影调与颜色")
    p.add_argument("--filter", default=None, help="Wratten 滤镜号 (例 25)")
    p.add_argument("--ev", type=float, default=0.0, help="曝光补偿 (档)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bits", type=int, default=8, choices=(8, 16))
    p.add_argument("--workers", default="auto", help='并行数 ("auto" 或整数; 不影响结果)')
    p.add_argument("--output-px", type=int, default=None, help="输出长边像素 (默认原尺寸)")
    p.add_argument("--timing", default="auto", choices=("auto", "greyworld", "fixed"), help="full 模式的冲印店定光")
    p.add_argument("--raw-p99", type=float, default=0.85, help="RAW: 亮度第 99 百分位缩放到这个线性值 (0 = 不缩放)")
    p.add_argument("--list", action="store_true", help="列出胶片、显影液与画幅")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv=None):
    import silvergrain as sg
    a = parser().parse_args(argv)
    if a.list:
        _list()
        return 0
    if not a.input or not a.output:
        parser().print_usage()
        return 2
    fmt = a.format if a.format in sg.FILM_FORMATS else float(a.format)
    workers = a.workers if a.workers == "auto" else int(a.workers)
    t = time.time()
    src = read_input(a.input, a.raw_p99)
    r = sg.apply_to_image(src, a.film, a.developer, a.grain_scale, fmt, a.output_px, a.mode, a.ev, filter=a.filter,
                          timing=a.timing, seed=a.seed, workers=workers)
    write_output(a.output, r.linear, a.bits)
    if not a.quiet:
        h, w = r.linear.shape[:2]
        print(f"{a.output}: {w} × {h}, {r.info.get('film_name', a.film)}, {r.info.get('mode')}, "
              f"渲染法 {r.info.get('method')}, {time.time() - t:.1f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
