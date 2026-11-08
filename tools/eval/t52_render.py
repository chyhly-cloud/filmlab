"""T52 整帧渲染: 不同画幅 (胶片面积) 下的整张图 (docs/tasks/T52_GRAIN_VISIBILITY.md)。

流程同 T48 / T50 / T51 的整帧 (EXIF 测光, rawio.SOURCE_STOPS = 0, HALO_FORWARD = 4, 冲印店逐帧定光, roll = None, 颗粒开)。
画幅 (只改物理量: 每个输出像素对应的胶片面积):
  35   36 × 24 mm, 原 RAW 半尺寸 (约 3000 × 2000, 约 12 µm/像素 = Noritsu 35 mm 标准尺寸);
  half 24 × 18 mm (半格, 4:3), 110  17 × 13 mm (110 胶卷)。
  · 场景先在整张 35 mm 画面上测光 (同一场景、同一曝光), 再按画幅的长宽比居中裁切 (长边方向与原图相同),
    重采样到与 35 mm 相同的输出像素总数 (用户要求), Params.frame_width_mm = 画幅长边。
  · 冲印店空间传递 (T51, Noritsu): 光学 σ 与 USM 半径按输出像素 (T51 的单位约定); 传感器孔径 = 冲印店在该画幅上的
    输出像素间距。输出像素数与 35 mm 相同时该间距就是渲染像素本身 (渲染像素已按面积积分), 所以按画幅登记一个
    pitch_um = 渲染像素间距的同一冲印店 (只在本脚本里, spatial.LABS 运行时加一项; 库代码不变)。
用法: python3 -I t52_render.py 代际 配方 画幅 长边像素(35 mm) 输出.npy RAW [进程数] [颗粒倍数]
输出: 线性 sRGB float16 (与 T51 相同), 旁边 *_meta.json (像素间距、裁切、定光)。
"""
import json
import logging
import resource
import sys
import time
import warnings
import zlib
from dataclasses import replace
from pathlib import Path

logging.getLogger("exifread").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from scipy.ndimage import zoom

FORMATS = {"35": (36.0, 24.0), "half": (24.0, 18.0), "110": (17.0, 13.0)}


def crop_to_format(img, fmt):
    """居中裁切到画幅的长宽比 (长边方向不变), 再重采样到 img 的像素总数。返回 (图, 裁切框 (y0, y1, x0, x1))。"""
    L, S = FORMATS[fmt]
    h, w = img.shape[:2]
    a = L / S
    if h >= w:
        hh, ww = (h, int(round(h / a))) if h / w <= a else (int(round(w * a)), w)
    else:
        ww, hh = (w, int(round(w / a))) if w / h <= a else (int(round(h * a)), h)
    y0, x0 = (h - hh) // 2, (w - ww) // 2
    c = img[y0:y0 + hh, x0:x0 + ww]
    k = np.sqrt(h * w / (hh * ww))
    if abs(k - 1) > 1e-3:
        c = np.clip(zoom(c, (k, k, 1), order=1), 0, None)
    return c, (int(y0), int(y0 + hh), int(x0), int(x0 + ww))


def main():
    gen, rec, fmt, size, dst, f = sys.argv[1:7]
    size = int(size)
    workers = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    gr = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    from filmlab import engine, generation, grain, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.scanning import spatial
    t = time.time()
    grain.SILVER_WORKERS = workers
    generation.use("t51_bw" if gen == "digital" else gen)
    rawio.SOURCE_STOPS = 0.0
    engine.HALO_FORWARD = 4.0
    ex = rawio.exif_exposure(f)
    rgb = rawio.load_raw(f, max_size=size)
    if max(rgb.shape[:2]) != size:
        k = size / max(rgb.shape[:2])
        rgb = np.clip(zoom(rgb, (k, k, 1), order=1), 0, None)
    m, _ = rawio.film_metering(rgb, rawio.clip_luminance(f), rng=np.random.default_rng(zlib.crc32(Path(f).stem.encode())),
                               t_original=ex[0] if ex else None)
    box = (0, m.shape[0], 0, m.shape[1])
    if fmt != "35":
        m, box = crop_to_format(m, fmt)
    L = FORMATS[fmt][0]
    px_um = L * 1000.0 / max(m.shape[:2])
    kw = {}
    lab = None if gen == "digital" else Engine(Params(**{**recipe(rec), "timing": "frame", "roll": None})).scan_spatial_lab()
    if lab is not None and fmt != "35":
        name = f"{[k for k, v in spatial.LABS.items() if v == lab][0]}@{fmt}"
        spatial.LABS[name] = replace(lab, pitch_um=px_um, name=lab.name + f" ({fmt}, 同像素数)")
        kw["scan_spatial"] = name
    if gen == "digital":
        out = m
    else:
        eng = Engine(Params(**{**recipe(rec), "timing": "frame", "roll": None, "frame": 0, "grain": gr,
                               "frame_width_mm": L, **kw}))
        out = eng.render(m)
        assert abs(eng.px_um - px_um) < 1e-6
    np.save(dst, np.clip(out, 0, 4).astype(np.float16))
    meta = dict(gen=gen, recipe=rec, format=fmt, frame_mm=FORMATS[fmt], px_um=px_um, shape=list(out.shape[:2]),
                crop_box_in_35mm=box, grain=gr, lab=kw.get("scan_spatial"))
    if gen != "digital":
        tm = eng.scanner.timing
        meta["timing"] = dict(K=np.asarray(tm.K).tolist(), s=np.asarray(tm.s).tolist())
    json.dump(meta, open(dst.replace(".npy", "_meta.json"), "w"), ensure_ascii=False)
    print(dst, gen, rec, fmt, out.shape, f"{px_um:.2f} µm/px", f"{time.time() - t:.0f} s",
          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
