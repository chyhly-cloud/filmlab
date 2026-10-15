"""夜景对比: 胶片摄影师用 CineStill 800T 按 EI 800 测光拍同一场景 (受相机光圈 / 手持快门的物理极限约束),
经过 RAW 降噪、高光重建、光晕、C-41 冲洗与扫描。
用法: python examples/night.py out/night RAW...
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filmlab.engine import Engine, Params          # noqa: E402
from filmlab.io import contact_sheet, save_srgb    # noqa: E402
import zlib                                         # noqa: E402
import numpy as np                                  # noqa: E402
from filmlab.rawio import clip_luminance, exif_exposure, film_metering, load_raw  # noqa: E402
from filmlab.recipes import recipe                 # noqa: E402

outdir = Path(sys.argv[1])
outdir.mkdir(parents=True, exist_ok=True)
tiles, labels = [], []
for i, f in enumerate(sys.argv[2:]):
    t = time.time()
    ex = exif_exposure(f)
    rgb, note = film_metering(load_raw(f, max_size=1600), clip_luminance(f),
                              rng=np.random.default_rng(zlib.crc32(Path(f).stem.encode())),
                              t_original=ex[0] if ex else None)
    print(f"{Path(f).stem}: {note}")
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": i}))
    tiles.append(save_srgb(eng.render(rgb), outdir / f"{Path(f).stem}.jpg"))
    labels.append(f"CineStill 800T · {Path(f).stem}")
    print(f"{Path(f).stem}: {time.time() - t:.1f}s", flush=True)
for k in range(0, len(tiles), 6):
    contact_sheet(tiles[k:k + 6], labels[k:k + 6], outdir / f"sheet_{k // 6 + 1}.jpg", cols=3, width=620)
