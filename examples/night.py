"""夜景对比: 摄影师的曝光意图不变, 换用 CineStill 800T 按 EI 800 测光拍同一场景,
经过 RAW 降噪、高光重建、光晕、C-41 冲洗与扫描。
用法: python examples/night.py out/night RAW...
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filmlab.engine import Engine, Params          # noqa: E402
from filmlab.io import contact_sheet, save_srgb    # noqa: E402
from filmlab.rawio import film_exposure, load_raw  # noqa: E402
from filmlab.recipes import recipe                 # noqa: E402

outdir = Path(sys.argv[1])
outdir.mkdir(parents=True, exist_ok=True)
tiles, labels = [], []
for i, f in enumerate(sys.argv[2:]):
    t = time.time()
    rgb = film_exposure(load_raw(f, max_size=1600))
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": i}))
    tiles.append(save_srgb(eng.render(rgb), outdir / f"{Path(f).stem}.jpg"))
    labels.append(f"CineStill 800T · {Path(f).stem}")
    print(f"{Path(f).stem}: {time.time() - t:.1f}s", flush=True)
for k in range(0, len(tiles), 6):
    contact_sheet(tiles[k:k + 6], labels[k:k + 6], outdir / f"sheet_{k // 6 + 1}.jpg", cols=3, width=620)
