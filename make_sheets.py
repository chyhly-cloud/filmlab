"""对样张渲染所有配方, 生成对比图。用法: python make_sheets.py samples/a.NEF out/"""
import sys, time
from pathlib import Path
import numpy as np
from filmlab.rawio import load_raw, auto_expose
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe, RECIPES
from filmlab.io import save_srgb, contact_sheet

src, outdir = Path(sys.argv[1]), Path(sys.argv[2])
size = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
outdir.mkdir(parents=True, exist_ok=True)
rgb = auto_expose(load_raw(src, max_size=size))
ref = np.clip(rgb, 0, None)
tiles, labels = [save_srgb(ref / (1 + ref / 4), outdir / "digital.jpg")], ["数码直出 (线性+简单色调映射)"]
for name in RECIPES:
    t = time.time()
    r = Engine(Params(**recipe(name)))
    tiles.append(save_srgb(r.render(rgb), outdir / f"{name}.jpg"))
    labels.append(name)
    print(f"{name:36s} {time.time() - t:5.1f}s", flush=True)
for i in range(0, len(tiles), 12):
    contact_sheet(tiles[i:i + 12], labels[i:i + 12], outdir / f"sheet_{i // 12 + 1}.jpg", cols=4, width=480)
