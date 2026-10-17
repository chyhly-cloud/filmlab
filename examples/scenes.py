"""同一配方在多种场景下的表现: 数码直出 / 第 1 卷 / 第 2 卷 (不同胶卷与冲洗, 物理随机范围内的差异)。
负片按冲印店扫描输出, 扫描仪按整张底片自动校正。
用法: python examples/scenes.py cinestill_800t out/cinestill  场景RAW... [--ev 场景:EV ...]"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filmlab.engine import Engine, Params          # noqa: E402
from filmlab.io import contact_sheet, save_srgb    # noqa: E402
from filmlab.rawio import auto_expose, load_raw    # noqa: E402
from filmlab.recipes import recipe                 # noqa: E402


def main():
    name, outdir, *rest = sys.argv[1:]
    files, ev = [], {}
    for a in rest:
        if ":" in a:
            k, v = a.split(":")
            ev[k] = float(v)
        else:
            files.append(a)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tiles, labels = [], []
    for f in files:
        stem = Path(f).stem
        t = time.time()
        key = 0.18 * 2.0 ** ev.get(stem, 0.0)
        cam = auto_expose(load_raw(f, max_size=1400, white_balance="camera"), key)
        day = auto_expose(load_raw(f, max_size=1400, white_balance="daylight"), key)
        c = np.clip(cam, 0, None)
        tiles.append(save_srgb(c / (1 + c / 4), outdir / f"{stem}_digital.jpg"))
        labels.append(f"{stem} 数码直出")
        for roll in (1, 2):
            eng = Engine(Params(**{**recipe(name), "timing": "frame", "roll": roll, "frame": len(tiles)}))
            tiles.append(save_srgb(eng.render(day), outdir / f"{stem}_roll{roll}.jpg"))
            labels.append(f"{name} 第{roll}卷")
        print(f"{stem}: {time.time() - t:.1f}s", flush=True)
    for i in range(0, len(tiles), 12):
        contact_sheet(tiles[i:i + 12], labels[i:i + 12], outdir / f"sheet_{i // 12 + 1}.jpg", cols=3, width=520)


if __name__ == "__main__":
    main()
