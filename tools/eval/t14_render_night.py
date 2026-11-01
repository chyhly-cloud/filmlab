"""夜景 RAW → 胶片 (examples/night.py 流程), 可选环境变量 STOPS (rawio.SOURCE_STOPS)。用法: GEN STOPS 输出目录 RAW..."""
import sys, zlib, warnings, os
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path[:0] = ["/home/user/filmlab"]

def job(a):
    gen, stops, out, f, W = a
    from filmlab import generation, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.io import srgb_encode
    from PIL import Image
    generation.use(gen)
    rawio.SOURCE_STOPS = stops
    if os.environ.get("MAXS"):
        rawio.MAX_STOPS = float(os.environ["MAXS"])
    ex = rawio.exif_exposure(f)
    rgb = rawio.load_raw(f, max_size=W)
    m, _ = rawio.film_metering(rgb, rawio.clip_luminance(f), rng=np.random.default_rng(zlib.crc32(Path(f).stem.encode())),
                               t_original=ex[0] if ex else None)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))
    img = srgb_encode(np.clip(eng.render(m), 0, 1))
    p = Path(out) / f"{Path(f).stem}_{gen}_{stops:+g}{'_m'+os.environ['MAXS'] if os.environ.get('MAXS') else ''}.jpg"
    Image.fromarray((img * 255).astype(np.uint8)).save(p, quality=92)
    return str(p)

if __name__ == "__main__":
    import multiprocessing as mp
    gens, stops, out = sys.argv[1].split(","), [float(s) for s in sys.argv[2].split(",")], sys.argv[3]
    W = int(os.environ.get("W", "1400"))
    os.makedirs(out, exist_ok=True)
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as p:
        for r in p.imap(job, [(g, s, out, f, W) for f in sys.argv[4:] for g in gens for s in stops], chunksize=1):
            print(r, flush=True)
