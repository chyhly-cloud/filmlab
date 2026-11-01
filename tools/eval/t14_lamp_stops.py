"""真实夜景 RAW: 截断光源重建后, 比胶片中灰亮多少档 (按 examples/night.py 的流程)。"""
import sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path[:0] = ["/home/user/filmlab"]
from scipy import ndimage
import rawpy
from filmlab import generation, rawio
generation.use(sys.argv[1])

def job(f):
    ex = rawio.exif_exposure(f)
    lc = rawio.clip_luminance(f)
    rgb = rawio.load_raw(f, max_size=1400)
    m, note = rawio.film_metering(rgb, lc, rng=np.random.default_rng(zlib.crc32(Path(f).stem.encode())),
                                  t_original=ex[0] if ex else None)
    Y = m @ rawio.LUMA
    clip_m = (m / np.maximum(rgb, 1e-9))[rgb > 0.05].mean()   # 胶片单位里的数码截断点
    with rawpy.imread(f) as r:
        sat = rawio.sensor_saturation(r)
    s = ndimage.zoom(sat.astype(float), (Y.shape[0] / sat.shape[0], Y.shape[1] / sat.shape[1]), order=1)[:Y.shape[0], :Y.shape[1]] > 0.5
    lab, n = ndimage.label(s)
    pk = ndimage.maximum(Y, lab, range(1, n + 1)) if n else np.array([np.nan])
    st = np.log2(np.asarray(pk) / 0.184)
    return f"{Path(f).name:14s} L_clip={lc if lc is None else round(lc,1)!s:7s} 截断点=中灰+{np.log2(clip_m/0.184):4.1f}档  截断块{n:4d}  灯峰值 中位 +{np.nanmedian(st):4.1f} / 最大 +{np.nanmax(st):4.1f} 档"

if __name__ == "__main__":
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(4) as p:
        for r in p.map(job, sys.argv[2:]):
            print(r, flush=True)
