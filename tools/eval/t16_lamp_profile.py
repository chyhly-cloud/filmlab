"""T16 诊断: 沿一盏截断灯的水平剖面, 打印 截断判定 / RAW / 重建 / 输入(档) / 三层曝光(档) / 成片。
用法: python3 t16_lamp_profile.py 代际 RAW [cy cx] [输出图.png]   (cy cx 为下采样网格坐标; 缺省取最大截断块中心)"""
import sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path[:0] = [str(Path(__file__).resolve().parents[2])]
import rawpy
from scipy import ndimage
from filmlab import generation, rawio
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.io import srgb_encode

gen, f = sys.argv[1], sys.argv[2]
generation.use(gen)
ex = rawio.exif_exposure(f)
rgb_clip = rawio.load_raw(f, max_size=1400, highlights="clip")
rgb = rawio.load_raw(f, max_size=1400)
m, _ = rawio.film_metering(rgb, rawio.clip_luminance(f), rng=np.random.default_rng(zlib.crc32(Path(f).stem.encode())),
                           t_original=ex[0] if ex else None)
eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))
tn = {}
out = np.clip(eng.scan(eng.develop_negative(m, trace=tn)), 0, None)
with rawpy.imread(f) as r:
    sat = rawio.sensor_saturation(r)
s = ndimage.zoom(sat.astype(float), (m.shape[0] / sat.shape[0], m.shape[1] / sat.shape[1]), order=1)[:m.shape[0], :m.shape[1]]
if len(sys.argv) > 4 and sys.argv[3].isdigit():
    cy, cx = int(sys.argv[3]), int(sys.argv[4])
else:
    lab, n = ndimage.label(s > 0.25)
    k = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    cy, cx = (int(v) for v in ndimage.center_of_mass(lab == k))
H = tn["total_exposure"]
print(f"{gen}  中心 ({cy},{cx})  px {36000 / max(m.shape[:2]):.1f} µm")
for dx in range(0, 75, 3):
    x = cx + dx
    if x >= m.shape[1]:
        break
    print(f"dx={dx:3d} 截断={s[cy, x]:.2f} RAW {np.round(rgb_clip[cy, x], 2)} 重建 {np.round(rgb[cy, x], 2)} "
          f"H档 {np.round(np.log2(np.maximum(H[cy, x], 1e-9) / 1.2), 1)} 成片 {np.round(out[cy, x], 2)}")
if sys.argv[-1].endswith(".png"):
    from PIL import Image
    img = srgb_encode(np.clip(out, 0, 1))
    Image.fromarray((img * 255).astype(np.uint8)).save(sys.argv[-1])
