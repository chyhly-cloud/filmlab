"""T14 synthetic point lights: lamps of different brightness (stops above mid-grey) on a dark background, through next_lab_halo; measures halation radius and colour.
Usage: [GEN=…] python3 t14_point_source.py out.png"""
import sys, warnings, os
import numpy as np
warnings.filterwarnings("ignore")
sys.path[:0] = ["/home/user/filmlab"]
from filmlab import generation
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.io import srgb_encode
from PIL import Image
GEN = os.environ.get("GEN", "next_lab_halo")
if os.environ.get("PLATE"):
    from filmlab import engine as _e; _e.PLATE_REFLECTANCE = float(os.environ["PLATE"])
generation.use(GEN)
W, Hh = 1400, 934
px = 36000 / W
STOPS = [4, 7, 10, 13, 16]
bg = 0.184 / 2 ** 3.0
img = np.full((Hh, W, 3), bg) * np.array([1.0, 0.9, 0.75])
yy, xx = np.mgrid[:Hh, :W]
cents = []
for i, s in enumerate(STOPS):
    cx, cy = 140 + i * 280, 300
    for rad, cyo in ((1.5, 0), (5, 330)):
        m = np.hypot(yy - cy - cyo, xx - cx) <= rad
        img[m] = 0.184 * 2 ** s * np.array([1.0, 0.85, 0.6])
        cents.append((s, rad, cx, cy + cyo))
eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))
tn = {}
neg = eng.develop_negative(img, trace=tn)
out = np.clip(eng.scan(neg), 0, None)
srgb = srgb_encode(np.clip(out, 0, 1))
Image.fromarray((srgb * 255).astype(np.uint8)).save(sys.argv[1])
Ht = tn["total_exposure"]
bgo = np.median(out[600:620, 50:100], 0)
print(f"{GEN} px={px:.1f}µm  背景成片 {np.round(bgo,3)}")
for s, rad, cx, cy in cents:
    print(f"\n+{s} 档 半径{rad*px:.0f}µm")
    for r in (0, 4, 8, 12, 20, 30, 45, 60):
        o = out[cy, cx + r]; h = Ht[cy, cx + r] / 0.184
        print(f"  r={r*px:5.0f}µm  H/中灰 {np.round(h,3)}  成片 {np.round(o,3)}  R/(R+G+B) {o[0]/o.sum():.2f}")
