"""三联对比图: 数码原片 | 本系统 CineStill 800T | 真实 800T 扫描 (按特征点对齐到扫描)。
用法: python3 tools/eval/show_pairs.py SillyStill配对目录 [输出目录]"""
import sys, os, numpy as np, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import pair_compare as pc
from PIL import Image, ImageDraw, ImageFont
from filmlab.io import srgb_encode
from skimage.transform import warp
D = sys.argv[1]   # SillyStill 配对目录 (table_cafe_*, film*/digital*)
OUT = sys.argv[2] if len(sys.argv) > 2 else "out/showcase_pairs"
os.makedirs(OUT, exist_ok=True)
PAIRS = [("table_cafe_film.JPG", "table_cafe_digital.ARW", "咖啡店 · 窗外直射阳光 (RAW)"),
         ("film1.jpeg", "digital1.jpeg", "夜晚走廊 · 冷白 LED + 暖色窗"),
         ("film3.jpeg", "digital3.jpeg", "阴天建筑"),
         ("raw-film.jpeg", "raw-digital.jpeg", "阴影中的木墙与滑板车"),
         ("film5.jpeg", "digital5.jpeg", "室内霓虹灯管")]
font = None
for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(f): font = ImageFont.truetype(f, 26); break
font = font or ImageFont.load_default()
def model_of(moving, fixed):
    from skimage.feature import ORB, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import ProjectiveTransform
    from skimage.exposure import equalize_adapthist
    kp, de = [], []
    for g in (equalize_adapthist(pc._gray(moving)), equalize_adapthist(pc._gray(fixed))):
        o = ORB(n_keypoints=3000, fast_threshold=0.03); o.detect_and_extract(g); kp.append(o.keypoints); de.append(o.descriptors)
    m = match_descriptors(de[0], de[1], cross_check=True, max_ratio=0.85)
    mod, _ = ransac((kp[0][m[:, 0]][:, ::-1], kp[1][m[:, 1]][:, ::-1]), ProjectiveTransform, min_samples=4, residual_threshold=3, max_trials=5000)
    return mod
for i, (fs, ds, title) in enumerate(PAIRS):
    real = Image.open(f"{D}/{fs}").convert("RGB"); real.thumbnail((1200, 1200))
    real = np.asarray(real).astype(float) / 255
    rgb, is_raw = pc.load_digital(f"{D}/{ds}", max_size=1600)
    dig = srgb_encode(np.clip(rgb / max(np.percentile(rgb @ [0.2126, 0.7152, 0.0722], 99), 1e-6) * 0.9, 0, 1)) if is_raw else np.clip(pc.colour.cctf_encoding(rgb), 0, 1)
    sim = srgb_encode(pc.render(rgb, 0.5, seed=i)[0])
    mod = model_of(sim, real)
    W = lambda a: np.nan_to_num(warp(a, mod.inverse, output_shape=real.shape[:2], order=1, cval=0))
    panels = [W(dig), W(sim), real]
    h, w = real.shape[:2]; pad = 46
    canvas = Image.new("RGB", (w * 3 + 20, h + pad), (20, 20, 20)); d = ImageDraw.Draw(canvas)
    for k, (p, lab) in enumerate(zip(panels, ["数码原片", "本系统 · CineStill 800T 模拟", "真实 CineStill 800T 冲印店扫描"])):
        canvas.paste(Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8)), (k * (w + 10), pad))
        d.text((k * (w + 10) + 8, 8), lab, fill=(235, 235, 235), font=font)
    d.text((w * 3 + 20 - 8 - d.textlength(title, font=font), 8), title, fill=(160, 160, 160), font=font) if hasattr(d, "textlength") else None
    canvas.save(f"{OUT}/pair_{i+1}.jpg", quality=90); print("saved", i + 1, title)
