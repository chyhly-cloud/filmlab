"""T03 目视对照: 完整画面 (真实 | next | next_oblique) 与光源最密集处的 3 倍局部, 对齐到真实扫描并标全图 ΔE00 中位。

配置与验收相同 (实际自动定光、roll=None、grain=0)。被测代码由环境变量 REPO 指定 (默认本仓库), 对比的代际由 GENS 指定 (逗号分隔, 默认 next,next_oblique)。
用法: FILMLAB_DATA=… [REPO=…] python3 t03_sheet.py 输出目录 场景…
"""
import os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(Path(__file__).resolve().parent), str(REPO)]
GENS = tuple(os.environ.get("GENS", "next,next_oblique").split(","))


def render(args):
    gen, sid = args
    from filmlab import generation
    generation.use(gen)
    import oracle_timing as ot, pair_compare as pc
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    from filmlab.io import srgb_encode
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": seed % 36, "grain": 0}))
    return srgb_encode(np.clip(eng.render(m), 0, 1)), m @ np.array([0.2126, 0.7152, 0.0722])


def sheet(out, sid, pool):
    import colour
    import oracle_timing as ot
    from t02_sheet import strip
    from scipy.ndimage import uniform_filter
    from skimage.transform import warp, ProjectiveTransform
    Hm, real = ot.geometry(sid)
    tf = ProjectiveTransform(Hm)
    outs = pool.map(render, [(g, sid) for g in GENS])
    lab = lambda x: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(x, 0, 1)))
    Lr = lab(real)
    tiles, labels = [real], ["真实扫描"]
    for name, (sim, _) in zip(GENS, outs):
        w = warp(sim, tf.inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
        ok = np.all(np.isfinite(w), -1)
        de = np.median(colour.delta_E(lab(np.where(ok[..., None], w, 0))[ok], Lr[ok], method="CIE 2000"))
        tiles.append(np.where(ok[..., None], w, 0)); labels.append(f"{name}  ΔE00 {de:.2f}")
    f1 = os.path.join(out, f"t03_full_{sid}.jpg")
    strip(tiles, labels, 420).save(f1, quality=90)
    Y = warp(outs[0][1], tf.inverse, output_shape=real.shape[:2], order=1, cval=0)
    c = 160
    dens = uniform_filter((Y > 8.0).astype(float), c)
    dens[:c // 2] = dens[-c // 2:] = 0; dens[:, :c // 2] = dens[:, -c // 2:] = 0
    if dens.max() <= 0:
        return [f1]
    y, x = np.unravel_index(np.argmax(dens), dens.shape)
    crops = [t[y - c // 2:y + c // 2, x - c // 2:x + c // 2] for t in tiles]
    f2 = os.path.join(out, f"t03_lamp_{sid}.jpg")
    strip(crops, [l.split("  ")[0] for l in labels], 3 * c).save(f2, quality=90)
    return [f1, f2]


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(2, maxtasksperchild=1) as p:
        for sid in sys.argv[2:]:
            print(*sheet(out, sid, p), flush=True)
