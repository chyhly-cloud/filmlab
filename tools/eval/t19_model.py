"""T19: 模型在夜景 RAW 上的光晕统计量, 与参考集 (t19_reference.py) 用同一测量代码。
流程同 examples/night.py (EXIF 测光), 长边 1600 像素 (≈ 22.5 µm/像素, 与参考扫描相近), 输出按 8 位 sRGB 量化后再解码测量。
用法: python3 t19_model.py 输出.json 代际[,代际…] RAW…     环境变量 STOPS (rawio.SOURCE_STOPS), CACHE (成片缓存目录)"""
import json, os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]


def job(a):
    gen, f = a
    cache = Path(os.environ.get("CACHE", "/tmp/t19_cache")); cache.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(f).stem}_{gen}_{os.environ.get('STOPS', '0')}"
    c = cache / f"{tag}.png"
    from PIL import Image
    if not c.exists():
        from filmlab import generation, rawio
        from filmlab.engine import Engine, Params
        from filmlab.recipes import recipe
        from filmlab.io import srgb_encode
        generation.use(gen)
        rawio.SOURCE_STOPS = float(os.environ.get("STOPS", "0"))
        ex = rawio.exif_exposure(f)
        rgb = rawio.load_raw(f, max_size=1600)
        m, _ = rawio.film_metering(rgb, rawio.clip_luminance(f), rng=np.random.default_rng(zlib.crc32(Path(f).stem.encode())),
                                   t_original=ex[0] if ex else None)
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))
        Image.fromarray((np.clip(srgb_encode(np.clip(eng.render(m), 0, 1)), 0, 1) * 255 + 0.5).astype(np.uint8)).save(c)
    import t19_halo_stats as hs
    im = np.asarray(Image.open(c).convert("RGB")) / 255.0
    um = 36000.0 / max(im.shape[:2])
    pr = hs.profiles(hs.srgb_to_lin(im), um)
    return gen, Path(f).name, {f"{s[0]}-{s[1]}": (np.nanmedian(np.array(v), 0).tolist() if v else None) for s, v in pr.items()}


if __name__ == "__main__":
    import multiprocessing as mp
    import t19_halo_stats as hs
    dst, gens, files = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as p:
        rows = p.map(job, [(g, f) for g in gens for f in files], chunksize=1)
    res = {}
    for g in gens:
        res[g] = {"per_image": {f: v for gg, f, v in rows if gg == g}, "summary": {}}
        for s in hs.SIZES:
            k = f"{s[0]}-{s[1]}"
            v = [np.array(d[k]) for gg, f, d in rows if gg == g and d[k] is not None]
            if v:
                sm = hs.summarize(v)
                res[g]["summary"][k] = {a: (np.round(x, 4).tolist() if hasattr(x, "tolist") else x) for a, x in sm.items()}
    json.dump(res, open(dst, "w"), indent=1)
    for g in gens:
        print("==", g)
        for k, v in res[g]["summary"].items():
            print(f"  灯芯半径 {k} µm 图数 {v['n']}\n     Δr {v['dr']}\n     Y  {v['Y']}")
