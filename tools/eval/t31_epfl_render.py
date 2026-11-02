"""T31: EPFL 配对 8 夜景, 在真实扫描几何里保存: 真实扫描 (线性), 模型成片 (除碳 / 带 remjet), 模型直接曝光与光晕增量, 灯芯掩码。
与 red_band.py 同一渲染流程 (代际 GENERATION, 默认 t30_base)。用法: FILMLAB_DATA=… python3 t31_epfl_render.py 输出目录 场景…"""
import os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools/eval"), str(REPO)]


def job(args):
    out, sid = args
    gen = os.environ.get("GENERATION", "t30_base")
    f = f"{out}/{sid}_{gen}{os.environ.get('TAG', '')}.npz"
    if os.path.exists(f):
        return sid
    import colour, rawpy
    import oracle_timing as ot, pair_compare as pc
    from scipy.ndimage import zoom
    from skimage.transform import warp, ProjectiveTransform
    from filmlab import generation, rawio, materials
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    generation.use(gen)
    Hm, real = ot.geometry(sid)
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    res = {}
    for tag, extra in (("base", {}), ("remjet", {"film": materials.emulsion("kodak_vision3_500t")})):
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": seed % 36, "grain": 0, **extra}))
        tr = {}
        neg = eng.develop_negative(m, trace=tr)
        res[tag] = (np.clip(eng.scan(neg), 0, None), tr)
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    core = zoom(sat.astype(float), (m.shape[0] / sat.shape[0], m.shape[1] / sat.shape[1]), order=1) > 0.25
    tf = ProjectiveTransform(Hm)
    W = lambda a, o=1: warp(a, tf.inverse, output_shape=real.shape[:2], order=o, cval=np.nan).astype(np.float32)
    tr = res["base"][1]
    np.savez_compressed(f, real=colour.models.eotf_sRGB(real).astype(np.float32), out=W(res["base"][0]),
                        out_rj=W(res["remjet"][0]), inp=W(m), direct=W(tr["direct_exposure"]),
                        halo=W(tr["total_exposure"] - tr["scattered_exposure"]), core=W(core.astype(float)) > 0.5)
    return sid


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as p:
        for s in p.imap_unordered(job, [(out, s) for s in sys.argv[2:]]):
            print("done", s, flush=True)
