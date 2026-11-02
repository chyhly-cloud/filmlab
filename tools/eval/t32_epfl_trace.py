"""T32: EPFL 配对帧的逐级中间量 (数码输入几何, 不重采样模型)。真实扫描反向配准到数码几何。
保存: inp (测光后线性 RGB), logH (三层总曝光, 含散射 / 光晕), halo_frac (光晕增量 / 总曝光), dye (三层染料量),
Dn (S4b 归一化底片密度), out (成片线性 sRGB), real (真实扫描线性 sRGB, NaN = 画外), core (传感器饱和灯芯), 定光参数。
渲染流程与 t31_epfl_render.py / accept_t02.scene_job 相同 (timing=frame, roll=None, grain=0)。
用法: FILMLAB_DATA=… GENERATION=t31_base [EV=档 TAG=后缀] python3 t32_epfl_trace.py 输出目录 场景…
R1-17-18 (灰阶楔) 用 tools/eval/epfl_wedge_geometry.json 的手工几何 (只有几何, 无亮度拟合)。"""
import json, os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools/eval"), str(REPO)]


def wedge_geometry(shape):
    """手工几何 (epfl_wedge_geometry.json) → 本次数码尺寸的单应性与真实扫描。"""
    import oracle_timing as ot
    from PIL import Image
    g = json.load(open(REPO / "tools/eval/epfl_wedge_geometry.json"))
    s = shape[1] / g["source_size"][0]
    S = np.diag([1 / s, 1 / s, 1.0])
    _, scan, _ = ot.sources("R1-17-18")
    real = Image.open(scan).convert("RGB"); real.thumbnail((1600, 1600)); real = np.asarray(real, float) / 255
    assert real.shape[1] == g["scan_size"][0], real.shape
    return np.asarray(g["H"]) @ S, real


def job(args):
    out, sid = args
    gen = os.environ.get("GENERATION", "t31_base")
    f = f"{out}/{sid}_{gen}{os.environ.get('TAG', '')}.npz"
    if os.path.exists(f):
        return sid
    import colour, rawpy
    import oracle_timing as ot, pair_compare as pc
    from scipy.ndimage import zoom
    from skimage.transform import warp, ProjectiveTransform
    from filmlab import generation, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    generation.use(gen)
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    Hm, real = wedge_geometry(rgb.shape) if sid == "R1-17-18" else ot.geometry(sid)
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    m = m * 2.0 ** float(os.environ.get("EV", "0"))     # 诊断用的整体曝光偏移 (档; 图卡帧的胶片曝光未知), 默认 0
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": seed % 36, "grain": 0}))
    tr = {}
    neg = eng.develop_negative(m, trace=tr)
    sc = {}
    o = np.clip(eng.scan(neg, trace=sc), 0, None)
    t = sc["timing"]
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    core = np.zeros(m.shape[:2], bool) if sat is None else \
        zoom(sat.astype(float), (m.shape[0] / sat.shape[0], m.shape[1] / sat.shape[1]), order=1)[:m.shape[0], :m.shape[1]] > 0.25
    # 真实扫描 → 数码几何 (Hm: 数码 → 扫描)
    rl = warp(colour.models.eotf_sRGB(real), ProjectiveTransform(Hm), output_shape=m.shape[:2], order=1, cval=np.nan)
    H = tr["total_exposure"]
    hf = (H - tr["scattered_exposure"]) / np.maximum(H, 1e-9)
    f16 = lambda a: np.asarray(a, np.float16)
    np.savez_compressed(f, inp=f16(m), logH=neg.log_exposure.astype(np.float32), halo_frac=f16(hf),
                        dye=neg.state.dye.astype(np.float32), Dn=sc["Dn"].astype(np.float32), out=o.astype(np.float32),
                        real=rl.astype(np.float32), core=core, Hm=Hm, px_um=eng.px_um,
                        K=np.asarray(t.K, float), s=np.asarray(t.s, float),
                        M_off=np.asarray(t.M[0] if t.M is not None else np.zeros(3), float),
                        M_r=np.asarray(t.M[1] if t.M is not None else np.ones(3), float))
    return sid


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as p:
        for s in p.imap_unordered(job, [(out, s) for s in sys.argv[2:]]):
            print("done", s, flush=True)
