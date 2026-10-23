"""光晕强度 / 半径的配对检验: 亮光源外侧暗区的红色边缘 (真实扫描 vs 模拟 vs 关闭光晕的模拟)。

只看暗区 (数码输入里该处场景本身很暗), 按到亮光源 (数码输入很亮) 的距离 (胶片上的 µm) 分档,
取 log(R/G)、log(B/G) 与亮度的中位数, 减去远处 (1000–1500 µm) 的水平 → 光源周围"多出来的红"。
历史模式的有/无光晕分别自动定光，其差值包含定光变化，不是孤立输运光晕。
固定定光的因果比较见 halation_chain_audit.py；本工具保留旧模式用于限定追溯。
真实扫描的值与模拟同法计算; 不拟合任何参数。

用法: python3 halo_edge.py 输出.json [场景…]
"""
import json, os, sys, zlib, warnings, hashlib
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'tools/eval'), str(REPO)]

BINS = np.array([0, 50, 100, 150, 200, 250, 300, 400, 500, 650, 800, 1000, 1500])
SRC = float(os.environ.get("HALO_SRC", "8.0"))     # 光源: 数码输入亮度 > 中灰的 8 倍 (+3 档)
# HALO_KNOWN=1: 只用数码 RAW 未截断的光源 (亮度是测量值, 不是重建先验); 截断区 1500 µm 内全部排除。
# 截断点只比中灰高约 2.8 档, 所以此时 HALO_SRC 需要降到约 3。
DARK = float(os.environ.get("HALO_DARK", "0.1"))    # 暗区: 数码输入亮度 < 中灰的约一半 (光源旁的数码镜头眩光使更严的阈值取不到近处)
FRAME_MM = 36.0
SCENES = ["R1-00-1", "R1-01-2", "R1-02-3", "R1-03-4", "R1-04-5", "R1-05-6", "R1-06-7", "R1-07-8",
          "R1-08-9", "R1-09-10", "R1-10-11", "R1-13-14"]


def linear_input(array, encoding):
    """Explicit domain contract: rendered RGB is linear; scan files are encoded."""
    if encoding == 'linear_srgb':
        return np.asarray(array, float)
    if encoding == 'srgb':
        from colour.models import eotf_sRGB
        return eotf_sRGB(array)
    raise ValueError('Encoding must be linear_srgb or srgb')


def profile(lin, d_um, dark):
    lin = np.clip(lin, 1e-4, None)
    lr, lb = np.log(lin[..., 0] / lin[..., 1]), np.log(lin[..., 2] / lin[..., 1])
    Y = np.log10(lin @ [0.2126, 0.7152, 0.0722])
    out = []
    for a, b in zip(BINS[:-1], BINS[1:]):
        m = dark & (d_um >= a) & (d_um < b)
        out.append([int(m.sum())] + ([float(np.median(v[m])) for v in (lr, lb, Y)] if m.sum() > 30 else [np.nan] * 3))
    return np.array(out)


def job(sid):
    import ablation_flags as af; af.install()
    import colour
    import oracle_timing as ot
    import pair_compare as pc
    from scipy.ndimage import distance_transform_edt
    from skimage.transform import warp, ProjectiveTransform
    from filmlab import generation
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    generation.use(os.environ.get("GENERATION", "next"))
    Hm, real = ot.geometry(sid)
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    tf = ProjectiveTransform(Hm)
    wp = lambda a: warp(a, tf.inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
    sims = {}; timings = {}
    for name, halo in (("sim", True), ("nohalo", False)):
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": seed % 36}))
        if not halo:
            eng.halo = None
        sims[name] = linear_input(np.clip(wp(np.clip(eng.render(m), 0, 1)), 0, 1), 'linear_srgb')
        t = eng.scanner.timing
        timings[name] = {'K':t.K.tolist(), 's':t.s.tolist(),
                         'M':None if t.M is None else [a.tolist() for a in t.M],
                         'x_knee':t.x_knee, 'x_white':t.x_white}
    v = eng.v
    Yin = wp(m @ np.array([0.2126, 0.7152, 0.0722]))
    ok = np.isfinite(Yin)
    src = ok & (Yin > SRC)
    um_px = FRAME_MM * 1000.0 / max(real.shape[:2])
    d_um = distance_transform_edt(~src) * um_px
    dark = ok & (Yin < DARK)
    if os.environ.get("HALO_KNOWN"):
        from filmlab.rawio import load_raw
        clipped = load_raw(rawp, max_size=1400, highlights="clip")
        clipped = clipped[0] if isinstance(clipped, tuple) else clipped
        recon = wp((np.abs(rgb - clipped).max(-1) > 1e-4 * max(rgb.max(), 1e-6)).astype(float)) > 0.01
        near = distance_transform_edt(~recon) * um_px < 1500.0
        src &= ~near; dark &= ~near
        d_um = distance_transform_edt(~src) * um_px
    out = {"n_src": int(src.sum()), "um_px": um_px, "base_um": v.base_um, "forward": v.forward,
           "diffuse": v.diffuse_fraction, "real": profile(linear_input(real, 'srgb'), d_um, dark).tolist(),
           'contract': {'version':2, 'simulation_input':'linear_srgb', 'scan_input':'srgb',
                        'profile_domain':'linear_srgb', 'primaries':'sRGB', 'white':'D65',
                        'simulation_dtype':str(sims['sim'].dtype), 'scan_storage_bits':8,
                        'timing':'historical independently automatic per variant; NOT a causal halo ablation',
                        'generation':generation.active(), 'seed':seed, 'grain':eng.p.grain,
                        'geometry':Hm.tolist(), 'timings':timings,
                        'halo_scale':float(os.environ.get('HALO','1')),
                        'known_source_only':bool(os.environ.get('HALO_KNOWN')),
                        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
           'legacy_double_decoded':{}}
    for k, s in sims.items():
        out[k] = profile(s, d_um, dark & np.all(np.isfinite(s), -1)).tolist()
        # Reproduce the old bug on these SAME pixels, without another render.
        out['legacy_double_decoded'][k] = profile(linear_input(s, 'srgb'), d_um,
                                                 dark & np.all(np.isfinite(s), -1)).tolist()
    return sid, out


def excess(p):
    p = np.asarray(p, float)
    return p[:, 1:] - p[-1, 1:]


def report(res):
    if any(r.get('contract',{}).get('version') != 2 for r in res.values()):
        raise ValueError('Legacy halo_edge cache has no corrected domain contract; regenerate it with explicit scenes')
    print("光源外侧暗区的 log(R/G) 相对远处 (1000–1500 µm) 的增量 (×100); 列 = 距离档上沿 µm")
    print(f"{'':22s}" + "".join(f"{b:>6d}" for b in BINS[1:-1]))
    agg = {k: [] for k in ("real", "sim", "nohalo")}
    for sid, r in res.items():
        if r["n_src"] < 50:
            continue
        for k in agg:
            e = excess(r[k])[:-1, 0] * 100
            agg[k].append(e)
            print(f"{sid:9s} {k:7s} {'':5s}" + "".join(f"{x:6.1f}" for x in e))
    print()
    for k, v in agg.items():
        print(f"{'中位':9s} {k:7s} {'':5s}" + "".join(f"{x:6.1f}" for x in np.nanmedian(v, 0)))
    h_sim = np.nanmedian(np.array(agg["sim"]) - np.array(agg["nohalo"]), 0)
    h_real = np.nanmedian(np.array(agg["real"]) - np.array(agg["nohalo"]), 0)
    print(f"{'光晕部分':9s} {'模拟':7s} {'':5s}" + "".join(f"{x:6.1f}" for x in h_sim))
    print(f"{'光晕部分':9s} {'真实':7s} {'':5s}" + "".join(f"{x:6.1f}" for x in h_real))


if __name__ == "__main__":
    out = sys.argv[1]
    if os.path.exists(out) and len(sys.argv) == 2:
        report(json.load(open(out))); sys.exit()
    run = sys.argv[2:] or SCENES
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4"))) as p:
        res = dict(p.imap(job, run))
    json.dump(res, open(out, "w"), indent=1)
    report(res)
