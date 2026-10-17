"""用法: [ONLY=场景,..] [RECIP=(aR,aG,aB)] [SKYMIX=HP1] [NODAY=1] [OUTNPZ=..] pair_stage_trace.py
逐阶段追踪: 9 组配对中, 每个色相区的颜色分离 (相对场景) 在各阶段保留多少。
k = 该阶段 (r-g, b-g) 对数分离 在场景分离方向上的投影 / 场景分离 (减去中性格子的偏移)。"""
import sys, os, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")
from PIL import Image
import pair_compare as pc
from filmlab import scanner as scm
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.rawio import film_metering
from filmlab.io import srgb_encode
from skimage.transform import warp
SS = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/pairs/sillystill"
EP = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/epfl/data"
SITE = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/versions/site/img/"
SCENES = [("cafe", f"{SS}/table_cafe_digital.ARW", 1.5), ("corridor", f"{SS}/digital1.jpeg", 0.5),
          ("building", f"{SS}/digital3.jpeg", 0.5), ("woodwall", f"{SS}/raw-digital.jpeg", 0.5), ("neon", f"{SS}/digital5.jpeg", 0.5),
          ("dusk", f"{EP}/digital/P1480502.RW2", 0.53), ("skyline", f"{EP}/digital/P1480511.RW2", 0.53),
          ("glass", f"{EP}/digital/P1480513.RW2", 0.53), ("campus", f"{EP}/digital/P1480514.RW2", 0.53)]
RECIP = eval(os.environ.get("RECIP", "None")); TIMES = dict(dusk=6, skyline=30, glass=1, campus=2)
ONLY = os.environ.get("ONLY")
if ONLY: SCENES = [s for s in SCENES if s[0] in ONLY.split(",")]
SKYMIX = os.environ.get("SKYMIX")
def skymix_corr(rgb):
    """诊断: 天空像素 = 暮光 (D150×λ^-1) + 钠灯天光 (HP1) 的非负混合 → 按真实光谱求层曝光, 与 D55 spectral 重建之比。"""
    import colour
    from filmlab import lights, spectral, film
    from filmlab.materials import emulsion
    from scipy.optimize import nnls
    W = spectral.WAVELENGTHS; sh = colour.SpectralShape(W[0], W[-1], W[1] - W[0])
    Sx = 10 ** emulsion("kodak_vision3_500t").stock.log_sensitivity
    tw = colour.sd_CIE_illuminant_D_series(colour.temperature.CCT_to_xy_CIE_D(15000)).copy().align(sh).values * (W / 550.0) ** -1
    hp = colour.SDS_ILLUMINANTS[SKYMIX].copy().align(sh).values
    B = np.stack([tw, hp]); C = lights.camera_rgb(B)            # 2×3
    h, w = rgb.shape[:2]; flat = np.clip(rgb.reshape(-1, 3), 1e-6, None)
    coef = np.linalg.lstsq(C.T, flat.T, rcond=None)[0].T          # 无约束最小二乘
    fit = coef @ C; res = np.linalg.norm(flat - fit, axis=1) / np.linalg.norm(flat, axis=1)
    ok = (coef > 0).all(1) & (res < 0.05) & (np.arange(h * w) // w < h * 0.5)   # 上半幅, 两分量为正, 残差 < 5%
    Ht = np.clip(coef, 0, None) @ (B @ Sx)
    Hd = film.exposure_response(emulsion("kodak_vision3_500t").stock, spectral.illuminant("D55"), None, "spectral")(flat)
    corr = np.log10(np.clip(Ht, 1e-12, None) / np.clip(Hd, 1e-12, None))
    corr -= corr[:, 1:2]                                          # 只改层间比, 绿层 (亮度) 不变
    corr[~ok] = 0
    print("天空混合像素比例", ok.mean().round(3), "R 层修正中位", np.median(corr[ok, 0]).round(3) if ok.any() else None, flush=True)
    from scipy.ndimage import gaussian_filter
    c = corr.reshape(h, w, 3)
    return np.stack([gaussian_filter(c[..., k], 2) for k in range(3)], -1)
if os.environ.get("NODAY"):
    from filmlab import lights as _L
    _cw = _L.class_weights
    def _cw2(ill, sigma=0.25, feas=None, sharpness=6.0):
        if feas is not None: feas = feas.copy(); feas[..., 0] = 1e-3     # 夜间: 日光类不可行
        return _cw(ill, sigma, feas, sharpness)
    _L.class_weights = _cw2
SM = scm.SATURATION_MATRIX.copy()
scm.SATURATION_MATRIX = np.eye(3)
STAGES = ["scene", "logH", "dye", "Dn", "DnM", "Dp", "pre", "post", "out"]
out_all = {k: [] for k in STAGES + ["real", "labR", "labS", "name"]}
IDX = {"cafe":0,"corridor":1,"building":2,"woodwall":3,"neon":4,"dusk":5,"skyline":6,"glass":7,"campus":8}
for name, path, bias in SCENES:
    i = IDX[name]
    rgb = pc.load_digital(path, 1400)[0]
    m, _ = film_metering(rgb, rng=np.random.default_rng(i), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": i % 36}))
    rec = {}
    lo, dev = eng._log_exposure, eng._develop
    def dv(x, **kw):
        if x.ndim == 3 and x.shape[:2] == m.shape[:2]:
            if RECIP and TIMES.get(name, 0) > 1:
                # 低照度倒易律失效: 照度低于 "1 s 达到中灰" 的部分按层损失 a_c·log10(照度比) (Kodak 5219: 10 s +1 档 + CC10R)
                ref = np.median(x[..., 1]) + 0.53 * 0.30103
                x = x - np.array(RECIP)[None, None, :] * np.maximum(0.0, np.log10(TIMES[name]) + ref - x[..., 1:2])
            if SKYMIX:
                x = x + skymix_corr(m)
            rec["logH"] = x.copy()
        return dev(x, **kw)
    eng._develop = dv
    sc = eng.scanner; to_rgb, rp = sc.to_rgb, sc._reference_print
    sc.to_rgb = lambda Dn: (rec.__setitem__("Dn", Dn.copy()), to_rgb(Dn))[1]
    sc._reference_print = lambda Dp, k, **kw: (rec.__setitem__("Dp", Dp.copy()), rp(Dp, k, **kw))[1]
    ot = scm._output_tint
    scm._output_tint = lambda x, t: (rec.__setitem__("pre", x.copy()), ot(np.clip(x, 0, None) @ SM.T, t))[1]
    out = eng.render(m)
    scm._output_tint = ot
    h, w = m.shape[:2]
    Dn = rec["Dn"].reshape(h, w, 3); M = sc.M
    st = dict(scene=np.log10(np.clip(m, 1e-5, None)), logH=rec["logH"], dye=eng.state.dye, Dn=-Dn,
              DnM=-(M[0] + M[1] * Dn), Dp=-rec["Dp"].reshape(h, w, 3),
              pre=np.log10(np.clip(rec["pre"].reshape(h, w, 3), 1e-5, None)),
              post=np.log10(np.clip(rec["pre"].reshape(h, w, 3) @ SM.T, 1e-5, None)),
              out=np.log10(np.clip(out, 1e-5, None)))
    real = np.asarray(Image.open(SITE + f"{name}_real.jpg"), float) / 255
    sim8 = srgb_encode(np.clip(out, 0, 1))
    # 对齐: 用与比较页同样的方法求单应性
    from skimage.feature import ORB, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import ProjectiveTransform
    from skimage.exposure import equalize_adapthist
    g = lambda a: equalize_adapthist(a @ np.array([.2126, .7152, .0722]))
    kp, de = [], []
    for im in (sim8, real):
        o = ORB(n_keypoints=3000, fast_threshold=0.03); o.detect_and_extract(g(im)); kp.append(o.keypoints); de.append(o.descriptors)
    mt = match_descriptors(de[0], de[1], cross_check=True, max_ratio=0.85)
    model, inl = ransac((kp[0][mt[:, 0]][:, ::-1], kp[1][mt[:, 1]][:, ::-1]), ProjectiveTransform, min_samples=4, residual_threshold=3, max_trials=5000)
    c = 12; H, W = real.shape[0] // c * c, real.shape[1] // c * c
    def cells(a):
        return np.median(a[:H, :W].reshape(H // c, c, W // c, c, -1).transpose(0, 2, 1, 3, 4).reshape(-1, c * c, a.shape[-1]), 1)
    mask = cells(warp(np.ones((h, w, 1)), model.inverse, output_shape=real.shape[:2], order=0, cval=0))[:, 0] > 0.99
    for k in STAGES:
        out_all[k].append(cells(warp(st[k], model.inverse, output_shape=real.shape[:2], order=1, cval=0))[mask])
    out_all["real"].append(cells(np.log10(np.clip(colour.cctf_decoding(real), 1e-5, None)))[mask])
    lab = lambda x: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(x, 0, 1)))
    out_all["labR"].append(cells(lab(real))[mask])
    out_all["labS"].append(cells(lab(warp(sim8, model.inverse, output_shape=real.shape[:2], order=1, cval=0)))[mask])
    out_all["name"].append(np.full(mask.sum(), i))
    print(name, "内点", inl.sum(), "格子", mask.sum(), flush=True)
np.savez(os.environ.get("OUTNPZ", "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/trace/cells.npz"), **{k: np.concatenate(v) for k, v in out_all.items()})
