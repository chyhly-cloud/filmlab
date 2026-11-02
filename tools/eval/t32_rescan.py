"""从存档的 logH (S1 不变) 重新显影 + 扫描, 用于只改 S2–S4 的候选。核对: t31_base 应与存档 out 一致。
evaluate(gen_setup, sid) → out (数码几何) 与 ΔE00 (全图 / 光源旁, 真实扫描几何, 同 accept_t02)。"""
import numpy as np, sys, os, zlib, warnings
warnings.filterwarnings('ignore')
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parent), str(__import__("pathlib").Path(__file__).resolve().parents[2])]
OLD = os.environ.get("FILMLAB_DATA", "")
os.environ.setdefault("FILMLAB_DATA", OLD)
def render(setup, sid, tag="_t31_base"):
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    setup()
    z = np.load(f"epfl/{sid}{tag}.npz")
    seed = zlib.crc32(sid.encode()) % 1000
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": seed % 36, "grain": 0}))
    eng.px_um = float(z["px_um"])
    logH = z["logH"].astype(float)
    import t32_counterfactuals as cands
    if getattr(cands, "LOGH_HOOK", None) is not None:
        logH = cands.LOGH_HOOK(logH, z)
    st = eng._develop(logH, blur_px=12.0 / eng.px_um)
    eng._agitation(st); eng._grain(st)
    tr = {}
    out = np.clip(eng.scan(st, trace=tr), 0, None)
    return out, tr, z
def de_scores(out, z, sid):
    import colour, oracle_timing as ot
    from scipy.ndimage import distance_transform_edt
    from skimage.transform import warp, ProjectiveTransform
    from filmlab.io import srgb_encode
    Hm, real = ot.geometry(sid) if sid != "R1-17-18" else (z["Hm"], None)
    sim = srgb_encode(np.clip(out, 0, 1))
    tf = ProjectiveTransform(z["Hm"])
    w = warp(sim, tf.inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
    ok = np.all(np.isfinite(w), -1)
    lab = lambda x: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(x, 0, 1)))
    de = colour.delta_E(lab(np.where(ok[..., None], w, 0)), lab(real), method="CIE 2000")
    core = warp(z["core"].astype(float), tf.inverse, output_shape=real.shape[:2], order=1, cval=0) > 0.5
    d = distance_transform_edt(~core) * 36000.0 / max(real.shape[:2])
    near = ok & (d > 0) & (d <= 300)
    return float(np.median(de[ok])), (float(np.median(de[near])) if near.sum() > 100 else np.nan)
def use(gen):
    def f():
        from filmlab import generation
        generation.use(gen)
    return f
if __name__ == "__main__":
    out, tr, z = render(use("t31_base"), "R1-03-4")
    print("核对 t31_base 重扫 vs 存档:", np.abs(out - z["out"]).max())
    print(de_scores(out, z, "R1-03-4"))
