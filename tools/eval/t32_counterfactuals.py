import os
import numpy as np
def k5219():
    from filmlab import generation, stocks, materials, engine, lights
    from filmlab.scanning import conversion, paper
    generation.use("t31_base")
    stocks.REVISIONS.clear(); stocks.REVISIONS.update({"kodak_vision3_500t": "kodak_raw"})
    stocks.load_stock.cache_clear(); materials.emulsion.cache_clear()
    engine._material_calibration.cache_clear(); engine._reference_timing.cache_clear()
    conversion._PROFILES.clear(); paper._PAPERS.clear()
def nodir():
    from filmlab import generation, chemistry
    generation.use("t31_base")
    if not hasattr(chemistry, "_A0"):
        chemistry._A0 = chemistry._dir_matrix
    chemistry._dir_matrix = lambda a, neg=True, _f=chemistry._A0: np.diag(np.diag(_f(a, neg)))
def base():
    from filmlab import generation, chemistry
    generation.use("t31_base")
    if hasattr(chemistry, "_A0"):
        chemistry._dir_matrix = chemistry._A0

RAW5219 = os.environ.get("KODAK5219_RAW", "raw5219")
def redcurve():
    """S2 反事实: 红层曲线按 Kodak 5219 Status M (ECN-2) 与模型 Status M 之比逐点缩放 (只改红层的形状 / 上限, 数据范围外沿用端点比)。"""
    base()
    from filmlab import materials, chemistry
    from filmlab.stocks import WAVELENGTHS as W
    import numpy as np
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    def curve(n, zero=False):
        a = np.loadtxt(f"{RAW5219}/{n}.csv", delimiter=","); a = a[np.argsort(a[:, 0])]
        return np.interp(W, a[:, 0], a[:, 1], left=0 if zero else None, right=0 if zero else None)
    F = np.stack([curve(f"statusM_{c}", True) for c in "rgb"], 1); F /= F.sum(0)
    h = np.loadtxt(f"{RAW5219}/density_curve_r.csv", delimiter=","); h = h[np.argsort(h[:, 0])]
    eng = Engine(Params(**{**recipe("cinestill_800t"), "process": "ECN-2", "roll": None, "grain": 0, "timing": "frame"}))
    e = eng.e
    x = e.log_exposure
    st = chemistry.run(e, eng.proc, np.repeat(x[:, None], 3, 1), 0.0, tau_factor=1.0)
    D = -np.log10(np.clip(eng._transmittance(st) @ F, 1e-9, None))
    dmin_m = D[0, 0]
    kd = np.interp(x, h[:, 0], h[:, 1])
    ratio = np.clip((kd - h[0, 1]) / np.maximum(D[:, 0] - dmin_m, 1e-3), 0.5, 1.5)
    ratio = np.where(x < h[0, 0] + 0.3, 1.0, np.where(x > h[-1, 0], ratio[np.argmin(abs(x - h[-1, 0]))], ratio))
    orig = materials.emulsion
    def em(key, _o=orig, _r=ratio):
        ee = _o(key)
        if key != "kodak_vision3_500t": return ee
        S = ee.S_nom.copy(); S[:, 0] = S[:, 0] * _r
        return ee.with_(S_nom=S)
    em.cache_clear = orig.cache_clear
    materials.emulsion = em
    import filmlab.recipes as rc, filmlab.engine as en
    rc.emulsion = em; en.emulsion = em
def scanbw():
    """S4a 反事实: 扫描绿通道 σ 16 → 35 nm (扫描仪通道光谱未测定)。"""
    base()
    from filmlab import scanner
    from filmlab.scanning import conversion
    m = scanner.SCANNERS["frontier"]
    from dataclasses import replace
    scanner.SCANNERS["frontier"] = replace(m, led_widths=(m.led_widths[0], 35.0, m.led_widths[2]))
    conversion._PROFILES.clear(); conversion._PRINTING.clear()
def papci():
    """S4d 反事实: 数字相纸按通道独立 (每通道 = 中性灰在该相纸密度下的输出), 去掉相纸染料串扰与 D50→sRGB 的跨通道作用。"""
    base()
    from filmlab.scanning import paper
    def rp(self, Dp, output_black=paper.OUTPUT_BLACK, chunk=1 << 16):
        k = self.k
        return (output_black + (1 - output_black) * (10.0 ** -Dp - k) / (1 - k)).reshape(-1, 3)
    paper.DigitalPaper.reference_print = rp
LOGH_HOOK = None
def s1g():
    """S1 反事实: 红色物体 (胶片曝光 logH_R − logH_G > 0.3) 的绿层曝光 ×1.5 (渐入), 其余不变。"""
    base()
    global LOGH_HOOK
    import numpy as np
    G0 = np.array([-0.83960607, -0.84922118, -0.8927258])
    def hook(lh, z=None):
        v = lh - G0
        w = np.clip((v[..., 0] - v[..., 1] - 0.2) / 0.2, 0, 1) * (v[..., 0] > v[..., 2])
        lh = lh.copy(); lh[..., 1] += np.log10(1.5) * w
        return lh
    LOGH_HOOK = hook

def s1b():
    """S1 反事实: 红色物体 (胶片 logH_R − logH_G > 0.3) 的蓝层曝光不高于相机 B 所示 (log10(B_raw/0.184) + 中灰), 其余不变。"""
    base()
    global LOGH_HOOK
    import numpy as np
    G0 = np.array([-0.83960607, -0.84922118, -0.8927258])
    def hook(lh, z):
        v = lh - G0
        w = np.clip((v[..., 0] - v[..., 1] - 0.2) / 0.2, 0, 1)
        cam = np.log10(np.clip(z["inp"][..., 2].astype(float), 1e-6, None) / 0.184) + G0[2]
        lh = lh.copy(); lh[..., 2] = np.where(w > 0, (1 - w) * lh[..., 2] + w * np.minimum(lh[..., 2], cam), lh[..., 2])
        return lh
    LOGH_HOOK = hook
