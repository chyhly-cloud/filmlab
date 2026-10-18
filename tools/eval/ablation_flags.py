"""消融实验开关 (只在实验进程内生效, 不改正式代码)。环境变量:
WB65=1   RAW 日光白平衡参考按 D65 (LibRaw 实际约定) 而非 D55
NOMAT=1  去掉 scanner.SATURATION_MATRIX
NOTINT=1 去掉 LAB_TINT (输出严格中性)
BAL0=1   逐帧颜色校正力度 0 (亮度 / 反差的逐帧定标保留)
MASK=signed|signed_pos   500T 染料谱保留带符号的成色剂遮罩项 (mask_variant.py)
PID=1    扫描仪印放密度换算 P = 单位矩阵 (直接用窄带扫描密度)
LED=noritsu   扫描仪 LED 峰值改用 Noritsu 型估计
SUN=1    日光类加入 CIE D40 / D45 (低太阳高度的直射阳光; CIE 日光公式适用于 4000K 以上)
TAU=<值>  ECN-2 胶片在 C-41 中的有效显影程度 τ (冻结区间 [1.0, 1.3] 内的敏感性; 现行 1/0.85)
DYESHIFT=none|cyan  C-41 中染料光谱平移: 不平移 / 只平移青色 (品红、黄的平移无实测数据)
CORR=a,b   为所有引擎打开 Params.corrections 中的计算修正
HALO=<k>  光晕强度 × k (片基背面反射能力的倍数; 画面与定光中灰同时缩放)
LIGHT=<类别>  光源识别预言机: 整幅只用该类光源 (daylight / tungsten / hps / fluorescent / led / metal_halide)"""
import os


_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:          # 同一进程只装一次 (进程池里一个进程会跑多个任务)
        return
    _INSTALLED = True
    import numpy as np
    if os.environ.get("GENERATION"):
        from filmlab import generation
        generation.use(os.environ["GENERATION"])
    from dataclasses import replace
    from functools import lru_cache
    from filmlab import scanner as scm
    if os.environ.get("WB65"):
        from filmlab import lights
        lights.WB_REFERENCE = "D65"
    if os.environ.get("NOMAT"):
        scm.SATURATION_MATRIX = np.eye(3)
    if os.environ.get("NOTINT"):
        for k, m in list(scm.SCANNERS.items()):
            scm.SCANNERS[k] = replace(m, output_tint=((0.0, 0.0, 0.0), (100.0, 0.0, 0.0)))
    if os.environ.get("MASK"):
        import mask_variant
        mask_variant.install(os.environ["MASK"])
    if os.environ.get("PID"):
        from filmlab.scanning import conversion
        conversion.printing_matrix = lambda *a, **k: np.eye(3)
    if os.environ.get("LED") == "noritsu":
        f = scm.SCANNERS["frontier"]; n = scm.SCANNERS["noritsu"]
        scm.SCANNERS["frontier"] = replace(f, led_peaks=n.led_peaks, led_widths=n.led_widths)
    if os.environ.get("SUN"):
        import colour
        from filmlab import lights
        orig = lights.classes.__wrapped__

        def daylight_d(T):
            xy = colour.temperature.CCT_to_xy_CIE_D(T * 1.4388 / 1.4380)
            v = colour.sd_CIE_illuminant_D_series(xy).align(lights.SHAPE).values
            return np.clip(v, 0, None) / v.max()

        def classes():
            out = list(orig())
            d = out[0]
            out[0] = replace(d, members=tuple((f"D{t // 100}", daylight_d(t)) for t in (4000, 4500)) + d.members)
            return tuple(out)
        lights.classes = lru_cache(maxsize=None)(classes)
        lights.class_gamuts.cache_clear()
        # 只改变光源识别 (似然、色域); 曝光用的具体光源仍从原成员中按种子抽取, 避免随机抽样序列改变
        def pick_members(seed):
            rng = np.random.default_rng(seed)
            return {c.key: c.members[rng.integers(len(c.members))] for c in orig()}
        lights.pick_members = pick_members
    if os.environ.get("TAU"):
        from filmlab import chemistry, engine
        run0, tau = chemistry.run, float(os.environ["TAU"])

        def run(e, process, logH, blur_px=0.0, tau_factor=1.0):
            if e.native_process == "ECN-2" and any(isinstance(st, chemistry.Develop) and st.developer == "C-41 CD"
                                                   for st in process.steps):
                tau_factor = tau_factor * tau * chemistry.DEVELOPERS["ECN-2 CD"].activity
            return run0(e, process, logH, blur_px, tau_factor)
        chemistry.run = run
    if os.environ.get("DYESHIFT"):
        from filmlab import chemistry
        ds0, mode = chemistry.dye_spectra, os.environ["DYESHIFT"]

        def dye_spectra(dd, wl, native, actual, shift_nm, broaden, shift_other_nm=None):
            if mode == "none":
                return dd
            return ds0(dd, wl, native, actual, shift_nm, broaden, 0.0)
        chemistry.dye_spectra = dye_spectra
    if os.environ.get("CORR"):
        from filmlab import engine as _eng
        init_e = _eng.Engine.__init__
        corr = tuple(os.environ["CORR"].split(","))

        def init_c(self, p):
            init_e(self, replace(p, corrections=tuple(set(p.corrections) | set(corr))))
        _eng.Engine.__init__ = init_c
    if os.environ.get("HALO"):
        from filmlab import film as _film
        hr0, kh = _film.halation_response, float(os.environ["HALO"])

        def halation_response(stock, light, scene_filter, v, back_reflect, method="spectral"):
            return hr0(stock, light, scene_filter, v, back_reflect * kh, method)
        _film.halation_response = halation_response
    if os.environ.get("LIGHT"):
        from filmlab import lights
        k = [c.key for c in lights.classes()].index(os.environ["LIGHT"])

        def one_hot(ill, *a, **kw):
            w = np.zeros(ill.shape[:-1] + (len(lights.classes()),)); w[..., k] = 1.0
            return w
        lights.class_weights = one_hot
    if os.environ.get("BAL0"):
        init = scm.Scanner.__init__

        def init0(self, *a, **k):
            init(self, *a, **k)
            self.strength = 0.0
        scm.Scanner.__init__ = init0


def tag():
    return "_".join(k for k in ("WB65", "NOMAT", "NOTINT", "BAL0", "MASK", "PID", "LED", "SUN", "LIGHT", "TAU", "DYESHIFT", "CORR", "HALO") if os.environ.get(k)) or "base"
