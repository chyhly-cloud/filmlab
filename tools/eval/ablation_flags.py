"""消融实验开关 (只在实验进程内生效, 不改正式代码)。环境变量:
WB65=1   RAW 日光白平衡参考按 D65 (LibRaw 实际约定) 而非 D55
NOMAT=1  去掉 scanner.SATURATION_MATRIX
NOTINT=1 去掉 LAB_TINT (输出严格中性)
BAL0=1   逐帧颜色校正力度 0 (亮度 / 反差的逐帧定标保留)
MASK=signed|signed_pos   500T 染料谱保留带符号的成色剂遮罩项 (mask_variant.py)
PID=1    扫描仪印放密度换算 P = 单位矩阵 (直接用窄带扫描密度)
LED=noritsu   扫描仪 LED 峰值改用 Noritsu 型估计
SUN=1    日光类加入 CIE D40 / D45 (低太阳高度的直射阳光; CIE 日光公式适用于 4000K 以上)
LIGHT=<类别>  光源识别预言机: 整幅只用该类光源 (daylight / tungsten / hps / fluorescent / led / metal_halide)"""
import os


_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:          # 同一进程只装一次 (进程池里一个进程会跑多个任务)
        return
    _INSTALLED = True
    import numpy as np
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
    return "_".join(k for k in ("WB65", "NOMAT", "NOTINT", "BAL0", "MASK", "PID", "LED", "SUN", "LIGHT") if os.environ.get(k)) or "base"
