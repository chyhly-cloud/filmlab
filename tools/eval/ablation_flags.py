"""消融实验开关 (只在实验进程内生效, 不改正式代码)。环境变量:
WB65=1   RAW 日光白平衡参考按 D65 (LibRaw 实际约定) 而非 D55
NOMAT=1  去掉 scanner.SATURATION_MATRIX
NOTINT=1 去掉 LAB_TINT (输出严格中性)
BAL0=1   逐帧颜色校正力度 0 (亮度 / 反差的逐帧定标保留)
MASK=signed|signed_pos   500T 染料谱保留带符号的成色剂遮罩项 (mask_variant.py)
PID=1    扫描仪印放密度换算 P = 单位矩阵 (直接用窄带扫描密度)
LED=noritsu   扫描仪 LED 峰值改用 Noritsu 型估计"""
import os


def install():
    import numpy as np
    from dataclasses import replace
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
    if os.environ.get("BAL0"):
        init = scm.Scanner.__init__

        def init0(self, *a, **k):
            init(self, *a, **k)
            self.strength = 0.0
        scm.Scanner.__init__ = init0


def tag():
    return "_".join(k for k in ("WB65", "NOMAT", "NOTINT", "BAL0", "MASK", "PID", "LED") if os.environ.get(k)) or "base"
