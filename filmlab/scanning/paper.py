"""S4d 数字相纸: 负片密度 → 正像 (输出参照的线性 RGB)。

当前模型以富士 Crystal Archive 相纸印放作为扫描输出的代理 (文件黑点另作映射)。这是模型假设,
不是已测定的 Frontier 文件响应; 相纸材料曲线不能单独确定扫描/显示输出 LUT。
影调用相纸实测特性曲线的形状 (三层曲线对齐后平均, 中性);
颜色按真实相纸染料光谱计算 (相纸三种染料都不纯, 串扰让印放比"三通道独立"低饱和)。
证据边界见 docs/EPFL_HIGHLIGHT_VALIDATION.md。
"""
import numpy as np

from ..materials import emulsion

# 数字相纸最大密度对应的输出亮度 (冲印店扫描的黑不是 0; 在调参 / 验证两半真实扫描上都改善)
OUTPUT_BLACK = 0.002


def neutral_paper_curve(paper):
    """相纸三层特性曲线的平均影调。三层的感光度位置不同, 曲线在横轴上错开 ——
    先按各层中间密度处对齐再平均, 否则平均会把曲线抹宽, 反差被低估。"""
    x, S = paper.log_exposure, paper.S_nom - paper.S_nom.min(0)
    mids = [x[np.argmax(S[:, c] >= 0.5 * S[:, c].max())] for c in range(3)]
    ref = mids[1]
    aligned = np.stack([np.interp(x + (mids[c] - ref), x, S[:, c]) for c in range(3)], 1)
    c = aligned.mean(1)
    return c - c.min()


class DigitalPaper:
    def __init__(self, paper_key, paper_white=False):
        self.paper_white = paper_white          # True: 以相纸白 (而不是 D50 完美白) 为白点做色适应 → 相纸白 = 输出白
        self.paper = emulsion(paper_key)
        self.x = self.paper.log_exposure
        self.curve = neutral_paper_curve(self.paper)
        self.k = 10.0 ** -self.curve.max()           # 相纸最大密度的反射率 (输出黑点)
        self._lut = None

    def density(self, xe):
        """相纸曝光 (每通道 log H) → 相纸密度。"""
        return np.interp(xe, self.x, self.curve)

    def mono(self, Dp):
        out = (10.0 ** -Dp - self.k) / (1.0 - self.k)
        return np.repeat(out.mean(-1, keepdims=True), 3, -1)

    def reference_print(self, Dp, output_black=OUTPUT_BLACK, chunk=1 << 16):
        """相纸密度 (…, 3) → 线性 sRGB (展平, (n, 3))。每个通道驱动相纸的一层: 先求出中性灰阶上各层的
        染料量 (扫描仪的中性标定), 通道密度 Dp_c 按这一关系换算为第 c 层的染料量, 再由染料光谱、
        D50 观看光源算出颜色。相纸白 = 白, 相纸最大密度 (k) = 黑 (相对介质渲染 + 黑点补偿)。"""
        from .. import spectral
        if self._lut is None:
            self._lut = self._neutral_layers()
        Dg, layers, illum, white, wxy = self._lut
        flat = Dp.reshape(-1, 3)
        out = np.empty_like(flat)
        ps, k = self.paper.stock, self.k
        for i in range(0, flat.shape[0], chunk):
            d = np.stack([np.interp(flat[i:i + chunk, c], Dg, layers[:, c]) for c in range(3)], -1)
            R = 10.0 ** -(d @ ps.dye_density.T + ps.base_density)
            xyz = spectral.spectrum_to_xyz(R, illum) / white[1]
            b0 = output_black
            xyz = b0 * white / white[1] + (1.0 - b0) * (xyz - k * white / white[1]) / (1.0 - k)
            out[i:i + chunk] = spectral.xyz_to_srgb(xyz, wxy)
        return out

    def _neutral_layers(self):
        """中性灰阶: 对每个视觉密度 D, 求相纸三层的染料量, 使反射光在 D50 下呈中性、亮度 = 10^-D (相对纸白)。"""
        from scipy.optimize import least_squares
        from .. import spectral
        ps = self.paper.stock
        illum = spectral.illuminant("D50")
        white = spectral.spectrum_to_xyz(10.0 ** -ps.base_density, illum)
        import colour
        wxy = colour.XYZ_to_xy(white) if self.paper_white else spectral.white_xy(illum)
        wn = white / white[1]
        Dg = np.linspace(0.0, float(self.curve.max()) + 0.05, 64)
        layers = np.zeros((len(Dg), 3))
        d0 = np.zeros(3)
        for j, D in enumerate(Dg):
            def resid(d):
                R = 10.0 ** -(d @ ps.dye_density.T + ps.base_density)
                xyz = spectral.spectrum_to_xyz(R, illum) / white[1]
                return np.log10(np.clip(xyz, 1e-9, None)) - np.log10(wn * 10.0 ** -D)
            # 初值取在边界内 (从边界 d=0 出发时优化器会停在原地, 曾导致低密度全部被映射成纸白)
            d0 = least_squares(resid, np.maximum(d0, 0.0) + max(D, 1e-3), bounds=(0, 6)).x
            layers[j] = d0
        return Dg, layers, illum, white, wxy


_PAPERS = {}


def digital_paper(paper_key, paper_white=False):
    """同一种数字相纸只标定一次 (中性灰阶的染料量求解较慢)。"""
    key = (paper_key, paper_white)
    if key not in _PAPERS:
        _PAPERS[key] = DigitalPaper(paper_key, paper_white)
    return _PAPERS[key]
