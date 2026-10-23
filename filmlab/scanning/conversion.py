"""S4b 胶片类型换算 (扫描仪软件, 仍在负片密度域)。

  去片基   减去片头未曝光处 (片基 + 橙色遮罩) 的扫描密度。
  印放密度 窄带扫描密度 → 相纸"看到"的宽带印放密度 (P, 3×3, 在密度域线性)。
  档案     按胶片档案的标准灰阶把三通道反差拉齐到绿通道 (gain), 记下档案中灰的通道平衡 (gray0)。

Frontier SP-3000 手册: 彩色负片印放通常用主通道, 没有找到按胶片型号的彩色档案的证据, 所以档案是
扫描仪的通用通道, 用典型的 C-41 负片 (Portra 800) 近似; 实际胶片与它的差别是成像的一部分。
"""
import numpy as np

from ..materials import emulsion
from .optics import measure

_PROFILES = {}
_PRINTING = {}


def printing_matrix(m, profile_key: str, paper_key: str):
    """窄带扫描密度 → 相纸印放密度 (3×3)。

    相纸用宽带感光度透过底片曝光: 每种染料的多余吸收会影响相纸的几层 (串扰), 这让印放的颜色
    比窄带测量低饱和。当前模型以银盐相纸印放作为扫描输出代理, 所以把窄带密度
    换算成印放密度 (类似 Cineon 的 printing density)。换算按扫描仪通用档案胶片的染料光谱拟合 ——
    冲印店只有一套通用的换算, 实际胶片与它不同的部分就是成像的一部分。
    这是当前虚拟印放代理的构造, 不是由 Frontier 的数字输出 LUT 测量直接得到的换算。"""
    key = (m, profile_key, paper_key)
    if key not in _PRINTING:
        from .. import spectral
        film, paper = emulsion(profile_key).stock, emulsion(paper_key).stock
        rng = np.random.default_rng(0)
        d = rng.uniform(0, 2.5, (2000, 3))
        T = 10.0 ** -(d @ film.dye_density.T + film.base_density)
        T0 = 10.0 ** -film.base_density
        dn = measure(T, m) - measure(T0[None, :], m)
        S = spectral.illuminant("TH-KG3")[:, None] * 10.0 ** paper.log_sensitivity
        dp = -np.log10((T @ S) / (T0 @ S))
        _PRINTING[key] = np.linalg.lstsq(dn, dp, rcond=None)[0]
    return _PRINTING[key]


def film_profile(profile, m):
    """扫描仪的胶片档案: 用该胶片的标准灰阶楔 (原生工艺、标称冲洗) 求每通道的 γ 归一化与灰点。
    profile: 胶片名或 Emulsion; m: 扫描仪型号。"""
    key = (profile if isinstance(profile, str) else profile.key, m)
    if key not in _PROFILES:
        _PROFILES[key] = _film_profile(profile, m)
    return _PROFILES[key]


def _film_profile(profile, m):
    from ..engine import Engine, Params, GRAY
    eng = Engine(Params(film=profile, grain=0, roll=None, output="print"))
    steps = np.linspace(-1.0, 1.0, 9)
    rgb = (GRAY * 10.0 ** steps)[:, None] * np.ones(3)
    H = eng._expose_lin(rgb, eng.M_ref, compensation=False)
    st = eng._develop(eng._log_exposure(H))
    D = measure(eng._transmittance(st), m) - measure(10.0 ** -eng.e.stock.base_density, m)
    D = D @ printing_matrix(m, eng.e.key if eng.e.design != "bw" else m.default_profile, m.virtual_paper)
    slope = np.polyfit(steps, D, 1)[0]                 # 每通道 dD / dlogH
    gain = slope[1] / slope                            # 拉齐到绿通道的反差
    gray = D[len(steps) // 2] * gain                   # 中灰在各通道的位置
    return gain, gray - gray[1]


class FilmTypeConversion:
    """扫描密度 → 归一化底片密度 Dn (片基 = 0, 三通道反差按档案拉齐)。"""

    def __init__(self, m, profile, prof_key, base_T):
        self.m = m
        self.P = printing_matrix(m, prof_key, m.virtual_paper)
        self.gain, self.gray0 = film_profile(profile, m)
        self.base = measure(base_T, m)

    def __call__(self, D_scan):
        return ((D_scan - self.base) @ self.P) * self.gain
