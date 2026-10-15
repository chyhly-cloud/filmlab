"""S4' 扫描: 冲印店的底片扫描仪 (Fuji Frontier / Noritsu 一类)。

如今大多数人看到的胶片照片都不是相纸印放的, 而是冲印店扫描出来的。扫描仪本身是成像链的一部分:

  1. 测量 (物理)   窄带 R/G/B 光源透过底片, 传感器读透射率 → 三通道密度。
                   窄带测量"看"染料与橙色遮罩的方式和相纸的宽带感光不同。
  2. 去片基 (物理) 测片头未曝光处 (片基 + 橙色遮罩) 的密度并减掉。
  3. 胶片档案      按操作员选的胶片类型, 把三通道的反差 (γ) 拉齐、灰阶对中。
                   扫描仪里没有 CineStill / 电影负片的档案 —— 冲印店通常用通用的 Kodak 档案,
                   档案与实际胶片的不匹配也是成像的一部分。
  4. 数字相纸      Frontier 的设计目标是复现富士 Crystal Archive 相纸的印放效果,
                   这里用该相纸实测特性曲线的影调形状做反转 (负 → 正)。真实相纸的三层染料
                   组合起来是中性的, 所以三个通道共用一条中性曲线 (三层曲线的平均)。
     黑白胶片按冲印店的做法用灰度模式扫描, 档案为该胶片本身。
  5. 自动校正      按整张底片的平均密度 (同 LATD) 校正亮度和颜色。为了不把日落、夜景这类
                   主体"校死", 颜色只校正一部分 —— 校正力度因扫描仪设置与操作员而异,
                   在一个范围内随机 (variation.py)。

⚠ 光源峰值波长、档案选择、校正力度为估计值 (厂商未公开)。
"""
from dataclasses import dataclass
import numpy as np

from . import chemistry
from .materials import emulsion
from .stocks import WAVELENGTHS


@dataclass(frozen=True)
class ScannerModel:
    name: str
    led_peaks: tuple          # R G B 光源峰值 (nm)
    led_widths: tuple         # 高斯 σ (nm)
    virtual_paper: str        # 数字相纸
    default_profile: str      # 没有对应档案时冲印店用的通用档案


SCANNERS = {
    "frontier": ScannerModel("Frontier 型 (估计参数)", (650.0, 540.0, 450.0), (14.0, 16.0, 14.0),
                             "fujifilm_crystal_archive_typeii", "kodak_portra_800"),
    "noritsu": ScannerModel("Noritsu 型 (估计参数)", (660.0, 525.0, 455.0), (11.0, 13.0, 11.0),
                            "kodak_portra_endura", "kodak_portra_800"),
}

# 扫描仪自带档案的胶片 (主流民用 / 专业负片)。电影负片、CineStill 不在其中。
SCANNER_PROFILES = {
    "fujifilm_c200", "fujifilm_pro_400h", "fujifilm_xtra_400", "kodak_ektar_100", "kodak_gold_200",
    "kodak_portra_160", "kodak_portra_400", "kodak_portra_800", "kodak_ultramax_400",
}


def led_response(m: ScannerModel):
    L = np.stack([np.exp(-0.5 * ((WAVELENGTHS - p) / w) ** 2) for p, w in zip(m.led_peaks, m.led_widths)], 1)
    return L / L.sum(0)


def measure(T, m: ScannerModel):
    """光谱透射率 (..., 81) → 扫描密度 (..., 3)。"""
    return -np.log10(np.clip(T @ led_response(m), 1e-6, None))


_PROFILES = {}


def film_profile(profile, scanner: str):
    """扫描仪的胶片档案: 用该胶片的标准灰阶楔 (原生工艺、标称冲洗) 求每通道的 γ 归一化与灰点。
    profile: 胶片名或 Emulsion。"""
    key = (profile if isinstance(profile, str) else profile.key, scanner)
    if key not in _PROFILES:
        _PROFILES[key] = _film_profile(profile, scanner)
    return _PROFILES[key]


def _film_profile(profile, scanner: str):
    from .engine import Engine, Params, GRAY
    m = SCANNERS[scanner]
    eng = Engine(Params(film=profile, grain=0, roll=None, output="print"))
    steps = np.linspace(-1.0, 1.0, 9)
    rgb = (GRAY * 10.0 ** steps)[:, None] * np.ones(3)
    H = eng._expose_lin(rgb, eng.M_ref, compensation=False)
    st = eng._develop(eng._log_exposure(H))
    D = measure(eng._transmittance(st), m) - measure(10.0 ** -eng.e.stock.base_density, m)
    slope = np.polyfit(steps, D, 1)[0]                 # 每通道 dD / dlogH
    gain = slope[1] / slope                            # 拉齐到绿通道的反差
    gray = D[len(steps) // 2] * gain                   # 中灰在各通道的位置
    return gain, gray - gray[1]


class Scanner:
    def __init__(self, engine, scanner="frontier", profile=None, strength=0.75):
        self.eng = engine
        self.m = SCANNERS[scanner]
        e = engine.e
        self.mono = e.design == "bw"
        if profile is None:
            profile = e if self.mono else (e.key if e.key in SCANNER_PROFILES else self.m.default_profile)
        self.profile = profile if isinstance(profile, str) else profile.key
        self.gain, self.offset = film_profile(profile, scanner)
        self.base = measure(engine._transmittance_base(), self.m)
        self.paper = emulsion(self.m.virtual_paper)
        c = self.paper.S_nom.mean(1)
        self.curve = c - c.min()                       # 中性的数字相纸曲线
        self.strength = 0.0 if self.mono else strength
        self.k = np.zeros(3)

    def normalized_density(self, T):
        """去片基 → 档案 γ 归一化 → 灰点对中。"""
        return (measure(T, self.m) - self.base) * self.gain - self.offset

    def to_rgb(self, Dn):
        """归一化底片密度 → 数字相纸 → 线性 RGB。"""
        x = self.paper.log_exposure
        Dp = np.interp(-Dn + self.k, x, self.curve)
        out = 10.0 ** -Dp
        if self.mono:
            out = np.repeat(out.mean(-1, keepdims=True), 3, -1)
        return out

    def balance(self, ref_density, mid_gray=0.18):
        """自动校正: 让参考密度 (灰卡或整张平均) 印成中灰。亮度全校正, 颜色按 strength 部分校正。"""
        x = self.paper.log_exposure
        need = np.interp(-np.log10(mid_gray), self.curve, x)                     # 数字相纸上中灰需要的曝光
        k = need + ref_density                                                   # logH = −Dn + k
        m = k[1]
        self.k = m + self.strength * (k - m)

    def render(self, T):
        """底片光谱透射率 → 线性 sRGB。"""
        return self.to_rgb(self.normalized_density(T))

    def describe(self):
        if self.mono:
            return f"扫描: {self.m.name}, 灰度模式, 数字相纸 = {self.paper.name} (中性曲线)"
        return (f"扫描: {self.m.name}, 胶片档案 = {self.profile}"
                f"{'' if self.profile == self.eng.e.key else ' (没有该胶片的档案, 用通用档案)'}, "
                f"数字相纸 = {self.paper.name}, 颜色自动校正 {self.strength:.0%}")
