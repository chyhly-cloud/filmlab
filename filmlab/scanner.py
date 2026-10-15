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
  5. 反转与校正    和真实的底片扫描仪一样, 片基 (未曝光处) 是每个通道的黑点, 中灰的平衡
                   靠调整各通道的反差 (斜率) 完成 —— 黑点锚定, 黑始终是中性的。
                   亮度按整张底片的平均密度 (同 LATD) 校正, 但最多 ±1 档 (不把夜景拉成白天);
                   颜色只校正一部分, 以免把日落、夜景这类主体"校死" —— 力度因扫描仪设置与
                   操作员而异, 在一个范围内随机 (variation.py)。

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
        self.gain, _ = film_profile(profile, scanner)
        self.base = measure(engine._transmittance_base(), self.m)
        self.paper = emulsion(self.m.virtual_paper)
        c = self.paper.S_nom.mean(1)
        self.curve = c - c.min()                       # 中性的数字相纸曲线
        self.strength = 0.0 if self.mono else strength
        self.K, self.s = 0.0, np.ones(3)

    def normalized_density(self, T):
        """去片基 (片基 = 0, 即黑点) → 按档案拉齐三通道反差。"""
        return (measure(T, self.m) - self.base) * self.gain

    def to_rgb(self, Dn):
        """归一化底片密度 → 数字相纸 → 线性 RGB。片基 (Dn=0) 在三个通道映射到同一点 → 中性的黑。"""
        x = self.paper.log_exposure
        Dp = np.interp(self.K - Dn * self.s, x, self.curve)
        out = 10.0 ** -Dp
        if self.mono:
            out = np.repeat(out.mean(-1, keepdims=True), 3, -1)
        return out

    def balance(self, ref_density, mid_gray=0.18, nominal=None, density_limit=0.3):
        """自动校正: 让参考密度 (灰卡或整张平均) 印成中灰。数字相纸曝光 logH = K − s·Dn:
        K 决定亮度 (以绿通道为准), 各通道斜率 s 决定颜色平衡 (黑点 Dn=0 不受影响)。
        nominal = None: 按灰卡定光 (亮度全校正, 颜色按 strength 部分校正)。
        nominal = 标准灰卡的密度: 按整张平均自动校正时, 亮度相对标准只校正 ±density_limit (log10,
                  0.3 ≈ 1 档); 颜色在标准灰卡的平衡上, 再按 strength 部分跟随整张平均。"""
        x = self.paper.log_exposure
        need = np.interp(-np.log10(mid_gray), self.curve, x)       # 数字相纸上中灰需要的曝光

        def solve(r):
            r = np.maximum(np.asarray(r, float), 0.05)
            return need + r[1], r[1] / r                           # 让 r 在三通道都印成中灰
        if nominal is None:
            self.K, s_full = solve(ref_density)
            self.s = 1.0 + self.strength * (s_full - 1.0)
            return
        K0, s0 = solve(nominal)
        Kf, sf = solve(ref_density)
        self.K = K0 + float(np.clip(Kf - K0, -density_limit, density_limit))
        self.s = s0 + self.strength * (sf - s0)

    def render(self, T):
        """底片光谱透射率 → 线性 sRGB。"""
        return self.to_rgb(self.normalized_density(T))

    def describe(self):
        if self.mono:
            return f"扫描: {self.m.name}, 灰度模式, 数字相纸 = {self.paper.name} (中性曲线)"
        return (f"扫描: {self.m.name}, 胶片档案 = {self.profile}"
                f"{'' if self.profile == self.eng.e.key else ' (没有该胶片的档案, 用通用档案)'}, "
                f"数字相纸 = {self.paper.name}, 颜色自动校正 {self.strength:.0%}")
