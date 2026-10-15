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
  5. 反转与校正    和光学印放一样, 校正只有整体反差 (自动色阶的拉伸, 三通道相同) 与每通道的
                   曝光偏移 (C·M·Y 校正键 = 印放光号)。偏移校正后中间调中性, 剩余偏色落在暗部
                   且方向与原偏色相反 (负片扫描的交叉偏色)。
                   亮度按整张底片的平均密度 (同 LATD) 校正, 但最多 ±1 档 (不把夜景拉成白天);
                   颜色只校正一部分且有范围上限, 以免把日落、夜景这类主体"校死" —— 力度因扫描仪
                   设置与操作员而异, 在一个范围内随机 (variation.py)。

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


def neutral_paper_curve(paper):
    """相纸三层特性曲线的平均影调。三层的感光度位置不同, 曲线在横轴上错开 ——
    先按各层中间密度处对齐再平均, 否则平均会把曲线抹宽, 反差被低估。"""
    x, S = paper.log_exposure, paper.S_nom - paper.S_nom.min(0)
    mids = [x[np.argmax(S[:, c] >= 0.5 * S[:, c].max())] for c in range(3)]
    ref = mids[1]
    aligned = np.stack([np.interp(x + (mids[c] - ref), x, S[:, c]) for c in range(3)], 1)
    c = aligned.mean(1)
    return c - c.min()


COLOR_LIMIT = 0.45   # 自动颜色校正的最大偏移 (logH, ≈18 个印放光号), 估计值


class Scanner:
    def __init__(self, engine, scanner="frontier", profile=None, strength=0.75, stretch_max=1.25):
        self.eng = engine
        self.m = SCANNERS[scanner]
        e = engine.e
        self.mono = e.design == "bw"
        if profile is None:
            profile = e if self.mono else (e.key if e.key in SCANNER_PROFILES else self.m.default_profile)
        self.profile = profile if isinstance(profile, str) else profile.key
        self.gain, self.gray0 = film_profile(profile, scanner)   # 档案: 反差拉齐 + 该胶片中灰的通道平衡
        self.base = measure(engine._transmittance_base(), self.m)
        self.paper = emulsion(self.m.virtual_paper)
        self.curve = neutral_paper_curve(self.paper)   # 中性的数字相纸曲线
        self.strength = 0.0 if self.mono else strength
        self.stretch_max = stretch_max
        self.K, self.s = np.zeros(3), np.ones(3)

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

    def balance(self, ref_density, mid_gray=0.18, nominal=None, density_limit=0.18, color_limit=COLOR_LIMIT):
        """自动定标。数字相纸曝光 logH_c = K_c − s·Dn_c (Dn: 去片基后、按档案拉齐反差的底片密度)。

        和光学印放一样, 校正只有两种: 整体反差 s (三通道相同, 反差由胶片档案决定) 与
        每通道的曝光偏移 K_c (= 印放光号 / 扫描仪的 C·M·Y 校正键)。不逐通道改反差。

        灰点  参考密度 (灰卡或整张平均) → 中灰。亮度总是满足; 颜色按 strength 部分校正,
              且每通道偏移不超过 color_limit (logH, 扫描仪校正键的范围)。
              偏移校正的结果: 中间调被校正, 剩下的偏色在暗部 (方向与原偏色相反) —— 这就是
              负片扫描的"交叉偏色", 例如钠灯夜景校正后暗部偏蓝紫。
        黑点  光学印放时, 正常曝光负片的片基 (Dn=0) 恰好印到相纸的最大密度 (纯黑);
              薄 (欠曝) 的负片片基印不到纯黑, 暗部发灰 —— 这是真实的物理现象。
              扫描仪会做自动色阶, 把反差拉伸让黑更深, 但拉伸幅度有限 (stretch_max, 扫描设置,
              在范围内随机); 超出上限时就保留发灰的黑。
        不校正时的颜色 = 胶片档案里的中灰平衡 (gray0)。档案与实际胶片不符时 (CineStill 用通用的
        Portra 800 档案), 二者中灰的差别就是需要自动校正去补的偏色, 校正不完的部分留在成片里。
        nominal = None: 按灰卡定标。
        nominal = 标准灰卡的密度: 按整张平均自动校正时, 亮度相对标准只移动 ±density_limit
                  (底片密度, ≈1 档) —— 不把夜景拉成白天。"""
        x = self.paper.log_exposure
        need = np.interp(-np.log10(mid_gray), self.curve, x)        # 数字相纸上中灰需要的曝光
        x_black = float(x[np.argmax(self.curve >= 0.97 * self.curve.max())])
        ref = np.asarray(ref_density, float)
        c0 = 0.0 if self.mono else self.gray0                      # 不校正时的通道平衡 (档案)
        if nominal is None:
            g = ref[1]
        else:
            r0 = np.asarray(nominal, float)
            g = float(np.clip(ref[1], r0[1] - density_limit, r0[1] + density_limit))
        g = max(g, 0.05)
        stretch = float(np.clip((x_black - need) / g, 1.0, self.stretch_max))   # 让片基尽量接近纯黑
        corr = np.clip(self.strength * ((ref - ref[1]) - c0), -color_limit / stretch, color_limit / stretch)
        self.s = np.full(3, stretch)
        self.K = need + stretch * (g + c0 + corr)                   # 绿通道: 灰点精确落在中灰

    def render(self, T):
        """底片光谱透射率 → 线性 sRGB。"""
        return self.to_rgb(self.normalized_density(T))

    def describe(self):
        if self.mono:
            return f"扫描: {self.m.name}, 灰度模式, 数字相纸 = {self.paper.name} (中性曲线)"
        return (f"扫描: {self.m.name}, 胶片档案 = {self.profile}"
                f"{'' if self.profile == self.eng.e.key else ' (没有该胶片的档案, 用通用档案)'}, "
                f"数字相纸 = {self.paper.name}, 颜色自动校正 {self.strength:.0%}")
