"""S4' 扫描: 冲印店的底片扫描仪 (Fuji Frontier / Noritsu 一类)。

如今大多数人看到的胶片照片都不是相纸印放的, 而是冲印店扫描出来的。扫描仪本身是成像链的一部分:

  1. 测量 (物理)   窄带 R/G/B 光源透过底片, 传感器读透射率 → 三通道密度。
                   窄带测量"看"染料与橙色遮罩的方式和相纸的宽带感光不同。
  2. 去片基 (物理) 测片头未曝光处 (片基 + 橙色遮罩) 的密度并减掉。
  3. 胶片档案      把三通道的反差 (γ) 拉齐、灰阶对中。Frontier SP-3000 手册: 印放通常用主通道
                   ("Master 0"), 按 DX 码登记胶片只写到黑白 / 单色片; 没有找到按胶片型号的彩色负片档案
                   的证据。所以这里的"档案"是扫描仪的通用通道, 用典型的 C-41 负片 (Portra 800) 近似 ——
                   ⚠ "冲印店用 Portra 800 档案扫 CineStill" 只是网上的说法, 未证实。电影负片、CineStill
                   与通用通道的不匹配是成像的一部分, 由逐张自动校正部分补偿 (Fuji 专利 US6668096B1:
                   按每张的密度直方图、LATD、高光侧 (最低密度) 与暗部侧求灰平衡)。
  4. 数字相纸      Frontier 的设计目标是复现富士 Crystal Archive 相纸的印放效果 (输出为文件:
                   相纸的最大密度写成文件的黑, 而不是相纸在房间里的反射率),
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

实现按真实处理顺序拆分在 filmlab/scanning/ (光学 → 胶片类型换算 → 定光 → 数字相纸 → 输出调整),
本模块的 Scanner 把它们组合起来, 保持原有接口与输出; 见 docs/ARCHITECTURE.md。
"""
from dataclasses import dataclass
import numpy as np

from .scanning import conversion, timing as tm
from .scanning.optics import led_response, measure                       # noqa: F401 (原接口)
from .scanning.paper import OUTPUT_BLACK, neutral_paper_curve, digital_paper   # noqa: F401
from .scanning.timing import (COLOR_LIMIT, HIGHLIGHT_AIM, HL_PCT, CONTRAST_RANGE, SLOPE_LIMIT,  # noqa: F401
                              SHOULDER_WHITE, GRAY_BALANCE, GB_CHROMA_THRESHOLD, PrintTiming)
from .scanning.conditioning import output_tint as _output_tint           # noqa: F401 (按名字在调用时查找)


@dataclass(frozen=True)
class ScannerModel:
    name: str
    led_peaks: tuple          # R G B 光源峰值 (nm)
    led_widths: tuple         # 高斯 σ (nm)
    virtual_paper: str        # 数字相纸
    default_profile: str      # 没有对应档案时冲印店用的通用档案
    output_tint: tuple = ((0.0, 0.0, 0.0), (100.0, 0.0, 0.0))   # 输出灰平衡目标 (L, a, b) 锚点; 默认严格中性


# 扫描仪灰平衡之后的"饱和度校正运算矩阵" (Fuji US6668096B1: 条件设定部决定灰平衡 LUT 与
# "an operation matrix for performing saturation correction"), 作用于线性输出 RGB, 行和 = 1 (中性不变)。
# 系数未公开: 由 SillyStill 的 5 对 CineStill 800T 实拍配对 (同一三脚架的数码原片 + 冲印店扫描) 拟合,
# 每帧允许独立的通道增益 (冲印店逐张定光), 矩阵共用。留一验证 5 对全部改善 (ΔE 中位 11.5→9.7 等);
# 在未参与拟合的 ref3 验证集 B 上 23.8→20.8, 测试集 C 19.8→18.5。tools/eval/fit_scan_matrix.py
SATURATION_MATRIX = np.array([[0.790, -0.064, 0.274],
                              [0.099, 0.765, 0.136],
                              [-0.053, 0.200, 0.853]])


# 冲印店扫描仪输出的灰平衡目标 (随亮度变化的色调, Lab a/b)。Fuji 专利: 扫描仪按暗部 / 高光侧建立
# 灰平衡 LUT, 但其目标值未公开。由一半真实 800T 冲印店扫描 (调参集 A) 估计: 暗部略偏暖品红、中间调偏暖、
# 高光偏黄绿; 在另一半 (验证集 B, 未参与估计) 上综合误差 29.5 → 23.9 (真实 A 与 B 之间为 26–27)。
# (L, a, b) 锚点, 纯黑 / 纯白保持中性。只由 800T 扫描估计, 可能包含尚未建模的胶片交叉偏色。
LAB_TINT = ((0.0, 0.0, 0.0), (10.0, 1.6, 1.1), (45.0, 3.4, 6.1), (85.0, -3.7, 4.2), (100.0, 0.0, 0.0))


SCANNERS = {
    "frontier": ScannerModel("Frontier 型 (估计参数)", (650.0, 540.0, 450.0), (14.0, 16.0, 14.0),
                             "fujifilm_crystal_archive_typeii", "kodak_portra_800", LAB_TINT),
    "frontier_neutral": ScannerModel("Frontier 型, 严格中性输出", (650.0, 540.0, 450.0), (14.0, 16.0, 14.0),
                                     "fujifilm_crystal_archive_typeii", "kodak_portra_800"),
    "noritsu": ScannerModel("Noritsu 型 (估计参数)", (660.0, 525.0, 455.0), (11.0, 13.0, 11.0),
                            "kodak_portra_endura", "kodak_portra_800"),
}

# 扫描仪自带档案的胶片 (主流民用 / 专业负片)。电影负片、CineStill 不在其中。
SCANNER_PROFILES = {
    "fujifilm_c200", "fujifilm_pro_400h", "fujifilm_xtra_400", "kodak_ektar_100", "kodak_gold_200",
    "kodak_portra_160", "kodak_portra_400", "kodak_portra_800", "kodak_ultramax_400",
}


def printing_matrix(scanner: str, profile_key: str, paper_key: str):
    """窄带扫描密度 → 相纸印放密度 (见 scanning.conversion.printing_matrix)。"""
    return conversion.printing_matrix(SCANNERS[scanner], profile_key, paper_key)


def film_profile(profile, scanner: str):
    """扫描仪的胶片档案 (见 scanning.conversion.film_profile)。"""
    return conversion.film_profile(profile, SCANNERS[scanner])


class Scanner:
    """组合扫描的各个阶段。conditioning: "lab" = 输出调整用本模块的 SATURATION_MATRIX 与型号的 output_tint
    (冲印店风格); None = 不做输出调整 (物理参考)。"""

    def __init__(self, engine, scanner="frontier", profile=None, strength=0.75, stretch_max=1.25,
                 conditioning="lab", gray_balance=None, paper_white=False):
        self.eng = engine
        self.m = SCANNERS[scanner]
        e = engine.e
        self.mono = e.design == "bw"
        if profile is None:
            profile = e if self.mono else (e.key if e.key in SCANNER_PROFILES else self.m.default_profile)
        self.profile = profile if isinstance(profile, str) else profile.key
        prof_key = profile if isinstance(profile, str) else (self.m.default_profile if self.mono else profile.key)
        # S4b 胶片类型换算 (档案: 反差拉齐 + 该胶片中灰的通道平衡)
        self.conversion = conversion.FilmTypeConversion(self.m, profile, prof_key, engine._transmittance_base())
        # S4d 数字相纸
        self.digital_paper = digital_paper(self.m.virtual_paper, paper_white)
        self.paper = self.digital_paper.paper
        self.curve = self.digital_paper.curve
        # S4c 定光 (操作员变量)
        self.strength = 0.0 if self.mono else strength
        self.stretch_max = stretch_max
        self.timing = PrintTiming()
        self.gray_balance = gray_balance
        # S4e 输出调整
        self.conditioning = conditioning

    # ---- 兼容原接口的属性
    P = property(lambda self: self.conversion.P)
    gain = property(lambda self: self.conversion.gain)
    base = property(lambda self: self.conversion.base)
    gray0 = property(lambda self: self.conversion.gray0)
    K = property(lambda self: self.timing.K, lambda self, v: setattr(self.timing, "K", v))
    s = property(lambda self: self.timing.s, lambda self, v: setattr(self.timing, "s", v))
    M = property(lambda self: self.timing.M, lambda self, v: setattr(self.timing, "M", v))
    x_knee = property(lambda self: self.timing.x_knee, lambda self, v: setattr(self.timing, "x_knee", v))
    x_white = property(lambda self: self.timing.x_white, lambda self, v: setattr(self.timing, "x_white", v))

    @property
    def _c0(self):
        return 0.0 if self.mono else self.gray0

    # ---- S4a + S4b
    def normalized_density(self, T):
        """去片基 (片基 = 0, 即黑点) → 按档案拉齐三通道反差。"""
        return self.conversion(measure(T, self.m))

    # ---- S4c
    def balance(self, ref_density, mid_gray=0.18, nominal=None, density_limit=0.18, color_limit=COLOR_LIMIT,
                stretch=None):
        """灰点定光 (见 scanning.timing.gray_point)。"""
        self.K, self.s = tm.gray_point(self.digital_paper, ref_density, self._c0, self.strength, self.stretch_max,
                                       mid_gray, nominal, density_limit, color_limit, stretch)

    def balance_frame(self, Dn, nominal, mid_gray=0.18, density_limit=0.3, latd_normal=0.0,
                      slope_limit=SLOPE_LIMIT, color_limit=COLOR_LIMIT):
        """按整张底片自动定光 (见 scanning.timing.frame_auto)。"""
        self.timing = tm.frame_auto(self.digital_paper, Dn, nominal, self._c0, self.strength, self.stretch_max,
                                    mid_gray, density_limit, latd_normal, slope_limit, color_limit,
                                    gray_balance=self.gray_balance)

    # ---- S4d + S4e
    def to_rgb(self, Dn):
        """归一化底片密度 → 数字相纸 → 线性 RGB。片基 (Dn=0) 在三个通道映射到同一点 → 中性的黑。"""
        Dp = self.digital_paper.density(self.timing.paper_exposure(Dn))
        if self.mono:
            return self.digital_paper.mono(Dp)
        return self._reference_print(Dp, self.digital_paper.k)

    def _reference_print(self, Dp, k, chunk=1 << 16):
        """数字相纸的颜色 (参考印放), 再做输出调整。"""
        out = self.digital_paper.reference_print(Dp, OUTPUT_BLACK, chunk)
        return self._condition(out).reshape(Dp.shape)

    def _condition(self, rgb):
        if self.conditioning is None:
            return np.clip(rgb, 0.0, None)
        return _output_tint(np.clip(rgb, 0.0, None) @ SATURATION_MATRIX.T, self.m.output_tint)

    def render(self, T):
        """底片光谱透射率 → 线性 sRGB。"""
        return self.to_rgb(self.normalized_density(T))

    def describe(self):
        if self.mono:
            return f"扫描: {self.m.name}, 灰度模式, 数字相纸 = {self.paper.name} (中性曲线)"
        return (f"扫描: {self.m.name}, 胶片档案 = {self.profile}"
                f"{'' if self.profile == self.eng.e.key else ' (扫描仪通用通道, 以该档案近似)'}, "
                f"数字相纸 = {self.paper.name}, 颜色自动校正 {self.strength:.0%}"
                f"{'' if self.conditioning else ', 无输出调整 (物理参考)'}")
