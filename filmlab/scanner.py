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
_PRINTING = {}


def printing_matrix(scanner: str, profile_key: str, paper_key: str):
    """窄带扫描密度 → 相纸"看到"的印放密度 (3×3, 在密度域线性)。

    相纸用宽带的感光度透过底片曝光: 每种染料的多余吸收会影响相纸的几层 (串扰), 这让印放的颜色
    比窄带测量低饱和。Frontier 的目标是复现银盐相纸印放 (Fuji 2001), 所以扫描后要把窄带密度
    换算成印放密度 (同 Cineon 的 printing density 概念)。换算按扫描仪的通用档案胶片的染料光谱
    拟合 —— 冲印店只有一套通用的换算, 实际胶片与它不同的部分就是成像的一部分。"""
    key = (scanner, profile_key, paper_key)
    if key not in _PRINTING:
        from . import spectral
        film, paper = emulsion(profile_key).stock, emulsion(paper_key).stock
        m = SCANNERS[scanner]
        rng = np.random.default_rng(0)
        d = rng.uniform(0, 2.5, (2000, 3))
        T = 10.0 ** -(d @ film.dye_density.T + film.base_density)
        T0 = 10.0 ** -film.base_density
        dn = measure(T, m) - measure(T0[None, :], m)
        S = spectral.illuminant("TH-KG3")[:, None] * 10.0 ** paper.log_sensitivity
        dp = -np.log10((T @ S) / (T0 @ S))
        _PRINTING[key] = np.linalg.lstsq(dn, dp, rcond=None)[0]
    return _PRINTING[key]


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
    D = D @ printing_matrix(scanner, eng.e.key if eng.e.design != "bw" else SCANNERS[scanner].default_profile,
                            SCANNERS[scanner].virtual_paper)
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


COLOR_LIMIT = 0.6    # 自动颜色校正的最大偏移 (底片密度), 估计值
# 高光侧 (绿通道密度 95–99%) 的目标反射率: 冲印店扫描给高光留有余量。没有公开数据,
# 由 32 张真实 800T 夜景 / 室内扫描的亮度 95 百分位推得 (中位数 0.58–0.76)。
HIGHLIGHT_AIM = 0.65
# 高光侧取绿通道密度的哪一段 (百分位)。夜景里最密的 1–3% 是光源与光晕 (在胶片肩部), 不是场景的高光;
# 印放机拟合影调范围时不应按光源压缩反差。
HL_PCT = [92.0, 96.0]
# 逐张反差拟合的上下限。94 张真实 800T 扫描显示冲印店基本保持底片自身的反差 (拟合到相纸全范围
# 会让影调范围比真实宽 5–15), 所以不做逐张反差拉伸。
CONTRAST_RANGE = (1.0, 1.0)
# 灰平衡对单通道反差的最大改变。真实扫描: 暗部接近中性而高光保留胶片的黄绿交叉偏色 —— 冲印店主要
# 按偏移校正, 只能少量改变通道反差 (±15%, 由 94 张真实扫描的综合误差最小推得)。
SLOPE_LIMIT = 0.15


SHOULDER_WHITE = 0.93   # 高光肩部的渐近亮度 (真实 800T 扫描 99 百分位中位数 0.74–0.91), 估计值


class Scanner:
    def __init__(self, engine, scanner="frontier", profile=None, strength=0.75, stretch_max=1.25):
        self.eng = engine
        self.m = SCANNERS[scanner]
        e = engine.e
        self.mono = e.design == "bw"
        if profile is None:
            profile = e if self.mono else (e.key if e.key in SCANNER_PROFILES else self.m.default_profile)
        self.profile = profile if isinstance(profile, str) else profile.key
        prof_key = profile if isinstance(profile, str) else (self.m.default_profile if self.mono else profile.key)
        self.P = printing_matrix(scanner, prof_key, self.m.virtual_paper)
        self.gain, self.gray0 = film_profile(profile, scanner)   # 档案: 反差拉齐 + 该胶片中灰的通道平衡
        self.base = measure(engine._transmittance_base(), self.m)
        self.paper = emulsion(self.m.virtual_paper)
        self.curve = neutral_paper_curve(self.paper)   # 中性的数字相纸曲线
        self._print_lut = None
        self.strength = 0.0 if self.mono else strength
        self.stretch_max = stretch_max
        self.K, self.s, self.M = np.zeros(3), np.ones(3), None
        self.x_knee = None

    def normalized_density(self, T):
        """去片基 (片基 = 0, 即黑点) → 按档案拉齐三通道反差。"""
        return ((measure(T, self.m) - self.base) @ self.P) * self.gain

    def to_rgb(self, Dn):
        """归一化底片密度 → 数字相纸 → 线性 RGB。片基 (Dn=0) 在三个通道映射到同一点 → 中性的黑。"""
        x = self.paper.log_exposure
        if self.M is not None:              # 逐张的两点灰平衡 (每通道线性映射)
            Dn = self.M[0] + self.M[1] * Dn
        xe = self.K - Dn * self.s
        if self.x_knee is not None:
            # 高光肩部 (Fuji 2001: 高光区灰阶非线性): 比高光侧更亮的部分 (光源、反光) 在相纸曝光域里
            # 平滑地压向接近纸白的 x_white, 而不是越过相纸趾部被截成纯白
            xk, xw = self.x_knee, self.x_white
            d = np.maximum(xk - xe, 0.0)
            xe = np.where(xe < xk, xk - (xk - xw) * (1.0 - np.exp(-d / max(xk - xw, 1e-3))), xe)
        Dp = np.interp(xe, x, self.curve)
        k = 10.0 ** -self.curve.max()
        if self.mono:
            out = (10.0 ** -Dp - k) / (1.0 - k)
            return np.repeat(out.mean(-1, keepdims=True), 3, -1)
        return self._reference_print(Dp, k)

    def _reference_print(self, Dp, k, chunk=1 << 16):
        """数字相纸的颜色: 按真实相纸染料光谱计算的参考印放 (Fuji: Frontier 的目标是复现银盐相纸印放)。
        相纸的三种染料都不纯 (青染料也吸收一些绿光, 品红吸收一些蓝光 …), 这种串扰让印放的颜色
        比"三通道各自独立"低饱和 —— 不能把三个通道直接当作 RGB 输出。
        每个通道驱动相纸的一层: 先求出中性灰阶上各层的染料量 (扫描仪的中性标定), 通道密度 Dp_c
        按这一关系换算为第 c 层的染料量, 再由染料光谱、D50 观看光源算出颜色。
        输出是数字文件: 相纸白 = 白, 相纸最大密度 (k) = 黑 (相对介质渲染 + 黑点补偿)。"""
        from . import spectral
        if self._print_lut is None:
            self._print_lut = self._neutral_layers()
        Dg, layers, illum, white, wxy = self._print_lut
        shape = Dp.shape
        flat = Dp.reshape(-1, 3)
        out = np.empty_like(flat)
        ps = self.paper.stock
        for i in range(0, flat.shape[0], chunk):
            d = np.stack([np.interp(flat[i:i + chunk, c], Dg, layers[:, c]) for c in range(3)], -1)
            R = 10.0 ** -(d @ ps.dye_density.T + ps.base_density)
            xyz = spectral.spectrum_to_xyz(R, illum) / white[1]
            xyz = (xyz - k * white / white[1]) / (1.0 - k)
            out[i:i + chunk] = spectral.xyz_to_srgb(xyz, wxy)
        return out.reshape(shape)

    def _neutral_layers(self):
        """中性灰阶: 对每个视觉密度 D, 求相纸三层的染料量, 使反射光在 D50 下呈中性、亮度 = 10^-D (相对纸白)。"""
        from scipy.optimize import least_squares
        from . import spectral
        ps = self.paper.stock
        illum = spectral.illuminant("D50")
        white = spectral.spectrum_to_xyz(10.0 ** -ps.base_density, illum)
        wxy = spectral.white_xy(illum)
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

    def balance(self, ref_density, mid_gray=0.18, nominal=None, density_limit=0.18, color_limit=COLOR_LIMIT,
                stretch=None):
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
        if stretch is None:
            stretch = float(np.clip((x_black - need) / g, 1.0, self.stretch_max))   # 让片基尽量接近纯黑
        corr = np.clip(self.strength * ((ref - ref[1]) - c0), -color_limit / stretch, color_limit / stretch)
        self.s = np.full(3, stretch)
        self.K = need + stretch * (g + c0 + corr)                   # 绿通道: 灰点精确落在中灰

    def balance_frame(self, Dn, nominal, mid_gray=0.18, density_limit=0.3, latd_normal=0.0,
                      slope_limit=SLOPE_LIMIT, color_limit=COLOR_LIMIT):
        """按整张底片自动定标 (Frontier 型; Fuji 专利 US6668096B1: 由每张的密度直方图求平均密度 /
        LATD、高光侧与暗部侧, 再建立各通道的灰平衡 LUT)。

        亮度  LATD (整张平均透射光的密度) 相对"正常场景"的偏离, 校正不超过 ±density_limit。
        颜色  两点灰平衡: 暗部侧 (绿通道密度最低的 1–5% 像素) 与高光侧 (95–99%, 避开光源本身)
              两个区域在每个通道上都映射到中性 (以档案的中灰平衡为中性)。每通道是一个线性映射
              (偏移 + 斜率) —— 斜率能补偿胶片在非设计光源下各层落在曲线不同位置造成的反差差异
              (日光下的灯光片)。力度 strength 部分校正, 斜率变化不超过 ±slope_limit, 偏移不超过
              color_limit —— 防止把钠灯等近单色场景的弱通道拉成极端反差。"""
        Dn = np.asarray(Dn, float)
        ok = np.all(np.isfinite(Dn), -1)
        g = Dn[ok, 1]
        lo, hi = np.percentile(g, [1, 5] + HL_PCT).reshape(2, 2)
        sh = Dn[ok][(g >= lo[0]) & (g <= lo[1])].mean(0)
        hl = Dn[ok][(g >= hi[0]) & (g <= hi[1])].mean(0)
        c0 = 0.0 if self.mono else self.gray0
        span = hl - sh
        full = np.where(np.abs(span) > 0.02, (span[1]) / np.where(np.abs(span) > 0.02, span, 1.0), 1.0)
        r = np.clip(1.0 + self.strength * (full - 1.0), 1.0 - slope_limit, 1.0 + slope_limit)
        # 不校正: 只按档案的中灰平衡 (Dn_c − c0_c); 完全校正: 暗部侧映射到绿通道的暗部侧。
        # 部分校正: 暗部侧的目标在两者之间按 strength 插值, 偏离"不校正"不超过 color_limit
        # 颜色偏移的两种依据 (Fuji 专利: 由 LATD 与暗部 / 高光侧共同求灰平衡):
        #   暗部侧中性 —— 夜景里暗部不受灯光颜色影响, 是可靠的中性参考;
        #   LATD 中性 (印放机的积分中性法) —— 日光 / 蓝调里暗部是天光照的阴影, 本身偏蓝。
        lat = -np.log10(np.mean(10.0 ** -(r * Dn[ok]), 0))
        none_l, none_sh = lat - c0, r * sh - c0
        off_l = none_l + np.clip(self.strength * (lat[1] - none_l), -color_limit, color_limit) - lat
        off_s = none_sh + np.clip(self.strength * (r[1] * sh[1] - none_sh), -color_limit, color_limit) - r * sh
        wl = 0.5          # 两者各半 (94 张真实扫描的综合误差: 仅暗部 16.9, 仅 LATD 17.7, 各半 16.7)
        off = wl * off_l + (1 - wl) * off_s
        r[1], off[1] = 1.0, 0.0
        self.M = (off, r)
        Dm = off + r * Dn[ok]
        latd = -np.log10(np.mean(10.0 ** -Dm, 0))
        # 反差: 把这张底片暗部侧到高光侧的密度范围放进相纸的再现范围 (Fuji 2001: 反差按相纸狭窄的
        # 再现范围拟合)。相纸再现范围取曲线 3%–97% 之间的曝光范围; 反差变化有上下限 (估计)。
        x, cv = self.paper.log_exposure, self.curve
        x_hl = float(np.interp(-np.log10(HIGHLIGHT_AIM), cv, x))      # 高光侧落在相纸上的位置
        span_paper = float(x[np.argmax(cv >= 0.97 * cv.max())] - x_hl)
        span_film = max(float(hl[1] - sh[1]), 0.05)
        stretch = float(np.clip(span_paper / span_film, CONTRAST_RANGE[0], CONTRAST_RANGE[1]))
        # 亮度与反差只由绿通道决定; 颜色已在 M 中校正, 这里不再做通道偏移
        self.gray0, saved = np.zeros(3), self.gray0
        self.balance(latd - latd_normal, mid_gray, nominal=nominal, density_limit=density_limit,
                     color_limit=0.0, stretch=stretch)
        self.gray0 = saved
        self.K = np.full(3, self.K[1])
        # 肩部起点: 高光侧在相纸上的位置, 但不比高光目标更亮 (亮度校正把高光侧推过目标时, 从目标处开始压缩)
        x_band = float(self.K[1] - stretch * float(off[1] + r[1] * hl[1]))
        self.x_knee = max(x_band, x_hl)
        self.x_white = float(np.interp(-np.log10(SHOULDER_WHITE), cv, x))

    def render(self, T):
        """底片光谱透射率 → 线性 sRGB。"""
        return self.to_rgb(self.normalized_density(T))

    def describe(self):
        if self.mono:
            return f"扫描: {self.m.name}, 灰度模式, 数字相纸 = {self.paper.name} (中性曲线)"
        return (f"扫描: {self.m.name}, 胶片档案 = {self.profile}"
                f"{'' if self.profile == self.eng.e.key else ' (扫描仪通用通道, 以该档案近似)'}, "
                f"数字相纸 = {self.paper.name}, 颜色自动校正 {self.strength:.0%}")
