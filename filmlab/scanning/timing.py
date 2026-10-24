"""S4c 定光 (自动校色): 冲印店对一张底片能做的操作, 发生在负片密度域、成正像之前。

和光学印放一样, 自由度只有:
  · 每通道的曝光偏移 K_c (= 印放光号 / 扫描仪的 C·M·Y 校正键);
  · 整体反差 s (三通道相同);
  · 灰平衡 LUT (Fuji US6668096B1): 每通道 Dn → off + r·Dn, 斜率变化受限;
  · 高光肩部 (Fuji 2001: 高光区灰阶非线性)。
数字相纸曝光 xe_c = K_c − s · (off_c + r_c · Dn_c), 再进入相纸特性曲线 (paper.py)。

这些是 **操作员 / 扫描设置变量**, 不是胶片物理: 力度、上限、取样区间都只有估计值。
策略: standard (胶片在设计光源下的中灰一次定标) / gray (拍摄光下的灰卡) / frame (按整张底片自动) /
固定 PrintTiming (由评估工具给定, 例如按真实扫描拟合的定光预言机)。
"""
from dataclasses import dataclass, field

import numpy as np

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
# 成片 99 百分位统计不能辨识扫描仪响应的渐近值。0.93 和下述指数形状均为经验呈现假设,
# 不是相纸或扫描仪测得的物理参数; 见 docs/SHOULDER_ABLATION.md。默认保留以便历史复现。
SHOULDER_WHITE = 0.93
# 逐帧灰平衡方法: "two_point" = 现行 (暗部 / 整体平均偏移 + 受限斜率);
# "us6160634" = Fuji US6160634 三种估计 0.2:0.5:0.3 加权 (待定: ref3 B 20.78→20.48, C 18.50→17.67,
# EPFL 夜景 13.25→13.20, SillyStill 12.00→12.04; 见 tools/eval/CRITERIA.md)。高彩度阈值在调参集 A 上选出 0.2。
GRAY_BALANCE = "two_point"
GB_CHROMA_THRESHOLD = 0.2


@dataclass
class PrintTiming:
    """一张底片的定光结果。"""
    K: np.ndarray = field(default_factory=lambda: np.zeros(3))
    s: np.ndarray = field(default_factory=lambda: np.ones(3))
    M: tuple | None = None            # 灰平衡 (off, r)
    x_knee: float | None = None       # 高光肩部起点 (相纸曝光)
    x_white: float | None = None

    def paper_exposure(self, Dn):
        """Dn → 每通道的数字相纸曝光 xe。"""
        if self.M is not None:
            Dn = self.M[0] + self.M[1] * Dn
        xe = self.K - Dn * self.s
        if self.x_knee is not None:
            # 比高光侧更亮的部分 (光源、反光) 在相纸曝光域里平滑地压向接近纸白的 x_white,
            # 而不是越过相纸趾部被截成纯白
            xk, xw = self.x_knee, self.x_white
            d = np.maximum(xk - xe, 0.0)
            xe = np.where(xe < xk, xk - (xk - xw) * (1.0 - np.exp(-d / max(xk - xw, 1e-3))), xe)
        return xe


def gray_point(paper, ref_density, gray0, strength, stretch_max, mid_gray=0.18, nominal=None,
               density_limit=0.18, color_limit=COLOR_LIMIT, stretch=None):
    """灰点定光 → (K, s)。参考密度 (灰卡或整张平均) → 中灰。亮度总是满足; 颜色按 strength 部分校正,
    每通道偏移不超过 color_limit。偏移校正后中间调中性, 剩下的偏色在暗部 (方向与原偏色相反) ——
    负片扫描的"交叉偏色"。
    黑点: 正常曝光负片的片基 (Dn=0) 恰好印到相纸最大密度; 薄负片片基印不到纯黑 (暗部发灰, 真实现象)。
    扫描仪的自动色阶把反差拉伸让黑更深, 但不超过 stretch_max。
    不校正时的颜色 = 档案的中灰平衡 gray0 (CineStill 用通用 Portra 800 档案时, 二者之差就是要补的偏色)。
    nominal = 标准灰卡的密度: 按整张平均校正时, 亮度相对标准只移动 ±density_limit (不把夜景拉成白天)。"""
    x, curve = paper.x, paper.curve
    need = np.interp(-np.log10(mid_gray), curve, x)            # 数字相纸上中灰需要的曝光
    x_black = float(x[np.argmax(curve >= 0.97 * curve.max())])
    ref = np.asarray(ref_density, float)
    c0 = gray0
    if nominal is None:
        g = ref[1]
    else:
        r0 = np.asarray(nominal, float)
        g = float(np.clip(ref[1], r0[1] - density_limit, r0[1] + density_limit))
    g = max(g, 0.05)
    if stretch is None:
        stretch = float(np.clip((x_black - need) / g, 1.0, stretch_max))   # 让片基尽量接近纯黑
    corr = np.clip(strength * ((ref - ref[1]) - c0), -color_limit / stretch, color_limit / stretch)
    return need + stretch * (g + c0 + corr), np.full(3, stretch)


def frame_auto(paper, Dn, nominal, gray0, strength, stretch_max, mid_gray=0.18, density_limit=0.3,
               latd_normal=0.0, slope_limit=SLOPE_LIMIT, color_limit=COLOR_LIMIT,
               gray_balance=None, chroma_thr=None, shoulder="exponential", highlight_aim=None, shoulder_white=None):
    """按整张底片自动定光 (Frontier 型; Fuji US6668096B1: 由每张的密度直方图求平均密度 / LATD、
    高光侧与暗部侧, 再建立各通道的灰平衡 LUT) → PrintTiming。

    亮度  整张平均密度相对"正常场景"的偏离, 校正不超过 ±density_limit。
    颜色  两点灰平衡: 暗部侧 (绿通道密度最低的 1–5%) 与高光侧 (HL_PCT) 在每个通道上映射到中性
          (以档案的中灰平衡为中性)。每通道一个线性映射 (偏移 + 斜率), 力度 strength 部分校正,
          斜率变化不超过 ±slope_limit, 偏移不超过 color_limit。"""
    if shoulder not in ("exponential", "paper_only"):
        raise ValueError(f"Unknown scan shoulder: {shoulder}")
    gray_balance = GRAY_BALANCE if gray_balance is None else gray_balance
    chroma_thr = GB_CHROMA_THRESHOLD if chroma_thr is None else chroma_thr
    Dn = np.asarray(Dn, float)
    ok = np.all(np.isfinite(Dn), -1)
    g = Dn[ok, 1]
    lo, hi = np.percentile(g, [1, 5] + HL_PCT).reshape(2, 2)
    sh = Dn[ok][(g >= lo[0]) & (g <= lo[1])].mean(0)
    hl = Dn[ok][(g >= hi[0]) & (g <= hi[1])].mean(0)
    c0 = gray0
    span = hl - sh
    full = np.where(np.abs(span) > 0.02, (span[1]) / np.where(np.abs(span) > 0.02, span, 1.0), 1.0)
    r = np.clip(1.0 + strength * (full - 1.0), 1.0 - slope_limit, 1.0 + slope_limit)
    # 颜色偏移的两种依据 (Fuji 专利: 由 LATD 与暗部 / 高光侧共同求灰平衡):
    #   暗部侧中性 —— 夜景里暗部不受灯光颜色影响, 是可靠的中性参考;
    #   LATD 中性 (印放机的积分中性法) —— 日光 / 蓝调里暗部是天光照的阴影, 本身偏蓝。
    lat = -np.log10(np.mean(10.0 ** -(r * Dn[ok]), 0))
    none_l, none_sh = lat - c0, r * sh - c0
    off_l = none_l + np.clip(strength * (lat[1] - none_l), -color_limit, color_limit) - lat
    off_s = none_sh + np.clip(strength * (r[1] * sh[1] - none_sh), -color_limit, color_limit) - r * sh
    wl = 0.5          # 两者各半 (94 张真实扫描的综合误差: 仅暗部 16.9, 仅 LATD 17.7, 各半 16.7)
    off = wl * off_l + (1 - wl) * off_s
    r[1], off[1] = 1.0, 0.0
    if gray_balance in ("us6160634", "us6160634_axis"):
        off, r = gray_balance_us6160634(Dn[ok], sh, hl, c0, color_limit, chroma_thr,
                                        axis=gray_balance == "us6160634_axis")
    Dm = off + r * Dn[ok]
    # 亮度依据: 数码冲印机由密度直方图求平均密度 (Fuji US6668096B1), 不是光学印放机光电池的积分透射。
    # 在 39 张正常曝光的日光 RAW 上, 平均密度相对灰卡的离散 (σ 0.076) 小于 LATD (σ 0.097)。
    latd = Dm.mean(0)
    # 反差: 把暗部侧到高光侧的密度范围放进相纸的再现范围 (相纸曲线 3%–97%), 上下限 CONTRAST_RANGE。
    x, cv = paper.x, paper.curve
    x_hl = float(np.interp(-np.log10(HIGHLIGHT_AIM if highlight_aim is None else highlight_aim), cv, x))
    span_paper = float(x[np.argmax(cv >= 0.97 * cv.max())] - x_hl)
    span_film = max(float(hl[1] - sh[1]), 0.05)
    stretch = float(np.clip(span_paper / span_film, CONTRAST_RANGE[0], CONTRAST_RANGE[1]))
    # 亮度与反差只由绿通道决定; 颜色已在 M 中校正, 这里不再做通道偏移
    K, s = gray_point(paper, latd - latd_normal, np.zeros(3), strength, stretch_max, mid_gray, nominal=nominal,
                      density_limit=density_limit, color_limit=0.0, stretch=stretch)
    K = np.full(3, K[1])
    if shoulder == "paper_only":
        # 独立消融候选: K/s/M 与默认完全相同, 不拟合新白点、不重定光。
        # 相纸趾部仍在, 所以此候选不等于高动态范围输出。
        return PrintTiming(K=K, s=s, M=(off, r))
    # 肩部起点: 高光侧在相纸上的位置, 但不比高光目标更亮
    x_band = float(K[1] - stretch * float(off[1] + r[1] * hl[1]))
    return PrintTiming(K=K, s=s, M=(off, r), x_knee=max(x_band, x_hl),
                       x_white=float(np.interp(-np.log10(SHOULDER_WHITE if shoulder_white is None else shoulder_white), cv, x)))


def gray_balance_us6160634(D, sh, hl, c0, color_limit, chroma_thr, k=(0.2, 0.5, 0.3), axis=False):
    """Fuji US6160634 的灰平衡: 三种估计按 k1:k2:k3 = 0.2:0.5:0.3 加权 (专利示例值)。
    在标准化密度 u = Dn − c0 中, 中性 ⇔ u_r = u_g = u_b; 每通道 u' = o + r·u。
      1. 该胶片类型的平均灰轴 (大量照片统计)  → 不做逐帧校正: o = 0, r = 1
      2. 本帧高光 / 暗部参考点连成的灰轴  → 两点映射到绿通道
      3. 剔除高彩度像素后的平均密度  → 只校正偏移
    估计 1 占 0.2, 总体只校正约 80%, 与 Fuji US7146041 "减弱到 60–80%" 一致。"""
    u = D - c0
    ush, uhl = sh - c0, hl - c0
    span = uhl - ush
    r2 = np.where(np.abs(span) > 0.02, span[1] / np.where(np.abs(span) > 0.02, span, 1.0), 1.0)
    o2 = ush[1] - r2 * ush
    if axis:
        # 专利原文: 标准化 (按该胶片类型的平均灰平衡) 之后, 各通道密度差大的像素是高彩度被摄体 → 相对灰轴度量。
        # 近中性像素太少时 (整幅都是高彩度) 估计 3 不可用 → 不做逐帧偏移 (等同估计 1), 而不是对全部像素取平均。
        chroma = np.max(np.abs(u - u[:, 1:2]), axis=1)
        keep = chroma <= chroma_thr
        o3 = (lambda m: m[1] - m)(u[keep].mean(0)) if keep.sum() > 0.03 * len(u) else np.zeros(3)
    else:
        # 现行 (候选 C): 相对本帧中位色度量, 均匀偏色画面的像素都会被当作近中性
        dev = u - np.median(u, 0)
        chroma = np.max(np.abs(dev - dev[:, 1:2]), axis=1)
        keep = chroma <= chroma_thr
        m3 = u[keep].mean(0) if keep.sum() > 0.03 * len(u) else u.mean(0)
        o3 = m3[1] - m3
    r = k[0] + k[1] * r2 + k[2]
    o = k[1] * o2 + k[2] * o3
    o = np.clip(o, -color_limit, color_limit)
    r[1], o[1] = 1.0, 0.0
    return o - r * c0, r
