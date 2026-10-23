"""冲洗化学: 把冲洗工艺写成化学步骤序列, 每一步作用在乳剂状态上。

乳剂状态 (每像素、每层, 单位为染料当量密度):
    latent   潜影 (log H), 显影液只会显影带潜影的颗粒
    halide   尚未显影的卤化银
    silver   金属银
    dye      染料
    coupler  剩余成色剂 (彩色胶片)
    fogged   反转浴之后, 剩余卤化银全部变为可显影

步骤:
    Develop(显影液, 时间, 温度)   黑白显影液只生成银; 彩色显影液生成银 + 等量染料 (受成色剂限制)
    Reversal()                     E-6 反转浴: 化学灰化剩余卤化银
    Bleach(效率)                   银 → 卤化银 (漂白); 效率 < 1 即"漂白旁路"
    Fix()                          溶掉卤化银 (定影)
    RemjetRemoval()                电影胶片的预浴, 去掉炭黑背涂

所以"交叉冲洗""漂白旁路""推冲"都不是特殊效果, 只是把某种胶片放进了某个流程。
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from .materials import Emulsion

# ------------------------------------------------------- 显影动力学 --
# 用 Kodak Portra 800 的标准 / 推一档 / 推两档实测曲线拟合 (C-41 显影 3:15 / 3:45 / 4:15)。
# 注意数据的横轴: 三条曲线在 logH=0 处密度相同 (0.757 / 0.761 / 0.762) —— 横轴是相对各自
# 标称感光度 (EI 800 / 1600 / 3200) 的曝光, 即"欠曝 n 档 + 推 n 档"。拟合前先换回同一物理曝光
# (推 n 档的曲线向右移 0.301·n)。在同一曝光下, 显影更充分只会让密度更高, 不会更低:
#   推一档: 全段密度 ×1.25 (含趾部); 推两档: 趾部 ×2–3, 有真实的感光度提升。
#   S(τ)(x) = S_nom(x + SHIFT·ln τ) · τ^A
#   (更多颗粒被显影 → 感光度提升; 每个显影中心长出更多银 → 密度按比例增加。两项都只增不减)
# τ = 相对显影程度 (时间 × 温度系数 × 显影液活性比)。
# 不建模时残差 0.32D。见 tests/test_system.py。
# ECN-2 胶片进 C-41 (CineStill): τ = 1/0.85 ≈ 1.18 是估计值, 公开资料不能辨识 (见 tools/eval/CRITERIA.md
# "C-41 τ 冻结"): CineStill 2012 对照称密度"几乎相同" (偏向 τ≈1), 2020 文章给出 C-41 / ECN-2 的工艺目标 gamma
# 0.6–0.65 / 0.45–0.55 (偏向 τ≈1.15–1.3)。冻结范围 τ ∈ [1.0, 1.3], 名义值不再调整。
# 每层分别拟合 (R, G, B): 各层乳剂对显影程度的响应不同 (每层残差 0.020 / 0.028 / 0.029 D,
# 三层共用一组参数时为 0.048 D)。绿感层随显影延长增加的反差明显较少。
KIN_A = np.array([1.100, 0.651, 0.965])      # 密度倍增 (τ 的指数)
KIN_SHIFT = np.array([0.807, 1.192, 1.044])  # 感光度提升 (logH / ln τ)
KIN_PIVOT = 0.413      # 黑白显影反差变化的支点 (× 该层最大密度)


# 黑白显影液 (含 E-6 第一显影): 没有 DIR 成色剂, 延长显影让带潜影的弱颗粒也被显影 → 真实的感光度提升。
# 按 Kodak E-6 推冲定义标定: 第一显影 6 → 8 分钟 (τ=1.33) 后欠曝一档的中灰应回到原位。
# 在 Provia 100F / Ektachrome 100 / Velvia 100 上求得 (含反差项), 三者误差 < 0.06 档。
KIN_BW_SHIFT = 0.742
KIN_BW_A = 0.35       # 黑白显影的反差随时间增长较缓 (估计值)


def kinetic_curve(curve: np.ndarray, tau: float, x: np.ndarray = None, kind: str = "color") -> np.ndarray:
    """标准显影的成像曲线 → 显影程度 τ 下的曲线。τ=1 时严格不变。
    color: 彩色显影 (C-41 实测拟合), 曲线沿曝光轴平移 (感光度) 并按比例增加密度 (反差);
    bw:    黑白显影, 曲线沿曝光轴平移 (感光度变化) 并略增反差。"""
    if abs(tau - 1.0) < 1e-9:
        return curve
    if kind == "bw":
        shifted = np.stack([np.interp(x + KIN_BW_SHIFT * np.log(tau), x, curve[:, c]) for c in range(3)], 1)
        dp = KIN_PIVOT * curve.max(0)
        return np.clip(dp + (shifted - dp) * tau ** KIN_BW_A, 0, None)
    shifted = np.stack([np.interp(x + KIN_SHIFT[c] * np.log(tau), x, curve[:, c]) for c in range(3)], 1)
    return shifted * tau ** KIN_A[None, :]


# ----------------------------------------------------------- 显影液 --
@dataclass(frozen=True)
class Developer:
    name: str
    kind: str              # "color" | "bw"
    t_nom: float           # 标准显影时间 (分钟)
    temp_nom: float        # 标准温度 (°C)
    activity: float        # 相对活性 (同一乳剂上的显影能力)
    temp_doubling: float = 6.0   # 温度每升高这么多 °C, 显影速度翻倍 (估计值)
    grain: float = 1.0     # 颗粒倍数 (黑白显影液差异大)
    acutance: float = 0.0  # 边缘效应 (显影液在高密度区局部耗竭)
    agent: str = ""        # 彩色显影剂 (决定染料的化学结构, 见 dye_spectra)


# 时间 / 温度来自公开的 Kodak 工艺说明 (Z-131, Z-119, E-6, D-76 数据); 活性比为估计值
DEVELOPERS = {
    "C-41 CD": Developer("C-41 彩色显影 (CD-4)", "color", 3.25, 37.8, 1.00, agent="CD-4"),
    "ECN-2 CD": Developer("ECN-2 彩色显影 (CD-3)", "color", 3.00, 41.1, 0.85, agent="CD-3"),
    "E-6 FD": Developer("E-6 第一显影 (黑白 PQ)", "bw", 6.00, 38.0, 1.60),
    "E-6 CD": Developer("E-6 彩色显影 (CD-3)", "color", 6.00, 38.0, 0.80, agent="CD-3"),
    "RA-4 CD": Developer("RA-4 相纸彩色显影", "color", 0.75, 35.0, 1.00, agent="CD-3"),
    "D-76": Developer("Kodak D-76 原液", "bw", 6.75, 20.0, 1.00),
    "Rodinal 1+50": Developer("Rodinal 1+50", "bw", 11.0, 20.0, 0.95, grain=1.35, acutance=0.6),
    "HC-110 B": Developer("HC-110 稀释 B", "bw", 5.0, 20.0, 1.03, grain=1.1, acutance=0.2),
    "Xtol": Developer("Xtol 原液", "bw", 6.0, 20.0, 1.00, grain=0.85, acutance=0.1),
}

# 每种设计的乳剂在哪种显影液里形成影像 (S_nom 就是在这个显影液、标准时间下的结果)
NATIVE_IMAGE_DEVELOPER = {"C-41": "C-41 CD", "ECN-2": "ECN-2 CD", "E-6": "E-6 FD",
                          "BW-D76": "D-76", "RA-4": "RA-4 CD"}




def native_dye_agent(e: Emulsion) -> str:
    """乳剂的成色剂是按哪种彩色显影剂设计的 (原生工艺里形成染料的显影液)。"""
    proc = PROCESSES.get(e.native_process)
    if proc is None:
        return ""
    agents = [DEVELOPERS[s.developer].agent for s in proc.steps
              if isinstance(s, Develop) and DEVELOPERS[s.developer].kind == "color"]
    return agents[-1] if agents else ""


def dye_spectra(dye_density: np.ndarray, wavelengths: np.ndarray, native: str, actual: str,
                shift_nm: float, broaden: float, shift_other_nm: float | None = None) -> np.ndarray:
    """成色剂与非设计的显影剂反应: 偶氮甲碱染料的供电子端来自显影剂, 吸收光谱随之改变。
    CD-4 (N-羟乙基) 比 CD-3 (N-甲磺酰胺乙基, 吸电子) 供电子更强 → 吸收峰偏长波。
    实测 (US 5,183,729 表中常规青色成色剂, CD-3 对 CD-4): 青色峰平移 +7…+15 nm, 带宽基本不变
    → shift_nm 用于青色 (峰 > 600 nm), broaden 默认 1。
    品红 / 黄没有找到公开对比数据: 方向按同一供电子机理取正, 幅度 shift_other_nm 为估计值。
    展宽时保持积分吸收 (振子强度) 不变。"""
    if not native or not actual or native == actual:
        return dye_density
    sign = 1.0 if (native, actual) == ("CD-3", "CD-4") else -1.0
    # 带符号的染料谱 (Kodak 原始测量): 负值是被消耗的有色成色剂 (遮罩补偿), 其吸收与显影剂无关, 不随显影剂平移
    coupler = np.minimum(dye_density, 0.0)
    dye_density = dye_density - coupler
    out = np.empty_like(dye_density)
    for c in range(dye_density.shape[1]):
        d = dye_density[:, c]
        peak = wavelengths[np.argmax(d)]
        sh = shift_nm if (peak > 600 or shift_other_nm is None) else shift_other_nm
        src = peak + (wavelengths - peak - sign * sh) / broaden
        new = np.interp(src, wavelengths, d, left=d[0], right=d[-1])
        out[:, c] = new * d.sum() / max(new.sum(), 1e-12)
    return out + coupler


# -------------------------------------------------------------- 步骤 --
@dataclass(frozen=True)
class Develop:
    developer: str
    minutes: float | None = None   # None = 标准时间
    temp: float | None = None


@dataclass(frozen=True)
class Reversal:
    pass


@dataclass(frozen=True)
class Bleach:
    efficiency: float = 1.0


@dataclass(frozen=True)
class Fix:
    pass


@dataclass(frozen=True)
class RemjetRemoval:
    pass


@dataclass(frozen=True)
class Process:
    name: str
    steps: tuple
    note: str = ""

    def push(self, stops: float) -> "Process":
        """推 / 拉冲: 延长 / 缩短成像显影步骤 (第一个 Develop)。
        C-41: 每档 +30 秒 (Kodak); E-6: 第一显影 每档约 +2 分钟; 黑白: 每档时间 ×1.4。"""
        steps = list(self.steps)
        i = next(k for k, s in enumerate(steps) if isinstance(s, Develop))
        d = steps[i]
        dev = DEVELOPERS[d.developer]
        t = d.minutes or dev.t_nom
        if d.developer in ("C-41 CD", "ECN-2 CD"):
            t = t + 0.5 * stops
        elif d.developer == "E-6 FD":
            t = t + 2.0 * stops
        else:
            t = t * 1.4 ** stops
        steps[i] = Develop(d.developer, max(t, 0.3), d.temp)
        return Process(f"{self.name} {'push' if stops > 0 else 'pull'} {abs(stops):g}", tuple(steps))

    def with_bleach(self, efficiency: float) -> "Process":
        """漂白旁路 (efficiency=0) / 部分漂白。"""
        steps = tuple(Bleach(efficiency) if isinstance(s, Bleach) else s for s in self.steps)
        return Process(f"{self.name} bleach {efficiency:.0%}", steps)


PROCESSES = {
    "C-41": Process("C-41", (Develop("C-41 CD"), Bleach(), Fix()), "彩色负片"),
    "ECN-2": Process("ECN-2", (RemjetRemoval(), Develop("ECN-2 CD"), Bleach(), Fix()), "电影负片"),
    "E-6": Process("E-6", (Develop("E-6 FD"), Reversal(), Develop("E-6 CD"), Bleach(), Fix()), "彩色反转片"),
    "BW-D76": Process("D-76", (Develop("D-76"), Fix()), "黑白负片"),
    "BW-Rodinal": Process("Rodinal 1+50", (Develop("Rodinal 1+50"), Fix()), "黑白负片, 锐利粗颗粒"),
    "BW-HC110": Process("HC-110 B", (Develop("HC-110 B"), Fix()), "黑白负片"),
    "BW-Xtol": Process("Xtol", (Develop("Xtol"), Fix()), "黑白负片, 细颗粒"),
    "RA-4": Process("RA-4", (Develop("RA-4 CD"), Bleach(), Fix()), "彩色相纸 (漂定合一)"),
}


# -------------------------------------------------------------- 状态 --
@dataclass
class FilmState:
    latent: np.ndarray             # (..., 3) log H
    halide: np.ndarray
    silver: np.ndarray
    dye: np.ndarray
    coupler: np.ndarray
    fogged: bool = False
    latent_used: bool = False
    remjet: bool = False
    image_silver: np.ndarray = None   # 成像显影形成的银 (决定颗粒)
    grain_factor: float = 1.0
    dye_agent: str = ""               # 形成染料的显影剂
    log: list = field(default_factory=list)

    def density(self, silver_covering: float):
        """最终每层的 (染料密度, 中性银密度, 残留卤化银造成的乳白)。"""
        return self.dye, self.silver * silver_covering, self.halide


def initial_state(e: Emulsion, logH: np.ndarray) -> FilmState:
    z = np.zeros_like(logH)
    return FilmState(latent=logH, halide=z + e.halide_cap, silver=z.copy(), dye=z.copy(),
                     coupler=z + e.coupler_cap, remjet=e.remjet)


# -------------------------------------------------------------- 执行 --
def _dir_matrix(amount, negative=True):
    """A[i, j]: 第 i 层显影释放的抑制剂对第 j 层的作用。
    标定目标: Kodak US 6,210,870 实测 C-41 负片 "分色曝光 gamma / 中性曝光 gamma":
    R 1.15–1.49, G 1.42–1.64, B 1.29–1.50 (绿层受到的层间效应最强)。
    这组值给出 Portra 800 [1.31 1.45 1.33]; 500T(C-41) 绿层受肩部限制为 1.31。
    R↔B 不相邻 (中间隔着绿层), 扩散距离更远 → 打折。
    反转片没有对应实测 (且第一显影是黑白显影), 保持原来较弱的估计值。"""
    if not negative:
        A = np.full((3, 3), 0.30 * amount)
        np.fill_diagonal(A, 0.15 * amount)
        A[2, 0] *= 0.5
        A[0, 2] *= 0.5
        return A
    A = np.full((3, 3), 0.60 * amount)
    A[:, 1] = 0.80 * amount
    np.fill_diagonal(A, 0.15 * amount)
    A[2, 0] *= 0.75
    A[0, 2] *= 0.75
    return A


def _curves_at(x, xs, curves):
    out = np.empty_like(x)
    for c in range(3):
        out[..., c] = np.interp(x[..., c], xs, curves[:, c])
    return out


def _intrinsic_curve(e: Emulsion, S: np.ndarray, A: np.ndarray):
    """厂商曲线已包含 DIR 抑制。反推"无抑制时"的曲线 C, 使中性灰阶被抑制后正好回到 S。"""
    x = e.log_exposure
    x0 = np.maximum.accumulate(x[:, None] - (S / e.halide_cap) @ A, axis=0)
    return np.stack([np.interp(x, x0[:, c], S[:, c]) for c in range(3)], 1)


def develop_image(e: Emulsion, logH, tau, inhib_amount, blur_px, kind="color"):
    """成像显影: 只有带潜影的颗粒被显影。包含动力学 (τ) 与 DIR / 边缘效应。"""
    x = e.log_exposure
    S = kinetic_curve(e.S_nom, tau, x, kind)
    if inhib_amount <= 0:
        return _curves_at(logH, x, S)
    A = _dir_matrix(inhib_amount, e.design == "negative")
    if e.design in ("bw", "paper"):     # 单层银影像: 只有同层的边缘效应
        A = np.diag(np.diag(A))
    C = _intrinsic_curve(e, S, A)
    first = _curves_at(logH, x, S) / e.halide_cap
    if blur_px > 0.3:   # 抑制剂在乳剂中扩散
        first = gaussian_filter(first, (blur_px, blur_px, 0)) if first.ndim == 3 else first
    return _curves_at(logH - first @ A, x, C)


def run(e: Emulsion, process: Process, logH: np.ndarray, blur_px: float = 0.0,
        tau_factor: float = 1.0) -> FilmState:
    """tau_factor: 这一次冲洗实际的显影程度偏差 (时间 / 温度 / 药液活性公差, 见 variation.py)。"""
    st = initial_state(e, logH)
    native_dev = DEVELOPERS[NATIVE_IMAGE_DEVELOPER[e.native_process]]
    for step in process.steps:
        if isinstance(step, RemjetRemoval):
            st.remjet = False
        elif isinstance(step, Develop):
            dev = DEVELOPERS[step.developer]
            minutes = step.minutes or dev.t_nom
            temp = dev.temp_nom if step.temp is None else step.temp
            if st.fogged:
                # 反转浴之后: 剩余卤化银全部可显影, 显影到底
                dS = st.halide.copy()
                st.log.append(f"{dev.name}: 显影全部剩余卤化银")
            elif not st.latent_used:
                tau = (minutes / dev.t_nom) * 2.0 ** ((temp - dev.temp_nom) / dev.temp_doubling) \
                    * dev.activity / native_dev.activity * tau_factor
                inhib = e.dir_amount if dev.kind == "color" or e.design != "negative" else 0.3 * e.dir_amount
                inhib += dev.acutance
                dS = np.minimum(develop_image(e, logH, tau, inhib, blur_px * (1 + 2 * dev.acutance), dev.kind),
                                st.halide)
                st.latent_used = True
                st.image_silver = dS
                st.grain_factor = dev.grain
                st.log.append(f"{dev.name}: {minutes:.2f} 分钟 @ {temp:.1f}°C, 显影程度 τ={tau:.2f}")
            else:
                st.log.append(f"{dev.name}: 没有可显影的潜影")
                continue
            st.halide = st.halide - dS
            st.silver = st.silver + dS
            if dev.kind == "color":
                new_dye = np.minimum(dS, st.coupler)
                st.dye = st.dye + new_dye
                if np.any(new_dye > 0):
                    st.dye_agent = dev.agent
                st.coupler = st.coupler - new_dye
                if not e.has_couplers:
                    st.log.append("  乳剂没有成色剂 → 不生成染料")
        elif isinstance(step, Reversal):
            st.fogged = True
            st.log.append("反转: 剩余卤化银被化学灰化")
        elif isinstance(step, Bleach):
            back = st.silver * step.efficiency
            st.silver = st.silver - back
            st.halide = st.halide + back
            st.log.append(f"漂白: {step.efficiency:.0%} 的银变回卤化银")
        elif isinstance(step, Fix):
            st.halide = np.zeros_like(st.halide)
            st.log.append("定影: 溶去卤化银")
    return st
