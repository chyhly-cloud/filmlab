"""染料云 / κ 银量路径的银盐颗粒物理 (T45 / T46; 原 filmlab/grain.py, docs/tasks/T45_GRAIN.md)。对所有胶片通用, 只由每种胶片档案里的物理量驱动。

链条 (每一层 / 黑白片只有一层):
  1. 卤化银晶体: 3 个亚群 (快 / 中 / 慢), 每个亚群的等效圆直径 ECD 呈对数正态 (σ_ln = SIGMA_LN), 涂布量 A_k
     (该亚群全部显影时贡献的密度)。T 颗粒 (彩色负片) 体积 v = a·t; 三维晶体 (黑白) v = a^1.5。
  2. 曝光: 投影面积为 a 的晶体吸收并有效的光子数 λ = Φ·a·H (泊松), 至少 r 个才形成可显影潜影 →
     P_hit = P(N ≥ r) = gammainc(r, λ)。H 为绝对曝光 (lux·s), 由感光度定: 净密度 0.1 处 H = 0.8 / EI。
     Φ = η × 该层波段的光子数 (光源为胶片设计光源)。灰雾: 与曝光无关的显影概率 f = 1 − exp(−a / a_f)。
     显影概率 p = 1 − (1 − f)(1 − P_hit)。
  3. 显影: 每个显影晶体生成一团染料云 (彩色; 积分密度 m = κ·v, κ 含成色剂产率与 DIR 抑制) 或银粒
     (黑白; m = κ_Ag·v, κ_Ag 由显影银覆盖力定)。
  4. 叠加: 每像素内某尺寸档的晶体数 M (涂布堆积使其方差小于泊松: Var M = S0·M̄, S0 为零波数结构因子),
     其中 Binomial(M, p) 个显影 → 该像素该档的密度 = N·m / 像素面积。方差 Σ A_k m_k [p(1−p) + S0·p²] / 面积
     (Selwyn 面积律, 与渲染分辨率无关)。云的空间尺寸大于 0.3 像素时, 再按云的高斯轮廓分布。
  5. 均值: 模型的显影化学 (chemistry.run: 冲洗工艺、推拉、DIR) 给出的染料 / 银量 d 不变; 晶体模型只决定
     d 由哪些晶体、以什么离散方式组成: 每像素按 d 反求有效曝光 (晶体模型名义密度 = d), 再按光子统计得各档显影概率。

不在输出端加统计噪声: 颗粒加在底片的染料 / 银量上, 之后照常经扫描仪或相纸。

T46 (docs/tasks/T46_GRAIN_GENERAL.md) 在同一链条上加三项, 默认值 (0 / None) 时与 T45 逐位相同:
  2'. 量子效率随晶体尺寸: 每 µm² 投影面积的有效光子数 Φ·(a/a0)^β (小晶体电子–空穴复合使潜影效率下降; β 为跨片种通用常数)。
  3'. 每个显影晶体的染料 / 银产量随机 (伽马分布, 变异系数 yield_cv; 由冲洗工艺定): 每档 N 个显影晶体的总产量
     ~ Gamma(N/cv², m·cv²), 均值 N·m 不变, 方差 N·m²·cv² (复合泊松: 每晶体方差乘以 1 + cv²)。
  3''. 染料云 / 银丝团的空间大小随晶体尺寸: 每个亚群单独展开, 高斯 σ_k = √((ECD_k/4)² + L²)
     (ECD/4: 显影晶体圆盘的二阶矩; L: 氧化显影剂 / 成色剂的扩散长度, 冲洗工艺参数 cloud_L_um)。
  颗粒的位置与数目只来自卤化银晶体 (涂布堆积 → 尺寸 → 光子统计显影); 染料云只决定这一晶体的颜色、产量与边缘大小。
"""
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from .constants import (APERTURE_UM2, BAND_PHOTONS, ETA_REF, PHOTONS_R, QUAD, SIGMA_LN, SURFACE_RATIO_3D,  # noqa: F401
                        check_grain_scale)


@dataclass(frozen=True)
class GrainSpec:
    """一种胶片的颗粒物理参数 (每层 R G B 一行; 黑白片三行相同, 只生成一个场)。"""
    ecd_um: tuple               # (3 层, 3 亚群) 晶体中位等效圆直径 µm
    amount: tuple               # (3, 3) 亚群全部显影时的密度 (模型染料 / 银当量单位)
    kappa: tuple                # (3,) 每 µm³ 显影晶体的积分密度 (密度·µm²/µm³)
    fog_area_um2: tuple         # (3,) 灰雾面积尺度 a_f
    packing_s0: tuple           # (3,) 晶体堆积的零波数结构因子 S0
    ei: float                   # 感光度 (净密度 0.1 处 H = 0.8/EI)
    illuminant: str = "BB3200"  # 设计光源 (决定各层的光子数)
    eta: float = ETA_REF
    thickness_um: float | None = 0.12   # T 颗粒厚度; None = 三维晶体 (v = a^1.5)
    single_layer: bool = False  # 黑白: 只有一层银, 三个通道同一个场
    cloud_um: float = 1.0       # 染料云 / 银粒的空间高斯 σ (µm)
    sigma_ln: tuple = (SIGMA_LN, SIGMA_LN, SIGMA_LN)   # (3,) 每层晶体 ECD 的对数正态宽度
    source: str = ""
    eta_beta: float = 0.0       # T46: 每面积量子效率 ∝ (a/a0)^β (0 = 与尺寸无关, T45)
    eta_a0_um2: float = 1.0     # T46: β 的参考投影面积 (µm²)
    yield_cv: float = 0.0       # T46: 每个显影晶体染料 / 银产量的变异系数 (伽马; 冲洗工艺参数)
    cloud_L_um: float | None = None   # T46: 染料云扩散长度 (µm); None = T45 (整场统一 σ = cloud_um)
    process: str = ""           # T46: 冲洗工艺 (只作记录)

    def phi(self):
        """(3,) 每层每 lux·s、每 µm² 投影面积的有效光子数。黑白全色单层吸收三个波段之和。"""
        b = np.asarray(BAND_PHOTONS[self.illuminant])
        return self.eta * (np.full(3, b.sum()) if self.single_layer else b)

    def with_(self, **kw):
        return replace(self, **kw)


def _nodes(spec: GrainSpec, c: int):
    """第 c 层各亚群的尺寸求积: 返回 [(n_k 每 µm² 晶体数, π_q 数目分数, a_q 面积, v_q 体积)]。
    ECD 按数目呈对数正态; A_k (全部显影的密度) = n_k · Σ π_q m_q。"""
    x, w = np.polynomial.hermite_e.hermegauss(QUAD)
    w = w / w.sum()
    out = []
    for e, amt in zip(spec.ecd_um[c], spec.amount[c]):
        a = np.pi * (e * np.exp(spec.sigma_ln[c] * x) / 2.0) ** 2
        v = a * spec.thickness_um if spec.thickness_um is not None else a ** 1.5
        m = spec.kappa[c] * v
        n = amt / float((w * m).sum())
        out.append((n, w, a, v, m))
    return out


def develop_probability(spec: GrainSpec, c: int, H, a):
    """(..., Q) 面积 a 的晶体在曝光 H (lux·s) 下的显影概率 p 与灰雾概率 f。"""
    from scipy.special import gammainc
    f = 1.0 - np.exp(-a / spec.fog_area_um2[c])
    if spec.eta_beta:
        a_eff = a * (a / spec.eta_a0_um2) ** spec.eta_beta    # 有效吸收截面 (T46)
        hit = gammainc(PHOTONS_R, spec.phi()[c] * a_eff * np.asarray(H)[..., None])
    else:
        hit = gammainc(PHOTONS_R, spec.phi()[c] * a * np.asarray(H)[..., None])
    return 1.0 - (1.0 - f) * (1.0 - hit), f


def layer_curves(spec: GrainSpec, c: int, H, reversal=False):
    """名义净密度 D(H) 与 48 µm 孔径 RMS 颗粒度 σ(H) (解析)。
    每亚群: 晶体总数按涂布堆积 (方差 S0 × 期望), 尺寸独立抽取, 显影独立 (二项), 每晶体产量随机 (cv) →
    方差/面积 = n [E(m²p)(1 + cv²) − E(mp)² + S0·E(mp)²]。"""
    D = 0.0; var = 0.0
    for n, w, a, v, m in _nodes(spec, c):
        p, f = develop_probability(spec, c, H, a)
        if reversal:
            p, f = 1.0 - p, np.zeros_like(f)
        Emp = (w * m * p).sum(-1); Em2p = (w * m * m * p).sum(-1)
        if spec.yield_cv:
            Em2p = Em2p * (1.0 + spec.yield_cv ** 2)        # 每晶体产量随机 (复合泊松)
        D = D + n * ((w * m * (p - f)).sum(-1))
        var = var + n * (Em2p - Emp ** 2 + spec.packing_s0[c] * Emp ** 2)
    return D, np.sqrt(var / APERTURE_UM2)


def speed_points(e, spec: GrainSpec):
    """每层原生特性曲线净密度 0.1 处的 log H (模型的曝光轴)。"""
    x, S = e.log_exposure, e.S_nom
    net = S - S.min(0)
    return np.array([np.interp(0.1, np.maximum.accumulate(net[:, c]), x) for c in range(3)])


def absolute_exposure(e, spec: GrainSpec, logH):
    return 0.8 / spec.ei * 10.0 ** (np.asarray(logH) - speed_points(e, spec))


def _packed_counts(Mbar, s0, shape, rng, target=8.0):
    """每像素晶体数: 均值 Mbar, 孔径内方差 ≈ S0 × 期望数 (涂布堆积的零波数结构因子 S0 < 1, 即比泊松更均匀)。
    像素内期望晶体数少 (大晶体、细像素) 时, 先在 b×b 像素的块上取数 (块内期望 ≥ target, 方差 S0·期望),
    再把块内晶体按均匀随机位置逐级二分到像素 (块内为随机位置)。这样孔径方差与渲染分辨率无关。"""
    h, w = shape
    if Mbar <= 0:                         # 涂布量为 0 的亚群 (NNLS 可给 0): 没有晶体
        return np.zeros(shape, np.int64)
    b = 1 if Mbar >= target else int(2 ** np.ceil(np.log2(np.sqrt(target / max(Mbar, 1e-12)))))
    H, W = -(-h // b), -(-w // b)
    mu = Mbar * b * b
    x = np.maximum(mu + np.sqrt(s0 * mu) * rng.standard_normal((H, W)), 0.0)
    K = (np.floor(x) + (rng.random((H, W)) < x - np.floor(x))).astype(np.int64)
    while K.shape[0] < H * b:            # 逐级二分: 先行后列
        k1 = rng.binomial(K, 0.5); n = np.empty((K.shape[0] * 2, K.shape[1]), np.int64)
        n[0::2], n[1::2] = k1, K - k1
        k1 = rng.binomial(n, 0.5); K = np.empty((n.shape[0], n.shape[1] * 2), np.int64)
        K[:, 0::2], K[:, 1::2] = k1, n - k1
    return K[:h, :w]


def _state_table(spec: GrainSpec, c: int, reversal=False):
    """有效曝光表: log10 H (lux·s) → 名义净密度 D_phys。用于由显影化学给出的密度反求晶体的显影状态。"""
    lh = np.linspace(-7.0, 4.0, 1101)
    return lh, layer_curves(spec, c, 10.0 ** lh, reversal)[0]


def apply_dye_grain(spec: GrainSpec, tgt, px_um, rng, reversal=False, strength=1.0, factor=1.0):
    """在显影后的密度上实现颗粒 (就地修改 tgt: (h, w, 3) 每层染料 / 银量, 模型染料当量单位)。

    每像素的晶体显影状态由该像素的密度 d 反求: 有效曝光 H_eff 使晶体模型的名义净密度 = d (原生冲洗下的
    特性曲线, 与 Kodak σ(D) 的测量条件一致)。显影化学 (冲洗工艺、推拉、DIR、边缘效应) 对哪些晶体显影的影响
    都已在 d 里; d 超过晶体模型的最大密度时, 超出部分视为每晶体的染料量增加 (比例 ρ)。
    每亚群: 像素内晶体总数 (涂布堆积) → 按尺寸分布逐档二项分配 → 每档按显影概率二项显影 → 密度 = Σ N·m / 像素面积。
    reversal: 反转片 (第一显影负像, 染料 = 未显影晶体)。strength: 颗粒倍数; factor: 额外倍数 (filmlab 的 grain_factor)。
    随机数的调用顺序与 filmlab.grain.apply (T54) 完全相同。"""
    from scipy.ndimage import gaussian_filter
    area = px_um * px_um
    rev = reversal
    layers = (1,) if spec.single_layer else (0, 1, 2)
    shape = tgt.shape[:2]
    for c in layers:
        lh, Dt = _state_table(spec, c, rev)
        order = np.argsort(Dt, kind="stable")
        d = np.clip(tgt[..., c], 0, None)
        Dmax = Dt.max()
        Heff = 10.0 ** np.interp(np.minimum(d, Dmax), Dt[order], lh[order])
        rho = np.maximum(d / Dmax, 1.0)
        if spec.cloud_L_um is not None or spec.yield_cv:
            fluct, floor = _layer_fluct_t46(spec, c, Heff, area, shape, rng, rev, px_um)
            fluct *= rho * strength * factor
            if spec.single_layer:
                for k in range(3):
                    tgt[..., k] = np.clip(tgt[..., k] + fluct, floor, None)
            else:
                tgt[..., c] = np.clip(tgt[..., c] + fluct, floor, None)
            continue
        fluct = np.zeros(shape)
        floor = 0.0
        for n, w, a, v, m in _nodes(spec, c):
            p, f = develop_probability(spec, c, Heff, a)
            if rev:
                p, f = 1.0 - p, np.zeros_like(f)
            floor -= n * float((w * m * f).sum())
            M = _packed_counts(n * area, spec.packing_s0[c], shape, rng)      # 该亚群每像素晶体总数
            rest, cum = M, 0.0
            for q in range(len(w)):                                          # 尺寸: 逐档二项分配
                cq = rest if q == len(w) - 1 else rng.binomial(rest, min(w[q] / max(1.0 - cum, 1e-12), 1.0))
                rest = rest - cq; cum += w[q]
                pq = np.clip(p[..., q], 0.0, 1.0)
                N = rng.binomial(cq, pq)
                fluct += (N - n * area * w[q] * pq) * (m[q] / area)
        sig = spec.cloud_um / px_um
        if sig > 0.3:
            fluct = gaussian_filter(fluct, sig)
        fluct *= rho * strength * factor
        # 灰雾晶体的密度在片基里 (模型的 Dmin), 所以染料 / 银量可低于 0, 下限为 −灰雾密度
        if spec.single_layer:
            for k in range(3):
                tgt[..., k] = np.clip(tgt[..., k] + fluct, floor, None)
        else:
            tgt[..., c] = np.clip(tgt[..., c] + fluct, floor, None)


def cloud_sigma_um(spec: GrainSpec, c: int):
    """(3,) 每个亚群的染料云 / 银丝团高斯 σ (µm): 显影晶体圆盘 (σ = ECD/4) 与扩散长度 L 的卷积。"""
    L = spec.cloud_L_um if spec.cloud_L_um is not None else spec.cloud_um
    return np.sqrt((np.asarray(spec.ecd_um[c]) / 4.0) ** 2 + L ** 2)


def _layer_fluct_t46(spec: GrainSpec, c, Heff, area, shape, rng, rev, px_um):
    """T46 的一层: 晶体数 (涂布堆积) → 尺寸档 (二项) → 显影 (二项) → 每晶体产量 (伽马) → 每亚群按自己的云大小展开。"""
    from scipy.ndimage import gaussian_filter
    fluct = np.zeros(shape)
    floor = 0.0
    cv2 = spec.yield_cv ** 2
    sig = cloud_sigma_um(spec, c)
    for k, (n, w, a, v, m) in enumerate(_nodes(spec, c)):
        p, f = develop_probability(spec, c, Heff, a)
        if rev:
            p, f = 1.0 - p, np.zeros_like(f)
        floor -= n * float((w * m * f).sum())
        M = _packed_counts(n * area, spec.packing_s0[c], shape, rng)
        rest, cum = M, 0.0
        fk = np.zeros(shape)
        for q in range(len(w)):
            cq = rest if q == len(w) - 1 else rng.binomial(rest, min(w[q] / max(1.0 - cum, 1e-12), 1.0))
            rest = rest - cq; cum += w[q]
            pq = np.clip(p[..., q], 0.0, 1.0)
            N = rng.binomial(cq, pq)
            if cv2 > 0:
                Y = rng.gamma(N / cv2, m[q] * cv2)            # N 个晶体的总产量: 均值 N·m, 方差 N·m²·cv²
            else:
                Y = N * m[q]
            fk += (Y - n * area * w[q] * pq * m[q]) / area
        s = sig[k] / px_um
        fluct += gaussian_filter(fk, s) if s > 0.3 else fk
    return fluct, floor


# ----------------------------------------------------------------- 档案 --
def fit_amounts(e, ecd_um, kappa, fog_area_um2, packing_s0, ei, illuminant, eta, thickness_um, single_layer=False,
                sigma_ln=(SIGMA_LN,) * 3):
    """给定晶体尺寸, 由该胶片自己的原生特性曲线 (非负最小二乘) 定各亚群涂布量 → GrainSpec。"""
    from scipy.optimize import nnls
    proto = GrainSpec(ecd_um=tuple(map(tuple, ecd_um)), amount=((1.0,) * 3,) * 3, kappa=tuple(kappa),
                      fog_area_um2=tuple(fog_area_um2), packing_s0=tuple(packing_s0), ei=ei, illuminant=illuminant,
                      eta=eta, thickness_um=thickness_um, single_layer=single_layer,
                      sigma_ln=tuple(sigma_ln))
    x, S = e.log_exposure, e.S_nom
    xs = speed_points(e, proto)
    amounts = []
    for c in range(3):
        net = np.maximum.accumulate(S[:, c] - S[:, c].min())
        H = 0.8 / ei * 10.0 ** (x - xs[c])
        cols = []
        for k in range(3):
            one = proto.with_(amount=tuple(tuple(1.0 if (cc == c and kk == k) else 0.0 for kk in range(3))
                                           for cc in range(3)))
            cols.append(layer_curves(one, c, H)[0])
        ok = (net < 0.97 * net.max()) & (net > 0.005)
        amt, _ = nnls(np.stack(cols, 1)[ok], net[ok])
        amounts.append(tuple(float(v) for v in amt))
    return proto.with_(amount=tuple(amounts))


def scaled(base: GrainSpec, e, ei, illuminant, eta, kind="dye", kappa=None, thickness_um="same"):
    """同一乳剂技术、不同感光度的胶片: 晶体尺寸按 a ∝ 1/(Φ·H_s) 换算 (H_s = 0.8/EI), 涂布量由自己的曲线定。"""
    phi0 = base.phi()
    phi1 = base.with_(eta=eta, illuminant=illuminant, single_layer=(kind == "silver")).phi()
    k = np.sqrt((phi0 * 0.8 / base.ei) / (phi1 * 0.8 / ei))
    ecd = np.asarray(base.ecd_um) * k[:, None]
    t = base.thickness_um if thickness_um == "same" else thickness_um
    return fit_amounts(e, ecd, kappa if kappa is not None else base.kappa, base.fog_area_um2, base.packing_s0,
                       ei, illuminant, eta, t, single_layer=(kind == "silver"), sigma_ln=base.sigma_ln)


# ------------------------------------------------------- T46: 跨片种通用推导 --
def exposure_axis(e, ei, design=None):
    """(256, 3) 每层特性曲线横轴对应的绝对曝光 H (lux·s), 由感光度的标准定义定:
    负片 / 黑白 (ISO 5800 / ISO 6 的简化): 每层净密度 0.1 处 H = 0.8/EI (同 T45);
    反转片 (ISO 2240): 绿层 Dmin+0.2 与 Dmin+2.0 两点曝光的几何平均 H_m = 10/EI, 三层共用同一绝对曝光轴。"""
    x = e.log_exposure
    design = design or e.design
    if design != "reversal":
        S = e.S_nom
        xs = np.array([np.interp(0.1, np.maximum.accumulate(S[:, c] - S[:, c].min()), x) for c in range(3)])
        return 0.8 / ei * 10.0 ** (x[:, None] - xs[None, :])
    D = e.stock.density_curves[:, 1]
    pos = D - D.min()
    dec = np.minimum.accumulate(pos[::-1])[::-1] if pos[0] < pos[-1] else np.maximum.accumulate(pos[::-1])[::-1]
    i = np.argsort(dec)
    x_hi = np.interp(0.2, dec[i], x[i]); x_lo = np.interp(min(2.0, 0.9 * dec.max()), dec[i], x[i])
    xm = 0.5 * (x_hi + x_lo)
    return np.repeat((10.0 / ei * 10.0 ** (x - xm))[:, None], 3, 1)


def derive_spec(e, ei, illuminant, U, kind="dye", thickness_um=0.12, process=None, single_layer=None):
    """T46: 只由该胶片的公开物理量 (感光度、设计光源、晶体形态、染料 / 银、特性曲线、冲洗工艺) 与跨片种通用常数 U
    生成卤化银晶体群 → GrainSpec。

    U: lam (3 层 × 3 亚群, 速度点处每个亚群中位晶体吸收的有效光子数; 乳剂结构常数), eta (a0 处每面积量子效率),
       beta (量子效率随面积的指数), a0, sigma_ln (3,), kappa_dye (3,: 青 / 品红 / 黄 每 µm³ 显影晶体的积分密度),
       kappa_ag (银粒), fog_area (3,), s0 (3,)。
    晶体尺寸: Φ_c·η·a·(a/a0)^β·H_s = λ_ck  →  a = (λ_ck·a0^β / (Φ_c·η·H_s))^(1/(1+β))。涂布量由自己的特性曲线 (NNLS) 定。
    process: dict(yield_cv, cloud_L_um, name) 冲洗工艺的最后一步参数。"""
    from scipy.optimize import nnls
    single = (kind == "silver") if single_layer is None else single_layer
    design = e.design
    Hx = exposure_axis(e, ei, design)
    b = np.asarray(BAND_PHOTONS[illuminant], float)
    # 增感染料吸附在晶体表面, 吸收面积 = 表面积: T 颗粒 ≈ 2a (两个大面), 立方体 = 6a → 同一投影面积下三维晶体多吸收 3 倍
    eta = U["eta"] * (SURFACE_RATIO_3D if thickness_um is None else 1.0)
    phi = eta * (np.full(3, b.sum()) if single else b)
    beta, a0 = U["beta"], U["a0"]
    lam = np.asarray(U["lam"], float)
    # 速度点 H_s: 每层净密度 0.1 的曝光 (反转片取同一绝对轴上的第一显影负像净密度 0.1)
    x = e.log_exposure; S = e.S_nom
    Hs = np.array([10 ** np.interp(0.1, np.maximum.accumulate(S[:, c] - S[:, c].min()), np.log10(Hx[:, c]))
                   for c in range(3)])
    a = (lam * a0 ** beta / (phi[:, None] * Hs[:, None])) ** (1.0 / (1.0 + beta))
    ecd = 2.0 * np.sqrt(a / np.pi)
    pr = process or {}
    ks = pr.get("kappa_scale", 1.0)
    kap = (U["kappa_ag"] * ks,) * 3 if kind == "silver" else tuple(k * ks for k in U["kappa_dye"])
    proto = GrainSpec(ecd_um=tuple(map(tuple, ecd)), amount=((1.0,) * 3,) * 3, kappa=kap,
                      fog_area_um2=tuple(U["fog_area"]), packing_s0=tuple(U["s0"]), ei=float(ei), illuminant=illuminant,
                      eta=eta, thickness_um=thickness_um, single_layer=single, sigma_ln=tuple(U["sigma_ln"]),
                      eta_beta=beta, eta_a0_um2=a0, yield_cv=pr.get("yield_cv", 0.0), cloud_L_um=pr.get("cloud_L_um"),
                      process=pr.get("name", ""))
    amounts = []
    for c in range(3):
        net = np.maximum.accumulate(S[:, c] - S[:, c].min())
        cols = []
        for k in range(3):
            one = proto.with_(amount=tuple(tuple(1.0 if (cc == c and kk == k) else 0.0 for kk in range(3))
                                           for cc in range(3)))
            cols.append(layer_curves(one, c, Hx[:, c])[0])
        ok = (net < 0.97 * net.max()) & (net > 0.005)
        amt, _ = nnls(np.stack(cols, 1)[ok], net[ok])
        amounts.append(tuple(float(v) for v in amt))
    return proto.with_(amount=tuple(amounts))


def structure_from(spec: GrainSpec, e):
    """一个已标定档案的乳剂结构常数: 速度点处每亚群中位晶体的有效光子数 λ_ck (= Φ_c·η·a_ck·(a/a0)^β·H_s,c)。"""
    Hx = exposure_axis(e, spec.ei)
    S = e.S_nom
    Hs = np.array([10 ** np.interp(0.1, np.maximum.accumulate(S[:, c] - S[:, c].min()), np.log10(Hx[:, c]))
                   for c in range(3)])
    a = np.pi * (np.asarray(spec.ecd_um) / 2.0) ** 2
    eff = (a / spec.eta_a0_um2) ** spec.eta_beta if spec.eta_beta else 1.0
    return spec.phi()[:, None] * a * eff * Hs[:, None]


# ================================================================== T53 ==
def scale_grain(spec: GrainSpec, k):
    """T53: 颗粒粗细 k 的染料云 / κ 银量档案 (T45 / T46 路径)。每亚群全部显影的密度 (amount) 不变; 每晶体积分密度
    m = κ·v 须 ∝ k² (投影足迹 ×k², 不透明度不变): T 颗粒 v ∝ k² 自然成立 (染料量守恒); 三维晶体 v ∝ k³, κ ×1/k
    (银: 覆盖力 ∝ 1/√a)。云的扩散长度 L (或 T45 的整场 σ) ×k。k = 1 原样返回。"""
    k = check_grain_scale(k)
    if k == 1.0:
        return spec
    kap = tuple(float(x) / k for x in spec.kappa) if spec.thickness_um is None else spec.kappa
    return spec.with_(ecd_um=tuple(tuple(float(e) * k for e in row) for row in spec.ecd_um),
                      fog_area_um2=tuple(float(f) * k * k for f in spec.fog_area_um2), eta=spec.eta / (k * k),
                      eta_a0_um2=spec.eta_a0_um2 * k * k, kappa=kap, cloud_um=spec.cloud_um * k,
                      cloud_L_um=None if spec.cloud_L_um is None else spec.cloud_L_um * k,
                      source=(spec.source + f"; 颗粒粗细 ×{k:g}").lstrip("; "))
