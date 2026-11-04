"""颗粒的银盐物理模型 (T45, docs/tasks/T45_GRAIN.md)。对所有胶片通用, 只由每种胶片档案里的物理量驱动。

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

SIGMA_LN = 0.30                 # 晶体 ECD 的对数正态宽度 (T 颗粒乳剂典型变异系数约 0.3)
QUAD = 5                        # 对数正态的 Gauss–Hermite 求积点数
PHOTONS_R = 4                   # 可显影潜影所需的有效光子数 (潜影中心约 4 个银原子)
ETA_REF = 0.05                  # Φ = η × 波段光子数 的参考 η (吸收率 × 潜影量子效率; 量级估计, 只影响绝对晶体尺寸)
APERTURE_UM2 = np.pi * 24.0 ** 2   # 48 µm 圆孔 (Kodak 漫射 RMS 颗粒度的测量孔径)
# 每 lux·s 每 µm² 的光子数 (400–500 / 500–600 / 600–700 nm; 普朗克谱, 按 CIE 明视觉换算到 lux), 通道顺序 R G B
BAND_PHOTONS = {"BB3200": (6408.0, 3576.0, 1336.0), "D55": (4090.0, 3790.0, 2914.0)}


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


def apply(e, spec: GrainSpec, st, px_um, rng, strength=1.0):
    """在显影后的状态上实现颗粒 (就地修改 st.dye 或 st.silver)。

    每像素的晶体显影状态由该像素的密度 d 反求: 有效曝光 H_eff 使晶体模型的名义净密度 = d (原生冲洗下的
    特性曲线, 与 Kodak σ(D) 的测量条件一致)。显影化学 (冲洗工艺、推拉、DIR、边缘效应) 对哪些晶体显影的影响
    都已在 d 里; d 超过晶体模型的最大密度时, 超出部分视为每晶体的染料量增加 (比例 ρ)。
    每亚群: 像素内晶体总数 (涂布堆积) → 按尺寸分布逐档二项分配 → 每档按显影概率二项显影 → 密度 = Σ N·m / 像素面积。"""
    from scipy.ndimage import gaussian_filter
    tgt = st.dye if e.has_couplers and np.any(st.dye != 0) else st.silver
    if tgt.ndim < 3:
        return
    area = px_um * px_um
    rev = e.design == "reversal"
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
            fluct *= rho * strength * getattr(st, "grain_factor", 1.0)
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
        fluct *= rho * strength * getattr(st, "grain_factor", 1.0)
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
SURFACE_RATIO_3D = 3.0          # 立方体表面积 / T 颗粒两面面积 (同一投影面积): 6a / 2a
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


# ================================================================== T47 ==
# 显式银粒 (docs/tasks/T47_TRIX_SILVER_GRAIN.md)。对所有黑白片通用, 只读胶片档案 (SilverSpec) 里的物理量:
#   ① 卤化银晶体: K 个亚群, ECD 对数正态 (离散为 bins 档), 三维晶体 v = a^1.5, 每亚群涂布银量 coat (g Ag/m²)
#      → 每 µm² 晶体数 n = coat / (每晶体含银 m), m = 3.72e-12 g/µm³ × v (AgBr 中的银)。位置: 泊松 (或 Thomas 团簇)。
#   ② 曝光: 有效光子数 ~ 泊松(φ·a·(a/a0)^β·H), ≥ r 个才可显影 (T46 通用常数); 灰雾 f = 1 − exp(−a/a_f)。
#   ③ 显影: 每个显影晶体在原位变成一团银丝 (丝团): 银质量守恒 —— 丝的总长 ℓ = V_Ag / (π w²/4), V_Ag = m / 10.49 g/cm³;
#      丝在半径 R = √(spread·a/π) 的球内做有持续长度的随机游走 (几个显影中心各长一根), 投影为不透明细丝的并集。
#   ④ 透射: 每个微元 T = exp(−Σ τ_i), τ_i = −ln(1 − g·c_i), c_i 为第 i 颗银丝团对该微元的投影覆盖率;
#      g = 1 (镜面收集) 或 1/Q (漫射收集: 被银散射的光有一部分仍被收集, Callier 系数 Q)。
#      微元透射率经光学 PSF (可选) 在透射率域积分到像素, 最后取 −log10。
#   平均透射率的闭式解 (泊松布尔模型): E[T] = exp(−Σ n_kq p_kq A_kq), A = ∫(1 − T_grain) = g × 丝团投影不透明面积;
#      D = −log10 E[T]。渲染器对每颗银粒的单粒积分 ∫(1 − T) 按面积精确守恒, 所以任何采样间距下宏观密度都等于闭式解。
AGBR_AG_G_UM3 = 3.72e-12        # 每 µm³ AgBr 含银 (g): 6.47 g/cm³ × 108/188
AG_G_UM3 = 10.49e-12            # 金属银密度 (g/µm³)
_FINE_UM = 0.01                 # 银丝团几何的参考栅格 (µm)


@dataclass(frozen=True)
class SilverSpec:
    """一种黑白片的显式银粒物理档案 (T47)。所有量都有物理单位; 见 materials.TRIX_400。"""
    ecd_um: tuple               # (K,) 每亚群晶体投影等效圆直径的中位 (µm)
    sigma_ln: tuple             # (K,) ECD 对数正态宽度
    coat_g_m2: tuple            # (K,) 每亚群涂布银量 (g Ag / m²)
    ei: float                   # 感光度 (净密度 0.1 处 H = 0.8/EI)
    eta: float                  # 每 µm² 投影面积、每 lux·s 的有效光子数 = eta × 光源三波段光子数之和 (含表面吸收系数)
    beta: float                 # 量子效率随投影面积 (a/a0)^β (T46)
    fog_area_um2: float         # 灰雾面积尺度 a_f (T46)
    base_density: float         # 片基 (含防光晕染色) 的密度, 与银无关
    a0_um2: float = 1.0
    photons_r: float = 4.0      # 显影阈值光子数 (T46: 4; 显影工艺参数)
    illuminant: str = "D55"
    spread: float = 4.0         # 银丝团投影面积 / 晶体投影面积 (显影工艺参数)
    filament_um: float = 0.04   # 银丝直径 (显影工艺参数)
    filaments: int = 4          # 每颗晶体的显影中心 (丝) 数
    persistence: float = 0.5    # 银丝持续长度 / 丝团半径
    callier_q: float = 1.4      # 漫射与镜面收集的密度比 (Callier Q)
    bins: int = 9               # 每亚群的尺寸档数 (±2.5 σ)
    shapes: int = 6             # 每档随机丝团形状数
    cluster_mu: float = 0.0     # Thomas 团簇: 每团平均晶体数 (0 = 泊松)
    cluster_um: float = 0.0     # Thomas 团簇: 晶体相对团中心的高斯 σ (µm)
    thickness_um: float | None = None   # T47b: T 颗粒厚度 (体积 a·t); None = 三维晶体 (体积 a^1.5)
    process: str = ""
    source: str = ""

    def with_(self, **kw):
        return replace(self, **kw)

    def phi(self):
        return self.eta * float(np.sum(BAND_PHOTONS.get(self.illuminant, BAND_PHOTONS["D55"])))


def silver_bins(spec: SilverSpec):
    """离散尺寸档: dict(ecd, a, m, V, w, n) 形状 (K, B)。n = 每 µm² 该档晶体数 (含尺寸权重)。"""
    x = np.linspace(-2.5, 2.5, spec.bins)
    w = np.exp(-0.5 * x * x); w /= w.sum()
    ecd = np.asarray(spec.ecd_um, float)[:, None] * np.exp(np.asarray(spec.sigma_ln, float)[:, None] * x[None])
    a = np.pi * ecd ** 2 / 4.0
    m = AGBR_AG_G_UM3 * (a ** 1.5 if spec.thickness_um is None else a * spec.thickness_um)
    n_k = np.asarray(spec.coat_g_m2, float) * 1e-12 / (m * w[None]).sum(1)
    return dict(ecd=ecd, a=a, m=m, V=m / AG_G_UM3, w=np.broadcast_to(w, a.shape), n=n_k[:, None] * w[None])


def silver_develop(spec: SilverSpec, H, a):
    """(..., K, B) 显影概率 (含灰雾)。H: 曝光 (lux·s)。"""
    from scipy.special import gammainc
    f = 1.0 - np.exp(-a / spec.fog_area_um2)
    lam = spec.phi() * a * (a / spec.a0_um2) ** spec.beta * np.asarray(H, float)[..., None, None]
    return 1.0 - (1.0 - f) * (1.0 - gammainc(spec.photons_r, lam))


def _ball_mask(V, ecd, R, w, nf, persist, rng):
    """一颗显影晶体的银丝团 (投影不透明的并集, 参考栅格 _FINE_UM)。返回 (bool 掩膜, 半宽 µm)。
    nf 根丝从晶体投影内的显影中心出发, 每步 w/2, 方向角随机扩散 (持续长度 persist·R), 碰到半径 R 的边界折回;
    丝的总长 = 银体积 / 丝截面 (质量守恒)。"""
    d = _FINE_UM
    half = R + w
    n = int(np.ceil(2 * half / d)) + 1
    mask = np.zeros((n, n), bool)
    step = w / 2.0
    L = V / (np.pi * w * w / 4.0)
    steps = max(int(np.ceil(L / nf / step)), 1)
    r0 = np.sqrt(rng.random(nf)) * ecd / 2.0
    t0 = rng.random(nf) * 2 * np.pi
    p = np.stack([r0 * np.cos(t0), r0 * np.sin(t0)], 1)
    th = rng.random(nf) * 2 * np.pi
    kappa = np.sqrt(2.0 * step / max(persist * R, step))
    pts = np.empty((steps, nf, 2))
    for i in range(steps):
        th = th + kappa * rng.standard_normal(nf)
        q = p + step * np.stack([np.cos(th), np.sin(th)], 1)
        out = np.hypot(q[:, 0], q[:, 1]) > R
        if out.any():
            th[out] = np.arctan2(-p[out, 1], -p[out, 0]) + (rng.random(out.sum()) - 0.5) * np.pi
            q[out] = p[out] + step * np.stack([np.cos(th[out]), np.sin(th[out])], 1)
        p = q
        pts[i] = p
    ij = np.floor((pts.reshape(-1, 2) + half) / d).astype(int)
    rad = int(np.ceil(w / 2 / d))
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            if (dy * dy + dx * dx) * d * d <= (w / 2) ** 2 + 1e-12:
                yy = np.clip(ij[:, 1] + dy, 0, n - 1); xx = np.clip(ij[:, 0] + dx, 0, n - 1)
                mask[yy, xx] = True
    return mask, half


def _geom_key(spec: SilverSpec):
    """银丝团几何只取决于晶体尺寸与显影形态 (与涂布量、曝光、光学无关)。"""
    return replace(spec, coat_g_m2=(1.0,) * len(spec.coat_g_m2), ei=400.0, eta=1.0, beta=0.0, fog_area_um2=1.0,
                   base_density=0.0, photons_r=4.0, callier_q=1.0, cluster_mu=0.0, cluster_um=0.0, process="", source="",
                   illuminant="D55", a0_um2=1.0)


def silver_library(spec: SilverSpec):
    """每档 (k, q) shapes 个随机银丝团 (确定性种子)。返回 dict(masks[k][q] = [(mask, half)], A_spec (K, B) 平均投影不透明面积 µm²)。"""
    return _silver_library(_geom_key(spec))


@lru_cache(maxsize=16)
def _silver_library(spec: SilverSpec):
    b = silver_bins(spec)
    K, B = b["a"].shape
    rng = np.random.default_rng(4747)
    masks, A = [], np.zeros((K, B))
    for k in range(K):
        row = []
        for q in range(B):
            R = np.sqrt(spec.spread * b["a"][k, q] / np.pi)
            lst = [_ball_mask(b["V"][k, q], b["ecd"][k, q], R, spec.filament_um, spec.filaments, spec.persistence, rng)
                   for _ in range(spec.shapes)]
            A[k, q] = np.mean([m.sum() for m, _ in lst]) * _FINE_UM ** 2
            row.append(lst)
        masks.append(row)
    return dict(masks=masks, A_spec=A)


def silver_gain(spec: SilverSpec, geometry="diffuse"):
    """单颗银粒的遮光系数 g: 镜面 1, 漫射 1/Q。"""
    return 1.0 if geometry == "specular" else 1.0 / spec.callier_q


def silver_density(spec: SilverSpec, H, geometry="diffuse"):
    """闭式解: 银层 (不含片基) 的密度 D(H) = log10(e) Σ n p g A (泊松布尔模型 E[T] = exp(−Σ n p g A))。"""
    b = silver_bins(spec)
    A = silver_library(spec)["A_spec"] * silver_gain(spec, geometry)
    p = silver_develop(spec, H, b["a"])
    return np.log10(np.e) * (b["n"] * p * A).sum((-1, -2))


def silver_budget(spec: SilverSpec, geometry="diffuse"):
    """银量审计: 涂布银 (g/m²)、全部显影时的密度上限、覆盖力 (密度 / 显影银 g/m²)。"""
    b = silver_bins(spec)
    A = silver_library(spec)["A_spec"] * silver_gain(spec, geometry)
    coat = float(np.sum(spec.coat_g_m2))
    dmax = float(np.log10(np.e) * (b["n"] * A).sum())
    return dict(coat_g_m2=coat, D_max=dmax, covering_power=dmax / coat)


def silver_developed_mass(spec: SilverSpec, H):
    """(…) 显影银 g/m² (每颗显影晶体的银质量 × 数量)。"""
    b = silver_bins(spec)
    p = silver_develop(spec, H, b["a"])
    return (b["n"] * p * b["m"]).sum((-1, -2)) * 1e12


def _overlap_weights(n_in, d_in, start, d_out):
    """一维面积权重: 输入格 i 覆盖 [start + i·d_in, start + (i+1)·d_in), 输出格 j 覆盖 [j·d_out, (j+1)·d_out)。
    返回 (j0, W[n_out, n_in]), W = 重叠长度 / d_out。"""
    lo = start; hi = start + n_in * d_in
    j0 = int(np.floor(lo / d_out)); j1 = int(np.ceil(hi / d_out))
    e_in = start + np.arange(n_in + 1) * d_in
    e_out = (j0 + np.arange(j1 - j0 + 1)) * d_out
    W = np.clip(np.minimum(e_out[1:, None], e_in[None, 1:]) - np.maximum(e_out[:-1, None], e_in[None, :-1]), 0, None)
    return j0, W / d_out


def silver_stamps(spec: SilverSpec, delta_um: float, nphase: int = 4):
    return _silver_stamps(_geom_key(spec), float(delta_um), int(nphase))


@lru_cache(maxsize=16)
def _silver_stamps(spec: SilverSpec, delta_um: float, nphase: int = 4):
    """把参考栅格上的银丝团按面积精确重采样到间距 delta 的微元网格 (每个亚格相位一份)。
    返回 stamps[k][q] = (oy, ox, cov[nphase, nphase, shapes, h, w]) —— 晶体中心在微元 (0,0) 的亚格相位 (py, px) 时,
    覆盖率 cov 的左上角相对该微元的偏移 (oy, ox)。各相位、各形状对齐到同一窗口。"""
    lib = silver_library(spec)
    out = []
    for row in lib["masks"]:
        r2 = []
        for lst in row:
            n = lst[0][0].shape[0]; half = lst[0][1]
            per = {}
            j0s, j1s = [], []
            for py in range(nphase):
                ph = (py + 0.5) / nphase * delta_um
                j0, W = _overlap_weights(n, _FINE_UM, ph - half, delta_um)
                per[py] = (j0, W)
                j0s.append(j0); j1s.append(j0 + W.shape[0])
            o = min(j0s); size = max(j1s) - o
            cov = np.zeros((nphase, nphase, len(lst), size, size), np.float32)
            for s, (m, _) in enumerate(lst):
                mf = m.astype(np.float64)
                for py in range(nphase):
                    jy, Wy = per[py]
                    t = Wy @ mf
                    for px in range(nphase):
                        jx, Wx = per[px]
                        c = t @ Wx.T                              # 覆盖率 = 重叠面积 / delta²
                        cov[py, px, s, jy - o:jy - o + c.shape[0], jx - o:jx - o + c.shape[1]] = c
            r2.append((o, o, np.clip(cov, 0, 1)))
        out.append(r2)
    return out


SILVER_WORKERS = 1              # 整帧渲染的进程数 (按像素行分块; 结果与进程数无关)


def _tiers(spec: SilverSpec, px_um, sub, count_sub, geometry):
    """每档的位置分辨率: 丝团直径 ≥ 1.5 微元 → 显式 (连续位置 + 丝团形状); 否则按粗格 (px/count_sub) 计数,
    位置分辨到粗格 (粗格内单粒覆盖率 = g·A/δ_B², 精确保持单粒 ∫(1−T))。"""
    b = silver_bins(spec)
    delta = px_um / sub
    dB = px_um / count_sub
    A = silver_library(spec)["A_spec"] * silver_gain(spec, geometry)
    diam = 2 * np.sqrt(spec.spread * b["a"] / np.pi)
    explicit = (diam >= 1.5 * delta) | (A >= 0.5 * dB * dB)
    return b, explicit, A, delta, dB


def _render_rows(spec, H, px_um, seed, geometry, sub, count_sub, nphase, r0, r1, psf_um=0.0, margin_rows=None):
    """像素行 [r0, r1) 的银层平均透射率 (每像素)。晶体按像素行用独立种子生成 (seed, 行号), 所以相邻块重叠的
    边缘行生成的是同一批晶体: 跨块边界的银丝团在两边都按同一位置计入, 分块结果与整体一次渲染相同。
    粗格档: 每亚群每粗格的显影晶体数 ~ Poisson(Σ_q λ_q), 每粒 τ_eff 使 1 − e^(−τ_eff) = 各档按 λ_q 加权的
    1 − e^(−τ_q) (平均透射率精确等于闭式解)。"""
    h, w = H.shape
    b, explicit, A, delta, dB = _tiers(spec, px_um, sub, count_sub, geometry)
    K, B = b["a"].shape
    g = silver_gain(spec, geometry)
    stamps = silver_stamps(spec, float(delta), nphase)
    taus, M = {}, 0
    for k in range(K):
        for q in range(B):
            if explicit[k, q]:
                oy, ox, cov = stamps[k][q]
                taus[k, q] = (oy, -np.log1p(-np.clip(g * cov, 0, 1 - 1e-6)).astype(np.float32))
                M = max(M, -oy, oy + cov.shape[-1])
    M += 1
    cs = count_sub
    bB = 1.0 - np.clip(A / (dB * dB), 0, 1 - 1e-6)                    # 粗格档: 每粒 (1 − 覆盖率) = e^(−τ)
    clus = spec.cluster_mu > 0
    if margin_rows is None:
        margin_rows = int(np.ceil((M / sub) + (4 * spec.cluster_um / px_um if clus else 0))) + 1
    nr = r1 - r0
    Hb, Wb = nr * sub + 2 * M, w * sub + 2 * M
    bufA = np.zeros(Hb * Wb, np.float64)
    bufB = np.zeros((nr * cs, w * cs), np.float64)
    area = px_um * px_um
    ks, qs = np.nonzero(explicit)
    groupsB = [(k, np.nonzero(~explicit[k])[0]) for k in range(K) if (~explicit[k]).any()]
    for r in range(max(0, r0 - margin_rows), min(h, r1 + margin_rows)):
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, r, 0])     # 显式档 (边缘行在相邻块里会重新生成)
        rngB = np.random.default_rng([int(seed) & 0xFFFFFFFF, r, 1])    # 粗格档 (只在本块)
        p = silver_develop(spec, H[r], b["a"])                         # (w, K, B)
        if clus:
            _cluster_row(spec, H, b, p, explicit, taus, -np.log(bB), rng, r, r0, nr, w, sub, cs, nphase, px_um, delta,
                         dB, M, Wb, bufA, bufB)
            continue
        if r0 <= r < r1:                                               # 粗格计数 (只在本像素行内)
            for k, qq in groupsB:
                lq = b["n"][k, qq] * p[:, k, qq] * (area / (cs * cs))  # (w, nq)
                L = lq.sum(1)
                keep = (lq * (1.0 - bB[k, qq])).sum(1) / np.maximum(L, 1e-300)
                tau = -np.log(np.clip(1.0 - keep, 1e-12, 1.0))
                N = rngB.poisson(np.broadcast_to(L[None, :, None], (cs, w, cs)))
                bufB[(r - r0) * cs:(r - r0 + 1) * cs] += (N * tau[None, :, None]).reshape(cs, w * cs)
        if not len(ks):
            continue
        lamA = b["n"][ks, qs] * p[:, ks, qs] * area                   # (w, ne)
        N = rng.poisson(lamA)
        idx, val = [], []
        y_lo = (r - r0) * sub                                          # 本行微元的起始行 (相对缓冲区内部)
        for j, (k, q) in enumerate(zip(ks, qs)):
            G = int(N[:, j].sum())
            if G == 0:
                continue
            x = np.repeat(np.arange(w), N[:, j])
            fy = (r - r0 + rng.random(G)) * sub
            fx = (x + rng.random(G)) * sub
            sh = rng.integers(0, spec.shapes, G)
            ii, vv = _stamp_entries(fy, fx, sh, taus[k, q], nphase, M, Hb, Wb)
            idx.append(ii); val.append(vv)
        if idx:
            idx = np.concatenate(idx); val = np.concatenate(val)
            lo = int(idx.min()) if idx.size else 0
            hi = int(idx.max()) + 1 if idx.size else 0
            if hi > lo:
                bufA[lo:hi] += np.bincount(idx - lo, weights=val, minlength=hi - lo)
    TA = np.exp(-bufA.reshape(Hb, Wb)[M:M + nr * sub, M:M + w * sub])
    T = TA * np.repeat(np.repeat(np.exp(-bufB), sub // cs, 0), sub // cs, 1)
    if psf_um > 0:
        from scipy.ndimage import gaussian_filter
        T = gaussian_filter(T, psf_um / delta, mode="nearest")
    return T.reshape(nr, sub, w, sub).mean((1, 3))


def _stamp_entries(fy, fx, s, tau_entry, nphase, M, Hb, Wb):
    """显式银粒: 按 (形状, 亚格相位) 取丝团的微元 τ → (缓冲区展平下标, τ)。fy, fx: 晶体中心的微元坐标 (连续)。"""
    oy, tau = tau_entry
    hh = tau.shape[-1]
    cy = np.floor(fy).astype(np.int64); cx = np.floor(fx).astype(np.int64)
    py = np.minimum(((fy - cy) * nphase).astype(np.int64), nphase - 1)
    px = np.minimum(((fx - cx) * nphase).astype(np.int64), nphase - 1)
    ok = (cy + oy + M >= 0) & (cy + oy + hh + M <= Hb) & (cx + oy + M >= 0) & (cx + oy + hh + M <= Wb)
    if not ok.all():
        cy, cx, py, px, s = cy[ok], cx[ok], py[ok], px[ok], s[ok]
    ar = np.arange(hh)
    offs = (ar[:, None] * Wb + ar[None, :]).ravel()
    base = (cy + oy + M) * Wb + (cx + oy + M)
    flat = tau.reshape(-1, hh * hh)
    val = flat[(py * nphase + px) * tau.shape[2] + s]
    return (base[:, None] + offs[None, :]).ravel(), val.ravel().astype(np.float64)


def _deposit(buf, Hb, Wb, M, fy, fx, s, tau_entry, nphase):
    """把显式银粒的 τ 累加到 buf (展平)。"""
    step = 1 << 18
    for i in range(0, len(fy), step):
        ii, vv = _stamp_entries(fy[i:i + step], fx[i:i + step], s[i:i + step], tau_entry, nphase, M, Hb, Wb)
        if ii.size:
            lo = int(ii.min()); hi = int(ii.max()) + 1
            buf[lo:hi] += np.bincount(ii - lo, weights=vv, minlength=hi - lo)


def _cluster_row(spec, H, b, p_row, explicit, taus, tB, rng, r, r0, nr, w, sub, cs, nphase, px_um, delta, dB, M,
                 Wb, bufA, bufB):
    """Thomas 团簇: 本像素行的团中心 (泊松, 每 µm² n_tot/μ 个), 每团 Poisson(μ) 个晶体, 相对团中心高斯 σ 偏移;
    晶体按各档数目比例取尺寸, 在它实际所在像素的曝光下独立显影。显式档按丝团沉积, 粗格档计入所在粗格。"""
    h = H.shape[0]
    n = b["n"].ravel(); ntot = n.sum()
    mu, sc = spec.cluster_mu, spec.cluster_um
    Np = rng.poisson(ntot / mu * px_um * px_um, w)
    P = int(Np.sum())
    if P == 0:
        return
    cx = (np.repeat(np.arange(w), Np) + rng.random(P)) * px_um
    cy = (r + rng.random(P)) * px_um
    nd = rng.poisson(mu, P)
    D = int(nd.sum())
    yy = np.repeat(cy, nd) + sc * rng.standard_normal(D)
    xx = np.repeat(cx, nd) + sc * rng.standard_normal(D)
    cls = rng.choice(n.size, D, p=n / ntot)
    iy = np.floor(yy / px_um).astype(np.int64); ix = np.floor(xx / px_um).astype(np.int64)
    ok = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
    yy, xx, cls, iy, ix = yy[ok], xx[ok], cls[ok], iy[ok], ix[ok]
    k, q = np.unravel_index(cls, b["a"].shape)
    pd = silver_develop(spec, H[iy, ix], b["a"])[np.arange(len(k)), k, q]
    dev = rng.random(len(k)) < pd
    yy, xx, k, q = yy[dev], xx[dev], k[dev], q[dev]
    fy = yy / delta - r0 * sub; fx = xx / delta
    for kk, qq in zip(*np.nonzero(explicit)):
        sel = (k == kk) & (q == qq)
        if sel.any():
            _deposit(bufA, bufA.size // Wb, Wb, M, fy[sel], fx[sel], rng.integers(0, spec.shapes, int(sel.sum())),
                     taus[kk, qq], nphase)
    selB = ~explicit[k, q]
    if selB.any():                                   # tB: 每粒 τ (粗格内覆盖率)
        by = np.floor(yy[selB] / dB).astype(np.int64) - r0 * cs; bx = np.floor(xx[selB] / dB).astype(np.int64)
        okB = (by >= 0) & (by < nr * cs) & (bx >= 0) & (bx < w * cs)
        np.add.at(bufB, (by[okB], bx[okB]), tB[k[selB][okB], q[selB][okB]])


_RENDER_ARGS = None


def _render_task(rr):
    spec, H, px_um, seed, geometry, sub, count_sub, nphase = _RENDER_ARGS
    return _render_rows(spec, H, px_um, seed, geometry, sub, count_sub, nphase, rr[0], rr[1])


def render_silver(spec: SilverSpec, H, px_um, seed, geometry="diffuse", sub=16, count_sub=4, nphase=4, psf_um=0.0,
                  rows_per_task=16, workers=None):
    """显式银粒 → 每像素银层平均透射率 (片基除外)。H: (h, w) 每像素曝光 (lux·s); px_um: 像素边长 (µm)。
    微元间距 px/sub, 粗格 px/count_sub (sub 须为 count_sub 的倍数)。psf_um > 0 时在微元透射率上做高斯 PSF
    (仅整块渲染, 用于显微镜视场); 整帧扫描的 PSF 由扫描仪在像素透射率上做 (engine "scan_psf")。"""
    global _RENDER_ARGS
    H = np.asarray(H, float)
    h, w = H.shape
    assert sub % count_sub == 0
    if psf_um > 0 or h <= rows_per_task:
        return _render_rows(spec, H, px_um, seed, geometry, sub, count_sub, nphase, 0, h, psf_um=psf_um)
    tasks = [(i, min(h, i + rows_per_task)) for i in range(0, h, rows_per_task)]
    workers = SILVER_WORKERS if workers is None else workers
    silver_library(spec); silver_stamps(spec, float(px_um / sub), nphase)
    _RENDER_ARGS = (spec, H, px_um, seed, geometry, sub, count_sub, nphase)
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_render_task, tasks, chunksize=1)
    else:
        parts = [_render_task(t) for t in tasks]
    _RENDER_ARGS = None
    return np.concatenate(parts, 0)


def _silver_autocorr(spec: SilverSpec, h_um=0.02):
    return _silver_autocorr_g(_geom_key(spec), h_um)


@lru_cache(maxsize=16)
def _silver_autocorr_g(spec: SilverSpec, h_um=0.02):
    """每档银丝团不透明投影的自相关 γ(r) = ∫ m(u) m(u + r) du (形状平均, µm²), 统一放在间距 h_um 的居中网格。"""
    lib = silver_library(spec)
    f = int(round(h_um / _FINE_UM))
    big = max(m.shape[0] for row in lib["masks"] for lst in row for m, _ in lst)
    n = 2 * big // f + 3
    n += (n + 1) % 2
    out = np.zeros(silver_bins(spec)["a"].shape + (n, n))
    for k, row in enumerate(lib["masks"]):
        for q, lst in enumerate(row):
            acc = 0.0
            for m, _ in lst:
                s = m.shape[0]
                F = np.fft.rfft2(m.astype(float), (2 * s, 2 * s))
                ac = np.fft.fftshift(np.fft.irfft2(F * np.conj(F), (2 * s, 2 * s)))   # 中心在 (s, s)
                c = ac[: 2 * s // f * f, : 2 * s // f * f].reshape(2 * s // f, f, 2 * s // f, f).sum((1, 3))
                acc = acc + c
            acc = acc / len(lst) * _FINE_UM ** 4 / h_um ** 2      # 每 h² 格的平均 γ (µm²)
            c0 = (s // f)                                          # 中心格
            m = acc.shape[0]
            o = n // 2 - c0
            out[k, q, max(o, 0):o + m, max(o, 0):o + m] = acc[max(-o, 0):, max(-o, 0):][: n - max(o, 0), : n - max(o, 0)]
    return out, h_um


def silver_covariance(spec: SilverSpec, H, geometry="diffuse"):
    """透射率场的协方差 (闭式解, 泊松布尔模型): Cov_T(r) = E[T]² (exp(Σ λ_j g² γ_j(r)) − 1), λ_j = n_j p_j。
    返回 (E[T], Cov 网格, 网格间距 µm)。E[T] 为银层 (不含片基)。"""
    gam, h = _silver_autocorr(spec)
    b = silver_bins(spec)
    g = silver_gain(spec, geometry)
    lam = b["n"] * silver_develop(spec, H, b["a"])
    X = np.tensordot(lam, gam, axes=([0, 1], [0, 1])) * g * g
    ET = 10.0 ** -float(silver_density(spec, H, geometry))
    return ET, ET * ET * np.expm1(X), h


def window_variance(cov, h, sides_um, psf_um=0.0, adjacent=True):
    """方孔 (边长 d) ⊛ 高斯 PSF 窗口下孔径平均的方差; adjacent=True 时为 Schmidt 估计器 (相邻方孔差的方差 / 2
    = 方差 − 相邻方孔协方差)。cov: 居中网格 (间距 h)。"""
    from scipy.ndimage import gaussian_filter
    f = max(1, int(round(0.1 / h)))                    # 窗口 ≥ 数 µm: 先按块平均到 ~0.1 µm 网格 (积分不变)
    if f > 1:
        n0 = cov.shape[0]; c0 = n0 // 2
        lo = (c0 - f // 2) % f
        m = (n0 - lo) // f
        cov = cov[lo:lo + m * f, lo:lo + m * f].reshape(m, f, m, f).mean((1, 3))
        h = h * f
    pad = int(np.ceil((4 * psf_um * np.sqrt(2) + max(sides_um) * (2 if adjacent else 1)) / h)) + 2
    C = np.pad(cov, pad)
    if psf_um > 0:
        C = gaussian_filter(C, np.sqrt(2) * psf_um / h, mode="constant", truncate=5.0)
    n = C.shape[0]
    x = (np.arange(n) - n // 2) * h
    out = []
    for d in sides_um:
        L = np.clip(1 - np.abs(x) / d, 0, None) / d * h
        v = L @ C @ L
        if adjacent:
            La = np.clip(1 - np.abs(x - d) / d, 0, None) / d * h
            v = v - La @ C @ L
        out.append(float(v))
    return np.array(out)


def silver_granularity(spec: SilverSpec, H, sides_um, psf_um=0.0, geometry="diffuse", adjacent=True):
    """孔径密度 RMS (闭式解, 透射率→密度线性化 σ_D = σ_T / (E[T] ln10))。"""
    ET, C, h = silver_covariance(spec, H, geometry)
    return np.sqrt(window_variance(C, h, sides_um, psf_um, adjacent)) / (ET * np.log(10))


def silver_structure(ei, U, eta_scale, illuminant="D55", thickness_um=None, scale_tgrain=True):
    """T46 通用乳剂结构 (单层黑白取中间层: λ_k, σ_ln, a_f) + 有效量子效率 η × 表面系数 × eta_scale。
    表面系数: 三维晶体 3 (六面吸附 / T 颗粒两面), T 颗粒 1 (thickness_um 不为 None, 体积 a·t; T47b)。
    eta_scale: T47 对 T46 η 的修正 (显式银粒模型下的有效量子效率; Schmidt Tri-X 标定)。T47b 检验它对三维与 T 颗粒都适用
    (scale_tgrain=False 只用于对照: T 颗粒不乘, 即 T47 原文"只修正三维晶体"的读法)。
    晶体中位投影面积 a_k = (λ_k a0^β / (φ H_s))^(1/(1+β)), H_s = 0.8/EI。返回 (ecd (K,), sigma_ln (K,), eta, a_f)。"""
    lam = np.asarray(U["lam"], float)[1]
    eta = U["eta"] * (SURFACE_RATIO_3D * eta_scale if thickness_um is None else (eta_scale if scale_tgrain else 1.0))
    phi = eta * float(np.sum(BAND_PHOTONS[illuminant]))
    a = (lam * U["a0"] ** U["beta"] / (phi * 0.8 / ei)) ** (1.0 / (1.0 + U["beta"]))
    return 2.0 * np.sqrt(a / np.pi), np.full(3, float(U["sigma_ln"][1])), eta, float(U["fog_area"][1])


def silver_derive(ei, U, eta_scale, H, netD, gross_fog, morph, illuminant="D55", geometry="diffuse", process="", source="",
                  thickness_um=None, scale_tgrain=True):
    """T47 档案: 晶体结构 (silver_structure) + 每亚群涂布银量 (g/m²) 由该胶片自己的特性曲线 (净密度 netD 对曝光 H,
    lux·s; 收集几何 geometry) 非负最小二乘; 片基密度 = 总灰雾 gross_fog − 模型灰雾。morph: spread, filament_um,
    callier_q 等 (SilverSpec 字段)。"""
    from scipy.optimize import nnls
    ecd, sig, eta, af = silver_structure(ei, U, eta_scale, illuminant, thickness_um, scale_tgrain)
    proto = SilverSpec(ecd_um=tuple(map(float, ecd)), sigma_ln=tuple(map(float, sig)), coat_g_m2=(1.0, 1.0, 1.0),
                       ei=float(ei), eta=float(eta), beta=float(U["beta"]), a0_um2=float(U["a0"]), fog_area_um2=af,
                       base_density=0.0, illuminant=illuminant, process=process, source=source,
                       thickness_um=thickness_um, **morph)
    H = np.asarray(H, float)
    cols = []
    for k in range(3):
        one = proto.with_(coat_g_m2=tuple(1.0 if j == k else 0.0 for j in range(3)))
        cols.append(silver_density(one, H, geometry) - silver_density(one, 0.0, geometry))
    coat, _ = nnls(np.stack(cols, 1), np.asarray(netD, float))
    sp = proto.with_(coat_g_m2=tuple(float(c) for c in coat))
    fog = float(silver_density(sp, 0.0, geometry))
    return sp.with_(base_density=float(gross_fog - fog))


def silver_curve(spec: SilverSpec, logH, geometry="diffuse"):
    """银层密度 (含灰雾, 不含片基) 对 log10 H。"""
    return silver_density(spec, 10.0 ** np.asarray(logH, float), geometry)


def apply_silver(e, spec: SilverSpec, st, px_um, seed, strength=1.0):
    """T47: 黑白片的显式银粒 (就地修改 st.silver)。显影化学给出的银密度 d (漫射, 含灰雾) 由同一晶体模型的特性曲线
    反求每像素有效曝光 H_eff; 在 H_eff 下渲染显式银粒 (漫射收集), 每像素密度 = −log10 (像素平均透射率)。
    strength ≠ 1 时按 d + strength·(D_px − d) (颗粒倍数)。"""
    tgt = st.silver
    if tgt.ndim < 3:
        return
    lh = np.linspace(-7.0, 4.0, 2201)
    Dt = silver_curve(spec, lh)
    Dt = np.maximum.accumulate(Dt + 1e-12 * np.arange(len(Dt)))
    d = tgt[..., 1]
    Heff = 10.0 ** np.interp(np.clip(d, Dt[0], Dt[-1]), Dt, lh)
    T = render_silver(spec, Heff, px_um, seed)
    D = -np.log10(np.clip(T, 1e-12, None))
    D = d + strength * (D - d) if strength != 1.0 else D
    for c in range(3):
        tgt[..., c] = D
