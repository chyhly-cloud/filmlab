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
    hit = gammainc(PHOTONS_R, spec.phi()[c] * a * np.asarray(H)[..., None])
    return 1.0 - (1.0 - f) * (1.0 - hit), f


def layer_curves(spec: GrainSpec, c: int, H, reversal=False):
    """名义净密度 D(H) 与 48 µm 孔径 RMS 颗粒度 σ(H) (解析)。
    每亚群: 晶体总数按涂布堆积 (方差 S0 × 期望), 尺寸独立抽取, 显影独立 (二项) →
    方差/面积 = n [E(m²p) − E(mp)² + S0·E(mp)²]。"""
    D = 0.0; var = 0.0
    for n, w, a, v, m in _nodes(spec, c):
        p, f = develop_probability(spec, c, H, a)
        if reversal:
            p, f = 1.0 - p, np.zeros_like(f)
        Emp = (w * m * p).sum(-1); Em2p = (w * m * m * p).sum(-1)
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
