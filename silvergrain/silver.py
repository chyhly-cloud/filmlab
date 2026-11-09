"""显式银粒路径 (T47 / T47b / T50 / T53; 原 filmlab/grain.py)。对所有黑白片通用, 只读胶片档案 (SilverSpec) 里的物理量。"""
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from .constants import (AG_G_UM3, AGBR_AG_G_UM3, BAND_PHOTONS, SURFACE_RATIO_3D, _FINE_UM,  # noqa: F401
                        check_grain_scale)

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
    # 团簇的含义: "crystal" = 晶体在乳剂里成团、各自独立显影 (T47 的对照); "developed" = 显影晶体成团 (T50: 感染显影 /
    # 银聚集, 由显影液决定; cluster_mu = 每团平均显影晶体数, 团中心密度 = 显影晶体密度 / cluster_mu)
    cluster_mode: str = "crystal"
    thickness_um: float | None = None   # T47b: T 颗粒厚度 (体积 a·t); None = 三维晶体 (体积 a^1.5)
    process: str = ""
    source: str = ""
    fine_um: float = _FINE_UM   # T53: 银丝团几何的参考栅格 (µm); 颗粒粗细 k 的三维晶体按 k 缩放 (几何整体放大), 见 scale_silver

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


def _ball_mask(V, ecd, R, w, nf, persist, rng, d=_FINE_UM):
    """一颗显影晶体的银丝团 (投影不透明的并集, 参考栅格 d, 默认 _FINE_UM)。返回 (bool 掩膜, 半宽 µm)。
    nf 根丝从晶体投影内的显影中心出发, 每步 w/2, 方向角随机扩散 (持续长度 persist·R), 碰到半径 R 的边界折回;
    丝的总长 = 银体积 / 丝截面 (质量守恒)。"""
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
            lst = [_ball_mask(b["V"][k, q], b["ecd"][k, q], R, spec.filament_um, spec.filaments, spec.persistence, rng,
                              spec.fine_um)
                   for _ in range(spec.shapes)]
            A[k, q] = np.mean([m.sum() for m, _ in lst]) * spec.fine_um ** 2
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
    S = (b["n"] * p * A).sum((-1, -2))
    if spec.cluster_mu > 0:                       # T50: Thomas 团簇 (簇内重叠浪费覆盖, 密度低于泊松)
        S = _cluster_mean_exponent(spec, S, _cluster_N(spec, b, p))
    return np.log10(np.e) * S


# ------------------------------------------------------------------ T50: Thomas 团簇的闭式解 --
# 显影晶体 = Neyman–Scott / Thomas 过程: 团中心泊松 (密度 ρ = n_tot/μ), 每团 Poisson(μ) 颗晶体, 相对团中心各向同性高斯 σ,
# 每颗按尺寸档比例取尺寸、独立显影 (与 _cluster_row 的显式渲染相同)。布尔足迹 (银丝团, 覆盖率 g·cov) 远小于 σ 时
# (点足迹近似), 由聚类过程的概率生成泛函:
#   E[T] = exp(−ρ ∫ (1 − e^{−Φ(c)}) dc),  Φ(c) = m k(c),  m = μ/N · S (每团期望覆盖面积), S = Σ n p g A, ρ = N/μ
#        = exp(−ρ 2πσ² Ein(x)),  x = m / (2πσ²),  Ein(x) = E1(x) + ln x + γ
#   N = 晶体总数 n_tot (cluster_mode "crystal": 团内晶体独立显影) 或显影晶体密度 Σ n p (cluster_mode "developed": 团由显影晶体
#   组成, 感染显影 / 银聚集; T50 用这一种)。两种情形的概率生成泛函形式相同, 只是 ρ、m 的含义不同。
#   E[T(0) T(r)] / E[T]² = exp(X1(r) + X2(r)):
#     X1 = s · Σ λ g² γ(r)            单颗晶体自身的重叠 (同泊松项), s = ∫ k e^{−2Φ} = (1 − e^{−2x}) / (2x)
#     X2 = ρ (u ⋆ u)(r), u(c) = 1 − e^{−Φ(c)} = 1 − exp(−x e^{−|c|²/2σ²})   同一团内不同晶体的交叉项
# μ → 0 时 x → 0、s → 1、X2 → 0, 回到泊松布尔模型。见 docs/tasks/T50_BW_DEVELOPERS.md。
def _ein(x):
    from scipy.special import exp1
    x = np.asarray(x, float)
    xs = np.maximum(x, 1e-12)
    return np.where(x < 1e-6, x - x * x / 4, exp1(xs) + np.log(xs) + np.euler_gamma)


def _cluster_N(spec: SilverSpec, b, p):
    """团中心密度 × μ: crystal 模式为晶体总数, developed 模式为显影晶体密度 (随 H)。"""
    if spec.cluster_mode == "developed":
        return np.maximum((b["n"] * p).sum((-1, -2)), 1e-30)
    return float(b["n"].sum())


def _cluster_sig2(spec: SilverSpec, b=None):
    """有效团簇方差: σ² + 足迹的空间方差 (丝团半径 R² = spread·a/π, 圆盘每轴方差 R²/4; 按足迹面积加权)。
    点足迹近似在 σ 与足迹同量级时偏差变大 (单元测试), 用高斯近似足迹修正。"""
    b = silver_bins(spec) if b is None else b
    R2 = spec.spread * b["a"] / np.pi
    w = b["n"] * silver_library(spec)["A_spec"]          # 按足迹面积加权 (重叠由面积决定)
    return spec.cluster_um ** 2 + float((w * R2 / 4).sum() / w.sum())


def _cluster_x(spec: SilverSpec, S, N):
    return spec.cluster_mu * np.asarray(S, float) / (np.asarray(N, float) * 2 * np.pi * _cluster_sig2(spec))


def _cluster_mean_exponent(spec: SilverSpec, S, N):
    """−ln E[T] (团簇) = ρ 2πσ² Ein(x) = S · Ein(x)/x。"""
    x = _cluster_x(spec, S, N)
    return np.asarray(S, float) * np.where(x < 1e-9, 1.0, _ein(x) / np.maximum(x, 1e-300))


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
                j0, W = _overlap_weights(n, spec.fine_um, ph - half, delta_um)
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
    if spec.cluster_mode == "developed":
        _cluster_row_developed(spec, H, b, explicit, taus, tB, rng, r, r0, nr, w, sub, cs, nphase, px_um, delta, dB, M,
                               Wb, bufA, bufB)
        return
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


def _cluster_row_developed(spec, H, b, explicit, taus, tB, rng, r, r0, nr, w, sub, cs, nphase, px_um, delta, dB, M,
                           Wb, bufA, bufB):
    """T50 显影晶体成团 (感染显影 / 银聚集): 团中心 ~ 泊松 (每 µm² Σ n p(H)/μ, H 取团中心所在像素), 每团 Poisson(μ) 颗
    显影晶体 (尺寸按 n p(H) 的比例), 相对团中心各向同性高斯 σ; 全部显影 (闭式解 _cluster_covariance 的同一过程)。"""
    h = H.shape[0]
    mu, sc = spec.cluster_mu, spec.cluster_um
    lam = (b["n"][None] * silver_develop(spec, H[r], b["a"])).reshape(w, -1)       # (w, K·B) 每 µm²
    L = lam.sum(1)
    Np = rng.poisson(L / mu * px_um * px_um)
    P = int(Np.sum())
    if P == 0:
        return
    pix = np.repeat(np.arange(w), Np)
    cx = (pix + rng.random(P)) * px_um
    cy = (r + rng.random(P)) * px_um
    nd = rng.poisson(mu, P)
    D = int(nd.sum())
    if D == 0:
        return
    par = np.repeat(np.arange(P), nd)
    yy = cy[par] + sc * rng.standard_normal(D)
    xx = cx[par] + sc * rng.standard_normal(D)
    cum = np.cumsum(lam, 1) / np.maximum(L[:, None], 1e-300)
    cls = np.minimum((cum[pix[par]] < rng.random(D)[:, None]).sum(1), lam.shape[1] - 1)
    iy = np.floor(yy / px_um).astype(np.int64); ix = np.floor(xx / px_um).astype(np.int64)
    ok = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
    yy, xx, cls = yy[ok], xx[ok], cls[ok]
    k, q = np.unravel_index(cls, b["a"].shape)
    fy = yy / delta - r0 * sub; fx = xx / delta
    for kk, qq in zip(*np.nonzero(explicit)):
        sel = (k == kk) & (q == qq)
        if sel.any():
            _deposit(bufA, bufA.size // Wb, Wb, M, fy[sel], fx[sel], rng.integers(0, spec.shapes, int(sel.sum())),
                     taus[kk, qq], nphase)
    selB = ~explicit[k, q]
    if selB.any():
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
    fine = spec.fine_um
    f = max(1, int(round(h_um / fine)))
    if fine != _FINE_UM:                  # T53: 放大的几何 (参考栅格 ≠ 0.01 µm), 网格取栅格的整数倍
        h_um = f * fine
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
            acc = acc / len(lst) * fine ** 4 / h_um ** 2          # 每 h² 格的平均 γ (µm²)
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
    if spec.cluster_mu > 0:
        return _cluster_covariance(spec, H, geometry, X, h, ET)
    return ET, ET * ET * np.expm1(X), h


def _cluster_covariance(spec, H, geometry, X, h, ET, hc=0.1):
    """T50: Thomas 团簇的协方差 (见 _ein 上方的推导), 放在间距 hc (µm) 的居中网格上 (window_variance 本来也块平均到 ~0.1 µm)。"""
    A = silver_library(spec)["A_spec"] * silver_gain(spec, geometry)
    b = silver_bins(spec)
    p = silver_develop(spec, H, b["a"])
    S = float((b["n"] * p * A).sum())
    N = float(_cluster_N(spec, b, p))
    x = float(_cluster_x(spec, S, N))
    s = 1.0 if x < 1e-9 else -np.expm1(-2 * x) / (2 * x)
    sig, rho = np.sqrt(_cluster_sig2(spec, b)), N / spec.cluster_mu
    # 短程项: expm1(s X) 先块平均到 hc
    f = max(1, int(round(hc / h)))
    C1 = np.expm1(s * X)
    n0 = C1.shape[0]; c0 = n0 // 2
    lo = (c0 - f // 2) % f; m = (n0 - lo) // f
    C1 = C1[lo:lo + m * f, lo:lo + m * f].reshape(m, f, m, f).mean((1, 3))
    hc = h * f
    # 长程项: X2 = ρ (u ⋆ u), FFT
    half = max(m // 2 + 1, int(np.ceil(6 * sig / hc)))
    n = 2 * half + 1
    xs = (np.arange(n) - half) * hc
    R2 = xs[:, None] ** 2 + xs[None, :] ** 2
    u = -np.expm1(-x * np.exp(-R2 / (2 * sig * sig)))
    F = np.fft.rfft2(u, (2 * n, 2 * n))
    ac = np.fft.irfft2(F * F, (2 * n, 2 * n))          # u 对称: u ⋆ u = u * u
    X2 = rho * hc * hc * ac[half:half + n, half:half + n]
    out = np.expm1(X2)
    o = half - m // 2
    out[o:o + m, o:o + m] += C1 * np.exp(X2[o:o + m, o:o + m])
    return ET, ET * ET * out, hc


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
    pois = proto.with_(cluster_mu=0.0)
    cols = []
    for k in range(3):
        one = pois.with_(coat_g_m2=tuple(1.0 if j == k else 0.0 for j in range(3)))
        cols.append(silver_density(one, H, geometry) - silver_density(one, 0.0, geometry))
    coat, _ = nnls(np.stack(cols, 1), np.asarray(netD, float))
    if proto.cluster_mu > 0:
        # T50 团簇: 密度对涂布银非线性 (簇内重叠)。目标曲线映射回泊松等效的光学厚度 S_P 再拟合 (映射依赖总晶体数, 迭代)
        ln10 = np.log(10.0)
        b0 = None
        for _ in range(30):
            sp = proto.with_(coat_g_m2=tuple(float(c) for c in coat))
            b0 = silver_bins(sp)
            A0 = silver_library(sp)["A_spec"] * silver_gain(sp, geometry)
            def phi(Hx):                          # S_cl / S_P = Ein(x)/x, x 由当前涂布的显影晶体平均足迹定
                pp = silver_develop(sp, Hx, b0["a"])
                Sx = (b0["n"] * pp * A0).sum((-1, -2)); Nx = _cluster_N(sp, b0, pp)
                return _cluster_mean_exponent(sp, Sx, Nx) / np.maximum(Sx, 1e-300), Sx
            f0, S0 = phi(0.0)
            Scl = float(f0 * S0) + ln10 * np.asarray(netD, float)
            fH, _ = phi(H)
            SP = Scl / fH
            new, _ = nnls(np.stack(cols, 1), (SP - float(S0)) / ln10)
            if np.max(np.abs(new - coat)) < 1e-6 * max(1.0, float(np.max(coat))):
                coat = new
                break
            coat = new
    sp = proto.with_(coat_g_m2=tuple(float(c) for c in coat))
    fog = float(silver_density(sp, 0.0, geometry))
    return sp.with_(base_density=float(gross_fog - fog))


# ================================================================== T53 ==
# 颗粒粗细 k (docs/tasks/T53_GRAIN_COARSENESS.md): 同一胶片的一个假想乳剂, 每个线性颗粒尺寸都乘 k ——
#   晶体 ECD (T 颗粒: 投影面积 ×k², 厚度 0.12 µm 不变)、显影银丝团 / 染料云的大小、显影液成团半径 σ。
# 守恒量 = 每个亚群全部显影时的投影遮光截面 Σ n·g·A (泊松布尔模型的指数; 染料: 积分密度 Σ n·m), 即每颗显影晶体的
#   投影足迹放大 k 倍、不透明度轮廓不变, 每 µm² 的晶体数 n ∝ k⁻²。于是:
#   · 平均密度 (特性曲线) 不变: E[T] = exp(−Σ n p g A) 中 n·A 与 p 都不变;
#   · 透射率协方差是 k = 1 的空间放大 Cov_k(r) = Cov_1(r/k), 所以颗粒结构 ×k, Selwyn σ√A ∝ k;
#   · 银量: 三维晶体每颗银 ∝ k³, 涂布银 ∝ k (k = 1 的覆盖力 ∝ 1/√a, Farnell & Solman 1970; 保持银量不变时密度 ∝ 1/k,
#     特性曲线不能不变); T 颗粒每颗银 ∝ k², 涂布银约不变 (由覆盖截面守恒算出, 见 scale_silver)。
#   · 银丝团: 丝径 ∝ 晶体平均厚度 v/a (三维 ∝ k, T 颗粒不变), 丝团的填充率不变; 三维晶体的丝团就是 k = 1 的丝团整体放大
#     (同一随机序列, 参考栅格 ×k), T 颗粒的丝团按新尺寸重新生成。
#   · 感光度 (唯一的非物理假设): 真实的大晶体捕获更多光子, 按本模型的 λ = φ·a·(a/a0)^β·H 会快 k^(2(1+β)) 倍;
#     这里每颗晶体在每个曝光下的有效光子数保持不变 (η ×k⁻², a0 ×k², 灰雾面积 a_f ×k²), 感光度由用户另设 (EI)。
def _subpop_cover(spec: SilverSpec):
    """(K,) 每个亚群全部显影时的投影遮光截面 Σ_q n_q·A_q (每 µm², 不含收集几何 g)。"""
    return (silver_bins(spec)["n"] * silver_library(spec)["A_spec"]).sum(1)


def scale_silver(spec: SilverSpec, k):
    """T53: 颗粒粗细 k 的显式银粒档案 (见上方说明)。k = 1 原样返回。"""
    k = check_grain_scale(k)
    if k == 1.0:
        return spec
    g3 = spec.thickness_um is None                    # 三维晶体: 丝径与参考栅格 ×k (丝团整体放大)
    sc = spec.with_(ecd_um=tuple(float(e) * k for e in spec.ecd_um), fog_area_um2=spec.fog_area_um2 * k * k,
                    a0_um2=spec.a0_um2 * k * k, eta=spec.eta / (k * k), cluster_um=spec.cluster_um * k,
                    filament_um=spec.filament_um * (k if g3 else 1.0), fine_um=spec.fine_um * (k if g3 else 1.0),
                    source=(spec.source + f"; 颗粒粗细 ×{k:g}").lstrip("; "))
    c1, ck = _subpop_cover(spec), _subpop_cover(sc)
    coat = np.asarray(spec.coat_g_m2, float) * np.where(ck > 0, c1 / np.maximum(ck, 1e-300), 0.0)
    return sc.with_(coat_g_m2=tuple(float(c) for c in coat))


def silver_curve(spec: SilverSpec, logH, geometry="diffuse"):
    """银层密度 (含灰雾, 不含片基) 对 log10 H。"""
    return silver_density(spec, 10.0 ** np.asarray(logH, float), geometry)


def silver_grain_density(spec: SilverSpec, d, px_um, seed, strength=1.0, render=None):
    """T47: 黑白片的显式银粒。d: (h, w) 每像素银层密度 (漫射, 含灰雾, 不含片基), 即显影化学 / 特性曲线给出的平均密度。
    由同一晶体模型的特性曲线反求每像素有效曝光 H_eff; 在 H_eff 下渲染显式银粒 (漫射收集), 返回每像素密度
    −log10 (像素平均透射率)。strength ≠ 1 时按 d + strength·(D_px − d) (颗粒倍数)。
    render: 渲染函数 (默认 render_silver; filmlab 传入自己的包装以保留 SILVER_WORKERS)。与 filmlab.grain.apply_silver (T54) 逐位相同。"""
    render = render_silver if render is None else render
    lh = np.linspace(-7.0, 4.0, 2201)
    Dt = silver_curve(spec, lh)
    Dt = np.maximum.accumulate(Dt + 1e-12 * np.arange(len(Dt)))
    Heff = 10.0 ** np.interp(np.clip(d, Dt[0], Dt[-1]), Dt, lh)
    T = render(spec, Heff, px_um, seed)
    D = -np.log10(np.clip(T, 1e-12, None))
    return d + strength * (D - d) if strength != 1.0 else D
