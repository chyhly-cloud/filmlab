"""S1 曝光阶段: 光谱曝光积分与胶片内部的光学效应 (散射、光晕)。

约定:
- 图像数组为 (H, W, 3), 通道顺序为乳剂层 R(青染料) G(品红染料) B(黄染料)。
- 曝光量 H 为线性, logH = log10(H)。
显影与冲洗见 chemistry.py。
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from . import spectral
from .stocks import Stock, WAVELENGTHS


# ---------------------------------------------------------------- 曝光 ----
def _spec(light):
    return spectral.illuminant(light) if isinstance(light, str) else np.asarray(light, float)


def _filt(scene_filter):
    return np.ones(81) if scene_filter is None else np.asarray(scene_filter, float)


def exposure_response(stock: Stock, light, scene_filter=None, method="spectral"):
    """场景线性 RGB (日光白平衡) → 三层乳剂曝光量。

    H_layer = Σ_λ 反射率(λ) · I(λ) · F(λ) · 10^logS_layer(λ)
    I 为照明光源的光谱 (名称或 81 点光谱), 反射率由 S0 光谱重建得到 (先按光源颜色除去照明);
    F 为胶片相机镜头前的滤镜 (数码 RAW 拍摄时没有), S 为乳剂光谱感光度。
    method: spectral = 除去照明后重建反射光谱 (日光下最准);
            linear   = 该光源下训练出的线性映射 (人造光源下更稳更准, 见 tools/check_lights.py)。"""
    from . import lights
    sens = 10.0 ** stock.log_sensitivity
    if method.startswith("linear"):          # linear / linear_wp / linear_xyz / linear_wp_xyz (T16)
        return lights.linear_response(_spec(light), _filt(scene_filter)[:, None] * sens,
                                      white_preserving="_wp" in method, space=_space(method))
    return lights.response_under(_spec(light), _filt(scene_filter)[:, None] * sens)


def _space(method):
    """linear 方法名的后缀 → lights.linear_response 的拟合空间: _xyz / _bxyz (T16) / 无 = sRGB。"""
    return "srgb_bxyz" if method.endswith("_bxyz") else ("xyz" if method.endswith("_xyz") else "srgb")


def expose(rgb_lin, response):
    return np.clip(response(rgb_lin), 1e-9, None)


# ------------------------------------------------------------- 光学效应 ----
def _fresnel_unpolarized(theta, n1=1.5, n2=1.0):
    """片基 (n≈1.5) → 空气界面的菲涅耳反射率; 超过临界角为全反射。"""
    s = n1 / n2 * np.sin(theta)
    out = np.ones_like(theta)
    ok = s < 1.0
    ci = np.cos(theta[ok])
    ct = np.sqrt(1.0 - s[ok] ** 2)
    rs = ((n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)) ** 2
    rp = ((n1 * ct - n2 * ci) / (n1 * ct + n2 * ci)) ** 2
    out[ok] = 0.5 * (rs + rp)
    return out


def halation_kernel(px_um, base_um=130.0, n_base=1.5, forward=1.5, r_max_um=1500.0, *, oblique_density=0.0,
                    oblique_w=0.0, return_gain=None, plate=0.0, plate_diffuse=False):
    """片基内反射的光学核: 到达片基的单位漫射光, 回到乳剂时在半径 r 处的分布。

    穿过浑浊乳剂到达片基的漫射光按 cos^forward θ 分布 (1 = 朗伯, 越大越前向)。
    以角度 θ 进入片基的光在背面按菲涅耳反射 (θ > 临界角 ≈ 42° 时全反射),
    回到乳剂时横向偏移 r = 2·t·tanθ。所以光晕是一个环: 半径 ≈ 2·130µm·tan42° ≈ 235µm,
    这是在给定角分布、平面界面与片基厚度假设下的几何结果; 环半径测试只能验证实现自洽,
    不能独立证明真实 CineStill 的传播核。见 docs/HALATION_CHAIN_AUDIT.md。

    return_gain (T15): θ → 返回光斜穿该层时吸收相对垂直吸收的倍数 (film.return_gain), None = 垂直吸收 (原行为)。
    plate (T15): 相机压片板的 (镜面) 反射率。低于临界角、穿出片基背面的光被压片板反射后
      以同一角度回到乳剂 (横向偏移同为 2·t·tanθ, 忽略压片板与片基之间的空气隙), 0 = 不计 (原行为)。
    plate_diffuse (T36): 压片板按朗伯 (哑光) 漫反射处理, 见 _plate_lambert_kernel; False = 上述镜面处理 (逐位不变)。"""
    if plate_diffuse and plate > 0:
        k = halation_kernel(px_um, base_um, n_base, forward, r_max_um, oblique_density=oblique_density,
                            oblique_w=oblique_w, return_gain=return_gain, plate=0.0)
        return k + _plate_lambert_kernel(px_um, base_um, n_base, forward, r_max_um, oblique_density, oblique_w,
                                         return_gain, plate, k.shape[0] // 2)
    if return_gain is None and plate == 0:
        return _halation_kernel_base(px_um, base_um, n_base, forward, r_max_um, oblique_density, oblique_w)
    th = np.linspace(1e-4, np.radians(89.5), 4000)
    r = 2.0 * base_um * np.tan(th)
    ang = np.sin(th) * np.cos(th) ** forward
    ang /= (ang * np.gradient(th)).sum()
    R = _fresnel_unpolarized(th, n_base)
    refl = np.where(R < 1.0, R + (1.0 - R) ** 2 * plate / (1.0 - plate * R), 1.0)
    w = refl * ang * np.gradient(th)
    if oblique_density != 0:
        w = w * oblique_attenuation(th, oblique_density, oblique_w)
    if return_gain is not None:
        w = w * return_gain(th)
    n = int(np.ceil(r_max_um / px_um))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1] * px_um
    rr = np.hypot(yy, xx)
    edges = np.append(r, r[-1] * 1.01)
    ring_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    k = np.interp(rr, r, w / ring_area, right=0.0) * px_um ** 2
    return k * (w[r <= r_max_um].sum() / max(k.sum(), 1e-12))


def _plate_lambert_kernel(px_um, base_um, n_base, forward, r_max_um, oblique_density, oblique_w, return_gain, plate,
                          n_half):
    """T36: 低于临界角穿出片基背面、被哑光 (朗伯) 压片板漫反射回乳剂的光 (片基背面菲涅耳反射 R(θ) 部分仍按镜面留在主核里)。

    下行: 以 θ 到达片基背面的光 (权重 = 主核同一 cos^m 角分布 × 下行斜程 A(θ)), 透出 1 − R(θ), 横移 a = t·tanθ。
    压片板: 漫反射率 ρ, 朗伯 (出射强度 ∝ cosθ_a)。回到片基背面时按菲涅耳 (空气 → 片基) 透入, 其余被背面反射回压片板,
      往返的几何级数 → 透入片基的比例 ρ·T_d / (1 − ρ·R_d), R_d = 朗伯入射的半球反射率, T_d = 1 − R_d。
      空气隙 (压片板压紧, 远小于 t) 忽略: 在空气隙里来回不产生横移。
    回程: 由光学扩展量守恒 (n²·cosθ'·dΩ' = cosθ_a·dΩ_a), 透入片基的光在 θ' ∈ [0, θc] 内按 (1 − R(θ'))·2n²·sinθ'·cosθ' 分布,
      横移 b = t·tanθ', 方位与下行无关 (漫反射); 回到乳剂时该层吸收按 θ' 的斜光程 return_gain(θ')。
    落点 = 两段横移的矢量和 |a + b| (相对方位均匀), 用细网格上两段环分布的二维卷积求径向分布, 再按解析总量栅格化。
    能量: Σ_θ<θc w(θ)·(1 − R(θ)) · ρ·T_d/(1 − ρ·R_d) · <g(θ')>, 不重新归一 (吸收的就是损失)。"""
    from scipy.signal import fftconvolve
    tc = np.arcsin(1.0 / n_base)
    th = np.linspace(1e-4, np.radians(89.5), 4000)
    ang = np.sin(th) * np.cos(th) ** forward
    ang /= (ang * np.gradient(th)).sum()
    R = _fresnel_unpolarized(th, n_base)
    w_out = ang * np.gradient(th) * (1.0 - R) * (th < tc)                     # 透出片基背面
    if oblique_density != 0:
        w_out = w_out * oblique_attenuation(th, oblique_density, oblique_w)
    tp = np.linspace(1e-5, tc - 1e-6, 2000)                                     # 回程角 θ' (片基内)
    Rp = _fresnel_unpolarized(tp, n_base)
    w_ret = (1.0 - Rp) * 2.0 * n_base ** 2 * np.sin(tp) * np.cos(tp) * np.gradient(tp)
    T_d = w_ret.sum()                                                           # 朗伯入射透入片基的比例
    R_d = 1.0 - T_d
    g_ret = return_gain(tp) if return_gain is not None else np.ones_like(tp)
    E = w_out.sum() * plate * T_d / (1.0 - plate * R_d) * (w_ret * g_ret).sum() / T_d
    # 两段横移的环分布 (细网格, 步长 t/130), 二维卷积 → 径向分布
    ds = base_um / 130.0
    a_max = base_um * np.tan(tc)
    m = int(np.ceil(a_max / ds)) + 2
    y, x = np.mgrid[-m:m + 1, -m:m + 1] * ds
    rho_g = np.hypot(y, x)
    def ring(theta, wt):
        rr = base_um * np.tan(theta)
        edges = np.append(rr, rr[-1] * 1.0001)
        area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        return np.interp(rho_g, rr, wt / area, right=0.0)
    sel = th < tc
    conv = fftconvolve(ring(th[sel], w_out[sel]), ring(tp, w_ret * g_ret), mode="full")
    M = conv.shape[0] // 2
    yy, xx = np.mgrid[-M:M + 1, -M:M + 1] * ds
    rc = np.hypot(yy, xx).ravel()
    nb = int(np.ceil(rc.max() / ds)) + 1
    idx = np.minimum((rc / ds).astype(int), nb - 1)
    dens = np.bincount(idx, conv.ravel(), nb) / np.maximum(np.bincount(idx, None, nb), 1)
    r_c = (np.arange(nb) + 0.5) * ds
    r_c[0] = 0.0
    yy, xx = np.mgrid[-n_half:n_half + 1, -n_half:n_half + 1] * px_um
    k = np.interp(np.hypot(yy, xx), r_c, dens, right=0.0)
    return k * (E / max(k.sum(), 1e-30))


def _halation_kernel_base(px_um, base_um, n_base, forward, r_max_um, oblique_density, oblique_w):
    """原 halation_kernel (逐位不变)。"""
    th = np.linspace(1e-4, np.radians(89.5), 4000)
    r = 2.0 * base_um * np.tan(th)
    ang = np.sin(th) * np.cos(th) ** forward
    ang /= (ang * np.gradient(th)).sum()
    w = _fresnel_unpolarized(th, n_base) * ang * np.gradient(th)
    n = int(np.ceil(r_max_um / px_um))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1] * px_um
    rr = np.hypot(yy, xx)
    edges = np.append(r, r[-1] * 1.01)
    ring_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    k = np.interp(rr, r, w / ring_area, right=0.0) * px_um ** 2
    total = w[r <= r_max_um].sum()
    scale = total / max(k.sum(), 1e-12)
    if oblique_density == 0:
        return k * scale
    # Correct raster integration to the attenuated analytic total (T03 v2),
    # never back to the original unattenuated energy.
    w = w * oblique_attenuation(th, oblique_density, oblique_w)
    k = np.interp(rr, r, w / ring_area, right=0.0) * px_um ** 2
    total = w[r <= r_max_um].sum()
    return k * (total / max(k.sum(), 1e-12))


def oblique_attenuation(theta, density, w_above=0.0):
    """Extra oblique optical depth below the scattering point; no fit constants.

    w_above = 0: uniform-depth average (T03). w_above > 0 (T11): that fraction of
    the light was scattered above the layer and crosses it fully (10^-D·u); the
    rest is the uniform-depth average. w_above comes from scatter_above_red()."""
    if not np.isfinite(density) or density < 0:
        raise ValueError("Oblique density must be finite and nonnegative")
    x = density * (1.0 / np.cos(theta) - 1.0) * np.log(10.0)
    uniform = np.divide(-np.expm1(-x), x, out=np.ones_like(x), where=x != 0)
    if w_above == 0:
        return uniform
    return w_above * np.exp(-x) + (1.0 - w_above) * uniform


def scatter_above_red(stock: Stock, v):
    """红感层波段内, 散射发生在红感层之上 (蓝层、绿层) 的比例 (T11)。
    与 emulsion_stack 同一吸收与散射设定, 红层感光度加权, 计入光向下穿过各层的衰减; 不拟合。"""
    S = 10.0 ** stock.log_sensitivity
    A = np.asarray(v.layer_absorbance)[None, :] * S / S.max(0)
    Ay = v.yellow_filter / (1.0 + np.exp((WAVELENGTHS - 505.0) / 9.0))
    turb = 0.15 * (550.0 / WAVELENGTHS) ** 3.0
    aR, aG, aB = A[:, 0] + turb, A[:, 1] + turb, A[:, 2] + turb
    t = lambda d: 10.0 ** -d
    sc = lambda a: (1 - t(a)) * turb / np.maximum(a, 1e-9)          # 该层被散射 (而非吸收) 的比例
    above = sc(aB) + t(aB + Ay) * sc(aG)
    inside = t(aB + Ay + aG) * sc(aR)
    w = S[:, 0]
    return float((above * w).sum() / ((above + inside) * w).sum())


def scatter_above_layers(stock: Stock, v):
    """T36: 按层的 w —— 各感光层波段内, 散射发生在该层之上的比例 (同 scatter_above_red 的叠层设定, 该层感光度加权)。
    红层: = scatter_above_red (之上为蓝层、绿层); 绿层: 之上只有蓝层 (散射) 与黄滤光层 (只吸收、不散射);
    蓝层: 之上没有层, w = 0 (即 T03 的均匀深度平均)。"""
    S = 10.0 ** stock.log_sensitivity
    A = np.asarray(v.layer_absorbance)[None, :] * S / S.max(0)
    Ay = v.yellow_filter / (1.0 + np.exp((WAVELENGTHS - 505.0) / 9.0))
    turb = 0.15 * (550.0 / WAVELENGTHS) ** 3.0
    aG, aB = A[:, 1] + turb, A[:, 2] + turb
    t = lambda d: 10.0 ** -d
    sc = lambda a: (1 - t(a)) * turb / np.maximum(a, 1e-9)
    above, inside = sc(aB), t(aB + Ay) * sc(aG)
    wG = float((above * S[:, 1]).sum() / ((above + inside) * S[:, 1]).sum())
    return scatter_above_red(stock, v), wG, 0.0


def oblique_densities(stock: Stock, v):
    """Vertical density to the base, weighted by each layer's sensitivity."""
    down, _, _ = emulsion_stack(stock, v.layer_absorbance, v.yellow_filter)
    sens = 10.0 ** stock.log_sensitivity
    return -np.log10((down[:, None] * sens).sum(axis=0) / sens.sum(axis=0))


# 乳剂每层的散射消光密度 (550 nm, ∝ λ^-3; 卤化银颗粒散射)。决定光晕中绿层与红层之比 (T24); 默认 0.15 (未定标的原设定)
TURBIDITY = 0.15
# 有色成色剂遮罩在叠层中的位置 (T31, docs/tasks/T31_HALO_COLOUR.md)。None = 原放置 (逐位不变):
#   直接曝光红层前 mask/2、到达片基前 mask、返回绿层前 mask/2、返回蓝层前 mask (绿光共经过 1.5 次)。
# "layers" = 按 Vision3 层结构: 青成色剂的遮罩 (吸收绿、蓝绿; mask 中 ≥ MASK_SPLIT_NM 的部分) 均匀在红感层,
#   品红成色剂的黄色遮罩 (吸收蓝; < MASK_SPLIT_NM 的部分) 均匀在绿感层。返回光先整层穿过红感层才到绿感层 (绿光共 2 次)。
#   两者的光谱分界只影响蓝区 (蓝层光晕本来 ≈ 0), 不影响绿/红之比。由 generation 按代际设置。
MASK_PLACEMENT = None
MASK_SPLIT_NM = 480.0


def emulsion_stack(stock: Stock, layer_absorbance, yellow_filter, turbidity=None, turbidity_power=3.0, path=1.0):
    """彩色胶片的乳剂叠层 (自上而下): 蓝感层 → 黄滤光层 → 绿感层 → 红感层 → 片基。

    光在叠层中被三种东西衰减:
      · 用归一化光谱感光度形状近似感光层吸收; 感光度含潜影/显影响应,
        不能据此识别未显影材料的吸收系数或量子效率, 这里是代理而非光学测量
      · 有色成色剂 (橙色遮罩): 用已冲洗片基密度减去透明片基近似拍摄时的吸收;
        这不是未冲洗乳剂透射率的独立测量 (反转片不使用这个遮罩代理)
      · 卤化银颗粒的散射: 短波长散射强得多 (∝ λ^-p, 颗粒尺寸与波长相近的 Mie 区, p≈2–4)

    返回 (到达片基的透射率 T_down(λ),
          直接曝光时各层吸收的比例 direct(λ, 3),
          反射回来后各层吸收的比例 back(λ, 3))   列顺序 R G B 层。
    反射光从下往上走, 最先遇到红感层, 而蓝绿光在下行与返回中都被遮罩和散射削弱
    —— 模型由此产生偏红的二次曝光。吸收密度、散射与角度依赖仍未独立定标;
    本函数只计算垂直光程, 并非完整的辐射传输求解。
    散射消光与 v.diffuse_fraction 独立, 没有闭合的方向输运能量账本;
    独立的假想材料验证器见 tools/eval/photon_transport.py。"""
    S = 10.0 ** stock.log_sensitivity
    A = np.asarray(layer_absorbance)[None, :] * S / S.max(0)       # (81,3) R G B 层的吸收密度
    Ay = yellow_filter / (1.0 + np.exp((WAVELENGTHS - 505.0) / 9.0))  # 黄滤光层挡蓝光
    mask = np.clip(stock.base_density - stock.base_density.min(), 0, None) if stock.kind == "negative" else 0.0
    turb = (TURBIDITY if turbidity is None else turbidity) * (550.0 / WAVELENGTHS) ** turbidity_power      # 每层散射消光密度
    aR, aG, aB = A[:, 0] + turb, A[:, 1] + turb, A[:, 2] + turb
    if path != 1.0:      # 斜穿: 各层的光程 (密度) 同乘 1/cosθ (T15 return_gain)
        A, Ay, mask, turb = A * path, Ay * path, mask * path, turb * path
    t = lambda d: 10.0 ** -d
    absorbed = lambda a_layer, a_abs: (1 - t(a_layer)) * (a_abs / np.maximum(a_layer, 1e-9))
    # 直接曝光: 只有被感光层"吸收"(而非散射) 的部分形成潜影
    direct = np.stack([t(aB + Ay + aG + mask / 2) * absorbed(aR, A[:, 0]),
                       t(aB + Ay) * absorbed(aG, A[:, 1]),
                       absorbed(aB, A[:, 2])], 1)
    T_down = t(aB + Ay + aG + aR + mask)
    back = np.stack([absorbed(aR, A[:, 0]),
                     t(aR + mask / 2) * absorbed(aG, A[:, 1]),
                     t(aR + aG + Ay + mask) * absorbed(aB, A[:, 2])], 1)
    if MASK_PLACEMENT == "layers" and stock.kind == "negative":
        wm = 1.0 / (1.0 + np.exp((WAVELENGTHS - MASK_SPLIT_NM) / 10.0))
        Mm, Mc = mask * wm, mask * (1.0 - wm)          # 绿感层的黄色遮罩, 红感层的 (偏红) 遮罩
        direct = np.stack([t(aB + Ay + aG + Mm + Mc / 2) * absorbed(aR, A[:, 0]),
                           t(aB + Ay + Mm / 2) * absorbed(aG, A[:, 1]),
                           absorbed(aB, A[:, 2])], 1)
        back = np.stack([t(Mc / 2) * absorbed(aR, A[:, 0]),
                         t(aR + Mc + Mm / 2) * absorbed(aG, A[:, 1]),
                         t(aR + aG + Ay + mask) * absorbed(aB, A[:, 2])], 1)
    elif MASK_PLACEMENT is not None and MASK_PLACEMENT != "layers":
        raise ValueError(f"unknown MASK_PLACEMENT {MASK_PLACEMENT!r}")
    return T_down, direct, back


def halation_response(stock: Stock, light, scene_filter, v, back_reflect: float, method="spectral"):
    """场景线性 RGB → 每层"将以光晕形式返回的曝光" (未含空间核), 与直接曝光同一单位。

    = 光 × 穿透整个乳剂到达片基的比例 × 被散射为漫射光的比例 × 片基背面的反射能力
      × 返回途中被该层吸收的比例; 再按"该层直接吸收"与 datasheet 感光度的比值换算单位。
    back_reflect: 无 remjet 的透明片基 = 1 (菲涅耳 + 全反射); remjet 炭黑背涂 ≈ 0;
      防光晕染料层 = 光往返两次被吸收后的剩余比例。
    输出是感光度标定的曝光单位, 不是入射光能量比例; 不能把三层 H 相加当作能量守恒检查。"""
    from . import lights
    E, F = _spec(light), _filt(scene_filter)
    T_down, direct, back = emulsion_stack(stock, v.layer_absorbance, v.yellow_filter)
    sens = 10.0 ** stock.log_sensitivity
    unit = ((E * F) @ sens) / ((E * F) @ direct)      # 吸收 → datasheet 曝光单位 (按中灰)
    w = F[:, None] * T_down[:, None] * back * v.diffuse_fraction * back_reflect * unit[None, :]
    if method.startswith("linear"):
        return lights.linear_response(E, w, white_preserving="_wp" in method, space=_space(method))
    return lights.response_under(E, w)


def multi_bounce_kernel(px_um, base_um, forward, rho, bounces=4, *, oblique_density=0.0, oblique_w=0.0,
                        return_gain=None, plate=0.0, plate_diffuse=False):
    """多次反射: 光回到乳剂后, 在乳剂正面 (乳剂/空气, 同样 >42° 全反射) 再被反射回片基,
    每多一次往返横向再偏移 2·t·tanθ, 形成 2 倍、3 倍半径的外环。
    rho = 每多一次往返保留的比例 ≈ (该波段穿透整个乳剂的透射率)², 所以外环越来越红。
    近似限制: rho 不依赖入射光谱/角度, 后续界面未逐次乘菲涅耳系数, 返回吸收未按斜光程变化;
    不包含完整的上下行重复吸收及散射输运。见 docs/HALATION_CHAIN_AUDIT.md。"""
    k1 = halation_kernel(px_um, base_um=base_um, forward=forward,
                         r_max_um=1500.0 * bounces, oblique_density=oblique_density, oblique_w=oblique_w,
                         return_gain=return_gain, plate=plate, plate_diffuse=plate_diffuse)
    total = k1.copy()
    for n in range(2, bounces + 1):
        kn = halation_kernel(px_um, base_um=base_um * n, forward=forward, r_max_um=1500.0 * bounces,
                             oblique_density=oblique_density, oblique_w=oblique_w, return_gain=return_gain, plate=plate,
                             plate_diffuse=plate_diffuse)
        total += rho ** (n - 1) * kn
    return total


def layer_halation_kernel(px_um, v, rho=0.0, oblique_density=0.0, oblique_w=0.0, return_gain=None, plate=0.0,
                          plate_diffuse=False):
    """One layer's kernel, shared by picture convolution and gray-card timing."""
    if rho > 0.01:
        return multi_bounce_kernel(px_um, v.base_um, v.forward, rho,
                                   oblique_density=oblique_density, oblique_w=oblique_w,
                                   return_gain=return_gain, plate=plate, plate_diffuse=plate_diffuse)
    return halation_kernel(px_um, base_um=v.base_um, forward=v.forward,
                           oblique_density=oblique_density, oblique_w=oblique_w, return_gain=return_gain, plate=plate,
                           plate_diffuse=plate_diffuse)


def halation(H, halo_source, px_um, v, rho=(0.0, 0.0, 0.0), *, oblique_densities=(0.0, 0.0, 0.0),
             oblique_ws=(0.0, 0.0, 0.0), return_gains=(None, None, None), plate=0.0, plate_diffuse=False):
    """Spread each layer's halo source with its own base-reflection kernel."""
    from scipy.signal import fftconvolve
    out = H.copy()
    for c in range(3):
        if halo_source[..., c].max() > 0:
            k = layer_halation_kernel(px_um, v, rho[c], oblique_densities[c], oblique_ws[c], return_gains[c], plate,
                                      plate_diffuse)
            out[..., c] += fftconvolve(halo_source[..., c], k, mode="same")
    return out


def return_gain(stock: Stock, v, light=None):
    """T15: 返回光以角度 θ 斜穿乳剂时, 各层吸收相对垂直返回的倍数 g_c(θ) (三个函数, 无拟合)。

    halation_response 按垂直光程计算返回吸收 (back), 但进入光晕核的光都以 θ 返回 (全反射部分 θ > 42°),
    在每层走过 1/cosθ 倍的光程。g_c(θ) = Σλ w(λ)·back_c(λ, 1/cosθ) / Σλ w(λ)·back_c(λ, 1),
    w(λ) = 光源 × 到达片基的透射率 (默认胶片参考光源), 即到达片基、将要返回的光的光谱。"""
    E = _spec(light if light is not None else stock.reference_illuminant)
    T_down, _, back1 = emulsion_stack(stock, v.layer_absorbance, v.yellow_filter)
    wl = E * T_down
    den = wl @ back1
    s = np.geomspace(1.0, 1.0 / np.cos(np.radians(89.5)), 200)
    num = np.array([wl @ emulsion_stack(stock, v.layer_absorbance, v.yellow_filter, path=si)[2] for si in s])
    g = num / np.maximum(den, 1e-30)
    return tuple((lambda th, gc=g[:, c]: np.interp(1.0 / np.cos(th), s, gc)) for c in range(3))


def bounce_retention(stock: Stock, v):
    """每层波段内, 光再往返一次穿透整个乳剂后保留的比例 (多次反射用)。"""
    T_down, _, _ = emulsion_stack(stock, v.layer_absorbance, v.yellow_filter)
    S = 10.0 ** stock.log_sensitivity
    return tuple(float((T_down ** 2 * S[:, c]).sum() / S[:, c].sum()) for c in range(3))


def emulsion_scatter(H, px_um, sigma_um=4.0):
    """乳剂内部的光散射 (决定胶片 MTF)。"""
    s = sigma_um / px_um
    if s < 0.3:
        return H
    return gaussian_filter(H, (s, s, 0))


# T41: 乳剂正面散射按 Kodak VISION3 5219 官方 MTF (H-1-5219 Rev 3-26 第 3 页, 3200K / ECN-2 / Status M; 独立数字化与第三方数字化相差 ≤ 2%)。
# 官方曲线除以低频平台 1.03 (显影邻接效应, 模型另由 DIR 抑制剂扩散给出) 后, 按二维指数 PSF ∝ exp(−r/a) (浑浊薄层的扩散近似,
# MTF = (1 + (2πaf)²)^−1.5) 逐层拟合 20–70 c/mm (蓝层 12–70): a = 2.04 / 2.31 / 3.37 µm (R G B 层), log RMS 0.04 / 0.06 / 0.06。
# 原设定 (三层同一高斯 σ = 4 µm) 在 50 c/mm 给 45%, 官方 R/G 为 57% / 50% (/1.03), 蓝层形状不符 (log RMS 0.25)。
# 见 docs/tasks/T41_NEAR_FIELD_GLOW.md。r > 25 µm 的能量 ≤ 5e-3 (蓝层), 不产生灯芯外 25–200 µm 的近场光;
# 在 22.5 µm/像素上只有约 3% (R) / 4% (G) / 10% (B) 落到相邻像素 (原设定 σ/像素 < 0.3 时整项跳过)。
EMULSION_PSF_A_UM = (2.04, 2.31, 3.37)


def emulsion_scatter_kodak(H, px_um, a_um=EMULSION_PSF_A_UM):
    """T41: 逐层二维指数 PSF 的乳剂正面散射 (按像素面积积分, 归一)。a/像素 < 0.1 时 (核几乎全在中心像素) 直接返回。"""
    from scipy.signal import fftconvolve
    out = np.empty_like(H, dtype=float)
    for c in range(H.shape[-1]):
        a = a_um[c] / px_um
        if a < 0.1:
            out[..., c] = H[..., c]
            continue
        n = int(np.ceil(12 * a))
        sub = 5                                    # 每像素 5×5 子采样, 近似像素面积积分
        g = (np.arange(-n * sub - sub // 2, n * sub + sub // 2 + 1) + 0.0) / sub
        yy, xx = np.meshgrid(g, g, indexing="ij")
        k = np.exp(-np.hypot(yy, xx) / a)
        k = k.reshape(2 * n + 1, sub, 2 * n + 1, sub).sum((1, 3))
        k /= k.sum()
        x = np.pad(np.asarray(H[..., c], float), n, mode="reflect")
        out[..., c] = fftconvolve(x, k, mode="valid")
    return out
