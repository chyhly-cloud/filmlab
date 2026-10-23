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
    if method == "linear_wp":
        return lights.linear_response(_spec(light), _filt(scene_filter)[:, None] * sens, white_preserving=True)
    fn = lights.linear_response if method == "linear" else lights.response_under
    return fn(_spec(light), _filt(scene_filter)[:, None] * sens)


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


def halation_kernel(px_um, base_um=130.0, n_base=1.5, forward=1.5, r_max_um=1500.0):
    """片基内反射的光学核: 到达片基的单位漫射光, 回到乳剂时在半径 r 处的分布。

    穿过浑浊乳剂到达片基的漫射光按 cos^forward θ 分布 (1 = 朗伯, 越大越前向)。
    以角度 θ 进入片基的光在背面按菲涅耳反射 (θ > 临界角 ≈ 42° 时全反射),
    回到乳剂时横向偏移 r = 2·t·tanθ。所以光晕是一个环: 半径 ≈ 2·130µm·tan42° ≈ 235µm,
    这是在给定角分布、平面界面与片基厚度假设下的几何结果; 环半径测试只能验证实现自洽,
    不能独立证明真实 CineStill 的传播核。见 docs/HALATION_CHAIN_AUDIT.md。"""
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
    return k * (total / max(k.sum(), 1e-12))


def emulsion_stack(stock: Stock, layer_absorbance, yellow_filter, turbidity=0.15, turbidity_power=3.0):
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
    turb = turbidity * (550.0 / WAVELENGTHS) ** turbidity_power      # 每层散射消光密度
    aR, aG, aB = A[:, 0] + turb, A[:, 1] + turb, A[:, 2] + turb
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
    if method == "linear_wp":
        return lights.linear_response(E, w, white_preserving=True)
    fn = lights.linear_response if method == "linear" else lights.response_under
    return fn(E, w)


def multi_bounce_kernel(px_um, base_um, forward, rho, bounces=4):
    """多次反射: 光回到乳剂后, 在乳剂正面 (乳剂/空气, 同样 >42° 全反射) 再被反射回片基,
    每多一次往返横向再偏移 2·t·tanθ, 形成 2 倍、3 倍半径的外环。
    rho = 每多一次往返保留的比例 ≈ (该波段穿透整个乳剂的透射率)², 所以外环越来越红。
    近似限制: rho 不依赖入射光谱/角度, 后续界面未逐次乘菲涅耳系数, 返回吸收未按斜光程变化;
    不包含完整的上下行重复吸收及散射输运。见 docs/HALATION_CHAIN_AUDIT.md。"""
    k1 = halation_kernel(px_um, base_um=base_um, forward=forward,
                         r_max_um=1500.0 * bounces)
    total = k1.copy()
    for n in range(2, bounces + 1):
        kn = halation_kernel(px_um, base_um=base_um * n, forward=forward, r_max_um=1500.0 * bounces)
        total += rho ** (n - 1) * kn
    return total


def halation(H, halo_source, px_um, v, rho=(0.0, 0.0, 0.0)):
    """把每层的光晕源按片基光学核 (含多次反射) 在空间上铺开, 加到曝光量上。"""
    from scipy.signal import fftconvolve
    out = H.copy()
    for c in range(3):
        if halo_source[..., c].max() > 0:
            k = multi_bounce_kernel(px_um, v.base_um, v.forward, rho[c]) if rho[c] > 0.01 \
                else halation_kernel(px_um, base_um=v.base_um, forward=v.forward)
            out[..., c] += fftconvolve(halo_source[..., c], k, mode="same")
    return out


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
