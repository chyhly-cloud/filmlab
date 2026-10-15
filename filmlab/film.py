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
def _light(scene_illuminant, scene_filter):
    light = spectral.illuminant(scene_illuminant).copy()
    return light * scene_filter if scene_filter is not None else light


def exposure_response(stock: Stock, scene_illuminant: str, scene_filter=None):
    """场景线性 RGB → 三层乳剂曝光量。

    H_layer = Σ_λ 场景反射光谱(λ) · I(λ) · F(λ) · 10^logS_layer(λ)
    场景光谱由 S0 光谱重建得到; I 为拍摄光源, F 为镜头前滤镜, S 为乳剂光谱感光度。"""
    sens = 10.0 ** stock.log_sensitivity
    return spectral.spectral_response(_light(scene_illuminant, scene_filter)[:, None] * sens)


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
    与实拍 CineStill 800T 中测得的 150–270µm 一致 (不是拟合出来的)。"""
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
      · 感光层自身的吸收 ∝ 其光谱感光度 (增感染料吸收什么光, 就对什么光敏感)
      · 有色成色剂 (橙色遮罩): 冲洗前就在乳剂里, 吸收蓝绿光。它的密度就是 datasheet 的
        片基密度减去透明片基 (反转片没有遮罩)
      · 卤化银颗粒的散射: 短波长散射强得多 (∝ λ^-p, 颗粒尺寸与波长相近的 Mie 区, p≈2–4)

    返回 (到达片基的透射率 T_down(λ),
          直接曝光时各层吸收的比例 direct(λ, 3),
          反射回来后各层吸收的比例 back(λ, 3))   列顺序 R G B 层。
    反射光从下往上走, 最先遇到红感层, 而蓝绿光在下行与返回中都被遮罩和散射削弱
    —— 这就是光晕是红色的原因。"""
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


def halation_response(stock: Stock, scene_illuminant: str, scene_filter, v, back_reflect: float):
    """场景线性 RGB → 每层"将以光晕形式返回的曝光" (未含空间核), 与直接曝光同一单位。

    = 光 × 穿透整个乳剂到达片基的比例 × 被散射为漫射光的比例 × 片基背面的反射能力
      × 返回途中被该层吸收的比例; 再按"该层直接吸收"与 datasheet 感光度的比值换算单位。
    back_reflect: 无 remjet 的透明片基 = 1 (菲涅耳 + 全反射); remjet 炭黑背涂 ≈ 0;
                  防光晕染料层 = 光往返两次被吸收后的剩余比例。"""
    light = _light(scene_illuminant, scene_filter)
    T_down, direct, back = emulsion_stack(stock, v.layer_absorbance, v.yellow_filter)
    sens = 10.0 ** stock.log_sensitivity
    unit = (light @ sens) / (light @ direct)          # 吸收 → datasheet 曝光单位 (按中灰)
    w = light[:, None] * T_down[:, None] * back * v.diffuse_fraction * back_reflect * unit[None, :]
    return spectral.spectral_response(w)


def halation(H, halo_source, px_um, v):
    """把每层的光晕源按片基光学核在空间上铺开, 加到曝光量上。"""
    from scipy.signal import fftconvolve
    k = halation_kernel(px_um, base_um=v.base_um, forward=v.forward)
    out = H.copy()
    for c in range(3):
        if halo_source[..., c].max() > 0:
            out[..., c] += fftconvolve(halo_source[..., c], k, mode="same")
    return out


def emulsion_scatter(H, px_um, sigma_um=4.0):
    """乳剂内部的光散射 (决定胶片 MTF)。"""
    s = sigma_um / px_um
    if s < 0.3:
        return H
    return gaussian_filter(H, (s, s, 0))
