"""由胶片的公开物理量与通用常数推导卤化银晶体群 (T46 / T47 / T47b / T50 / T53): Emulsion = derive(film, developer, grain_scale)。

彩色片 (染料云, T46): 晶体尺寸 a = (λ_ck a0^β / (Φ_c η H_s))^(1/(1+β)) (λ_ck: 5219 标定的乳剂结构, η、β 跨片种通用,
  H_s = 0.8/EI 由感光度定, Φ_c 由设计光源定), 每亚群涂布量由该片自己的特性曲线 (非负最小二乘) 定; 冲洗工艺给出每晶体
  产量 CV、扩散长度 L、κ 倍数。= filmlab.materials.grain_spec_t46。
黑白片 (显式银粒, T47 / T47b): 同一晶体结构 (中间层), 有效量子效率 × 表面系数 (三维 3) × 6.61 (T47 标定, T47b 对 T 颗粒同样适用),
  每亚群涂布银量 (g/m²) 由该片自己的 D-76 类曲线定; 银丝团形态为 D-76 类先验。显影液 (T50) 只改显影阈值 r 与显影晶体成团。
  = filmlab.materials.silver_spec (代际 t50_bw / t51_bw 的 "<片>@<显影液>")。
颗粒粗细 k (T53): 所有颗粒线尺度 ×k 的假想乳剂 (投影光学截面守恒, 曲线不变; 感光度不随 k 是唯一的非物理假设)。
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import database as db
from . import dye, silver
from .constants import ETA_REF, check_grain_scale
from .films import Developer, Film, _colour_curve, developer as _developer, film as _film


# ------------------------------------------------------------ 通用常数 --
@lru_cache(maxsize=None)
def base_5219():
    """T45 的 5219 标定 (GrainSpec): 晶体结构由 Kodak H-1-5219 拟合, 涂布量由 500T 自己的特性曲线定。"""
    g = db.grain_5219()
    e = _colour_curve("kodak_vision3_500t")
    return dye.fit_amounts(e, g["ecd_um"], g["kappa"], g["fog_area_um2"], g["packing_s0"], g["ei"], g["illuminant"], ETA_REF,
                           g["thickness_um"], sigma_ln=g["sigma_ln"]).with_(source="Kodak H-1-5219 (2022) 拟合")


def universal(eta=None, beta=None):
    """T46 通用常数 U (dye.derive_spec / silver.silver_structure 的输入): 乳剂结构 λ_ck (3 层 × 3 亚群) 取 5219 的 T45 标定;
    η、β 跨片种通用 (留一检验的全体拟合值); κ、灰雾面积、堆积 S0 同 5219; κ_Ag 黑白。= filmlab.materials.grain_universal。"""
    t46 = db.universal()["t46"]
    base = base_5219()
    lam = dye.structure_from(base, _colour_curve("kodak_vision3_500t"))
    return dict(lam=lam, eta=t46["eta"] if eta is None else eta, beta=t46["beta"] if beta is None else beta,
                a0=t46["a0_um2"], sigma_ln=base.sigma_ln, kappa_dye=base.kappa, kappa_ag=t46["kappa_ag"],
                fog_area=base.fog_area_um2, s0=base.packing_s0)


def process_params(name):
    """冲洗工艺 → dict(yield_cv, cloud_L_um, kappa_scale, name)。= filmlab.materials.process_grain。"""
    P = db.process_params()
    name = str(name or "")
    for k in ("ECN-2", "C-41", "E-6"):
        if name.startswith(k):
            return dict(P[k], name=k)
    return dict(P["BW"], name=name or "BW")


@lru_cache(maxsize=64)
def silver_curve_table(spec):
    """(log10 H, 单调化的银层密度) 细表 (2201 点, −7 … 4; 与 silver.silver_grain_density 同一张表)。
    整帧上的平均密度与反求曝光都按这张表插值 (直接算闭式解要 (h, w, 亚群, 尺寸档) 的数组)。"""
    lh = np.linspace(-7.0, 4.0, 2201)
    Dt = silver.silver_curve(spec, lh)
    return lh, np.maximum.accumulate(Dt + 1e-12 * np.arange(len(Dt)))


# ------------------------------------------------------------- 乳剂 --
@dataclass(frozen=True)
class Emulsion:
    """一卷 "胶片 + 显影液 + 颗粒粗细" 的晶体群。path: "silver" (黑白显式银粒, spec = silver.SilverSpec) 或
    "dye" (彩色染料云, spec = dye.GrainSpec)。"""
    film: Film
    developer: Developer
    grain_scale: float
    spec: object

    @property
    def path(self):
        return "silver" if self.film.is_bw else "dye"

    @property
    def base_density(self):
        """片基 (+ 防光晕染色) 的密度: 黑白 = 总灰雾 − 模型灰雾 (T47); 彩色按层模型不含片基 (0)。"""
        return float(self.spec.base_density) if self.path == "silver" else 0.0

    @property
    def curve(self):
        return _colour_curve(self.film.key) if self.path == "dye" else None

    def exposure_axis(self):
        """彩色: (n, 3) 档案横轴对应的绝对曝光 H (lux·s; 每层净 0.1 处 H = 0.8/EI, 反转片按 ISO 2240)。"""
        return dye.exposure_axis(self.curve, self.film.ei)

    def mean_density(self, H):
        """平均 (闭式) 成像密度, 不含片基。黑白: H (…) → 银层漫射密度 (含灰雾); 彩色: H (…, 3) 每层曝光 → 每层染料当量密度
        (该片的数据册特性曲线, 即 filmlab 原生冲洗的染料量)。"""
        if self.path == "silver":
            lh, Dt = silver_curve_table(self.spec)
            return np.interp(np.log10(np.maximum(np.asarray(H, float), 1e-30)), lh, Dt)
        Hx = self.exposure_axis()
        D = self.curve.density_curves
        H = np.asarray(H, float)
        out = np.empty(H.shape)
        for c in range(3):
            out[..., c] = np.interp(np.log10(np.maximum(H[..., c], 1e-30)), np.log10(Hx[:, c]), D[:, c])
        return out

    def describe(self):
        s = self.spec
        if self.path == "silver":
            b = silver.silver_budget(s)
            return (f"{self.film.name} / {self.developer.name}, k = {self.grain_scale:g}: 三维晶体" if s.thickness_um is None
                    else f"{self.film.name} / {self.developer.name}, k = {self.grain_scale:g}: T 颗粒 {s.thickness_um} µm") + \
                f"; 中位 ECD {np.round(s.ecd_um, 3)} µm; 涂布银 {b['coat_g_m2']:.2f} g/m²; 覆盖力 {b['covering_power']:.2f}; r = {s.photons_r:g}" + \
                (f"; 成团 μ {s.cluster_mu:g}, σ {s.cluster_um:g} µm" if s.cluster_mu > 0 else "")
        return f"{self.film.name} / {self.developer.name}, k = {self.grain_scale:g}: 快亚群 ECD {np.round(np.asarray(s.ecd_um)[:, 0], 3)} µm (R G B)"


@lru_cache(maxsize=None)
def _silver_spec_d76(film_key):
    """= filmlab.materials._silver_spec(key, "bw", "BW-D76", "t48")。"""
    f = _film(film_key)
    c = f.curve()
    U = universal()
    eta = db.universal()["t47"]["eta3d_scale"]
    from .films import _bw_def
    src = _bw_def(film_key)["spec_source"]
    return silver.silver_derive(f.ei, U, eta, 0.8 / f.ei * 10.0 ** c["x"], c["net"], c["gross_fog"], db.t47_morph(),
                                process="BW-D76", source=src, thickness_um=f.thickness_um)


def developer_mod(film_key, dev: Developer):
    """显影液对 SilverSpec 的修改 (= filmlab.materials.bw_developer 的 mod): r 与显影晶体成团。"""
    mod = dict(photons_r=float(dev.photons_r(film_key)))
    if dev.cluster_mu > 0:
        mod.update(cluster_mu=float(dev.cluster_mu), cluster_um=float(dev.cluster_um), cluster_mode="developed")
    return mod


@lru_cache(maxsize=None)
def _dye_spec(film_key, process):
    """= filmlab.materials._grain_spec_t46 (彩色片)。"""
    f = _film(film_key)
    e = _colour_curve(film_key)
    sp = dye.derive_spec(e, f.ei, f.illuminant, universal(), kind="dye", thickness_um=f.thickness_um,
                         process=process_params(process))
    return sp.with_(source=f"T46 通用推导 (EI {f.ei:g}, {f.illuminant}, {'T 颗粒' if f.thickness_um else '三维晶体'}, 工艺 {process})")


@lru_cache(maxsize=None)
def _derive(film_key, dev_key, k):
    f = _film(film_key)
    dev = _developer(dev_key, f)
    if f.is_bw:
        if dev.kind != "bw":
            raise ValueError(f"黑白片 {f.name} 需要黑白显影液 ({', '.join(d for d in ('d76', 'xtol', 'rodinal', 'fx39'))})")
        sp = silver.scale_silver(_silver_spec_d76(film_key).with_(**developer_mod(film_key, dev)), k)
    else:
        if dev.kind != "colour":
            raise ValueError(f"彩色片 {f.name} 需要彩色冲洗工艺 (C-41 / ECN-2 / E-6)")
        sp = dye.scale_grain(_dye_spec(film_key, dev.key), k)
    return Emulsion(f, dev, float(k), sp)


def derive(film, developer=None, grain_scale=1.0):
    """胶片 + 显影液 (None = 原生冲洗: 黑白 D-76, 彩色按片) + 颗粒粗细 k (0.5–3) → Emulsion。"""
    f = _film(film)
    dev = _developer(developer, f)
    return _derive(f.key, dev.key, check_grain_scale(grain_scale))
