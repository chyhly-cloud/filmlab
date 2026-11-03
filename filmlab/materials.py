"""材料层: 把 datasheet 数据 (Stock) 换算成乳剂的物理状态参数 (Emulsion)。

统一单位: 每层的卤化银、银、染料都用"染料当量密度"表示 (1 单位银在彩色显影中生成 1 单位染料)。

datasheet 的特性曲线是"标准冲洗后的染料密度"。由此反推出每种乳剂在其**原生显影液**里的
成像银曲线 S_nom(logH), 再由化学模型在任何冲洗流程下重新计算出结果:
  负片 (C-41/ECN-2): 彩色显影直接成像, S_nom = 染料曲线
  反转片 (E-6):      第一显影 (黑白) 形成负像银, 剩余卤化银在彩色显影中成色,
                     所以 S_nom = Dmax − 染料曲线, 卤化银总量 = Dmax
  黑白:              S_nom = 银密度曲线, 没有成色剂
"""
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from .stocks import Stock, load_stock

# 银的光学覆盖力 (每单位染料当量的银产生的中性密度)。彩色胶片的银颗粒细, 取 0.6 (估计值)
SILVER_COVERING_COLOR = 0.6

NATIVE_PROCESS = {"negative": "C-41", "reversal": "E-6", "bw": "BW-D76", "paper": "RA-4"}
ECN2_STOCKS = {"kodak_vision3_50d", "kodak_vision3_200t", "kodak_vision3_250d",
               "kodak_vision3_500t", "kodak_2383", "kodak_2393"}


@dataclass(frozen=True)
class Emulsion:
    key: str
    name: str
    stock: Stock              # 光谱数据: 感光度、染料、片基
    design: str               # negative | reversal | bw | paper
    native_process: str       # 设计时的冲洗工艺
    log_exposure: np.ndarray  # (256,)
    S_nom: np.ndarray         # (256,3) 原生显影液、标准时间下的成像银
    halide_cap: np.ndarray    # (3,) 可显影卤化银总量
    coupler_cap: np.ndarray   # (3,) 成色剂可生成的最大染料量 (黑白 = 0)
    silver_covering: float
    dir_amount: float         # DIR 成色剂 / 层间效应强度
    halation: float           # 片基背面的反射能力: 透明片基 1 (菲涅耳+全反射), remjet ≈ 0, 防光晕层 = 往返剩余比例
    remjet: bool = False      # 电影负片的炭黑背涂
    speed_loss: np.ndarray = None   # (3,) 过期造成的感光度损失 (logH)
    fog_exposure: np.ndarray = None  # (3,) 过期造成的潜影灰雾 (等效曝光 logH)

    def with_(self, **kw):
        return replace(self, **kw)

    @property
    def has_couplers(self):
        return bool(np.any(self.coupler_cap > 0))


# 防光晕层: 光往返两次穿过吸收层后剩下的比例 (估计值)。strong ≈ 透射 10% 两次; weak ≈ 35% 两次
HALATION = {"strong": 0.01, "weak": 0.12}
# T33: DIR 层间抑制量 (Emulsion.dir_amount) 按档案键覆盖, 由 filmlab.generation 按代际设置。
# 空 = 原取值 (彩色 1.0, 未定标; 逐位不变)。见 docs/tasks/T33_BLUE_LAYER_DIR.md
DIR_AMOUNT = {}


def from_stock(stock: Stock) -> Emulsion:
    x, D = stock.log_exposure, stock.density_curves
    if stock.kind == "paper":
        design = "paper"
    elif stock.bw:
        design = "bw"
    else:
        design = "reversal" if stock.is_positive else "negative"

    if design == "reversal":
        dmax = D.max(0)
        S = dmax - D
        halide, coupler = dmax.copy(), dmax.copy()
    elif design == "bw":
        S = D
        halide, coupler = D.max(0) * 1.3, np.zeros(3)
    else:
        S = D
        halide, coupler = D.max(0) * 1.4, D.max(0) * 1.4   # 过量: Kodak 推两档数据的 Dmax 比标准高约 33%

    native = NATIVE_PROCESS[design]
    if stock.key in ECN2_STOCKS:
        native = "ECN-2"
    return Emulsion(
        key=stock.key, name=stock.name, stock=stock, design=design, native_process=native,
        log_exposure=x, S_nom=S, halide_cap=halide, coupler_cap=coupler,
        silver_covering=1.0 if design == "bw" else SILVER_COVERING_COLOR,
        dir_amount=DIR_AMOUNT.get(stock.key, {"bw": 0.3, "paper": 0.0}.get(design, 1.0)),
        halation=0.0 if stock.key in ECN2_STOCKS else HALATION.get(stock.antihalation, 0.01),
        remjet=stock.key in ECN2_STOCKS,
        speed_loss=np.zeros(3), fog_exposure=np.full(3, -np.inf),
    )


@lru_cache(maxsize=None)
def emulsion(key: str) -> Emulsion:
    return from_stock(load_stock(key))


# ------------------------------------------------------------ 材料变体 --
def without_remjet(e: Emulsion, name=None) -> Emulsion:
    """去掉 remjet 背涂 (CineStill 的做法): 片基背面失去吸光层 → 强光晕。"""
    return e.with_(remjet=False, halation=1.0, name=name or f"{e.name} (no remjet)",
                   key=e.key + "_noremjet")


def aged(e: Emulsion, years: float, seed: int = 0) -> Emulsion:
    """过期胶片的乳剂老化 (发生在拍摄之前):
    - 宇宙射线 / 本底辐射 / 热 → 在卤化银中形成随机潜影 = 一个均匀的"灰雾曝光"
    - 增感剂分解 → 感光度下降; 蓝感层在最上层且颗粒最大, 劣化最快"""
    rng = np.random.default_rng(seed)
    k = years / 10.0
    loss = np.array([0.10, 0.13, 0.20]) * k + rng.normal(0, 0.02, 3) * k
    # 灰雾曝光放在感光点以下约 1.6 logH, 随年数增加
    x = e.log_exposure
    g = e.S_nom[:, 1]
    speed = x[np.argmax(g > g.min() + 0.1)] if e.design != "reversal" else x[np.argmax(g > 0.1)]
    fog = speed - 1.9 + np.array([0.0, 0.05, 0.25]) + 0.35 * np.log2(1 + k)
    return e.with_(speed_loss=loss, fog_exposure=fog, name=f"{e.name} (expired {years:g}y)",
                   key=f"{e.key}_aged{years:g}")


def bw_film(base="kodak_portra_400", sensitivity="pan", contrast=1.1) -> Emulsion:
    """黑白负片乳剂: 单一卤化银, 银影像光谱平坦, 没有成色剂。
    (目前没有黑白胶片的 datasheet 数字化数据, 由彩色负片的数据推导, 属于近似)"""
    s = load_stock(base)
    lin = 10.0 ** s.log_sensitivity
    w = {"pan": [0.9, 1.0, 1.0], "ortho": [0.0, 1.0, 1.15], "blue": [0, 0, 1.0]}[sensitivity]
    sens = np.log10(np.clip(lin @ np.array(w), 1e-10, None))
    curve = s.density_curves.mean(1) * contrast
    st = s.with_(
        key=f"bw_{sensitivity}", name=f"B&W {sensitivity}", bw=True,
        log_sensitivity=np.repeat(sens[:, None], 3, 1),
        dye_density=np.full((81, 3), 1.0 / 3.0),
        base_density=np.full(81, 0.18),
        density_curves=np.repeat(curve[:, None], 3, 1),
    )
    return from_stock(st)


def bw_paper(base="kodak_portra_endura", grade=2.5) -> Emulsion:
    """多级反差黑白相纸 (蓝 + 绿感), 银影像光谱平坦。grade 0..5 对应反差。"""
    p = load_stock(base)
    lin = 10.0 ** p.log_sensitivity
    sens = np.log10(np.clip(lin[:, 1] + lin[:, 2], 1e-10, None))
    x, c = p.log_exposure, p.density_curves.mean(1)
    mid = x[np.argmax(c > c.max() / 2)]
    curve = np.interp(mid + (x - mid) * (0.6 + 0.2 * grade), x, c)
    st = p.with_(
        key="bw_paper", name=f"B&W paper grade {grade:g}", bw=True,
        log_sensitivity=np.repeat(sens[:, None], 3, 1),
        dye_density=np.full((81, 3), 1.0 / 3.0),
        base_density=np.full(81, 0.05),
        density_curves=np.repeat(curve[:, None], 3, 1),
    )
    e = from_stock(st)
    return e.with_(design="paper", native_process="BW-D76", coupler_cap=np.zeros(3),
                   silver_covering=1.0)


# ------------------------------------------------------------- 颗粒 (T45) --
# 颗粒的银盐物理参数 (filmlab.grain; docs/tasks/T45_GRAIN.md)。通道顺序 R G B; 每层 3 个晶体亚群 (快 / 中 / 慢)。
# 5219 (Vision3 500T): 晶体尺寸、κ、灰雾面积尺度 a_f、堆积 S0 由 Kodak H-1-5219 (2022) 的漫射 RMS 颗粒度曲线与
#   特性曲线联合拟合 (tools/eval/t45_grain_fit.py); 涂布量由仓库的原生特性曲线 (非负最小二乘) 定。
#   来源: https://www.kodak.com/content/products-brochures/Film/VISION3_5219_7219_Technical-data.pdf
GRAIN_5219 = dict(
    ecd_um=((1.808, 0.499, 0.114), (2.332, 0.583, 0.138), (3.338, 0.812, 0.214)),
    kappa=(2.492, 2.602, 3.195), fog_area_um2=(28.35, 43.42, 21.38), packing_s0=(0.0, 0.0, 0.0),
    ei=500.0, illuminant="BB3200", thickness_um=0.12, sigma_ln=(0.194, 0.186, 0.199))
# 其他片种: 同一模型, 只换该片种自己的物理量 —— 感光度 EI、设计光源、有效量子效率 η (相对 5219), 晶体形态与银 / 染料。
#   晶体尺寸按 a ∝ 1/(Φ·0.8/EI) 由 5219 的亚群结构换算, 涂布量由该片种自己的特性曲线定。
#   η 比值: 5213 / 5203 由各自 Kodak 颗粒度曲线定 (每片种 1 个数, 见 T45 文档"通用性检验"); 黑白由 Tri-X 400 (F-4017) 的 17 定。
#   黑白: 三维晶体 (v = a^1.5), 显影银 κ_Ag = 覆盖力 0.5 密度/(g/m²) × 每 µm³ AgBr 含银 3.72e-12 g = 1.86 (假设的典型覆盖力)。
GRAIN_FILMS = {
    "kodak_vision3_500t": dict(eta_ratio=1.0, source="Kodak H-1-5219 (2022) 拟合"),
    "kodak_vision3_200t": dict(eta_ratio=0.791, ei=200.0, illuminant="BB3200",
                               source="https://www.freestylephoto.com/static/pdf/product_pdf/kodak/KODAK_VISION3_200T_Brochure.pdf"),
    "kodak_vision3_50d": dict(eta_ratio=0.354, ei=50.0, illuminant="D55",
                              source="https://www.kodak.com/content/pdfs/motion/KODAK-VISION3-50D-5203-7203-brochure.pdf"),
    "bw": dict(eta_ratio=1.136, ei=400.0, illuminant="D55", kind="silver", kappa=(1.86, 1.86, 1.86), thickness_um=None,
               source="https://kodakprofessional.com/sites/default/files/wysiwyg/film/f4017_trix_320400.pdf (Tri-X 400: 17)"),
}
# 没有专门参数的片种: η 比值取已标定三者 (0.79 / 0.35 / 1.14) 的中位 0.79, 其余同上换算 (假设, 未逐一核对官方颗粒度)
GRAIN_DEFAULT_ETA_RATIO = 0.79


def grain_spec(e: Emulsion):
    """该乳剂的颗粒物理参数 (filmlab.grain.GrainSpec)。按 (档案键, 设计) 缓存。"""
    return _grain_spec(e.stock.key, e.design, e.name, e.stock.reference_illuminant)


@lru_cache(maxsize=None)
def _grain_spec(stock_key, design, name, ref_illum):
    import re
    from . import grain
    base_e = emulsion("kodak_vision3_500t")
    base = grain.fit_amounts(base_e, GRAIN_5219["ecd_um"], GRAIN_5219["kappa"], GRAIN_5219["fog_area_um2"],
                             GRAIN_5219["packing_s0"], GRAIN_5219["ei"], GRAIN_5219["illuminant"], grain.ETA_REF,
                             GRAIN_5219["thickness_um"], sigma_ln=GRAIN_5219["sigma_ln"]).with_(source=GRAIN_FILMS["kodak_vision3_500t"]["source"])
    if stock_key == "kodak_vision3_500t":
        return base
    e = _emulsion_for_grain(stock_key)
    if design == "bw":
        p = GRAIN_FILMS["bw"]
    elif stock_key in GRAIN_FILMS:
        p = GRAIN_FILMS[stock_key]
    else:
        m = re.search(r"(\d{2,4})", name)
        ill = "BB3200" if str(ref_illum).startswith("BB") and float(re.sub(r"\D", "", str(ref_illum)) or 5500) < 4500 else "D55"
        p = dict(eta_ratio=GRAIN_DEFAULT_ETA_RATIO, ei=float(m.group(1)) if m else 200.0, illuminant=ill,
                 source="默认: 5219 结构按感光度换算 (未标定)")
    kw = {k: p[k] for k in ("kappa", "thickness_um") if k in p}
    sp = grain.scaled(base, e, p["ei"], p["illuminant"], grain.ETA_REF * p["eta_ratio"], kind=p.get("kind", "dye"), **kw)
    return sp.with_(source=p["source"])


def _emulsion_for_grain(stock_key):
    if stock_key.startswith("bw_"):
        return bw_film(sensitivity=stock_key[3:]) if stock_key[3:] in ("pan", "ortho", "blue") else bw_film()
    return emulsion(stock_key)


# ------------------------------------------------------------- 颗粒 (T46) --
# 跨片种通用的银盐颗粒 (docs/tasks/T46_GRAIN_GENERAL.md; 留一检验 tools/eval/t46_grain_loo.py)。
# 每个颗粒 = 一个卤化银晶体: 位置 (涂布堆积)、尺寸 (对数正态亚群)、形态 (T 颗粒 v = a·t / 三维 v = a^1.5)、
# 是否显影 (吸收 ≥ 4 个有效光子) 都按晶体决定; 彩色片的显影晶体在漂白前原位生成一团染料云, 黑白片保留银。
#
# 通用常数 (所有片种共用; 依据见文档第 3 节):
#   乳剂结构 λ_ck: 速度点处快 / 中 / 慢亚群中位晶体吸收的有效光子数, σ_ln, 每种染料的 κ, 灰雾面积 a_f, 堆积 S0
#     —— 取 T45 的 5219 标定 (只用它的"结构", 绝对尺寸不再来自 5219, 而由下面的 η、β 与各片自己的感光度定)
#   η (a0 = 1 µm² 处每面积量子效率), β (量子效率随投影面积的指数) —— 多片种官方颗粒度联合定 (留一检验)
#   T 颗粒厚度 t = 0.12 µm (Kodak T-GRAIN 公开描述的 0.1–0.2 µm 量级, T45 假设)
#   银粒 κ_Ag = 显影银覆盖力 0.5 密度/(g/m²) × 每 µm³ AgBr 含银 3.72e-12 g = 1.86 (T45 假设)
GRAIN_T46_ETA = 0.0248          # 留一检验的全体拟合值 (tools/eval/t46/loo.json)
GRAIN_T46_BETA = 0.875
GRAIN_T46_A0 = 1.0
GRAIN_KAPPA_AG = 1.86
GRAIN_T_UM = 0.12
# 冲洗工艺的最后一步 (显影晶体 → 染料云 / 银丝团): 每晶体产量的变异系数 (伽马) 与扩散长度 L (µm)。
#   没有找到单个染料云产量的显微统计; CV 0.3 为量级估计 (成色剂油滴计数的泊松起伏 + 大 T 颗粒的部分显影),
#   L 由 EPFL 800T 扫描颗粒自相关的 1 µm 量级定 (T45 第 3.2 节); 各工艺之间没有公开的可区分数据, 取同值。
#   kappa_scale: 每 µm³ 显影晶体的积分密度相对彩色负片 (5219 标定) 的倍数, 由该工艺官方颗粒度的族常数定 (σ ∝ √κ):
#     E-6 5.41 (彩色显影把第一显影剩下的晶体全部显影、成色剂过量; 负片成色剂不足 + DIR 抑制, 只有理论产率的 3–4%),
#     黑白 1.36 (κ_Ag = 1.86 × 1.36 = 2.52, 即显影银覆盖力 0.68 密度/(g/m²))。
GRAIN_PROCESS = {
    "C-41": dict(yield_cv=0.3, cloud_L_um=1.0, kappa_scale=1.0),
    "ECN-2": dict(yield_cv=0.3, cloud_L_um=1.0, kappa_scale=1.0),
    "E-6": dict(yield_cv=0.3, cloud_L_um=1.0, kappa_scale=5.407),
    "BW": dict(yield_cv=0.3, cloud_L_um=0.2, kappa_scale=1.357),     # 银丝团: 覆盖力随机 (丝团疏密) + 约晶体大小的空间范围
}
# 每个档案的公开物理量 (感光度、设计光源、晶体形态); 未列出的按名称里的感光度、彩色 T 颗粒、档案光源推定。
GRAIN_PHYS = {
    "kodak_vision3_500t": dict(ei=500, illuminant="BB3200"), "kodak_vision3_200t": dict(ei=200, illuminant="BB3200"),
    "kodak_vision3_250d": dict(ei=250, illuminant="D55"), "kodak_vision3_50d": dict(ei=50, illuminant="D55"),
    "kodak_portra_160": dict(ei=160), "kodak_portra_400": dict(ei=400), "kodak_portra_800": dict(ei=800),
    "kodak_ektar_100": dict(ei=100), "kodak_gold_200": dict(ei=200), "kodak_ultramax_400": dict(ei=400),
    "fujifilm_pro_400h": dict(ei=400), "fujifilm_xtra_400": dict(ei=400), "fujifilm_c200": dict(ei=200),
    "fujifilm_provia_100f": dict(ei=100), "fujifilm_velvia_100": dict(ei=100), "kodak_ektachrome_100": dict(ei=100),
    "kodak_kodachrome_64": dict(ei=64, thickness_um=None),
    "bw": dict(ei=400, thickness_um=None),        # bw_film 通用全色: 按 Tri-X 400 (传统三维晶体)
}


def grain_universal(eta=None, beta=None):
    """T46 通用常数 U (filmlab.grain.derive_spec)。乳剂结构取 5219 的 T45 标定。"""
    from . import grain
    base = _grain_spec("kodak_vision3_500t", "negative", "Kodak Vision3 500T", "BB3200")
    lam = grain.structure_from(base, emulsion("kodak_vision3_500t"))
    return dict(lam=lam, eta=GRAIN_T46_ETA if eta is None else eta, beta=GRAIN_T46_BETA if beta is None else beta,
                a0=GRAIN_T46_A0, sigma_ln=base.sigma_ln, kappa_dye=base.kappa, kappa_ag=GRAIN_KAPPA_AG,
                fog_area=base.fog_area_um2, s0=base.packing_s0)


def process_grain(process_name: str):
    name = str(process_name or "")
    for k in ("ECN-2", "C-41", "E-6"):
        if name.startswith(k):
            return dict(GRAIN_PROCESS[k], name=k)
    return dict(GRAIN_PROCESS["BW"], name=name or "BW")


def grain_spec_t46(e: Emulsion, process_name: str = None):
    """T46 的颗粒档案: 只由该胶片的公开物理量 + 通用常数 + 冲洗工艺生成。"""
    return _grain_spec_t46(e.stock.key, e.design, e.name, e.stock.reference_illuminant,
                           process_name or e.native_process)


@lru_cache(maxsize=None)
def _grain_spec_t46(stock_key, design, name, ref_illum, process_name):
    import re
    from . import grain
    e = _emulsion_for_grain(stock_key)
    p = dict(GRAIN_PHYS["bw"] if design == "bw" else GRAIN_PHYS.get(stock_key, {}))
    if "ei" not in p:
        m = re.search(r"(\d{2,4})", name)
        p["ei"] = float(m.group(1)) if m else 200.0
    if "illuminant" not in p:
        ri = str(ref_illum)
        p["illuminant"] = "BB3200" if ri.startswith("BB") and float(re.sub(r"\D", "", ri) or 5500) < 4500 else "D55"
    kind = "silver" if design == "bw" else "dye"
    t = p.get("thickness_um", GRAIN_T_UM)
    sp = grain.derive_spec(e, p["ei"], p["illuminant"], grain_universal(), kind=kind, thickness_um=t,
                           process=process_grain(process_name))
    return sp.with_(source=f"T46 通用推导 (EI {p['ei']:g}, {p['illuminant']}, {'T 颗粒' if t else '三维晶体'}, 工艺 {process_name})")
