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
    if stock_key in BW_SILVER_FILMS or stock_key in bw_t48_data():   # T47 / T47b / T48 黑白片 (曲线由晶体模型算出)
        return bw_silver_film(stock_key)
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


# ------------------------------------------------------------- 颗粒 (T47) --
# 显式银粒 (filmlab.grain.SilverSpec / render_silver; docs/tasks/T47_TRIX_SILVER_GRAIN.md)。黑白片通用。
# 显影后银粒形态 (显影工艺参数; 先验, 未逐片实测):
#   spread      银丝团投影面积 / 晶体投影面积 = 4 (丝团直径约为晶体的 2 倍: 化学显影的银丝从显影中心长出、伸出晶体轮廓)
#   filament_um 银丝直径 0.04 µm (电镜下化学显影银丝为数十纳米量级); 银质量守恒决定丝长, 丝团的不透明度由此算出
#   callier_q   漫射 / 镜面收集的密度比 Q = 1.4 (粗颗粒黑白负片的经典范围 1.3–1.5; https://en.wikipedia.org/wiki/Callier_effect)
GRAIN_T47_MORPH = dict(spread=4.0, filament_um=0.04, filaments=4, persistence=0.5, callier_q=1.4)
# Schmidt 显微镜颗粒的收集几何 (页面未写明; 假设): 漫射。依据见 T47 文档第 4 节 (镜面口径与 Kodak 48 µm 官方值矛盾)。
GRAIN_T47_MICRO_GEOMETRY = "diffuse"
# 三维晶体 (传统立方 / 八面体) 的有效量子效率相对 T46 通用值 (η × 3 表面吸收) 的倍数 —— T47 标定量:
#   由 Schmidt Tri-X / D-76 显微颗粒的 20 / 25 µm 平台定 (tools/eval/t47_trix.py → tools/eval/t47/calib.json)。
#   T46 已指出三维晶体按通用 β 推出的晶体偏大 (Tri-X 颗粒偏粗 1.31 倍); 这里只修正这一个已有物理量。
#   T47b (tools/eval/t47b_bw_loo.py): 这个修正属于显式银粒模型 (T46 的 η 是在染料云 / κ 模型下定的), 对 T 颗粒同样适用
#   (grain.silver_structure 的 scale_tgrain=True); 只乘三维时 T 颗粒片偏粗 1.5–1.9 倍, 两者都乘时 T-Max / Delta 为 0.90–1.14。
GRAIN_T47_ETA3D_SCALE = 6.608337
# Schmidt 的 Tri-X 400 / D-76 原液 20 °C 9 min 特性曲线 (tools/eval/t47/schmidt_density_19.csv; 漫射密度计 —— 几何检验):
#   (净 D 0.10 处为零点的 log10 相对曝光, 净 D); 片基 + 灰雾 0.318。来源 https://www.bernhard-w-schmidt.de/en/einzeln-trix
TRIX_D76_CURVE = (
    (
     -0.351661, -0.198357, -0.050612, 0.101300, 0.254443, 0.401886, 0.556097, 0.700907, 0.856097, 1.008704,
     1.156172, 1.310667, 1.458189, 1.610382, 1.760926, 1.909405, 2.060395, 2.209405, 2.363445),
    (
     0.014539, 0.045951, 0.081639, 0.136748, 0.214511, 0.282623, 0.394426, 0.526526, 0.639961, 0.750542, 0.840658,
     0.930406, 1.020022, 1.119872, 1.218745, 1.307940, 1.396306, 1.486511, 1.576268))
TRIX_GROSS_FOG = 0.318
# Kodak F-4017 第 7 页光谱感光度 (D = 1.0 > 灰雾, log 1/(erg/cm²); tools/eval/t47/kodak_f4017_p7_spectral.csv), 全色
TRIX_SPECTRAL = (
    (300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410,
     415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525,
     530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640,
     650, 655, 660, 667),
    (2.508, 2.508, 2.525, 2.546, 2.568, 2.592, 2.616, 2.638, 2.657, 2.673, 2.684, 2.694, 2.698, 2.703, 2.708, 2.711,
     2.711, 2.710, 2.708, 2.703, 2.698, 2.692, 2.687, 2.679, 2.671, 2.660, 2.648, 2.632, 2.616, 2.597, 2.574, 2.551,
     2.522, 2.484, 2.448, 2.405, 2.360, 2.319, 2.281, 2.251, 2.238, 2.224, 2.232, 2.248, 2.270, 2.292, 2.317, 2.340,
     2.362, 2.383, 2.401, 2.419, 2.432, 2.444, 2.452, 2.454, 2.446, 2.429, 2.403, 2.384, 2.380, 2.375, 2.392, 2.410,
     2.414, 2.392, 2.330, 2.189, 1.932, 1.000, 0.720, 0.430, 0.000))
TRIX_KEY = "kodak_trix_400"
# T47b: 其他黑白片 (Schmidt 同一显微镜 / 同一 D-76 原液 20 °C 的曲线, tools/eval/t47/schmidt_bw_d76.csv;
#   (净 D 0.10 处为零点的 log10 相对曝光, 净 D)。T-Max 100 9 min EI 200、T-Max 400 11 min EI 640 (Schmidt 实测有效感光度)。
#   来源 https://www.bernhard-w-schmidt.de/en/einzeln-tmx 、https://www.bernhard-w-schmidt.de/en/einzeln-tmy
TMX_D76_CURVE = (
    (
     -0.3334, -0.0334, 0.1166, 0.4166, 0.5666, 0.7166, 0.8666, 1.0166, 1.1666, 1.4666, 1.6166, 1.7666, 1.9166,
     2.0666),
    (
     0.0000, 0.0869, 0.1457, 0.3952, 0.5129, 0.6362, 0.7651, 0.9053, 1.0258, 1.2220, 1.3257, 1.4070, 1.4994,
     1.5891))
TMY_D76_CURVE = (
    (
     -0.2769, -0.1269, 0.0231, 0.1731, 0.3231, 0.4731, 0.6231, 0.7731, 0.9231, 1.0731, 1.2231, 1.3731, 1.5231,
     1.6731, 1.8231, 1.9731, 2.1231, 2.2731, 2.4231),
    (
     0.0000, 0.0336, 0.1121, 0.2017, 0.2998, 0.3895, 0.4736, 0.6221, 0.7258, 0.8295, 0.9529, 1.0762, 1.2079,
     1.3649, 1.5218, 1.6900, 1.8469, 2.0123, 2.2141))
# 每种黑白片的显式银粒档案: 感光度 (与所用曲线一致)、晶体形态 (T 颗粒厚度 / None = 三维)、D-76 曲线、片基 + 灰雾。
#   光谱感光度都取 Tri-X 的 F-4017 全色曲线 (T-Max / Plus-X 的光谱未数字化; T-Max 数据册写明蓝光感光度略低, 未计入)。
#   Plus-X 125: 没有 D-76 曲线 (Schmidt 未测, Kodak 官方曲线为 HC-110), 曲线形状借 Tri-X D-76、感光度 125 —— 假设。
BW_SILVER_FILMS = {
    TRIX_KEY: dict(name="Kodak Tri-X 400", ei=400.0, thickness_um=None, curve="TRIX_D76_CURVE", gross_fog=0.318,
                   source="Tri-X 400: Schmidt D-76 曲线 + T47 标定"),
    "kodak_tmax_100": dict(name="Kodak T-Max 100", ei=200.0, thickness_um=GRAIN_T_UM, curve="TMX_D76_CURVE", gross_fog=0.2186,
                           source="T-Max 100 (T-GRAIN): Schmidt D-76 9 min 曲线 (EI 200)"),
    "kodak_tmax_400": dict(name="Kodak T-Max 400", ei=640.0, thickness_um=GRAIN_T_UM, curve="TMY_D76_CURVE", gross_fog=0.2607,
                           source="T-Max 400 (T-GRAIN): Schmidt D-76 11 min 曲线 (EI 640)"),
    "kodak_plusx_125": dict(name="Kodak Plus-X 125", ei=125.0, thickness_um=None, curve="TRIX_D76_CURVE", gross_fog=0.318,
                            source="Plus-X 125 (三维): 曲线形状借 Tri-X D-76 (假设), EI 125"),
}


# ------------------------------------------------------------- 黑白片 (T48) --
# 每种黑白片自己的光谱感光度与 D-76 类特性曲线 (filmlab/data/bw_t48.json; 数字化读数 tools/eval/t48/, 来源 URL 逐条;
# docs/tasks/T48_BW_SPECTRAL_CURVES.md)。按代际选用: generation.use 设 BW_PROFILE_SET ("t47" = T47b 的档案, 现有代际逐位不变;
# "t48" = 本表)。T47 里没有的片种 (Ilford、Acros) 两种设置下都用本表。
# "t49" (T49b, docs/tasks/T49_BW_FILTERS.md): 同 "t48", 但 T-Max 100 与 Ilford 片的光谱按滤镜系数检验修正
# (filmlab/data/bw_t49_spectra.json); 曲线与晶体档案不变。
BW_PROFILE_SET = "t47"


@lru_cache(maxsize=None)
def bw_t49_spectra():
    import json
    from pathlib import Path
    return json.loads((Path(__file__).parent / "data" / "bw_t49_spectra.json").read_text())["films"]


@lru_cache(maxsize=None)
def bw_t48_data():
    import json
    from pathlib import Path
    return json.loads((Path(__file__).parent / "data" / "bw_t48.json").read_text())


def bw_film_def(key, profile_set=None):
    """黑白片档案的定义: dict(name, ei, thickness_um, curve (x, net), gross_fog, spectral (lam, logS), source)。
    x = 以净 D 0.10 为零点的 log10 相对曝光 (绝对曝光由官方感光度定: 净 D 0.10 处 H = 0.8/EI)。"""
    ps = BW_PROFILE_SET if profile_set is None else profile_set
    if ps == "t49":
        d = bw_film_def(key, "t48")
        if key in bw_t49_spectra():
            sp = bw_t49_spectra()[key]
            d = dict(d, spectral=(sp["lam"], sp["logS"]), source=d["source"] + "; 光谱 T49b")
        return d
    t48 = bw_t48_data()
    if key in BW_SILVER_FILMS and (ps != "t48" or key not in t48):
        f = BW_SILVER_FILMS[key]
        return dict(name=f["name"], ei=f["ei"], thickness_um=f["thickness_um"], curve=globals()[f["curve"]],
                    gross_fog=f["gross_fog"], spectral=TRIX_SPECTRAL, source=f["source"])
    d = t48[key]
    c = d["curve"]
    if c is None:                     # Tri-X: 曲线沿用 T47 (Schmidt D-76 9 min)
        curve, fog, src = TRIX_D76_CURVE, TRIX_GROSS_FOG, "Tri-X 400: Schmidt D-76 曲线 + T47 标定"
    else:
        curve, fog, src = (c["x"], c["net"]), c["gross_fog"], f"T48: {d['name']}; 曲线 {c['process']}"
    return dict(name=d["name"], ei=float(d["ei"]), thickness_um=d["thickness"], curve=curve, gross_fog=fog,
                spectral=(d["spectral"]["lam"], d["spectral"]["logS"]), source=src)


def bw_profile_key(key):
    """档案缓存的键: 片种 + 选用的数据集 (Tri-X 在两种设置下相同)。"""
    if BW_PROFILE_SET == "t49" and key in bw_t49_spectra():
        return "t49"
    t48 = bw_t48_data()
    if key in t48 and (BW_PROFILE_SET in ("t48", "t49") or key not in BW_SILVER_FILMS):
        return "t48" if key != TRIX_KEY else "t47"
    return "t47"


def _curve_set(profile_set):
    """晶体 / 银档案只取决于特性曲线: T49b 只改光谱, 曲线同 t48。"""
    return "t48" if profile_set == "t49" else profile_set


def silver_spec(e: Emulsion, process_name: str = None):
    """T47 黑白片的显式银粒档案 (filmlab.grain.SilverSpec)。Tri-X 400: 晶体由 T46 通用结构 + 三维 η 标定量, 涂布银量由
    Schmidt D-76 曲线 (漫射) 定; 其他黑白片: 同一组常数, 涂布银量由该片自己的特性曲线定 (感光度按 GRAIN_PHYS)。
    T48: 有档案的黑白片按 bw_film_def (BW_PROFILE_SET)。"""
    k = e.stock.key
    if k in BW_SILVER_FILMS or k in bw_t48_data():
        return _silver_spec(k, e.design, process_name or e.native_process, _curve_set(bw_profile_key(k)))
    return _silver_spec(k, e.design, process_name or e.native_process)


@lru_cache(maxsize=None)
def _silver_spec(stock_key, design, process_name, profile_set="t47"):
    from . import grain
    U = grain_universal()
    eta = GRAIN_T47_ETA3D_SCALE
    if stock_key in BW_SILVER_FILMS or stock_key in bw_t48_data():
        f = bw_film_def(stock_key, profile_set)
        x, net = map(np.asarray, f["curve"])
        return grain.silver_derive(f["ei"], U, eta, 0.8 / f["ei"] * 10.0 ** x, net, f["gross_fog"], GRAIN_T47_MORPH,
                                   process=str(process_name), source=f["source"], thickness_um=f["thickness_um"])
    e = _emulsion_for_grain(stock_key)
    ei = float(GRAIN_PHYS["bw"]["ei"])
    H = grain.exposure_axis(e, ei)[:, 1]
    S = e.S_nom[:, 1]
    net = np.maximum.accumulate(S - S.min())
    ok = net < 0.97 * net.max()
    return grain.silver_derive(ei, U, eta, H[ok], net[ok], float(S.min()), GRAIN_T47_MORPH, process=str(process_name),
                               source=f"T47 通用推导 (EI {ei:g}, 三维晶体, 自己的特性曲线)")


def silver_fog_offset(e: Emulsion, spec):
    """显影化学的银密度 (S_nom, 含曲线的灰雾) 与晶体模型银层密度之差中与曝光无关的部分 (非颗粒的中性灰雾, 不显式渲染)。"""
    from . import grain
    return float(e.S_nom[:, 1].min() - grain.silver_density(spec, 0.0))


def trix_film() -> Emulsion:
    """Kodak Tri-X 400 (400TX) 黑白负片 (T47): 特性曲线由显式银粒的同一晶体模型算出 (漫射密度, 绝对 log10 lux·s 横轴),
    光谱感光度取 Kodak F-4017 (全色)。片基密度 = Schmidt 的片基 + 灰雾 − 模型灰雾。"""
    return bw_silver_film(TRIX_KEY)


def bw_silver_film(key: str) -> Emulsion:
    """有档案的黑白片 (T47b BW_SILVER_FILMS; T48 bw_t48.json): 特性曲线由显式银粒的晶体模型算出, 光谱感光度取该片档案。"""
    return _bw_silver_film(key, bw_profile_key(key))


@lru_cache(maxsize=None)
def _bw_silver_film(key, profile_set="t47"):
    from . import grain
    from .stocks import WAVELENGTHS
    base = load_stock("kodak_portra_400")
    spec = _silver_spec(key, "bw", "BW-D76", _curve_set(profile_set))
    f = bw_film_def(key, profile_set)
    x = np.linspace(-4.5, 1.5, 256)
    D = grain.silver_curve(spec, x, "diffuse")
    lam, ls = map(np.asarray, f["spectral"])
    logS = np.where(WAVELENGTHS <= lam.max(), np.interp(WAVELENGTHS, lam, ls), -10.0)
    # 绝对尺度与仓库其他档案一致 (测光按包装感光度另行定标; 只有光谱形状起作用): D55 下的总响应与通用全色片相同
    from .spectral import illuminant
    d55 = illuminant("D55")
    ref = bw_film().stock.log_sensitivity[:, 1]
    logS = np.where(logS > -9, logS + np.log10((d55 * 10.0 ** ref).sum() / (d55 * 10.0 ** logS).sum()), logS)
    st = base.with_(key=key, name=f["name"], bw=True, kind="negative",
                    log_sensitivity=np.repeat(logS[:, None], 3, 1),
                    dye_density=np.full((81, 3), 1.0 / 3.0),
                    base_density=np.full(81, spec.base_density),
                    log_exposure=x, density_curves=np.repeat(D[:, None], 3, 1),
                    reference_illuminant="D55", target_print=None)
    return from_stock(st)
