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
    halation: float           # 光晕强度, 由防光晕层 / remjet 决定
    remjet: bool = False      # 电影负片的炭黑背涂
    speed_loss: np.ndarray = None   # (3,) 过期造成的感光度损失 (logH)
    fog_exposure: np.ndarray = None  # (3,) 过期造成的潜影灰雾 (等效曝光 logH)

    def with_(self, **kw):
        return replace(self, **kw)

    @property
    def has_couplers(self):
        return bool(np.any(self.coupler_cap > 0))


HALATION = {"strong": 0.008, "weak": 0.035}


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
        dir_amount={"bw": 0.3, "paper": 0.0}.get(design, 1.0),
        halation=HALATION.get(stock.antihalation, 0.01),
        remjet=stock.key in ECN2_STOCKS,
        speed_loss=np.zeros(3), fog_exposure=np.full(3, -np.inf),
    )


@lru_cache(maxsize=None)
def emulsion(key: str) -> Emulsion:
    return from_stock(load_stock(key))


# ------------------------------------------------------------ 材料变体 --
def without_remjet(e: Emulsion, name=None) -> Emulsion:
    """去掉 remjet 背涂 (CineStill 的做法): 片基背面失去吸光层 → 强光晕。"""
    return e.with_(remjet=False, halation=0.08, name=name or f"{e.name} (no remjet)",
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
