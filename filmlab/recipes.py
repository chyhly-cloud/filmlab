"""配方 = 材料 + 冲洗工艺 + 拍摄条件。

这里不包含任何"调色"——每个配方只是在描述一个真实存在的组合:
哪种胶片、在什么光下用什么滤镜拍、送进哪种冲洗流程、印在哪种相纸上。
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .chemistry import PROCESSES
from .materials import (BW_SILVER_FILMS, aged, bw_developer, bw_film, bw_silver_film, bw_t48_data, bw_t50_developers, emulsion,
                        trix_film, without_remjet)
from .stocks import WAVELENGTHS


# ------------------------------------------------------- 镜头滤镜 --
def lens_filter(kind: str) -> np.ndarray:
    """镜头前滤镜的光谱透射率 (近似 Wratten 系列)。"""
    def lp(cut, w=12):
        return 1.0 / (1.0 + np.exp(-(WAVELENGTHS - cut) / w))
    table = {
        "yellow": lambda: lp(495),        # Wratten 8
        "orange": lambda: lp(550),        # Wratten 21
        "red": lambda: lp(595),           # Wratten 25
        "green": lambda: np.exp(-0.5 * ((WAVELENGTHS - 540) / 55) ** 2) * 0.9 + 0.05,
        "blue": lambda: np.exp(-0.5 * ((WAVELENGTHS - 450) / 50) ** 2) * 0.9 + 0.03,
        "85": lambda: 1.0 - 0.45 / (1.0 + np.exp((WAVELENGTHS - 520) / 25)),   # 灯光片日光下用
        "80a": lambda: 1.0 - 0.7 / (1.0 + np.exp(-(WAVELENGTHS - 560) / 30)),  # 日光片灯光下用
    }
    if kind in table:                     # 原有的近似曲线 (bw_pan_yellow 等配方用), 保持不变
        return table[kind]()
    num = wratten_number(kind)
    if num is not None:
        return wratten(num)
    return np.ones(81)


# T49: Kodak Wratten 滤镜的实测光谱透过率 (silvergrain/data/wratten_filters.json, T55 前为 filmlab/data/; 读自 Kodak Wratten Filters 1965 的表)
WRATTEN_ALIASES = {"k2": "8", "x1": "11", "g": "15", "a": "25"}


@lru_cache(maxsize=None)
def wratten_data() -> dict:
    from silvergrain import database
    return json.loads(database.path("wratten_filters.json").read_text())


def wratten_number(kind) -> str | None:
    """'25' / 'w25' / 'wratten_25' / 'A' → '25'; 不是数据里的 Wratten 滤镜时返回 None。"""
    k = str(kind).lower()
    for p in ("wratten_", "wratten", "w"):
        if k.startswith(p) and k[len(p):] in wratten_data()["filters"]:
            k = k[len(p):]
            break
    k = WRATTEN_ALIASES.get(k, k)
    return k if k in wratten_data()["filters"] else None


def wratten(number) -> np.ndarray:
    """Wratten 滤镜在 81 点波长上的透过率。表外: 短于首个读数为 0, 长于 700 nm 保持 700 nm 的值。"""
    f = wratten_data()["filters"][str(number)]
    lam, t = np.asarray(f["lam"], float), np.asarray(f["T"], float)
    return np.where(WAVELENGTHS < lam[0], 0.0, np.interp(WAVELENGTHS, lam, t))


# 胶片数据表没有给出某个滤镜的系数时: 先用 Kodak 全色片的通用系数 (Wratten Filters 1965, p.17), 仍没有的
# (No. 21 橙、No. 47 蓝) 用给出了该系数的各片的中位数 (日光: No. 21 = HP5 2.6 / FP4 3 / Acros 4 → 3;
# No. 47 = Tri-X 6 / T-Max 100 8 / T-Max 400 10 / Plus-X 6 → 7)。见 docs/tasks/T49_BW_FILTERS.md。
def filter_factor(film: str, number, light: str = "daylight") -> tuple[float, str]:
    """摄影师用的滤镜系数 (曝光倍数) 与它的出处: 该片数据表 → Kodak 全色片通用表 → 其他片的中位数。"""
    num = wratten_number(number)
    fac = wratten_data()["factors"][light]
    if film in fac and num in fac[film]:
        return float(fac[film][num]), film
    if num in fac["kodak_panchromatic_class"]:
        return float(fac["kodak_panchromatic_class"][num]), "kodak_panchromatic_class"
    vals = [d[num] for f, d in fac.items() if num in d]
    if not vals:
        raise KeyError(f"没有 Wratten {number} 在{light}下的滤镜系数")
    return float(np.median(vals)), "median"


# ------------------------------------------- S0 扩展: 近红外估算 --
def near_infrared_estimate(strength=1.0):
    """Kodak Aerochrome III 1443 红外彩色反转片。
    三层: 绿感 → 黄染料, 红感 → 品红染料, 近红外感 → 青染料 (配黄滤镜挡掉蓝光)。

    普通 RAW 没有近红外 (传感器前有 IR 截止滤镜), 这里按植被的"红边"规律估算:
    叶绿素强吸收红光、强反射 700–900nm, 所以绿色植被 → 高 IR。
    全光谱改装相机的 RAW 可以把真实 IR 通道接到这里。"""
    def hook(rgb, H):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        veg = np.clip(g - 0.5 * (r + b), 0, None) / (np.abs(g) + 1e-4)
        ir = Y * (0.55 + strength * 4.0 * veg) + 0.35 * np.clip(r, 0, None)
        scale = H[..., 1].mean() / (Y.mean() + 1e-6)
        # 层顺序 [红感/青染料, 绿感/品红, 蓝感/黄] ← [IR, 红, 绿]
        return np.stack([ir * scale, H[..., 0], H[..., 1]], -1)
    return hook


# --------------------------------------------------------------- 配方 --
def recipe(name: str) -> dict:
    """返回 engine.Params 的关键字参数。"""
    if "+" in name:
        # 配方后缀 (可叠加, 按顺序): "<配方>+w<Wratten 号>" 滤镜 (T49)、"+g<k>" 颗粒粗细 (T53)、"+f<画幅>" 画幅 (T53)。
        # 例 "kodak_trix_400+w25"、"kodak_trix_400+g1.5"、"kodak_trix_400@rodinal+w25+g2"、"kodak_trix_400+fhalf"。
        base, *mods = name.split("+")
        kw = dict(recipe(base))
        for mod in mods:
            tag, val = mod[:1], mod[1:]
            if tag == "w":
                # T49 黑白滤镜: 滤镜加在镜头前的光谱上 (与片种无关); 曝光按摄影师的做法补偿: 不带滤镜测光,
                # 再乘该片数据表公布的滤镜系数 (filter_factor)。
                if wratten_number(val) is None:
                    raise KeyError(f"未知的 Wratten 滤镜: {val}")
                film = kw["film"] if isinstance(kw["film"], str) else base
                kw["lens_filter"] = wratten(wratten_number(val))
                kw["filter_factor"] = filter_factor(film, val)[0]
            elif tag == "g":
                # T53 颗粒粗细 k: 同一胶片的假想乳剂, 每个线性颗粒尺寸 ×k (engine.Params.grain_scale; 范围 0.5–3)
                from .grain import check_grain_scale
                try:
                    k = float(val)
                except ValueError:
                    raise ValueError(f"颗粒粗细后缀须为 +g<数>, 例 +g1.5: {mod!r}") from None
                kw["grain_scale"] = check_grain_scale(k)
            elif tag == "f":
                # T53 画幅 (engine.FILM_FORMATS 的键, 例 +fhalf、+f110、+f645): 每个输出像素对应的胶片面积
                from .engine import FILM_FORMATS
                if val not in FILM_FORMATS:
                    raise KeyError(f"未知画幅 {val!r}; 可选 {', '.join(FILM_FORMATS)}")
                kw["film_format"] = val
            else:
                raise KeyError(f"未知的配方后缀 +{mod}")
        return kw
    if name == "cinestill_800t_reference":
        # 物理参考 (docs/ARCHITECTURE.md): 同一张 CineStill 800T 底片, 扫描时按胶片在设计光源 (3200K) 下的中灰
        # 一次定光、不做逐帧自动校色 (操作员变量), 不做输出调整 (冲印店风格)。用来检验胶片物理与扫描硬件。
        return dict(recipe("cinestill_800t"), timing="standard", output_conditioning=None, scanner="frontier_neutral")
    if name == "cinestill_800t_frontier_plain":
        # 冲印店扫描预设 (可选, 不是默认): Frontier 逐帧自动定光, 不加输出调整 (饱和度矩阵 + LAB_TINT)。
        # 依据 EPFL 同店 RAW 配对 (tools/eval/CRITERIA.md "输出调整的拆分"): 不加调整时彩度比 1.04 (加上 0.88),
        # 自动定光误差 11.79 vs 12.32, 目视更接近; 但 ref3 网络总体明显偏好加调整 → 只适用于这类冲印店。
        return dict(recipe("cinestill_800t"), output_conditioning=None)
    if name == "cinestill_800t":
        # CineStill 800T 的实际生产与冲洗:
        #   1. 乳剂 = Kodak Vision3 500T (3200K 灯光平衡, 设计在 ECN-2 中冲洗)
        #   2. 无 remjet 吸光背涂 → 片基界面反射 (仅超过临界角才全反射) → 二次曝光
        #   3. 标称 EI 800, 乳剂本身 500 → 按 800 测光等于欠曝 log2(800/500) ≈ 0.68 档
        #   4. 用 C-41 而非 ECN-2 冲洗 → 显影程度升高 (τ 由公开资料不可辨识, 冻结在 [1.0, 1.3], 见 chemistry.py)
        return dict(film=without_remjet(emulsion("kodak_vision3_500t"), "CineStill 800T"),
                    process="C-41", paper="kodak_portra_endura", exposure_ev=-np.log2(800 / 500))
    if name == "kodak_trix_400":
        # Kodak Tri-X 400 / D-76 (T47): 特性曲线由显式银粒的晶体模型算出, 光谱感光度取 Kodak F-4017
        return dict(film=trix_film())
    if "@" in name:
        # T50 显影液 (t50_bw): "<片>@<显影液>", 例 "kodak_trix_400@rodinal"; 推冲 "kodak_trix_400@d76_ei1600"
        # (按提高后的 EI 拍摄: exposure_ev = −log2(EI/包装感光度))
        film, dev, _ = bw_developer(name)
        kw = dict(film=bw_silver_film(name))
        push = bw_t50_developers()["push"].get(film, {}).get(dev)
        if push:
            kw["exposure_ev"] = float(push["exposure_ev"])
        return kw
    if name in BW_SILVER_FILMS or name in bw_t48_data():
        # T47b: T-Max 100 / 400 (T 颗粒)、Plus-X 125 (三维), D-76; 同一显式银粒模型 (materials.BW_SILVER_FILMS)
        return dict(film=bw_silver_film(name))
    if name.startswith("bw_"):
        # bw_pan, bw_pan_red, bw_ortho, bw_pan_rodinal ...
        parts = name.split("_")[1:]
        sens = parts[0]
        kw = dict(film=bw_film(sensitivity=sens))
        for p in parts[1:]:
            if p in ("rodinal", "hc110", "xtol"):
                kw["process"] = {"rodinal": "BW-Rodinal", "hc110": "BW-HC110", "xtol": "BW-Xtol"}[p]
            else:
                kw["lens_filter"] = lens_filter(p)
        return kw
    if name == "aerochrome":
        return dict(film="kodak_ektachrome_100", lens_filter=lens_filter("yellow"),
                    exposure_hook=near_infrared_estimate())
    if name.startswith("xpro_"):
        # 交叉冲洗: 把胶片放进另一种工艺。正片 → C-41, 负片 → E-6
        src = name[len("xpro_"):]
        e = emulsion(src)
        if e.design == "reversal":
            return dict(film=src, process="C-41")
        # 负片走 E-6 得到的正像很密, 实际操作通常过曝 1–2 档
        return dict(film=src, process="E-6", exposure_ev=1.5)
    if name.startswith("bypass_"):
        return dict(film=name[len("bypass_"):], bleach=0.0)
    if name.startswith(("push", "pull")):
        head, src = name.split("_", 1)
        n = float(head[4:]) * (1 if head.startswith("push") else -1)
        return dict(film=src, push=n, exposure_ev=-n)       # 按提高后的 EI 拍摄
    if name.startswith("expired"):
        # expired15_kodak_gold_200
        head, src = name.split("_", 1)
        return dict(film=aged(emulsion(src), float(head[7:] or 15)))
    if name.startswith("standard_"):
        # 冲印店不做逐张校正 (按标准光号印放): 光源与胶片不匹配的偏色会保留
        return dict(film=name[len("standard_"):], timing="standard")
    return dict(film=name)


RECIPES = [
    # 原生工艺
    "kodak_portra_400", "kodak_portra_160", "kodak_ektar_100", "kodak_gold_200",
    "kodak_ultramax_400", "fujifilm_pro_400h", "fujifilm_c200", "fujifilm_xtra_400",
    "kodak_vision3_50d", "kodak_vision3_250d", "kodak_vision3_500t",
    "fujifilm_velvia_100", "fujifilm_provia_100f", "kodak_ektachrome_100", "kodak_kodachrome_64",
    "bw_pan", "bw_pan_yellow", "bw_pan_red", "bw_ortho", "bw_pan_rodinal",
    # 改变拍摄或冲洗条件
    "push2_kodak_portra_400", "pull1_kodak_portra_400", "push1_fujifilm_velvia_100",
    "xpro_fujifilm_velvia_100", "xpro_kodak_portra_400",
    "bypass_kodak_vision3_250d", "bypass_kodak_portra_400",
    "expired15_kodak_gold_200", "cinestill_800t", "cinestill_800t_reference", "cinestill_800t_frontier_plain", "standard_kodak_vision3_500t",
    "aerochrome",
]

assert all(k in PROCESSES for k in ("C-41", "E-6", "ECN-2"))
