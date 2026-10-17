"""配方 = 材料 + 冲洗工艺 + 拍摄条件。

这里不包含任何"调色"——每个配方只是在描述一个真实存在的组合:
哪种胶片、在什么光下用什么滤镜拍、送进哪种冲洗流程、印在哪种相纸上。
"""
import numpy as np

from .chemistry import PROCESSES
from .materials import aged, bw_film, emulsion, without_remjet
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
    return table[kind]() if kind in table else np.ones(81)


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
    if name == "cinestill_800t_reference":
        # 物理参考 (docs/ARCHITECTURE.md): 同一张 CineStill 800T 底片, 扫描时按胶片在设计光源 (3200K) 下的中灰
        # 一次定光、不做逐帧自动校色 (操作员变量), 不做输出调整 (冲印店风格)。用来检验胶片物理与扫描硬件。
        return dict(recipe("cinestill_800t"), timing="standard", output_conditioning=None, scanner="frontier_neutral")
    if name == "cinestill_800t":
        # CineStill 800T 的实际生产与冲洗:
        #   1. 乳剂 = Kodak Vision3 500T (3200K 灯光平衡, 设计在 ECN-2 中冲洗)
        #   2. 出厂前去掉 remjet 背涂 → 片基背面全反射 → 红色光环
        #   3. 标称 EI 800, 乳剂本身 500 → 按 800 测光等于欠曝 log2(800/500) ≈ 0.68 档
        #   4. 用 C-41 而非 ECN-2 冲洗 → 显影程度升高 (τ 由公开资料不可辨识, 冻结在 [1.0, 1.3], 见 chemistry.py)
        return dict(film=without_remjet(emulsion("kodak_vision3_500t"), "CineStill 800T"),
                    process="C-41", paper="kodak_portra_endura", exposure_ev=-np.log2(800 / 500))
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
    "expired15_kodak_gold_200", "cinestill_800t", "cinestill_800t_reference", "standard_kodak_vision3_500t",
    "aerochrome",
]

assert all(k in PROCESSES for k in ("C-41", "E-6", "ECN-2"))
