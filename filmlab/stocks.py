"""胶片 / 相纸数据加载。

数据来自 spektrafilm 项目（Andrea Volpato, CC BY-SA 4.0），
由厂商 datasheet 的光谱感光度、染料光谱密度、特性曲线数字化而来。
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import json

import numpy as np

PROFILE_DIR = Path(__file__).parent / "data" / "profiles"
WAVELENGTHS = np.arange(380.0, 781.0, 5.0)  # 81 个波段, 与数据一致


@dataclass(frozen=True)
class Stock:
    key: str
    name: str
    kind: str                    # "negative" | "positive" (反转片) | "paper"
    log_sensitivity: np.ndarray  # (81, 3) 每层乳剂 log10 光谱感光度, 列顺序 R G B 层
    dye_density: np.ndarray      # (81, 3) C M Y 染料的光谱密度 (单位浓度)
    base_density: np.ndarray     # (81,) 片基 + 橙色遮罩 (负片) 的光谱密度
    log_exposure: np.ndarray     # (256,) 特性曲线横轴
    density_curves: np.ndarray   # (256, 3) 特性曲线 (H&D), 各层染料浓度 vs log H
    reference_illuminant: str
    target_print: str | None
    antihalation: str
    bw: bool = False             # 黑白: 银密度光谱平坦

    @property
    def is_positive(self):
        return self.kind == "positive"

    def with_(self, **kw):
        return replace(self, **kw)


def _clean(a):
    """datasheet 只覆盖约 400–700nm, 测量范围外按边缘值外推 (不能当作"透明")。"""
    a = np.array(a, dtype=float)
    cols = a if a.ndim == 2 else a[:, None]
    for c in range(cols.shape[1]):
        ok = ~np.isnan(cols[:, c])
        cols[:, c] = np.interp(WAVELENGTHS, WAVELENGTHS[ok], cols[ok, c]) if ok.any() else 0.0
    return a


# 胶片数据的修订 (按管线代际启用, 见 filmlab.generation): 档案键 → 修订函数
REVISIONS = {}


def _kodak_raw_5219(st: Stock) -> Stock:
    """Vision3 5219: 改用 Kodak 原始测量的染料光谱 (保留有色成色剂遮罩的负值), 并按 Kodak 实测的 Status M 特性曲线
    重新解混分层曲线 (tools/eval/unmix_5219.py)。独立验证: 合成的中性中间调光谱密度与 Kodak 实测相差 RMS 0.006 D
    (spektrafilm 解混 + 截断谱为 0.026 D, 且整片 Status M 的蓝 / 红反差比 Kodak 实测高约 10% / 8%)。
    Kodak 曲线覆盖 log H −2.38…+2.41; 超出部分沿用原档案曲线的形状, 在边界处连续衔接。"""
    k = np.load(PROFILE_DIR / "kodak_vision3_500t_kodak_raw.npz")
    x, C = k["x"], k["C"]
    xs, D0 = st.log_exposure, st.density_curves
    D = np.empty_like(D0)
    for c in range(3):
        inside = np.interp(xs, x, C[:, c])
        lo = D0[:, c] - np.interp(x[0], xs, D0[:, c]) + C[0, c]
        hi = D0[:, c] - np.interp(x[-1], xs, D0[:, c]) + C[-1, c]
        D[:, c] = np.where(xs < x[0], lo, np.where(xs > x[-1], hi, inside))
    return st.with_(dye_density=np.asarray(k["E"], float), density_curves=np.clip(D, 0, None))


@lru_cache(maxsize=None)
def load_stock(key: str) -> Stock:
    st = _load_stock(key)
    rev = REVISIONS.get(key)
    return REVISIONS_FN[rev](st) if rev else st


REVISIONS_FN = {"kodak_raw": _kodak_raw_5219}


def _load_stock(key: str) -> Stock:
    d = json.loads((PROFILE_DIR / f"{key}.json").read_text())
    info, D = d["info"], d["data"]
    assert np.allclose(D["wavelengths"], WAVELENGTHS)
    logS = np.asarray(D["log_sensitivity"], float)
    logS = np.where(np.isnan(logS), -10.0, logS)  # 测量范围外 = 不感光
    kind = "paper" if info["support"] == "paper" else info["type"]
    ref = str(info.get("reference_illuminant", "D55"))
    if ref == "T":
        logS, ref = _rebalance_tungsten(logS), "BB3200"
    return Stock(
        key=key,
        name=info["name"],
        kind=kind,
        log_sensitivity=logS,
        dye_density=np.clip(_clean(D["channel_density"]), 0, None),
        base_density=_clean(D["base_density"]),
        log_exposure=np.asarray(D["log_exposure"], float),
        density_curves=np.clip(np.nan_to_num(np.asarray(D["density_curves"], float)), 0, None),
        reference_illuminant=ref,
        target_print=info.get("target_print"),
        antihalation=info.get("antihalation", "strong"),
    )


def _rebalance_tungsten(logS):
    """灯光片的感光度平衡: Kodak 数据手册 (如 5219) 规定灯光片按 3200K 钨丝灯平衡、感光测定也用
    3200K 曝光。spektrafilm 档案在归一化时用的参考光源 "T" 是 colour 库的 Incandescent 光谱
    (约 2813K), 按它平衡后胶片在 2813K 下才中性, 在真正的 3200K 下偏冷。
    这里按 3200K 黑体重新平衡: 中灰在 3200K 下三层曝光之比与原档案在 "T" 下相同。"""
    import colour
    from . import spectral
    S = 10.0 ** logS
    inc = colour.SDS_LIGHT_SOURCES["Incandescent"].copy().align(colour.SpectralShape(380, 780, 5)).values
    bb = spectral.illuminant("BB3200")
    r_t = (inc[:, None] * S).sum(0) / inc.sum()
    r_b = (bb[:, None] * S).sum(0) / bb.sum()
    k = (r_t / r_t[1]) / (r_b / r_b[1])            # 让 3200K 下的通道比 = 原来 T 下的通道比
    return logS + np.log10(k)[None, :]


def list_stocks():
    out = {"negative": [], "positive": [], "paper": []}
    for p in sorted(PROFILE_DIR.glob("*.json")):
        s = load_stock(p.stem)
        out[s.kind].append(s.key)
    return out
