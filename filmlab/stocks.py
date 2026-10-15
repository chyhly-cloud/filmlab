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


@lru_cache(maxsize=None)
def load_stock(key: str) -> Stock:
    d = json.loads((PROFILE_DIR / f"{key}.json").read_text())
    info, D = d["info"], d["data"]
    assert np.allclose(D["wavelengths"], WAVELENGTHS)
    logS = np.asarray(D["log_sensitivity"], float)
    logS = np.where(np.isnan(logS), -10.0, logS)  # 测量范围外 = 不感光
    kind = "paper" if info["support"] == "paper" else info["type"]
    return Stock(
        key=key,
        name=info["name"],
        kind=kind,
        log_sensitivity=logS,
        dye_density=np.clip(_clean(D["channel_density"]), 0, None),
        base_density=_clean(D["base_density"]),
        log_exposure=np.asarray(D["log_exposure"], float),
        density_curves=np.clip(np.nan_to_num(np.asarray(D["density_curves"], float)), 0, None),
        reference_illuminant=str(info.get("reference_illuminant", "D55")),
        target_print=info.get("target_print"),
        antihalation=info.get("antihalation", "strong"),
    )


def list_stocks():
    out = {"negative": [], "positive": [], "paper": []}
    for p in sorted(PROFILE_DIR.glob("*.json")):
        s = load_stock(p.stem)
        out[s.kind].append(s.key)
    return out
