"""冲印店影调: 输出码值 ↔ 底片密度的实测曲线 (data/lab_tone.json; T55b)。冲印店属性 (同 filmlab T44 的黑位), 不是对颗粒的拟合。

曲线写成 code = T(u), u = D − D_118 (该帧码值 118 = sRGB 中灰处的底片密度; 冲印店逐帧定光只平移密度轴)。
帧间合并在码值域 (每个码值上各帧局部斜率的中位, 再积分), 保留颗粒所见的局部斜率。
  · "rpl_noritsu" 黑白: 14 帧 (Tri-X、HP5+) 的帧间中位, 银密度; 彩色: 31 帧 (5 种彩负) 每通道, 模型染料当量密度。
  · 测量范围外 (黑白约码值 16–217) 按端点斜率线性外推到 0 / 255 (外推段在 info 里标明)。
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import database as db

LABS = ("rpl_noritsu",)


@dataclass(frozen=True, eq=False)
class ToneCurve:
    u: np.ndarray               # 底片密度 (相对锚点), 严格递增
    code: np.ndarray            # 8 位码值 (浮点), 严格递增, 两端外推到 0 / 255
    measured: tuple             # (u_min, u_max) 实测覆盖的范围
    label: str = ""

    def code_of(self, u):
        return np.interp(u, self.u, self.code)

    def u_of(self, code):
        return np.interp(code, self.code, self.u)

    def slope(self, u):
        """dCode/du (每单位密度的码值)。"""
        g = np.gradient(self.code, self.u)
        return np.interp(u, self.u, g)


def _extend(u, c, label):
    u = np.asarray(u, float); c = np.asarray(c, float)
    c = c + 1e-6 * np.arange(len(c))                     # 严格递增
    s_lo = (c[2] - c[0]) / (u[2] - u[0]); s_hi = (c[-1] - c[-3]) / (u[-1] - u[-3])
    s_lo, s_hi = max(s_lo, 1.0), max(s_hi, 1.0)
    u0 = u[0] - c[0] / s_lo; u1 = u[-1] + (255.0 - c[-1]) / s_hi
    return ToneCurve(np.concatenate([[u0], u, [u1]]), np.concatenate([[0.0], c, [255.0]]), (float(u[0]), float(u[-1])), label)


@lru_cache(maxsize=None)
def tone(lab="rpl_noritsu", kind="bw"):
    """kind "bw" → (ToneCurve,); "colour" → (R, G, B)。"""
    d = db.load_json("lab_tone.json")
    if lab not in d or "bw" not in d[lab]:
        raise KeyError(f"没有冲印店影调 {lab!r} (可选: {', '.join(LABS)}); {d.get(lab, {}).get('note', '')}")
    L = d[lab]
    if kind == "bw":
        c = L["bw"]["curve"]
        return (_extend(c["u"], c["code"], f"{lab} 黑白"),)
    return tuple(_extend(L["colour"][ch]["curve"]["u"], L["colour"][ch]["curve"]["code"], f"{lab} {ch}") for ch in "RGB")
