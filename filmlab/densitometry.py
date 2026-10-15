"""虚拟密度计与灰阶楔: 用冲洗行业自己的检测方法来检验系统。

Status M (负片) / Status A (反转片、相纸) 是 ISO 5-3 规定的密度计光谱响应。
标准的数值表是付费的; 这里按公开的峰值波长用窄带高斯近似 (近似, 仅用于相对比较)。
"""
import numpy as np

from . import chemistry, film
from .engine import Engine, Params
from .stocks import WAVELENGTHS

_PEAKS = {"M": (645.0, 530.0, 445.0), "A": (610.0, 540.0, 440.0)}
_WIDTH = {"M": (15.0, 18.0, 15.0), "A": (12.0, 15.0, 12.0)}


def status_response(kind="M"):
    """(81,3) 列为 R G B 密度计通道。"""
    return np.stack([np.exp(-0.5 * ((WAVELENGTHS - p) / w) ** 2)
                     for p, w in zip(_PEAKS[kind], _WIDTH[kind])], 1)


def measure(transmittance, kind="M"):
    """光谱透射率 (..., 81) → 积分密度 (..., 3)。"""
    R = status_response(kind)
    return -np.log10(np.clip(transmittance @ R / R.sum(0), 1e-12, None))


def step_wedge(params: Params, steps=np.arange(-2.0, 2.01, 0.15)):
    """在参考光源下拍一条中性灰阶楔 (logH 相对中灰), 按 params 的工艺冲洗。
    返回 (logH 偏移, 分析密度 = 每层染料+银, 积分密度 = 密度计读数 − 片基)。"""
    eng = Engine(params)
    rgb = (0.184 * 10.0 ** steps)[:, None] * np.ones(3)
    H = eng._expose_lin(rgb, eng.M_ref, compensation=False)
    st = eng._develop(eng._log_exposure(H))
    analytic = st.dye + st.silver * eng.e.silver_covering
    kind = "A" if eng.polarity == "positive" else "M"
    T = eng._transmittance(st)
    base = 10.0 ** -eng.e.stock.base_density
    integral = measure(T, kind) - measure(base, kind)
    return steps, analytic, integral, eng


def report(params: Params):
    """打印一条灰阶楔的测量结果 (类似冲洗厂的控制条报告)。"""
    steps, an, integ, eng = step_wedge(params)
    print(eng.describe())
    print(" logH   分析密度 R   G    B  |  Status 读数 R   G    B")
    for s, a, i in zip(steps, an, integ):
        print(f"{s:+.2f}  {a[0]:6.2f} {a[1]:5.2f} {a[2]:5.2f} | {i[0]:10.2f} {i[1]:5.2f} {i[2]:5.2f}")
