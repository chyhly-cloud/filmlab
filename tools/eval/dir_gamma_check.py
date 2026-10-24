"""DIR 层间效应的独立口径 (T01 验收): 分色 gamma / 中性 gamma, 以 Status M 积分密度、在直线段取平均斜率。

专利 (Kodak US 6,210,870) 的比值是扫描/测量密度上的特性曲线斜率。这里:
  · 用 CineStill 实际冲洗 (C-41) 的完整负片透射率 (染料谱 + 片基 + 残银/卤化银) 计 Status M 密度;
  · 中性: 三层同 logH; 分色: 只扫目标层, 其余层固定在曝光轴最低值;
  · gamma = 该层 Status M 密度在"高于最低密度 0.3–1.3 D"区间对 logH 的最小二乘斜率 (直线段, 不用最大瞬时斜率)。
用法: python3 dir_gamma_check.py [参数.json]   (缺省 = 现值)
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools/eval"), str(REPO)]
RANGES = {"R": (1.15, 1.49), "G": (1.42, 1.64), "B": (1.29, 1.50)}


def measure(params):
    import accept_calibration as acc
    acc.apply(params)
    from filmlab import densitometry
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    eng = Engine(Params(**{**recipe("cinestill_800t"), "roll": None, "grain": 0}))
    x = np.linspace(eng.e.log_exposure[0], eng.e.log_exposure[-1], 1201)

    def sm(logH):
        st = eng._develop(logH[None])
        T = eng._transmittance(st)[0]
        return densitometry.measure(T, "M")

    def gamma(D, c):
        d = D[:, c] - D[:, c].min()
        k = (d > 0.3) & (d < 1.3)
        return float(np.polyfit(x[k], D[k, c], 1)[0]) if k.sum() > 10 else np.nan
    Dn = sm(np.repeat(x[:, None], 3, 1))
    out = {}
    for c, name in enumerate("RGB"):
        q = np.full((len(x), 3), x[0]); q[:, c] = x
        out[name] = gamma(sm(q), c) / gamma(Dn, c)
    return out


if __name__ == "__main__":
    params = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    r = measure(params)
    print(json.dumps({k: round(v, 3) for k, v in r.items()}), "范围内:",
          all(RANGES[k][0] <= v <= RANGES[k][1] for k, v in r.items()))
