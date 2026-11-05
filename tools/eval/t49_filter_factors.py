"""T49 独立检验: 滤镜系数 = 中性灰在无滤镜 / 有滤镜时的有效曝光之比 (Kodak Wratten Filters 1965 p.16 的定义:
匹配同一密度所需的曝光倍数; 滤镜不改变特性曲线形状, 所以就是 Σ E·S / Σ E·S·T)。
E = 光源光谱, S = 该片在模型里的光谱感光度 (配方渲染用的同一档案, T48), T = Wratten 透过率 (filmlab/data/wratten_filters.json)。
与数据表公布的系数比较 (tools/eval/t49/filter_factors_published.csv), 误差 = log2(模型 / 公布) 档。

用法: python3 tools/eval/t49_filter_factors.py [输出.json] [代际, 默认 t48_bw; T49b 修正后 t49_bw]
"""
import csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from filmlab import generation, materials                      # noqa: E402
from filmlab.recipes import recipe, wratten, wratten_data     # noqa: E402
from filmlab.spectral import _blackbody, illuminant            # noqa: E402
from filmlab.stocks import WAVELENGTHS                          # noqa: E402

FILMS = ["kodak_trix_400", "kodak_tmax_100", "kodak_tmax_400", "kodak_plusx_125", "ilford_hp5_plus", "ilford_fp4_plus",
         "ilford_delta_100", "ilford_delta_400", "ilford_pan_f_plus", "fuji_acros_100"]
FILTERS = ["8", "11", "12", "15", "21", "25", "47", "58"]
LIGHTS = {"daylight": ["D55", "D65", "BB5500"], "tungsten": ["BB3200", "A"]}


def spec(name):
    return _blackbody(float(name[2:])) if name.startswith("BB") else illuminant(name)


def sensitivity(film, gen="t48_bw"):
    generation.use(gen)
    r = recipe(film)["film"]
    e = r if not isinstance(r, str) else materials.emulsion(r)
    return 10.0 ** e.stock.log_sensitivity[:, 1]


def factor(S, E, f):
    return float((E * S).sum() / (E * S * wratten(f)).sum())


def published():
    out = {}
    with open(ROOT / "tools/eval/t49/filter_factors_published.csv") as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")):
            out.setdefault(r["light"], {}).setdefault(r["film"], {})[r["filter"]] = float(r["factor"])
    return out


def ilford_variant(film, planck=True, k=1.0):
    """诊断: 从 T48 原始读数重建 Ilford 光谱, 纵轴尺度 k 倍、是否扣 2850K 光源 (T48 的做法: 扣, k = 1)。"""
    lam, v = [], []
    with open(ROOT / "tools/eval/t48/bw_spectral_raw.csv") as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")):
            if r["film"] == film:
                lam.append(float(r["wavelength_nm"])); v.append(float(r["raw_value"]))
    lam, v = np.array(lam), k * np.array(v)
    ls = np.interp(WAVELENGTHS, lam, v)
    if planck:
        P = np.log10(_blackbody(2850.0)); ls = ls - (P - P.max())
    return np.where(WAVELENGTHS <= lam.max(), 10.0 ** ls, 0.0)


def main(out=None, gen="t48_bw"):
    pub = published()
    S = {f: sensitivity(f, gen) for f in FILMS}
    print("代际", gen)
    res = dict(model={}, errors={})
    for light, names in LIGHTS.items():
        for name in names:
            E = spec(name)
            res["model"][name] = {f: {k: factor(S[f], E, k) for k in FILTERS} for f in FILMS}
            errs = [(f, k, np.log2(res["model"][name][f][k] / p)) for f, d in pub[light].items() if f in S for k, p in d.items()]
            e = np.array([x[2] for x in errs])
            res["errors"][name] = dict(n=len(e), mean=float(e.mean()), rms=float(np.sqrt((e ** 2).mean())),
                                       over_third=int((abs(e) > 1 / 3).sum()), items=[[f, k, float(x)] for f, k, x in errs])
            print(f"{light:8s} {name:6s} n={len(e)} 平均 {e.mean():+.2f} RMS {np.sqrt((e**2).mean()):.2f} 档, |误差|>1/3 档 {int((abs(e) > 1/3).sum())}")
    print("\n日光 D55: 模型系数 / 公布 (误差 档)")
    for f in FILMS:
        p = pub["daylight"].get(f, {})
        print(f"  {f:18s} " + " ".join(f"{k}:{res['model']['D55'][f][k]:5.2f}" + (f"/{p[k]:<4g}{np.log2(res['model']['D55'][f][k] / p[k]):+.2f}" if k in p else " " * 10) for k in FILTERS))
    # 诊断: Ilford 光谱的读法
    res["ilford_diag"] = {}
    E = spec("D55")
    for f in ("ilford_hp5_plus", "ilford_fp4_plus"):
        for planck in (True, False):
            for k in (1.0, 2.0):
                Sv = ilford_variant(f, planck, k)
                e = [np.log2(factor(Sv, E, n) / p) for n, p in pub["daylight"][f].items()]
                key = f"{f} planck={planck} k={k}"
                res["ilford_diag"][key] = dict(mean=float(np.mean(e)), rms=float(np.sqrt(np.mean(np.square(e)))))
                print(f"  诊断 {key:40s} 平均 {np.mean(e):+.2f} RMS {np.sqrt(np.mean(np.square(e))):.2f}")
    if out:
        json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else "t48_bw")
