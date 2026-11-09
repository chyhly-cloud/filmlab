"""T49: 黑白摄影的镜头滤镜 (Kodak Wratten 实测透过率) 与按公布滤镜系数的曝光补偿。"""
import sys
from pathlib import Path

import numpy as np
import pytest

from filmlab import generation
from filmlab.engine import GRAY, Engine, Params
from filmlab.recipes import filter_factor, lens_filter, recipe, wratten, wratten_data, wratten_number
from filmlab.stocks import WAVELENGTHS

ROOT = Path(__file__).resolve().parents[1]
FILTERS = ["8", "11", "12", "15", "21", "25", "47", "58"]


def _engine(name, **kw):
    generation.use("t48_bw")
    return Engine(Params(**{**recipe(name), "timing": "frame", "roll": None, "frame": 0, "grain": 0, **kw}))


def test_old_approximate_filters_unchanged():
    # bw_pan_yellow 等旧配方用的近似曲线不变 (现有代际逐位不变的前提)
    lp = lambda cut, w=12: 1.0 / (1.0 + np.exp(-(WAVELENGTHS - cut) / w))
    assert np.array_equal(lens_filter("yellow"), lp(495))
    assert np.array_equal(lens_filter("red"), lp(595))
    assert np.array_equal(lens_filter("85"), 1.0 - 0.45 / (1.0 + np.exp((WAVELENGTHS - 520) / 25)))
    assert np.array_equal(lens_filter("unknown"), np.ones(81))


def test_wratten_data_and_aliases():
    assert sorted(wratten_data()["filters"], key=int) == FILTERS
    for k in FILTERS:
        t = wratten(k)
        assert t.shape == (81,) and t.min() >= 0 and t.max() <= 0.92
    # 表中读数 (Kodak Wratten Filters 1965)
    assert wratten("8")[WAVELENGTHS == 550][0] == pytest.approx(0.884)
    assert wratten("25")[WAVELENGTHS == 590][0] == pytest.approx(0.126)
    assert wratten("47")[WAVELENGTHS == 380][0] == pytest.approx(0.010)
    assert wratten("58")[WAVELENGTHS == 530][0] == pytest.approx(0.536)
    # 截止型滤镜在短波为 0, 长于 700 nm 保持 700 nm 的值
    assert wratten("25")[WAVELENGTHS < 580].max() == 0
    assert wratten("25")[-1] == pytest.approx(0.895)
    for a, n in (("K2", "8"), ("X1", "11"), ("G", "15"), ("A", "25"), ("w58", "58"), ("wratten_47", "47"), ("21", "21")):
        assert wratten_number(a) == n
        assert np.array_equal(lens_filter(a), wratten(n))
    assert wratten_number("29") is None


def test_filter_factor_lookup():
    assert filter_factor("kodak_trix_400", "25") == (8.0, "kodak_trix_400")
    assert filter_factor("kodak_tmax_100", "8") == (1.5, "kodak_tmax_100")
    assert filter_factor("ilford_hp5_plus", "25") == (4.0, "ilford_hp5_plus")
    assert filter_factor("fuji_acros_100", "21") == (4.0, "fuji_acros_100")
    assert filter_factor("kodak_trix_400", "8", "tungsten") == (1.5, "kodak_trix_400")
    # 数据表没给的: Kodak 全色片通用表, 再没有就取其他片的中位数
    assert filter_factor("ilford_delta_100", "15") == (3.0, "kodak_panchromatic_class")
    assert filter_factor("kodak_trix_400", "21") == (3.0, "median")       # HP5 2.6 / FP4 3 / Acros 4
    assert filter_factor("ilford_hp5_plus", "47") == (7.0, "median")      # 6 / 8 / 10 / 6


def test_recipe_syntax():
    r = recipe("kodak_trix_400+w25")
    assert np.array_equal(r["lens_filter"], wratten("25")) and r["filter_factor"] == 8.0
    r = recipe("ilford_hp5_plus+wK2")
    assert np.array_equal(r["lens_filter"], wratten("8")) and r["filter_factor"] == 1.7
    assert "lens_filter" not in recipe("kodak_trix_400") and "filter_factor" not in recipe("kodak_trix_400")
    with pytest.raises(KeyError):
        recipe("kodak_trix_400+w29")


def test_exposure_compensation_by_published_factor():
    # 摄影师的做法: 不带滤镜测光 → 与无滤镜同一 log_k, 再加 log10(公布系数)
    e0 = _engine("kodak_trix_400")
    for w, F in (("8", 2.0), ("25", 8.0), ("47", 6.0)):
        e = _engine(f"kodak_trix_400+w{w}")
        assert e.p.filter_factor == F
        assert e.log_k - e0.log_k == pytest.approx(np.log10(F), abs=1e-12)
        # 中性灰的曝光 = 无滤镜 × 公布系数 / 该片在拍摄光下的实际系数
        g = np.full((1, 1, 3), GRAY)
        H0 = e0._expose_lin(g, e0.M)[0, 0, 1] * 10 ** e0.log_k
        H = e._expose_lin(g, e.M)[0, 0, 1] * 10 ** e.log_k
        actual = H0 / H * F
        assert 1.0 < actual < 20.0


def test_ttl_metering_kept_when_no_factor():
    # filter_factor=None 时仍是原来的透过滤镜测光 (中性灰在参考光源下曝光与无滤镜相同)
    e0 = _engine("kodak_trix_400")
    e = _engine("kodak_trix_400", lens_filter=wratten("25"))
    g = np.full((1, 1, 3), GRAY)
    a = e0._expose_lin(g, e0.M_ref)[0, 0, 1] * 10 ** e0.log_k
    b = e._expose_lin(g, e.M_ref)[0, 0, 1] * 10 ** e.log_k
    assert a == pytest.approx(b, rel=1e-9)


def test_filter_darkens_complementary_colour():
    # 红滤镜: 红物体变亮、蓝物体变暗 (相对中性灰); 蓝滤镜相反
    e0, er, eb = _engine("kodak_trix_400"), _engine("kodak_trix_400+w25"), _engine("kodak_trix_400+w47")
    cols = np.array([[[0.18, 0.18, 0.18], [0.40, 0.04, 0.03], [0.03, 0.06, 0.40]]])
    rel = lambda e: (lambda H: H / H[0])(e._expose_lin(cols, e.M)[0, :, 1])
    r0, rr, rb = rel(e0), rel(er), rel(eb)
    assert rr[1] > 2 * r0[1] and rr[2] < 0.5 * r0[2]
    assert rb[1] < 0.5 * r0[1] and rb[2] > 1.3 * r0[2]


def test_filter_factor_validation_regression():
    # 独立检验 (tools/eval/t49_filter_factors.py): Tri-X 与 T-Max 400 (Kodak 自己的光谱与系数) 日光 D55 下
    # 除 Tri-X No. 58 外都在 1/3 档以内; 全部 40 项的 RMS 记录在文档里 (0.59 档)
    sys.path.insert(0, str(ROOT / "tools" / "eval"))
    import t49_filter_factors as T
    pub = T.published()["daylight"]
    E = T.spec("D55")
    for film in ("kodak_trix_400", "kodak_tmax_400"):
        S = T.sensitivity(film)
        for k, p in pub[film].items():
            err = np.log2(T.factor(S, E, k) / p)
            if (film, k) == ("kodak_trix_400", "58"):
                assert 0.5 < err < 0.8
            else:
                assert abs(err) < 1 / 3, (film, k, err)


# ---------------------------------------------------------------- T49b --
def test_t49b_generation_and_profile_set():
    from filmlab import materials
    assert generation.GENERATIONS["t49_bw"] == generation.GENERATIONS["t48_bw"]
    assert generation.BW_PROFILES["t49_bw"] == "t49" and generation.BW_PROFILES["t48_bw"] == "t48"
    generation.use("t49_bw")
    assert materials.BW_PROFILE_SET == "t49"
    generation.use("t48_bw")
    assert materials.BW_PROFILE_SET == "t48"


def test_t49b_only_tmax100_and_ilford_change():
    import json
    from filmlab import materials
    films = json.loads((ROOT / "silvergrain/data/bw_t49_spectra.json").read_text())["films"]
    assert sorted(films) == sorted(["kodak_tmax_100", "ilford_hp5_plus", "ilford_fp4_plus", "ilford_delta_100",
                                    "ilford_delta_400", "ilford_pan_f_plus"])
    S = {}
    for g in ("t48_bw", "t49_bw"):
        generation.use(g)
        S[g] = {k: materials.bw_silver_film(k).stock.log_sensitivity.copy() for k in
                ("kodak_trix_400", "kodak_tmax_100", "kodak_tmax_400", "kodak_plusx_125", "fuji_acros_100", "ilford_hp5_plus")}
        S[g + "_spec"] = materials._silver_spec("ilford_hp5_plus", "bw", "BW-D76", materials._curve_set(materials.bw_profile_key("ilford_hp5_plus")))
    for k in ("kodak_trix_400", "kodak_tmax_400", "kodak_plusx_125", "fuji_acros_100"):
        assert np.array_equal(S["t48_bw"][k], S["t49_bw"][k])
    for k in ("kodak_tmax_100", "ilford_hp5_plus"):
        assert not np.array_equal(S["t48_bw"][k], S["t49_bw"][k])
    # 曲线与晶体 / 银档案不变 (只改光谱): 同一个缓存对象
    assert S["t48_bw_spec"] is S["t49_bw_spec"]


def test_t49b_ilford_single_wedge_constant():
    import csv, json
    d = json.loads((ROOT / "silvergrain/data/bw_t49_spectra.json").read_text())
    k = d["ilford_wedge_constant"]["k"]
    assert 2.5 < k < 3.5
    raw = {}
    for r in csv.DictReader(l for l in open(ROOT / "tools/eval/t48/bw_spectral_raw.csv") if not l.startswith("#")):
        raw.setdefault(r["film"], {})[float(r["wavelength_nm"])] = float(r["raw_value"])
    planck = lambda l: 1 / ((l * 1e-9) ** 5 * (np.exp(1.4388e-2 / (l * 1e-9 * 2850.0)) - 1))
    for f in ("ilford_hp5_plus", "ilford_fp4_plus", "ilford_delta_100", "ilford_delta_400", "ilford_pan_f_plus"):
        sp = d["films"][f]
        lam = np.array(sp["lam"]); v = np.array([raw[f][l] for l in lam])
        ls = k * v - np.log10(planck(lam))
        assert np.allclose(np.array(sp["logS"]), ls - ls.max(), atol=1e-4)      # 同一个 k, 不按片选读法
        assert sp["predicted"] == (f not in ("ilford_hp5_plus", "ilford_fp4_plus"))
    assert d["films"]["kodak_tmax_100"]["definition"].startswith("log10 1/(erg/cm²) 达到 D-min+0.3")


def test_t49b_filter_factor_check_improves():
    sys.path.insert(0, str(ROOT / "tools" / "eval"))
    import t49_filter_factors as T
    pub = T.published()["daylight"]
    E = T.spec("D55")
    def rms(gen, films):
        e = [np.log2(T.factor(T.sensitivity(f, gen), E, k) / p) for f in films for k, p in pub[f].items()]
        return float(np.sqrt(np.mean(np.square(e))))
    films = [f for f in T.FILMS if f in pub]
    assert rms("t49_bw", films) < rms("t48_bw", films) - 0.1           # 0.59 → 0.42 档
    assert rms("t49_bw", ["kodak_tmax_100"]) < rms("t48_bw", ["kodak_tmax_100"])
    assert rms("t49_bw", ["ilford_hp5_plus", "ilford_fp4_plus"]) == pytest.approx(0.643, abs=0.01)
