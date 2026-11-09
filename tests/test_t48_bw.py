"""T48: 黑白片各自的光谱感光度与 D-76 类特性曲线 (silvergrain/data/bw_t48.json, T55 前在 filmlab/data/; 代际 t48_bw; docs/tasks/T48_BW_SPECTRAL_CURVES.md)。"""
import numpy as np
import pytest

from filmlab import generation, grain, materials
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe

FILMS = ["kodak_trix_400", "kodak_tmax_100", "kodak_tmax_400", "kodak_plusx_125", "ilford_hp5_plus", "ilford_fp4_plus",
         "ilford_delta_100", "ilford_delta_400", "ilford_pan_f_plus", "fuji_acros_100"]


@pytest.fixture(autouse=True)
def _restore_generation():
    prev = generation.active()
    yield
    generation.use(prev)


def test_generation_t48_bw():
    assert generation.GENERATIONS["t48_bw"] == generation.GENERATIONS["t47_bw"]
    generation.use("t48_bw"); assert materials.BW_PROFILE_SET == "t48"
    generation.use("t47_bw"); assert materials.BW_PROFILE_SET == "t47"
    generation.use("t46_base"); assert materials.BW_PROFILE_SET == "t47"


def test_data_complete_and_sourced():
    D = materials.bw_t48_data()
    assert set(D) == set(FILMS)
    for k, d in D.items():
        lam = np.asarray(d["spectral"]["lam"]); ls = np.asarray(d["spectral"]["logS"])
        assert lam.min() <= 405 and lam.max() >= 640 and np.isclose(ls.max(), 0, atol=1e-3), k
        assert d["spectral"]["source"] and d["ei"] > 0
        if d["curve"] is not None:
            assert d["curve"]["source"] and d["curve"]["process"]
            assert np.all(np.diff(d["curve"]["net"]) >= -1e-9) and max(d["curve"]["net"]) > 1.0
    assert D["kodak_plusx_125"]["spectral"].get("borrowed")


@pytest.mark.parametrize("key", FILMS)
def test_t48_curves_fit_and_budget(key):
    """每种片: 晶体模型 (漫射) 复现自己的曲线 |ΔD| ≤ 0.10, 银量上限高于曲线最高一级; 官方感光度。"""
    generation.use("t48_bw")
    e = materials.bw_silver_film(key)
    sp = materials.silver_spec(e)
    f = materials.bw_film_def(key)
    x, net = map(np.asarray, f["curve"])
    H = 0.8 / f["ei"] * 10 ** x
    model = grain.silver_density(sp, H) - grain.silver_density(sp, 0.0)
    assert np.max(np.abs(model - net)) <= 0.10
    assert grain.silver_budget(sp)["D_max"] - grain.silver_density(sp, 0.0) > net.max()
    assert f["ei"] == float(materials.bw_t48_data()[key]["ei"])


def test_trix_identical_in_t47_and_t48_and_tmax_t47_unchanged():
    generation.use("t47_bw")
    a = materials.silver_spec(materials.bw_silver_film("kodak_trix_400"))
    t47 = materials.silver_spec(materials.bw_silver_film("kodak_tmax_100"))
    s47 = materials.bw_silver_film("kodak_tmax_100").stock.log_sensitivity
    generation.use("t48_bw")
    b = materials.silver_spec(materials.bw_silver_film("kodak_trix_400"))
    t48 = materials.silver_spec(materials.bw_silver_film("kodak_tmax_100"))
    s48 = materials.bw_silver_film("kodak_tmax_100").stock.log_sensitivity
    assert a == b
    assert t47.ei == 200.0 and t48.ei == 100.0          # t47_bw: Schmidt 有效感光度; t48_bw: 官方感光度
    assert not np.allclose(s47, s48)


def test_spectral_differences_reach_the_image():
    """同一张红 / 绿 / 蓝色块, 不同光谱感光度的片给出不同的相对曝光 (定光前)。"""
    generation.use("t48_bw")
    img = np.zeros((24, 72, 3)); img[:, :24] = (0.4, 0.05, 0.05); img[:, 24:48] = (0.05, 0.3, 0.05); img[:, 48:] = (0.05, 0.05, 0.4)
    rel = {}
    for k in ("kodak_trix_400", "ilford_delta_400", "fuji_acros_100"):
        e = Engine(Params(**{**recipe(k), "roll": None, "frame": 0, "grain": 0, "scene_lights": None}))
        H = e._expose_lin(img, e.M)[0, :, 1]
        rel[k] = np.log10([H[:24].mean() / H[24:48].mean(), H[48:].mean() / H[24:48].mean()])
    assert abs(rel["kodak_trix_400"][0] - rel["ilford_delta_400"][0]) > 0.03 or abs(rel["kodak_trix_400"][1] - rel["fuji_acros_100"][1]) > 0.03


def test_t48_new_films_render_with_grain():
    generation.use("t48_bw")
    img = np.full((32, 32, 3), 0.18)
    for k in ("ilford_hp5_plus", "fuji_acros_100"):
        s = Engine(Params(**{**recipe(k), "timing": "frame", "roll": None, "frame": 0, "grain": 1.0,
                             "frame_width_mm": 0.3})).develop_negative(img).state.silver
        assert s[..., 1].std() > 0.003
