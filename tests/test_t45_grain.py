"""T45: 颗粒按银盐物理 (filmlab.grain; 代际 t45_base = t44_base + grain_silver + scan_psf; docs/tasks/T45_GRAIN.md)。
现有代际不变 (旧的 _grain 路径保留); 新模块对所有胶片通用, 只由胶片档案的物理参数驱动。"""
import json
from pathlib import Path

import numpy as np
import pytest

from filmlab import engine, generation, grain, materials
from filmlab.chemistry import FilmState
from filmlab.engine import Engine, Params
from filmlab.materials import bw_film, emulsion, without_remjet
from filmlab.recipes import recipe

EVAL = Path(__file__).resolve().parents[1] / "tools" / "eval"


@pytest.fixture(autouse=True)
def _restore_generation():
    prev = generation.active()
    yield
    generation.use(prev)


def _box_sigma48(a, um):
    """48 µm 圆孔等面积方盒 (42.5 µm) 的 RMS, 由相邻整数盒按面积律插值。"""
    n = 42.5 / um
    vs = []
    for b in {max(1, int(np.floor(n))), int(np.ceil(n))}:
        h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
        m = a[:h, :w].reshape(h // b, b, w // b, b, -1).mean((1, 3))
        vs.append(m.var((0, 1)) * (b * um) ** 2)
    return np.sqrt(np.mean(vs, 0)) / 42.5


def _simulate(e, d0, um, size_um=3000.0, seed=1):
    sp = materials.grain_spec(e)
    N = int(size_um / um)
    st = FilmState(latent=None, halide=None, silver=np.zeros((N, N, 3)), dye=np.full((N, N, 3), d0), coupler=None)
    grain.apply(e, sp, st, um, np.random.default_rng(seed))
    return sp, st


def _analytic(sp, c, d0):
    lh, Dt = grain._state_table(sp, c)
    i = np.argsort(Dt)
    return float(grain.layer_curves(sp, c, 10 ** np.interp(d0, Dt[i], lh[i]))[1])


def test_generation_is_t44_base_plus_grain():
    assert generation.GENERATIONS["t45_base"] == ("D65", generation.GENERATIONS["t44_base"][1] + ("grain_silver", "scan_psf"), {})
    for g, (_, corr, _) in generation.GENERATIONS.items():
        if g not in ("t45_base", "t46_base", "t47_bw", "t48_bw", "t49_bw", "t50_bw", "t51_base", "t51_bw"):   # t46_base = t45_base + grain_t46 (T46); t47_bw = t46_base + grain_t47 (T47)
            assert "grain_silver" not in corr and "scan_psf" not in corr, g


@pytest.mark.parametrize("d0", [0.1, 0.65])
@pytest.mark.parametrize("um", [22.5, 6.6])
def test_simulation_matches_analytic_and_is_resolution_independent(d0, um):
    """模拟的 48 µm RMS = 解析 (Selwyn 面积律, 与渲染分辨率无关); 均值不变。"""
    e = without_remjet(emulsion("kodak_vision3_500t"))
    sp, st = _simulate(e, d0, um)
    s = _box_sigma48(st.dye, um)
    for c in range(3):
        assert abs(st.dye[..., c].mean() - d0) < 0.004
        assert abs(s[c] / _analytic(sp, c, d0) - 1) < 0.12, (c, s[c], _analytic(sp, c, d0))


def test_5219_matches_kodak_datasheet():
    """5219 每层 σ(D) 与 Kodak H-1-5219 的漫射 RMS 颗粒度一致 (log 误差 RMS < 0.1)。"""
    import sys
    sys.path.insert(0, str(EVAL))
    import t45_grain_check as chk
    K, _ = chk.kodak("k5219_gran.json")
    e = emulsion("kodak_vision3_500t")
    for c, (net, s) in K.items():
        D, sg = chk.model_curve(e, c)
        i = np.argsort(D)
        sel = net <= min(D.max(), net.max()) * 0.97
        err = np.log(np.interp(net[sel], D[i], sg[i]) / s[sel])
        assert np.sqrt(np.mean(err ** 2)) < 0.1, c
    # 层间: 蓝层颗粒最粗 (3200K 下蓝光子最少, 同感光度需要更大的晶体)
    sp = materials.grain_spec(e)
    assert sp.ecd_um[2][0] > sp.ecd_um[1][0] > sp.ecd_um[0][0]


def test_generic_films_have_specs():
    """模块通用: 其他片种 (彩色负片 / 黑白 / 反转) 都由档案参数得到 GrainSpec; 慢片晶体更小。"""
    s500 = materials.grain_spec(emulsion("kodak_vision3_500t"))
    s50 = materials.grain_spec(emulsion("kodak_vision3_50d"))
    sbw = materials.grain_spec(bw_film())
    assert s50.ecd_um[1][0] < s500.ecd_um[1][0]
    assert sbw.single_layer and sbw.thickness_um is None
    for k in ("kodak_portra_400", "fujifilm_velvia_100", "kodak_gold_200"):
        sp = materials.grain_spec(emulsion(k))
        assert np.all(np.asarray(sp.amount) >= 0) and np.asarray(sp.amount).sum() > 0


def test_bw_grain_is_one_neutral_field_and_matches_tri_x():
    e = bw_film()
    sp = materials.grain_spec(e)
    N = 64
    st = FilmState(latent=None, halide=None, silver=np.full((N, N, 3), 0.8), dye=np.zeros((N, N, 3)), coupler=None)
    grain.apply(e, sp, st, 10.0, np.random.default_rng(0))
    assert np.array_equal(st.silver[..., 0], st.silver[..., 1]) and np.array_equal(st.silver[..., 1], st.silver[..., 2])
    lh, Dt = grain._state_table(sp, 1)
    i = np.argsort(Dt)
    s1 = float(grain.layer_curves(sp, 1, 10 ** np.interp(1.0, Dt[i], lh[i]))[1])
    assert abs(s1 / 0.017 - 1) < 0.05                    # Kodak F-4017 Tri-X 400: 17


def test_t45_without_grain_equals_t44_at_research_resolution():
    """颗粒关闭、像素 ≥ 16 µm (扫描点扩散不生效) 时 t45 与 t44 逐位相同: 颗粒之外没有改动。"""
    rng = np.random.default_rng(1)
    img = np.full((60, 90, 3), 0.02) + rng.uniform(0, 0.05, (60, 90, 3))
    img[30, 45] = 50.0
    outs = []
    for g in ("t44_base", "t45_base"):
        generation.use(g)
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))
        outs.append(eng.render(img))
    assert np.array_equal(outs[0], outs[1])


def test_scan_psf_constant_and_threshold():
    assert 3.0 < engine.SCAN_PSF_UM < 7.0
    assert engine.SCAN_PSF_UM / 22.5 < 0.3                # 研究分辨率 (22.5 µm/像素) 不生效


def test_t45_grain_on_engine_is_mean_preserving():
    generation.use("t45_base")
    img = np.full((96, 96, 3), 0.05)
    a = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0,
                         "frame_width_mm": 2.0})).develop_negative(img).state.dye
    b = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 1.0,
                         "frame_width_mm": 2.0})).develop_negative(img).state.dye
    assert np.allclose(a.mean((0, 1)), b.mean((0, 1)), atol=0.01)
    assert np.all(b.std((0, 1)) > 0)


def test_digitized_sources_have_urls():
    for f in ("k5219_gran.json", "k5213_gran.json", "k5203_gran.json"):
        d = json.load(open(EVAL / "t45" / f))
        assert d["source"].startswith("http") or "http" in d["source"]
