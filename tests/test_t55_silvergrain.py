"""T55: 银盐颗粒独立为 silvergrain 包 (docs/tasks/T55_SILVERGRAIN_STANDALONE.md)。filmlab 只经 filmlab.grain 的薄接口调用它;
silvergrain 自己的胶片数据库与推导必须与 filmlab 的档案逐位相同, 不导入 filmlab。"""
from types import SimpleNamespace

import numpy as np
import pytest

import silvergrain
from silvergrain import database as sgdb, dye as sg_dye, emulsion as sg_em, silver as sg_silver
from filmlab import generation, grain, materials


def test_shim_reexports_silvergrain_objects():
    assert grain.SilverSpec is sg_silver.SilverSpec and grain.GrainSpec is sg_dye.GrainSpec
    assert grain.silver_granularity is sg_silver.silver_granularity and grain.derive_spec is sg_dye.derive_spec
    assert grain._state_table is sg_dye._state_table and grain._nodes is sg_dye._nodes
    assert grain.GRAIN_SCALE_RANGE == (0.5, 3.0) and grain.ETA_REF == 0.05


def test_materials_constants_come_from_silvergrain():
    U = sgdb.universal()
    assert materials.GRAIN_T46_ETA == U["t46"]["eta"] == 0.0248
    assert materials.GRAIN_T47_ETA3D_SCALE == 6.608337
    assert materials.GRAIN_T47_MORPH == dict(spread=4.0, filament_um=0.04, filaments=4, persistence=0.5, callier_q=1.4)
    assert materials.GRAIN_PROCESS["E-6"]["kappa_scale"] == 5.407
    assert materials.TRIX_D76_CURVE[1][0] == 0.014539 and materials.TRIX_GROSS_FOG == 0.318
    assert materials.bw_t48_data() is sgdb.bw_t48()


def test_universal_and_colour_curves_identical():
    U0, U1 = materials.grain_universal(), sg_em.universal()
    assert all(np.array_equal(np.asarray(U0[k]), np.asarray(U1[k])) for k in U0)
    for k in ("kodak_portra_400", "kodak_vision3_500t", "fujifilm_provia_100f"):
        c = silvergrain.film(k).curve()
        e = materials.emulsion(k)
        assert np.array_equal(c.density_curves, e.stock.density_curves) and np.array_equal(c.S_nom, e.S_nom)


@pytest.mark.parametrize("key,dev", [("kodak_trix_400", "d76"), ("kodak_trix_400", "rodinal"), ("kodak_tmax_400", "xtol"),
                                     ("ilford_hp5_plus", "fx39"), ("ilford_delta_100", "d76")])
def test_bw_specs_identical_to_filmlab(key, dev):
    generation.use("t51_bw")
    try:
        fk = key if dev == "d76" else f"{key}@{dev}"
        a = materials.silver_spec(materials.bw_silver_film(fk))
        assert a == silvergrain.derive(key, dev).spec
        assert grain.scale_silver(a, 2.0) == silvergrain.derive(key, dev, 2.0).spec
    finally:
        generation.use("current")


def test_colour_specs_identical_to_filmlab():
    for k in ("kodak_portra_400", "kodak_ektar_100", "fujifilm_provia_100f", "kodak_vision3_50d"):
        assert materials.grain_spec_t46(materials.emulsion(k)) == silvergrain.derive(k).spec


def test_silver_render_identical_to_filmlab_apply_silver():
    sp = silvergrain.derive("kodak_trix_400").spec
    d = np.linspace(0.3, 1.2, 48)[None, :].repeat(40, 0)
    st = SimpleNamespace(silver=np.repeat(d[..., None], 3, -1).copy())
    grain.apply_silver(SimpleNamespace(), sp, st, 12.0, 5)
    r = silvergrain.render(silvergrain.derive("kodak_trix_400"), density=d, px_um=12.0, seed=5, method="explicit")
    assert np.array_equal(st.silver[..., 1], r.density)


def test_dye_apply_identical_to_silvergrain():
    sp = silvergrain.derive("kodak_portra_400").spec
    d = np.full((24, 24, 3), 0.7)
    st = SimpleNamespace(dye=d.copy(), silver=np.zeros_like(d))
    e = SimpleNamespace(has_couplers=True, design="negative")
    grain.apply(e, sp, st, 10.0, np.random.default_rng(3))
    t = d.copy()
    sg_dye.apply_dye_grain(sp, t, 10.0, np.random.default_rng(3))
    assert np.array_equal(st.dye, t)


def test_silver_workers_still_settable():
    sp = silvergrain.derive("kodak_trix_400").spec
    H = np.full((40, 24), 0.02)
    a = grain.render_silver(sp, H, 12.0, 1)
    old = grain.SILVER_WORKERS
    try:
        grain.SILVER_WORKERS = 2
        b = grain.render_silver(sp, H, 12.0, 1)
    finally:
        grain.SILVER_WORKERS = old
    assert np.array_equal(a, b)
