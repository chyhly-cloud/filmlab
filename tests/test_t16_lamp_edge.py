"""T16: blue layer mapped in XYZ (non-negative), edge-colour reconstruction, defaults unchanged."""
import numpy as np
import pytest
from filmlab import generation, lights, rawio
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe


@pytest.fixture(autouse=True)
def _restore():
    prev = generation.active()
    yield
    generation.use(prev)


def _eng(gen):
    generation.use(gen)
    return Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))


def test_blue_xyz_keeps_red_green_and_is_nonnegative_for_out_of_gamut():
    eng = _eng("next_lab_halo")
    sens = 10.0 ** eng.e.stock.log_sensitivity
    spec = lights.pick_members(eng.v.frame_seed)["hps"][1]
    fs = lights.linear_response(spec, sens, True, "srgb")
    fb = lights.linear_response(spec, sens, True, "srgb_bxyz")
    x = np.array([[1.29, 1.01, -0.09], [0.4, 0.3, 0.2]])
    np.testing.assert_array_equal(fb(x)[:, :2], fs(x)[:, :2])
    assert fs(x)[0, 2] < 0 < fb(x)[0, 2]           # sRGB 映射给出负蓝层曝光, XYZ 蓝层为正


def test_edge_colour_matches_boundary():
    rgb = np.full((40, 40, 3), 0.02)
    yy, xx = np.mgrid[:40, :40]
    r = np.hypot(yy - 20, xx - 20)
    rgb[r < 12] = [1.3, 1.0, -0.09]                  # 灯边实测颜色 (sRGB 色域外)
    clip = (r < 8).astype(float)
    rgb[r < 8] = 1.0
    prev = rawio.RECON_COLOUR
    try:
        rawio.RECON_COLOUR = "edge"
        out = rawio.reconstruct_highlights(rgb, clip, np.random.default_rng(0), None)
    finally:
        rawio.RECON_COLOUR = prev
    c_in = out[20, 20] / (out[20, 20] @ rawio.LUMA)
    c_edge = np.array([1.3, 1.0, -0.09]) / (np.array([1.3, 1.0, -0.09]) @ rawio.LUMA)
    np.testing.assert_allclose(c_in, c_edge, rtol=1e-6)


def test_production_generations_unaffected():
    for g in ("current", "next", "next_lab_halo"):
        eng = _eng(g)
        assert "blue_xyz" not in eng.p.corrections
        assert rawio.RECON_COLOUR == "ring"
