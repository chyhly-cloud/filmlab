"""T44: 冲印店输出的逐通道黑位 (lab_shadow; 代际 t44_base = t31_base + lab_shadow; docs/tasks/T44_LAB_SHADOW_TONE.md)。
现有代际不变; 新项是输出级的逐通道仿射 β + (1 − β)·out (白不变、黑抬到实测值), 不改变定光与负片。"""
import numpy as np
import pytest

from filmlab import engine, film, generation, rawio
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe


@pytest.fixture(autouse=True)
def _restore_generation():
    prev = generation.active()
    yield
    generation.use(prev)


def _engine(g):
    generation.use(g)
    return Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))


def test_generation_is_t31_base_plus_lab_shadow():
    assert generation.GENERATIONS["t44_base"] == ("D65", generation.GENERATIONS["t31_base"][1] + ("lab_shadow",), {})
    generation.use("t44_base")
    assert (rawio.METERING, rawio.HIGHLIGHT_RECON, rawio.RECON_COLOUR, rawio.APERTURE_RANGE, film.MASK_PLACEMENT,
            rawio.STOPS_PRIOR) == ("session", "stops", "ring", None, "layers", (6.0, 8.0))
    for g, (_, corr, _) in generation.GENERATIONS.items():
        if g not in ("t44_base", "t45_base", "t46_base", "t47_bw", "t48_bw", "t49_bw", "t50_bw", "t51_base", "t51_bw", "t56_base", "t56_bw"):   # t45/t46_base、t47_bw = t44_base + 颗粒 (T45 / T46 / T47)
            assert "lab_shadow" not in corr, g


def test_measured_constants():
    b = np.asarray(engine.LAB_SHADOW_BLACK)
    assert b.shape == (3,) and np.all(b >= 0) and np.all(b < 0.01)
    assert b[0] > b[1]                         # 片基附近偏品红: R 抬得比 G 多 (T40 G/R 0.54–0.64)


def test_existing_generation_has_no_shadow_black():
    eng = _engine("t31_base")
    assert eng.scanner.shadow_black is None


def test_output_is_per_channel_affine_of_t31():
    rgb = np.array([[[0.40, 0.05, 0.03], [0.03, 0.10, 0.35], [1e-4, 1e-4, 1e-4], [0.18, 0.18, 0.18]]]) * np.ones((6, 1, 1))
    out = {g: _engine(g).render(rgb) for g in ("t31_base", "t44_base")}
    b = np.asarray(engine.LAB_SHADOW_BLACK)
    np.testing.assert_allclose(out["t44_base"], b + (1 - b) * out["t31_base"], rtol=1e-9, atol=1e-12)


def test_black_raised_white_kept():
    eng = _engine("t44_base")
    sc = eng.scanner
    base = sc.to_rgb(np.zeros((1, 3)))[0]                # 片基 (Dn = 0)
    eng0 = _engine("t31_base")
    base0 = eng0.scanner.to_rgb(np.zeros((1, 3)))[0]
    b = np.asarray(engine.LAB_SHADOW_BLACK)
    np.testing.assert_allclose(base - base0, b * (1 - base0), rtol=1e-9)
    grey, grey0 = sc.to_rgb(eng._lab_gray[None])[0], eng0.scanner.to_rgb(eng0._lab_gray[None])[0]
    assert np.max(np.abs(grey / grey0 - 1)) < 0.02        # 中灰几乎不变 (< 2%)


def test_dark_background_threshold(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/eval"))
    import t19_halo_stats as hs
    monkeypatch.delenv("BG_LIFT", raising=False)
    np.testing.assert_allclose(hs.dark_bg_max("x_t44_base_0_f4.png"), 0.02)
    monkeypatch.setenv("BG_LIFT", "t44_base:0.003,0.0015,0.0015")
    np.testing.assert_allclose(hs.dark_bg_max("x_t44_base_0_f4.png"), [0.02294, 0.02147, 0.02147])
    np.testing.assert_allclose(hs.dark_bg_max("x_t31_base_0_f4.png"), 0.02)
    np.testing.assert_allclose(hs.dark_bg_max("ref_scan.jpg"), 0.02)
