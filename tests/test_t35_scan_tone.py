"""T35: 冲印店扫描影调的整体反差 (lab_contrast, engine.LAB_SCAN_CONTRAST, EPFL 灰阶楔实测)。
现有代际不变; 新项只在 t35_base 启用, 只改固定光号的整体反差 s (支点 = 设计光源中灰), 负片与 Dn 不变。"""
import numpy as np
import pytest

from filmlab import engine, film, generation, rawio
from filmlab.engine import Engine, GRAY, Params
from filmlab.recipes import recipe


@pytest.fixture(autouse=True)
def _restore_generation():
    prev = generation.active()
    yield
    generation.use(prev)


def _eng(g, **kw):
    generation.use(g)
    return Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0, **kw}))


def test_generation():
    wb, corr, rev = generation.GENERATIONS["t35_base"]
    assert (wb, corr, rev) == ("D65", generation.GENERATIONS["t31_base"][1] + ("lab_contrast",), {})
    generation.use("t35_base")
    assert (rawio.METERING, rawio.HIGHLIGHT_RECON, rawio.RECON_COLOUR, rawio.APERTURE_RANGE, film.MASK_PLACEMENT) == \
        ("session", "stops", "ring", None, "layers")
    for g, c in generation.GENERATIONS.items():
        if g != "t35_base":
            assert "lab_contrast" not in c[1]
    assert 0.8 <= engine.LAB_SCAN_CONTRAST <= 0.9


def test_contrast_only_changes_timing():
    e31, e35 = _eng("t31_base"), _eng("t35_base")
    t31, t35 = e31._lab_base, e35._lab_base
    np.testing.assert_array_equal(t31.s, np.ones(3))                    # t31_base: 固定光号的 s 仍为 1
    np.testing.assert_allclose(t35.s, engine.LAB_SCAN_CONTRAST)
    np.testing.assert_array_equal(e31._lab_gray, e35._lab_gray)         # 负片 / Dn 不变
    # 支点 = 设计光源中灰: 绿通道的数字相纸曝光不变, 中灰的亮度不变
    g = e31._lab_gray
    x31, x35 = t31.paper_exposure(g), t35.paper_exposure(g)
    assert abs(x31[1] - x35[1]) < 1e-12
    assert np.max(np.abs(x31 - x35)) < 0.01
    # 相纸曝光对 Dn 的斜率 = −s
    d = np.array([0.3, 0.3, 0.3])
    np.testing.assert_allclose((t35.paper_exposure(g + d) - x35) / d, -engine.LAB_SCAN_CONTRAST)
    np.testing.assert_allclose((t31.paper_exposure(g + d) - x31) / d, -1.0)


def test_grey_steps_lower_output_contrast():
    """中性灰阶 (设计光源, 中灰 ±1 档) 的输出: 中灰亮度相同, 两侧向中灰收拢。"""
    outs = {}
    for g in ("t31_base", "t35_base"):
        e = _eng(g)
        rgb = (GRAY * 2.0 ** np.array([-2.0, 0.0, 1.0]))[:, None, None] * np.ones((1, 1, 3))
        H = e._expose_lin(rgb.reshape(-1, 3), e.M_ref, compensation=False)
        st = e._develop(e._log_exposure(H))
        Dn = e.scanner.normalized_density(e._transmittance(st))
        e.scanner.timing = e._lab_base
        outs[g] = e.scanner.to_rgb(Dn)[:, 1]
    a, b = outs["t31_base"], outs["t35_base"]
    assert b[0] > a[0] and b[2] < a[2]
    assert np.log10(b[2] / b[0]) < np.log10(a[2] / a[0])


def test_other_timings_unaffected():
    for timing in ("gray", "standard"):
        a, b = _eng("t31_base", timing=timing), _eng("t35_base", timing=timing)
        np.testing.assert_array_equal(a.scanner.timing.K, b.scanner.timing.K)
        np.testing.assert_array_equal(a.scanner.timing.s, b.scanner.timing.s)
