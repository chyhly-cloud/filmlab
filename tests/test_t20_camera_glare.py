"""T20: camera-side glare stage (not film): energy conserving, uniform field unchanged, off by default."""
import numpy as np
import pytest
from filmlab import generation
from filmlab.engine import Engine, Params
from filmlab.optics import lens_glare, glare_kernel
from filmlab.recipes import recipe


@pytest.fixture(autouse=True)
def _restore():
    prev = generation.active()
    yield
    generation.use(prev)


def test_kernel_normalised_and_tail():
    k = glare_kernel(20.0, 100.0)
    assert abs(k.sum() - 1.0) < 1e-12
    n = k.shape[0] // 2
    assert k[n, n] > k[n, n + 10] > k[n, n + 100] > 0


def test_uniform_field_and_energy():
    x = np.full((120, 160, 3), 0.18)
    np.testing.assert_allclose(lens_glare(x, 22.5, 0.05, 100.0), x, atol=1e-12)
    x[60, 80] = 500.0
    y = lens_glare(x, 22.5, 0.05, 100.0)
    assert abs((y - 0.18).sum() / (x - 0.18).sum() - 1.0) < 1e-3
    assert y[60, 80, 0] < x[60, 80, 0] and y[60, 90, 0] > 0.18


def test_off_in_production():
    for g in ("current", "next", "next_lab_halo", "next_lab_halo_e2"):
        generation.use(g)
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))
        assert "camera_glare" not in eng.p.corrections
