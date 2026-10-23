"""Stage diagnostics must distinguish surviving density from display information loss."""
import numpy as np
import pytest

from filmlab import generation
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from tools.eval.highlight_transfer import stages, wedge


@pytest.mark.parametrize("timing", ["standard", "gray", "frame"])
def test_diagnostic_preserves_render_and_isolates_shoulder(timing):
    rgb = np.repeat((0.184 * 2.0 ** np.linspace(-4, 8, 49))[None, :, None], 3, -1)
    params = Params(**{**recipe("cinestill_800t"), "timing": timing, "roll": None, "grain": 0})
    expected = Engine(params).render(rgb)
    engine = Engine(params)
    data = stages(engine, engine.develop_negative(rgb))
    assert np.array_equal(data["output_rgb"], expected)
    # Ablation is at fixed negative and timing, not a newly auto-balanced image.
    paper, sc = engine.scanner.digital_paper, engine.scanner
    assert np.array_equal(data["without_extra_shoulder_rgb"],
                          sc._reference_print(paper.density(data["paper_exposure_linear"]), paper.k))
    if timing == "frame":
        assert np.any(data["shoulder_slope"] < 0.1)
        assert not np.array_equal(data["without_extra_shoulder_rgb"], data["output_rgb"])
    else:
        assert np.array_equal(data["without_extra_shoulder_rgb"], data["output_rgb"])


def test_density_information_can_survive_when_export_codes_do_not(tmp_path):
    try:
        generation.use("next")
        _, ev, data = wedge("frame", tmp_path)
        a, b = [int(np.argmin(abs(ev - v))) for v in (8, 10)]
        # Two stops remain distinguishable in negative density and floating RGB.
        assert data["Dn"][0, b, 1] > data["Dn"][0, a, 1] + 0.1
        assert not np.array_equal(data["output_rgb"][0, a], data["output_rgb"][0, b])
        # They have collapsed to the same non-white display code: this is neither
        # sensor saturation nor a floating-point hard clip at RGB=1.
        assert np.array_equal(data["encoded_u8"][0, a], data["encoded_u8"][0, b])
        assert np.all(data["output_rgb"][0, [a, b]] < 1)
    finally:
        generation.use("current")
