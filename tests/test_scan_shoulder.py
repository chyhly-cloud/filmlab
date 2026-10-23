"""The experimental shoulder ablation must not refit timing or film colour."""
from dataclasses import replace

import numpy as np
import pytest

from filmlab import generation, spectral
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.scanning.timing import PrintTiming, SHOULDER_WHITE


@pytest.mark.parametrize("conditioning", [None, "lab"])
def test_paper_only_changes_only_active_shoulder(conditioning):
    rgb = np.repeat((.184 * 2.0 ** np.linspace(-6, 10, 129))[None, :, None], 3, -1)
    p = Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None,
                  "grain": 0, "output_conditioning": conditioning})
    baseline, candidate = Engine(p), Engine(replace(p, scan_shoulder="paper_only"))
    negative_a, negative_b = baseline.develop_negative(rgb), candidate.develop_negative(rgb)
    assert np.array_equal(negative_a.log_exposure, negative_b.log_exposure)
    assert np.array_equal(negative_a.dye, negative_b.dye)
    ta, tb = {}, {}
    a, b = baseline.scan(negative_a, trace=ta), candidate.scan(negative_b, trace=tb)
    assert np.array_equal(ta["Dn"], tb["Dn"])
    sa, sb = baseline.scanner, candidate.scanner
    for field in ("K", "s"):
        assert np.array_equal(getattr(sa.timing, field), getattr(sb.timing, field))
    for va, vb in zip(sa.M, sb.M):
        assert np.array_equal(va, vb)
    assert sb.x_knee is None and sb.x_white is None
    mask = (sb.timing.paper_exposure(tb["Dn"]) >= sa.x_knee).all(-1)
    assert mask.any() and (~mask).any()
    assert np.array_equal(a[mask], b[mask])
    # Independently disable the baseline's shoulder at fixed timing. This must
    # exactly reproduce the candidate, with no new exposure/contrast fit.
    sa.timing = replace(sa.timing, x_knee=None, x_white=None)
    assert np.array_equal(sa.to_rgb(ta["Dn"]), b)


@pytest.mark.parametrize("timing", ["gray", "standard"])
def test_paper_only_does_not_change_non_frame_scan(timing):
    rgb = np.repeat((.184 * 2.0 ** np.linspace(-4, 6, 33))[None, :, None], 3, -1)
    p = Params(**{**recipe("cinestill_800t"), "timing": timing, "grain": 0, "roll": None})
    assert np.array_equal(Engine(p).render(rgb), Engine(replace(p, scan_shoulder="paper_only")).render(rgb))


def test_float_ceiling_exists_before_display_quantization():
    engine = Engine(Params(**{**recipe("cinestill_800t"), "grain": 0, "roll": None}))
    sc, paper = engine.scanner, engine.scanner.digital_paper
    xw = float(np.interp(-np.log10(SHOULDER_WHITE), paper.curve, paper.x))
    sc.timing = PrintTiming(K=np.zeros(3), s=np.ones(3), x_knee=xw + .25, x_white=xw)
    # Both inputs are far beyond the exponential's knee; no encoding or rounding
    # is involved in this convergence to a model-imposed, sub-unity float ceiling.
    sc.conditioning = None
    result = sc.to_rgb(np.full((2, 3), [[20.0], [30.0]]))
    expected = np.maximum(paper.reference_print(paper.density(np.full((1, 3), xw))), 0)
    assert np.allclose(result, expected, rtol=0, atol=1e-12)
    assert np.all(result < 1)
    # The full colour transform explains the small difference from neutral 247.
    assert np.array_equal(np.floor(spectral.srgb_encode(expected[0]) * 255 + .5), [247, 247, 246])


def test_candidate_is_not_a_generation_default():
    try:
        for name in ("current", "next", "next_sensor"):
            generation.use(name)
            assert Params().scan_shoulder == "exponential"
        with pytest.raises(ValueError, match="Unknown scan shoulder"):
            Engine(Params(scan_shoulder="unknown"))
    finally:
        generation.use("current")
