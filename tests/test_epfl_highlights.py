"""Local contrast must measure transfer slope, not reward a brightness offset."""
import numpy as np

from tools.eval.epfl_highlights import summaries


def test_local_contrast_separates_slope_from_exposure_offset():
    ev = np.broadcast_to(np.linspace(-3, 3, 160), (80, 160))
    real = np.repeat((.1 * 2.0 ** (.5 * ev))[..., None], 3, -1)
    steeper = np.repeat((.05 * 2.0 ** ev)[..., None], 3, -1)
    result = summaries(ev, np.ones(ev.shape, bool), real,
                       {"offset_only": real * 2, "slope_change": steeper})
    central = [r for r in result if r["status"] == "ok" and -1 <= r["EV"][0] < 1]
    assert len(central) == 4
    for row in central:
        paths = row["paths"]
        assert abs(paths["real"]["local_gain"] - .5) < 1e-5
        assert abs(paths["offset_only"]["local_gain"] - paths["real"]["local_gain"]) < 1e-8
        assert abs(paths["slope_change"]["local_gain"] - 1) < 1e-5


def test_missing_measured_pixels_are_insufficient_not_a_pass():
    ev = np.zeros((40, 40))
    rgb = np.ones((40, 40, 3)) * .18
    rows = summaries(ev, np.zeros(ev.shape, bool), rgb, {"model": rgb})
    assert all(r["status"] == "insufficient" and "paths" not in r for r in rows)
