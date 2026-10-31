"""T07 next_lab (lab_measured + session metering) synthetic checks; no scene data."""
from dataclasses import replace

import numpy as np
import pytest
from filmlab import engine as eng_mod, generation, rawio
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe

RGB = np.random.default_rng(3).uniform(0.02, 0.6, (40, 56, 3))


@pytest.fixture
def restore():
    old = generation.active()
    yield
    generation.use(old)


def render(gen, **kw):
    generation.use(gen)
    e = Engine(Params(**{**recipe('cinestill_800t'), 'roll': None, 'grain': 0, 'timing': 'frame', **kw}))
    tr = {}
    out = e.scan(e.negative(RGB), trace=tr)
    return e, out, tr


def test_generation_switches_metering(restore):
    generation.use('next_lab')
    assert rawio.METERING == 'session'
    m, _ = rawio.film_metering(RGB)
    np.testing.assert_array_equal(m, RGB * 10.0 ** rawio.SESSION_LOG_SCALE)
    generation.use('next')
    assert rawio.METERING == 'meter'


def test_flag_off_is_unchanged(restore):
    # next with an explicit corrections tuple equal to next's: same output as the plain next path
    _, a, _ = render('next')
    _, b, _ = render('next', corrections=generation.GENERATIONS['next'][1])
    np.testing.assert_array_equal(a, b)


def test_frame_timing_is_measured_lab_colour(restore):
    e, _, tr = render('next_lab')
    Dn = tr['Dn'].reshape(-1, 3)
    v = Dn[np.all(np.isfinite(Dn), -1)].mean(0) - e._lab_gray
    (aR, kR), (aB, kB) = eng_mod.LAB_MEASURED_COLOUR['R'], eng_mod.LAB_MEASURED_COLOUR['B']
    t0 = e._lab_base
    off0 = t0.M[0] if t0.M is not None else np.zeros(3)
    expect = np.asarray(off0) + [aR - kR * (v[0] - v[1]), 0.0, aB - kB * (v[2] - v[1])]
    np.testing.assert_allclose(tr['timing'].M[0], expect, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(tr['timing'].K, t0.K)          # brightness not adjusted per frame
    assert e.scanner.conditioning is None                         # no other-lab output conditioning


def test_other_timings_unaffected(restore):
    for timing in ('standard', 'gray'):
        _, a, _ = render('next', timing=timing)
        _, b, _ = render('next', timing=timing, corrections=generation.GENERATIONS['next_lab'][1])
        np.testing.assert_array_equal(a, b)
