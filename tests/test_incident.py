import numpy as np
import pytest

from filmlab.engine import Engine, Params
from filmlab.incident import IncidentSpectrum, apply_incident, spectral_responses
from filmlab import film, lights
from filmlab.recipes import recipe


def engine():
    return Engine(Params(**{**recipe('cinestill_800t'), 'grain': 0, 'roll': None, 'timing': 'frame'}))


def test_known_incident_spectrum_ignores_background_and_integrates_filter():
    e = engine()
    e.p.lens_filter = np.linspace(.1, .9, 81)
    e.p.exposure_ev = 1
    mask = np.zeros((48, 48), bool); mask[22:26, 22:26] = True
    amplitude = np.ones(mask.shape) * .7
    spec = lights._blackbody(2700)
    source = IncidentSpectrum(spec, amplitude, mask, 'Controlled blackbody')
    observed = []
    for background in ([.02]*3, [.001, .01, .04], [.04, .01, .001]):
        rgb = np.broadcast_to(background, (48, 48, 3)).copy()
        rgb[mask] = lights.camera_rgb(spec) * .7
        trace = {}
        e.develop_negative(rgb, incident_sources=[source], trace=trace)
        observed.append((trace['direct_exposure'][mask], trace['halo_source'][mask]))
    expected = 1.4 * ((spec * e.p.lens_filter) @ (10 ** e.e.stock.log_sensitivity))
    for direct, halo in observed:
        np.testing.assert_allclose(direct, np.broadcast_to(expected, direct.shape), rtol=1e-14)
        np.testing.assert_array_equal(direct, observed[0][0])
        np.testing.assert_array_equal(halo, observed[0][1])


def test_trace_and_empty_incident_sources_leave_existing_negative_bitwise_equal():
    e = engine(); rgb = np.random.default_rng(4).uniform(.01, 2, (40, 48, 3))
    baseline = e.develop_negative(rgb)
    candidate = e.develop_negative(rgb, incident_sources=(), trace={})
    np.testing.assert_array_equal(baseline.log_exposure, candidate.log_exposure)
    np.testing.assert_array_equal(baseline.dye, candidate.dye)


def test_bright_pixels_are_not_automatically_classified_as_emitters():
    e = engine(); rgb = np.full((40, 48, 3), .02); rgb[20, 20] = 100
    expected, _ = e._expose_mixed(rgb * 2 ** e.p.exposure_ev)
    trace = {}; e.develop_negative(rgb, trace=trace)
    np.testing.assert_array_equal(trace['direct_exposure'], expected)


def test_overlapping_or_invalid_spectral_rays_are_rejected():
    e = engine(); rgb = np.ones((4, 4, 3))
    source = IncidentSpectrum(np.ones(81), np.ones((4, 4)), np.ones((4, 4), bool), 'known')
    with pytest.raises(ValueError, match='Overlapping'):
        e.develop_negative(rgb, incident_sources=[source, source])
    with pytest.raises(ValueError, match='nonnegative'):
        e.develop_negative(rgb, incident_sources=[IncidentSpectrum(-np.ones(81), source.amplitude, source.mask, 'known')])


def test_spectral_input_requires_nonblank_evidence_reference():
    e=engine();rgb=np.ones((4,4,3))
    with pytest.raises(ValueError,match='provenance'):
        e.develop_negative(rgb,incident_sources=[IncidentSpectrum(np.ones(81),np.ones((4,4)),np.ones((4,4),bool),'  ')])


def test_known_spectra_additivity_homogeneity_and_filter_commutation():
    e=engine();shape=(2,3);mask=np.ones(shape,bool)
    z=np.zeros((*shape,3));amp=np.ones(shape)
    a=lights._blackbody(2700);b=lights._sd('LED-B3')
    def response(spectrum, amount=1.):
        return apply_incident(e,z,z,[IncidentSpectrum(spectrum,amp*amount,mask,'Controlled spectrum')])
    for ra,rb,rab in zip(response(a),response(b),response(a+b)):
        np.testing.assert_allclose(rab,ra+rb,rtol=3e-15,atol=1e-15)
    for actual,expected in zip(response(a,2.7),response(a)):
        np.testing.assert_allclose(actual,2.7*expected,rtol=3e-15)
    filt=np.linspace(.1,.9,81)
    expected=response(a*filt)
    e.p.lens_filter=filt
    for actual,want in zip(response(a),expected):
        np.testing.assert_allclose(actual,want,rtol=3e-15)


def test_fixed_reference_preserves_old_return_calibration():
    e=engine();sens,response=spectral_responses(e)
    E=film._spec(e.e.stock.reference_illuminant)
    down,direct,back=film.emulsion_stack(e.e.stock,e.v.layer_absorbance,e.v.yellow_filter)
    old=(E@(down[:,None]*back))*(E@sens)/(E@direct)*e.v.diffuse_fraction*e._back_reflect
    np.testing.assert_allclose(E@response,old,rtol=3e-15)
