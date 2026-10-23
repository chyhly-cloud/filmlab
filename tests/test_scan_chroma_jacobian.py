"""Independent scanning-domain contracts, not assertions that the model fits film."""
import numpy as np
import pytest
from tools.eval.scan_chroma_jacobian import linear_lab,condition_original,central_difference,crosses_threshold,rotation_degrees


def test_linear_lab_does_not_decode_srgb_again():
    np.testing.assert_allclose(linear_lab(np.full((1,3),.18))[0,0],49.4961076,atol=1e-5)


def test_empirical_intervention_is_exact_original_conditioning():
    from filmlab.scanner import Scanner,SCANNERS
    # _condition uses only these fields; no engine/calibration is needed here.
    from types import SimpleNamespace
    scanner=SimpleNamespace(conditioning='lab',m=SCANNERS['frontier'])
    rgb=np.array([[-.05,.2,.1],[.1,.6,.2],[1.05,.9,.95]])
    actual,_=condition_original(rgb,scanner.m)
    np.testing.assert_allclose(actual,Scanner._condition(scanner,rgb),atol=1e-14)


def test_black_clamp_can_reverse_a_response_without_zero_crossing():
    base=np.array([[-.0052441,.09510964,.00588725]])
    direction=np.array([[.11940939,.06127081,-.00381062]]);eps=1e-5
    minus=base-eps*direction;plus=base+eps*direction
    raw=central_difference(linear_lab(minus),linear_lab(plus),eps)
    displayed=central_difference(linear_lab(np.maximum(minus,0)),linear_lab(np.maximum(plus,0)),eps)
    assert raw[0,1]>40 and displayed[0,1]<-9
    assert not crosses_threshold(minus,plus,0).any()
    # Exactly on the clamp boundary, two one-sided slopes instead disagree.
    atzero=np.array([[-eps,0,0]]);above=-atzero
    assert crosses_threshold(atzero,above,0).all()


def test_cyan_density_increment_uses_beer_lambert_and_zero_direction_is_undefined():
    density=np.linspace(.1,1,81);cyan=np.linspace(.02,1.1,81);eps=1e-4
    plus=10**(-(density+eps*cyan));minus=10**(-(density-eps*cyan))
    np.testing.assert_allclose(plus/minus,10**(-2*eps*cyan),rtol=5e-16)
    assert np.isnan(rotation_degrees([[0,0]],[[1,0]])[0])
    np.testing.assert_allclose(rotation_degrees([[1,0]],[[0,1]]),90)
