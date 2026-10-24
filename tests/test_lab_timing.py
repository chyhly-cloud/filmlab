"""T04 tests use only synthetic inputs; never load any real scene."""
from dataclasses import replace
import numpy as np
import pytest
from filmlab.scanning.timing import PrintTiming
from tools.eval.lab_timing import (oracle_fit, absolute_controls, DefinitionMismatch,
    check_closure, boundary_keys, boundary_stop, stopped_result)


def test_oracle_o1_recovers_known_density_units():
    import oracle_timing as ot
    rng = np.random.default_rng(20261025)
    densities = rng.uniform(.6, 1.6, (240, 3))
    t0 = PrintTiming(K=np.full(3, 1.2), s=np.full(3, 1.1),
                     M=(np.array([-.13, 0., .09]), np.array([1.02, 1., .98])))
    truth = np.array([.15, .05, -.06])
    target = replace(t0, K=t0.K+truth[0], M=(t0.M[0]+[truth[1],0,truth[2]], t0.M[1]))
    # Synthetic invertible response, with the real oracle's exact fitter/Lab map.
    render = lambda t: 1/(1+np.exp(-t.paper_exposure(densities)))
    fitted, x = oracle_fit(render, t0, densities, ot.lab_lin(render(target)), np.ones(240,bool), 1)
    np.testing.assert_allclose(x, truth, atol=1e-6, rtol=0)
    values = absolute_controls(t0, fitted, x)
    np.testing.assert_allclose(values['c_auto'], [-.13,.09], atol=1e-12)
    np.testing.assert_allclose(values['c_lab'], [-.08,.03], atol=1e-6)
    assert abs(values['K_lab']-1.35) < 1e-6


def test_definition_mismatch_stops():
    t0=PrintTiming(K=np.ones(3),M=(np.zeros(3),np.ones(3)))
    bad=replace(t0,K=t0.K+.1,M=(np.array([-.2,0,.1]),np.ones(3)))
    with pytest.raises(DefinitionMismatch):
        absolute_controls(t0,bad,[.1,.2,.1])


def test_closure_stops_at_tolerance_and_keeps_result_schema():
    good=check_closure('synthetic', 1.+.5e-6,1.)
    bad=check_closure('synthetic', 1.+2e-6,1.)
    assert good['closed'] and not bad['closed']
    result=stopped_result([bad])
    assert result['metadata']['status']=='stopped_auto_not_closed'
    assert result['frames']==result['ledger']=={}
    assert result['verdict'] is None
    assert check_closure('synthetic',float('nan'),1.)['closed'] is False


def test_boundary_rule_is_strict_and_counts_frames_not_parameters():
    assert boundary_keys([0,0,0])==[]
    assert boundary_keys([1.49,0,0])==['K']
    assert not boundary_stop(['a','b','c','a'])
    assert boundary_stop(['a','b','c','d'])

