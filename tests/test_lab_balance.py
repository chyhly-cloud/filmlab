"""T05 fixed light offsets: synthetic arrays only, no acceptance imports."""
from dataclasses import replace
import numpy as np
import pytest
from filmlab import generation
from filmlab.engine import (Engine, Params, _lab_balance, LAB_BALANCE_DELTA_K,
                            LAB_BALANCE_DELTA_R, LAB_BALANCE_DELTA_B)
from filmlab.scanning.timing import PrintTiming
from filmlab.recipes import recipe


@pytest.fixture(autouse=True)
def restore_generation():
    old = generation.active()
    yield
    generation.use(old)


def image():
    return np.random.default_rng(20261026).uniform(.015, .7, (8, 12, 3))


def make(corrections, timing='frame', name='cinestill_800t'):
    return Engine(Params(**{**recipe(name), 'roll': None, 'grain': 0,
                            'timing': timing, 'corrections': corrections}))


def test_fixed_constants_are_spec_values():
    assert LAB_BALANCE_DELTA_K == -0.11437142396955352
    assert LAB_BALANCE_DELTA_R == -0.12100152025600766
    assert LAB_BALANCE_DELTA_B == -0.008793146365340334
    assert generation.GENERATIONS['next_balance'] == (
        'D65', generation.GENERATIONS['next'][1]+('lab_balance',), {})


@pytest.mark.parametrize('empty', [False, True])
def test_exact_o1_unit_mapping_and_other_fields_preserved(empty):
    t = PrintTiming(K=np.array([.3,.5,.7]), s=np.array([.8,1.1,1.2]),
                    M=None if empty else (np.array([.1,.2,-.3]), np.array([.9,1,1.1])),
                    x_knee=-.5, x_white=-1.)
    k0=t.K.copy()
    off0,r0=(np.zeros(3),np.ones(3)) if empty else t.M
    actual=_lab_balance(t)
    np.testing.assert_allclose(actual.K-t.K, np.full(3,LAB_BALANCE_DELTA_K), atol=1e-12,rtol=0)
    np.testing.assert_allclose(actual.M[0]-off0,[LAB_BALANCE_DELTA_R,0,LAB_BALANCE_DELTA_B],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(actual.M[1],r0)
    assert actual.s is t.s and actual.x_knee==t.x_knee and actual.x_white==t.x_white
    np.testing.assert_array_equal(t.K,k0)
    # Same mapping as oracle_timing.fit(1), without importing/running an oracle.
    expected=replace(t,K=t.K+LAB_BALANCE_DELTA_K,
                     M=(off0+np.array([LAB_BALANCE_DELTA_R,0,LAB_BALANCE_DELTA_B]),r0))
    density=np.random.default_rng(117).uniform(0,2,(25,3))
    np.testing.assert_array_equal(actual.paper_exposure(density),expected.paper_exposure(density))


def test_frame_shift_is_after_auto_balance_and_does_not_accumulate():
    generation.use('next')
    corr=generation.GENERATIONS['next'][1]
    a,b=make(corr),make(corr+('lab_balance',))
    neg=a.develop_negative(image())
    old,new,again={},{},{}
    a.scan(neg,trace=old);b.scan(neg,trace=new);b.scan(neg,trace=again)
    t,u=old['timing'],new['timing']
    np.testing.assert_array_equal(new['Dn'],old['Dn'])
    np.testing.assert_allclose(u.K-t.K,LAB_BALANCE_DELTA_K,atol=1e-12,rtol=0)
    np.testing.assert_allclose(u.M[0]-t.M[0],[LAB_BALANCE_DELTA_R,0,LAB_BALANCE_DELTA_B],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(u.M[1],t.M[1])
    np.testing.assert_array_equal(u.s,t.s)
    assert (u.x_knee,u.x_white)==(t.x_knee,t.x_white)
    np.testing.assert_array_equal(again['timing'].K,u.K)
    np.testing.assert_array_equal(again['timing'].M[0],u.M[0])


@pytest.mark.parametrize('timing,name', [('gray','cinestill_800t'),('standard','cinestill_800t'),
                                        ('standard','cinestill_800t_reference')])
def test_nonframe_and_physical_reference_are_bitwise_unchanged(timing,name):
    generation.use('next')
    corr=generation.GENERATIONS['next'][1]
    a,b=make(corr,timing,name),make(corr+('lab_balance',),timing,name)
    np.testing.assert_array_equal(a.render(image()),b.render(image()))
    np.testing.assert_array_equal(a.scanner.timing.K,b.scanner.timing.K)
    assert a.scanner.timing.M is None and b.scanner.timing.M is None


@pytest.mark.parametrize('gen', ['current','next','legacy'])
def test_default_generations_and_disabled_flag_are_bitwise_unchanged(gen,monkeypatch):
    import filmlab.engine as engine
    generation.use(gen)
    assert 'lab_balance' not in Params().corrections
    # The disabled path must not even execute the new arithmetic.
    monkeypatch.setattr(engine,'_lab_balance',lambda t: pytest.fail('disabled correction executed'))
    params={**recipe('cinestill_800t'),'timing':'frame','roll':None,'grain':0}
    a=Engine(Params(**params)).render(image())
    b=Engine(Params(**params,corrections=generation.GENERATIONS[gen][1])).render(image())
    np.testing.assert_array_equal(a,b)
