"""Locality must hold once the incident-light assignment is already fixed."""
import numpy as np
import pytest
from filmlab import generation,lights
from filmlab.engine import Engine,Params
from filmlab.recipes import recipe
from tools.eval.illumination_inference_trial import weighted_responses


@pytest.fixture(scope='module')
def engine():
    generation.use('next_sensor')
    try:
        yield Engine(Params(**{**recipe('cinestill_800t'),'roll':None,'grain':0,'output_conditioning':None}))
    finally:
        generation.use('current')


def test_fixed_local_light_is_not_removed_when_unrelated_canvas_grows(engine):
    members=lights.pick_members(engine.v.frame_seed);values=[];old=[]
    for size in [32,128]:
        rgb=np.full((size,size,3),.18);W=np.zeros((size,size,6))
        W[...,0]=1.;W[:4,:4,0]=0.;W[:4,:4,4]=1.
        h,s=weighted_responses(engine,rgb,W,members)
        before,_=weighted_responses(engine,rgb,W,members,.003)
        values.append((h[1,1],s[1,1]));old.append(before[1,1])
    np.testing.assert_allclose(values[0][0],values[1][0],rtol=1e-13,atol=0)
    np.testing.assert_allclose(values[0][1],values[1][1],rtol=1e-13,atol=0)
    # Demonstrate that this actually exercises the old global-pruning boundary.
    assert np.max(old[0])>100*np.max(old[1])


def test_no_pruning_preserves_full_response_when_every_class_was_retained(engine):
    rng=np.random.default_rng(73);rgb=rng.uniform(.03,.5,(8,8,3))
    W=rng.uniform(.1,1,(8,8,6));W/=W.sum(-1,keepdims=True)
    members=lights.pick_members(engine.v.frame_seed)
    a=weighted_responses(engine,rgb,W,members,.003)
    b=weighted_responses(engine,rgb,W,members,0.)
    for old,new in zip(a,b):np.testing.assert_array_equal(old,new)
