import numpy as np
import pytest

from filmlab.brackets import MeasuredHighlightPatch,replace_measured_highlights
from filmlab.rawio import XYZ_TO_SRGB


def test_measured_replacement_preserves_hdr_and_all_unverified_pixels():
    rgb=np.random.default_rng(4).random((8,9,3))
    sat=np.zeros(rgb.shape,bool);sat[3:5,3:6,1]=True
    a=np.full((4,5,3),4.);trusted=np.ones((4,5),bool);trusted[1,1]=False
    patch=MeasuredHighlightPatch((2,2),a,trusted,'1/2 s RAW x8; three channels validated')
    out,mask=replace_measured_highlights(rgb,sat,[patch],np.eye(3))
    assert mask.sum()==5
    np.testing.assert_array_equal(out[~mask],rgb[~mask])
    np.testing.assert_allclose(out[mask],np.broadcast_to(np.array([4.,4.,4.])@XYZ_TO_SRGB.T,(5,3)))
    assert out[mask].max()>1


def test_bright_unsaturated_reflector_is_not_replaced_or_classified():
    rgb=np.full((3,3,3),100.);sat=np.zeros(rgb.shape,bool)
    patch=MeasuredHighlightPatch((0,0),np.ones_like(rgb),np.ones((3,3),bool),'measurement')
    out,mask=replace_measured_highlights(rgb,sat,[patch],np.eye(3))
    assert not mask.any()
    np.testing.assert_array_equal(out,rgb)


def test_conflicting_measurements_require_explicit_selection():
    rgb=np.ones((3,3,3));sat=np.ones(rgb.shape,bool)
    p=MeasuredHighlightPatch((0,0),rgb,np.ones((3,3),bool),'measurement')
    with pytest.raises(ValueError,match='Choose one'):
        replace_measured_highlights(rgb,sat,[p,p],np.eye(3))


def test_measured_patch_requires_nonblank_evidence_reference():
    rgb=np.ones((3,3,3));sat=np.ones(rgb.shape,bool)
    p=MeasuredHighlightPatch((0,0),rgb,np.ones((3,3),bool),'  ')
    with pytest.raises(ValueError,match='provenance'):
        replace_measured_highlights(rgb,sat,[p],np.eye(3))
