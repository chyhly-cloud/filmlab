import numpy as np
from tools.eval.base_colour_audit import cell_rows,exclusion_distance


def test_surface_selection_does_not_depend_on_model_colour_error():
    real=np.full((48,48,3),.2);mask=np.ones((48,48),bool)
    same=cell_rows({'real':real,'model':real.copy()},mask)
    wrong=cell_rows({'real':real,'model':np.full_like(real,.8)},mask)
    assert len(same)==len(wrong)==4
    assert [v['yx'] for v in same]==[v['yx'] for v in wrong]


def test_unobserved_border_is_not_a_highlight():
    assert np.isinf(exclusion_distance(np.zeros((10,10),bool),5)).all()
    mask=np.zeros((10,10),bool);mask[4,4]=True
    assert exclusion_distance(mask,5)[4,4]==0
    assert exclusion_distance(mask,5)[4,6]==10


def test_invalid_registration_support_is_not_filled_as_a_dark_surface():
    real=np.full((24,24,3),.2);model=real.copy();model[5,5]=np.nan
    assert not cell_rows({'real':real,'model':model},np.ones((24,24),bool))
